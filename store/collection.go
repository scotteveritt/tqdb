package store

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/mathutil"
	"github.com/scotteveritt/tqdb/quantize"
)

// CollectionConfig extends tqdb.Config with an optional embedding function.
type CollectionConfig struct {
	tqdb.Config
	EmbedFunc tqdb.EmbeddingFunc
}

// Collection stores compressed vectors for batch search.
// It uses contiguous index storage and the rotated-space optimization
// to avoid O(d²) per-vector decompression during search.
//
// Scoring uses inner products in rotated space, matching the paper's approach.
// Since vectors are unit-normalized before quantization, inner product ≈ cosine similarity.
//
// Safe for concurrent Add and Search after construction.
type Collection struct {
	mu        sync.RWMutex
	quantizer *quantize.TurboQuantMSE
	dim       int
	embedFunc tqdb.EmbeddingFunc

	// Contiguous storage for cache-friendly search.
	// allIndices stores all vectors' indices end-to-end: vec0[0..d-1] | vec1[0..d-1] | ...
	allIndices []uint8
	norms      []float32 // original L2 norms (used for dequantization, not search ranking)
	ids        []string
	contents   []string
	dataFields []map[string]any
	idIndex    map[string]int // O(1) ID lookup
	deleted    []bool         // tombstone-based delete
}

// NewCollection creates a new Collection with the given quantizer config.
func NewCollection(cfg tqdb.Config) (*Collection, error) {
	q, err := quantize.NewMSE(cfg)
	if err != nil {
		return nil, err
	}
	return &Collection{
		quantizer: q,
		dim:       q.Rotation().WorkDim(),
		idIndex:   make(map[string]int),
	}, nil
}

// NewCollectionWithConfig creates a new Collection with the given config including embed func.
func NewCollectionWithConfig(cfg CollectionConfig) (*Collection, error) {
	q, err := quantize.NewMSE(cfg.Config)
	if err != nil {
		return nil, err
	}
	return &Collection{
		quantizer: q,
		dim:       q.Rotation().WorkDim(),
		embedFunc: cfg.EmbedFunc,
		idIndex:   make(map[string]int),
	}, nil
}

// Add compresses and stores a vector with its ID and data fields.
func (c *Collection) Add(id string, vec []float64, data map[string]any) {
	cv := c.quantizer.Quantize(vec)
	c.addCompressed(id, "", cv.Indices, cv.Norm, data)
}

// AddFloat32 is a convenience wrapper for float32 vectors.
func (c *Collection) AddFloat32(id string, vec []float32, data map[string]any) {
	c.Add(id, mathutil.Float32ToFloat64(vec), data)
}

// AddCompressed stores a pre-compressed vector.
func (c *Collection) AddCompressed(id string, cv *tqdb.CompressedVector, data map[string]any) {
	c.addCompressed(id, "", cv.Indices, cv.Norm, data)
}

func (c *Collection) addCompressed(id, content string, indices []uint8, norm float32, data map[string]any) {
	c.mu.Lock()
	idx := len(c.norms)
	c.allIndices = append(c.allIndices, indices...)
	c.norms = append(c.norms, norm)
	c.ids = append(c.ids, id)
	c.contents = append(c.contents, content)
	c.dataFields = append(c.dataFields, data)
	c.deleted = append(c.deleted, false)
	c.idIndex[id] = idx
	c.mu.Unlock()
}

// AddDocument compresses and stores a Document.
// If doc.Embedding is nil and an EmbeddingFunc is set, it auto-embeds doc.Content.
func (c *Collection) AddDocument(ctx context.Context, doc tqdb.Document) error {
	vec, err := c.resolveEmbedding(ctx, doc)
	if err != nil {
		return err
	}
	cv := c.quantizer.Quantize(vec)
	c.mu.Lock()
	idx := len(c.norms)
	c.allIndices = append(c.allIndices, cv.Indices...)
	c.norms = append(c.norms, cv.Norm)
	c.ids = append(c.ids, doc.ID)
	c.contents = append(c.contents, doc.Content)
	c.dataFields = append(c.dataFields, doc.Data)
	c.deleted = append(c.deleted, false)
	c.idIndex[doc.ID] = idx
	c.mu.Unlock()
	return nil
}

// AddDocuments adds multiple documents concurrently.
// concurrency controls the maximum number of concurrent embedding calls.
func (c *Collection) AddDocuments(ctx context.Context, docs []tqdb.Document, concurrency int) error {
	if concurrency <= 0 {
		concurrency = 1
	}

	type prepared struct {
		doc     tqdb.Document
		indices []uint8
		norm    float32
		err     error
	}

	results := make([]prepared, len(docs))
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for i, doc := range docs {
		wg.Add(1)
		go func(i int, doc tqdb.Document) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			vec, err := c.resolveEmbedding(ctx, doc)
			if err != nil {
				results[i] = prepared{doc: doc, err: err}
				return
			}
			cv := c.quantizer.Quantize(vec)
			results[i] = prepared{doc: doc, indices: cv.Indices, norm: cv.Norm}
		}(i, doc)
	}
	wg.Wait()

	// Check for errors before inserting any.
	for _, r := range results {
		if r.err != nil {
			return fmt.Errorf("embedding %q: %w", r.doc.ID, r.err)
		}
	}

	// Insert all in order while holding the lock.
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, r := range results {
		idx := len(c.norms)
		c.allIndices = append(c.allIndices, r.indices...)
		c.norms = append(c.norms, r.norm)
		c.ids = append(c.ids, r.doc.ID)
		c.contents = append(c.contents, r.doc.Content)
		c.dataFields = append(c.dataFields, r.doc.Data)
		c.deleted = append(c.deleted, false)
		c.idIndex[r.doc.ID] = idx
	}
	return nil
}

func (c *Collection) resolveEmbedding(ctx context.Context, doc tqdb.Document) ([]float64, error) {
	if doc.Embedding != nil {
		return doc.Embedding, nil
	}
	if c.embedFunc == nil {
		return nil, fmt.Errorf("tqdb: document %q has no embedding and no EmbeddingFunc configured", doc.ID)
	}
	f32, err := c.embedFunc(ctx, doc.Content)
	if err != nil {
		return nil, fmt.Errorf("tqdb: embed %q: %w", doc.ID, err)
	}
	return mathutil.Float32ToFloat64(f32), nil
}

// GetByID returns a document by its ID.
func (c *Collection) GetByID(id string) (tqdb.Document, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	idx, ok := c.idIndex[id]
	if !ok || c.deleted[idx] {
		return tqdb.Document{}, false
	}
	return tqdb.Document{
		ID:      c.ids[idx],
		Content: c.contents[idx],
		Data:    c.dataFields[idx],
		// Note: Embedding not returned (it's compressed).
	}, true
}

// Delete marks documents as deleted by ID. Deleted documents are excluded from searches.
func (c *Collection) Delete(ids ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, id := range ids {
		idx, ok := c.idIndex[id]
		if !ok {
			continue
		}
		c.deleted[idx] = true
		delete(c.idIndex, id)
	}
	return nil
}

// Upsert inserts or replaces a vector with its data fields.
// If the ID already exists, the old entry is tombstoned and a new entry is appended.
func (c *Collection) Upsert(id string, vec []float64, data map[string]any) {
	cv := c.quantizer.Quantize(vec)
	c.mu.Lock()
	// Tombstone old entry if it exists.
	if oldIdx, ok := c.idIndex[id]; ok {
		c.deleted[oldIdx] = true
	}
	idx := len(c.norms)
	c.allIndices = append(c.allIndices, cv.Indices...)
	c.norms = append(c.norms, cv.Norm)
	c.ids = append(c.ids, id)
	c.contents = append(c.contents, "")
	c.dataFields = append(c.dataFields, data)
	c.deleted = append(c.deleted, false)
	c.idIndex[id] = idx
	c.mu.Unlock()
}

// UpsertDocument inserts or replaces a document.
func (c *Collection) UpsertDocument(ctx context.Context, doc tqdb.Document) error {
	vec, err := c.resolveEmbedding(ctx, doc)
	if err != nil {
		return err
	}
	cv := c.quantizer.Quantize(vec)
	c.mu.Lock()
	if oldIdx, ok := c.idIndex[doc.ID]; ok {
		c.deleted[oldIdx] = true
	}
	idx := len(c.norms)
	c.allIndices = append(c.allIndices, cv.Indices...)
	c.norms = append(c.norms, cv.Norm)
	c.ids = append(c.ids, doc.ID)
	c.contents = append(c.contents, doc.Content)
	c.dataFields = append(c.dataFields, doc.Data)
	c.deleted = append(c.deleted, false)
	c.idIndex[doc.ID] = idx
	c.mu.Unlock()
	return nil
}

// ListIDs returns the IDs of all non-deleted entries.
func (c *Collection) ListIDs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]string, 0, len(c.idIndex))
	for id := range c.idIndex {
		out = append(out, id)
	}
	return out
}

// Count returns the number of non-deleted entries.
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.idIndex)
}

// Search finds the top-k most similar vectors to the query.
//
// Scoring uses the inner product in rotated space: <Pi*q_hat, centroids[idx]>
// where q_hat = q/||q||. Since stored vectors are unit-normalized before quantization,
// this equals cosine similarity without the noise of centroid-norm correction.
func (c *Collection) Search(query []float64, topK int) []tqdb.Result {
	return c.SearchWithOptions(query, tqdb.SearchOptions{TopK: topK})
}

// SearchWithOptions performs a vector similarity search with VS2-aligned options.
func (c *Collection) SearchWithOptions(query []float64, opts tqdb.SearchOptions) []tqdb.Result {
	if opts.TopK <= 0 {
		opts.TopK = 10
	}
	var filterFn func(map[string]any) bool
	if opts.Filter != nil {
		filterFn = opts.Filter.Match
	}
	return c.searchInternal(query, opts.TopK, filterFn, opts)
}

// SemanticSearch embeds a text string and performs vector similarity search.
func (c *Collection) SemanticSearch(ctx context.Context, text string, opts tqdb.SearchOptions) ([]tqdb.Result, error) {
	if c.embedFunc == nil {
		return nil, fmt.Errorf("tqdb: no EmbeddingFunc configured for semantic search")
	}
	f32, err := c.embedFunc(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("tqdb: embed query: %w", err)
	}
	query := mathutil.Float32ToFloat64(f32)
	return c.SearchWithOptions(query, opts), nil
}

// Query performs filter-only retrieval (no vector similarity scoring).
func (c *Collection) Query(opts tqdb.QueryOptions) []tqdb.Result {
	if opts.PageSize <= 0 {
		opts.PageSize = 100
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []tqdb.Result
	for i, data := range c.dataFields {
		if c.deleted[i] {
			continue
		}
		if opts.Filter != nil && !opts.Filter.Match(data) {
			continue
		}
		results = append(results, tqdb.Result{
			ID:      c.ids[i],
			Content: c.contents[i],
			Data:    data,
		})
		if len(results) >= opts.PageSize {
			break
		}
	}
	return results
}

// SearchFloat32 is a convenience wrapper for float32 queries.
func (c *Collection) SearchFloat32(query []float32, topK int) []tqdb.Result {
	return c.Search(mathutil.Float32ToFloat64(query), topK)
}

func (c *Collection) searchInternal(query []float64, topK int, filterFn func(map[string]any) bool, opts tqdb.SearchOptions) []tqdb.Result {
	c.mu.RLock()
	n := len(c.norms)
	if n == 0 {
		c.mu.RUnlock()
		return nil
	}

	d := c.dim
	centroids := c.quantizer.Codebook().Centroids

	// Rotate the unit query once.
	queryRotated := c.quantizer.GetBuf()
	queryNorm := mathutil.Norm(query)
	if queryNorm < 1e-15 {
		c.quantizer.PutBuf(queryRotated)
		c.mu.RUnlock()
		return nil
	}

	// Normalize query before rotation so the inner product = cosine similarity.
	invQN := 1.0 / queryNorm
	unitQuery := c.quantizer.GetBuf()
	for i, v := range query {
		unitQuery[i] = v * invQN
	}
	c.quantizer.Rotation().Rotate(queryRotated, unitQuery[:c.quantizer.GetConfig().Dim])
	c.quantizer.PutBuf(unitQuery)

	// Top-k via sorted-insert (with extra capacity for offset).
	effectiveK := topK + opts.Offset
	type scored struct {
		idx   int
		score float64
	}
	topBuf := make([]scored, 0, effectiveK+1)
	minScore := -math.MaxFloat64

	allIdx := c.allIndices

	for i := range n {
		// Skip deleted entries.
		if c.deleted[i] {
			continue
		}

		// Apply data field filter before scoring.
		if filterFn != nil && !filterFn(c.dataFields[i]) {
			continue
		}

		// Inner product: <Pi*q_hat, centroids[idx]>
		// For unit-normalized stored vectors, this ~= cosine similarity.
		indices := allIdx[i*d : i*d+d : i*d+d]

		var dot float64
		j := 0
		for ; j <= d-4; j += 4 {
			i0, i1, i2, i3 := indices[j], indices[j+1], indices[j+2], indices[j+3]
			dot += queryRotated[j]*centroids[i0] +
				queryRotated[j+1]*centroids[i1] +
				queryRotated[j+2]*centroids[i2] +
				queryRotated[j+3]*centroids[i3]
		}
		for ; j < d; j++ {
			dot += queryRotated[j] * centroids[indices[j]]
		}

		// Apply MinScore filter after scoring.
		if opts.MinScore > 0 && dot < opts.MinScore {
			continue
		}

		// Fast rejection.
		if len(topBuf) >= effectiveK && dot <= minScore {
			continue
		}

		pos := sort.Search(len(topBuf), func(p int) bool {
			return topBuf[p].score < dot
		})
		topBuf = append(topBuf, scored{})
		copy(topBuf[pos+1:], topBuf[pos:])
		topBuf[pos] = scored{idx: i, score: dot}

		if len(topBuf) > effectiveK {
			topBuf = topBuf[:effectiveK]
		}
		if len(topBuf) == effectiveK {
			minScore = topBuf[effectiveK-1].score
		}
	}

	c.quantizer.PutBuf(queryRotated)

	// Apply offset: skip the first opts.Offset results.
	if opts.Offset > 0 && opts.Offset < len(topBuf) {
		topBuf = topBuf[opts.Offset:]
	} else if opts.Offset >= len(topBuf) {
		topBuf = nil
	}

	// Trim to topK after offset.
	if len(topBuf) > topK {
		topBuf = topBuf[:topK]
	}

	results := make([]tqdb.Result, len(topBuf))
	for i, s := range topBuf {
		results[i] = tqdb.Result{
			ID:      c.ids[s.idx],
			Score:   s.score,
			Content: c.contents[s.idx],
			Data:    c.dataFields[s.idx],
		}
	}

	c.mu.RUnlock()
	return results
}

// Len returns the number of vectors in the collection (including deleted).
func (c *Collection) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.norms)
}

// Size returns the approximate memory usage in bytes for stored vectors.
func (c *Collection) Size() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	n := int64(len(c.norms))
	// allIndices: n*d bytes, norms: n*4, per-entry overhead ~64 bytes
	return n*int64(c.dim) + n*4 + n*64
}

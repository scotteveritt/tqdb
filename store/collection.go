package store

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/distancer"
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
	allIndices []uint8
	norms      []float32
	ids        []string
	contents   []string
	dataFields []map[string]any
	idIndex    map[string]int
	deleted    []bool

	// Indexes (built by CreateIndex, nil until then).
	filterIdx *filterIndex
	ivfIdx    *ivfIndex
	hnswIdx   *hnswIndex
	hnswVecs  []float32 // precomputed dequantized vectors for HNSW NEON distance
}

// filterIndex provides O(1) lookup for Eq/In filter operations.
// Built by CreateIndex on declared filter fields.
type filterIndex struct {
	fields map[string]map[any][]int // field → value → vector indices
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
// Silently skips if the ID already exists. Use Upsert to replace.
func (c *Collection) Add(id string, vec []float64, data map[string]any) {
	c.mu.RLock()
	_, exists := c.idIndex[id]
	c.mu.RUnlock()
	if exists {
		return
	}
	cv := c.quantizer.Quantize(vec)
	c.addCompressed(id, "", cv.Indices, cv.Norm, data)
}

// AddFloat32 is a convenience wrapper for float32 vectors.
func (c *Collection) AddFloat32(id string, vec []float32, data map[string]any) {
	c.Add(id, mathutil.Float32ToFloat64(vec), data)
}

// AddCompressed stores a pre-compressed vector.
// Silently skips if the ID already exists.
func (c *Collection) AddCompressed(id string, cv *tqdb.CompressedVector, data map[string]any) {
	c.mu.RLock()
	_, exists := c.idIndex[id]
	c.mu.RUnlock()
	if exists {
		return
	}
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

	// Maintain filter index incrementally.
	if c.filterIdx != nil && data != nil {
		for field, fieldIdx := range c.filterIdx.fields {
			if val, ok := data[field]; ok {
				fieldIdx[val] = append(fieldIdx[val], idx)
			}
		}
	}

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

// CreateIndex builds filter indexes and IVF partitions for fast search.
// After calling CreateIndex, searches use partition pruning (O(√N) instead of O(N))
// and inverted index lookups for Eq/In filters (O(1) instead of O(N)).
// Matches VS2's CreateIndex API.
func (c *Collection) CreateIndex(cfg tqdb.IndexConfig) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := len(c.norms)

	// Build filter inverted indexes.
	if len(cfg.FilterFields) > 0 {
		idx := &filterIndex{
			fields: make(map[string]map[any][]int, len(cfg.FilterFields)),
		}
		for _, field := range cfg.FilterFields {
			idx.fields[field] = make(map[any][]int)
		}
		for i, data := range c.dataFields {
			if c.deleted[i] {
				continue
			}
			for _, field := range cfg.FilterFields {
				if val, ok := data[field]; ok {
					idx.fields[field][val] = append(idx.fields[field][val], i)
				}
			}
		}
		c.filterIdx = idx
	}

	// Resolve auto index type.
	indexType := cfg.Type
	if indexType == tqdb.IndexAuto {
		if n < 10_000 {
			indexType = tqdb.IndexNone
		} else {
			indexType = tqdb.IndexIVF
		}
	}

	// Build ANN index.
	switch indexType {
	case tqdb.IndexHNSW:
		// HNSW graph index over quantized vectors.
		// Precompute dequantized float32 vectors for NEON-accelerated distance.
		centroids32 := c.quantizer.Codebook().Centroids32
		d := c.dim
		allIdx := c.allIndices

		// Materialize float32 vectors: decodedVecs[i*d : (i+1)*d] = centroids32[indices[j]]
		decodedVecs := make([]float32, n*d)
		for i := range n {
			off := i * d
			for j := range d {
				decodedVecs[off+j] = centroids32[allIdx[off+j]]
			}
		}

		distFunc := func(a, b int) float32 {
			return distancer.NegDot(decodedVecs[a*d:a*d+d], decodedVecs[b*d:b*d+d])
		}

		h := newHNSW(hnswConfig{
			M:              cfg.M,
			EfConstruction: cfg.EfConstruction,
		})
		for i := range n {
			if c.deleted[i] {
				continue
			}
			h.Insert(i, distFunc)
		}
		// Store decoded vectors for search-time distance computation.
		c.hnswIdx = h
		c.hnswVecs = decodedVecs

	case tqdb.IndexNone:
		// No ANN index. Filter indexes were already built above.

	case tqdb.IndexIVF:
		// IVF partitions (ScaNN-style k-means over rotated centroids).
		if n >= 100 {
			numPartitions := cfg.NumPartitions
			if numPartitions <= 0 {
				numPartitions = int(math.Sqrt(float64(n)))
				if numPartitions < 4 {
					numPartitions = 4
				}
			}
			nProbe := cfg.NProbe
			if nProbe <= 0 {
				nProbe = int(math.Sqrt(float64(numPartitions)))
				if nProbe < 1 {
					nProbe = 1
				}
			}

			c.ivfIdx = buildIVF(
				c.allIndices,
				c.quantizer.Codebook().Centroids,
				c.dim,
				n,
				numPartitions,
				nProbe,
				c.deleted,
			)
		}
	}
}

// filterCandidates returns the set of vector indices matching the filter
// using the inverted index. Returns nil if the filter can't be resolved
// via the index (falls back to brute-force evaluation).
func (c *Collection) filterCandidates(f tqdb.Filter) map[int]struct{} {
	if c.filterIdx == nil {
		return nil
	}
	return resolveFilter(f, c.filterIdx)
}

// resolveFilter recursively resolves a filter against the inverted index.
// Returns nil if the filter can't be resolved (requires brute-force).
func resolveFilter(f tqdb.Filter, idx *filterIndex) map[int]struct{} {
	switch ft := f.(type) {
	case interface{ Field() string; Value() any }:
		// Eq filter — direct lookup
		field := ft.Field()
		val := ft.Value()
		if fieldIdx, ok := idx.fields[field]; ok {
			if ids, ok := fieldIdx[val]; ok {
				set := make(map[int]struct{}, len(ids))
				for _, id := range ids {
					set[id] = struct{}{}
				}
				return set
			}
			return map[int]struct{}{} // field indexed but value not found
		}
		return nil // field not indexed, fall back

	case interface{ Filters() []tqdb.Filter }:
		// And/Or — resolve sub-filters
		subs := ft.Filters()
		if len(subs) == 0 {
			return nil
		}

		// Check if this is And (intersect) or Or (union)
		// Use Match on empty data to distinguish: And({}) = true, Or({}) = false
		isAnd := f.Match(map[string]any{})

		if isAnd {
			// Intersect: start with first resolved set, intersect rest
			var result map[int]struct{}
			for _, sub := range subs {
				resolved := resolveFilter(sub, idx)
				if resolved == nil {
					continue // can't resolve this sub, skip it
				}
				if result == nil {
					result = resolved
				} else {
					// Intersect
					for id := range result {
						if _, ok := resolved[id]; !ok {
							delete(result, id)
						}
					}
				}
			}
			return result
		}

		// Union
		result := map[int]struct{}{}
		allResolved := true
		for _, sub := range subs {
			resolved := resolveFilter(sub, idx)
			if resolved == nil {
				allResolved = false
				continue
			}
			for id := range resolved {
				result[id] = struct{}{}
			}
		}
		if !allResolved {
			return nil // can't fully resolve Or, fall back
		}
		return result

	default:
		return nil // complex filter, fall back to brute-force
	}
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
	centroids32 := c.quantizer.Codebook().Centroids32

	// Rotate the unit query once. Use a single pooled buffer for normalize+rotate.
	queryNorm := mathutil.Norm(query)
	if queryNorm < 1e-15 {
		c.mu.RUnlock()
		return nil
	}

	queryRotated := c.quantizer.GetBuf()
	// Normalize into the first `dim` elements of queryRotated (reuse as temp).
	invQN := 1.0 / queryNorm
	origDim := c.quantizer.GetConfig().Dim
	for i := range origDim {
		queryRotated[i] = query[i] * invQN
	}
	// Rotate in-place: rotate needs src != dst, so use a second buf only for rotation.
	unitBuf := c.quantizer.GetBuf()
	copy(unitBuf[:origDim], queryRotated[:origDim])
	c.quantizer.Rotation().Rotate(queryRotated, unitBuf[:origDim])
	c.quantizer.PutBuf(unitBuf)

	// Convert rotated query to float32 for scoring (5-7% faster, zero recall loss).
	qr32 := make([]float32, d)
	for i := range d {
		qr32[i] = float32(queryRotated[i])
	}

	// Top-k via sorted-insert (with extra capacity for offset + rescore).
	effectiveK := topK + opts.Offset
	if opts.Rescore > 0 && opts.Rescore > effectiveK {
		effectiveK = opts.Rescore
	}
	type scored struct {
		idx   int
		score float64
	}
	topBuf := make([]scored, 0, effectiveK+1)
	minScore := -math.MaxFloat64

	allIdx := c.allIndices
	deleted := c.deleted

	// Inline scoring function with manual binary search (avoids sort.Search closure alloc).
	insertTopK := func(i int, dot float64) {
		if opts.MinScore > 0 && dot < opts.MinScore {
			return
		}
		if len(topBuf) >= effectiveK && dot <= minScore {
			return
		}

		// Binary search: find insertion point where topBuf[pos].score < dot.
		lo, hi := 0, len(topBuf)
		for lo < hi {
			mid := int(uint(lo+hi) >> 1) // avoid overflow
			if topBuf[mid].score < dot {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		topBuf = append(topBuf, scored{})
		copy(topBuf[lo+1:], topBuf[lo:])
		topBuf[lo] = scored{idx: i, score: dot}

		if len(topBuf) > effectiveK {
			topBuf = topBuf[:effectiveK]
		}
		if len(topBuf) == effectiveK {
			minScore = topBuf[effectiveK-1].score
		}
	}

	scoreAndInsert := func(i int) {
		indices := allIdx[i*d : i*d+d : i*d+d]
		var dot0, dot1 float32
		j := 0
		for ; j <= d-8; j += 8 {
			dot0 += qr32[j]*centroids32[indices[j]] +
				qr32[j+1]*centroids32[indices[j+1]] +
				qr32[j+2]*centroids32[indices[j+2]] +
				qr32[j+3]*centroids32[indices[j+3]]
			dot1 += qr32[j+4]*centroids32[indices[j+4]] +
				qr32[j+5]*centroids32[indices[j+5]] +
				qr32[j+6]*centroids32[indices[j+6]] +
				qr32[j+7]*centroids32[indices[j+7]]
		}
		for ; j < d; j++ {
			dot0 += qr32[j] * centroids32[indices[j]]
		}
		insertTopK(i, float64(dot0+dot1))
	}

	// Build candidate set from indexes (IVF + filter).
	var filterCands map[int]struct{}
	if opts.Filter != nil && c.filterIdx != nil {
		filterCands = c.filterCandidates(opts.Filter)
	}

	switch {
	case c.hnswIdx != nil:
		// HNSW path: graph-based search with NEON-accelerated distance.
		ef := opts.Ef
		if ef <= 0 {
			ef = effectiveK * 10 // default: 10x topK for quantized asymmetric scoring
			if ef < 100 {
				ef = 100
			}
		}

		// Use precomputed decoded vectors + NEON for query-to-vector distance.
		hnswVecs := c.hnswVecs
		distToQuery := func(nodeID int) float32 {
			if hnswVecs != nil {
				return distancer.NegDot(qr32, hnswVecs[nodeID*d:nodeID*d+d])
			}
			idx := allIdx[nodeID*d : nodeID*d+d]
			var dot float32
			for j := range d {
				dot += qr32[j] * centroids32[idx[j]]
			}
			return -dot
		}

		hnswResults := c.hnswIdx.Search(distToQuery, effectiveK, ef)

		// Convert HNSW results to topBuf.
		for _, r := range hnswResults {
			i := int(r.id)
			if deleted[i] {
				continue
			}
			// Apply filter if set.
			if filterCands != nil {
				if _, ok := filterCands[i]; !ok {
					continue
				}
			} else if filterFn != nil && !filterFn(c.dataFields[i]) {
				continue
			}
			insertTopK(i, float64(-r.dist)) // convert back to positive score
		}

	case c.ivfIdx != nil:
		// IVF path: iterate partitions directly, no map allocation.
		topPartitions := c.ivfIdx.findNearestPartitions(queryRotated, c.ivfIdx.scoreBuf)

		switch {
		case filterCands != nil:
			// IVF + filter: intersect by checking filter membership.
			c.ivfIdx.forEachCandidate(topPartitions, func(i int) {
				if deleted[i] {
					return
				}
				if _, ok := filterCands[i]; !ok {
					return
				}
				scoreAndInsert(i)
			})
		case filterFn != nil:
			// IVF + brute-force filter (no inverted index for this filter).
			c.ivfIdx.forEachCandidate(topPartitions, func(i int) {
				if deleted[i] {
					return
				}
				if !filterFn(c.dataFields[i]) {
					return
				}
				scoreAndInsert(i)
			})
		default:
			// IVF only, no filter.
			c.ivfIdx.forEachCandidate(topPartitions, func(i int) {
				if deleted[i] {
					return
				}
				scoreAndInsert(i)
			})
		}
	case filterCands != nil:
		// Filter-only path (no IVF).
		for i := range filterCands {
			if deleted[i] {
				continue
			}
			scoreAndInsert(i)
		}
	default:
		// Brute-force path: score all vectors.
		for i := range n {
			if deleted[i] {
				continue
			}
			if filterFn != nil && !filterFn(c.dataFields[i]) {
				continue
			}
			scoreAndInsert(i)
		}
	}

	c.quantizer.PutBuf(queryRotated)

	// Rescore: dequantize top candidates and re-rank with exact cosine similarity.
	// Uses a single reusable buffer for all dequantizations to avoid per-candidate allocation.
	if opts.Rescore > 0 && len(topBuf) > 0 {
		allIdx := c.allIndices
		origDim := c.quantizer.GetConfig().Dim
		bits := c.quantizer.GetConfig().Bits
		recon := make([]float64, origDim) // single buffer reused for all candidates
		cv := tqdb.CompressedVector{Dim: origDim, Bits: bits}
		for k := range topBuf {
			cv.Norm = c.norms[topBuf[k].idx]
			cv.Indices = allIdx[topBuf[k].idx*d : topBuf[k].idx*d+d]
			c.quantizer.DequantizeTo(recon, &cv)
			topBuf[k].score = mathutil.CosineSimilarity(query, recon)
		}
		sort.Slice(topBuf, func(i, j int) bool {
			return topBuf[i].score > topBuf[j].score
		})
	}

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

// ForEach iterates all non-deleted entries, calling fn for each.
// Used for persistence (writing collection state to a Store file).
// The indices slice is a view into internal storage and must not be modified.
func (c *Collection) ForEach(fn func(id string, indices []uint8, norm float32, content string, data map[string]any)) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	d := c.dim
	for i, id := range c.ids {
		if c.deleted[i] {
			continue
		}
		fn(id, c.allIndices[i*d:i*d+d], c.norms[i], c.contents[i], c.dataFields[i])
	}
}

// AddRawDocument stores pre-compressed vector data with content and metadata.
// Unlike AddCompressed, this preserves the document content field.
// Used for loading from a Store file back into a Collection.
// Silently skips if the ID already exists.
func (c *Collection) AddRawDocument(id string, indices []uint8, norm float32, content string, data map[string]any) {
	c.mu.RLock()
	_, exists := c.idIndex[id]
	c.mu.RUnlock()
	if exists {
		return
	}
	c.addCompressed(id, content, indices, norm, data)
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

package tqdb

import (
	"math"
	"sort"
	"sync"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

// Result represents a search result from a Collection or Store.
type Result struct {
	ID       string
	Score    float64 // inner product similarity (≈ cosine sim for unit-normalized vectors)
	Metadata map[string]string
}

// Filter is a predicate for filtering entries during search.
type Filter func(metadata map[string]string) bool

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
	quantizer *TurboQuantMSE
	dim       int

	// Contiguous storage for cache-friendly search.
	// allIndices stores all vectors' indices end-to-end: vec0[0..d-1] | vec1[0..d-1] | ...
	allIndices []uint8
	norms      []float32 // original L2 norms (used for dequantization, not search ranking)
	ids        []string
	metadata   []map[string]string
}

// NewCollection creates a new Collection with the given quantizer config.
func NewCollection(cfg Config) (*Collection, error) {
	q, err := NewMSE(cfg)
	if err != nil {
		return nil, err
	}
	return &Collection{
		quantizer: q,
		dim:       q.rotation.WorkDim(),
	}, nil
}

// Add compresses and stores a vector with its ID and metadata.
func (c *Collection) Add(id string, vec []float64, metadata map[string]string) {
	cv := c.quantizer.Quantize(vec)
	c.addCompressed(id, cv.Indices, cv.Norm, metadata)
}

// AddFloat32 is a convenience wrapper for float32 vectors.
func (c *Collection) AddFloat32(id string, vec []float32, metadata map[string]string) {
	c.Add(id, mathutil.Float32ToFloat64(vec), metadata)
}

// AddCompressed stores a pre-compressed vector.
func (c *Collection) AddCompressed(id string, cv *CompressedVector, metadata map[string]string) {
	c.addCompressed(id, cv.Indices, cv.Norm, metadata)
}

func (c *Collection) addCompressed(id string, indices []uint8, norm float32, metadata map[string]string) {
	c.mu.Lock()
	c.allIndices = append(c.allIndices, indices...)
	c.norms = append(c.norms, norm)
	c.ids = append(c.ids, id)
	c.metadata = append(c.metadata, metadata)
	c.mu.Unlock()
}

// Search finds the top-k most similar vectors to the query.
//
// Scoring uses the inner product in rotated space: ⟨Π·q̂, centroids[idx]⟩
// where q̂ = q/‖q‖. Since stored vectors are unit-normalized before quantization,
// this equals cosine similarity without the noise of centroid-norm correction.
func (c *Collection) Search(query []float64, topK int) []Result {
	return c.searchInternal(query, topK, nil)
}

// SearchWithFilter finds the top-k most similar vectors matching the filter.
func (c *Collection) SearchWithFilter(query []float64, topK int, filter Filter) []Result {
	return c.searchInternal(query, topK, filter)
}

// SearchFloat32 is a convenience wrapper for float32 queries.
func (c *Collection) SearchFloat32(query []float32, topK int) []Result {
	return c.Search(mathutil.Float32ToFloat64(query), topK)
}

func (c *Collection) searchInternal(query []float64, topK int, filter Filter) []Result {
	c.mu.RLock()
	n := len(c.norms)
	if n == 0 {
		c.mu.RUnlock()
		return nil
	}

	d := c.dim
	centroids := c.quantizer.codebook.Centroids

	// Rotate the unit query once — O(d²) or O(d log d) for Hadamard.
	queryRotated := c.quantizer.getBuf()
	queryNorm := mathutil.Norm(query)
	if queryNorm < 1e-15 {
		c.quantizer.putBuf(queryRotated)
		c.mu.RUnlock()
		return nil
	}

	// Normalize query before rotation so the inner product = cosine similarity.
	invQN := 1.0 / queryNorm
	unitQuery := c.quantizer.getBuf()
	for i, v := range query {
		unitQuery[i] = v * invQN
	}
	c.quantizer.rotation.Rotate(queryRotated, unitQuery[:c.quantizer.config.Dim])
	c.quantizer.putBuf(unitQuery)

	// Top-k via sorted-insert.
	type scored struct {
		idx   int
		score float64
	}
	topBuf := make([]scored, 0, topK+1)
	minScore := -math.MaxFloat64

	allIdx := c.allIndices

	for i := range n {
		if filter != nil && !filter(c.metadata[i]) {
			continue
		}

		// Inner product: ⟨Π·q̂, centroids[idx]⟩
		// For unit-normalized stored vectors, this ≈ cosine similarity.
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

		// Fast rejection.
		if len(topBuf) >= topK && dot <= minScore {
			continue
		}

		pos := sort.Search(len(topBuf), func(p int) bool {
			return topBuf[p].score < dot
		})
		topBuf = append(topBuf, scored{})
		copy(topBuf[pos+1:], topBuf[pos:])
		topBuf[pos] = scored{idx: i, score: dot}

		if len(topBuf) > topK {
			topBuf = topBuf[:topK]
		}
		if len(topBuf) == topK {
			minScore = topBuf[topK-1].score
		}
	}

	c.quantizer.putBuf(queryRotated)

	results := make([]Result, len(topBuf))
	for i, s := range topBuf {
		results[i] = Result{
			ID:       c.ids[s.idx],
			Score:    s.score,
			Metadata: c.metadata[s.idx],
		}
	}

	c.mu.RUnlock()
	return results
}

// Len returns the number of vectors in the collection.
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

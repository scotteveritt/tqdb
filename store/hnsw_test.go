package store

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/scotteveritt/tqdb"
)

// --- Unit tests for HNSW core ---

func TestHNSW_Empty(t *testing.T) {
	h := newHNSW(hnswConfig{})
	results := h.Search(func(_ int) float32 { return 0 }, 10, 50)
	if len(results) != 0 {
		t.Errorf("expected 0 results from empty index, got %d", len(results))
	}
}

func TestHNSW_SingleNode(t *testing.T) {
	h := newHNSW(hnswConfig{M: 16, EfConstruction: 100})
	h.Insert(0, func(_, _ int) float32 { return 0 })

	results := h.Search(func(_ int) float32 { return 0.1 }, 1, 10)
	if len(results) != 1 || results[0].id != 0 {
		t.Errorf("expected node 0, got %v", results)
	}
}

func TestHNSW_NeedleInHaystack(t *testing.T) {
	// Insert 1000 random vectors, then search for the needle.
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 32
	n := 1000

	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}

	distFunc := func(a, b int) float32 {
		return negDotF32(vecs[a], vecs[b])
	}

	h := newHNSW(hnswConfig{M: 16, EfConstruction: 200})
	for i := range n {
		h.Insert(i, distFunc)
	}

	// Search for a query that is exactly one of the vectors.
	query := vecs[42]
	distToQuery := func(nodeID int) float32 {
		return negDotF32(query, vecs[nodeID])
	}

	results := h.Search(distToQuery, 10, 100)
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].id != 42 {
		t.Errorf("expected needle (42) as top result, got %d with dist %f", results[0].id, results[0].dist)
	}
}

func TestHNSW_RecallAtK(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running HNSW recall test")
	}
	// Measure recall@10 over multiple queries.
	rng := rand.New(rand.NewPCG(99, 0))
	dim := 64
	n := 5000
	nQueries := 50
	k := 10

	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}

	distFunc := func(a, b int) float32 {
		return negDotF32(vecs[a], vecs[b])
	}

	h := newHNSW(hnswConfig{M: 16, EfConstruction: 200})
	for i := range n {
		h.Insert(i, distFunc)
	}

	var totalRecall float64
	for q := range nQueries {
		query := vecs[q]
		distToQuery := func(nodeID int) float32 {
			return negDotF32(query, vecs[nodeID])
		}

		// Exact brute-force top-k.
		type scored struct {
			id   int
			dist float32
		}
		exact := make([]scored, n)
		for i := range n {
			exact[i] = scored{i, distToQuery(i)}
		}
		sort.Slice(exact, func(i, j int) bool { return exact[i].dist < exact[j].dist })
		truthSet := make(map[uint32]bool, k)
		for i := range k {
			truthSet[uint32(exact[i].id)] = true
		}

		// HNSW search.
		results := h.Search(distToQuery, k, 100)
		hits := 0
		for _, r := range results {
			if truthSet[r.id] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	recall := totalRecall / float64(nQueries)
	t.Logf("HNSW recall@%d = %.1f%% (n=%d, d=%d, M=16, efSearch=100)", k, recall*100, n, dim)
	if recall < 0.90 {
		t.Errorf("recall too low: %.1f%% (expected > 90%%)", recall*100)
	}
}

// --- Integration: HNSW with quantized Collection ---

func TestHNSW_WithCollection(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running HNSW collection test")
	}
	dim := 64
	n := 2000
	k := 10

	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)

	rng := rand.New(rand.NewPCG(42, 0))
	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randomVector(dim, rng)
		coll.Add(fmt.Sprintf("doc-%d", i), vecs[i], nil)
	}

	// Build HNSW index over the quantized vectors.
	d := coll.dim
	centroids32 := coll.quantizer.Codebook().Centroids32
	allIdx := coll.allIndices

	// Distance function: negative inner product in quantized space.
	distFunc := func(a, b int) float32 {
		idxA := allIdx[a*d : a*d+d]
		idxB := allIdx[b*d : b*d+d]
		var dot float32
		for j := range d {
			dot += centroids32[idxA[j]] * centroids32[idxB[j]]
		}
		return -dot // negative because HNSW minimizes distance
	}

	h := newHNSW(hnswConfig{M: 16, EfConstruction: 200})
	for i := range n {
		h.Insert(i, distFunc)
	}

	// Compare HNSW search vs brute-force on the quantized collection.
	var totalRecall float64
	nQueries := 30

	for q := range nQueries {
		query := vecs[q]

		// Brute-force results from collection.
		bruteResults := coll.Search(query, k)
		truthSet := make(map[string]bool, k)
		for _, r := range bruteResults {
			truthSet[r.ID] = true
		}

		// HNSW search: need to compute query distance to each node.
		qr32 := make([]float32, d)
		queryRotated := coll.quantizer.GetBuf()
		qNorm := math.Sqrt(dotF64(query, query))
		unitQ := coll.quantizer.GetBuf()
		for i, v := range query {
			unitQ[i] = v / qNorm
		}
		coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
		for i := range d {
			qr32[i] = float32(queryRotated[i])
		}
		coll.quantizer.PutBuf(unitQ)
		coll.quantizer.PutBuf(queryRotated)

		distToQuery := func(nodeID int) float32 {
			idx := allIdx[nodeID*d : nodeID*d+d]
			var dot float32
			for j := range d {
				dot += qr32[j] * centroids32[idx[j]]
			}
			return -dot
		}

		hnswResults := h.Search(distToQuery, k, 100)

		hits := 0
		for _, r := range hnswResults {
			id := fmt.Sprintf("doc-%d", r.id)
			if truthSet[id] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	recall := totalRecall / float64(nQueries)
	t.Logf("HNSW+quantized recall@%d = %.1f%% (n=%d, d=%d)", k, recall*100, n, dim)
	if recall < 0.85 {
		t.Errorf("recall too low: %.1f%%", recall*100)
	}
}

// --- Benchmarks ---

func BenchmarkHNSW_Build_5K_d64(b *testing.B) {
	benchHNSWBuild(b, 5000, 64)
}

func BenchmarkHNSW_Search_5K_d64(b *testing.B) {
	benchHNSWSearch(b, 5000, 64, 10, 100)
}

func BenchmarkHNSW_Search_5K_d64_ef50(b *testing.B) {
	benchHNSWSearch(b, 5000, 64, 10, 50)
}

// --- Helpers ---

func negDotF32(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return -s
}

func dotF64(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func benchHNSWBuild(b *testing.B, n, dim int) {
	rng := rand.New(rand.NewPCG(42, 0))
	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}
	distFunc := func(a, bb int) float32 { return negDotF32(vecs[a], vecs[bb]) }

	b.ResetTimer()
	for range b.N {
		h := newHNSW(hnswConfig{M: 16, EfConstruction: 200})
		for i := range n {
			h.Insert(i, distFunc)
		}
	}
}

func benchHNSWSearch(b *testing.B, n, dim, k, ef int) {
	rng := rand.New(rand.NewPCG(42, 0))
	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}
	distFunc := func(a, bb int) float32 { return negDotF32(vecs[a], vecs[bb]) }

	h := newHNSW(hnswConfig{M: 16, EfConstruction: 200})
	for i := range n {
		h.Insert(i, distFunc)
	}

	queries := make([][]float32, 100)
	for i := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[i] = q
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := range b.N {
		q := queries[i%len(queries)]
		distToQuery := func(nodeID int) float32 { return negDotF32(q, vecs[nodeID]) }
		_ = h.Search(distToQuery, k, ef)
	}
}

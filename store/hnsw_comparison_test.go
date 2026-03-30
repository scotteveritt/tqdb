package store

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
)

// Comparative benchmarks: Brute-force vs IVF vs HNSW on the same data.

func setupCollectionN(b *testing.B, n, dim, bits int) (*Collection, []float64) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		coll.Add(fmt.Sprintf("%d", i), randomVector(dim, rng), nil)
	}
	return coll, randomVector(dim, rng)
}

// --- 10K d=128 ---

func BenchmarkCompare_BruteForce_10K_d128(b *testing.B) {
	coll, query := setupCollectionN(b, 10_000, 128, 8)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkCompare_HNSW_10K_d128(b *testing.B) {
	coll, query := setupCollectionN(b, 10_000, 128, 8)
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: 10})
	}
}

// --- 50K d=128 ---

func BenchmarkCompare_BruteForce_50K_d128(b *testing.B) {
	coll, query := setupCollectionN(b, 50_000, 128, 8)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkCompare_HNSW_50K_d128(b *testing.B) {
	coll, query := setupCollectionN(b, 50_000, 128, 8)
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: 10})
	}
}

// --- Recall comparison ---

func TestCompare_Recall_10K_d128(t *testing.T) {
	dim := 128
	n := 10_000
	k := 10
	nQueries := 50

	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randomVector(dim, rng)
		coll.Add(fmt.Sprintf("%d", i), vecs[i], nil)
	}

	// Build HNSW index.
	collHNSW, _ := NewCollection(cfg)
	for i := range n {
		collHNSW.Add(fmt.Sprintf("%d", i), vecs[i], nil)
	}
	collHNSW.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})

	var bruteRecall, hnswRecall float64

	for q := range nQueries {
		query := randomVector(dim, rng)
		_ = q

		// Brute-force results (ground truth for quantized search).
		bruteResults := coll.Search(query, k)
		truthSet := make(map[string]bool, k)
		for _, r := range bruteResults {
			truthSet[r.ID] = true
		}

		// HNSW results with higher ef for better recall.
		hnswResults := collHNSW.SearchWithOptions(query, tqdb.SearchOptions{TopK: k, Ef: 200})

		bruteHits := k // brute-force is the reference
		hnswHits := 0
		for _, r := range hnswResults {
			if truthSet[r.ID] {
				hnswHits++
			}
		}

		bruteRecall += float64(bruteHits) / float64(k)
		hnswRecall += float64(hnswHits) / float64(k)
	}

	bruteRecall /= float64(nQueries)
	hnswRecall /= float64(nQueries)

	t.Logf("10K d=128 8-bit:")
	t.Logf("  Brute-force recall: %.1f%% (reference)", bruteRecall*100)
	t.Logf("  HNSW recall:        %.1f%% (vs brute-force)", hnswRecall*100)

	if hnswRecall < 0.90 {
		t.Errorf("HNSW recall too low: %.1f%%", hnswRecall*100)
	}
}

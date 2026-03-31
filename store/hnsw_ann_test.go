package store

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/mathutil"
)

// ANN benchmark: measures recall@10 and QPS for HNSW vs brute-force
// on synthetic data at various scales, matching standard ANN benchmark methodology.
func TestANN_HNSW_vs_BruteForce(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running HNSW ANN benchmark")
	}
	for _, tc := range []struct {
		n, dim, bits int
	}{
		{10_000, 128, 8},
		{10_000, 128, 4},
	} {
		name := fmt.Sprintf("n=%d_d=%d_%dbit", tc.n, tc.dim, tc.bits)
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(42, 0))

			// Generate data.
			cfg := tqdb.Config{Dim: tc.dim, Bits: tc.bits, Rotation: tqdb.RotationHadamard}
			coll, _ := NewCollection(cfg)
			vecs := make([][]float64, tc.n)
			for i := range tc.n {
				vecs[i] = randomVector(tc.dim, rng)
				coll.Add(fmt.Sprintf("%d", i), vecs[i], nil)
			}

			// Build HNSW.
			collHNSW, _ := NewCollection(cfg)
			for i := range tc.n {
				collHNSW.Add(fmt.Sprintf("%d", i), vecs[i], nil)
			}
			collHNSW.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})

			// Generate 100 random queries.
			nQueries := 100
			k := 10
			queries := make([][]float64, nQueries)
			for i := range queries {
				queries[i] = randomVector(tc.dim, rng)
			}

			// Compute exact ground truth via brute-force cosine similarity.
			type scored struct {
				idx   int
				score float64
			}

			// Measure recall at different ef values.
			for _, ef := range []int{50, 100, 200, 400} {
				var totalRecall float64
				for _, query := range queries {
					// Exact top-k via raw cosine similarity.
					exact := make([]scored, tc.n)
					for i, v := range vecs {
						exact[i] = scored{i, mathutil.CosineSimilarity(query, v)}
					}
					sort.Slice(exact, func(a, b int) bool { return exact[a].score > exact[b].score })
					truthSet := make(map[int]bool, k)
					for i := range k {
						truthSet[exact[i].idx] = true
					}

					// HNSW search.
					results := collHNSW.SearchWithOptions(query, tqdb.SearchOptions{TopK: k, Ef: ef})
					hits := 0
					for _, r := range results {
						var idx int
						_, _ = fmt.Sscanf(r.ID, "%d", &idx)
						if truthSet[idx] {
							hits++
						}
					}
					totalRecall += float64(hits) / float64(k)
				}
				recall := totalRecall / float64(nQueries)
				t.Logf("  ef=%d: recall@%d = %.1f%%", ef, k, recall*100)
			}

			// Also measure brute-force recall (against exact cosine).
			var bruteRecall float64
			for _, query := range queries {
				exact := make([]scored, tc.n)
				for i, v := range vecs {
					exact[i] = scored{i, mathutil.CosineSimilarity(query, v)}
				}
				sort.Slice(exact, func(a, b int) bool { return exact[a].score > exact[b].score })
				truthSet := make(map[int]bool, k)
				for i := range k {
					truthSet[exact[i].idx] = true
				}
				results := coll.Search(query, k)
				hits := 0
				for _, r := range results {
					var idx int
					_, _ = fmt.Sscanf(r.ID, "%d", &idx)
					if truthSet[idx] {
						hits++
					}
				}
				bruteRecall += float64(hits) / float64(k)
			}
			bruteRecall /= float64(nQueries)
			t.Logf("  brute-force: recall@%d = %.1f%% (reference)", k, bruteRecall*100)
		})
	}
}

// Suppress unused import warning.
var _ = math.Sqrt

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

// ============================================================================
// Baseline benchmarks — establish current performance for each component.
// Run with: go test -bench=. -benchmem -count=1 ./store/
// ============================================================================

// --- End-to-end search ---

func BenchmarkSearch_BruteForce_10K_d128_4bit(b *testing.B) {
	coll, query := setupBench(b, 10_000, 128, 4, false)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkSearch_IVF_10K_d128_4bit(b *testing.B) {
	coll, query := setupBench(b, 10_000, 128, 4, true)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkSearch_BruteForce_10K_d128_8bit(b *testing.B) {
	coll, query := setupBench(b, 10_000, 128, 8, false)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkSearch_BruteForce_1K_d3072_4bit(b *testing.B) {
	coll, query := setupBench(b, 1_000, 3072, 4, false)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.Search(query, 10)
	}
}

func BenchmarkSearch_Rescore_10K_d128_4bit(b *testing.B) {
	coll, query := setupBench(b, 10_000, 128, 4, true)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: 10, Rescore: 30})
	}
}

// --- Isolated components ---

func BenchmarkScoreVec_d128_N1K(b *testing.B) {
	benchScoreVecRaw(b, 1_000, 128, 4)
}

func BenchmarkScoreVec_d128_N10K(b *testing.B) {
	benchScoreVecRaw(b, 10_000, 128, 4)
}

func BenchmarkScoreVec_d3072_N1K(b *testing.B) {
	benchScoreVecRaw(b, 1_000, 3072, 4)
}

func BenchmarkQueryRotation_d128(b *testing.B) {
	benchQueryRotation(b, 128)
}

func BenchmarkQueryRotation_d3072(b *testing.B) {
	benchQueryRotation(b, 3072)
}

func BenchmarkTopK_SortedInsert_K10_N10K(b *testing.B) {
	benchTopKInsert(b, 10, 10_000)
}

func BenchmarkTopK_SortedInsert_K10_N100K(b *testing.B) {
	benchTopKInsert(b, 10, 100_000)
}

func BenchmarkCosineSimilarity_d128(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	a := randomVector(128, rng)
	c := randomVector(128, rng)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = mathutil.CosineSimilarity(a, c)
	}
}

func BenchmarkNormalizeTo_d128(b *testing.B) {
	rng := rand.New(rand.NewPCG(42, 0))
	src := randomVector(128, rng)
	dst := make([]float64, 128)
	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = mathutil.NormalizeTo(dst, src)
	}
}

func BenchmarkQuantize_d128(b *testing.B) {
	benchQuantize(b, 128, 4)
}

func BenchmarkQuantize_d3072(b *testing.B) {
	benchQuantize(b, 3072, 4)
}

// ============================================================================
// Implementations
// ============================================================================

func setupBench(b *testing.B, n, dim, bits int, ivf bool) (*Collection, []float64) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, err := NewCollection(cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		coll.Add(fmt.Sprintf("%d", i), randomVector(dim, rng), nil)
	}
	if ivf {
		coll.CreateIndex(tqdb.IndexConfig{})
	}
	return coll, randomVector(dim, rng)
}

func benchScoreVecRaw(b *testing.B, n, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, err := NewCollection(cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		coll.Add(fmt.Sprintf("%d", i), randomVector(dim, rng), nil)
	}

	query := randomVector(dim, rng)
	d := coll.dim
	centroids := coll.quantizer.Codebook().Centroids
	allIdx := coll.allIndices

	// Pre-rotate query.
	queryRotated := coll.quantizer.GetBuf()
	qNorm := mathutil.Norm(query)
	unitQ := coll.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v / qNorm
	}
	coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
	coll.quantizer.PutBuf(unitQ)

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		var bestScore float64
		for i := range n {
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
			if dot > bestScore {
				bestScore = dot
			}
		}
		_ = bestScore
	}
	coll.quantizer.PutBuf(queryRotated)
}

func benchQueryRotation(b *testing.B, dim int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard}
	coll, err := NewCollection(cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	query := randomVector(dim, rng)

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		qBuf := coll.quantizer.GetBuf()
		unitQ := coll.quantizer.GetBuf()
		qNorm := mathutil.Norm(query)
		for i, v := range query {
			unitQ[i] = v / qNorm
		}
		coll.quantizer.Rotation().Rotate(qBuf, unitQ[:dim])
		coll.quantizer.PutBuf(unitQ)
		coll.quantizer.PutBuf(qBuf)
	}
}

func benchTopKInsert(b *testing.B, k, n int) {
	rng := rand.New(rand.NewPCG(42, 0))
	scores := make([]float64, n)
	for i := range scores {
		scores[i] = rng.Float64()
	}

	b.ResetTimer()
	b.ReportAllocs()

	type scored struct {
		idx   int
		score float64
	}

	for range b.N {
		topBuf := make([]scored, 0, k+1)
		minScore := -math.MaxFloat64

		for i, s := range scores {
			if len(topBuf) >= k && s <= minScore {
				continue
			}
			pos := sort.Search(len(topBuf), func(p int) bool {
				return topBuf[p].score < s
			})
			topBuf = append(topBuf, scored{})
			copy(topBuf[pos+1:], topBuf[pos:])
			topBuf[pos] = scored{idx: i, score: s}
			if len(topBuf) > k {
				topBuf = topBuf[:k]
			}
			if len(topBuf) == k {
				minScore = topBuf[k-1].score
			}
		}
		_ = topBuf
	}
}

func benchQuantize(b *testing.B, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, err := NewCollection(cfg)
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(42, 0))
	vec := randomVector(dim, rng)

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_ = coll.quantizer.Quantize(vec)
	}
}

// randomVector is declared in store_test.go

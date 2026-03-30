package store

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/scotteveritt/tqdb"
)

// Prototype: float32 scoring
//
// Current: centroids[]float64, queryRotated[]float64 → float64 FMA per element
// Proposed: centroids32[]float32, queryRotated32[]float32 → float32 FMA, float64 accumulator
//
// Benefits: halved centroid table and query vector sizes, better cache utilization.
// Risk: precision loss from float32 multiplication may affect recall.

// --- Benchmarks ---

func BenchmarkScore_Float64_d128_N10K(b *testing.B) {
	benchFloat64Scoring(b, 10_000, 128, 4)
}

func BenchmarkScore_Float32_d128_N10K(b *testing.B) {
	benchFloat32Scoring(b, 10_000, 128, 4)
}

func BenchmarkScore_Float64_d128_8bit_N10K(b *testing.B) {
	benchFloat64Scoring(b, 10_000, 128, 8)
}

func BenchmarkScore_Float32_d128_8bit_N10K(b *testing.B) {
	benchFloat32Scoring(b, 10_000, 128, 8)
}

func BenchmarkScore_Float64_d4096_N1K(b *testing.B) {
	benchFloat64Scoring(b, 1_000, 4096, 4)
}

func BenchmarkScore_Float32_d4096_N1K(b *testing.B) {
	benchFloat32Scoring(b, 1_000, 4096, 4)
}

func BenchmarkScore_Float64_d4096_8bit_N1K(b *testing.B) {
	benchFloat64Scoring(b, 1_000, 4096, 8)
}

func BenchmarkScore_Float32_d4096_8bit_N1K(b *testing.B) {
	benchFloat32Scoring(b, 1_000, 4096, 8)
}

// --- Recall comparison ---

func TestFloat32Scoring_RecallParity(t *testing.T) {
	for _, tc := range []struct {
		dim, bits, n int
	}{
		{128, 4, 10_000},
		{128, 8, 10_000},
		{4096, 4, 1_000},
	} {
		name := fmt.Sprintf("d=%d_%dbit_n=%d", tc.dim, tc.bits, tc.n)
		t.Run(name, func(t *testing.T) {
			cfg := tqdb.Config{Dim: tc.dim, Bits: tc.bits, Rotation: tqdb.RotationHadamard}
			coll, _ := NewCollection(cfg)
			rng := rand.New(rand.NewPCG(42, 0))
			for i := range tc.n {
				coll.Add(fmt.Sprintf("%d", i), randomVector(tc.dim, rng), nil)
			}

			d := coll.dim
			centroids64 := coll.quantizer.Codebook().Centroids
			centroids32 := make([]float32, len(centroids64))
			for i, c := range centroids64 {
				centroids32[i] = float32(c)
			}
			allIdx := coll.allIndices

			// Run 50 queries, compare top-10 rankings.
			nQueries := 50
			k := 10
			var matchCount, totalCount int

			for q := range nQueries {
				_ = q
				query := randomVector(tc.dim, rng)
				queryRotated := coll.quantizer.GetBuf()
				qN := vecNorm(query)
				unitQ := coll.quantizer.GetBuf()
				for i, v := range query {
					unitQ[i] = v / qN
				}
				coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:tc.dim])
				coll.quantizer.PutBuf(unitQ)

				// Float32 version of query.
				queryRotated32 := make([]float32, d)
				for i := range d {
					queryRotated32[i] = float32(queryRotated[i])
				}

				// Score all vectors with both methods.
				type scored struct {
					idx   int
					score float64
				}
				scores64 := make([]scored, tc.n)
				scores32 := make([]scored, tc.n)

				for i := range tc.n {
					indices := allIdx[i*d : i*d+d]

					// Float64 scoring (current).
					var dot64 float64
					for j := range d {
						dot64 += queryRotated[j] * centroids64[indices[j]]
					}
					scores64[i] = scored{i, dot64}

					// Float32 scoring (proposed).
					var dot32 float64 // accumulate in float64
					for j := range d {
						dot32 += float64(queryRotated32[j]) * float64(centroids32[indices[j]])
					}
					scores32[i] = scored{i, dot32}
				}

				// Get top-k for each.
				sort.Slice(scores64, func(a, b int) bool { return scores64[a].score > scores64[b].score })
				sort.Slice(scores32, func(a, b int) bool { return scores32[a].score > scores32[b].score })

				truth := make(map[int]bool, k)
				for i := range k {
					truth[scores64[i].idx] = true
				}
				for i := range k {
					if truth[scores32[i].idx] {
						matchCount++
					}
				}
				totalCount += k

				coll.quantizer.PutBuf(queryRotated)
			}

			recall := float64(matchCount) / float64(totalCount)
			t.Logf("d=%d %d-bit: float32 vs float64 ranking agreement: %.1f%%", tc.dim, tc.bits, recall*100)
			if recall < 0.99 {
				t.Errorf("float32 recall too low: %.1f%% (expected > 99%%)", recall*100)
			}
		})
	}
}

// --- Benchmark implementations ---

func benchFloat64Scoring(b *testing.B, n, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		coll.Add(fmt.Sprintf("%d", i), randomVector(dim, rng), nil)
	}

	query := randomVector(dim, rng)
	d := coll.dim
	centroids := coll.quantizer.Codebook().Centroids
	allIdx := coll.allIndices

	queryRotated := coll.quantizer.GetBuf()
	qN := vecNorm(query)
	unitQ := coll.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v / qN
	}
	coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
	coll.quantizer.PutBuf(unitQ)

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		var bestScore float64
		for i := range n {
			indices := allIdx[i*d : i*d+d : i*d+d]
			var dot0, dot1 float64
			j := 0
			for ; j <= d-8; j += 8 {
				dot0 += queryRotated[j]*centroids[indices[j]] +
					queryRotated[j+1]*centroids[indices[j+1]] +
					queryRotated[j+2]*centroids[indices[j+2]] +
					queryRotated[j+3]*centroids[indices[j+3]]
				dot1 += queryRotated[j+4]*centroids[indices[j+4]] +
					queryRotated[j+5]*centroids[indices[j+5]] +
					queryRotated[j+6]*centroids[indices[j+6]] +
					queryRotated[j+7]*centroids[indices[j+7]]
			}
			for ; j < d; j++ {
				dot0 += queryRotated[j] * centroids[indices[j]]
			}
			if dot0+dot1 > bestScore {
				bestScore = dot0 + dot1
			}
		}
		_ = bestScore
	}
	coll.quantizer.PutBuf(queryRotated)
}

func benchFloat32Scoring(b *testing.B, n, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		coll.Add(fmt.Sprintf("%d", i), randomVector(dim, rng), nil)
	}

	query := randomVector(dim, rng)
	d := coll.dim
	centroids64 := coll.quantizer.Codebook().Centroids
	allIdx := coll.allIndices

	// Convert centroids and query to float32.
	centroids32 := make([]float32, len(centroids64))
	for i, c := range centroids64 {
		centroids32[i] = float32(c)
	}

	queryRotated := coll.quantizer.GetBuf()
	qN := vecNorm(query)
	unitQ := coll.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v / qN
	}
	coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
	coll.quantizer.PutBuf(unitQ)

	qr32 := make([]float32, d)
	for i := range d {
		qr32[i] = float32(queryRotated[i])
	}

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		var bestScore float64
		for i := range n {
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
			score := float64(dot0 + dot1)
			if score > bestScore {
				bestScore = score
			}
		}
		_ = bestScore
	}
	coll.quantizer.PutBuf(queryRotated)
}


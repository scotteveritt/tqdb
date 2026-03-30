package store

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func vecNorm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

// Prototype: precomputed query×centroid table
//
// Instead of: score += queryRotated[j] * centroids[indices[j]]  (gather + FMA)
// Precompute: qc[j*numLevels + k] = queryRotated[j] * centroids[k]  for all j, k
// Then:       score += qc[j*numLevels + indices[j]]              (single lookup, no multiply)
//
// The table is d × numLevels × 8 bytes:
//   4-bit, d=128:  128 × 16 × 8 =  16 KB  (fits L1)
//   4-bit, d=4096: 4096 × 16 × 8 = 512 KB (L2)
//   8-bit, d=128:  128 × 256 × 8 = 256 KB (borderline)
//   8-bit, d=4096: 4096 × 256 × 8 = 8 MB  (too big)

func BenchmarkScoreVec_Current_d128_N10K(b *testing.B) {
	benchCurrentScoring(b, 10_000, 128, 4)
}

func BenchmarkScoreVec_Precomputed_d128_N10K(b *testing.B) {
	benchPrecomputedScoring(b, 10_000, 128, 4)
}

func BenchmarkScoreVec_Current_d128_8bit_N10K(b *testing.B) {
	benchCurrentScoring(b, 10_000, 128, 8)
}

func BenchmarkScoreVec_Precomputed_d128_8bit_N10K(b *testing.B) {
	benchPrecomputedScoring(b, 10_000, 128, 8)
}

func BenchmarkScoreVec_Current_d4096_N1K(b *testing.B) {
	benchCurrentScoring(b, 1_000, 4096, 4)
}

func BenchmarkScoreVec_Precomputed_d4096_N1K(b *testing.B) {
	benchPrecomputedScoring(b, 1_000, 4096, 4)
}

// Current approach: gather + FMA (what we ship today)
func benchCurrentScoring(b *testing.B, n, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		vec := randomVector(dim, rng)
		coll.Add(fmt.Sprintf("%d", i), vec, nil)
	}

	query := randomVector(dim, rng)
	d := coll.dim
	centroids := coll.quantizer.Codebook().Centroids
	allIdx := coll.allIndices

	// Pre-rotate query.
	queryRotated := coll.quantizer.GetBuf()
	qNorm := vecNorm(query)
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
			score := dot0 + dot1
			if score > bestScore {
				bestScore = score
			}
		}
		_ = bestScore
	}
	coll.quantizer.PutBuf(queryRotated)
}

// Precomputed approach: build query×centroid table, then scoring is lookup-only
func benchPrecomputedScoring(b *testing.B, n, dim, bits int) {
	b.Helper()
	cfg := tqdb.Config{Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range n {
		vec := randomVector(dim, rng)
		coll.Add(fmt.Sprintf("%d", i), vec, nil)
	}

	query := randomVector(dim, rng)
	d := coll.dim
	centroids := coll.quantizer.Codebook().Centroids
	numLevels := len(centroids)
	allIdx := coll.allIndices

	// Pre-rotate query.
	queryRotated := coll.quantizer.GetBuf()
	qNorm := vecNorm(query)
	unitQ := coll.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v / qNorm
	}
	coll.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
	coll.quantizer.PutBuf(unitQ)

	// Precompute the query×centroid table.
	// qcTable[j*numLevels + k] = queryRotated[j] * centroids[k]
	qcTable := make([]float64, d*numLevels)
	for j := range d {
		for k := range numLevels {
			qcTable[j*numLevels+k] = queryRotated[j] * centroids[k]
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		var bestScore float64
		for i := range n {
			indices := allIdx[i*d : i*d+d : i*d+d]
			var dot0, dot1 float64
			j := 0
			nl := numLevels
			for ; j <= d-8; j += 8 {
				dot0 += qcTable[j*nl+int(indices[j])] +
					qcTable[(j+1)*nl+int(indices[j+1])] +
					qcTable[(j+2)*nl+int(indices[j+2])] +
					qcTable[(j+3)*nl+int(indices[j+3])]
				dot1 += qcTable[(j+4)*nl+int(indices[j+4])] +
					qcTable[(j+5)*nl+int(indices[j+5])] +
					qcTable[(j+6)*nl+int(indices[j+6])] +
					qcTable[(j+7)*nl+int(indices[j+7])]
			}
			for ; j < d; j++ {
				dot0 += qcTable[j*nl+int(indices[j])]
			}
			score := dot0 + dot1
			if score > bestScore {
				bestScore = score
			}
		}
		_ = bestScore
	}
	coll.quantizer.PutBuf(queryRotated)
}

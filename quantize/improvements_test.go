package quantize

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
)

// ============================================================================
// Hypothesis 1: UseExactPDF (Beta distribution) vs Gaussian codebook
//
// The Gaussian N(0, 1/√d) is the CLT approximation. The exact marginal
// of a rotated unit vector coordinate is Beta-distributed:
//   f(x) ∝ (1-x²)^((d-3)/2)
// For large d these converge, but at d=128 or d=64 the difference may matter.
// PolarQuant uses exact angle distributions at each level.
// ============================================================================

func TestHypothesis1_ExactPDF_vs_Gaussian(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running hypothesis test")
	}
	for _, d := range []int{64, 128, 768, 3072} {
		for _, bits := range []int{4, 5, 8} {
			name := fmt.Sprintf("d=%d_%dbit", d, bits)
			t.Run(name, func(t *testing.T) {
				recallGaussian := measureRecall(d, bits, false, 10000, 100)
				recallExact := measureRecall(d, bits, true, 10000, 100)
				delta := recallExact - recallGaussian
				t.Logf("d=%d %d-bit: Gaussian=%.3f%% Exact=%.3f%% delta=%+.3f%%",
					d, bits, recallGaussian*100, recallExact*100, delta*100)
			})
		}
	}
}

// ============================================================================
// Hypothesis 2: Sub-vector norm groups
//
// PolarQuant stores 8 fp16 radii capturing magnitude across subspaces.
// We store 1 float32 norm, losing all intra-vector magnitude structure.
//
// Idea: after rotation + quantization, compute the actual vs quantized
// norm per group of coordinates. Store a scale factor per group.
// At search time, multiply each group's contribution by its scale factor.
//
// This doesn't change the codebook or quantization — it's a post-hoc
// correction that captures per-subspace distortion.
// ============================================================================

func TestHypothesis2_SubvectorNormCorrection(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running hypothesis test")
	}
	for _, d := range []int{128, 768, 3072} {
		name := fmt.Sprintf("d=%d_4bit", d)
		t.Run(name, func(t *testing.T) {
			recallBaseline := measureRecall(d, 4, false, 10000, 100)
			recallCorrected := measureRecallWithGroupNorms(d, 4, 8, 10000, 100)
			delta := recallCorrected - recallBaseline
			t.Logf("d=%d 4-bit: baseline=%.3f%% groupNorm(k=8)=%.3f%% delta=%+.3f%%",
				d, recallBaseline*100, recallCorrected*100, delta*100)
		})
	}
}

// ============================================================================
// Hypothesis 3: Increased bit-width cost/benefit
//
// Measure the exact recall improvement per extra bit to find the sweet spot.
// ============================================================================

func TestHypothesis3_BitWidthTradeoff(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running hypothesis test")
	}
	for _, d := range []int{128, 3072} {
		for bits := 3; bits <= 8; bits++ {
			name := fmt.Sprintf("d=%d_%dbit", d, bits)
			t.Run(name, func(t *testing.T) {
				recall := measureRecall(d, bits, false, 5000, 100)
				bytesPerVec := codec.NextPow2(d) * bits / 8
				t.Logf("d=%d %d-bit: recall=%.3f%% bytes/vec=%d compression=%.1fx",
					d, bits, recall*100, bytesPerVec, float64(d*8)/float64(bytesPerVec))
			})
		}
	}
}

// ============================================================================
// Measurement infrastructure
// ============================================================================

// measureRecall computes recall@10 using a synthetic dataset.
// Generates n random unit vectors, quantizes them, searches with nQueries queries.
func measureRecall(dim, bits int, useExact bool, n, nQueries int) float64 { //nolint:unparam // nQueries is parameterized for clarity
	rng := rand.New(rand.NewPCG(42, 0))

	tq, err := NewMSE(tqdb.Config{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard,
		Seed: 42, UseExactPDF: useExact,
	})
	if err != nil {
		panic(err)
	}

	// Generate random unit vectors.
	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randUnitVec(dim, rng)
	}

	// Quantize all vectors.
	compressed := make([]*tqdb.CompressedVector, n)
	for i, v := range vecs {
		compressed[i] = tq.Quantize(v)
	}

	// Generate queries and measure recall@10.
	k := 10
	var totalRecall float64

	for q := range nQueries {
		query := randUnitVec(dim, rng)
		_ = q

		// Exact top-k via brute-force cosine similarity.
		type scored struct {
			idx   int
			score float64
		}
		exact := make([]scored, n)
		for i, v := range vecs {
			exact[i] = scored{i, cosineSim(query, v)}
		}
		sort.Slice(exact, func(i, j int) bool { return exact[i].score > exact[j].score })

		truthSet := make(map[int]bool, k)
		for i := range k {
			truthSet[exact[i].idx] = true
		}

		// Approximate top-k via asymmetric scoring.
		approx := make([]scored, n)
		for i, cv := range compressed {
			approx[i] = scored{i, tq.AsymmetricCosineSimilarity(query, cv)}
		}
		sort.Slice(approx, func(i, j int) bool { return approx[i].score > approx[j].score })

		hits := 0
		for i := range k {
			if truthSet[approx[i].idx] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	return totalRecall / float64(nQueries)
}

// measureRecallWithGroupNorms tests the sub-vector norm correction hypothesis.
// After quantization, it computes per-group scale factors and uses them during scoring.
func measureRecallWithGroupNorms(dim, bits, nGroups, n, nQueries int) float64 {
	rng := rand.New(rand.NewPCG(42, 0))

	tq, err := NewMSE(tqdb.Config{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard, Seed: 42,
	})
	if err != nil {
		panic(err)
	}

	workDim := tq.Rotation().WorkDim()
	groupSize := workDim / nGroups

	// Generate random unit vectors.
	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randUnitVec(dim, rng)
	}

	// Quantize and compute per-group scale factors.
	type quantizedVec struct {
		cv          *tqdb.CompressedVector
		groupScales []float64 // nGroups scale factors
	}

	quantized := make([]quantizedVec, n)
	centroids := tq.Codebook().Centroids

	for i, v := range vecs {
		cv := tq.Quantize(v)

		// Compute the actual rotated vector.
		rotBuf := tq.GetBuf()
		unitBuf := tq.GetBuf()
		norm := math.Sqrt(dot(v, v))
		for j := range len(v) {
			unitBuf[j] = v[j] / norm
		}
		tq.Rotation().Rotate(rotBuf, unitBuf[:dim])

		// Compute per-group scale: ||actual_group|| / ||quantized_group||
		scales := make([]float64, nGroups)
		for g := range nGroups {
			start := g * groupSize
			end := start + groupSize
			var actualNormSq, quantNormSq float64
			for j := start; j < end && j < workDim; j++ {
				actualNormSq += rotBuf[j] * rotBuf[j]
				c := centroids[cv.Indices[j]]
				quantNormSq += c * c
			}
			if quantNormSq > 1e-15 {
				scales[g] = math.Sqrt(actualNormSq / quantNormSq)
			} else {
				scales[g] = 1.0
			}
		}

		tq.PutBuf(rotBuf)
		tq.PutBuf(unitBuf)

		quantized[i] = quantizedVec{cv: cv, groupScales: scales}
	}

	// Measure recall using group-corrected scoring.
	k := 10
	var totalRecall float64

	for range nQueries {
		query := randUnitVec(dim, rng)

		// Exact top-k.
		type scored struct {
			idx   int
			score float64
		}
		exact := make([]scored, n)
		for i, v := range vecs {
			exact[i] = scored{i, cosineSim(query, v)}
		}
		sort.Slice(exact, func(i, j int) bool { return exact[i].score > exact[j].score })
		truthSet := make(map[int]bool, k)
		for i := range k {
			truthSet[exact[i].idx] = true
		}

		// Approximate top-k with group norm correction.
		qRot := tq.GetBuf()
		qNorm := math.Sqrt(dot(query, query))
		unitQ := tq.GetBuf()
		for j := range len(query) {
			unitQ[j] = query[j] / qNorm
		}
		tq.Rotation().Rotate(qRot, unitQ[:dim])
		tq.PutBuf(unitQ)

		approx := make([]scored, n)
		for i, qv := range quantized {
			var score float64
			for g := range nGroups {
				start := g * groupSize
				end := start + groupSize
				var groupDot float64
				for j := start; j < end && j < workDim; j++ {
					groupDot += qRot[j] * centroids[qv.cv.Indices[j]]
				}
				score += groupDot * qv.groupScales[g]
			}
			approx[i] = scored{i, score}
		}
		sort.Slice(approx, func(i, j int) bool { return approx[i].score > approx[j].score })

		tq.PutBuf(qRot)

		hits := 0
		for i := range k {
			if truthSet[approx[i].idx] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}

	return totalRecall / float64(nQueries)
}

// --- Helpers ---

func randUnitVec(d int, rng *rand.Rand) []float64 {
	v := make([]float64, d)
	var norm float64
	for i := range v {
		v[i] = rng.NormFloat64()
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= norm
	}
	return v
}

func cosineSim(a, b []float64) float64 {
	var d, na, nb float64
	for i := range a {
		d += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	denom := math.Sqrt(na * nb)
	if denom < 1e-15 {
		return 0
	}
	return d / denom
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

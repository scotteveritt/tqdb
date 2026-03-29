package tqdb

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

func randomUnitVector(d int, rng *rand.Rand) []float64 {
	v := make([]float64, d)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	norm := mathutil.Norm(v)
	for i := range v {
		v[i] /= norm
	}
	return v
}

func randomVector(d int, rng *rand.Rand) []float64 {
	v := make([]float64, d)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	return v
}

func TestMSEDistortionBound(t *testing.T) {
	// Empirical MSE should be less than paper bound: sqrt(3)*pi/2 * (1/4^b)
	d := 128
	n := 1000
	rng := rand.New(rand.NewPCG(123, 0))

	for _, bits := range []int{1, 2, 3, 4} {
		q, err := NewMSE(Config{Dim: d, Bits: bits, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}

		totalMSE := 0.0
		for range n {
			vec := randomUnitVector(d, rng)
			cv := q.Quantize(vec)
			recon := q.Dequantize(cv)

			mse := 0.0
			for i := range d {
				diff := vec[i] - recon[i]
				mse += diff * diff
			}
			mse /= float64(d)
			totalMSE += mse
		}
		avgMSE := totalMSE / float64(n)

		bound := math.Sqrt(3) * math.Pi / 2.0 * math.Pow(4.0, -float64(bits))
		t.Logf("bits=%d: avg MSE per coord = %.6f, paper bound = %.6f (%.1f%%)",
			bits, avgMSE, bound, 100*avgMSE/bound)

		if avgMSE > bound*1.1 {
			t.Errorf("bits=%d: MSE %.6f exceeds paper bound %.6f", bits, avgMSE, bound)
		}
	}
}

func TestMSECosineSimilarityPreservation(t *testing.T) {
	d := 128
	n := 500
	rng := rand.New(rand.NewPCG(456, 0))

	for _, bits := range []int{2, 3, 4} {
		q, err := NewMSE(Config{Dim: d, Bits: bits, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}

		totalSim := 0.0
		for range n {
			vec := randomVector(d, rng)
			cv := q.Quantize(vec)
			recon := q.Dequantize(cv)
			sim := mathutil.CosineSimilarity(vec, recon)
			totalSim += sim
		}
		avgSim := totalSim / float64(n)

		t.Logf("bits=%d: avg cosine similarity = %.6f", bits, avgSim)

		// At 4-bit, expect >= 0.995. At 2-bit, >= 0.90
		minSim := map[int]float64{2: 0.88, 3: 0.96, 4: 0.995}
		if avgSim < minSim[bits] {
			t.Errorf("bits=%d: avg cosine sim %.4f below minimum %.4f", bits, avgSim, minSim[bits])
		}
	}
}

func TestMSEQuantizeFloat32(t *testing.T) {
	d := 64
	q, err := NewMSE(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	vec32 := make([]float32, d)
	for i := range vec32 {
		vec32[i] = float32(i) * 0.1
	}

	cv := q.QuantizeFloat32(vec32)
	recon32 := q.DequantizeFloat32(cv)

	if len(recon32) != d {
		t.Fatalf("got %d elements, want %d", len(recon32), d)
	}

	// Check cosine similarity
	vec64 := mathutil.Float32ToFloat64(vec32)
	recon64 := mathutil.Float32ToFloat64(recon32)
	sim := mathutil.CosineSimilarity(vec64, recon64)
	if sim < 0.99 {
		t.Errorf("float32 roundtrip cosine sim = %f, want >= 0.99", sim)
	}
}

func TestMSECosineSimilarityMethod(t *testing.T) {
	d := 128
	rng := rand.New(rand.NewPCG(789, 0))

	q, err := NewMSE(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	query := randomVector(d, rng)
	doc := randomVector(d, rng)

	cv := q.Quantize(doc)

	// CosineSimilarity method should match manual computation
	sim1 := q.CosineSimilarity(query, cv)
	recon := q.Dequantize(cv)
	sim2 := mathutil.CosineSimilarity(query, recon)

	if math.Abs(sim1-sim2) > 1e-10 {
		t.Errorf("CosineSimilarity=%f vs manual=%f", sim1, sim2)
	}
}

func TestNewMSEValidation(t *testing.T) {
	_, err := NewMSE(Config{Dim: 0, Bits: 4})
	if err == nil {
		t.Error("expected error for Dim=0")
	}

	_, err = NewMSE(Config{Dim: 128, Bits: 0})
	if err != nil {
		t.Error("Bits=0 should default to 4, got error:", err)
	}

	_, err = NewMSE(Config{Dim: 128, Bits: 9})
	if err == nil {
		t.Error("expected error for Bits=9")
	}
}

func TestAsymmetricCosineSimilarityMatchesExact(t *testing.T) {
	d := 128
	q, err := NewMSE(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(999, 0))

	for range 100 {
		query := randomVector(d, rng)
		doc := randomVector(d, rng)
		cv := q.Quantize(doc)

		exact := q.CosineSimilarity(query, cv)
		asym := q.AsymmetricCosineSimilarity(query, cv)

		if math.Abs(exact-asym) > 1e-10 {
			t.Errorf("exact=%f vs asymmetric=%f, diff=%e", exact, asym, exact-asym)
		}
	}
}

func TestAsymmetricCosineSimilarityBatch(t *testing.T) {
	d := 64
	q, err := NewMSE(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(888, 0))

	query := randomVector(d, rng)
	n := 50
	cvs := make([]*CompressedVector, n)
	for i := range n {
		cvs[i] = q.Quantize(randomVector(d, rng))
	}

	results := q.AsymmetricCosineSimilarityBatch(query, cvs)
	if len(results) != n {
		t.Fatalf("got %d results, want %d", len(results), n)
	}

	for i, cv := range cvs {
		single := q.AsymmetricCosineSimilarity(query, cv)
		if math.Abs(results[i]-single) > 1e-10 {
			t.Errorf("batch[%d]=%f vs single=%f", i, results[i], single)
		}
	}
}

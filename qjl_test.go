package tqdb

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

func TestQJLUnbiasedness(t *testing.T) {
	// The TurboQuantProd inner product estimator should be unbiased:
	// E[estimate] ≈ true inner product
	d := 128
	n := 2000
	rng := rand.New(rand.NewPCG(300, 0))

	for _, bits := range []int{2, 3, 4} {
		pq, err := NewProd(Config{Dim: d, Bits: bits, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}

		totalBias := 0.0
		for range n {
			x := randomVector(d, rng)
			y := randomVector(d, rng)

			cv := pq.Quantize(x)
			estimated := pq.InnerProduct(y, cv)
			actual := mathutil.Dot(x, y)

			totalBias += estimated - actual
		}
		avgBias := totalBias / float64(n)

		t.Logf("bits=%d: avg bias = %.6f", bits, avgBias)
		if math.Abs(avgBias) > 0.5 {
			t.Errorf("bits=%d: bias %.6f too large, estimator may not be unbiased", bits, avgBias)
		}
	}
}

func TestMSEOnlyBias(t *testing.T) {
	// MSE-only inner product (without QJL correction) has a systematic
	// scale factor of ~2/π ≈ 0.637 for unit vectors.
	// This test demonstrates the bias that QJL corrects.
	d := 128
	n := 1000
	rng := rand.New(rand.NewPCG(400, 0))

	q, err := NewMSE(Config{Dim: d, Bits: 3, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	totalRatio := 0.0
	count := 0
	for range n {
		x := randomUnitVector(d, rng)
		y := randomUnitVector(d, rng)

		cv := q.Quantize(x)
		recon := q.Dequantize(cv)
		estimated := mathutil.Dot(y, recon)
		actual := mathutil.Dot(x, y)

		if math.Abs(actual) > 0.01 {
			totalRatio += estimated / actual
			count++
		}
	}
	avgRatio := totalRatio / float64(count)

	t.Logf("MSE-only avg ratio estimate/actual = %.4f (expected < 1.0 due to norm shrinkage)", avgRatio)

	// MSE-only should show some shrinkage (ratio < 1.0), though it may not be
	// exactly 2/π since we're dequantizing (which partially restores the norm)
	if avgRatio > 1.05 {
		t.Errorf("MSE-only ratio %.4f > 1.05, unexpected", avgRatio)
	}
}

func TestProdQuantizerValidation(t *testing.T) {
	_, err := NewProd(Config{Dim: 128, Bits: 1})
	if err == nil {
		t.Error("expected error for Bits=1 with TurboQuantProd")
	}

	pq, err := NewProd(Config{Dim: 64, Bits: 4})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(500, 0))
	vec := randomVector(64, rng)
	cv := pq.Quantize(vec)
	recon := pq.Dequantize(cv)

	sim := mathutil.CosineSimilarity(vec, recon)
	t.Logf("TurboQuantProd d=64 bits=4: cosine sim after dequantize = %.6f", sim)
	if sim < 0.90 {
		t.Errorf("cosine sim %.4f too low for TurboQuantProd", sim)
	}
}

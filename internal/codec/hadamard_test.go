package codec

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

func TestFWHTSelfInverse(t *testing.T) {
	for _, n := range []int{4, 8, 16, 64, 256, 1024} {
		rng := rand.New(rand.NewPCG(42, 0))
		x := make([]float64, n)
		orig := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
			orig[i] = x[i]
		}
		fwht(x)
		fwht(x)
		for i := range n {
			if math.Abs(x[i]-orig[i]) > 1e-10 {
				t.Errorf("n=%d: fwht(fwht(x))[%d] = %f, want %f", n, i, x[i], orig[i])
				break
			}
		}
	}
}

func TestFWHTPreservesNorm(t *testing.T) {
	for _, n := range []int{8, 64, 256, 1024} {
		rng := rand.New(rand.NewPCG(42, 0))
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		normBefore := mathutil.Norm(x)
		fwht(x)
		normAfter := mathutil.Norm(x)
		if math.Abs(normBefore-normAfter)/normBefore > 1e-10 {
			t.Errorf("n=%d: norm changed", n)
		}
	}
}

func TestHadamardRotatorRoundtrip(t *testing.T) {
	for _, d := range []int{16, 64, 128, 768, 3072} {
		h := NewHadamardRotator(d, 42)
		rng := rand.New(rand.NewPCG(99, 0))
		vec := make([]float64, d)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}
		rotated := make([]float64, h.WorkDim())
		h.Rotate(rotated, vec)
		recovered := make([]float64, d)
		h.Unrotate(recovered, rotated)
		for i := range d {
			if math.Abs(vec[i]-recovered[i]) > 1e-9 {
				t.Errorf("d=%d: roundtrip mismatch at %d", d, i)
				break
			}
		}
	}
}

func TestHadamardRotatorNormPreservation(t *testing.T) {
	for _, d := range []int{64, 128, 768, 3072} {
		h := NewHadamardRotator(d, 42)
		rng := rand.New(rand.NewPCG(77, 0))
		vec := make([]float64, d)
		for i := range vec {
			vec[i] = rng.NormFloat64()
		}
		normBefore := mathutil.Norm(vec)
		dst := make([]float64, h.WorkDim())
		h.Rotate(dst, vec)
		normAfter := mathutil.Norm(dst)
		if math.Abs(normBefore-normAfter)/normBefore > 1e-9 {
			t.Errorf("d=%d: norm changed by %e", d, math.Abs(normBefore-normAfter)/normBefore)
		}
	}
}

func TestHadamardRotatorDeterministic(t *testing.T) {
	d := 128
	h1 := NewHadamardRotator(d, 42)
	h2 := NewHadamardRotator(d, 42)
	vec := make([]float64, d)
	for i := range vec {
		vec[i] = float64(i) * 0.1
	}
	dst1 := make([]float64, h1.WorkDim())
	dst2 := make([]float64, h2.WorkDim())
	h1.Rotate(dst1, vec)
	h2.Rotate(dst2, vec)
	for i := range h1.WorkDim() {
		if dst1[i] != dst2[i] {
			t.Fatalf("non-deterministic at %d", i)
		}
	}
}

func TestNextPow2(t *testing.T) {
	tests := []struct{ in, want int }{
		{1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8},
		{128, 128}, {129, 256}, {768, 1024}, {3072, 4096},
	}
	for _, tt := range tests {
		if got := NextPow2(tt.in); got != tt.want {
			t.Errorf("NextPow2(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

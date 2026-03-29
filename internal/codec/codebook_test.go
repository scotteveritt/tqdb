package codec

import (
	"math"
	"testing"
)

func TestCodebookSymmetry(t *testing.T) {
	// Centroids should be symmetric around 0 for the symmetric Gaussian distribution
	for _, d := range []int{64, 128, 256, 768} {
		for _, bits := range []int{1, 2, 3, 4} {
			cb := SolveCodebook(d, bits, false)
			sum := 0.0
			for _, c := range cb.Centroids {
				sum += c
			}
			if math.Abs(sum) > 0.01 {
				t.Errorf("d=%d bits=%d: centroid sum = %f, want ~0", d, bits, sum)
			}
		}
	}
}

func TestCodebookConvergence(t *testing.T) {
	// Solver should converge for various (d, bits) pairs
	for _, d := range []int{64, 128, 768, 3072} {
		for _, bits := range []int{2, 3, 4} {
			cb := SolveCodebook(d, bits, false)
			if cb == nil {
				t.Fatalf("d=%d bits=%d: solver returned nil", d, bits)
			}
			if len(cb.Centroids) != 1<<bits {
				t.Errorf("d=%d bits=%d: got %d centroids, want %d", d, bits, len(cb.Centroids), 1<<bits)
			}
			if len(cb.Boundaries) != (1<<bits)-1 {
				t.Errorf("d=%d bits=%d: got %d boundaries, want %d", d, bits, len(cb.Boundaries), (1<<bits)-1)
			}
			// Centroids should be sorted
			for i := 1; i < len(cb.Centroids); i++ {
				if cb.Centroids[i] <= cb.Centroids[i-1] {
					t.Errorf("d=%d bits=%d: centroids not sorted at index %d", d, bits, i)
				}
			}
		}
	}
}

func TestCodebookQuantizeDequantize(t *testing.T) {
	cb := SolveCodebook(128, 4, false)
	sigma := 1.0 / math.Sqrt(128.0)

	// Test that quantize+dequantize produces values close to input
	values := []float64{0.0, sigma, -sigma, 2 * sigma, -2 * sigma}
	indices := cb.Quantize(values)
	recon := cb.Dequantize(indices)

	for i, v := range values {
		diff := math.Abs(v - recon[i])
		if diff > sigma { // should be much less than sigma
			t.Errorf("value %f: reconstructed as %f, diff=%f", v, recon[i], diff)
		}
	}
}

func TestCodebookDistortionBound(t *testing.T) {
	// Paper bound: D_mse <= sqrt(3)*pi/2 * (1/4^b)
	for _, bits := range []int{1, 2, 3, 4} {
		cb := SolveCodebook(128, bits, false)
		bound := math.Sqrt(3) * math.Pi / 2.0 * math.Pow(4.0, -float64(bits))
		if cb.Distortion > bound*1.1 { // 10% tolerance for numerical integration
			t.Errorf("bits=%d: distortion=%f exceeds paper bound=%f", bits, cb.Distortion, bound)
		}
		t.Logf("bits=%d: distortion=%.6f, bound=%.6f (%.1f%% of bound)",
			bits, cb.Distortion, bound, 100*cb.Distortion/bound)
	}
}

func TestBetaPDF(t *testing.T) {
	// Beta PDF should integrate to 1 over [-1, 1]
	for _, d := range []int{10, 64, 128} {
		pdf := betaPDF(d)
		// Simple numerical integration
		n := 10000
		dx := 2.0 / float64(n)
		integral := 0.0
		for i := range n {
			x := -1.0 + (float64(i)+0.5)*dx
			integral += pdf(x) * dx
		}
		if math.Abs(integral-1.0) > 0.01 {
			t.Errorf("d=%d: Beta PDF integral = %f, want 1.0", d, integral)
		}
	}
}

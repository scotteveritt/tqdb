package codec

import (
	"math"
	"testing"
)

func TestRotationOrthogonality(t *testing.T) {
	// Q^T Q should be approximately identity
	for _, d := range []int{16, 64, 128} {
		rot := NewRotationMatrix(d, 42)

		// Check Q^T Q ≈ I by computing a few diagonal and off-diagonal entries
		for i := range d {
			// Diagonal: column i dot column i should be ~1
			colDotSelf := 0.0
			for k := range d {
				colDotSelf += rot.At(k, i) * rot.At(k, i)
			}
			if math.Abs(colDotSelf-1.0) > 1e-10 {
				t.Errorf("d=%d: Q^T*Q[%d,%d] = %f, want 1.0", d, i, i, colDotSelf)
			}

			// Off-diagonal: column i dot column (i+1)%d should be ~0
			j := (i + 1) % d
			colDotOther := 0.0
			for k := range d {
				colDotOther += rot.At(k, i) * rot.At(k, j)
			}
			if math.Abs(colDotOther) > 1e-10 {
				t.Errorf("d=%d: Q^T*Q[%d,%d] = %f, want 0.0", d, i, j, colDotOther)
			}
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	d := 128
	rot := NewRotationMatrix(d, 42)

	// Random vector
	vec := make([]float64, d)
	for i := range vec {
		vec[i] = float64(i*17%13) - 6.0 // deterministic pseudo-random
	}

	normBefore := 0.0
	for _, v := range vec {
		normBefore += v * v
	}
	normBefore = math.Sqrt(normBefore)

	rotated := make([]float64, d)
	rot.Rotate(rotated, vec)

	normAfter := 0.0
	for _, v := range rotated {
		normAfter += v * v
	}
	normAfter = math.Sqrt(normAfter)

	if math.Abs(normBefore-normAfter)/normBefore > 1e-10 {
		t.Errorf("norm changed: before=%f, after=%f", normBefore, normAfter)
	}
}

func TestRotationRoundtrip(t *testing.T) {
	d := 64
	rot := NewRotationMatrix(d, 42)

	vec := make([]float64, d)
	for i := range vec {
		vec[i] = float64(i) * 0.1
	}

	rotated := make([]float64, d)
	rot.Rotate(rotated, vec)

	recovered := make([]float64, d)
	rot.Unrotate(recovered, rotated)

	for i := range d {
		if math.Abs(vec[i]-recovered[i]) > 1e-10 {
			t.Errorf("roundtrip mismatch at %d: want %f, got %f", i, vec[i], recovered[i])
		}
	}
}

func TestRotationDeterministic(t *testing.T) {
	d := 64
	rot1 := NewRotationMatrix(d, 42)
	rot2 := NewRotationMatrix(d, 42)

	for i := range d {
		for j := range d {
			if rot1.At(i, j) != rot2.At(i, j) {
				t.Fatalf("non-deterministic: Q1[%d,%d]=%f != Q2[%d,%d]=%f",
					i, j, rot1.At(i, j), i, j, rot2.At(i, j))
			}
		}
	}
}

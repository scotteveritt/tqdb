package codec

import (
	"math"
	"math/rand/v2"
)

// HadamardRotator applies the Randomized Hadamard Transform:
//
//	R = D₂ · H̃ · D₁
//
// where D₁, D₂ are random ±1 diagonal matrices and H̃ is the
// normalized Walsh-Hadamard transform (self-inverse, orthonormal).
//
// For non-power-of-2 dimensions, input is zero-padded to the next
// power of 2. All padD coordinates are preserved in the rotated output
// to avoid information loss. Dim() returns the original dimension;
// WorkDim() returns the padded dimension.
//
// Memory: O(d) for two sign vectors. No d×d matrix.
// Compute: O(d log d) per rotation via the butterfly FWHT.
// Quality: Empirically better than random QR rotation (QuaRot, ICLR 2024).
type HadamardRotator struct {
	d      int       // original dimension
	padD   int       // padded to next power of 2
	signs1 []float64 // length padD, values ±1.0
	signs2 []float64 // length padD, values ±1.0
	buf    []float64 // scratch buffer for Unrotate, length padD
}

// NewHadamardRotator creates a Randomized Hadamard Transform rotator.
// Deterministic for a given (d, seed) pair.
func NewHadamardRotator(d int, seed uint64) *HadamardRotator {
	padD := NextPow2(d)
	rng := rand.New(rand.NewPCG(seed, 0))

	signs1 := make([]float64, padD)
	signs2 := make([]float64, padD)
	for i := range padD {
		if rng.IntN(2) == 0 {
			signs1[i] = 1.0
		} else {
			signs1[i] = -1.0
		}
		if rng.IntN(2) == 0 {
			signs2[i] = 1.0
		} else {
			signs2[i] = -1.0
		}
	}

	return &HadamardRotator{
		d:      d,
		padD:   padD,
		signs1: signs1,
		signs2: signs2,
		buf:    make([]float64, padD),
	}
}

// Rotate computes dst = D₂ · H̃ · D₁ · [src; 0].
// src has length Dim(), dst has length WorkDim().
func (h *HadamardRotator) Rotate(dst, src []float64) {
	padD := h.padD
	d := h.d

	// Copy src into dst with zero-padding.
	copy(dst[:d], src[:d])
	for i := d; i < padD; i++ {
		dst[i] = 0
	}

	// D₁: apply first random sign flip.
	s1 := h.signs1
	for i := range padD {
		dst[i] *= s1[i]
	}

	// H̃: normalized Walsh-Hadamard transform.
	fwht(dst[:padD])

	// D₂: apply second random sign flip.
	s2 := h.signs2
	for i := range padD {
		dst[i] *= s2[i]
	}
}

// Unrotate computes the inverse: first d coordinates of D₁ · H̃ · D₂ · src.
// src has length WorkDim(), dst has length Dim().
func (h *HadamardRotator) Unrotate(dst, src []float64) {
	padD := h.padD
	d := h.d
	buf := h.buf

	copy(buf[:padD], src[:padD])

	// Reverse order: D₂ first, then H̃, then D₁.
	s2 := h.signs2
	for i := range padD {
		buf[i] *= s2[i]
	}

	fwht(buf)

	s1 := h.signs1
	for i := range padD {
		buf[i] *= s1[i]
	}

	// Take first d coordinates.
	copy(dst[:d], buf[:d])
}

// Dim returns the original (unpadded) dimension.
func (h *HadamardRotator) Dim() int { return h.d }

// WorkDim returns the padded working dimension (next power of 2 >= Dim).
func (h *HadamardRotator) WorkDim() int { return h.padD }

// fwht performs an in-place normalized Fast Walsh-Hadamard Transform.
// len(x) must be a power of 2. The normalized transform is self-inverse:
// fwht(fwht(x)) == x (to floating-point precision).
//
// Complexity: O(n log n) additions/subtractions.
func fwht(x []float64) {
	n := len(x)
	invSqrt2 := 1.0 / math.Sqrt(2.0)
	h := 1
	for h < n {
		for i := 0; i < n; i += h * 2 {
			for j := i; j < i+h; j++ {
				a, b := x[j], x[j+h]
				x[j] = (a + b) * invSqrt2
				x[j+h] = (a - b) * invSqrt2
			}
		}
		h *= 2
	}
}

// NextPow2 returns the smallest power of 2 >= n.
func NextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

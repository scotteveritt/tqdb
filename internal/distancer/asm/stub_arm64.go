//go:build arm64

// To regenerate the assembly:
//   go install github.com/gorse-io/goat@latest
//   PATH="/opt/homebrew/opt/binutils/bin:/opt/homebrew/opt/llvm/bin:$PATH" \
//     goat ../c/dot_neon_arm64.c -O3 -e="--target=arm64" -e="-march=armv8-a+simd+fp" -o .

package asm

import "unsafe"

// DotNeon computes the dot product of two float32 slices using NEON FMA.
func DotNeon(x, y []float32) float32 {
	n := len(x)
	if n < 16 {
		var s float32
		for i := range x {
			s += x[i] * y[i]
		}
		return s
	}
	var res float32
	dot_neon(
		unsafe.Pointer(unsafe.SliceData(x)),
		unsafe.Pointer(unsafe.SliceData(y)),
		unsafe.Pointer(&res),
		unsafe.Pointer(&n),
	)
	return res
}

// NegDotNeon computes the negative dot product (for HNSW distance where lower = better).
func NegDotNeon(x, y []float32) float32 {
	n := len(x)
	if n < 16 {
		var s float32
		for i := range x {
			s += x[i] * y[i]
		}
		return -s
	}
	var res float32
	neg_dot_neon(
		unsafe.Pointer(unsafe.SliceData(x)),
		unsafe.Pointer(unsafe.SliceData(y)),
		unsafe.Pointer(&res),
		unsafe.Pointer(&n),
	)
	return res
}

// VecMulF64 computes dst[i] = a[i] * b[i] for float64 slices.
// Used for Hadamard sign flips.
func VecMulF64(dst, a, b []float64) {
	n := len(a)
	if n < 8 {
		for i := range a {
			dst[i] = a[i] * b[i]
		}
		return
	}
	vec_mul_f64(
		unsafe.Pointer(unsafe.SliceData(dst)),
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(unsafe.SliceData(b)),
		unsafe.Pointer(&n),
	)
}

// VecScaleF64 computes dst[i] = a[i] * scalar for float64 slices.
// Used for normalization.
func VecScaleF64(dst, a []float64, scalar float64) {
	n := len(a)
	if n < 8 {
		for i := range a {
			dst[i] = a[i] * scalar
		}
		return
	}
	vec_scale_f64(
		unsafe.Pointer(unsafe.SliceData(dst)),
		unsafe.Pointer(unsafe.SliceData(a)),
		unsafe.Pointer(&scalar),
		unsafe.Pointer(&n),
	)
}

// DotF64 computes the dot product of two float64 slices.
func DotF64(x, y []float64) float64 {
	n := len(x)
	if n < 8 {
		var s float64
		for i := range x {
			s += x[i] * y[i]
		}
		return s
	}
	var res float64
	dot_f64(
		unsafe.Pointer(unsafe.SliceData(x)),
		unsafe.Pointer(unsafe.SliceData(y)),
		unsafe.Pointer(&res),
		unsafe.Pointer(&n),
	)
	return res
}

// DotPrefetchNeon computes dot product while prefetching the next pair of vectors.
// Set nextA/nextB to nil on the last iteration.
func DotPrefetchNeon(x, y []float32, nextA, nextB []float32) float32 {
	n := len(x)
	if n < 16 {
		var s float32
		for i := range x {
			s += x[i] * y[i]
		}
		return s
	}
	var res float32
	var na, nb unsafe.Pointer
	if len(nextA) > 0 {
		na = unsafe.Pointer(unsafe.SliceData(nextA))
	}
	if len(nextB) > 0 {
		nb = unsafe.Pointer(unsafe.SliceData(nextB))
	}
	dot_prefetch_neon(
		unsafe.Pointer(unsafe.SliceData(x)),
		unsafe.Pointer(unsafe.SliceData(y)),
		unsafe.Pointer(&res),
		unsafe.Pointer(&n),
		na, nb,
	)
	return res
}

// L2Neon computes L2 squared distance using NEON.
func L2Neon(x, y []float32) float32 {
	n := len(x)
	if n < 16 {
		var s float32
		for i := range x {
			d := x[i] - y[i]
			s += d * d
		}
		return s
	}
	var res float32
	l2_neon(
		unsafe.Pointer(unsafe.SliceData(x)),
		unsafe.Pointer(unsafe.SliceData(y)),
		unsafe.Pointer(&res),
		unsafe.Pointer(&n),
	)
	return res
}

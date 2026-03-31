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

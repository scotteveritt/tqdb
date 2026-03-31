// Package distancer provides SIMD-accelerated distance functions.
// Uses NEON on ARM64, with pure Go fallback on other architectures.
package distancer

// --- float32 ---

// Dot computes the dot product of two float32 slices.
func Dot(a, b []float32) float32 { return dotImpl(a, b) }

// NegDot computes the negative dot product (lower = more similar).
func NegDot(a, b []float32) float32 { return negDotImpl(a, b) }

// DotPrefetch computes dot product while prefetching nextA/nextB into L1 cache.
// Pass nil for nextA/nextB on the last iteration.
func DotPrefetch(a, b, nextA, nextB []float32) float32 { return dotPrefetchImpl(a, b, nextA, nextB) }

// L2Squared computes the squared L2 distance.
func L2Squared(a, b []float32) float32 { return l2Impl(a, b) }

// --- float64 ---

// DotF64 computes the dot product of two float64 slices.
func DotF64(a, b []float64) float64 { return dotF64Impl(a, b) }

// VecMulF64 computes dst[i] = a[i] * b[i] (elementwise multiply, float64).
func VecMulF64(dst, a, b []float64) { vecMulF64Impl(dst, a, b) }

// VecScaleF64 computes dst[i] = a[i] * scalar (float64).
func VecScaleF64(dst, a []float64, scalar float64) { vecScaleF64Impl(dst, a, scalar) }

// Pure Go implementations (fallback).
var dotImpl = dotGo
var negDotImpl = negDotGo
var l2Impl = l2Go
var dotPrefetchImpl = dotPrefetchGo
var dotF64Impl = dotF64Go
var vecMulF64Impl = vecMulF64Go
var vecScaleF64Impl = vecScaleF64Go

func dotGo(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func negDotGo(a, b []float32) float32 {
	return -dotGo(a, b)
}

func dotPrefetchGo(a, b, _, _ []float32) float32 {
	return dotGo(a, b) // Go fallback ignores prefetch hints
}

func l2Go(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func dotF64Go(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func vecMulF64Go(dst, a, b []float64) {
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func vecScaleF64Go(dst, a []float64, scalar float64) {
	for i := range a {
		dst[i] = a[i] * scalar
	}
}

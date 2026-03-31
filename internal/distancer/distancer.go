// Package distancer provides SIMD-accelerated distance functions.
// Uses NEON on ARM64, with pure Go fallback on other architectures.
package distancer

// Dot computes the dot product of two float32 slices.
func Dot(a, b []float32) float32 { return dotImpl(a, b) }

// NegDot computes the negative dot product (lower = more similar).
func NegDot(a, b []float32) float32 { return negDotImpl(a, b) }

// L2Squared computes the squared L2 distance.
func L2Squared(a, b []float32) float32 { return l2Impl(a, b) }

// Pure Go implementations (fallback).
var dotImpl = dotGo
var negDotImpl = negDotGo
var l2Impl = l2Go

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

func l2Go(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

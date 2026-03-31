package mathutil

import (
	"math"

	"github.com/scotteveritt/tqdb/internal/distancer"
)

// Dot computes the dot product of two equal-length float64 slices.
// Uses NEON on ARM64 when available.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("mathutil.Dot: mismatched lengths")
	}
	return distancer.DotF64(a, b)
}

// Norm computes the L2 norm of a float64 slice.
func Norm(v []float64) float64 {
	return math.Sqrt(distancer.DotF64(v, v))
}

// NormalizeTo writes the unit vector of src into dst and returns the original norm.
// Uses NEON for the scale operation on ARM64.
func NormalizeTo(dst, src []float64) float64 {
	norm := Norm(src)
	if norm < 1e-15 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	distancer.VecScaleF64(dst, src, 1.0/norm)
	return norm
}

// CosineSimilarity computes the cosine similarity between two equal-length vectors.
// Uses NEON-accelerated dot products on ARM64.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("mathutil.CosineSimilarity: mismatched lengths")
	}
	n := len(a)
	if n == 0 {
		return 0
	}

	dot := distancer.DotF64(a, b)
	normASq := distancer.DotF64(a, a)
	normBSq := distancer.DotF64(b, b)
	denom := math.Sqrt(normASq * normBSq)
	if denom < 1e-30 {
		return 0
	}
	return dot / denom
}

// Float32ToFloat64 converts a []float32 to []float64.
func Float32ToFloat64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}

// Float64ToFloat32 converts a []float64 to []float32.
func Float64ToFloat32(f64 []float64) []float32 {
	f32 := make([]float32, len(f64))
	for i, v := range f64 {
		f32[i] = float32(v)
	}
	return f32
}

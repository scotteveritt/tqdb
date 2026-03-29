package mathutil

import "math"

// Dot computes the dot product of two equal-length float64 slices.
// Panics if lengths differ.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("mathutil.Dot: mismatched lengths")
	}
	n := len(a)
	if n == 0 {
		return 0
	}
	// Bounds-check elimination hint.
	_ = a[n-1]
	_ = b[n-1]

	// 4-way unrolled accumulator reduces dependency chains and
	// gives the compiler room to emit FMA instructions.
	var s0, s1, s2, s3 float64
	i := 0
	for ; i <= n-4; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// Norm computes the L2 norm of a float64 slice.
func Norm(v []float64) float64 {
	return math.Sqrt(Dot(v, v))
}

// NormalizeTo writes the unit vector of src into dst and returns the original norm.
// dst and src must have equal length. If the norm is near-zero, dst is zeroed and 0 is returned.
func NormalizeTo(dst, src []float64) float64 {
	norm := Norm(src)
	if norm < 1e-15 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	invNorm := 1.0 / norm
	for i, v := range src {
		dst[i] = v * invNorm
	}
	return norm
}

// CosineSimilarity computes the cosine similarity between two equal-length vectors.
// Uses a single pass over the data for dot, normA², and normB².
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("mathutil.CosineSimilarity: mismatched lengths")
	}
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]

	var dot, normASq, normBSq float64
	for i := range n {
		ai, bi := a[i], b[i]
		dot += ai * bi
		normASq += ai * ai
		normBSq += bi * bi
	}
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

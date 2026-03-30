package simd_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tphakala/simd/f64"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

// --- Correctness Tests ---

func TestDotProduct_Correctness(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1000, 3072} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			a, b := randVec(n), randVec(n)
			got := f64.DotProduct(a, b)
			want := scalarDot(a, b)
			if !closeEnough(got, want) {
				t.Errorf("n=%d: simd=%.15g, scalar=%.15g, diff=%g", n, got, want, math.Abs(got-want))
			}
		})
	}
}

func TestAdd_Correctness(t *testing.T) {
	for _, n := range []int{1, 7, 8, 128, 1000} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			a, b := randVec(n), randVec(n)
			got := make([]float64, n)
			want := make([]float64, n)
			f64.Add(got, a, b)
			for i := range n {
				want[i] = a[i] + b[i]
			}
			for i := range n {
				if !closeEnough(got[i], want[i]) {
					t.Fatalf("i=%d: got=%g, want=%g", i, got[i], want[i])
				}
			}
		})
	}
}

func TestScale_Correctness(t *testing.T) {
	for _, n := range []int{1, 7, 128, 3072} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			a := randVec(n)
			scalar := 3.14159
			got := make([]float64, n)
			want := make([]float64, n)
			f64.Scale(got, a, scalar)
			for i := range n {
				want[i] = a[i] * scalar
			}
			for i := range n {
				if !closeEnough(got[i], want[i]) {
					t.Fatalf("i=%d: got=%g, want=%g", i, got[i], want[i])
				}
			}
		})
	}
}

func TestEuclideanDistance_Correctness(t *testing.T) {
	for _, n := range []int{1, 8, 128, 1000} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			a, b := randVec(n), randVec(n)
			got := f64.EuclideanDistance(a, b)
			want := scalarEuclidean(a, b)
			if !closeEnough(got, want) {
				t.Errorf("n=%d: simd=%g, scalar=%g", n, got, want)
			}
		})
	}
}

func TestAddScaled_Correctness(t *testing.T) {
	for _, n := range []int{1, 8, 128, 1000} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			dst := randVec(n)
			src := randVec(n)
			alpha := 2.5
			want := make([]float64, n)
			copy(want, dst)
			for i := range n {
				want[i] += alpha * src[i]
			}
			f64.AddScaled(dst, alpha, src)
			for i := range n {
				if !closeEnough(dst[i], want[i]) {
					t.Fatalf("i=%d: got=%g, want=%g", i, dst[i], want[i])
				}
			}
		})
	}
}

// --- GatherDot Test (our custom kernel) ---

func TestGatherDot(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 15, 16, 128, 256, 3072} {
		t.Run(sizeLabel(n), func(t *testing.T) {
			query := randVec(n)
			table := randVec(16) // 4-bit: 16 centroids
			indices := make([]uint8, n)
			rng := rand.New(rand.NewPCG(42, 0))
			for i := range n {
				indices[i] = uint8(rng.IntN(16))
			}

			// SIMD: gather into buffer, then dot product.
			gathered := make([]float64, n)
			for j, idx := range indices {
				gathered[j] = table[idx]
			}
			got := f64.DotProduct(query, gathered)

			// Scalar reference.
			var want float64
			for j, idx := range indices {
				want += query[j] * table[idx]
			}

			if !closeEnough(got, want) {
				t.Errorf("n=%d: simd=%g, scalar=%g, diff=%g", n, got, want, math.Abs(got-want))
			}
		})
	}
}

// --- Benchmarks: SIMD vs Scalar ---

func BenchmarkDotProduct_Scalar_128(b *testing.B)  { benchScalarDot(b, 128) }
func BenchmarkDotProduct_SIMD_128(b *testing.B)    { benchSIMDDot(b, 128) }
func BenchmarkDotProduct_Scalar_3072(b *testing.B) { benchScalarDot(b, 3072) }
func BenchmarkDotProduct_SIMD_3072(b *testing.B)   { benchSIMDDot(b, 3072) }

func BenchmarkGatherDot_Scalar_128(b *testing.B) { benchScalarGatherDot(b, 128) }
func BenchmarkGatherDot_SIMD_128(b *testing.B)   { benchSIMDGatherDot(b, 128) }
func BenchmarkGatherDot_Scalar_3072(b *testing.B) { benchScalarGatherDot(b, 3072) }
func BenchmarkGatherDot_SIMD_3072(b *testing.B)   { benchSIMDGatherDot(b, 3072) }

func BenchmarkScale_Scalar_128(b *testing.B) { benchScalarScale(b, 128) }
func BenchmarkScale_SIMD_128(b *testing.B)   { benchSIMDScale(b, 128) }

func BenchmarkAddScaled_Scalar_128(b *testing.B) { benchScalarAddScaled(b, 128) }
func BenchmarkAddScaled_SIMD_128(b *testing.B)   { benchSIMDAddScaled(b, 128) }

func BenchmarkCosineSim_Scalar_128(b *testing.B) { benchScalarCosineSim(b, 128) }
func BenchmarkCosineSim_SIMD_128(b *testing.B)   { benchSIMDCosineSim(b, 128) }

// Compare against current mathutil.Dot (4-way unrolled scalar).
func BenchmarkDotProduct_MathutilDot_128(b *testing.B)  { benchMathutilDot(b, 128) }
func BenchmarkDotProduct_MathutilDot_3072(b *testing.B) { benchMathutilDot(b, 3072) }

// --- Benchmark implementations ---

func benchScalarDot(b *testing.B, n int) {
	a, c := randVec(n), randVec(n)
	b.ResetTimer()
	for range b.N {
		_ = scalarDot(a, c)
	}
}

func benchSIMDDot(b *testing.B, n int) {
	a, c := randVec(n), randVec(n)
	b.ResetTimer()
	for range b.N {
		_ = f64.DotProduct(a, c)
	}
}

func benchMathutilDot(b *testing.B, n int) {
	a, c := randVec(n), randVec(n)
	b.ResetTimer()
	for range b.N {
		_ = mathutil.Dot(a, c)
	}
}

func benchScalarGatherDot(b *testing.B, n int) {
	query := randVec(n)
	table := randVec(16)
	indices := makeIndices(n, 16)
	b.ResetTimer()
	for range b.N {
		var dot float64
		for j, idx := range indices {
			dot += query[j] * table[idx]
		}
		_ = dot
	}
}

func benchSIMDGatherDot(b *testing.B, n int) {
	query := randVec(n)
	table := randVec(16)
	indices := makeIndices(n, 16)
	gathered := make([]float64, n)
	b.ResetTimer()
	for range b.N {
		for j, idx := range indices {
			gathered[j] = table[idx]
		}
		_ = f64.DotProduct(query, gathered)
	}
}

func benchScalarScale(b *testing.B, n int) {
	src := randVec(n)
	dst := make([]float64, n)
	b.ResetTimer()
	for range b.N {
		for i, v := range src {
			dst[i] = v * 3.14
		}
	}
}

func benchSIMDScale(b *testing.B, n int) {
	src := randVec(n)
	dst := make([]float64, n)
	b.ResetTimer()
	for range b.N {
		f64.Scale(dst, src, 3.14)
	}
}

func benchScalarAddScaled(b *testing.B, n int) {
	dst := randVec(n)
	src := randVec(n)
	b.ResetTimer()
	for range b.N {
		for i := range n {
			dst[i] += 2.5 * src[i]
		}
	}
}

func benchSIMDAddScaled(b *testing.B, n int) {
	dst := randVec(n)
	src := randVec(n)
	b.ResetTimer()
	for range b.N {
		f64.AddScaled(dst, 2.5, src)
	}
}

func benchScalarCosineSim(b *testing.B, n int) {
	a, c := randVec(n), randVec(n)
	b.ResetTimer()
	for range b.N {
		_ = mathutil.CosineSimilarity(a, c)
	}
}

func benchSIMDCosineSim(b *testing.B, n int) {
	a, c := randVec(n), randVec(n)
	b.ResetTimer()
	for range b.N {
		dot := f64.DotProduct(a, c)
		normA := f64.DotProduct(a, a)
		normB := f64.DotProduct(c, c)
		denom := math.Sqrt(normA * normB)
		if denom > 1e-30 {
			_ = dot / denom
		}
	}
}

// --- Helpers ---

func randVec(n int) []float64 {
	rng := rand.New(rand.NewPCG(42, 0))
	v := make([]float64, n)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	return v
}

func makeIndices(n, maxVal int) []uint8 {
	rng := rand.New(rand.NewPCG(42, 0))
	idx := make([]uint8, n)
	for i := range idx {
		idx[i] = uint8(rng.IntN(maxVal))
	}
	return idx
}

func scalarDot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func scalarEuclidean(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

func closeEnough(a, b float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	avg := (math.Abs(a) + math.Abs(b)) / 2
	if avg < 1e-10 {
		return diff < 1e-10
	}
	return diff/avg < 1e-9
}

func sizeLabel(n int) string {
	return fmt.Sprintf("n=%d", n)
}

var _ = fmt.Sprintf // keep fmt import alive

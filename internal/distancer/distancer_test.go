package distancer

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestDot_Correctness(t *testing.T) {
	for _, n := range []int{1, 4, 7, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1000, 3072} {
		rng := rand.New(rand.NewPCG(42, 0))
		a := randF32(n, rng)
		b := randF32(n, rng)

		got := Dot(a, b)
		want := dotGo(a, b)
		if !closeF32(got, want) {
			t.Errorf("n=%d: Dot=%f, want=%f, diff=%f", n, got, want, math.Abs(float64(got-want)))
		}
	}
}

func TestNegDot_Correctness(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	a := randF32(128, rng)
	b := randF32(128, rng)
	got := NegDot(a, b)
	want := -dotGo(a, b)
	if !closeF32(got, want) {
		t.Errorf("NegDot=%f, want=%f", got, want)
	}
}

func TestL2_Correctness(t *testing.T) {
	for _, n := range []int{1, 16, 128, 3072} {
		rng := rand.New(rand.NewPCG(42, 0))
		a := randF32(n, rng)
		b := randF32(n, rng)
		got := L2Squared(a, b)
		want := l2Go(a, b)
		if !closeF32(got, want) {
			t.Errorf("n=%d: L2=%f, want=%f", n, got, want)
		}
	}
}

// --- Benchmarks ---

func BenchmarkDot_Go_128(b *testing.B)   { benchDotGo(b, 128) }
func BenchmarkDot_NEON_128(b *testing.B)  { benchDotNEON(b, 128) }
func BenchmarkDot_Go_3072(b *testing.B)   { benchDotGo(b, 3072) }
func BenchmarkDot_NEON_3072(b *testing.B) { benchDotNEON(b, 3072) }

func BenchmarkL2_Go_128(b *testing.B)   { benchL2Go(b, 128) }
func BenchmarkL2_NEON_128(b *testing.B)  { benchL2NEON(b, 128) }

func benchDotGo(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a, c := randF32(n, rng), randF32(n, rng)
	b.ResetTimer()
	for range b.N {
		_ = dotGo(a, c)
	}
}

func benchDotNEON(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a, c := randF32(n, rng), randF32(n, rng)
	b.ResetTimer()
	for range b.N {
		_ = Dot(a, c) // uses NEON via dispatch if available
	}
}

func benchL2Go(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a, c := randF32(n, rng), randF32(n, rng)
	b.ResetTimer()
	for range b.N {
		_ = l2Go(a, c)
	}
}

func benchL2NEON(b *testing.B, n int) {
	rng := rand.New(rand.NewPCG(42, 0))
	a, c := randF32(n, rng), randF32(n, rng)
	b.ResetTimer()
	for range b.N {
		_ = L2Squared(a, c)
	}
}

// --- Helpers ---

func randF32(n int, rng *rand.Rand) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func closeF32(a, b float32) bool {
	diff := math.Abs(float64(a - b))
	avg := (math.Abs(float64(a)) + math.Abs(float64(b))) / 2
	if avg < 1e-6 {
		return diff < 1e-6
	}
	return diff/avg < 1e-4 // float32 has ~7 digits, allow 0.01% error
}

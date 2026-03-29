package kvcache

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func randomVector(d int, rng *rand.Rand) []float64 {
	v := make([]float64, d)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	return v
}

func TestKVCacheBasic(t *testing.T) {
	kv, err := New(tqdb.KVCacheConfig{
		Layers:   2,
		Heads:    4,
		HeadDim:  64,
		Bits:     4,
		Rotation: tqdb.RotationHadamard,
	})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(42, 0))
	for range 100 {
		kv.AppendKey(0, 0, randomVector(64, rng))
		kv.AppendValue(0, 0, randomVector(64, rng))
	}

	if kv.SeqLen(0, 0) != 100 {
		t.Errorf("SeqLen = %d, want 100", kv.SeqLen(0, 0))
	}
	if kv.SeqLen(0, 1) != 0 {
		t.Errorf("SeqLen(0,1) = %d, want 0", kv.SeqLen(0, 1))
	}
}

func TestKVCacheNeedleInHaystack(t *testing.T) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 64, Bits: 4, Rotation: tqdb.RotationHadamard,
	})

	rng := rand.New(rand.NewPCG(99, 0))

	for range 500 {
		kv.AppendKey(0, 0, randomVector(64, rng))
		kv.AppendValue(0, 0, randomVector(64, rng))
	}

	needle := randomVector(64, rng)
	kv.AppendKey(0, 0, needle)
	kv.AppendValue(0, 0, randomVector(64, rng))

	for range 500 {
		kv.AppendKey(0, 0, randomVector(64, rng))
		kv.AppendValue(0, 0, randomVector(64, rng))
	}

	scores := kv.AttentionScores(0, 0, needle)
	if len(scores) != 1001 {
		t.Fatalf("got %d scores, want 1001", len(scores))
	}

	maxIdx := 0
	maxScore := scores[0]
	for i, s := range scores {
		if s > maxScore {
			maxScore = s
			maxIdx = i
		}
	}

	if maxIdx != 500 {
		t.Errorf("needle at position 500, but argmax is %d (score=%.4f, needle score=%.4f)",
			maxIdx, maxScore, scores[500])
	}
}

func TestKVCacheGetValue(t *testing.T) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 64, Bits: 4, Rotation: tqdb.RotationHadamard,
	})

	rng := rand.New(rand.NewPCG(77, 0))
	original := randomVector(64, rng)

	kv.AppendKey(0, 0, randomVector(64, rng))
	kv.AppendValue(0, 0, original)

	recovered := kv.GetValue(0, 0, 0)
	if len(recovered) != 64 {
		t.Fatalf("got %d elements, want 64", len(recovered))
	}

	var dot, normA, normB float64
	for i := range 64 {
		dot += original[i] * recovered[i]
		normA += original[i] * original[i]
		normB += recovered[i] * recovered[i]
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if sim < 0.99 {
		t.Errorf("value cosine sim = %.4f, want >= 0.99", sim)
	}
}

func TestKVCacheMemoryUsage(t *testing.T) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 32, Heads: 32, HeadDim: 128, Bits: 4, Rotation: tqdb.RotationHadamard,
	})

	rng := rand.New(rand.NewPCG(42, 0))
	for range 100 {
		for l := range 32 {
			for h := range 32 {
				kv.AppendKey(l, h, randomVector(128, rng))
				kv.AppendValue(l, h, randomVector(128, rng))
			}
		}
	}

	tqMem := kv.MemoryUsage()
	fp16Mem := kv.MemoryUsageFP16()
	ratio := float64(fp16Mem) / float64(tqMem)

	t.Logf("100 tokens, 32 layers, 32 heads, d=128:")
	t.Logf("  TQ cache:  %d MB", tqMem/(1<<20))
	t.Logf("  FP16 cache: %d MB", fp16Mem/(1<<20))
	t.Logf("  Compression: %.1fx", ratio)

	if ratio < 1.8 {
		t.Errorf("compression ratio %.1fx too low", ratio)
	}
}

// --- Packed mode tests ---

func TestKVCachePackedNeedle(t *testing.T) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 64, Bits: 4,
		Rotation: tqdb.RotationHadamard, PackIndices: true,
	})

	rng := rand.New(rand.NewPCG(99, 0))

	for range 500 {
		kv.AppendKey(0, 0, randomVector(64, rng))
		kv.AppendValue(0, 0, randomVector(64, rng))
	}

	needle := randomVector(64, rng)
	kv.AppendKey(0, 0, needle)
	kv.AppendValue(0, 0, randomVector(64, rng))

	for range 500 {
		kv.AppendKey(0, 0, randomVector(64, rng))
		kv.AppendValue(0, 0, randomVector(64, rng))
	}

	scores := kv.AttentionScores(0, 0, needle)
	maxIdx := 0
	for i, s := range scores {
		if s > scores[maxIdx] {
			maxIdx = i
		}
	}

	if maxIdx != 500 {
		t.Errorf("packed: needle at 500, argmax is %d", maxIdx)
	}
}

func TestKVCachePackedMemorySavings(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))

	// Unpacked
	kvUnpacked, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 128, Bits: 4,
		Rotation: tqdb.RotationHadamard, PackIndices: false,
	})
	// Packed
	kvPacked, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 128, Bits: 4,
		Rotation: tqdb.RotationHadamard, PackIndices: true,
	})

	for range 1000 {
		key := randomVector(128, rng)
		val := randomVector(128, rng)
		kvUnpacked.AppendKey(0, 0, key)
		kvUnpacked.AppendValue(0, 0, val)
		kvPacked.AppendKey(0, 0, key)
		kvPacked.AppendValue(0, 0, val)
	}

	unpackedMem := kvUnpacked.MemoryUsage()
	packedMem := kvPacked.MemoryUsage()
	ratio := float64(unpackedMem) / float64(packedMem)

	t.Logf("1000 tokens, d=128, 4-bit:")
	t.Logf("  Unpacked: %d KB", unpackedMem/1024)
	t.Logf("  Packed:   %d KB", packedMem/1024)
	t.Logf("  Savings:  %.1fx", ratio)

	if ratio < 1.8 {
		t.Errorf("packing ratio %.1fx too low, expected ~2x", ratio)
	}
}

func TestKVCachePackedGetValue(t *testing.T) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 64, Bits: 4,
		Rotation: tqdb.RotationHadamard, PackIndices: true,
	})

	rng := rand.New(rand.NewPCG(55, 0))
	original := randomVector(64, rng)

	kv.AppendKey(0, 0, randomVector(64, rng))
	kv.AppendValue(0, 0, original)

	recovered := kv.GetValue(0, 0, 0)

	var dot, normA, normB float64
	for i := range 64 {
		dot += original[i] * recovered[i]
		normA += original[i] * original[i]
		normB += recovered[i] * recovered[i]
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if sim < 0.99 {
		t.Errorf("packed GetValue cosine sim = %.4f, want >= 0.99", sim)
	}
}

func BenchmarkKVCacheAppendKey(b *testing.B) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 128, Bits: 4, Rotation: tqdb.RotationHadamard,
	})
	rng := rand.New(rand.NewPCG(42, 0))
	key := randomVector(128, rng)
	b.ResetTimer()
	for range b.N {
		kv.AppendKey(0, 0, key)
	}
}

func BenchmarkKVCacheAttentionScores(b *testing.B) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 128, Bits: 4, Rotation: tqdb.RotationHadamard,
	})
	rng := rand.New(rand.NewPCG(42, 0))
	for range 1000 {
		kv.AppendKey(0, 0, randomVector(128, rng))
	}
	query := randomVector(128, rng)
	b.ResetTimer()
	for range b.N {
		kv.AttentionScores(0, 0, query)
	}
}

func BenchmarkKVCacheAttentionScoresPacked(b *testing.B) {
	kv, _ := New(tqdb.KVCacheConfig{
		Layers: 1, Heads: 1, HeadDim: 128, Bits: 4,
		Rotation: tqdb.RotationHadamard, PackIndices: true,
	})
	rng := rand.New(rand.NewPCG(42, 0))
	for range 1000 {
		kv.AppendKey(0, 0, randomVector(128, rng))
	}
	query := randomVector(128, rng)
	b.ResetTimer()
	for range b.N {
		kv.AttentionScores(0, 0, query)
	}
}

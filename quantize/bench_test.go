package quantize

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
)

func BenchmarkCodebookSolve(b *testing.B) {
	for _, d := range []int{128, 768, 3072} {
		b.Run(fmt.Sprintf("d=%d_4bit", d), func(b *testing.B) {
			for range b.N {
				codec.SolveCodebook(d, 4, false)
			}
		})
	}
}

func BenchmarkQuantize(b *testing.B) {
	for _, d := range []int{128, 768} {
		for _, bits := range []int{3, 4} {
			name := fmt.Sprintf("d=%d_%dbit", d, bits)
			b.Run(name, func(b *testing.B) {
				q, _ := NewMSE(tqdb.Config{Dim: d, Bits: bits, Seed: 42})
				rng := rand.New(rand.NewPCG(42, 0))
				vec := randomVector(d, rng)
				b.ResetTimer()
				for range b.N {
					q.Quantize(vec)
				}
			})
		}
	}
}

func BenchmarkDequantize(b *testing.B) {
	for _, d := range []int{128, 768} {
		name := fmt.Sprintf("d=%d_4bit", d)
		b.Run(name, func(b *testing.B) {
			q, _ := NewMSE(tqdb.Config{Dim: d, Bits: 4, Seed: 42})
			rng := rand.New(rand.NewPCG(42, 0))
			vec := randomVector(d, rng)
			cv := q.Quantize(vec)
			b.ResetTimer()
			for range b.N {
				q.Dequantize(cv)
			}
		})
	}
}

func BenchmarkPack4Bit(b *testing.B) {
	indices := make([]uint8, 3072)
	for i := range indices {
		indices[i] = uint8(i % 16)
	}
	dst := make([]byte, codec.PackedSize(3072, 4))
	b.ResetTimer()
	for range b.N {
		codec.Pack4BitTo(dst, indices)
	}
}

func BenchmarkMarshalBinary(b *testing.B) {
	q, _ := NewMSE(tqdb.Config{Dim: 3072, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(42, 0))
	vec := randomVector(3072, rng)
	cv := q.Quantize(vec)
	b.ResetTimer()
	for range b.N {
		_, _ = cv.MarshalBinary()
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	q, _ := NewMSE(tqdb.Config{Dim: 128, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(42, 0))
	query := randomVector(128, rng)
	cv := q.Quantize(randomVector(128, rng))
	b.ResetTimer()
	for range b.N {
		q.CosineSimilarity(query, cv)
	}
}

func BenchmarkAsymmetricCosineSimilarity(b *testing.B) {
	q, _ := NewMSE(tqdb.Config{Dim: 128, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(42, 0))
	query := randomVector(128, rng)
	cv := q.Quantize(randomVector(128, rng))
	b.ResetTimer()
	for range b.N {
		q.AsymmetricCosineSimilarity(query, cv)
	}
}

func BenchmarkAsymmetricBatch100(b *testing.B) {
	q, _ := NewMSE(tqdb.Config{Dim: 128, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(42, 0))
	query := randomVector(128, rng)
	cvs := make([]*tqdb.CompressedVector, 100)
	for i := range cvs {
		cvs[i] = q.Quantize(randomVector(128, rng))
	}
	b.ResetTimer()
	for range b.N {
		q.AsymmetricCosineSimilarityBatch(query, cvs)
	}
}

func BenchmarkQuantizeHadamard(b *testing.B) {
	for _, d := range []int{128, 768} {
		name := fmt.Sprintf("d=%d_4bit", d)
		b.Run(name, func(b *testing.B) {
			q, _ := NewMSE(tqdb.Config{Dim: d, Bits: 4, Seed: 42, Rotation: tqdb.RotationHadamard})
			rng := rand.New(rand.NewPCG(42, 0))
			vec := randomVector(d, rng)
			b.ResetTimer()
			for range b.N {
				q.Quantize(vec)
			}
		})
	}
}

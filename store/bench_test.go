package store

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func BenchmarkCollectionSearch(b *testing.B) {
	for _, n := range []int{1000, 10000} {
		name := fmt.Sprintf("d=128_4bit_n=%d", n)
		b.Run(name, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(42, 0))
			coll, _ := NewCollection(tqdb.Config{Dim: 128, Bits: 4, Seed: 42})
			for i := range n {
				_ = coll.Add(fmt.Sprintf("doc-%d", i), randomVector(128, rng), nil)
			}
			query := randomVector(128, rng)
			b.ResetTimer()
			for range b.N {
				coll.Search(query, 10)
			}
		})
	}
}

func BenchmarkCollectionSearchHadamard(b *testing.B) {
	for _, n := range []int{1000, 10000} {
		name := fmt.Sprintf("d=128_4bit_n=%d", n)
		b.Run(name, func(b *testing.B) {
			rng := rand.New(rand.NewPCG(42, 0))
			coll, _ := NewCollection(tqdb.Config{Dim: 128, Bits: 4, Seed: 42, Rotation: tqdb.RotationHadamard})
			for i := range n {
				_ = coll.Add(fmt.Sprintf("doc-%d", i), randomVector(128, rng), nil)
			}
			query := randomVector(128, rng)
			b.ResetTimer()
			for range b.N {
				coll.Search(query, 10)
			}
		})
	}
}

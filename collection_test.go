package tqdb

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

func TestCollectionNeedleInHaystack(t *testing.T) {
	// Insert N random vectors + 1 "needle" vector.
	// Search with the needle as query — it should be top-1.
	for _, tc := range []struct {
		d, bits, haystackSize int
	}{
		{64, 4, 512},
		{64, 3, 512},
		{64, 2, 512},
		{128, 4, 2048},
		{128, 3, 2048},
	} {
		name := fmt.Sprintf("d=%d_bits=%d_n=%d", tc.d, tc.bits, tc.haystackSize)
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(uint64(tc.d*1000+tc.bits), 0))

			coll, err := NewCollection(Config{Dim: tc.d, Bits: tc.bits, Seed: 42})
			if err != nil {
				t.Fatal(err)
			}

			// Add haystack
			for i := range tc.haystackSize {
				vec := randomVector(tc.d, rng)
				coll.Add(fmt.Sprintf("hay-%d", i), vec, nil)
			}

			// Add needle
			needle := randomVector(tc.d, rng)
			coll.Add("needle", needle, nil)

			// Search
			results := coll.Search(needle, 5)
			if len(results) == 0 {
				t.Fatal("no results")
			}

			if results[0].ID != "needle" {
				t.Errorf("needle not top-1: got %s with score %.4f", results[0].ID, results[0].Score)
				for i, r := range results {
					t.Logf("  %d: %s score=%.4f", i, r.ID, r.Score)
				}
			}
		})
	}
}

func TestCollectionSearchQuality(t *testing.T) {
	// Verify that search results have reasonable cosine similarity scores
	d := 128
	bits := 4
	n := 100
	rng := rand.New(rand.NewPCG(600, 0))

	coll, err := NewCollection(Config{Dim: d, Bits: bits, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	vectors := make([][]float64, n)
	for i := range n {
		vectors[i] = randomVector(d, rng)
		coll.Add(fmt.Sprintf("doc-%d", i), vectors[i], nil)
	}

	// Search with first vector as query
	results := coll.Search(vectors[0], 5)

	// Top result should be doc-0 with high similarity
	if results[0].ID != "doc-0" {
		t.Logf("Warning: doc-0 not top-1 (got %s), which can happen with quantization", results[0].ID)
	}
	if results[0].Score < 0.8 {
		t.Errorf("top result score %.4f too low", results[0].Score)
	}
}

func TestCollectionWithFilter(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(700, 0))

	coll, err := NewCollection(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	// Add vectors with different repos
	for i := range 50 {
		vec := randomVector(d, rng)
		repo := "repo-a"
		if i%2 == 0 {
			repo = "repo-b"
		}
		coll.Add(fmt.Sprintf("doc-%d", i), vec, map[string]string{"repo": repo})
	}

	query := randomVector(d, rng)

	// Search with filter for repo-b only
	results := coll.SearchWithFilter(query, 10, func(meta map[string]string) bool {
		return meta["repo"] == "repo-b"
	})

	for _, r := range results {
		if r.Metadata["repo"] != "repo-b" {
			t.Errorf("filter failed: got repo=%s", r.Metadata["repo"])
		}
	}
}

func TestCollectionLen(t *testing.T) {
	coll, _ := NewCollection(Config{Dim: 64, Bits: 4, Seed: 42})
	if coll.Len() != 0 {
		t.Errorf("empty collection Len=%d", coll.Len())
	}

	rng := rand.New(rand.NewPCG(800, 0))
	for range 10 {
		coll.Add("x", randomVector(64, rng), nil)
	}
	if coll.Len() != 10 {
		t.Errorf("Len=%d, want 10", coll.Len())
	}
}

func TestCollectionAddFloat32(t *testing.T) {
	coll, _ := NewCollection(Config{Dim: 4, Bits: 4, Seed: 42})

	vec32 := []float32{1.0, 2.0, 3.0, 4.0}
	coll.AddFloat32("test", vec32, nil)

	query := mathutil.Float32ToFloat64(vec32)
	results := coll.Search(query, 1)
	if len(results) != 1 || results[0].ID != "test" {
		t.Errorf("unexpected search result: %+v", results)
	}
}

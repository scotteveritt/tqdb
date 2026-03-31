package store

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func TestIncrementalHNSW_AddAfterIndex(t *testing.T) {
	dim := 64
	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))

	// Add initial 500 vectors, build HNSW.
	for i := range 500 {
		coll.Add(fmt.Sprintf("initial-%d", i), randomVector(dim, rng), nil)
	}
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})

	// Add 100 more vectors AFTER index creation.
	needle := randomVector(dim, rng)
	coll.Add("needle", needle, nil)
	for i := range 99 {
		coll.Add(fmt.Sprintf("post-%d", i), randomVector(dim, rng), nil)
	}

	// Search for the needle - it should be found via HNSW graph.
	results := coll.SearchWithOptions(needle, tqdb.SearchOptions{TopK: 5, Ef: 200})
	if len(results) == 0 {
		t.Fatal("no results")
	}

	found := false
	for _, r := range results {
		if r.ID == "needle" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("needle not found in top-5 results after incremental HNSW insert")
		for i, r := range results {
			t.Logf("  result %d: %s (score %.4f)", i, r.ID, r.Score)
		}
	}
}

func TestIncrementalIVF_AddAfterIndex(t *testing.T) {
	dim := 64
	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))

	// Add initial 500 vectors, build IVF.
	for i := range 500 {
		coll.Add(fmt.Sprintf("initial-%d", i), randomVector(dim, rng), nil)
	}
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexIVF})

	// Add the needle AFTER index creation.
	needle := randomVector(dim, rng)
	coll.Add("needle", needle, nil)

	// Search - needle should be found via IVF partition assignment.
	results := coll.Search(needle, 5)
	if len(results) == 0 {
		t.Fatal("no results")
	}

	found := false
	for _, r := range results {
		if r.ID == "needle" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("needle not found in top-5 results after incremental IVF insert")
		for i, r := range results {
			t.Logf("  result %d: %s (score %.4f)", i, r.ID, r.Score)
		}
	}
}

func TestIncrementalHNSW_DeleteAfterIndex(t *testing.T) {
	dim := 64
	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))

	// Add vectors including one we'll search for then delete.
	target := randomVector(dim, rng)
	coll.Add("target", target, nil)
	for i := range 499 {
		coll.Add(fmt.Sprintf("other-%d", i), randomVector(dim, rng), nil)
	}
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})

	// Verify target is found before deletion.
	results := coll.SearchWithOptions(target, tqdb.SearchOptions{TopK: 1, Ef: 200})
	if len(results) == 0 || results[0].ID != "target" {
		t.Fatal("target not found before deletion")
	}

	// Delete target.
	coll.Delete("target")

	// Search again - target should NOT appear.
	results = coll.SearchWithOptions(target, tqdb.SearchOptions{TopK: 5, Ef: 200})
	for _, r := range results {
		if r.ID == "target" {
			t.Errorf("deleted target still appears in HNSW results")
		}
	}
}

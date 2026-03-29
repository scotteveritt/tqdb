package store

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb"
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

			coll, err := NewCollection(tqdb.Config{Dim: tc.d, Bits: tc.bits, Seed: 42})
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

	coll, err := NewCollection(tqdb.Config{Dim: d, Bits: bits, Seed: 42})
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

	coll, err := NewCollection(tqdb.Config{Dim: d, Bits: 4, Seed: 42})
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
		coll.Add(fmt.Sprintf("doc-%d", i), vec, map[string]any{"repo": repo})
	}

	query := randomVector(d, rng)

	// Search with filter for repo-b only
	results := coll.SearchWithOptions(query, tqdb.SearchOptions{
		TopK:   10,
		Filter: tqdb.Eq("repo", "repo-b"),
	})

	for _, r := range results {
		if r.Data["repo"] != "repo-b" {
			t.Errorf("filter failed: got repo=%s", r.Data["repo"])
		}
	}
}

func TestCollectionLen(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	if coll.Len() != 0 {
		t.Errorf("empty collection Len=%d", coll.Len())
	}

	rng := rand.New(rand.NewPCG(800, 0))
	for i := range 10 {
		coll.Add(fmt.Sprintf("x-%d", i), randomVector(64, rng), nil)
	}
	if coll.Len() != 10 {
		t.Errorf("Len=%d, want 10", coll.Len())
	}
}

func TestCollectionAddFloat32(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 4, Bits: 4, Seed: 42})

	vec32 := []float32{1.0, 2.0, 3.0, 4.0}
	coll.AddFloat32("test", vec32, nil)

	query := mathutil.Float32ToFloat64(vec32)
	results := coll.Search(query, 1)
	if len(results) != 1 || results[0].ID != "test" {
		t.Errorf("unexpected search result: %+v", results)
	}
}

// --- New CRUD tests ---

func TestCollectionAddDocument(t *testing.T) {
	coll, err := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(900, 0))
	vec := randomVector(64, rng)

	err = coll.AddDocument(context.Background(), tqdb.Document{
		ID:        "doc-1",
		Content:   "hello world",
		Data:      map[string]any{"lang": "go"},
		Embedding: vec,
	})
	if err != nil {
		t.Fatal(err)
	}

	if coll.Count() != 1 {
		t.Errorf("Count=%d, want 1", coll.Count())
	}

	doc, ok := coll.GetByID("doc-1")
	if !ok {
		t.Fatal("GetByID returned false")
	}
	if doc.Content != "hello world" {
		t.Errorf("Content=%q, want %q", doc.Content, "hello world")
	}
	if doc.Data["lang"] != "go" {
		t.Errorf("Data[lang]=%v, want go", doc.Data["lang"])
	}
}

func TestCollectionAddDocuments(t *testing.T) {
	coll, err := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(901, 0))
	docs := make([]tqdb.Document, 10)
	for i := range docs {
		docs[i] = tqdb.Document{
			ID:        fmt.Sprintf("doc-%d", i),
			Content:   fmt.Sprintf("content %d", i),
			Data:      map[string]any{"idx": float64(i)},
			Embedding: randomVector(64, rng),
		}
	}

	err = coll.AddDocuments(context.Background(), docs, 4)
	if err != nil {
		t.Fatal(err)
	}

	if coll.Count() != 10 {
		t.Errorf("Count=%d, want 10", coll.Count())
	}

	for i := range 10 {
		doc, ok := coll.GetByID(fmt.Sprintf("doc-%d", i))
		if !ok {
			t.Errorf("doc-%d not found", i)
		}
		if doc.Content != fmt.Sprintf("content %d", i) {
			t.Errorf("doc-%d content=%q", i, doc.Content)
		}
	}
}

func TestCollectionGetByID(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(902, 0))

		coll.Add("a", randomVector(64, rng), map[string]any{"key": "val"})

	doc, ok := coll.GetByID("a")
	if !ok {
		t.Fatal("not found")
	}
	if doc.ID != "a" {
		t.Errorf("ID=%q, want a", doc.ID)
	}
	if doc.Data["key"] != "val" {
		t.Errorf("Data[key]=%v, want val", doc.Data["key"])
	}

	_, ok = coll.GetByID("nonexistent")
	if ok {
		t.Error("expected not found for nonexistent ID")
	}
}

func TestCollectionDelete(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(903, 0))

	for i := range 5 {
		coll.Add(fmt.Sprintf("doc-%d", i), randomVector(64, rng), nil)
	}

	if coll.Count() != 5 {
		t.Fatalf("Count=%d, want 5", coll.Count())
	}

	_ = coll.Delete("doc-2", "doc-4")

	if coll.Count() != 3 {
		t.Errorf("Count after delete=%d, want 3", coll.Count())
	}

	_, ok := coll.GetByID("doc-2")
	if ok {
		t.Error("deleted doc-2 should not be found")
	}

	// Deleted entries should not appear in search results.
	query := randomVector(64, rng)
	results := coll.Search(query, 10)
	for _, r := range results {
		if r.ID == "doc-2" || r.ID == "doc-4" {
			t.Errorf("deleted ID %q appeared in search results", r.ID)
		}
	}
}

func TestCollectionUpsert(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(904, 0))

	vec1 := randomVector(64, rng)
	coll.Upsert("doc-1", vec1, map[string]any{"version": "v1"})

	doc, ok := coll.GetByID("doc-1")
	if !ok {
		t.Fatal("not found after upsert")
	}
	if doc.Data["version"] != "v1" {
		t.Errorf("version=%v, want v1", doc.Data["version"])
	}

	// Upsert same ID with new data.
	vec2 := randomVector(64, rng)
	coll.Upsert("doc-1", vec2, map[string]any{"version": "v2"})

	doc, ok = coll.GetByID("doc-1")
	if !ok {
		t.Fatal("not found after second upsert")
	}
	if doc.Data["version"] != "v2" {
		t.Errorf("version=%v, want v2", doc.Data["version"])
	}

	// Should be 1 active entry, not 2.
	if coll.Count() != 1 {
		t.Errorf("Count=%d, want 1", coll.Count())
	}
}

func TestCollectionUpsertDocument(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(905, 0))

	err := coll.UpsertDocument(context.Background(), tqdb.Document{
		ID:        "doc-1",
		Content:   "first",
		Embedding: randomVector(64, rng),
	})
	if err != nil {
		t.Fatal(err)
	}

	err = coll.UpsertDocument(context.Background(), tqdb.Document{
		ID:        "doc-1",
		Content:   "second",
		Embedding: randomVector(64, rng),
	})
	if err != nil {
		t.Fatal(err)
	}

	doc, ok := coll.GetByID("doc-1")
	if !ok {
		t.Fatal("not found")
	}
	if doc.Content != "second" {
		t.Errorf("Content=%q, want second", doc.Content)
	}
	if coll.Count() != 1 {
		t.Errorf("Count=%d, want 1", coll.Count())
	}
}

func TestCollectionListIDs(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	rng := rand.New(rand.NewPCG(906, 0))

	for i := range 5 {
		coll.Add(fmt.Sprintf("doc-%d", i), randomVector(64, rng), nil)
	}
	_ = coll.Delete("doc-2")

	ids := coll.ListIDs()
	if len(ids) != 4 {
		t.Errorf("ListIDs len=%d, want 4", len(ids))
	}

	idSet := make(map[string]bool)
	for _, id := range ids {
		idSet[id] = true
	}
	if idSet["doc-2"] {
		t.Error("deleted doc-2 should not be in ListIDs")
	}
}

func TestCollectionSearchWithOptions(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(907, 0))

	coll, _ := NewCollection(tqdb.Config{Dim: d, Bits: 4, Seed: 42})

	for i := range 50 {
		repo := "repo-a"
		if i%2 == 0 {
			repo = "repo-b"
		}
		coll.Add(fmt.Sprintf("doc-%d", i), randomVector(d, rng), map[string]any{"repo": repo})
	}

	query := randomVector(d, rng)

	// Test with Filter.
	results := coll.SearchWithOptions(query, tqdb.SearchOptions{
		TopK:   10,
		Filter: tqdb.Eq("repo", "repo-b"),
	})
	for _, r := range results {
		if r.Data["repo"] != "repo-b" {
			t.Errorf("filter failed: got repo=%v", r.Data["repo"])
		}
	}

	// Test with Offset.
	all := coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: 10})
	page2 := coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: 5, Offset: 5})
	if len(all) >= 10 && len(page2) > 0 {
		if all[5].ID != page2[0].ID {
			t.Errorf("offset mismatch: all[5]=%s, page2[0]=%s", all[5].ID, page2[0].ID)
		}
	}
}

func TestCollectionSemanticSearch(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(908, 0))

	// Create a simple embedding function that always returns the same vector.
	fixedVec := randomVector(d, rng)
	embedFunc := func(_ context.Context, _ string) ([]float32, error) {
		f32 := make([]float32, d)
		for i, v := range fixedVec {
			f32[i] = float32(v)
		}
		return f32, nil
	}

	coll, err := NewCollectionWithConfig(CollectionConfig{
		Config:    tqdb.Config{Dim: d, Bits: 4, Seed: 42},
		EmbedFunc: embedFunc,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Add a document using the embedding function.
	err = coll.AddDocument(context.Background(), tqdb.Document{
		ID:      "doc-1",
		Content: "hello world",
	})
	if err != nil {
		t.Fatal(err)
	}

	// Semantic search.
	results, err := coll.SemanticSearch(context.Background(), "anything", tqdb.SearchOptions{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].ID != "doc-1" {
		t.Errorf("expected doc-1, got %s", results[0].ID)
	}
}

func TestCollectionSemanticSearchNoEmbedFunc(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 64, Bits: 4, Seed: 42})
	_, err := coll.SemanticSearch(context.Background(), "test", tqdb.SearchOptions{TopK: 5})
	if err == nil {
		t.Error("expected error when no EmbeddingFunc")
	}
}

func TestCollectionQuery(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(909, 0))

	coll, _ := NewCollection(tqdb.Config{Dim: d, Bits: 4, Seed: 42})

	for i := range 20 {
		lang := "go"
		if i%3 == 0 {
			lang = "python"
		}
		coll.Add(fmt.Sprintf("doc-%d", i), randomVector(d, rng), map[string]any{"lang": lang})
	}

	results := coll.Query(tqdb.QueryOptions{
		PageSize: 100,
		Filter:   tqdb.Eq("lang", "python"),
	})

	for _, r := range results {
		if r.Data["lang"] != "python" {
			t.Errorf("query filter failed: got lang=%v", r.Data["lang"])
		}
	}
	// docs 0, 3, 6, 9, 12, 15, 18 = 7 docs.
	if len(results) != 7 {
		t.Errorf("query result count=%d, want 7", len(results))
	}
}

func TestCollectionMinScore(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(910, 0))

	coll, _ := NewCollection(tqdb.Config{Dim: d, Bits: 4, Seed: 42})

	for i := range 100 {
		coll.Add(fmt.Sprintf("doc-%d", i), randomVector(d, rng), nil)
	}

	query := randomVector(d, rng)

	// With a very high MinScore, should get few/no results.
	results := coll.SearchWithOptions(query, tqdb.SearchOptions{
		TopK:     50,
		MinScore: 0.99,
	})

	for _, r := range results {
		if r.Score < 0.99 {
			t.Errorf("MinScore filter failed: score=%.4f", r.Score)
		}
	}
}

func TestCollectionDeletedExcludedFromSearch(t *testing.T) {
	d := 64
	rng := rand.New(rand.NewPCG(911, 0))

	coll, _ := NewCollection(tqdb.Config{Dim: d, Bits: 4, Seed: 42})

	// Add a needle that's very similar to our query.
	needle := randomVector(d, rng)
	coll.Add("needle", needle, nil)

	for i := range 50 {
		coll.Add(fmt.Sprintf("hay-%d", i), randomVector(d, rng), nil)
	}

	// Verify needle is top-1 before deletion.
	results := coll.Search(needle, 1)
	if len(results) > 0 && results[0].ID != "needle" {
		t.Logf("needle not top-1 before delete (may happen with quantization), got %s", results[0].ID)
	}

	// Delete the needle.
	_ = coll.Delete("needle")

	// Needle should not appear in results.
	results = coll.Search(needle, 10)
	for _, r := range results {
		if r.ID == "needle" {
			t.Error("deleted needle appeared in search results")
		}
	}
}

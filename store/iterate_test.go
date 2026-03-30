package store

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func TestCollectionForEach(t *testing.T) {
	coll, _ := NewCollection(tqdb.Config{Dim: 128, Bits: 4, Rotation: tqdb.RotationHadamard})
	rng := rand.New(rand.NewPCG(42, 0))

	// Add 50 docs with content and metadata.
	for i := range 50 {
		vec := randomVector(128, rng)
		data := map[string]any{"repo": "test-repo", "idx": fmt.Sprintf("%d", i)}
		coll.Add(fmt.Sprintf("doc-%d", i), vec, data)
	}

	// Delete a few.
	if err := coll.Delete("doc-5", "doc-10", "doc-25"); err != nil {
		t.Fatal(err)
	}

	// ForEach should visit exactly 47 entries.
	var visited int
	var sawContent bool
	coll.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		visited++
		if id == "doc-5" || id == "doc-10" || id == "doc-25" {
			t.Errorf("ForEach visited deleted entry %s", id)
		}
		if len(indices) != coll.dim {
			t.Errorf("indices length %d, want %d", len(indices), coll.dim)
		}
		if norm <= 0 {
			t.Errorf("norm %f should be > 0", norm)
		}
		if data == nil {
			t.Error("data should not be nil")
		}
		if data["repo"] != "test-repo" {
			t.Errorf("data[repo] = %v, want test-repo", data["repo"])
		}
		_ = content // content is empty in Add() path, that's ok
		_ = sawContent
	})

	if visited != 47 {
		t.Errorf("ForEach visited %d entries, want 47", visited)
	}
}

func TestCollectionForEachWithContent(t *testing.T) {
	cfg := tqdb.Config{Dim: 32, Bits: 4, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollectionWithConfig(CollectionConfig{Config: cfg})

	// Use addCompressed directly to set content.
	rng := rand.New(rand.NewPCG(42, 0))
	vec := randomVector(32, rng)
	cv := coll.quantizer.Quantize(vec)
	coll.addCompressed("doc-with-content", "hello world", cv.Indices, cv.Norm, map[string]any{"k": "v"})

	var gotContent string
	coll.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		gotContent = content
	})
	if gotContent != "hello world" {
		t.Errorf("ForEach content = %q, want %q", gotContent, "hello world")
	}
}

func TestCollectionAddRawDocument(t *testing.T) {
	cfg := tqdb.Config{Dim: 128, Bits: 4, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)

	// Add a normal doc to get valid indices.
	rng := rand.New(rand.NewPCG(42, 0))
	vec := randomVector(128, rng)
	cv := coll.quantizer.Quantize(vec)

	// Now add via AddRawDocument.
	data := map[string]any{"repo": "myrepo", "language": "go"}
	coll.AddRawDocument("raw-doc", cv.Indices, cv.Norm, "some code content", data)

	if coll.Count() != 1 {
		t.Fatalf("Count = %d, want 1", coll.Count())
	}

	// Verify retrievable.
	doc, ok := coll.GetByID("raw-doc")
	if !ok {
		t.Fatal("GetByID returned false")
	}
	if doc.Content != "some code content" {
		t.Errorf("Content = %q, want %q", doc.Content, "some code content")
	}
	if doc.Data["repo"] != "myrepo" {
		t.Errorf("Data[repo] = %v, want myrepo", doc.Data["repo"])
	}

	// Duplicate should be silently skipped.
	coll.AddRawDocument("raw-doc", cv.Indices, cv.Norm, "different", nil)
	if coll.Count() != 1 {
		t.Errorf("Count after duplicate = %d, want 1", coll.Count())
	}
}

func TestStoreAddRawAndForEachCompressed(t *testing.T) {
	dir := t.TempDir()
	tqPath := filepath.Join(dir, "test.tq")

	cfg := tqdb.Config{Dim: 128, Bits: 4, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)

	// Add docs to collection.
	rng := rand.New(rand.NewPCG(42, 0))
	for i := range 20 {
		vec := randomVector(128, rng)
		data := map[string]any{"repo": "testrepo", "file": fmt.Sprintf("file%d.go", i)}
		coll.Add(fmt.Sprintf("doc-%d", i), vec, data)
	}

	// Write collection to store via ForEach + AddRaw.
	storeCfg := tqdb.StoreConfig{Dim: 128, Bits: 4, Rotation: tqdb.RotationHadamard}
	s, err := Create(tqPath, storeCfg)
	if err != nil {
		t.Fatal(err)
	}

	var written int
	coll.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		if err := s.AddRaw(id, indices, norm, content, data); err != nil {
			t.Fatal(err)
		}
		written++
	})
	if written != 20 {
		t.Fatalf("wrote %d, want 20", written)
	}

	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify file exists and has reasonable size.
	fi, err := os.Stat(tqPath)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Size() < 100 {
		t.Errorf("file too small: %d bytes", fi.Size())
	}

	// Read back via ForEachCompressed.
	s2, err := Open(tqPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s2.Close() }()

	if s2.Len() != 20 {
		t.Fatalf("store len = %d, want 20", s2.Len())
	}

	var read int
	seenIDs := map[string]bool{}
	s2.ForEachCompressed(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		read++
		seenIDs[id] = true
		if len(indices) != s2.workDim {
			t.Errorf("indices len %d, want %d", len(indices), s2.workDim)
		}
		if norm <= 0 {
			t.Errorf("norm %f should be > 0", norm)
		}
		if data["repo"] != "testrepo" {
			t.Errorf("data[repo] = %v, want testrepo", data["repo"])
		}
	})
	if read != 20 {
		t.Errorf("read %d entries, want 20", read)
	}
	for i := range 20 {
		if !seenIDs[fmt.Sprintf("doc-%d", i)] {
			t.Errorf("missing doc-%d", i)
		}
	}
}

func TestRoundtripCollectionViaStore(t *testing.T) {
	// The full roundtrip: Collection -> Store file -> new Collection
	dir := t.TempDir()
	tqPath := filepath.Join(dir, "roundtrip.tq")

	cfg := tqdb.Config{Dim: 64, Bits: 4, Rotation: tqdb.RotationHadamard}

	// Create and populate original collection.
	coll1, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(99, 0))
	query := randomVector(64, rng) // save for search later
	for i := range 100 {
		vec := randomVector(64, rng)
		data := map[string]any{"idx": fmt.Sprintf("%d", i), "lang": "go"}
		coll1.Add(fmt.Sprintf("d%d", i), vec, data)
	}

	// Persist: Collection -> Store.
	storeCfg := tqdb.StoreConfig{Dim: 64, Bits: 4, Rotation: tqdb.RotationHadamard}
	ws, _ := Create(tqPath, storeCfg)
	coll1.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		if err := ws.AddRaw(id, indices, norm, content, data); err != nil {
			t.Fatal(err)
		}
	})
	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}

	// Load: Store -> new Collection.
	coll2, _ := NewCollection(cfg)
	rs, _ := Open(tqPath)
	rs.ForEachCompressed(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		coll2.AddRawDocument(id, indices, norm, content, data)
	})
	if err := rs.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify counts match.
	if coll2.Count() != coll1.Count() {
		t.Fatalf("roundtrip count %d != original %d", coll2.Count(), coll1.Count())
	}

	// Search both and verify same top result.
	r1 := coll1.Search(query, 5)
	r2 := coll2.Search(query, 5)

	if len(r1) == 0 || len(r2) == 0 {
		t.Fatal("search returned no results")
	}
	if r1[0].ID != r2[0].ID {
		t.Errorf("top result mismatch: original=%s, roundtrip=%s", r1[0].ID, r2[0].ID)
	}

	// Verify metadata survives roundtrip.
	doc, ok := coll2.GetByID("d0")
	if !ok {
		t.Fatal("d0 not found in roundtrip collection")
	}
	if doc.Data["lang"] != "go" {
		t.Errorf("metadata lang = %v, want go", doc.Data["lang"])
	}
}

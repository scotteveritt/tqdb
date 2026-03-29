package tqdb

import (
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func TestFormatHeaderRoundtrip(t *testing.T) {
	hdr := &fileHeader{
		Dim:       3072,
		WorkDim:   4096,
		Bits:      4,
		Rotation:  1,
		UseExact:  0,
		Seed:      42,
		NumVecs:   10000,
		NormsOff:  64 + 10000*4096,
		CNormsOff: 64 + 10000*4096 + 10000*4,
		IDsOff:    64 + 10000*4096 + 10000*8,
		MetaOff:   64 + 10000*4096 + 10000*8 + 50000,
	}

	buf := make([]byte, fileHeaderSize)
	encodeHeader(buf, hdr)

	decoded, err := decodeHeader(buf)
	if err != nil {
		t.Fatal(err)
	}

	if decoded.Dim != hdr.Dim || decoded.WorkDim != hdr.WorkDim ||
		decoded.Bits != hdr.Bits || decoded.Rotation != hdr.Rotation ||
		decoded.Seed != hdr.Seed || decoded.NumVecs != hdr.NumVecs ||
		decoded.NormsOff != hdr.NormsOff || decoded.CNormsOff != hdr.CNormsOff ||
		decoded.IDsOff != hdr.IDsOff || decoded.MetaOff != hdr.MetaOff {
		t.Errorf("header mismatch:\n  encoded: %+v\n  decoded: %+v", hdr, decoded)
	}
}

func TestFormatHeaderMagic(t *testing.T) {
	buf := make([]byte, fileHeaderSize)
	copy(buf[0:4], "XXXX")
	_, err := decodeHeader(buf)
	if err == nil {
		t.Error("expected error for bad magic")
	}
}

func TestStoreCreateFlush(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.tq")

	store, err := CreateStore(path, StoreConfig{
		Dim: 64, Bits: 4, Rotation: RotationHadamard,
	})
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(42, 0))
	for i := range 100 {
		vec := randomVector(64, rng)
		meta := map[string]string{"idx": string(rune('0' + i%10))}
		if err := store.Add(
			"doc-"+string(rune('A'+i%26)),
			vec,
			meta,
		); err != nil {
			t.Fatal(err)
		}
	}

	if store.Len() != 100 {
		t.Errorf("Len=%d, want 100", store.Len())
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify file exists and is non-empty.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() < fileHeaderSize {
		t.Errorf("file too small: %d bytes", info.Size())
	}

	// Verify no temp file left behind.
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error("temp file not cleaned up")
	}
}

func TestStoreEmptyFlush(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.tq")

	store, _ := CreateStore(path, StoreConfig{Dim: 32, Bits: 4})
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	// Should be able to open an empty store.
	store2, err := OpenStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store2.Close() //nolint:errcheck

	if store2.Len() != 0 {
		t.Errorf("expected 0 vectors, got %d", store2.Len())
	}

	results := store2.Search(make([]float64, 32), 10)
	if len(results) != 0 {
		t.Errorf("expected 0 results from empty store, got %d", len(results))
	}
}

func TestStoreRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roundtrip.tq")

	rng := rand.New(rand.NewPCG(123, 0))
	d := 64

	// Write.
	store, _ := CreateStore(path, StoreConfig{
		Dim: d, Bits: 4, Rotation: RotationHadamard,
	})
	vecs := make([][]float64, 500)
	for i := range 500 {
		vecs[i] = randomVector(d, rng)
		_ = store.Add("doc-"+string(rune('A'+i%26))+string(rune('0'+i%10)), vecs[i], nil)
	}
	_ = store.Close()

	// Read.
	store2, err := OpenStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store2.Close() //nolint:errcheck

	if store2.Len() != 500 {
		t.Fatalf("Len=%d, want 500", store2.Len())
	}

	// Search with first vector — should find itself or very similar.
	results := store2.Search(vecs[0], 5)
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].Score < 0.9 {
		t.Errorf("top score %.4f too low", results[0].Score)
	}
}

func TestStoreSearchNeedle(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "needle.tq")

	rng := rand.New(rand.NewPCG(456, 0))
	d := 128

	store, _ := CreateStore(path, StoreConfig{
		Dim: d, Bits: 4, Rotation: RotationHadamard,
	})

	// Add haystack.
	for i := range 1000 {
		_ = store.Add(
			"hay-"+string(rune('0'+i/100))+string(rune('0'+(i/10)%10))+string(rune('0'+i%10)),
			randomVector(d, rng),
			nil,
		)
	}

	// Add needle.
	needle := randomVector(d, rng)
	_ = store.Add("needle", needle, nil)
	_ = store.Close()

	// Reopen and search.
	store2, _ := OpenStore(path)
	defer store2.Close() //nolint:errcheck

	results := store2.Search(needle, 5)
	if len(results) == 0 {
		t.Fatal("no results")
	}
	if results[0].ID != "needle" {
		t.Errorf("needle not top-1: got %s score=%.4f", results[0].ID, results[0].Score)
	}
}

func TestStoreSearchFilter(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "filter.tq")

	rng := rand.New(rand.NewPCG(789, 0))
	d := 64

	store, _ := CreateStore(path, StoreConfig{
		Dim: d, Bits: 4, Rotation: RotationHadamard,
	})

	for i := range 100 {
		repo := "repo-a"
		if i%2 == 0 {
			repo = "repo-b"
		}
		_ = store.Add("doc", randomVector(d, rng), map[string]string{"repo": repo})
	}
	_ = store.Close()

	store2, _ := OpenStore(path)
	defer store2.Close() //nolint:errcheck

	results := store2.SearchWithFilter(randomVector(d, rng), 10, func(meta map[string]string) bool {
		return meta["repo"] == "repo-b"
	})

	for _, r := range results {
		if r.Metadata["repo"] != "repo-b" {
			t.Errorf("filter failed: got repo=%s", r.Metadata["repo"])
		}
	}
}

func TestStoreInfo(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "info.tq")

	store, _ := CreateStore(path, StoreConfig{
		Dim: 128, Bits: 4, Rotation: RotationHadamard, Seed: 99,
	})
	rng := rand.New(rand.NewPCG(42, 0))
	for range 50 {
		_ = store.Add("x", randomVector(128, rng), nil)
	}
	_ = store.Close()

	store2, _ := OpenStore(path)
	defer store2.Close() //nolint:errcheck

	info := store2.Info()
	if info.Dim != 128 {
		t.Errorf("Dim=%d, want 128", info.Dim)
	}
	if info.WorkDim != 128 { // 128 is power of 2
		t.Errorf("WorkDim=%d, want 128", info.WorkDim)
	}
	if info.Bits != 4 {
		t.Errorf("Bits=%d, want 4", info.Bits)
	}
	if info.Rotation != RotationHadamard {
		t.Errorf("Rotation=%d, want Hadamard", info.Rotation)
	}
	if info.Seed != 99 {
		t.Errorf("Seed=%d, want 99", info.Seed)
	}
	if info.NumVecs != 50 {
		t.Errorf("NumVecs=%d, want 50", info.NumVecs)
	}
	if info.Compression < 1.0 {
		t.Errorf("Compression=%.2f, expected > 1", info.Compression)
	}
}

func TestStoreMatchesCollection(t *testing.T) {
	// Verify that Store search results match Collection search results.
	dir := t.TempDir()
	path := filepath.Join(dir, "match.tq")

	rng := rand.New(rand.NewPCG(111, 0))
	d := 64
	n := 200

	cfg := StoreConfig{Dim: d, Bits: 4, Rotation: RotationHadamard, Seed: 42}

	// Build collection.
	coll, _ := NewCollection(Config{Dim: d, Bits: 4, Rotation: RotationHadamard, Seed: 42})
	store, _ := CreateStore(path, cfg)

	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randomVector(d, rng)
		id := "v" + string(rune('A'+i%26))
		coll.Add(id, vecs[i], nil)
		_ = store.Add(id, vecs[i], nil)
	}
	_ = store.Close()

	store2, _ := OpenStore(path)
	defer store2.Close() //nolint:errcheck

	query := randomVector(d, rng)
	collResults := coll.Search(query, 10)
	storeResults := store2.Search(query, 10)

	if len(collResults) != len(storeResults) {
		t.Fatalf("result count: coll=%d store=%d", len(collResults), len(storeResults))
	}

	for i := range collResults {
		if collResults[i].ID != storeResults[i].ID {
			t.Errorf("rank %d: coll=%s store=%s", i, collResults[i].ID, storeResults[i].ID)
		}
		if math.Abs(collResults[i].Score-storeResults[i].Score) > 1e-6 {
			t.Errorf("rank %d: coll=%.6f store=%.6f", i, collResults[i].Score, storeResults[i].Score)
		}
	}
}

package store

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/scotteveritt/tqdb"
)

func TestHNSW_PersistRoundtrip(t *testing.T) {
	dim := 64
	n := 1000
	k := 10

	cfg := tqdb.Config{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	coll, _ := NewCollection(cfg)
	rng := rand.New(rand.NewPCG(42, 0))
	vecs := make([][]float64, n)
	for i := range n {
		vecs[i] = randomVector(dim, rng)
		coll.Add(fmt.Sprintf("%d", i), vecs[i], nil)
	}

	// Build HNSW index.
	coll.CreateIndex(tqdb.IndexConfig{Type: tqdb.IndexHNSW, M: 16, EfConstruction: 200})
	if coll.hnswIdx == nil {
		t.Fatal("HNSW index not built")
	}

	// Search before persistence.
	query := randomVector(dim, rng)
	resultsBefore := coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: k, Ef: 200})

	// Serialize graph.
	graphData := coll.hnswIdx.MarshalHNSW()
	t.Logf("graph size: %d bytes (%d nodes)", len(graphData), n)

	// Write .tq file with graph.
	dir := t.TempDir()
	tqPath := filepath.Join(dir, "hnsw.tq")
	storeCfg := tqdb.StoreConfig{Dim: dim, Bits: 8, Rotation: tqdb.RotationHadamard}
	s, err := Create(tqPath, storeCfg)
	if err != nil {
		t.Fatal(err)
	}
	coll.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
		if err := s.AddRaw(id, indices, norm, content, data); err != nil {
			t.Fatal(err)
		}
	})
	s.SetGraph(graphData)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify file size is reasonable.
	fi, _ := os.Stat(tqPath)
	t.Logf("file size: %d bytes (graph: %d bytes = %.0f%%)",
		fi.Size(), len(graphData), float64(len(graphData))/float64(fi.Size())*100)

	// Reopen and verify HNSW graph was loaded.
	s2, err := Open(tqPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s2.Close() }()

	if s2.hnswIdx == nil {
		t.Fatal("HNSW index not loaded from file")
	}
	if s2.hnswIdx.entryNode < 0 {
		t.Fatal("HNSW entry node not set")
	}
	if s2.hnswIdx.count != n {
		t.Errorf("HNSW node count = %d, want %d", s2.hnswIdx.count, n)
	}
	if s2.hnswVecs == nil {
		t.Fatal("decoded vectors not loaded")
	}

	// Search on the reloaded store using HNSW.
	// The Store doesn't have SearchWithOptions with HNSW yet,
	// but we can test the graph directly.
	qr32 := make([]float32, s2.workDim)
	queryRotated := s2.quantizer.GetBuf()
	qNorm := vecNorm(query)
	unitQ := s2.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v / qNorm
	}
	s2.quantizer.Rotation().Rotate(queryRotated, unitQ[:dim])
	for i := range s2.workDim {
		qr32[i] = float32(queryRotated[i])
	}
	s2.quantizer.PutBuf(unitQ)
	s2.quantizer.PutBuf(queryRotated)

	d := s2.workDim
	distToQuery := func(nodeID int) float32 {
		vec := s2.hnswVecs[nodeID*d : nodeID*d+d]
		var dot float32
		for j := range d {
			dot += qr32[j] * vec[j]
		}
		return -dot
	}

	hnswResults := s2.hnswIdx.Search(distToQuery, k, 200)
	if len(hnswResults) != k {
		t.Fatalf("HNSW search returned %d results, want %d", len(hnswResults), k)
	}

	// Compare with the pre-persistence results.
	if len(resultsBefore) > 0 && len(hnswResults) > 0 {
		t.Logf("before persistence: top-1 = %s (score %.4f)", resultsBefore[0].ID, resultsBefore[0].Score)
		t.Logf("after persistence:  top-1 = node %d (dist %.4f)", hnswResults[0].id, hnswResults[0].dist)
	}
}

func TestHNSW_MarshalUnmarshal(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	dim := 32
	n := 500

	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
	}

	distFunc := func(a, b int) float32 { return negDotF32(vecs[a], vecs[b]) }

	h := newHNSW(hnswConfig{M: 16, EfConstruction: 100})
	for i := range n {
		h.Insert(i, distFunc)
	}

	// Marshal.
	data := h.MarshalHNSW()
	t.Logf("marshaled %d nodes to %d bytes (%.1f bytes/node)", n, len(data), float64(len(data))/float64(n))

	// Unmarshal.
	h2, err := UnmarshalHNSW(data)
	if err != nil {
		t.Fatal(err)
	}

	// Verify structure.
	if h2.entryNode != h.entryNode {
		t.Errorf("entryNode: %d != %d", h2.entryNode, h.entryNode)
	}
	if h2.maxLevel != h.maxLevel {
		t.Errorf("maxLevel: %d != %d", h2.maxLevel, h.maxLevel)
	}
	if h2.M != h.M {
		t.Errorf("M: %d != %d", h2.M, h.M)
	}
	if len(h2.edges) != len(h.edges) {
		t.Fatalf("edges length: %d != %d", len(h2.edges), len(h.edges))
	}

	// Verify search produces same results.
	query := vecs[0]
	distToQuery := func(nodeID int) float32 { return negDotF32(query, vecs[nodeID]) }

	r1 := h.Search(distToQuery, 10, 100)
	r2 := h2.Search(distToQuery, 10, 100)

	if len(r1) != len(r2) {
		t.Fatalf("result count: %d != %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].id != r2[i].id {
			t.Errorf("rank %d: %d != %d", i, r1[i].id, r2[i].id)
		}
	}
}

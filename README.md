# tqdb

**The quantization-native vector database.** Pure Go. Single file. Embeddable.

Store vectors compressed. Search without decompressing. Open in milliseconds.

Built on Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with Hadamard rotation and IVF partitioning.

## Install

```bash
# Library
go get github.com/scotteveritt/tqdb

# CLI
go install github.com/scotteveritt/tqdb/cmd/tqdb@latest
```

## CLI Quick Start

```bash
# Initialize a workspace with an embedding provider
tqdb init --provider ollama

# Import data (JSONL with vectors, or --embed to auto-embed text)
tqdb import --from embeddings.jsonl

# Search by text (embeds via configured provider)
tqdb search "how does authentication work"

# Search with filters
tqdb search "error handling" --top 5 --filter repo=myrepo --filter language=go

# Inspect
tqdb info
tqdb count
tqdb bench --queries 100
tqdb export | head -5
```

The CLI supports three embedding providers (Vertex AI, OpenAI, Ollama) as
lightweight HTTP clients with no SDK dependencies. Configure once via
`tqdb init` or `~/.config/tqdb/config.yaml`.

## Go Library

```go
import (
    "github.com/scotteveritt/tqdb"
    "github.com/scotteveritt/tqdb/store"
)

// Create and populate a .tq file
s, _ := store.Create("index.tq", tqdb.StoreConfig{
    Dim: 3072, Bits: 8, Rotation: tqdb.RotationHadamard,
})
s.Add(tqdb.Document{
    ID: "doc-1", Content: "hello world", Embedding: vec,
    Data: map[string]any{"repo": "myrepo", "language": "go"},
})
s.Close() // writes the .tq file atomically

// Open (mmap, instant) and search
s, _ = store.Open("index.tq")
defer s.Close()
results := s.SearchWithOptions(query, tqdb.SearchOptions{
    TopK:   10,
    Filter: tqdb.And(tqdb.Eq("repo", "myrepo"), tqdb.Gt("stars", 10.0)),
})
```

### In-Memory Collection

For applications that need Add/Delete/Upsert during a session:

```go
coll, _ := store.NewCollection(tqdb.Config{
    Dim: 3072, Bits: 8, Rotation: tqdb.RotationHadamard,
})
coll.Add("id", vec, data)
coll.CreateIndex(tqdb.IndexConfig{
    FilterFields: []string{"repo", "language"},
})
results := coll.Search(query, 10)
```

## How It Works

1. **Normalize** the vector to unit length, store the magnitude separately
2. **Rotate** via Randomized Walsh-Hadamard Transform (O(d log d), 65 KB memory)
3. **Quantize** each coordinate with a Lloyd-Max codebook precomputed from the known Gaussian distribution (no training data needed)
4. **Search** by rotating the query once, then computing inner products via centroid table lookup (no decompression)

The codebook depends only on (dimension, bits), not on your data. This makes quantization data-oblivious: you can add vectors one at a time without retraining.

## Benchmarks

All measurements on Apple M4 Pro, 25K Gemini embeddings, d=3072.

### Search Performance

| Mode | Recall@10 | p50 | QPS |
|------|-----------|-----|-----|
| Brute-force (8-bit) | ~99% | 2.9ms | 343 |
| Brute-force (4-bit) | ~89% | 2.9ms | 343 |
| IVF + rescore | ~92% | 9.4ms | 106 |

### vs chromem-go

| Metric | chromem-go | tqdb | Improvement |
|--------|-----------|------|-------------|
| Startup | 6.2s | **10ms** | **620x** |
| Search | 72ms | **2.9ms** | **25x** |
| Disk | 397 MB (25K files) | **140 MB** (1 file) | **2.8x** |
| Recall@10 | 100% (exact) | **~99%** (8-bit) | -1% |

### Recall by Bit-Width

| Bits | Recall@10 (d=3072) | Recall@10 (d=128) | Bytes/Vector | Compression |
|------|-------------------|-------------------|-------------|-------------|
| 3 | 78% | 76% | d * 3/8 | 21x |
| 4 | 89% | 86% | d * 4/8 | 16x |
| 5 | 93% | 93% | d * 5/8 | 13x |
| 6 | 96% | 96% | d * 6/8 | 11x |
| **8** | **~99%** | **99%** | **d** | **8x** |

Default is 8-bit. With the current uint8 storage format, all bit-widths use the
same bytes on disk (one byte per coordinate). The compression ratios above apply
when bit-packing is enabled in the .tq format (4-bit packs 2 indices per byte).

### Standard ANN Benchmarks

Tested on canonical datasets from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks):

| Dataset | Type | d | N | 4-bit Recall@10 | 8-bit Recall@10 |
|---------|------|---|---|-----------------|-----------------|
| Gemini embeddings | Learned | 3072 | 25K | 91.9% | ~99% |
| GloVe-100 | Learned | 100 | 1.18M | 80.8% | 96.6% |
| SIFT-128 | SIFT descriptors | 128 | 1M | 50.9% | 89.3% |

4-bit TurboQuant works best on modern learned embeddings (Gemini, GloVe, OpenAI).
SIFT descriptors have distributional properties that don't match the Gaussian
codebook assumption, resulting in lower recall at low bit-widths.

## Comparison with TurboQuant Paper

We changed two things relative to the paper's approach, and measured each independently:

| Config | Rotation | Bit Allocation | Recall@10 (d=3072) |
|--------|----------|---------------|-------------------|
| Paper | QR | Prod (3+1) | ~85% |
| tqdb (rotation only) | **Hadamard** | Prod (3+1) | 89.2% |
| tqdb (both) | **Hadamard** | **MSE-only (4+0)** | **91.9%** |

Hadamard rotation contributes ~4.3% and MSE-only bit allocation ~2.6%.
Memory: 65 KB (Hadamard) vs 75 MB (QR).

## Features

### Filters

Composable filters matching [Google Vector Search 2.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/search) syntax:

```go
tqdb.Eq("repo", "tqdb")                    // exact match
tqdb.In("lang", "go", "rust", "python")    // set membership
tqdb.Gt("stars", 100.0)                    // numeric comparison
tqdb.And(filter1, filter2)                  // intersection
tqdb.Or(filter1, filter2)                   // union
tqdb.Contains("content", "vector")          // substring
```

Fields listed in `IndexConfig.FilterFields` get inverted indexes for O(1) lookup.

### IVF Partitioning

```go
coll.CreateIndex(tqdb.IndexConfig{
    FilterFields: []string{"repo", "language"},
    // IVF auto-tuned: sqrt(N) partitions, sqrt(P) probes
    // Set SkipIVF: true to build only filter indexes (faster startup)
})
```

### File Format

The `.tq` format is a single columnar file, memory-mapped for instant startup:

```
[Header 64B] [Indices N*packedRow] [Norms N*4B] [IDs] [Data JSON] [Contents]
```

Indices are bit-packed (4-bit: 2 per byte, 8-bit: 1 per byte). The header
stores dimension, bits, rotation type, and section offsets. IDs, metadata, and
content are lazily loaded on first access.

### CRUD Operations

```go
coll.Add("id", vec, data)                   // skip if duplicate
coll.AddDocument(ctx, doc)                  // auto-embed if EmbedFunc set
coll.AddDocuments(ctx, docs, concurrency)   // batch with concurrent embedding
coll.Upsert("id", vec, data)               // replace if exists
coll.Delete("id-1", "id-2")
doc, ok := coll.GetByID("id")
ids := coll.ListIDs()
```

### Persistence Roundtrip

```go
// Save Collection to .tq file
s, _ := store.Create("index.tq", cfg)
coll.ForEach(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
    s.AddRaw(id, indices, norm, content, data)
})
s.Close()

// Load .tq file back into Collection
s, _ = store.Open("index.tq")
s.ForEachCompressed(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
    coll.AddRawDocument(id, indices, norm, content, data)
})
s.Close()
```

## License

MIT

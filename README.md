# tqdb

**The quantization-native vector database.** Pure Go. Single file. Embeddable.

Store vectors compressed. Search without decompressing. Open in milliseconds.

Built on Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with Hadamard rotation, HNSW graph indexing, and NEON-accelerated distance kernels.

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

### In-Memory Collection with HNSW

```go
coll, _ := store.NewCollection(tqdb.Config{
    Dim: 3072, Bits: 8, Rotation: tqdb.RotationHadamard,
})
coll.Add("id", vec, data)

// Build HNSW graph index (sub-linear search, ~97% recall)
coll.CreateIndex(tqdb.IndexConfig{
    Type:           tqdb.IndexHNSW,
    M:              16,
    EfConstruction: 200,
    FilterFields:   []string{"repo", "language"},
})

results := coll.SearchWithOptions(query, tqdb.SearchOptions{
    TopK: 10,
    Ef:   100, // higher = better recall, slower
})
```

## How It Works

1. **Normalize** the vector to unit length, store the magnitude separately
2. **Rotate** via Randomized Walsh-Hadamard Transform (O(d log d), 65 KB memory)
3. **Quantize** each coordinate with a Lloyd-Max codebook precomputed from the known Gaussian distribution (no training data needed)
4. **Index** via HNSW graph for sub-linear search, or brute-force for small collections
5. **Search** by rotating the query once, then traversing the graph with NEON-accelerated distance (no decompression)

The codebook depends only on (dimension, bits), not on your data. This makes quantization data-oblivious: you can add vectors one at a time without retraining.

## Benchmarks

All measurements on Apple M4 Pro with NEON acceleration.

### Search Performance

**25K vectors, d=3072 (Gemini embeddings):**

| Mode | p50 | QPS |
|------|-----|-----|
| Brute-force | **700 us** | **1,467** |

**10K vectors, d=128:**

| Mode | Recall@10 | p50 | QPS |
|------|-----------|-----|-----|
| HNSW | ~97% | 81 us | **12,346** |
| Brute-force | ~99% | 399 us | 2,505 |

HNSW shines at lower dimensions where per-distance cost is cheap. At d=3072,
brute-force is fast enough that graph traversal overhead doesn't help until 100K+ vectors.

### vs chromem-go (25K vectors, d=3072)

| Metric | chromem-go | tqdb | Improvement |
|--------|-----------|------|-------------|
| Startup | 6.2s | **10ms** | **620x** |
| Search | 72ms | **700 us** | **103x** |
| Disk | 397 MB (25K files) | **140 MB** (1 file) | **2.8x** |
| Recall@10 | 100% (exact) | **~99%** (8-bit) | -1% |

### NEON Assembly Acceleration

GoAT-generated ARM64 NEON kernels for distance computation:

| Operation | Pure Go | NEON | Speedup |
|-----------|---------|------|---------|
| Dot product (f32, d=128) | 33.8 ns | 8.6 ns | **3.9x** |
| Dot product (f32, d=3072) | 701 ns | 142 ns | **5.0x** |
| L2 distance (f32, d=128) | 33.6 ns | 8.5 ns | **4.0x** |

On x86, pure Go fallbacks are used automatically.

### Recall by Bit-Width

| Bits | Recall@10 (d=3072) | Recall@10 (d=128) | Compression |
|------|-------------------|-------------------|-------------|
| 4 | 89% | 86% | 16x |
| 5 | 93% | 93% | 13x |
| **8** | **~99%** | **99%** | **8x** |

Default is 8-bit. Indices are bit-packed in the .tq format (4-bit stores
2 indices per byte).

### Standard ANN Benchmarks

| Dataset | Type | d | N | 4-bit Recall@10 | 8-bit Recall@10 |
|---------|------|---|---|-----------------|-----------------|
| Gemini embeddings | Learned | 3072 | 25K | 91.9% | ~99% |
| GloVe-100 | Learned | 100 | 1.18M | 80.8% | 96.6% |
| SIFT-128 | SIFT descriptors | 128 | 1M | 50.9% | 89.3% |

TurboQuant works best on modern learned embeddings (Gemini, GloVe, OpenAI).

## Features

### Filters

Composable filters matching [Google Vector Search 2.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/search) syntax:

```go
tqdb.Eq("repo", "tqdb")
tqdb.In("lang", "go", "rust", "python")
tqdb.Gt("stars", 100.0)
tqdb.And(filter1, filter2)
tqdb.Or(filter1, filter2)
tqdb.Contains("content", "vector")
```

### CRUD

```go
// Collection (in-memory, supports all operations)
coll.Add("id", vec, data)                   // skip if duplicate
coll.Upsert("id", vec, data)               // replace if exists
coll.Delete("id-1", "id-2")
coll.AddDocument(ctx, doc)                  // auto-embed via EmbedFunc
doc, ok := coll.GetByID("id")

// Store (file-backed, write-once)
s.Add(tqdb.Document{ID: "id", Embedding: vec, Content: "...", Data: data})
```

### File Format

The `.tq` format is a single columnar file, memory-mapped for instant startup:

```
[Header 64B] [Indices] [Norms] [IDs] [Data] [Contents] [HNSW Graph]
```

Indices are bit-packed. The HNSW graph section is optional (~134 bytes/node).
IDs, metadata, and content are lazily loaded on first access.

## License

MIT

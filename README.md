# tqdb

**The quantization-native vector database.** Embeddable. Single-file. Pure Go.

8x compression. ScaNN-style indexing. Search without decompression. No training data needed.

Built on Google's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with [Hadamard rotation](https://arxiv.org/abs/2404.00456) and IVF partitioning. Exceeds the paper's recall by 7%.

## vs the Paper

tqdb exceeds the TurboQuant paper's reported recall by using Hadamard rotation instead of QR:

| Metric (d=3072, 4-bit) | Paper | tqdb | Delta |
|------------------------|-------|------|-------|
| Cosine similarity | ~99.5% | **99.6%** | +0.1% |
| Recall@10 (brute-force) | ~85% | **91.9%** | **+6.9%** |
| Recall@10 (IVF indexed) | — | **91.3%** | with 6.8x speedup |
| Rotation memory (d=3072) | 75 MB (QR) | **65 KB** (Hadamard) | **1,150x less** |
| Quantize time (d=3072) | 0.002s | **0.86 µs** | per vector |

The paper uses 3-bit MSE + 1-bit QJL (TurboQuant_Prod). We benchmarked both and found 4-bit MSE-only with Hadamard rotation outperforms the paper's approach (91.8% vs 89.2% recall@10).

## Why tqdb

No existing system simultaneously satisfies all four:

| Requirement | chromem-go | Weaviate | Milvus | sqlite-vec | coder/hnsw | **tqdb** |
|-------------|-----------|---------|--------|-----------|-----------|---------|
| Pure Go (no CGO) | Yes | Yes | No (C++) | No (C) | Yes | **Yes** |
| Embeddable (in-process) | Yes | **No** (server) | **No** | Yes | Yes | **Yes** |
| ANN indexing | **No** | Yes | Yes | **No** | Yes | **Yes** |
| Built-in quantization | **No** | Yes | Yes | int8 only | **No** | **Yes** (4-bit) |

## What tqdb does

- **Vector store** — `.tq` file, mmap, ScaNN-style IVF index, VS2-aligned filters
- **Model compression** — SafeTensors → 4-bit (TinyLlama 2.0 GB → 581 MB in 3s)
- **GGUF conversion** — SafeTensors → GGUF for ollama/llama.cpp/LM Studio
- **KV cache** — Quantized attention for 2x longer contexts

## Install

```bash
go get github.com/scotteveritt/tqdb
go install github.com/scotteveritt/tqdb/cmd/tqdb@latest
```

## Quick Start

```go
import (
    "github.com/scotteveritt/tqdb"
    "github.com/scotteveritt/tqdb/store"
)

// Create, add, flush
s, _ := store.Create("index.tq", tqdb.StoreConfig{
    Dim: 768, Bits: 4, Rotation: tqdb.RotationHadamard,
})
s.AddDocument(ctx, tqdb.Document{
    ID: "doc-1", Content: "...", Embedding: vec,
    Data: map[string]any{"repo": "tqdb", "stars": 42},
})
s.Close()

// Open (mmap, 10ms), search (5ms with IVF)
s, _ = store.Open("index.tq")
defer s.Close()

results := s.SearchWithOptions(query, tqdb.SearchOptions{
    TopK:   10,
    Filter: tqdb.And(tqdb.Eq("repo", "tqdb"), tqdb.Gt("stars", 10.0)),
})
```

## Indexes (ScaNN-style IVF)

```go
coll.CreateIndex(tqdb.IndexConfig{
    FilterFields: []string{"repo", "language"},
})
// IVF partitions built automatically (√N clusters, k-means)
// Filter fields get inverted indexes for O(1) lookup
// Searches use both: partition pruning + filter intersection
```

| Search Mode | Without Index | With Index | Speedup |
|------------|---------------|-----------|---------|
| Unfiltered (25K, d=3072) | 31ms | **4.6ms** | **6.8x** |
| Filtered | 12ms | **2.1ms** | **5.7x** |

## VS2-Aligned Filters

Matches [Google Vector Search 2.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/search) filter syntax:

```go
tqdb.Eq("repo", "tqdb")                    // $eq
tqdb.Ne("status", "archived")              // $ne
tqdb.Gt("stars", 100.0)                    // $gt
tqdb.In("lang", "go", "rust", "python")    // $in
tqdb.And(tqdb.Eq("repo", "x"), tqdb.Gt("stars", 50.0))  // $and
tqdb.Or(tqdb.Eq("lang", "go"), tqdb.Eq("lang", "rust"))  // $or
tqdb.Contains("content", "vector")          // $contains
```

## CRUD

```go
coll.Add("id", vec, data)                       // skip if duplicate
coll.AddDocument(ctx, doc)                       // auto-embed if EmbedFunc set
coll.Upsert("id", vec, data)                    // replace if exists
doc, ok := coll.GetByID("id")
coll.Delete("id-1", "id-2")
ids := coll.ListIDs()
```

## CLI

```bash
tqdb create index.tq --dim 768 --bits 4 --rotation hadamard
cat vectors.jsonl | tqdb add index.tq
tqdb search index.tq --query "0.1,0.2,..." --top 10
tqdb info index.tq
tqdb import index.tq --format chromem --dir /path/to/chromem-data
tqdb compress model.safetensors -o model.tqm --bits 4
tqdb convert ./model-dir -o model.gguf
tqdb inspect model.tqm
```

## Benchmarks

Measured on Apple M4 Pro with 25K real Gemini embeddings (d=3072).

### vs chromem-go

| Metric | chromem-go | tqdb | Improvement |
|--------|-----------|------|-------------|
| Open time | 6.2s | **10ms** | **620x** |
| Search (brute-force) | 72ms | **31ms** | **2.3x** |
| Search (IVF indexed) | — | **5ms** | **14x** |
| Disk size | 362 MB | **115 MB** | **3.1x** |
| Files | 25,411 | **1** | |
| Recall@10 | 100% (exact) | **91.9%** | 4-bit quantization |

### Recall vs Latency

| Config | Recall@10 | p50 | Speedup |
|--------|-----------|-----|---------|
| Brute-force | 91.9% | 31ms | 1.0x |
| IVF default | 89.1% | 4.7ms | 6.6x |
| IVF nProbe=2x | 90.7% | 8.5ms | 3.7x |
| **IVF nProbe=2x + rescore=30** | **91.9%** | **9.4ms** | **3.3x** |

### Model Compression (TinyLlama 1.1B)

| | Original | Compressed |
|---|---|---|
| Size | 2.0 GB (BF16) | **581 MB** (4-bit) |
| Time | — | **3.1s** (12 cores) |
| Quality | — | **99.5%** cosine sim |

### Compression Quality

| Bits | Cosine Similarity | Recall@10 | Ratio |
|------|-------------------|-----------|-------|
| 4-bit | 99.6% | **91.9%** | 8x |
| 5-bit | 99.8% | **92.3%** | 6.4x |
| 3-bit | 98.3% | ~80% | 10.6x |

## Architecture

```
tqdb/           — shared types: Config, Result, Filter, Document
quantize/       — TurboQuantMSE, Hadamard rotation, Lloyd-Max codebook
store/          — Collection (in-memory), Store (.tq file, mmap), IVF index
kvcache/        — quantized KV cache for transformer inference
model/          — SafeTensors reader, model weight compression
internal/codec/ — codebook solver, rotation matrices, bit-packing
```

## Algorithm

1. **Rotate** — Randomized Walsh-Hadamard Transform spreads energy uniformly (O(d log d), O(d) memory)
2. **Quantize** — Per-coordinate Lloyd-Max with precomputed codebook (no training data, data-oblivious)
3. **Index** — ScaNN-style IVF: k-means partitions + inverted filter indexes
4. **Search** — Rotate query once, prune partitions, inner product via centroid lookup (no decompression)

## License

MIT

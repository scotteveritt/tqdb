# tqdb

**SQLite for quantized vectors.** Single-file, embedded, zero-config.

Compress embeddings to 4-bit. Search without decompression. Compress model weights. Convert to GGUF. One Go binary.

Based on Google's [TurboQuant](https://arxiv.org/abs/2504.19874) algorithm (ICLR 2026).

## What tqdb does

- **Vector store** — Compress embeddings 8x, search via mmap, one `.tq` file
- **Model compression** — Quantize SafeTensors weights to 4-bit (TinyLlama 2.0 GB → 581 MB in 3s)
- **GGUF conversion** — SafeTensors → GGUF for ollama/llama.cpp/LM Studio
- **KV cache** — Quantized attention for 2-4x longer contexts

## Install

```bash
# Library
go get github.com/scotteveritt/tqdb

# CLI
go install github.com/scotteveritt/tqdb/cmd/tqdb@latest
```

## CLI

```bash
# Vector store
tqdb create index.tq --dim 768 --bits 4 --rotation hadamard
cat vectors.jsonl | tqdb add index.tq
tqdb search index.tq --query "0.1,0.2,..." --top 10
tqdb info index.tq

# Import from chromem-go
tqdb import index.tq --format chromem --dir ~/.local/share/csgdaa-code/vectorize

# Model compression
tqdb compress model.safetensors -o model.tqm --bits 4
tqdb inspect model.tqm

# GGUF conversion (via ollama)
tqdb convert ./model-dir -o model.gguf
```

## Library — Vector Store

```go
import (
    "github.com/scotteveritt/tqdb"
    "github.com/scotteveritt/tqdb/store"
)

// Write
s, _ := store.Create("index.tq", tqdb.StoreConfig{
    Dim:      768,
    Bits:     4,
    Rotation: tqdb.RotationHadamard,
})
s.Add("doc-1", embedding, map[string]any{"repo": "myrepo"})
s.Close() // atomic flush to disk

// Read (mmap, instant open)
s, _ = store.Open("index.tq")
defer s.Close()

results := s.Search(query, 10)
for _, r := range results {
    fmt.Printf("%s: %.4f\n", r.ID, r.Score)
}

// Filtered search (VS2-aligned)
results = s.SearchWithOptions(query, tqdb.SearchOptions{
    TopK:   10,
    Filter: tqdb.And(tqdb.Eq("repo", "tqdb"), tqdb.Gt("stars", 50.0)),
})
```

## Library — KV Cache

```go
import (
    "github.com/scotteveritt/tqdb"
    "github.com/scotteveritt/tqdb/kvcache"
)

kv, _ := kvcache.New(tqdb.KVCacheConfig{
    Layers:      32,
    Heads:       32,
    HeadDim:     128,
    Bits:        4,
    PackIndices: true,      // 2 indices per byte
    Rotation:    tqdb.RotationHadamard,
})

// Quantized attention — keys never decompressed
kv.AppendKey(layer, head, keyVec)
kv.AppendValue(layer, head, valueVec)
scores := kv.AttentionScores(layer, head, queryVec) // Q_rot @ centroids[idx]^T
value := kv.GetValue(layer, head, pos)              // decompress on demand
```

## Library — Standalone Quantizer

```go
import (
    "github.com/scotteveritt/tqdb"
    "github.com/scotteveritt/tqdb/quantize"
)

q, _ := quantize.NewMSE(tqdb.Config{Dim: 768, Bits: 4, Rotation: tqdb.RotationHadamard})

cv := q.Quantize(embedding)                         // compress
recon := q.Dequantize(cv)                           // decompress
data, _ := cv.MarshalBinary()                       // serialize (~384 bytes)
sim := q.AsymmetricCosineSimilarity(query, cv)       // compare without decompressing
```

## Benchmarks

Measured on Apple M4 Pro with real data.

### vs chromem-go (25K vectors, d=3072)

| Metric | chromem-go | tqdb | Improvement |
|--------|-----------|------|-------------|
| Open time | 6.2 s | **9 ms** | **662x** |
| Search time | 72 ms | **31 ms** | **2.3x** |
| Disk size | 362 MB | **110 MB** | **3.3x smaller** |
| Files | 25,411 | **1** | Single file |
| Ranking overlap | — | **10/10 top-10** | Identical results |

### Model compression (TinyLlama 1.1B)

| | Original | Compressed |
|---|---|---|
| Size | 2.0 GB (BF16) | **581 MB** (4-bit packed) |
| Ratio | — | **3.6x** |
| Time | — | **3.1s** (concurrent, 12 cores) |
| Quality | — | **99.5%** cosine sim per tensor |

### Quantizer performance

| Operation | QR (d=128) | Hadamard (d=128) | Hadamard (d=768) |
|-----------|-----------|------------------|------------------|
| Quantize | 3.9 µs | **0.86 µs** | 7.3 µs |
| Search 10K | 389 µs | — | — |
| KV Append | — | 0.9 µs | — |
| KV Attention 1K | — | 81 µs | — |

### Compression quality

| Bits | Cosine Similarity | Ratio vs float32 |
|------|-------------------|-------------------|
| 4-bit | **99.5%** | **8x** |
| 3-bit | 98.3% | 10.6x |
| 2-bit | 94.0% | 15.9x |

## Rotation Modes

| Mode | Memory | Compute | Quality |
|------|--------|---------|---------|
| `RotationHadamard` | **O(d)** — 65 KB at d=3072 | **O(d log d)** | Better ([QuaRot](https://arxiv.org/abs/2404.00456)) |
| `RotationQR` | O(d²) — 75 MB at d=3072 | O(d²) | Paper original |

Hadamard is recommended for all use cases.

## File Format

The `.tq` file uses a columnar layout optimized for mmap-backed search:

```
[Header 64B] [Indices N×workDim] [Norms N×4B] [IDs] [Metadata]
```

- **Indices** — contiguous, read directly from mmap (zero deserialization)
- **Norms** — decoded to `[]float32` at open time (~800 KB for 100K vectors)
- **IDs/Metadata** — lazy-loaded only when search results are accessed

## Scaling

| Vectors | d=128 | d=768 (est.) |
|---------|-------|-------------|
| 1K | 43 µs | ~250 µs |
| 10K | 389 µs | ~2.5 ms |
| 100K | ~4 ms | ~25 ms |
| 1M | ~40 ms | ~250 ms |

For N > 500K, pair tqdb with an ANN index for candidate retrieval, then re-rank with `AsymmetricCosineSimilarityBatch`.

## Algorithm

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026):

1. **Rotate** — Randomized Walsh-Hadamard Transform spreads outlier energy uniformly
2. **Quantize** — Per-coordinate Lloyd-Max with a precomputed codebook (no training data)
3. **Search** — Pre-rotate query once, then inner product via centroid lookup (no decompression)

## License

MIT

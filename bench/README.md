# tqdb Benchmark Suite

Reproducible benchmarks comparing tqdb against other Go vector search systems
on standard ANN benchmark datasets.

## Datasets

| Dataset | Dims | Vectors | Queries | Metric | Source |
|---------|------|---------|---------|--------|--------|
| `glove-100-angular` | 100 | 1,183,514 | 10,000 | Angular | [ann-benchmarks](http://ann-benchmarks.com/) |
| `sift-128-euclidean` | 128 | 1,000,000 | 10,000 | Euclidean | [ann-benchmarks](http://ann-benchmarks.com/) |
| `codesearchnet-jina-768-cosine` | 768 | 1,374,067 | 10,000 | Cosine | [VIBE 2025](https://arxiv.org/abs/2505.17810) |
| `simplewiki-openai-3072-normalized` | 3072 | 260,372 | 10,000 | Cosine | [VIBE 2025](https://arxiv.org/abs/2505.17810) |

All datasets include pre-computed ground truth (top-100 exact nearest neighbors).

## Systems Under Test

| System | Version | Language | ANN | Quantization | Notes |
|--------|---------|----------|-----|-------------|-------|
| **tqdb** | current | Pure Go | IVF | TurboQuant 4-bit | Our system |
| **chromem-go** | v0.7.1 | Pure Go | None | None | Brute-force baseline |
| **coder/hnsw** | latest | Pure Go | HNSW | None | Graph-based baseline |
| **sqlite-vec** | v0.1.7 | C (CGO) | None | float32/int8 | SQL-based baseline |

## Reproducing

### Prerequisites

```bash
go 1.25+
make
~4 GB disk for datasets
~8 GB RAM for largest dataset (codesearchnet-768)
```

### Steps

```bash
# 1. Download datasets (~2 GB total)
make datasets

# 2. Run all benchmarks
make bench

# 3. View results
cat bench/results/summary.md
```

### Individual Commands

```bash
# Download a single dataset
make dataset-glove100
make dataset-sift128
make dataset-codesearchnet768
make dataset-simplewiki3072

# Run benchmark for a single system
make bench-tqdb
make bench-chromem
make bench-hnsw
make bench-sqlite-vec

# Run benchmark for a single dataset
make bench-glove100
make bench-sift128
```

## Metrics

For each (system, dataset) pair, we measure:

| Metric | Description |
|--------|-------------|
| **Recall@10** | Fraction of true top-10 neighbors found |
| **Recall@100** | Fraction of true top-100 neighbors found |
| **QPS** | Queries per second (1/mean_latency) |
| **p50 latency** | Median query latency |
| **p95 latency** | 95th percentile query latency |
| **Build time** | Time to index all vectors |
| **Memory** | Peak RSS during search |
| **Disk size** | On-disk index size |

Recall is measured against the dataset's pre-computed ground truth (exact nearest neighbors),
not against any system's own brute-force results.

## Methodology

1. Each system indexes the full `train` split
2. All 10,000 queries from the `test` split are executed sequentially
3. First 100 queries are warmup (excluded from timing)
4. Remaining 9,900 queries are timed
5. Results are compared against the ground truth `neighbors` array
6. All runs on the same machine, same conditions, no other workload

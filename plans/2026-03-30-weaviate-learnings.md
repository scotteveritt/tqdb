# Weaviate-Informed Improvement Plan

## Context

After analyzing Weaviate's codebase and comparing with tqdb's architecture,
we identified 10 improvements ranked by impact. This plan prioritizes them
into concrete phases with implementation details.

tqdb's current position: 2.7ms p50 brute-force search over 25K d=3072 vectors,
379 QPS, 140 MB single file, pure Go, no assembly. The goal is to close the
gap with Weaviate-class performance while maintaining embeddability and simplicity.

## Phase 1: HNSW Graph Index

**Priority: Highest. This is the single biggest improvement tqdb can make.**

### Why

IVF gives ~6x speedup but only works well at specific partition counts and has
an expensive build step (minutes of k-means at d=4096). HNSW gives 10-100x
speedup with near-perfect recall and incremental insertion (no rebuild).

Weaviate's numbers: ~5,000-8,000 QPS at 95%+ recall on SIFT-128 via HNSW.
tqdb's current: 379 QPS brute-force, ~200 QPS with IVF.

### Design

HNSW operates on the quantized representation, same as our current IVF.
The graph edges connect quantized vectors; scoring uses the centroid-lookup
inner product (no decompression). This is the same asymmetric scoring tqdb
already does.

Key parameters:
- `M` = 16 (edges per node, standard default)
- `efConstruction` = 200 (build-time beam width)
- `efSearch` = tunable at query time (recall/speed tradeoff)

### Implementation

```
store/hnsw.go          HNSW graph structure, insert, search
store/hnsw_test.go     correctness + benchmark tests
```

Core data structures:

```go
type hnswIndex struct {
    maxLevel  int
    entryNode uint32
    nodes     []hnswNode
    M         int           // max edges per layer
    efConst   int           // build-time beam width
    levelMul  float64       // 1 / ln(M)
}

type hnswNode struct {
    edges [][]uint32        // edges[level] = neighbor IDs at that level
}
```

Key algorithms from Weaviate:

**Level assignment**: `level = floor(-log(rand) * levelMul)` where
`levelMul = 1/ln(M)`. This gives exponentially fewer nodes at higher levels.

**Diversity heuristic** for neighbor selection: Accept a candidate only if
it's closer to the query than to ANY already-accepted neighbor. This prevents
clustering in dense regions and maintains recall in sparse ones.

**Search**: Greedy beam search from entry point, layer by layer top-down.
At each layer, expand the beam by checking neighbors of current best candidates.
ef parameter controls beam width (higher = better recall, slower).

**Tombstone deletion**: Mark nodes as deleted, skip during search. Background
goroutine repairs edges (reconnects around tombstoned nodes) lazily.

### Integration with tqdb

Add HNSW as an option alongside IVF and brute-force:

```go
coll.CreateIndex(tqdb.IndexConfig{
    Type: tqdb.IndexHNSW,       // new
    M: 16, EfConstruction: 200,
    FilterFields: []string{"repo", "language"},
})

// Search with HNSW
results := coll.SearchWithOptions(query, tqdb.SearchOptions{
    TopK: 10,
    Ef: 100,  // new: HNSW beam width
})
```

### .tq format changes

HNSW graph edges need to be persisted in the .tq file. Add an optional
graph section after the contents section:

```
[Header] [Indices] [Norms] [IDs] [Data] [Contents] [Graph edges]
```

Header gets a flag indicating whether HNSW edges are present.

### Estimated impact

| Metric | Current (brute-force) | With HNSW |
|--------|----------------------|-----------|
| QPS (d=128, 10K) | 2,647 | ~15,000-25,000 |
| QPS (d=3072, 25K) | 379 | ~2,000-5,000 |
| Recall@10 | ~99% (8-bit) | ~95-99% (tunable via ef) |
| Build time | instant | ~5-10s for 25K vectors |
| Memory overhead | 0 | ~M*2*4 bytes per vector |

### Effort: Large (2-3 weeks)

---

## Phase 2: Assembly Distance Kernels

**Priority: High. 3-5x speedup on the scoring inner loop.**

### Why

Go's compiler emits scalar FMADDD on ARM64, not NEON VFMLA. Weaviate
auto-generates NEON assembly via GoAT and gets 3-5x speedup on distance
computations.

Our gather+FMA pattern can't fully vectorize (random table lookups), but
the straight dot product paths (HNSW distance, cosine similarity for rescore,
Hadamard rotation) are contiguous and benefit from NEON.

### Design

Follow Weaviate's pattern:

```go
// internal/distancer/dot.go
var dotImpl func(a, b []float32) float32 = dotGo

func init() {
    if cpu.ARM64.HasASIMD {
        dotImpl = dotNEON
    }
    if cpu.X86.HasAVX2 {
        dotImpl = dotAVX2
    }
}

// internal/distancer/dot_neon_arm64.s
// Generated via GoAT from C with NEON intrinsics
```

### What to accelerate

| Function | Pattern | Expected speedup |
|----------|---------|-----------------|
| float32 dot product | contiguous FMA | 4x (NEON 4-wide) |
| L2 distance (HNSW) | contiguous sub+mul | 4x |
| Hadamard FWHT | contiguous add/sub | 2-3x |
| Cosine similarity (rescore) | contiguous 3-accumulator | 3x |

The gather+FMA scoring loop stays scalar (unavoidable without hardware gather).

### Tools

- **GoAT** (what Weaviate uses): C -> Go assembly transpiler
- Or **avo**: Go library for generating Go assembly programmatically
- Or hand-write the .s files (small enough, ~50 instructions each)

### Effort: Medium (1 week)

---

## Phase 3: Hardware Prefetch

**Priority: High. Single instruction, measurable improvement.**

### Why

Weaviate's search loop prefetches the next candidate's vector data while
processing the current one. This hides the L1 miss latency (3 cycles) when
scanning sequential vectors in brute-force mode.

### Design

In the brute-force scoring loop, add a prefetch hint for the next vector's
indices before scoring the current one:

```go
// In the scoring loop (conceptual - actual impl in assembly)
for i := range n {
    // Prefetch next vector's indices into L1
    if i+1 < n {
        prefetchL1(allIdx[(i+1)*d:])
    }
    // Score current vector
    score := gatherDot(qr32, allIdx[i*d:i*d+d], centroids32)
    insertTopK(i, score)
}
```

In Go assembly: `PRFM (R_NEXT), PLDL1KEEP`

This requires a tiny assembly function or the prefetch to be part of the
distance kernel from Phase 2.

### Effort: Small (1 day, pairs with Phase 2)

---

## Phase 4: Sharded Locks

**Priority: Medium. Enables concurrent writes.**

### Why

tqdb's Collection uses a single `sync.RWMutex`. Multiple goroutines calling
Add() or Upsert() serialize on this lock. Weaviate uses 512 striped locks.

### Design

```go
const numShards = 512

type Collection struct {
    shards [numShards]shard
}

type shard struct {
    mu sync.RWMutex
    // per-shard data
}

func (c *Collection) shardFor(id string) *shard {
    h := fnv1a(id)
    return &c.shards[h % numShards]
}
```

For search (read path), we still need to scan all shards, but multiple
concurrent searches don't block each other. For writes, only the target
shard is locked.

### Caveat

This is only valuable when tqdb is used as a long-running in-process DB
with concurrent writers (like csgdaa-vectorize's indexer). For the CLI
and Store (write-once) use cases, it doesn't matter.

### Effort: Medium (3-5 days)

---

## Phase 5: O(1)-Reset Visited Set (for HNSW)

**Priority: Medium. Required for efficient HNSW search.**

### Why

HNSW search needs a visited set to avoid revisiting nodes. Zeroing a
[]bool of size N on every query is O(N). Weaviate's version-byte trick
makes reset O(1):

```go
type visitedSet struct {
    visited []byte
    version byte
}

func (v *visitedSet) Reset() {
    v.version++
    if v.version == 0 {
        // Overflow every 255 resets: zero once
        clear(v.visited)
        v.version = 1
    }
}

func (v *visitedSet) Visit(id uint32) bool {
    if v.visited[id] == v.version {
        return false // already visited
    }
    v.visited[id] = v.version
    return true
}
```

### Effort: Small (part of HNSW implementation)

---

## Phase 6: Pool-of-Pools

**Priority: Medium. Reduces GC pressure.**

### Why

tqdb already pools float64 buffers via sync.Pool. Extend to:
- Result candidate heaps (topBuf slices)
- Filter candidate sets
- Temporary score buffers
- HNSW beam/candidate lists

Weaviate pools visited sets, heaps, and temp vectors aggressively.

### Design

```go
type searchPools struct {
    topBuf   sync.Pool // []scored
    visited  sync.Pool // visitedSet (for HNSW)
    qr32Buf  sync.Pool // []float32 (rotated query)
}
```

### Effort: Small (1-2 days)

---

## Phase 7: Blocked FWHT

**Priority: Low-Medium. Better cache behavior for rotation.**

### Why

Weaviate's RQ uses blocked FWHT with 64/256-element blocks and multiple
rounds, instead of a single full-dimension FWHT. This keeps the working
set in L1 cache during the butterfly passes.

For d=4096, our single FWHT works on 4096 float64s = 32 KB. The M4 Pro
has 128 KB L1d, so it fits. But for very large dimensions or when cache
is contested (concurrent queries), blocked FWHT could help.

### Design

Split the vector into blocks of 64 or 256 elements, FWHT each block
independently, then apply cross-block mixing (random permutation + sign flips).

This changes the mathematical properties of the rotation (it's no longer
a global Hadamard transform), but multiple rounds compensate. Weaviate
uses 3 rounds and notes "2 rounds leads to bias."

### Effort: Medium (1 week including recall validation)

---

## Phase 8: Dynamic Index Selection

**Priority: Low. Quality-of-life improvement.**

### Why

Weaviate auto-switches between flat (brute-force) and HNSW based on data
size. Below a threshold, brute-force is faster. Above it, HNSW wins.

### Design

```go
type dynamicIndex struct {
    flat *flatIndex
    hnsw *hnswIndex
    threshold int // switch point, e.g., 10,000
}

func (d *dynamicIndex) Search(query []float64, k int) []Result {
    if d.Count() < d.threshold {
        return d.flat.Search(query, k)
    }
    return d.hnsw.Search(query, k)
}
```

Also fixes the `Store.ensureIVF` bug where lazy IVF build takes minutes.
With dynamic indexing, small collections use brute-force and large collections
auto-build HNSW on first search.

### Effort: Small (2-3 days, after HNSW is implemented)

---

## Implementation Roadmap

### Now (next release)
- Phase 3: Hardware prefetch (pairs with current assembly investigation)
- Phase 6: Pool-of-pools for search buffers (low-hanging fruit)

### Next (1-2 weeks)
- Phase 1: HNSW graph index (the biggest win)
- Phase 5: O(1) visited set (required for HNSW)

### After HNSW (2-4 weeks)
- Phase 2: Assembly distance kernels (maximize HNSW throughput)
- Phase 8: Dynamic index selection (auto flat/HNSW)

### When needed
- Phase 4: Sharded locks (only if concurrent writes become a bottleneck)
- Phase 7: Blocked FWHT (only if rotation becomes a bottleneck at very high d)

## What We Don't Adopt from Weaviate

- **LSM persistence**: Too complex for an embeddable library. Our single-file
  .tq format is simpler and sufficient. Incremental updates can be handled
  via Collection + periodic flush.
- **Multi-tenancy / sharding**: Server-scale feature, not embeddable.
- **gRPC / GraphQL API**: Our REST API in `tqdb serve` is enough for now.
- **Module system**: Over-engineered for our use case. Built-in providers
  (Vertex, OpenAI, Ollama) are sufficient.
- **BM25 hybrid search**: Different use case. tqdb is vector-only.

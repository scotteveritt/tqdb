// tqdb-annbench: Standard ANN benchmark harness.
//
// Reads FVECS/IVECS dataset directories (converted from HDF5 via convert_hdf5.py).
// Benchmarks tqdb at multiple configurations and reports recall/latency.
//
// Usage:
//
//	go run ./bench/cmd/tqdb-annbench --dir bench/datasets/siftsmall-128-euclidean
package main

import (
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/store"
)

func main() {
	dir := flag.String("dir", "", "path to dataset directory with train.fvecs, test.fvecs, neighbors.ivecs")
	output := flag.String("output", "", "write markdown results to file (optional)")
	topK := flag.Int("k", 10, "recall@k")
	maxQueries := flag.Int("queries", 0, "max queries to run (0 = all)")
	maxPartitions := flag.Int("partitions", 0, "max IVF partitions (0 = auto √N)")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "error: --dir required")
		os.Exit(1)
	}

	name := filepath.Base(*dir)
	fmt.Printf("=== tqdb ANN Benchmark: %s ===\n\n", name)

	// Load dataset.
	fmt.Println("Loading dataset...")
	train := readFvecs(filepath.Join(*dir, "train.fvecs"))
	test := readFvecs(filepath.Join(*dir, "test.fvecs"))
	neighbors := readIvecs(filepath.Join(*dir, "neighbors.ivecs"))
	dim := len(train[0])
	fmt.Printf("  Train: %d vectors, d=%d\n", len(train), dim)
	fmt.Printf("  Test:  %d queries\n", len(test))
	fmt.Printf("  Truth: top-%d neighbors\n\n", len(neighbors[0]))

	// Limit queries if requested.
	if *maxQueries > 0 && *maxQueries < len(test) {
		test = test[:*maxQueries]
		neighbors = neighbors[:*maxQueries]
	}

	k := *topK
	warmup := 10
	if warmup > len(test) {
		warmup = 0
	}

	// Configs to benchmark. IVF configs with the same bits share a single collection build.
	type searchConfig struct {
		name    string
		rescore int
	}
	type benchGroup struct {
		bits    int
		ivf     bool
		configs []searchConfig
	}

	groups := []benchGroup{
		{bits: 0, ivf: false, configs: []searchConfig{{"float64 brute-force (baseline)", 0}}},
		{bits: 4, ivf: false, configs: []searchConfig{{"tqdb 4-bit brute-force", 0}}},
		{bits: 4, ivf: true, configs: []searchConfig{
			{"tqdb 4-bit IVF", 0},
			{"tqdb 4-bit IVF + rescore=30", 30},
		}},
		{bits: 5, ivf: false, configs: []searchConfig{{"tqdb 5-bit brute-force", 0}}},
		{bits: 8, ivf: false, configs: []searchConfig{{"tqdb 8-bit brute-force", 0}}},
	}

	type result struct {
		name      string
		recall    float64
		p50       time.Duration
		p95       time.Duration
		qps       float64
		buildTime time.Duration
	}
	var results []result

	for _, g := range groups {
		if g.bits == 0 {
			fmt.Printf("--- %s ---\n", g.configs[0].name)
			r := benchFloat64BruteForce(train, test, neighbors, k, warmup, dim)
			results = append(results, result{g.configs[0].name, r.recall, r.p50, r.p95, r.qps, r.buildTime})
			continue
		}

		// Build collection once for all configs in this group.
		fmt.Printf("--- Building %d-bit collection (IVF=%v) ---\n", g.bits, g.ivf)
		coll, buildTime := buildCollection(train, dim, g.bits, g.ivf, *maxPartitions)
		fmt.Printf("  Built in %s\n", buildTime.Round(time.Millisecond))

		// Run each search config on the shared collection.
		for _, cfg := range g.configs {
			fmt.Printf("--- %s ---\n", cfg.name)
			r := benchSearch(coll, test, neighbors, k, warmup, cfg.rescore)
			results = append(results, result{cfg.name, r.recall, r.p50, r.p95, r.qps, buildTime})
		}
	}

	// Print table.
	fmt.Printf("\n=== Results: %s (k=%d) ===\n\n", name, k)
	header := fmt.Sprintf("| %-35s | %10s | %10s | %10s | %8s | %8s |",
		"Config", "Recall@"+fmt.Sprint(k), "p50", "p95", "QPS", "Build")
	fmt.Println(header)
	fmt.Println("|", strings.Repeat("-", 35), "|", "---:", "|", "---:", "|", "---:", "|", "---:", "|", "---:", "|")
	for _, r := range results {
		fmt.Printf("| %-35s | %9.1f%% | %8.2fms | %8.2fms | %8.0f | %7.2fs |\n",
			r.name, r.recall*100,
			float64(r.p50.Microseconds())/1000,
			float64(r.p95.Microseconds())/1000,
			r.qps, r.buildTime.Seconds())
	}

	// Write markdown if requested.
	if *output != "" {
		f, _ := os.Create(*output)
		_, _ = fmt.Fprintf(f, "# Benchmark: %s\n\n", name)
		_, _ = fmt.Fprintf(f, "- Vectors: %d train, %d test, d=%d\n", len(train), len(test), dim)
		_, _ = fmt.Fprintf(f, "- Machine: %s (%d cores)\n", runtime.GOARCH, runtime.NumCPU())
		_, _ = fmt.Fprintf(f, "- Date: %s\n\n", time.Now().Format("2006-01-02"))
		_, _ = fmt.Fprintf(f, "| Config | Recall@%d | p50 | p95 | QPS | Build |\n", k)
		_, _ = fmt.Fprintf(f, "| --- | ---: | ---: | ---: | ---: | ---: |\n")
		for _, r := range results {
			_, _ = fmt.Fprintf(f, "| %s | %.1f%% | %.2fms | %.2fms | %.0f | %.2fs |\n",
				r.name, r.recall*100,
				float64(r.p50.Microseconds())/1000,
				float64(r.p95.Microseconds())/1000,
				r.qps, r.buildTime.Seconds())
		}
		_ = f.Close()
		fmt.Printf("\nResults written to %s\n", *output)
	}
}

type benchResult struct {
	recall    float64
	p50       time.Duration
	p95       time.Duration
	qps       float64
	buildTime time.Duration
}

// buildCollection creates a tqdb collection, adds all training vectors,
// and optionally builds an IVF index. Returns the collection and build time.
func buildCollection(train [][]float64, dim, bits int, useIVF bool, maxPartitions int) (*store.Collection, time.Duration) {
	buildStart := time.Now()
	coll, _ := store.NewCollection(tqdb.Config{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard,
	})
	for i, vec := range train {
		coll.Add(fmt.Sprintf("%d", i), vec, nil)
	}
	if useIVF {
		cfg := tqdb.IndexConfig{}
		if maxPartitions > 0 {
			cfg.NumPartitions = maxPartitions
		}
		coll.CreateIndex(cfg)
	}
	return coll, time.Since(buildStart)
}

// benchSearch runs search queries against an existing collection and returns recall/latency stats.
func benchSearch(coll *store.Collection, test [][]float64, neighbors [][]int32, k, warmup, rescore int) benchResult {
	var latencies []time.Duration
	var totalRecall float64

	for qi, query := range test {
		start := time.Now()
		var results []tqdb.Result
		if rescore > 0 {
			results = coll.SearchWithOptions(query, tqdb.SearchOptions{TopK: k, Rescore: rescore})
		} else {
			results = coll.Search(query, k)
		}
		elapsed := time.Since(start)

		if qi >= warmup {
			latencies = append(latencies, elapsed)
			truthSet := make(map[int32]bool, k)
			for _, idx := range neighbors[qi][:k] {
				truthSet[idx] = true
			}
			hits := 0
			for _, r := range results {
				var idx int
				fmt.Sscanf(r.ID, "%d", &idx) //nolint:errcheck
				if truthSet[int32(idx)] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
	}

	return computeStats(latencies, totalRecall, 0)
}

func benchFloat64BruteForce(train, test [][]float64, neighbors [][]int32, k, warmup, _ int) benchResult {
	n := len(train)
	buildStart := time.Now()
	// Precompute norms.
	norms := make([]float64, n)
	for i, v := range train {
		norms[i] = vecNorm(v)
	}
	buildTime := time.Since(buildStart)

	var latencies []time.Duration
	var totalRecall float64

	type scored struct {
		idx   int
		score float64
	}

	for qi, query := range test {
		qNorm := vecNorm(query)
		start := time.Now()

		// Top-k via sorted-insert (O(N×k) instead of O(N log N) full sort).
		topBuf := make([]scored, 0, k+1)
		minScore := -math.MaxFloat64

		for i, vec := range train {
			s := vecDot(query, vec) / (qNorm * norms[i])
			if len(topBuf) >= k && s <= minScore {
				continue
			}
			pos := sort.Search(len(topBuf), func(p int) bool { return topBuf[p].score < s })
			topBuf = append(topBuf, scored{})
			copy(topBuf[pos+1:], topBuf[pos:])
			topBuf[pos] = scored{i, s}
			if len(topBuf) > k {
				topBuf = topBuf[:k]
			}
			if len(topBuf) == k {
				minScore = topBuf[k-1].score
			}
		}
		elapsed := time.Since(start)

		if qi >= warmup {
			latencies = append(latencies, elapsed)
			truthSet := make(map[int32]bool, k)
			for _, idx := range neighbors[qi][:k] {
				truthSet[idx] = true
			}
			hits := 0
			for _, s := range topBuf[:k] {
				if truthSet[int32(s.idx)] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
	}

	return computeStats(latencies, totalRecall, buildTime)
}

func computeStats(latencies []time.Duration, totalRecall float64, buildTime time.Duration) benchResult {
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	n := len(latencies)
	var total time.Duration
	for _, d := range latencies {
		total += d
	}
	return benchResult{
		recall:    totalRecall / float64(n),
		p50:       latencies[n/2],
		p95:       latencies[n*95/100],
		qps:       float64(n) / total.Seconds(),
		buildTime: buildTime,
	}
}

// --- FVECS/IVECS readers ---

func readFvecs(path string) [][]float64 {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var vecs [][]float64
	for {
		var dim uint32
		if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			_ = f.Close()
			os.Exit(1)
		}
		raw := make([]float32, dim)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			_ = f.Close()
			fmt.Fprintf(os.Stderr, "error reading fvecs data: %v\n", err)
			os.Exit(1)
		}
		vec := make([]float64, dim)
		for i, v := range raw {
			vec[i] = float64(v)
		}
		vecs = append(vecs, vec)
	}
	_ = f.Close()
	return vecs
}

func readIvecs(path string) [][]int32 {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	var vecs [][]int32
	for {
		var dim uint32
		if err := binary.Read(f, binary.LittleEndian, &dim); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			_ = f.Close()
			fmt.Fprintf(os.Stderr, "error reading ivecs dim: %v\n", err)
			os.Exit(1)
		}
		vec := make([]int32, dim)
		if err := binary.Read(f, binary.LittleEndian, vec); err != nil {
			_ = f.Close()
			fmt.Fprintf(os.Stderr, "error reading ivecs data: %v\n", err)
			os.Exit(1)
		}
		vecs = append(vecs, vec)
	}
	_ = f.Close()
	return vecs
}

func vecDot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func vecNorm(v []float64) float64 {
	return math.Sqrt(vecDot(v, v))
}

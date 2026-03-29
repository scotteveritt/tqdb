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

	k := *topK
	warmup := 10
	if warmup > len(test) {
		warmup = 0
	}

	// Configs to benchmark.
	type benchConfig struct {
		name    string
		bits    int
		ivf     bool
		rescore int
	}
	configs := []benchConfig{
		{"float64 brute-force (baseline)", 0, false, 0},
		{"tqdb 4-bit brute-force", 4, false, 0},
		{"tqdb 4-bit IVF", 4, true, 0},
		{"tqdb 4-bit IVF + rescore=30", 4, true, 30},
		{"tqdb 5-bit brute-force", 5, false, 0},
		{"tqdb 8-bit brute-force", 8, false, 0},
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

	for _, cfg := range configs {
		fmt.Printf("--- %s ---\n", cfg.name)

		if cfg.bits == 0 {
			r := benchFloat64BruteForce(train, test, neighbors, k, warmup, dim)
			results = append(results, result{cfg.name, r.recall, r.p50, r.p95, r.qps, r.buildTime})
			continue
		}

		r := benchTQDB(train, test, neighbors, k, warmup, dim, cfg.bits, cfg.ivf, cfg.rescore)
		results = append(results, result{cfg.name, r.recall, r.p50, r.p95, r.qps, r.buildTime})
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

	for qi, query := range test {
		qNorm := vecNorm(query)
		start := time.Now()
		type scored struct {
			idx   int
			score float64
		}
		scores := make([]scored, n)
		for i, vec := range train {
			scores[i] = scored{i, vecDot(query, vec) / (qNorm * norms[i])}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		elapsed := time.Since(start)

		if qi >= warmup {
			latencies = append(latencies, elapsed)
			truthSet := make(map[int32]bool, k)
			for _, idx := range neighbors[qi][:k] {
				truthSet[idx] = true
			}
			hits := 0
			for _, s := range scores[:k] {
				if truthSet[int32(s.idx)] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}
	}

	return computeStats(latencies, totalRecall, buildTime)
}

func benchTQDB(train, test [][]float64, neighbors [][]int32, k, warmup, dim, bits int, useIVF bool, rescore int) benchResult {
	buildStart := time.Now()
	coll, _ := store.NewCollection(tqdb.Config{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard,
	})
	for i, vec := range train {
		coll.Add(fmt.Sprintf("%d", i), vec, nil)
	}
	if useIVF {
		coll.CreateIndex(tqdb.IndexConfig{})
	}
	buildTime := time.Since(buildStart)
	fmt.Printf("  Built in %s\n", buildTime.Round(time.Millisecond))

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

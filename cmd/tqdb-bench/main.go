// tqdb-bench: Comprehensive comparison benchmark between chromem-go (float32)
// and tqdb (4-bit quantized) search quality and performance.
//
// Usage:
//
//	go run ./cmd/tqdb-bench --chromem-dir ~/.local/share/csgdaa-code/vectorize --tqdb /tmp/chromem-import.tq --queries 200
package main

import (
	"compress/gzip"
	"encoding/gob"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/scotteveritt/tqdb"
)

type Document struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
	Content   string
}

func main() {
	chromemDir := flag.String("chromem-dir", "", "path to chromem-go data directory")
	tqdbPath := flag.String("tqdb", "", "path to .tq store file")
	numQueries := flag.Int("queries", 200, "number of random queries to run")
	topK := flag.Int("k", 10, "top-k for recall measurement")
	seed := flag.Uint64("seed", 42, "random seed for query selection")
	flag.Parse()

	if *chromemDir == "" || *tqdbPath == "" {
		fmt.Fprintln(os.Stderr, "usage: tqdb-bench --chromem-dir <dir> --tqdb <file> [--queries N] [--k K]")
		os.Exit(1)
	}

	// ================================================================
	// Load chromem-go data
	// ================================================================
	fmt.Println("Loading chromem-go vectors...")
	loadStart := time.Now()
	docs := loadChromemDocs(*chromemDir)
	loadTime := time.Since(loadStart)

	var memChromem runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memChromem)

	fmt.Printf("  %d vectors loaded in %s (heap: %s)\n\n",
		len(docs), loadTime.Round(time.Millisecond), fmtBytes(int64(memChromem.Alloc)))

	// ================================================================
	// Open tqdb store
	// ================================================================
	fmt.Println("Opening tqdb store...")
	openStart := time.Now()
	store, err := tqdb.OpenStore(*tqdbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer store.Close() //nolint:errcheck
	openTime := time.Since(openStart)

	var memTqdb runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memTqdb)

	fmt.Printf("  %d vectors opened in %s (heap: %s)\n\n",
		store.Len(), openTime.Round(time.Microsecond), fmtBytes(int64(memTqdb.Alloc)))

	// ================================================================
	// Select random queries from the dataset
	// ================================================================
	rng := rand.New(rand.NewPCG(*seed, 0))
	queryIndices := make([]int, *numQueries)
	for i := range queryIndices {
		queryIndices[i] = rng.IntN(len(docs))
	}

	k := *topK
	fmt.Printf("Running %d queries with k=%d...\n\n", *numQueries, k)

	// ================================================================
	// Run all queries on both systems
	// ================================================================
	results := make([]queryResult, *numQueries)

	for qi, docIdx := range queryIndices {
		doc := docs[docIdx]
		query32 := doc.Embedding
		query64 := f32to64(query32)
		qNorm := normF32(query32)

		// chromem-go search
		start := time.Now()
		cResults := chromemSearch(query32, docs, k, qNorm)
		results[qi].chromemTime = time.Since(start)
		results[qi].chromemIDs = make([]string, len(cResults))
		results[qi].chromemScores = make([]float64, len(cResults))
		for i, r := range cResults {
			results[qi].chromemIDs[i] = r.id
			results[qi].chromemScores[i] = r.score
		}

		// tqdb search
		start = time.Now()
		tResults := store.Search(query64, k)
		results[qi].tqdbTime = time.Since(start)
		results[qi].tqdbIDs = make([]string, len(tResults))
		results[qi].tqdbScores = make([]float64, len(tResults))
		for i, r := range tResults {
			results[qi].tqdbIDs[i] = r.ID
			results[qi].tqdbScores[i] = r.Score
		}

		if (qi+1)%50 == 0 {
			fmt.Printf("  %d/%d queries complete\n", qi+1, *numQueries)
		}
	}

	// ================================================================
	// Compute metrics
	// ================================================================
	fmt.Println("\n========================================")
	fmt.Println("  BENCHMARK RESULTS")
	fmt.Println("========================================")

	// --- Latency ---
	chromemLatencies := make([]time.Duration, *numQueries)
	tqdbLatencies := make([]time.Duration, *numQueries)
	for i, r := range results {
		chromemLatencies[i] = r.chromemTime
		tqdbLatencies[i] = r.tqdbTime
	}
	slices.Sort(chromemLatencies)
	slices.Sort(tqdbLatencies)

	fmt.Println("\n--- Latency (ms) ---")
	fmt.Printf("  %-12s %10s %10s %10s %10s %10s\n", "", "p50", "p90", "p95", "p99", "mean")
	fmt.Printf("  %-12s %10s %10s %10s %10s %10s\n",
		"chromem-go",
		fmtDur(percentile(chromemLatencies, 50)),
		fmtDur(percentile(chromemLatencies, 90)),
		fmtDur(percentile(chromemLatencies, 95)),
		fmtDur(percentile(chromemLatencies, 99)),
		fmtDur(mean(chromemLatencies)))
	fmt.Printf("  %-12s %10s %10s %10s %10s %10s\n",
		"tqdb",
		fmtDur(percentile(tqdbLatencies, 50)),
		fmtDur(percentile(tqdbLatencies, 90)),
		fmtDur(percentile(tqdbLatencies, 95)),
		fmtDur(percentile(tqdbLatencies, 99)),
		fmtDur(mean(tqdbLatencies)))
	fmt.Printf("  %-12s %10.1fx %10.1fx %10.1fx %10.1fx %10.1fx\n",
		"speedup",
		ratio(percentile(chromemLatencies, 50), percentile(tqdbLatencies, 50)),
		ratio(percentile(chromemLatencies, 90), percentile(tqdbLatencies, 90)),
		ratio(percentile(chromemLatencies, 95), percentile(tqdbLatencies, 95)),
		ratio(percentile(chromemLatencies, 99), percentile(tqdbLatencies, 99)),
		ratio(mean(chromemLatencies), mean(tqdbLatencies)))

	// --- Recall@k ---
	fmt.Println("\n--- Recall@k ---")
	for _, kVal := range []int{1, 3, 5, k} {
		if kVal > k {
			continue
		}
		totalRecall := 0.0
		for _, r := range results {
			recall := recallAtK(r.chromemIDs, r.tqdbIDs, kVal)
			totalRecall += recall
		}
		avgRecall := totalRecall / float64(*numQueries)
		fmt.Printf("  Recall@%-3d  %.4f  (%d/%d queries have perfect recall)\n",
			kVal, avgRecall, countPerfect(results, kVal), *numQueries)
	}

	// --- NDCG@k ---
	fmt.Println("\n--- NDCG@k ---")
	for _, kVal := range []int{1, 5, k} {
		if kVal > k {
			continue
		}
		totalNDCG := 0.0
		for _, r := range results {
			totalNDCG += ndcgAtK(r.chromemIDs, r.tqdbIDs, kVal)
		}
		fmt.Printf("  NDCG@%-4d   %.4f\n", kVal, totalNDCG/float64(*numQueries))
	}

	// --- Score correlation ---
	fmt.Println("\n--- Score Correlation (top-1) ---")
	var chromemTop1, tqdbTop1 []float64
	for _, r := range results {
		if len(r.chromemScores) > 0 && len(r.tqdbScores) > 0 {
			chromemTop1 = append(chromemTop1, r.chromemScores[0])
			tqdbTop1 = append(tqdbTop1, r.tqdbScores[0])
		}
	}
	pearson := pearsonCorrelation(chromemTop1, tqdbTop1)
	avgScoreDiff := avgDiff(chromemTop1, tqdbTop1)
	fmt.Printf("  Pearson r:        %.6f\n", pearson)
	fmt.Printf("  Avg score diff:   %.6f (chromem - tqdb)\n", avgScoreDiff)
	fmt.Printf("  Max score diff:   %.6f\n", maxDiff(chromemTop1, tqdbTop1))

	// --- Rank displacement ---
	fmt.Println("\n--- Rank Displacement ---")
	var totalDisp, maxDisp float64
	var dispCount int
	for _, r := range results {
		for rank, id := range r.chromemIDs {
			tqdbRank := indexOf(r.tqdbIDs, id)
			if tqdbRank >= 0 {
				d := math.Abs(float64(rank - tqdbRank))
				totalDisp += d
				if d > maxDisp {
					maxDisp = d
				}
				dispCount++
			}
		}
	}
	fmt.Printf("  Avg displacement: %.2f positions\n", totalDisp/float64(dispCount))
	fmt.Printf("  Max displacement: %.0f positions\n", maxDisp)

	// --- Startup ---
	fmt.Println("\n--- Startup ---")
	fmt.Printf("  chromem-go load:  %s\n", loadTime.Round(time.Millisecond))
	fmt.Printf("  tqdb open (mmap): %s\n", openTime.Round(time.Microsecond))
	fmt.Printf("  speedup:          %.0fx\n", float64(loadTime)/float64(openTime))

	// --- Storage ---
	chromemSize := dirSize(*chromemDir)
	tqdbInfo, _ := os.Stat(*tqdbPath)
	fmt.Println("\n--- Storage ---")
	fmt.Printf("  chromem-go:  %s (%d files)\n", fmtBytes(chromemSize), len(docs))
	fmt.Printf("  tqdb:        %s (1 file)\n", fmtBytes(tqdbInfo.Size()))
	fmt.Printf("  ratio:       %.1fx smaller\n", float64(chromemSize)/float64(tqdbInfo.Size()))
}

// ================================================================
// Search helpers
// ================================================================

type scored struct {
	id    string
	score float64
	meta  map[string]string
}

func chromemSearch(query []float32, docs []Document, k int, qNorm float64) []scored {
	scores := make([]scored, len(docs))
	for i, doc := range docs {
		scores[i] = scored{id: doc.ID, score: cosSimF32(query, doc.Embedding, qNorm), meta: doc.Metadata}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
	if k > len(scores) {
		k = len(scores)
	}
	return scores[:k]
}

func cosSimF32(a, b []float32, aNorm float64) float64 {
	var dot, bNormSq float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		bNormSq += bi * bi
	}
	bNorm := math.Sqrt(bNormSq)
	if bNorm < 1e-15 {
		return 0
	}
	return dot / (aNorm * bNorm)
}

func normF32(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func f32to64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

// ================================================================
// Metrics
// ================================================================

func recallAtK(truth, predicted []string, k int) float64 {
	if k > len(truth) {
		k = len(truth)
	}
	if k > len(predicted) {
		k = len(predicted)
	}
	truthSet := make(map[string]bool, k)
	for _, id := range truth[:k] {
		truthSet[id] = true
	}
	hits := 0
	for _, id := range predicted[:k] {
		if truthSet[id] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

func countPerfect(results []queryResult, k int) int {
	count := 0
	for _, r := range results {
		if recallAtK(r.chromemIDs, r.tqdbIDs, k) == 1.0 {
			count++
		}
	}
	return count
}

func ndcgAtK(truth, predicted []string, k int) float64 {
	if k > len(truth) {
		k = len(truth)
	}
	if k > len(predicted) {
		k = len(predicted)
	}

	// Relevance: position in truth list (higher = more relevant)
	relevance := make(map[string]float64, k)
	for i, id := range truth[:k] {
		relevance[id] = float64(k - i) // k for rank 0, k-1 for rank 1, etc.
	}

	// DCG of predicted
	dcg := 0.0
	for i := 0; i < k && i < len(predicted); i++ {
		rel := relevance[predicted[i]]
		dcg += rel / math.Log2(float64(i+2)) // +2 because log2(1) = 0
	}

	// Ideal DCG
	idcg := 0.0
	for i := 0; i < k; i++ {
		rel := float64(k - i)
		idcg += rel / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func pearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := range n {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	nf := float64(n)
	num := nf*sumXY - sumX*sumY
	den := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	if den < 1e-15 {
		return 0
	}
	return num / den
}

func avgDiff(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] - b[i]
	}
	return sum / float64(len(a))
}

func maxDiff(a, b []float64) float64 {
	var mx float64
	for i := range a {
		d := math.Abs(a[i] - b[i])
		if d > mx {
			mx = d
		}
	}
	return mx
}

func indexOf(slice []string, target string) int {
	for i, s := range slice {
		if s == target {
			return i
		}
	}
	return -1
}

// ================================================================
// Stats helpers
// ================================================================

func percentile(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := (p * len(sorted)) / 100
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func mean(durations []time.Duration) time.Duration {
	var total time.Duration
	for _, d := range durations {
		total += d
	}
	return total / time.Duration(len(durations))
}

func ratio(a, b time.Duration) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func fmtDur(d time.Duration) string {
	return fmt.Sprintf("%.1f", float64(d.Microseconds())/1000.0)
}

// ================================================================
// IO helpers
// ================================================================

func loadChromemDocs(dir string) []Document {
	var docs []Document
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(p, ".gob.gz") {
			return nil
		}
		f, err := os.Open(p)
		if err != nil {
			return nil
		}
		defer f.Close() //nolint:errcheck
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil
		}
		defer gz.Close() //nolint:errcheck
		var doc Document
		if err := gob.NewDecoder(gz).Decode(&doc); err != nil {
			return nil
		}
		if len(doc.Embedding) > 0 {
			docs = append(docs, doc)
		}
		return nil
	})
	return docs
}

func dirSize(path string) int64 {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}

func fmtBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

type queryResult struct {
	chromemIDs    []string
	chromemScores []float64
	chromemTime   time.Duration
	tqdbIDs       []string
	tqdbScores    []float64
	tqdbTime      time.Duration
}

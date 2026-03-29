// tqdb-test: End-to-end integration test + benchmark against live chromem-go data.
//
// Usage:
//
//	go run ./cmd/tqdb-test
package main

import (
	"compress/gzip"
	"context"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/mathutil"
	"github.com/scotteveritt/tqdb/store"
)

type chromemDoc struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
	Content   string
}

func main() {
	chromemDir := os.ExpandEnv("$HOME/.local/share/csgdaa-code/vectorize")
	storePath := "/tmp/tqdb-test.tq"

	fmt.Println("=== tqdb End-to-End Integration Test ===")

	// -----------------------------------------------
	// Phase 1: Import from chromem-go
	// -----------------------------------------------
	fmt.Println("Phase 1: Import chromem-go data")
	fmt.Printf("  Source: %s\n", chromemDir)

	importStart := time.Now()
	docs := loadChromemDocs(chromemDir)
	loadTime := time.Since(importStart)
	fmt.Printf("  Loaded %d docs in %s\n", len(docs), loadTime.Round(time.Millisecond))

	if len(docs) == 0 {
		fmt.Fprintln(os.Stderr, "error: no documents found")
		os.Exit(1)
	}

	dim := len(docs[0].Embedding)
	fmt.Printf("  Dimension: %d\n", dim)

	// Create tqdb store with content
	s, err := store.Create(storePath, tqdb.StoreConfig{
		Dim:      dim,
		Bits:     4,
		Rotation: tqdb.RotationHadamard,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	indexStart := time.Now()
	for _, doc := range docs {
		vec := f32to64(doc.Embedding)
		data := make(map[string]any, len(doc.Metadata))
		for k, v := range doc.Metadata {
			data[k] = v
		}
		if err := s.AddDocument(context.TODO(), tqdb.Document{
			ID:        doc.ID,
			Content:   doc.Content[:min(200, len(doc.Content))], // truncate for size
			Data:      data,
			Embedding: vec,
		}); err != nil {
			continue
		}
	}
	indexTime := time.Since(indexStart)
	fmt.Printf("  Indexed %d vectors in %s (%.0f vec/s)\n",
		s.Len(), indexTime.Round(time.Millisecond), float64(s.Len())/indexTime.Seconds())

	flushStart := time.Now()
	if err := s.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	flushTime := time.Since(flushStart)

	info, _ := os.Stat(storePath)
	fmt.Printf("  Flushed in %s (%s on disk)\n", flushTime.Round(time.Millisecond), fmtBytes(info.Size()))

	// -----------------------------------------------
	// Phase 2: Open store (mmap)
	// -----------------------------------------------
	fmt.Println("\nPhase 2: Open store (mmap)")
	openStart := time.Now()
	s2, err := store.Open(storePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	openTime := time.Since(openStart)
	fmt.Printf("  Opened %d vectors in %s\n", s2.Len(), openTime.Round(time.Microsecond))

	var memStats runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memStats)
	fmt.Printf("  Heap: %s\n", fmtBytes(int64(memStats.Alloc)))

	// -----------------------------------------------
	// Phase 3: Search benchmarks
	// -----------------------------------------------
	fmt.Println("\nPhase 3: Search benchmarks")

	rng := rand.New(rand.NewPCG(42, 0))
	numQueries := 100

	// Pick random query vectors from the dataset
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = rng.IntN(len(docs))
	}

	// --- Unfiltered search ---
	var unfilteredTimes []time.Duration
	for _, qi := range queryIndices {
		query := f32to64(docs[qi].Embedding)
		start := time.Now()
		_ = s2.Search(query, 10)
		unfilteredTimes = append(unfilteredTimes, time.Since(start))
	}
	sort.Slice(unfilteredTimes, func(i, j int) bool { return unfilteredTimes[i] < unfilteredTimes[j] })

	fmt.Printf("\n  Unfiltered search (top-10, %d queries):\n", numQueries)
	fmt.Printf("    p50:  %s\n", unfilteredTimes[len(unfilteredTimes)/2])
	fmt.Printf("    p95:  %s\n", unfilteredTimes[len(unfilteredTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(unfilteredTimes))

	// --- Filtered search (Eq on repo field) ---
	// Find a common repo value
	repoCounts := map[string]int{}
	for _, doc := range docs {
		repoCounts[doc.Metadata["repo"]]++
	}
	var topRepo string
	var topCount int
	for repo, count := range repoCounts {
		if count > topCount {
			topRepo = repo
			topCount = count
		}
	}
	fmt.Printf("\n  Filtered search (repo=%q, %d/%d docs, %d queries):\n",
		topRepo, topCount, len(docs), numQueries)

	var filteredTimes []time.Duration
	for _, qi := range queryIndices {
		query := f32to64(docs[qi].Embedding)
		start := time.Now()
		_ = s2.SearchWithOptions(query, tqdb.SearchOptions{
			TopK:   10,
			Filter: tqdb.Eq("repo", topRepo),
		})
		filteredTimes = append(filteredTimes, time.Since(start))
	}
	sort.Slice(filteredTimes, func(i, j int) bool { return filteredTimes[i] < filteredTimes[j] })

	fmt.Printf("    p50:  %s\n", filteredTimes[len(filteredTimes)/2])
	fmt.Printf("    p95:  %s\n", filteredTimes[len(filteredTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(filteredTimes))

	// --- Collection with filter index ---
	fmt.Printf("\n  Collection with filter index (repo=%q, %d queries):\n", topRepo, numQueries)

	coll, _ := store.NewCollection(tqdb.Config{
		Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard,
	})
	for _, doc := range docs {
		vec := f32to64(doc.Embedding)
		data := make(map[string]any, len(doc.Metadata))
		for k, v := range doc.Metadata {
			data[k] = v
		}
		_ = coll.Add("", vec, data)
	}

	// Build filter index on "repo" field
	coll.CreateIndex(tqdb.IndexConfig{FilterFields: []string{"repo"}})

	var indexedFilterTimes []time.Duration
	for _, qi := range queryIndices {
		query := f32to64(docs[qi].Embedding)
		start := time.Now()
		_ = coll.SearchWithOptions(query, tqdb.SearchOptions{
			TopK:   10,
			Filter: tqdb.Eq("repo", topRepo),
		})
		indexedFilterTimes = append(indexedFilterTimes, time.Since(start))
	}
	sort.Slice(indexedFilterTimes, func(i, j int) bool { return indexedFilterTimes[i] < indexedFilterTimes[j] })

	fmt.Printf("    p50:  %s (was %s without index)\n",
		indexedFilterTimes[len(indexedFilterTimes)/2],
		filteredTimes[len(filteredTimes)/2])
	fmt.Printf("    p95:  %s\n", indexedFilterTimes[len(indexedFilterTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(indexedFilterTimes))

	// --- IVF unfiltered search ---
	fmt.Printf("\n  IVF unfiltered search (top-10, %d queries):\n", numQueries)

	var ivfTimes []time.Duration
	for _, qi := range queryIndices {
		query := f32to64(docs[qi].Embedding)
		start := time.Now()
		_ = coll.Search(query, 10)
		ivfTimes = append(ivfTimes, time.Since(start))
	}
	sort.Slice(ivfTimes, func(i, j int) bool { return ivfTimes[i] < ivfTimes[j] })

	fmt.Printf("    p50:  %s (was %s brute-force)\n",
		ivfTimes[len(ivfTimes)/2],
		unfilteredTimes[len(unfilteredTimes)/2])
	fmt.Printf("    p95:  %s\n", ivfTimes[len(ivfTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(ivfTimes))

	// --- MinScore search ---
	var minScoreTimes []time.Duration
	for _, qi := range queryIndices {
		query := f32to64(docs[qi].Embedding)
		start := time.Now()
		_ = s2.SearchWithOptions(query, tqdb.SearchOptions{
			TopK:     100,
			MinScore: 0.7,
		})
		minScoreTimes = append(minScoreTimes, time.Since(start))
	}
	sort.Slice(minScoreTimes, func(i, j int) bool { return minScoreTimes[i] < minScoreTimes[j] })

	fmt.Printf("\n  MinScore search (score>0.7, top-100, %d queries):\n", numQueries)
	fmt.Printf("    p50:  %s\n", minScoreTimes[len(minScoreTimes)/2])
	fmt.Printf("    p95:  %s\n", minScoreTimes[len(minScoreTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(minScoreTimes))

	// --- Query (filter-only, no vector) ---
	var queryTimes []time.Duration
	for range numQueries {
		start := time.Now()
		_ = s2.Query(tqdb.QueryOptions{
			PageSize: 50,
			Filter:   tqdb.Eq("repo", topRepo),
		})
		queryTimes = append(queryTimes, time.Since(start))
	}
	sort.Slice(queryTimes, func(i, j int) bool { return queryTimes[i] < queryTimes[j] })

	fmt.Printf("\n  Query (filter-only, repo=%q, page=50, %d queries):\n", topRepo, numQueries)
	fmt.Printf("    p50:  %s\n", queryTimes[len(queryTimes)/2])
	fmt.Printf("    p95:  %s\n", queryTimes[len(queryTimes)*95/100])
	fmt.Printf("    mean: %s\n", mean(queryTimes))

	// -----------------------------------------------
	// Phase 4: Recall quality
	// -----------------------------------------------
	fmt.Println("\nPhase 4: Recall quality (vs float32 brute-force)")

	recallQueries := 50
	var recall1Sum, recall10Sum float64

	for _, qi := range queryIndices[:recallQueries] {
		query32 := docs[qi].Embedding
		query64 := f32to64(query32)
		qNorm := normF32(query32)

		// Float32 ground truth
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(docs))
		for i, d := range docs {
			scores[i] = scored{id: d.ID, score: cosSimF32(query32, d.Embedding, qNorm)}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

		// tqdb results
		results := s2.Search(query64, 10)

		// Recall@1
		if len(results) > 0 && results[0].ID == scores[0].id {
			recall1Sum += 1.0
		}

		// Recall@10
		truthSet := map[string]bool{}
		for _, s := range scores[:10] {
			truthSet[s.id] = true
		}
		hits := 0
		for _, r := range results {
			if truthSet[r.ID] {
				hits++
			}
		}
		recall10Sum += float64(hits) / 10.0
	}

	fmt.Printf("  Recall@1:  %.1f%% (%d queries)\n", recall1Sum/float64(recallQueries)*100, recallQueries)
	fmt.Printf("  Recall@10: %.1f%% (%d queries)\n", recall10Sum/float64(recallQueries)*100, recallQueries)

	// -----------------------------------------------
	// Phase 5: Content retrieval
	// -----------------------------------------------
	fmt.Println("\nPhase 5: Content retrieval")

	results := s2.Search(f32to64(docs[0].Embedding), 3)
	for i, r := range results {
		contentPreview := r.Content
		if len(contentPreview) > 80 {
			contentPreview = contentPreview[:80] + "..."
		}
		fmt.Printf("  %d. %s (score=%.4f)\n", i+1, r.ID, r.Score)
		if contentPreview != "" {
			fmt.Printf("     content: %s\n", contentPreview)
		}
		if repo, ok := r.Data["repo"]; ok {
			fmt.Printf("     repo: %s\n", repo)
		}
	}

	// -----------------------------------------------
	// Summary
	// -----------------------------------------------
	fmt.Println("\n=== Summary ===")
	fmt.Printf("  Vectors:        %d\n", s2.Len())
	fmt.Printf("  Dimension:      %d\n", dim)
	fmt.Printf("  File size:      %s\n", fmtBytes(info.Size()))
	fmt.Printf("  Open time:      %s\n", openTime.Round(time.Microsecond))
	fmt.Printf("  Search (p50):   %s\n", unfilteredTimes[len(unfilteredTimes)/2])
	fmt.Printf("  Filtered (p50): %s\n", filteredTimes[len(filteredTimes)/2])
	fmt.Printf("  Recall@10:      %.1f%%\n", recall10Sum/float64(recallQueries)*100)

	_ = s2.Close()
	_ = os.Remove(storePath)
}

func loadChromemDocs(dir string) []chromemDoc {
	var docs []chromemDoc
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
		var doc chromemDoc
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

func f32to64(v []float32) []float64 {
	return mathutil.Float32ToFloat64(v)
}

func cosSimF32(a, b []float32, aNorm float64) float64 {
	var dot, bNormSq float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		bNormSq += bi * bi
	}
	return dot / (aNorm * math.Sqrt(bNormSq))
}

func normF32(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func mean(durations []time.Duration) time.Duration {
	var total time.Duration
	for _, d := range durations {
		total += d
	}
	return total / time.Duration(len(durations))
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

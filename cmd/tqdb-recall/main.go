// tqdb-recall: Measure recall vs latency tradeoffs for IVF tuning.
package main

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/mathutil"
	"github.com/scotteveritt/tqdb/store"
)

type doc struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
}

func main() {
	fmt.Println("Loading chromem-go vectors...")
	docs := loadDocs()
	fmt.Printf("  %d vectors, d=%d\n\n", len(docs), len(docs[0].Embedding))

	dim := len(docs[0].Embedding)
	numQueries := 100
	rng := rand.New(rand.NewPCG(42, 0))
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = rng.IntN(len(docs))
	}

	// Compute float32 ground truth for all queries.
	fmt.Println("Computing float32 ground truth...")
	type truthResult struct {
		ids []string
	}
	truths := make([]truthResult, numQueries)
	for qi, idx := range queryIndices {
		query := docs[idx].Embedding
		qNorm := normF32(query)
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(docs))
		for i, d := range docs {
			scores[i] = scored{id: d.ID, score: cosSimF32(query, d.Embedding, qNorm)}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		ids := make([]string, 10)
		for i := range 10 {
			ids[i] = scores[i].id
		}
		truths[qi] = truthResult{ids: ids}
	}
	fmt.Println("  Done")

	// Test configurations.
	configs := []struct {
		name          string
		numPartitions int
		nProbe        int
		rescore       int
	}{
		{"brute-force (no IVF)", 0, 0, 0},
		{"IVF default (auto)", 0, 0, 0},       // will use auto
		{"IVF nProbe=2x", 0, 0, 0},            // double nProbe
		{"IVF nProbe=4x", 0, 0, 0},            // 4x nProbe
		{"IVF + rescore=30", 0, 0, 30},
		{"IVF + rescore=50", 0, 0, 50},
		{"IVF nProbe=2x + rescore=30", 0, 0, 30},
	}

	fmt.Printf("%-35s %10s %10s %10s\n", "Config", "Recall@10", "p50 (ms)", "p95 (ms)")
	fmt.Printf("%-35s %10s %10s %10s\n", "---", "---", "---", "---")

	for ci, cfg := range configs {
		coll, _ := store.NewCollection(tqdb.Config{
			Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard,
		})
		for _, d := range docs {
			_ = coll.Add(d.ID, mathutil.Float32ToFloat64(d.Embedding), nil)
		}

		if ci == 0 {
			// No index — brute force
		} else {
			autoP := int(math.Sqrt(float64(len(docs))))
			autoNProbe := int(math.Sqrt(float64(autoP)))

			np := autoP
			nprobe := autoNProbe

			switch ci {
			case 1: // default
			case 2: // 2x nProbe
				nprobe = autoNProbe * 2
			case 3: // 4x nProbe
				nprobe = autoNProbe * 4
			case 4: // default + rescore
			case 5: // default + rescore=50
			case 6: // 2x nProbe + rescore
				nprobe = autoNProbe * 2
			}

			coll.CreateIndex(tqdb.IndexConfig{
				NumPartitions: np,
				NProbe:        nprobe,
			})
		}

		var latencies []time.Duration
		var totalRecall float64

		for qi, idx := range queryIndices {
			query := mathutil.Float32ToFloat64(docs[idx].Embedding)
			start := time.Now()
			var results []tqdb.Result
			if ci == 0 {
				results = coll.Search(query, 10)
			} else {
				results = coll.SearchWithOptions(query, tqdb.SearchOptions{
					TopK:    10,
					Rescore: cfg.rescore,
				})
			}
			latencies = append(latencies, time.Since(start))

			// Compute recall@10
			truthSet := map[string]bool{}
			for _, id := range truths[qi].ids {
				truthSet[id] = true
			}
			hits := 0
			for _, r := range results {
				if truthSet[r.ID] {
					hits++
				}
			}
			totalRecall += float64(hits) / 10.0
		}

		sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
		avgRecall := totalRecall / float64(numQueries)
		p50 := latencies[len(latencies)/2]
		p95 := latencies[len(latencies)*95/100]

		fmt.Printf("%-35s %9.1f%% %9.1fms %9.1fms\n",
			cfg.name, avgRecall*100,
			float64(p50.Microseconds())/1000,
			float64(p95.Microseconds())/1000)
	}
}

func loadDocs() []doc {
	dir := os.ExpandEnv("$HOME/.local/share/csgdaa-code/vectorize")
	var docs []doc
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
		var d doc
		if err := gob.NewDecoder(gz).Decode(&d); err != nil {
			return nil
		}
		if len(d.Embedding) > 0 {
			docs = append(docs, d)
		}
		return nil
	})
	return docs
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

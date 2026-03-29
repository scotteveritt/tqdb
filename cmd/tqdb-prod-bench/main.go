// tqdb-prod-bench: Compare MSE-only vs Prod (MSE+QJL) recall and latency.
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
	"github.com/scotteveritt/tqdb/quantize"
)

type doc struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
}

func main() {
	fmt.Println("Loading vectors...")
	docs := loadDocs()
	dim := len(docs[0].Embedding)
	fmt.Printf("  %d vectors, d=%d\n\n", len(docs), dim)

	numQueries := 100
	rng := rand.New(rand.NewPCG(42, 0))
	queryIndices := make([]int, numQueries)
	for i := range queryIndices {
		queryIndices[i] = rng.IntN(len(docs))
	}

	// Float32 ground truth.
	fmt.Println("Computing ground truth...")
	type truth struct {
		ids []string
	}
	truths := make([]truth, numQueries)
	for qi, idx := range queryIndices {
		q := docs[idx].Embedding
		qN := normF32(q)
		type s struct {
			id    string
			score float64
		}
		scores := make([]s, len(docs))
		for i, d := range docs {
			scores[i] = s{d.ID, cosSimF32(q, d.Embedding, qN)}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		ids := make([]string, 10)
		for i := range 10 {
			ids[i] = scores[i].id
		}
		truths[qi] = truth{ids}
	}

	// ==============================
	// Config 1: 4-bit MSE-only
	// ==============================
	fmt.Println("\n=== 4-bit MSE-only (current default) ===")
	qMSE, _ := quantize.NewMSE(tqdb.Config{Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard})

	// Quantize all vectors with MSE.
	type mseEntry struct {
		id      string
		indices []uint8
	}
	mseEntries := make([]mseEntry, len(docs))
	for i, d := range docs {
		vec := mathutil.Float32ToFloat64(d.Embedding)
		cv := qMSE.Quantize(vec)
		mseEntries[i] = mseEntry{d.ID, cv.Indices}
	}

	centroids := qMSE.Codebook().Centroids
	workDim := qMSE.Rotation().WorkDim()

	// Search with MSE inner product.
	var mseRecall float64
	var mseTimes []time.Duration
	for qi, idx := range queryIndices {
		query := mathutil.Float32ToFloat64(docs[idx].Embedding)
		qNorm := mathutil.Norm(query)
		invQN := 1.0 / qNorm
		unitQ := make([]float64, dim)
		for i, v := range query {
			unitQ[i] = v * invQN
		}
		qRot := make([]float64, workDim)
		qMSE.Rotation().Rotate(qRot, unitQ[:dim])

		start := time.Now()
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(docs))
		for i, e := range mseEntries {
			var dot float64
			for j, cidx := range e.indices {
				dot += qRot[j] * centroids[cidx]
			}
			scores[i] = scored{e.id, dot}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		mseTimes = append(mseTimes, time.Since(start))

		truthSet := map[string]bool{}
		for _, id := range truths[qi].ids {
			truthSet[id] = true
		}
		hits := 0
		for _, s := range scores[:10] {
			if truthSet[s.id] {
				hits++
			}
		}
		mseRecall += float64(hits) / 10.0
	}
	sort.Slice(mseTimes, func(i, j int) bool { return mseTimes[i] < mseTimes[j] })
	fmt.Printf("  Recall@10: %.1f%%\n", mseRecall/float64(numQueries)*100)
	fmt.Printf("  Search p50: %s\n", mseTimes[len(mseTimes)/2])

	// ==============================
	// Config 2: 4-bit Prod (3+1)
	// ==============================
	fmt.Println("\n=== 4-bit Prod (3-bit MSE + 1-bit QJL) — paper's approach ===")
	qProd, _ := quantize.NewProd(tqdb.Config{Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard})

	// Quantize all vectors with Prod.
	type prodEntry struct {
		id           string
		cv           *tqdb.CompressedProdVector
	}
	fmt.Println("  Quantizing with Prod (3+1 bits)...")
	prodStart := time.Now()
	prodEntries := make([]prodEntry, len(docs))
	for i, d := range docs {
		vec := mathutil.Float32ToFloat64(d.Embedding)
		cv := qProd.Quantize(vec)
		prodEntries[i] = prodEntry{d.ID, cv}
	}
	fmt.Printf("  Quantized in %s\n", time.Since(prodStart).Round(time.Millisecond))

	// Search with Prod inner product (unbiased estimator).
	var prodRecall float64
	var prodTimes []time.Duration
	for qi, idx := range queryIndices {
		query := mathutil.Float32ToFloat64(docs[idx].Embedding)

		// Pre-project the query once (O(d²)), then each vector is O(d).
		sqProjected := qProd.ProjectQuery(query)

		start := time.Now()
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(docs))
		for i, e := range prodEntries {
			ip := qProd.InnerProductProjected(query, sqProjected, e.cv)
			scores[i] = scored{e.id, ip}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		prodTimes = append(prodTimes, time.Since(start))

		truthSet := map[string]bool{}
		for _, id := range truths[qi].ids {
			truthSet[id] = true
		}
		hits := 0
		for _, s := range scores[:10] {
			if truthSet[s.id] {
				hits++
			}
		}
		prodRecall += float64(hits) / 10.0
	}
	sort.Slice(prodTimes, func(i, j int) bool { return prodTimes[i] < prodTimes[j] })
	fmt.Printf("  Recall@10: %.1f%%\n", prodRecall/float64(numQueries)*100)
	fmt.Printf("  Search p50: %s\n", prodTimes[len(prodTimes)/2])

	// ==============================
	// Config 3: 5-bit MSE-only (for comparison — same storage as 4-bit Prod with signs)
	// ==============================
	fmt.Println("\n=== 5-bit MSE-only (extra bit comparison) ===")
	qMSE5, _ := quantize.NewMSE(tqdb.Config{Dim: dim, Bits: 5, Rotation: tqdb.RotationHadamard})

	mse5Entries := make([]mseEntry, len(docs))
	for i, d := range docs {
		vec := mathutil.Float32ToFloat64(d.Embedding)
		cv := qMSE5.Quantize(vec)
		mse5Entries[i] = mseEntry{d.ID, cv.Indices}
	}

	centroids5 := qMSE5.Codebook().Centroids
	workDim5 := qMSE5.Rotation().WorkDim()

	var mse5Recall float64
	var mse5Times []time.Duration
	for qi, idx := range queryIndices {
		query := mathutil.Float32ToFloat64(docs[idx].Embedding)
		qNorm := mathutil.Norm(query)
		invQN := 1.0 / qNorm
		unitQ := make([]float64, dim)
		for i, v := range query {
			unitQ[i] = v * invQN
		}
		qRot := make([]float64, workDim5)
		qMSE5.Rotation().Rotate(qRot, unitQ[:dim])

		start := time.Now()
		type scored struct {
			id    string
			score float64
		}
		scores := make([]scored, len(docs))
		for i, e := range mse5Entries {
			var dot float64
			for j, cidx := range e.indices {
				dot += qRot[j] * centroids5[cidx]
			}
			scores[i] = scored{e.id, dot}
		}
		sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })
		mse5Times = append(mse5Times, time.Since(start))

		truthSet := map[string]bool{}
		for _, id := range truths[qi].ids {
			truthSet[id] = true
		}
		hits := 0
		for _, s := range scores[:10] {
			if truthSet[s.id] {
				hits++
			}
		}
		mse5Recall += float64(hits) / 10.0
	}
	sort.Slice(mse5Times, func(i, j int) bool { return mse5Times[i] < mse5Times[j] })
	fmt.Printf("  Recall@10: %.1f%%\n", mse5Recall/float64(numQueries)*100)
	fmt.Printf("  Search p50: %s\n", mse5Times[len(mse5Times)/2])

	// ==============================
	// Summary
	// ==============================
	fmt.Println("\n=== Summary ===")
	fmt.Printf("%-35s %10s %12s\n", "Config", "Recall@10", "Search p50")
	fmt.Printf("%-35s %10s %12s\n", "---", "---", "---")
	fmt.Printf("%-35s %9.1f%% %12s\n", "4-bit MSE-only (current)",
		mseRecall/float64(numQueries)*100, mseTimes[len(mseTimes)/2])
	fmt.Printf("%-35s %9.1f%% %12s\n", "4-bit Prod (3+1, paper's method)",
		prodRecall/float64(numQueries)*100, prodTimes[len(prodTimes)/2])
	fmt.Printf("%-35s %9.1f%% %12s\n", "5-bit MSE-only (extra bit)",
		mse5Recall/float64(numQueries)*100, mse5Times[len(mse5Times)/2])
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
		dot += float64(a[i]) * float64(b[i])
		bNormSq += float64(b[i]) * float64(b[i])
	}
	return dot / (aNorm * math.Sqrt(bNormSq))
}

func normF32(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

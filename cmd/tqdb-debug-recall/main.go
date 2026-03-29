// tqdb-debug-recall: Diagnose why recall@10 doesn't match paper expectations.
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
	fmt.Printf("%d docs, d=%d\n\n", len(docs), len(docs[0].Embedding))

	dim := len(docs[0].Embedding)
	q, _ := quantize.NewMSE(tqdb.Config{Dim: dim, Bits: 4, Rotation: tqdb.RotationHadamard})

	rng := rand.New(rand.NewPCG(42, 0))

	// Measure actual cosine sim preservation
	var totalCosSim float64
	nSample := 500
	for range nSample {
		i := rng.IntN(len(docs))
		vec := mathutil.Float32ToFloat64(docs[i].Embedding)
		cv := q.Quantize(vec)
		recon := q.Dequantize(cv)
		totalCosSim += mathutil.CosineSimilarity(vec, recon)
	}
	fmt.Printf("Avg cos_sim(original, dequantized): %.4f%% (%d samples)\n\n",
		totalCosSim/float64(nSample)*100, nSample)

	// Deep-dive on recall misses
	numQueries := 50
	totalRecallBrute := 0.0
	totalRecallDecomp := 0.0
	totalNearMisses := 0

	for trial := range numQueries {
		idx := rng.IntN(len(docs))
		query32 := docs[idx].Embedding
		query64 := mathutil.Float32ToFloat64(query32)
		qNorm := normF32(query32)

		// Float32 ground truth
		type scored struct {
			id    string
			score float64
		}
		f32scores := make([]scored, len(docs))
		for i, d := range docs {
			f32scores[i] = scored{d.ID, cosSimF32(query32, d.Embedding, qNorm)}
		}
		sort.Slice(f32scores, func(i, j int) bool { return f32scores[i].score > f32scores[j].score })

		// Method 1: Inner product in rotated space (what our search does)
		qUnit := make([]float64, dim)
		invQN := 1.0 / math.Sqrt(mathutil.Dot(query64, query64))
		for i, v := range query64 {
			qUnit[i] = v * invQN
		}
		qRot := make([]float64, q.Rotation().WorkDim())
		q.Rotation().Rotate(qRot, qUnit[:dim])
		centroids := q.Codebook().Centroids

		ipScores := make([]scored, len(docs))
		for i, d := range docs {
			vec := mathutil.Float32ToFloat64(d.Embedding)
			cv := q.Quantize(vec)
			var dot float64
			for j, cidx := range cv.Indices {
				dot += qRot[j] * centroids[cidx]
			}
			ipScores[i] = scored{d.ID, dot}
		}
		sort.Slice(ipScores, func(i, j int) bool { return ipScores[i].score > ipScores[j].score })

		// Method 2: Dequantize and compute exact cosine sim
		decompScores := make([]scored, len(docs))
		for i, d := range docs {
			vec := mathutil.Float32ToFloat64(d.Embedding)
			cv := q.Quantize(vec)
			recon := q.Dequantize(cv)
			decompScores[i] = scored{d.ID, mathutil.CosineSimilarity(query64, recon)}
		}
		sort.Slice(decompScores, func(i, j int) bool { return decompScores[i].score > decompScores[j].score })

		// Compute recall for both methods
		truthSet := map[string]bool{}
		for _, s := range f32scores[:10] {
			truthSet[s.id] = true
		}

		hitsIP := 0
		for _, s := range ipScores[:10] {
			if truthSet[s.id] {
				hitsIP++
			}
		}
		hitsDecomp := 0
		for _, s := range decompScores[:10] {
			if truthSet[s.id] {
				hitsDecomp++
			}
		}

		totalRecallBrute += float64(hitsIP) / 10.0
		totalRecallDecomp += float64(hitsDecomp) / 10.0

		// Count near-misses: how close is the gap?
		if hitsIP < 10 {
			gap := f32scores[9].score - f32scores[10].score
			if gap < 0.01 {
				totalNearMisses++
			}
		}

		if trial < 5 && hitsIP < 10 {
			fmt.Printf("Trial %d: IP recall=%d/10, Decomp recall=%d/10\n", trial, hitsIP, hitsDecomp)
			fmt.Printf("  F32 #9 score: %.6f, F32 #10 score: %.6f, gap: %.6f\n",
				f32scores[9].score, f32scores[10].score,
				f32scores[9].score-f32scores[10].score)
			fmt.Println()
		}
	}

	fmt.Printf("Recall@10 (inner product, rotated): %.1f%%\n", totalRecallBrute/float64(numQueries)*100)
	fmt.Printf("Recall@10 (decompress + exact cos):  %.1f%%\n", totalRecallDecomp/float64(numQueries)*100)
	fmt.Printf("Near-misses (gap < 0.01):            %d/%d\n", totalNearMisses, numQueries)
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

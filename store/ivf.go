package store

import (
	"math"
	"math/rand/v2"
	"sort"
)

// ivfIndex holds IVF (Inverted File) partitions for approximate nearest neighbor search.
// Vectors are clustered into √N partitions via k-means on their rotated centroid representations.
// At search time, only the top-P nearest partitions are scored (P = nProbe).
type ivfIndex struct {
	numPartitions int
	nProbe        int         // number of partitions to search
	centroids     [][]float64 // [numPartitions][workDim] — partition centroids in rotated space
	partitions    [][]int     // [numPartitions][]int — vector indices per partition
}

// buildIVF constructs an IVF index over the collection's quantized vectors.
// It operates in rotated space (centroid lookups) matching the search inner loop.
func buildIVF(allIndices []uint8, codebookCentroids []float64, workDim, n, numPartitions, nProbe int, deleted []bool) *ivfIndex {
	if n < numPartitions {
		return nil // too few vectors for partitioning
	}

	// Reconstruct rotated-space vectors from quantized indices (centroid lookup).
	// These are the same values the search inner loop dots against.
	vectors := make([][]float64, n)
	for i := range n {
		if deleted[i] {
			continue
		}
		vec := make([]float64, workDim)
		off := i * workDim
		for j := range workDim {
			vec[j] = codebookCentroids[allIndices[off+j]]
		}
		vectors[i] = vec
	}

	// k-means++ initialization + Lloyd's algorithm.
	centroids := kmeanspp(vectors, numPartitions, deleted)
	centroids = lloyds(vectors, centroids, numPartitions, workDim, deleted, 20)

	// Assign each vector to its nearest partition.
	partitions := make([][]int, numPartitions)
	for i := range n {
		if deleted[i] {
			continue
		}
		nearest := nearestCentroid(vectors[i], centroids)
		partitions[nearest] = append(partitions[nearest], i)
	}

	return &ivfIndex{
		numPartitions: numPartitions,
		nProbe:        nProbe,
		centroids:     centroids,
		partitions:    partitions,
	}
}

// findNearestPartitions returns the indices of the top-P nearest partition centroids to the query.
func (idx *ivfIndex) findNearestPartitions(queryRotated []float64) []int {
	type scored struct {
		partition int
		score     float64
	}
	scores := make([]scored, idx.numPartitions)
	for p := range idx.numPartitions {
		var dot float64
		for j, v := range queryRotated {
			if j < len(idx.centroids[p]) {
				dot += v * idx.centroids[p][j]
			}
		}
		scores[p] = scored{partition: p, score: dot}
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	nProbe := idx.nProbe
	if nProbe > idx.numPartitions {
		nProbe = idx.numPartitions
	}

	result := make([]int, nProbe)
	for i := range nProbe {
		result[i] = scores[i].partition
	}
	return result
}

// candidatesFromPartitions returns the union of vector indices from the given partitions.
func (idx *ivfIndex) candidatesFromPartitions(partitionIDs []int) map[int]struct{} {
	total := 0
	for _, p := range partitionIDs {
		total += len(idx.partitions[p])
	}
	candidates := make(map[int]struct{}, total)
	for _, p := range partitionIDs {
		for _, vecIdx := range idx.partitions[p] {
			candidates[vecIdx] = struct{}{}
		}
	}
	return candidates
}

// --- k-means implementation ---

// kmeanspp selects initial centroids using k-means++ seeding.
func kmeanspp(vectors [][]float64, k int, deleted []bool) [][]float64 {
	n := len(vectors)
	rng := rand.New(rand.NewPCG(42, 0))

	// Find first non-deleted vector.
	first := 0
	for first < n && (deleted[first] || vectors[first] == nil) {
		first++
	}
	if first >= n {
		return nil
	}

	centroids := make([][]float64, 0, k)
	centroids = append(centroids, copyVec(vectors[first]))

	// Distance from each point to nearest existing centroid.
	minDist := make([]float64, n)
	for i := range n {
		minDist[i] = math.MaxFloat64
	}

	for len(centroids) < k {
		// Update minimum distances to nearest centroid.
		last := centroids[len(centroids)-1]
		var totalDist float64
		for i := range n {
			if deleted[i] || vectors[i] == nil {
				minDist[i] = 0
				continue
			}
			d := sqDist(vectors[i], last)
			if d < minDist[i] {
				minDist[i] = d
			}
			totalDist += minDist[i]
		}

		// Weighted random selection.
		if totalDist <= 0 {
			break
		}
		target := rng.Float64() * totalDist
		var cumulative float64
		chosen := first
		for i := range n {
			cumulative += minDist[i]
			if cumulative >= target {
				chosen = i
				break
			}
		}
		centroids = append(centroids, copyVec(vectors[chosen]))
	}

	return centroids
}

// lloyds runs Lloyd's algorithm for k-means refinement.
func lloyds(vectors, centroids [][]float64, k, dim int, deleted []bool, maxIter int) [][]float64 {
	n := len(vectors)
	assignments := make([]int, n)

	for iter := range maxIter {
		_ = iter

		// Assign each vector to nearest centroid.
		changed := false
		for i := range n {
			if deleted[i] || vectors[i] == nil {
				continue
			}
			nearest := nearestCentroid(vectors[i], centroids)
			if nearest != assignments[i] {
				assignments[i] = nearest
				changed = true
			}
		}

		if !changed {
			break
		}

		// Recompute centroids.
		newCentroids := make([][]float64, k)
		counts := make([]int, k)
		for p := range k {
			newCentroids[p] = make([]float64, dim)
		}

		for i := range n {
			if deleted[i] || vectors[i] == nil {
				continue
			}
			p := assignments[i]
			counts[p]++
			for j := range dim {
				newCentroids[p][j] += vectors[i][j]
			}
		}

		for p := range k {
			if counts[p] > 0 {
				invN := 1.0 / float64(counts[p])
				for j := range dim {
					newCentroids[p][j] *= invN
				}
			} else {
				// Empty cluster — keep old centroid.
				copy(newCentroids[p], centroids[p])
			}
		}

		centroids = newCentroids
	}

	return centroids
}

func nearestCentroid(vec []float64, centroids [][]float64) int {
	best := 0
	bestDot := -math.MaxFloat64
	for p, c := range centroids {
		var dot float64
		for j := range vec {
			if j < len(c) {
				dot += vec[j] * c[j]
			}
		}
		if dot > bestDot {
			bestDot = dot
			best = p
		}
	}
	return best
}

func sqDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		if i < len(b) {
			d := a[i] - b[i]
			sum += d * d
		}
	}
	return sum
}

func copyVec(v []float64) []float64 {
	c := make([]float64, len(v))
	copy(c, v)
	return c
}

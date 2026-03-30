package store

import (
	"math"
	"math/rand/v2"
)

// ivfIndex holds IVF (Inverted File) partitions for approximate nearest neighbor search.
// Vectors are clustered into √N partitions via k-means on their rotated centroid representations.
// At search time, only the top-P nearest partitions are scored (P = nProbe).
type ivfIndex struct {
	numPartitions int
	nProbe        int         // number of partitions to search
	centroids     [][]float64 // [numPartitions][workDim] — partition centroids in rotated space
	partitions    [][]int     // [numPartitions][]int — vector indices per partition

	// Reusable buffers to avoid per-search allocations.
	probeBuf []int     // reused by findNearestPartitions (length nProbe)
	scoreBuf []float64 // reused by findNearestPartitions (length numPartitions)
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
		probeBuf:      make([]int, nProbe),
		scoreBuf:      make([]float64, numPartitions),
	}
}

// findNearestPartitions returns the indices of the top-P nearest partition centroids
// using a partial selection (no full sort). Reuses pre-allocated buffers.
func (idx *ivfIndex) findNearestPartitions(queryRotated []float64, scoreBuf []float64) []int {
	k := idx.numPartitions

	// Compute dot products against all centroids.
	// Reuse caller-provided scoreBuf to avoid allocation.
	if len(scoreBuf) < k {
		scoreBuf = make([]float64, k)
	}
	for p := range k {
		c := idx.centroids[p]
		var dot float64
		j := 0
		for ; j <= len(c)-4; j += 4 {
			dot += queryRotated[j]*c[j] +
				queryRotated[j+1]*c[j+1] +
				queryRotated[j+2]*c[j+2] +
				queryRotated[j+3]*c[j+3]
		}
		for ; j < len(c); j++ {
			dot += queryRotated[j] * c[j]
		}
		scoreBuf[p] = dot
	}

	// Partial top-nProbe selection via linear scan (faster than sort for small nProbe).
	nProbe := idx.nProbe
	if nProbe > k {
		nProbe = k
	}

	// Reuse the result buffer stored on the index.
	result := idx.probeBuf
	if len(result) < nProbe {
		result = make([]int, nProbe)
		idx.probeBuf = result
	}
	result = result[:nProbe]

	// Initialize with first nProbe partitions.
	for i := range nProbe {
		result[i] = i
	}
	// Find the minimum score among the current top set.
	minIdx := 0
	minVal := scoreBuf[0]
	for i := 1; i < nProbe; i++ {
		if scoreBuf[result[i]] < minVal {
			minVal = scoreBuf[result[i]]
			minIdx = i
		}
	}
	// Scan remaining partitions, replacing the minimum when a better one is found.
	for p := nProbe; p < k; p++ {
		if scoreBuf[p] > minVal {
			result[minIdx] = p
			// Re-find minimum in the result set.
			minIdx = 0
			minVal = scoreBuf[result[0]]
			for i := 1; i < nProbe; i++ {
				if scoreBuf[result[i]] < minVal {
					minVal = scoreBuf[result[i]]
					minIdx = i
				}
			}
		}
	}
	return result
}

// candidatesFromPartitions iterates over candidate vector indices from selected partitions,
// calling fn for each candidate. Avoids allocating a map.
func (idx *ivfIndex) forEachCandidate(partitionIDs []int, fn func(int)) {
	if len(partitionIDs) == 1 {
		// Fast path: single partition, no dedup needed.
		for _, vecIdx := range idx.partitions[partitionIDs[0]] {
			fn(vecIdx)
		}
		return
	}
	// Multiple partitions: use a bitset for O(1) dedup when N is known,
	// or iterate partition slices directly (partitions are disjoint by construction).
	// IVF partitions are disjoint — each vector belongs to exactly one partition.
	// No dedup needed.
	for _, p := range partitionIDs {
		for _, vecIdx := range idx.partitions[p] {
			fn(vecIdx)
		}
	}
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

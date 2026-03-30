package store

import (
	"math"
	"math/rand/v2"
	"sync"
)

// hnswIndex implements the Hierarchical Navigable Small World graph for
// approximate nearest neighbor search. Operates on quantized vectors
// using the same centroid-lookup scoring as brute-force.
//
// Key design choices (informed by Weaviate's implementation):
//   - Exponential level assignment: -log(rand) / ln(M)
//   - Diversity heuristic for neighbor selection
//   - Tombstone-based lazy deletion
//   - O(1)-reset visited set
type hnswIndex struct {
	mu sync.RWMutex

	M         int     // max edges per layer (default 16)
	Mmax0     int     // max edges at layer 0 (default 2*M)
	efConst   int     // beam width during construction (default 200)
	levelMul  float64 // 1 / ln(M)
	maxLevel  int     // current max level in the graph
	entryNode int     // entry point node ID (-1 if empty)

	// Graph structure: edges[nodeID][level] = []neighborID
	edges [][]uint32Slice

	// Tombstones for lazy deletion
	deleted []bool

	// Node count (including deleted)
	count int

	// RNG for level assignment (deterministic for reproducibility)
	rng *rand.Rand
}

// uint32Slice is a named type for a neighbor list.
type uint32Slice = []uint32

// hnswConfig holds HNSW construction parameters.
type hnswConfig struct {
	M              int // max edges per layer (default 16)
	EfConstruction int // beam width during build (default 200)
	Seed           uint64
}

func (c *hnswConfig) withDefaults() hnswConfig {
	out := *c
	if out.M <= 0 {
		out.M = 16
	}
	if out.EfConstruction <= 0 {
		out.EfConstruction = 200
	}
	if out.Seed == 0 {
		out.Seed = 42
	}
	return out
}

// newHNSW creates an empty HNSW index.
func newHNSW(cfg hnswConfig) *hnswIndex {
	cfg = cfg.withDefaults()
	return &hnswIndex{
		M:         cfg.M,
		Mmax0:     cfg.M * 2,
		efConst:   cfg.EfConstruction,
		levelMul:  1.0 / math.Log(float64(cfg.M)),
		entryNode: -1,
		rng:       rand.New(rand.NewPCG(cfg.Seed, 0)),
	}
}

// assignLevel returns a random level for a new node.
// Level 0 is always included. Higher levels are exponentially rarer.
func (h *hnswIndex) assignLevel() int {
	return int(math.Floor(-math.Log(h.rng.Float64()) * h.levelMul))
}

// candidate is a scored node used in beam search.
type candidate struct {
	id   uint32
	dist float32
}

// Insert adds a node to the HNSW graph. distFunc(a, b) returns the
// distance between nodes a and b (lower = more similar for L2,
// higher = more similar for inner product). We use negative inner
// product as distance so lower = better.
func (h *hnswIndex) Insert(nodeID int, distFunc func(a, b int) float32) {
	h.mu.Lock()
	defer h.mu.Unlock()

	level := h.assignLevel()

	// Ensure edges slice is large enough.
	for len(h.edges) <= nodeID {
		h.edges = append(h.edges, nil)
		h.deleted = append(h.deleted, false)
	}

	// Allocate edge lists for this node at all levels up to `level`.
	h.edges[nodeID] = make([]uint32Slice, level+1)
	h.count++

	// First node: set as entry point and return.
	if h.entryNode < 0 {
		h.entryNode = nodeID
		h.maxLevel = level
		return
	}

	// Greedy search from top to the node's level, finding the single
	// nearest neighbor at each layer.
	ep := uint32(h.entryNode)
	epDist := distFunc(nodeID, int(ep))

	for lc := h.maxLevel; lc > level; lc-- {
		changed := true
		for changed {
			changed = false
			if lc < len(h.edges[ep]) {
				for _, neighbor := range h.edges[ep][lc] {
					if h.deleted[neighbor] {
						continue
					}
					d := distFunc(nodeID, int(neighbor))
					if d < epDist {
						ep = neighbor
						epDist = d
						changed = true
					}
				}
			}
		}
	}

	// From the node's level down to 0, do beam search and connect.
	for lc := min(level, h.maxLevel); lc >= 0; lc-- {
		candidates := h.searchLayer(nodeID, []candidate{{id: ep, dist: epDist}}, h.efConst, lc, distFunc)

		// Select neighbors using diversity heuristic.
		maxEdges := h.M
		if lc == 0 {
			maxEdges = h.Mmax0
		}
		neighbors := h.selectNeighborsDiversity(nodeID, candidates, maxEdges, distFunc)

		// Connect node to neighbors.
		h.edges[nodeID][lc] = neighbors

		// Connect neighbors back to node (bidirectional).
		for _, neighbor := range neighbors {
			if int(neighbor) >= len(h.edges) || h.edges[neighbor] == nil {
				continue
			}
			if lc >= len(h.edges[neighbor]) {
				continue
			}
			nEdges := h.edges[neighbor][lc]
			nEdges = append(nEdges, uint32(nodeID))

			// Prune if over capacity.
			if len(nEdges) > maxEdges {
				// Re-select using diversity heuristic.
				cands := make([]candidate, len(nEdges))
				for i, nid := range nEdges {
					cands[i] = candidate{id: nid, dist: distFunc(int(neighbor), int(nid))}
				}
				nEdges = h.selectNeighborsDiversity(int(neighbor), cands, maxEdges, distFunc)
			}
			h.edges[neighbor][lc] = nEdges
		}

		// Update entry point for next layer.
		if len(candidates) > 0 {
			ep = candidates[0].id
			epDist = candidates[0].dist
		}
	}

	// Update global entry point if this node has a higher level.
	if level > h.maxLevel {
		h.maxLevel = level
		h.entryNode = nodeID
	}
}

// Search finds the top-k nearest neighbors to a query node.
// distToQuery(nodeID) returns the distance from the query to the given node.
func (h *hnswIndex) Search(distToQuery func(nodeID int) float32, k, ef int) []candidate {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.entryNode < 0 || h.count == 0 {
		return nil
	}
	if ef < k {
		ef = k
	}

	ep := uint32(h.entryNode)
	epDist := distToQuery(int(ep))

	// Greedy descent from top level to level 1.
	for lc := h.maxLevel; lc > 0; lc-- {
		changed := true
		for changed {
			changed = false
			if lc < len(h.edges[ep]) {
				for _, neighbor := range h.edges[ep][lc] {
					if h.deleted[neighbor] {
						continue
					}
					d := distToQuery(int(neighbor))
					if d < epDist {
						ep = neighbor
						epDist = d
						changed = true
					}
				}
			}
		}
	}

	// Beam search at layer 0.
	results := h.searchLayer(-1, []candidate{{id: ep, dist: epDist}}, ef, 0,
		func(_, b int) float32 { return distToQuery(b) })

	// Return top-k.
	if len(results) > k {
		results = results[:k]
	}
	return results
}

// searchLayer performs beam search at a single layer.
// queryNode is the node being inserted (-1 for search queries).
// entryPoints are the starting candidates. ef is the beam width.
func (h *hnswIndex) searchLayer(queryNode int, entryPoints []candidate, ef, level int, distFunc func(a, b int) float32) []candidate {
	visited := newVisitedSet(len(h.edges))
	for _, ep := range entryPoints {
		visited.Visit(ep.id)
	}

	// Candidates: sorted by distance ascending (best first).
	cands := make([]candidate, len(entryPoints))
	copy(cands, entryPoints)

	// Results: keep up to ef best.
	results := make([]candidate, len(entryPoints))
	copy(results, entryPoints)

	for len(cands) > 0 {
		// Pop best candidate.
		best := cands[0]
		cands = cands[1:]

		// Worst result so far.
		worstDist := results[len(results)-1].dist
		if best.dist > worstDist && len(results) >= ef {
			break // No improvement possible.
		}

		// Expand neighbors.
		if int(best.id) < len(h.edges) && h.edges[best.id] != nil && level < len(h.edges[best.id]) {
			for _, neighbor := range h.edges[best.id][level] {
				if !visited.Visit(neighbor) {
					continue // Already visited.
				}
				if h.deleted[neighbor] {
					continue
				}

				var d float32
				if queryNode >= 0 {
					d = distFunc(queryNode, int(neighbor))
				} else {
					d = distFunc(0, int(neighbor)) // search mode: distFunc ignores first arg
				}

				if len(results) < ef || d < results[len(results)-1].dist {
					// Insert into results (sorted).
					results = insertSorted(results, candidate{id: neighbor, dist: d})
					if len(results) > ef {
						results = results[:ef]
					}

					// Insert into candidates (sorted).
					cands = insertSorted(cands, candidate{id: neighbor, dist: d})
				}
			}
		}
	}

	return results
}

// selectNeighborsDiversity implements the diversity heuristic:
// Accept a candidate only if it's closer to the base node than to
// any already-accepted neighbor. This prevents clustering.
func (h *hnswIndex) selectNeighborsDiversity(baseNode int, candidates []candidate, maxEdges int, distFunc func(a, b int) float32) []uint32 {
	// Sort candidates by distance.
	sortCandidates(candidates)

	selected := make([]uint32, 0, maxEdges)
	for _, c := range candidates {
		if len(selected) >= maxEdges {
			break
		}
		if int(c.id) == baseNode {
			continue
		}

		// Check diversity: is this candidate closer to base than to any selected?
		keep := true
		for _, s := range selected {
			distToSelected := distFunc(int(c.id), int(s))
			if distToSelected < c.dist {
				keep = false
				break
			}
		}
		if keep {
			selected = append(selected, c.id)
		}
	}

	// If diversity was too aggressive, fill with nearest remaining.
	if len(selected) < maxEdges {
		for _, c := range candidates {
			if len(selected) >= maxEdges {
				break
			}
			if int(c.id) == baseNode {
				continue
			}
			found := false
			for _, s := range selected {
				if s == c.id {
					found = true
					break
				}
			}
			if !found {
				selected = append(selected, c.id)
			}
		}
	}

	return selected
}

// insertSorted inserts a candidate into a sorted slice (ascending by dist).
func insertSorted(s []candidate, c candidate) []candidate {
	i := 0
	for i < len(s) && s[i].dist < c.dist {
		i++
	}
	s = append(s, candidate{})
	copy(s[i+1:], s[i:])
	s[i] = c
	return s
}

// sortCandidates sorts candidates by distance ascending.
func sortCandidates(c []candidate) {
	// Simple insertion sort (candidates lists are small).
	for i := 1; i < len(c); i++ {
		key := c[i]
		j := i - 1
		for j >= 0 && c[j].dist > key.dist {
			c[j+1] = c[j]
			j--
		}
		c[j+1] = key
	}
}

// --- Visited set with O(1) reset ---

type visitedSet struct {
	data    []byte
	version byte
}

func newVisitedSet(size int) *visitedSet {
	return &visitedSet{
		data:    make([]byte, size),
		version: 1,
	}
}

// Visit marks a node as visited. Returns true if it was NOT already visited.
func (v *visitedSet) Visit(id uint32) bool {
	if int(id) >= len(v.data) {
		return true // Out of bounds: treat as unvisited, don't track.
	}
	if v.data[id] == v.version {
		return false // Already visited.
	}
	v.data[id] = v.version
	return true
}

// Reset clears all visited markers in O(1).
func (v *visitedSet) Reset() {
	v.version++
	if v.version == 0 {
		// Overflow: zero the array (happens every 255 resets).
		clear(v.data)
		v.version = 1
	}
}

package store

import (
	"encoding/binary"
	"fmt"
	"math"
)

// HNSW graph binary format:
//
//   [GraphHeader: 16 bytes]
//     magic:       [4]byte "HNSW"
//     numNodes:    uint32
//     entryNode:   int32 (-1 if empty)
//     maxLevel:    uint8
//     M:           uint8
//     efConst:     uint16
//
//   [Per-node: variable length]
//     numLevels:   uint8 (0 if deleted/absent)
//     For each level:
//       numEdges:  uint16
//       edges:     numEdges * uint32

const hnswHeaderSize = 16

// MarshalHNSW serializes the HNSW graph to bytes.
func (h *hnswIndex) MarshalHNSW() []byte {
	h.mu.RLock()
	defer h.mu.RUnlock()

	// Calculate size.
	size := hnswHeaderSize
	for _, nodeLevels := range h.edges {
		size++ // numLevels byte
		if nodeLevels == nil {
			continue
		}
		for _, edges := range nodeLevels {
			size += 2 + len(edges)*4 // uint16 count + uint32 per edge
		}
	}

	buf := make([]byte, size)

	// Header.
	copy(buf[0:4], "HNSW")
	binary.LittleEndian.PutUint32(buf[4:8], uint32(len(h.edges)))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(int32(h.entryNode)))
	buf[12] = uint8(h.maxLevel)
	buf[13] = uint8(h.M)
	binary.LittleEndian.PutUint16(buf[14:16], uint16(h.efConst))

	// Per-node data.
	off := hnswHeaderSize
	for _, nodeLevels := range h.edges {
		if nodeLevels == nil {
			buf[off] = 0
			off++
			continue
		}
		buf[off] = uint8(len(nodeLevels))
		off++
		for _, edges := range nodeLevels {
			binary.LittleEndian.PutUint16(buf[off:off+2], uint16(len(edges)))
			off += 2
			for _, e := range edges {
				binary.LittleEndian.PutUint32(buf[off:off+4], e)
				off += 4
			}
		}
	}

	return buf
}

// UnmarshalHNSW deserializes an HNSW graph from bytes.
func UnmarshalHNSW(data []byte) (*hnswIndex, error) {
	if len(data) < hnswHeaderSize {
		return nil, fmt.Errorf("hnsw: data too short (%d bytes)", len(data))
	}
	if string(data[0:4]) != "HNSW" {
		return nil, fmt.Errorf("hnsw: invalid magic %q", data[0:4])
	}

	numNodes := int(binary.LittleEndian.Uint32(data[4:8]))
	entryNode := int(int32(binary.LittleEndian.Uint32(data[8:12])))
	maxLevel := int(data[12])
	M := int(data[13])
	efConst := int(binary.LittleEndian.Uint16(data[14:16]))

	h := &hnswIndex{
		M:         M,
		Mmax0:     M * 2,
		efConst:   efConst,
		levelMul:  1.0 / math.Log(float64(M)),
		maxLevel:  maxLevel,
		entryNode: entryNode,
		edges:     make([][]uint32Slice, numNodes),
		deleted:   make([]bool, numNodes),
		count:     numNodes,
	}

	off := hnswHeaderSize
	for i := range numNodes {
		if off >= len(data) {
			return nil, fmt.Errorf("hnsw: truncated at node %d", i)
		}
		numLevels := int(data[off])
		off++
		if numLevels == 0 {
			h.deleted[i] = true
			h.count--
			continue
		}
		h.edges[i] = make([]uint32Slice, numLevels)
		for lv := range numLevels {
			if off+2 > len(data) {
				return nil, fmt.Errorf("hnsw: truncated at node %d level %d", i, lv)
			}
			numEdges := int(binary.LittleEndian.Uint16(data[off : off+2]))
			off += 2
			if off+numEdges*4 > len(data) {
				return nil, fmt.Errorf("hnsw: truncated edges at node %d level %d", i, lv)
			}
			edges := make([]uint32, numEdges)
			for e := range numEdges {
				edges[e] = binary.LittleEndian.Uint32(data[off : off+4])
				off += 4
			}
			h.edges[i][lv] = edges
		}
	}

	return h, nil
}


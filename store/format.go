package store

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
)

// File format constants.
const (
	fileMagic      = "TQDB"
	fileVersion    = uint8(1)
	fileHeaderSize = 64
)

// .tq file layout:
//
//	[Header 64B]
//	[Indices: N × packedRowSize bytes, bit-packed per header.Bits]
//	[Norms: N × float32 LE]
//	[IDs: length-prefixed strings]
//	[Data: length-prefixed JSON blobs (map[string]any)]
//	[Contents: length-prefixed strings]
//
// Indices are bit-packed: 4-bit stores 2 indices per byte, 8-bit stores 1 per byte.
// packedRowSize = codec.PackedSize(workDim, bits).
// All offsets in the header are absolute byte positions.

// fileHeader is the in-memory representation of the .tq file header.
type fileHeader struct {
	Dim         uint16
	WorkDim     uint16
	Bits        uint8
	Rotation    uint8
	UseExact    uint8
	Seed        uint64
	NumVecs     uint32
	NormsOff    uint32
	IDsOff      uint32
	DataOff     uint32
	ContentsOff uint32
	GraphOff    uint32 // 0 = no HNSW graph stored
}

// encodeHeader writes the 64-byte header to dst.
func encodeHeader(dst []byte, h *fileHeader) {
	_ = dst[fileHeaderSize-1] // BCE
	copy(dst[0:4], fileMagic)
	dst[4] = fileVersion
	binary.LittleEndian.PutUint16(dst[5:7], h.Dim)
	binary.LittleEndian.PutUint16(dst[7:9], h.WorkDim)
	dst[9] = h.Bits
	dst[10] = h.Rotation
	dst[11] = h.UseExact
	binary.LittleEndian.PutUint64(dst[12:20], h.Seed)
	binary.LittleEndian.PutUint32(dst[20:24], h.NumVecs)
	binary.LittleEndian.PutUint32(dst[24:28], h.NormsOff)
	binary.LittleEndian.PutUint32(dst[28:32], h.IDsOff)
	binary.LittleEndian.PutUint32(dst[32:36], h.DataOff)
	binary.LittleEndian.PutUint32(dst[36:40], h.ContentsOff)
	binary.LittleEndian.PutUint32(dst[40:44], h.GraphOff)
	// bytes 44-63: reserved
}

// decodeHeader reads and validates the 64-byte header.
func decodeHeader(src []byte) (fileHeader, error) {
	if len(src) < fileHeaderSize {
		return fileHeader{}, fmt.Errorf("tqdb: file too small (%d bytes, need %d)", len(src), fileHeaderSize)
	}
	if string(src[0:4]) != fileMagic {
		return fileHeader{}, fmt.Errorf("tqdb: invalid magic %q (expected %q)", src[0:4], fileMagic)
	}
	if src[4] != fileVersion {
		return fileHeader{}, fmt.Errorf("tqdb: unsupported version %d (expected %d)", src[4], fileVersion)
	}
	h := fileHeader{
		Dim:         binary.LittleEndian.Uint16(src[5:7]),
		WorkDim:     binary.LittleEndian.Uint16(src[7:9]),
		Bits:        src[9],
		Rotation:    src[10],
		UseExact:    src[11],
		Seed:        binary.LittleEndian.Uint64(src[12:20]),
		NumVecs:     binary.LittleEndian.Uint32(src[20:24]),
		NormsOff:    binary.LittleEndian.Uint32(src[24:28]),
		IDsOff:      binary.LittleEndian.Uint32(src[28:32]),
		DataOff:     binary.LittleEndian.Uint32(src[32:36]),
		ContentsOff: binary.LittleEndian.Uint32(src[36:40]),
		GraphOff:    binary.LittleEndian.Uint32(src[40:44]),
	}
	if h.WorkDim < h.Dim {
		return fileHeader{}, fmt.Errorf("tqdb: workDim %d < dim %d", h.WorkDim, h.Dim)
	}
	return h, nil
}

// encodeFile serializes the complete .tq file into a byte slice.
func encodeFile(cfg tqdb.StoreConfig, workDim int, buf *writeBuffer) []byte {
	numVecs := len(buf.norms)
	wdim := workDim
	bits := cfg.Bits
	if bits <= 0 || bits > 8 {
		bits = 8
	}

	// Compute section offsets.
	// Indices are bit-packed: packedRowSize bytes per vector.
	packedRowSize := codec.PackedSize(wdim, bits)
	indicesSize := numVecs * packedRowSize
	normsSize := numVecs * 4

	normsOff := uint32(fileHeaderSize + indicesSize)
	idsOff := normsOff + uint32(normsSize)

	// IDs section size.
	idsSize := 0
	for _, id := range buf.ids {
		idsSize += 2 + len(id)
	}
	dataOff := idsOff + uint32(idsSize)

	// Data section size (JSON blobs).
	dataSize := 0
	for _, m := range buf.data {
		dataSize += 4
		if m != nil {
			dataSize += len(m)
		}
	}
	contentsOff := dataOff + uint32(dataSize)

	// Contents section size.
	contentsSize := 0
	for _, c := range buf.contents {
		contentsSize += 4 + len(c)
	}

	// Graph section (optional, after contents).
	graphOff := uint32(0)
	graphSize := len(buf.graphData)
	if graphSize > 0 {
		graphOff = contentsOff + uint32(contentsSize)
	}

	totalSize := int(contentsOff) + contentsSize + graphSize
	data := make([]byte, totalSize)

	// Header.
	useExact := uint8(0)
	if cfg.UseExactPDF {
		useExact = 1
	}
	hdr := &fileHeader{
		Dim:         uint16(cfg.Dim),
		WorkDim:     uint16(wdim),
		Bits:        uint8(cfg.Bits),
		Rotation:    uint8(cfg.Rotation),
		UseExact:    useExact,
		Seed:        cfg.Seed,
		NumVecs:     uint32(numVecs),
		NormsOff:    normsOff,
		IDsOff:      idsOff,
		DataOff:     dataOff,
		ContentsOff: contentsOff,
		GraphOff:    graphOff,
	}
	encodeHeader(data[:fileHeaderSize], hdr)

	// Indices section (bit-packed).
	if bits == 8 {
		// No packing needed, direct copy.
		copy(data[fileHeaderSize:], buf.allIndices)
	} else {
		off := fileHeaderSize
		for i := range numVecs {
			row := buf.allIndices[i*wdim : i*wdim+wdim]
			codec.PackIndicesTo(data[off:off+packedRowSize], row, bits)
			off += packedRowSize
		}
	}

	// Norms section.
	off := int(normsOff)
	for _, n := range buf.norms {
		binary.LittleEndian.PutUint32(data[off:off+4], math.Float32bits(n))
		off += 4
	}

	// IDs section.
	off = int(idsOff)
	for _, id := range buf.ids {
		binary.LittleEndian.PutUint16(data[off:off+2], uint16(len(id)))
		copy(data[off+2:], id)
		off += 2 + len(id)
	}

	// Data section (JSON blobs).
	off = int(dataOff)
	for _, m := range buf.data {
		if m == nil {
			binary.LittleEndian.PutUint32(data[off:off+4], 0)
			off += 4
		} else {
			binary.LittleEndian.PutUint32(data[off:off+4], uint32(len(m)))
			copy(data[off+4:], m)
			off += 4 + len(m)
		}
	}

	// Contents section.
	off = int(contentsOff)
	for _, c := range buf.contents {
		binary.LittleEndian.PutUint32(data[off:off+4], uint32(len(c)))
		copy(data[off+4:], c)
		off += 4 + len(c)
	}

	// Graph section (optional).
	if graphSize > 0 {
		copy(data[graphOff:], buf.graphData)
	}

	return data
}

// decodeFloat32s reads n float32 values from a little-endian byte slice.
func decodeFloat32s(src []byte, n int) []float32 {
	dst := make([]float32, n)
	for i := range n {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4 : i*4+4]))
	}
	return dst
}

// decodeIDs reads length-prefixed string IDs from a byte slice.
func decodeIDs(src []byte, n int) []string {
	ids := make([]string, n)
	off := 0
	for i := range n {
		idLen := int(binary.LittleEndian.Uint16(src[off : off+2]))
		ids[i] = string(src[off+2 : off+2+idLen])
		off += 2 + idLen
	}
	return ids
}

// decodeDataRaw reads length-prefixed JSON blobs from a byte slice.
func decodeDataRaw(src []byte, n int) [][]byte {
	meta := make([][]byte, n)
	off := 0
	for i := range n {
		metaLen := int(binary.LittleEndian.Uint32(src[off : off+4]))
		if metaLen > 0 {
			meta[i] = src[off+4 : off+4+metaLen]
		}
		off += 4 + metaLen
	}
	return meta
}

// decodeContents reads length-prefixed strings from a byte slice.
func decodeContents(src []byte, n int) []string {
	contents := make([]string, n)
	off := 0
	for i := range n {
		cLen := int(binary.LittleEndian.Uint32(src[off : off+4]))
		if cLen > 0 {
			contents[i] = string(src[off+4 : off+4+cLen])
		}
		off += 4 + cLen
	}
	return contents
}

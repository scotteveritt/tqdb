package store

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/scotteveritt/tqdb"
)

// File format constants.
const (
	fileMagic      = "TQDB"
	fileVersion    = uint8(1)
	fileHeaderSize = 64
)

// fileHeader is the in-memory representation of the .tq file header.
type fileHeader struct {
	Dim       uint16
	WorkDim   uint16
	Bits      uint8
	Rotation  uint8
	UseExact  uint8
	Seed      uint64
	NumVecs   uint32
	NormsOff  uint32
	CNormsOff uint32
	IDsOff    uint32
	MetaOff   uint32
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
	binary.LittleEndian.PutUint32(dst[28:32], h.CNormsOff)
	binary.LittleEndian.PutUint32(dst[32:36], h.IDsOff)
	binary.LittleEndian.PutUint32(dst[36:40], h.MetaOff)
	// bytes 40-63: reserved (leave zeroed)
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
		Dim:       binary.LittleEndian.Uint16(src[5:7]),
		WorkDim:   binary.LittleEndian.Uint16(src[7:9]),
		Bits:      src[9],
		Rotation:  src[10],
		UseExact:  src[11],
		Seed:      binary.LittleEndian.Uint64(src[12:20]),
		NumVecs:   binary.LittleEndian.Uint32(src[20:24]),
		NormsOff:  binary.LittleEndian.Uint32(src[24:28]),
		CNormsOff: binary.LittleEndian.Uint32(src[28:32]),
		IDsOff:    binary.LittleEndian.Uint32(src[32:36]),
		MetaOff:   binary.LittleEndian.Uint32(src[36:40]),
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

	// Compute section offsets.
	indicesSize := numVecs * wdim
	normsSize := numVecs * 4

	normsOff := uint32(fileHeaderSize + indicesSize)
	idsOff := normsOff + uint32(normsSize)

	// Compute IDs section size.
	idsSize := 0
	for _, id := range buf.ids {
		idsSize += 2 + len(id) // uint16 len + bytes
	}
	metaOff := idsOff + uint32(idsSize)

	// Compute metadata section size.
	metaSize := 0
	for _, m := range buf.metadata {
		metaSize += 4 // uint32 len
		if m != nil {
			metaSize += len(m) // raw JSON bytes
		}
	}

	totalSize := int(metaOff) + metaSize
	data := make([]byte, totalSize)

	// Header.
	useExact := uint8(0)
	if cfg.UseExactPDF {
		useExact = 1
	}
	hdr := &fileHeader{
		Dim:       uint16(cfg.Dim),
		WorkDim:   uint16(wdim),
		Bits:      uint8(cfg.Bits),
		Rotation:  uint8(cfg.Rotation),
		UseExact:  useExact,
		Seed:      cfg.Seed,
		NumVecs:   uint32(numVecs),
		NormsOff:  normsOff,
		CNormsOff: normsOff, // deprecated: no longer used, kept for format compat
		IDsOff:    idsOff,
		MetaOff:   metaOff,
	}
	encodeHeader(data[:fileHeaderSize], hdr)

	// Indices section.
	copy(data[fileHeaderSize:], buf.allIndices)

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

	// Metadata section.
	off = int(metaOff)
	for _, m := range buf.metadata {
		if m == nil {
			binary.LittleEndian.PutUint32(data[off:off+4], 0)
			off += 4
		} else {
			binary.LittleEndian.PutUint32(data[off:off+4], uint32(len(m)))
			copy(data[off+4:], m)
			off += 4 + len(m)
		}
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

// decodeMetadata reads length-prefixed JSON blobs from a byte slice.
func decodeMetadataRaw(src []byte, n int) [][]byte {
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

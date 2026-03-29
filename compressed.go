package tqdb

import (
	"github.com/scotteveritt/tqdb/internal/codec"
	"encoding/binary"
	"fmt"
	"math"
)

// CompressedVector is the output of TurboQuantMSE.Quantize().
// It stores quantization indices and the original vector norm.
type CompressedVector struct {
	Dim     int     // original dimension
	Bits    int     // quantization bits per coordinate
	Norm    float32 // original L2 norm
	Indices []uint8 // quantization indices (one per coordinate, values 0..2^bits-1)
}

const headerSize = 7 // 2 (dim) + 1 (bits) + 4 (norm)

// MarshalBinary encodes a CompressedVector to a compact binary format.
// Layout: [dim:2][bits:1][norm:4][packed_indices:...]
func (cv *CompressedVector) MarshalBinary() ([]byte, error) {
	if cv.Dim > 65535 {
		return nil, fmt.Errorf("turboquant: dimension %d exceeds uint16 max", cv.Dim)
	}

	pSize := codec.PackedSize(cv.Dim, cv.Bits)
	buf := make([]byte, headerSize+pSize)
	cv.appendTo(buf)
	return buf, nil
}

// AppendBinary appends the binary encoding to dst and returns the extended slice.
// Avoids allocation when dst has sufficient capacity.
func (cv *CompressedVector) AppendBinary(dst []byte) []byte {
	pSize := codec.PackedSize(cv.Dim, cv.Bits)
	need := headerSize + pSize
	dst = grow(dst, need)
	off := len(dst) - need
	cv.appendTo(dst[off:])
	return dst
}

func (cv *CompressedVector) appendTo(buf []byte) {
	binary.LittleEndian.PutUint16(buf[0:2], uint16(cv.Dim))
	buf[2] = uint8(cv.Bits)
	binary.LittleEndian.PutUint32(buf[3:7], math.Float32bits(cv.Norm))
	codec.PackIndicesTo(buf[headerSize:], cv.Indices, cv.Bits)
}

// UnmarshalBinary decodes a CompressedVector from binary format.
func (cv *CompressedVector) UnmarshalBinary(data []byte) error {
	if len(data) < headerSize {
		return fmt.Errorf("turboquant: data too short (%d bytes, need >= %d)", len(data), headerSize)
	}

	dim := int(binary.LittleEndian.Uint16(data[0:2]))
	bits := int(data[2])
	if bits < 1 || bits > 8 {
		return fmt.Errorf("turboquant: invalid bits %d", bits)
	}

	needPacked := codec.PackedSize(dim, bits)
	if len(data)-headerSize < needPacked {
		return fmt.Errorf("turboquant: data too short for %d indices at %d-bit (%d bytes, need %d)",
			dim, bits, len(data)-headerSize, needPacked)
	}

	cv.Dim = dim
	cv.Bits = bits
	cv.Norm = math.Float32frombits(binary.LittleEndian.Uint32(data[3:7]))
	cv.Indices = codec.UnpackIndices(data[headerSize:headerSize+needPacked], dim, bits)
	return nil
}

// Size returns the serialized size in bytes.
func (cv *CompressedVector) Size() int {
	return headerSize + codec.PackedSize(cv.Dim, cv.Bits)
}

// CompressedProdVector adds QJL data for unbiased inner product estimation.
type CompressedProdVector struct {
	CompressedVector
	Signs        []int8  // QJL sign bits, length = QJL projection dimension
	ResidualNorm float32 // ‖residual‖₂
}

// grow appends n zero bytes to dst and returns the extended slice.
func grow(dst []byte, n int) []byte {
	if cap(dst)-len(dst) >= n {
		return dst[:len(dst)+n]
	}
	buf := make([]byte, len(dst)+n)
	copy(buf, dst)
	return buf
}

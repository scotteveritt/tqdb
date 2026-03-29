package tqdb

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"unsafe"
)

// SafeTensorsFile provides read access to a SafeTensors (.safetensors) file.
// The file is memory-mapped for zero-copy access to tensor data.
type SafeTensorsFile struct {
	data      []byte       // mmap'd or loaded file data
	release   func() error // Unmap or no-op
	dataStart int          // byte offset where tensor data begins
	tensors   map[string]TensorInfo
	metadata  map[string]string
}

// TensorInfo describes a single tensor in a SafeTensors file.
type TensorInfo struct {
	Name        string
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// OpenSafeTensors opens a SafeTensors file for reading.
func OpenSafeTensors(path string) (*SafeTensorsFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tqdb: open safetensors: %w", err)
	}
	defer f.Close() //nolint:errcheck

	data, release, err := mapFile(f)
	if err != nil {
		return nil, fmt.Errorf("tqdb: mmap safetensors: %w", err)
	}

	if len(data) < 8 {
		_ = release()
		return nil, fmt.Errorf("tqdb: safetensors file too small")
	}

	// Read 8-byte header length.
	headerLen := binary.LittleEndian.Uint64(data[0:8])
	if headerLen > 100_000_000 {
		_ = release()
		return nil, fmt.Errorf("tqdb: safetensors header too large (%d bytes)", headerLen)
	}

	headerEnd := 8 + int(headerLen)
	if headerEnd > len(data) {
		_ = release()
		return nil, fmt.Errorf("tqdb: safetensors file truncated (header says %d, file has %d)", headerEnd, len(data))
	}

	// Parse JSON header.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data[8:headerEnd], &raw); err != nil {
		_ = release()
		return nil, fmt.Errorf("tqdb: parse safetensors header: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	var metadata map[string]string

	for k, v := range raw {
		if k == "__metadata__" {
			_ = json.Unmarshal(v, &metadata)
			continue
		}
		var ti TensorInfo
		if err := json.Unmarshal(v, &ti); err != nil {
			_ = release()
			return nil, fmt.Errorf("tqdb: parse tensor %q: %w", k, err)
		}
		ti.Name = k
		tensors[k] = ti
	}

	return &SafeTensorsFile{
		data:      data,
		release:   release,
		dataStart: headerEnd,
		tensors:   tensors,
		metadata:  metadata,
	}, nil
}

// Close releases the mmap.
func (sf *SafeTensorsFile) Close() error {
	if sf.release != nil {
		return sf.release()
	}
	return nil
}

// TensorNames returns all tensor names, sorted alphabetically.
func (sf *SafeTensorsFile) TensorNames() []string {
	names := make([]string, 0, len(sf.tensors))
	for k := range sf.tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Tensor returns info about a named tensor.
func (sf *SafeTensorsFile) Tensor(name string) (TensorInfo, bool) {
	ti, ok := sf.tensors[name]
	return ti, ok
}

// NumTensors returns the total number of tensors.
func (sf *SafeTensorsFile) NumTensors() int {
	return len(sf.tensors)
}

// Metadata returns the __metadata__ map (may be nil).
func (sf *SafeTensorsFile) Metadata() map[string]string {
	return sf.metadata
}

// TensorBytes returns the raw bytes for a tensor.
func (sf *SafeTensorsFile) TensorBytes(name string) ([]byte, error) {
	ti, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tqdb: tensor %q not found", name)
	}
	start := sf.dataStart + int(ti.DataOffsets[0])
	end := sf.dataStart + int(ti.DataOffsets[1])
	if start < 0 || end > len(sf.data) || start > end {
		return nil, fmt.Errorf("tqdb: tensor %q: invalid offsets [%d, %d]", name, start, end)
	}
	return sf.data[start:end], nil
}

// NumElements returns the total number of elements in a tensor.
func (ti TensorInfo) NumElements() int64 {
	if len(ti.Shape) == 0 {
		return 0
	}
	n := int64(1)
	for _, s := range ti.Shape {
		n *= s
	}
	return n
}

// BytesPerElement returns the size of one element for the tensor's dtype.
func (ti TensorInfo) BytesPerElement() int {
	switch ti.DType {
	case "F64", "I64", "U64", "C64":
		return 8
	case "F32", "I32", "U32":
		return 4
	case "F16", "BF16", "I16", "U16":
		return 2
	case "I8", "U8", "BOOL", "F8_E5M2", "F8_E4M3", "F8_E8M0":
		return 1
	default:
		return 0
	}
}

// Rows returns the number of rows (first dimension) for a 2D tensor.
// Returns 1 for 1D tensors, 0 for empty tensors.
func (ti TensorInfo) Rows() int {
	if len(ti.Shape) == 0 {
		return 0
	}
	if len(ti.Shape) == 1 {
		return 1
	}
	return int(ti.Shape[0])
}

// Cols returns the row dimension (last dimension) for a tensor.
func (ti TensorInfo) Cols() int {
	if len(ti.Shape) == 0 {
		return 0
	}
	return int(ti.Shape[len(ti.Shape)-1])
}

// ReadRowFloat64 reads a single row of a tensor as float64.
// For 2D tensors, row is the first-dimension index.
// Supports F32, F16, BF16 dtypes.
func (sf *SafeTensorsFile) ReadRowFloat64(name string, row int) ([]float64, error) {
	ti, ok := sf.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tqdb: tensor %q not found", name)
	}

	cols := ti.Cols()
	bpe := ti.BytesPerElement()
	if bpe == 0 {
		return nil, fmt.Errorf("tqdb: unsupported dtype %q", ti.DType)
	}

	rowBytes := cols * bpe
	offset := sf.dataStart + int(ti.DataOffsets[0]) + row*rowBytes
	if offset+rowBytes > len(sf.data) {
		return nil, fmt.Errorf("tqdb: tensor %q row %d out of bounds", name, row)
	}

	raw := sf.data[offset : offset+rowBytes]
	out := make([]float64, cols)

	switch ti.DType {
	case "F32":
		for i := range cols {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:])))
		}
	case "F16":
		for i := range cols {
			out[i] = float64(f16ToF32(binary.LittleEndian.Uint16(raw[i*2:])))
		}
	case "BF16":
		for i := range cols {
			out[i] = float64(bf16ToF32(binary.LittleEndian.Uint16(raw[i*2:])))
		}
	case "F64":
		for i := range cols {
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		}
	default:
		return nil, fmt.Errorf("tqdb: unsupported dtype %q for float conversion", ti.DType)
	}

	return out, nil
}

// bf16ToF32 converts a BF16 value to float32.
// BF16 is the top 16 bits of a float32 — just shift left by 16.
func bf16ToF32(b uint16) float32 {
	bits := uint32(b) << 16
	return *(*float32)(unsafe.Pointer(&bits))
}

// f16ToF32 converts an IEEE 754 half-precision float to float32.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	switch exp {
	case 0:
		if mant == 0 {
			// Zero
			return math.Float32frombits(sign << 31)
		}
		// Subnormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	case 31:
		// Inf or NaN
		return math.Float32frombits((sign << 31) | (0xFF << 23) | (mant << 13))
	default:
		// Normal
		return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
	}
}

package tqdb

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"unsafe"

	"github.com/scotteveritt/tqdb/internal/codec"
)

// RotationType selects the orthogonal rotation algorithm.
type RotationType int

const (
	// RotationQR uses a Haar-random orthogonal matrix via QR decomposition.
	// Memory: O(d²). Compute: O(d²) per vector. The paper's original approach.
	RotationQR RotationType = iota

	// RotationHadamard uses the Randomized Walsh-Hadamard Transform (D₂·H̃·D₁).
	// Memory: O(d). Compute: O(d log d) per vector.
	// Empirically better quality than QR (QuaRot, ICLR 2024).
	RotationHadamard
)

// Config controls quantizer behavior.
type Config struct {
	Dim         int          // embedding dimension (required)
	Bits        int          // bits per coordinate: 1-8 (default: 4)
	Seed        uint64       // rotation matrix seed (default: 42)
	Rotation    RotationType // rotation algorithm (default: RotationQR)
	UseExactPDF bool         // use exact Beta PDF for codebook (default: false, uses Gaussian approx)
}

// Validate checks that the Config fields are within valid ranges.
func (c Config) Validate() error {
	if c.Dim < 2 {
		return fmt.Errorf("turboquant: Dim must be >= 2, got %d", c.Dim)
	}
	if c.Bits < 1 || c.Bits > 8 {
		return fmt.Errorf("turboquant: Bits must be 1-8, got %d", c.Bits)
	}
	return nil
}

// WithDefaults returns a copy with zero fields filled to defaults.
func (c Config) WithDefaults() Config {
	if c.Bits == 0 {
		c.Bits = 4
	}
	if c.Seed == 0 {
		c.Seed = 42
	}
	return c
}

// Result represents a search result from a Collection or Store.
type Result struct {
	ID      string
	Score   float64 // inner product similarity (≈ cosine sim for unit-normalized vectors)
	Content string
	Data    map[string]any
}

// SearchOptions controls vector similarity search (VS2: SearchDataObjectsRequest).
type SearchOptions struct {
	TopK     int     // max results (default: 10)
	MinScore float64 // minimum similarity threshold (0 = no filter)
	Offset   int     // skip first N results (pagination)
	Filter   Filter  // data field filter
	Rescore  int     // rescore top-N with exact dequantized distance (0 = disabled, recommended: 3×TopK)
	Ef       int     // HNSW: search beam width (0 = auto, higher = better recall, slower)
}

// QueryOptions controls filter-only retrieval (VS2: QueryDataObjectsRequest).
type QueryOptions struct {
	PageSize int    // max results (default: 100)
	Filter   Filter // required
}

// IndexType selects the ANN index algorithm.
type IndexType int

const (
	IndexAuto IndexType = iota // Auto-select based on N and d (default)
	IndexIVF                   // ScaNN-style IVF partitioning
	IndexHNSW                  // Hierarchical Navigable Small World graph
	IndexNone                  // No ANN index (brute-force only, filter indexes still built)
)

// IndexConfig controls index creation (VS2-aligned).
type IndexConfig struct {
	FilterFields  []string  // fields to build inverted indexes on
	Type          IndexType // index algorithm: IndexIVF (default) or IndexHNSW
	NumPartitions int       // IVF: partitions (0 = auto √N)
	NProbe        int       // IVF: partitions to search per query (0 = auto √NumPartitions)
	SkipIVF       bool      // IVF: if true, only build filter indexes
	M             int       // HNSW: max edges per layer (default 16)
	EfConstruction int      // HNSW: build-time beam width (default 200)
}

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

// StoreConfig controls store creation.
type StoreConfig struct {
	Dim         int          // embedding dimension (required)
	Bits        int          // bits per coordinate: 1-8 (default: 4)
	Rotation    RotationType // rotation algorithm (default: RotationQR)
	Seed        uint64       // rotation matrix seed (default: 42)
	UseExactPDF bool         // use exact Beta PDF for codebook
}

// ToConfig converts StoreConfig to a quantizer Config.
func (c StoreConfig) ToConfig() Config {
	return Config{
		Dim:         c.Dim,
		Bits:        c.Bits,
		Rotation:    c.Rotation,
		Seed:        c.Seed,
		UseExactPDF: c.UseExactPDF,
	}
}

// WithDefaults returns a copy with zero fields filled to defaults.
func (c StoreConfig) WithDefaults() StoreConfig {
	if c.Bits == 0 {
		c.Bits = 4
	}
	if c.Seed == 0 {
		c.Seed = 42
	}
	return c
}

// StoreInfo contains statistics about a store.
type StoreInfo struct {
	Path        string
	Dim         int
	WorkDim     int
	Bits        int
	Rotation    RotationType
	Seed        uint64
	NumVecs     int
	FileSize    int64
	IndexBytes  int64
	Compression float64 // ratio vs float32
}

// KVCacheConfig controls KV cache creation.
type KVCacheConfig struct {
	Layers      int          // number of transformer layers
	Heads       int          // number of attention heads
	HeadDim     int          // dimension per head (typically 64 or 128)
	Bits        int          // quantization bits per coordinate (default: 4)
	PackIndices bool         // bit-pack indices in storage (default: false)
	Rotation    RotationType // rotation algorithm (default: RotationHadamard)
	Seed        uint64       // rotation seed (default: 42)
}

// ModelConfig controls model weight quantization.
type ModelConfig struct {
	Bits     int          // bits per coordinate (default: 4)
	Rotation RotationType // rotation algorithm (default: RotationHadamard)
	Seed     uint64       // rotation seed (default: 42)
	Workers  int          // concurrent workers per tensor (default: GOMAXPROCS)
}

// TQModelHeader is the JSON header of a .tqm file.
type TQModelHeader struct {
	Version  int                      `json:"version"`
	Source   string                   `json:"source,omitempty"`
	Bits     int                      `json:"bits"`
	Packed   bool                     `json:"packed"`
	Rotation string                   `json:"rotation"`
	Seed     uint64                   `json:"seed"`
	Tensors  map[string]TQMTensorInfo `json:"tensors"`
}

// TQMTensorInfo describes a quantized tensor in a .tqm file.
type TQMTensorInfo struct {
	Shape     []int64 `json:"shape"`
	OrigDType string  `json:"orig_dtype"`
	Rows      int     `json:"rows"`
	RowDim    int     `json:"row_dim"`
	WorkDim   int     `json:"work_dim"`
	Offset    int64   `json:"offset"`
	Size      int64   `json:"size"`
	AvgCosSim float64 `json:"avg_cos_sim"`
}

// CompressModelProgress reports compression progress.
// Called from the main goroutine with the current tensor and row counts.
type CompressModelProgress func(tensorName string, tensorIdx, totalTensors int, rows, totalRows int)

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

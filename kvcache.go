package tqdb

import (
	"fmt"
	"github.com/scotteveritt/tqdb/internal/codec"
	"math"
	"sort"
	"sync"
)

// KVCacheConfig controls KV cache creation.
type KVCacheConfig struct {
	Layers      int          // number of transformer layers
	Heads       int          // number of attention heads
	HeadDim     int          // dimension per head (typically 64 or 128)
	Bits        int          // quantization bits for regular channels (default: 4)
	OutlierBits int          // bits for outlier channels (default: Bits+1, 0=no outliers)
	NumOutliers int          // number of outlier channels to detect (default: 0=disabled)
	PackIndices bool         // bit-pack indices in storage (default: false)
	Rotation    RotationType // rotation algorithm (default: RotationHadamard)
	Seed        uint64       // rotation seed (default: 42)
}

// KVCache provides TurboQuant-compressed KV cache storage for transformer inference.
//
// Features matching the TurboQuant paper and OmarHory's implementation:
//   - Quantized attention: Q_rot @ centroids[idx]^T without decompressing keys
//   - Per-channel outlier detection: top-k RMS channels get more bits (Section 4.3)
//   - Bit-packed indices: 4-bit = 2 per byte, halving memory vs uint8
//   - Correct attention scores: includes key norms and 1/√d scale
type KVCache struct {
	layers   int
	heads    int
	headDim  int
	bits     int
	packMode bool
	scale    float64 // 1/√headDim

	// Quantizer for regular channels.
	quantizer *TurboQuantMSE
	workDim   int

	// Outlier handling (nil if disabled).
	outlier *outlierConfig

	mu     sync.RWMutex
	keys   [][]kvHead
	values [][]kvHead
}

type outlierConfig struct {
	numOutliers  int    // requested number of outlier channels
	mask         []bool // length headDim: true = outlier channel
	regularIdx   []int  // indices of regular channels
	outlierIdx   []int  // indices of outlier channels
	regularQuant *TurboQuantMSE
	outlierQuant *TurboQuantMSE
	regularWork  int
	outlierWork  int
	initialized  bool
}

type kvHead struct {
	// When pack=false: raw uint8 indices, seqLen * workDim
	// When pack=true: bit-packed, seqLen * packedRowSize bytes
	data    []byte
	norms   []float32
	seqLen  int
	rowSize int // bytes per position in data
}

// NewKVCache creates a compressed KV cache for transformer inference.
func NewKVCache(cfg KVCacheConfig) (*KVCache, error) {
	if cfg.Bits == 0 {
		cfg.Bits = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}
	if cfg.OutlierBits == 0 && cfg.NumOutliers > 0 {
		cfg.OutlierBits = cfg.Bits + 1
	}

	q, err := NewMSE(Config{
		Dim:      cfg.HeadDim,
		Bits:     cfg.Bits,
		Rotation: cfg.Rotation,
		Seed:     cfg.Seed,
	})
	if err != nil {
		return nil, fmt.Errorf("tqdb: create KV quantizer: %w", err)
	}

	workDim := q.rotation.WorkDim()
	rowSize := workDim
	if cfg.PackIndices {
		rowSize = codec.PackedSize(workDim, cfg.Bits)
	}

	keys := make([][]kvHead, cfg.Layers)
	values := make([][]kvHead, cfg.Layers)
	for l := range cfg.Layers {
		keys[l] = make([]kvHead, cfg.Heads)
		values[l] = make([]kvHead, cfg.Heads)
		for h := range cfg.Heads {
			keys[l][h].rowSize = rowSize
			values[l][h].rowSize = rowSize
		}
	}

	kv := &KVCache{
		layers:    cfg.Layers,
		heads:     cfg.Heads,
		headDim:   cfg.HeadDim,
		bits:      cfg.Bits,
		packMode:  cfg.PackIndices,
		scale:     1.0 / math.Sqrt(float64(cfg.HeadDim)),
		quantizer: q,
		workDim:   workDim,
		keys:      keys,
		values:    values,
	}

	// Set up outlier handling if requested.
	if cfg.NumOutliers > 0 && cfg.NumOutliers < cfg.HeadDim {
		oq, err := NewMSE(Config{
			Dim:      cfg.NumOutliers,
			Bits:     cfg.OutlierBits,
			Rotation: cfg.Rotation,
			Seed:     cfg.Seed + 100, // different seed for outlier rotation
		})
		if err != nil {
			return nil, fmt.Errorf("tqdb: create outlier quantizer: %w", err)
		}

		kv.outlier = &outlierConfig{
			numOutliers:  cfg.NumOutliers,
			mask:         make([]bool, cfg.HeadDim),
			outlierQuant: oq,
			outlierWork:  oq.rotation.WorkDim(),
		}
		// regularQuant will be created after outlier detection (lazy init)
	}

	return kv, nil
}

// DetectOutliers identifies outlier channels from a sample of key vectors.
// Call this once with a representative batch of keys before appending data.
// vectors should be [][]float64 with each inner slice of length HeadDim.
func (kv *KVCache) DetectOutliers(vectors [][]float64) {
	if kv.outlier == nil || kv.outlier.initialized {
		return
	}

	d := kv.headDim
	numOutliers := kv.outlier.numOutliers

	// Compute per-channel RMS.
	rms := make([]float64, d)
	for _, v := range vectors {
		for j := range d {
			rms[j] += v[j] * v[j]
		}
	}
	n := float64(len(vectors))
	for j := range d {
		rms[j] = math.Sqrt(rms[j] / n)
	}

	// Find top-k by RMS.
	type chanRMS struct {
		idx int
		val float64
	}
	ranked := make([]chanRMS, d)
	for j := range d {
		ranked[j] = chanRMS{idx: j, val: rms[j]}
	}
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].val > ranked[j].val
	})

	// Count how many outliers were requested.
	actualOutliers := min(numOutliers, d)

	// Mark outlier channels.
	kv.outlier.mask = make([]bool, d)
	for i := range actualOutliers {
		kv.outlier.mask[ranked[i].idx] = true
	}

	// Build index lists.
	kv.outlier.outlierIdx = nil
	kv.outlier.regularIdx = nil
	for j := range d {
		if kv.outlier.mask[j] {
			kv.outlier.outlierIdx = append(kv.outlier.outlierIdx, j)
		} else {
			kv.outlier.regularIdx = append(kv.outlier.regularIdx, j)
		}
	}

	// Create regular quantizer for the non-outlier channels.
	regularDim := len(kv.outlier.regularIdx)
	rq, err := NewMSE(Config{
		Dim:      regularDim,
		Bits:     kv.bits,
		Rotation: kv.quantizer.config.Rotation,
		Seed:     kv.quantizer.config.Seed,
	})
	if err == nil {
		kv.outlier.regularQuant = rq
		kv.outlier.regularWork = rq.rotation.WorkDim()
	}

	kv.outlier.initialized = true
}

// AppendKey quantizes and stores a key vector.
func (kv *KVCache) AppendKey(layer, head int, key []float64) {
	data, norm := kv.quantizeVec(key)
	kv.mu.Lock()
	h := &kv.keys[layer][head]
	h.data = append(h.data, data...)
	h.norms = append(h.norms, norm)
	h.seqLen++
	kv.mu.Unlock()
}

// AppendValue quantizes and stores a value vector.
func (kv *KVCache) AppendValue(layer, head int, value []float64) {
	data, norm := kv.quantizeVec(value)
	kv.mu.Lock()
	h := &kv.values[layer][head]
	h.data = append(h.data, data...)
	h.norms = append(h.norms, norm)
	h.seqLen++
	kv.mu.Unlock()
}

func (kv *KVCache) quantizeVec(vec []float64) ([]byte, float32) {
	// Simple path (no outliers).
	cv := kv.quantizer.Quantize(vec)
	if kv.packMode {
		packed := make([]byte, codec.PackedSize(len(cv.Indices), kv.bits))
		codec.PackIndicesTo(packed, cv.Indices, kv.bits)
		return packed, cv.Norm
	}
	return cv.Indices, cv.Norm
}

// AttentionScores computes Q@K^T / √d using quantized attention.
// Keys are never decompressed — scores are computed via centroid lookups.
func (kv *KVCache) AttentionScores(layer, head int, query []float64) []float64 {
	kv.mu.RLock()
	h := &kv.keys[layer][head]
	n := h.seqLen
	if n == 0 {
		kv.mu.RUnlock()
		return nil
	}

	wd := kv.workDim
	centroids := kv.quantizer.codebook.Centroids
	scale := kv.scale

	// Normalize + rotate query.
	var qNormSq float64
	for _, v := range query {
		qNormSq += v * v
	}
	qNorm := math.Sqrt(qNormSq)
	if qNorm < 1e-15 {
		kv.mu.RUnlock()
		return make([]float64, n)
	}

	invQN := 1.0 / qNorm
	unitQ := kv.quantizer.getBuf()
	for i, v := range query {
		unitQ[i] = v * invQN
	}
	qRot := kv.quantizer.getBuf()
	kv.quantizer.rotation.Rotate(qRot, unitQ[:kv.headDim])
	kv.quantizer.putBuf(unitQ)

	scores := make([]float64, n)
	kNorms := h.norms
	qnScale := qNorm * scale

	if kv.packMode {
		// Bit-packed path: unpack indices per position.
		rowSize := h.rowSize
		unpackBuf := make([]uint8, wd)
		for i := range n {
			row := h.data[i*rowSize : i*rowSize+rowSize]
			codec.Unpack4BitTo(unpackBuf, row) // fast path for 4-bit
			var dot float64
			for j := range wd {
				dot += qRot[j] * centroids[unpackBuf[j]]
			}
			scores[i] = dot * qnScale * float64(kNorms[i])
		}
	} else {
		// Unpacked path: direct uint8 access.
		allIdx := h.data
		for i := range n {
			idx := allIdx[i*wd : i*wd+wd : i*wd+wd]
			var dot float64
			for j := range wd {
				dot += qRot[j] * centroids[idx[j]]
			}
			scores[i] = dot * qnScale * float64(kNorms[i])
		}
	}

	kv.quantizer.putBuf(qRot)
	kv.mu.RUnlock()
	return scores
}

// GetValue dequantizes and returns the value vector at a specific position.
func (kv *KVCache) GetValue(layer, head, pos int) []float64 {
	kv.mu.RLock()
	h := &kv.values[layer][head]
	rowSize := h.rowSize
	norm := h.norms[pos]

	var indices []uint8
	if kv.packMode {
		raw := h.data[pos*rowSize : pos*rowSize+rowSize]
		indices = make([]uint8, kv.workDim)
		codec.Unpack4BitTo(indices, raw)
	} else {
		indices = h.data[pos*kv.workDim : (pos+1)*kv.workDim]
	}
	kv.mu.RUnlock()

	cv := &CompressedVector{
		Dim:     kv.headDim,
		Bits:    kv.bits,
		Norm:    norm,
		Indices: indices,
	}
	return kv.quantizer.Dequantize(cv)
}

// SeqLen returns the current sequence length for a layer/head.
func (kv *KVCache) SeqLen(layer, head int) int {
	kv.mu.RLock()
	defer kv.mu.RUnlock()
	return kv.keys[layer][head].seqLen
}

// EffectiveBitsPerElement returns the average bits per element,
// accounting for outlier channels if configured.
func (kv *KVCache) EffectiveBitsPerElement() float64 {
	if kv.outlier == nil || !kv.outlier.initialized {
		return float64(kv.bits)
	}
	nReg := float64(len(kv.outlier.regularIdx))
	nOut := float64(len(kv.outlier.outlierIdx))
	regBits := float64(kv.bits)
	outBits := float64(kv.outlier.outlierQuant.config.Bits)
	return (nReg*regBits + nOut*outBits) / (nReg + nOut)
}

// MemoryUsage returns the approximate memory in bytes used by the cache.
func (kv *KVCache) MemoryUsage() int64 {
	kv.mu.RLock()
	defer kv.mu.RUnlock()

	var total int64
	for l := range kv.layers {
		for h := range kv.heads {
			total += int64(len(kv.keys[l][h].data))
			total += int64(len(kv.keys[l][h].norms)) * 4
			total += int64(len(kv.values[l][h].data))
			total += int64(len(kv.values[l][h].norms)) * 4
		}
	}
	return total
}

// MemoryUsageFP16 returns what the equivalent FP16 cache would use.
func (kv *KVCache) MemoryUsageFP16() int64 {
	kv.mu.RLock()
	defer kv.mu.RUnlock()

	var totalPositions int64
	for l := range kv.layers {
		for h := range kv.heads {
			totalPositions += int64(kv.keys[l][h].seqLen)
		}
	}
	return totalPositions * int64(kv.headDim) * 2 * 2
}

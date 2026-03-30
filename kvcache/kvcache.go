package kvcache

import (
	"fmt"
	"math"
	"sync"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
	"github.com/scotteveritt/tqdb/quantize"
)

// KVCache provides TurboQuant-compressed KV cache storage for transformer inference.
//
// Features matching the TurboQuant paper and OmarHory's implementation:
//   - Quantized attention: Q_rot @ centroids[idx]^T without decompressing keys
//   - Bit-packed indices: 4-bit = 2 per byte, halving memory vs uint8
//   - Correct attention scores: includes key norms and 1/√d scale
//
// Per-channel outlier detection (Section 4.3) is planned but not yet implemented.
type KVCache struct {
	layers   int
	heads    int
	headDim  int
	bits     int
	packMode bool
	scale    float64 // 1/√headDim

	quantizer *quantize.TurboQuantMSE
	workDim   int

	mu     sync.RWMutex
	keys   [][]kvHead
	values [][]kvHead
}

type kvHead struct {
	// When pack=false: raw uint8 indices, seqLen * workDim
	// When pack=true: bit-packed, seqLen * packedRowSize bytes
	data    []byte
	norms   []float32
	seqLen  int
	rowSize int // bytes per position in data
}

// New creates a compressed KV cache for transformer inference.
func New(cfg tqdb.KVCacheConfig) (*KVCache, error) {
	if cfg.Bits == 0 {
		cfg.Bits = 4
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}

	q, err := quantize.NewMSE(tqdb.Config{
		Dim:      cfg.HeadDim,
		Bits:     cfg.Bits,
		Rotation: cfg.Rotation,
		Seed:     cfg.Seed,
	})
	if err != nil {
		return nil, fmt.Errorf("tqdb: create KV quantizer: %w", err)
	}

	workDim := q.Rotation().WorkDim()
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

	return &KVCache{
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
	}, nil
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
	centroids := kv.quantizer.Codebook().Centroids
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
	unitQ := kv.quantizer.GetBuf()
	for i, v := range query {
		unitQ[i] = v * invQN
	}
	qRot := kv.quantizer.GetBuf()
	kv.quantizer.Rotation().Rotate(qRot, unitQ[:kv.headDim])
	kv.quantizer.PutBuf(unitQ)

	scores := make([]float64, n)
	kNorms := h.norms
	qnScale := qNorm * scale

	if kv.packMode {
		// Bit-packed path: unpack indices per position.
		// Buffer hoisted outside loop to avoid per-position allocation.
		rowSize := h.rowSize
		unpackBuf := make([]uint8, wd)
		for i := range n {
			row := h.data[i*rowSize : i*rowSize+rowSize]
			codec.Unpack4BitTo(unpackBuf, row)
			var dot0, dot1 float64
			j := 0
			for ; j <= wd-8; j += 8 {
				dot0 += qRot[j]*centroids[unpackBuf[j]] +
					qRot[j+1]*centroids[unpackBuf[j+1]] +
					qRot[j+2]*centroids[unpackBuf[j+2]] +
					qRot[j+3]*centroids[unpackBuf[j+3]]
				dot1 += qRot[j+4]*centroids[unpackBuf[j+4]] +
					qRot[j+5]*centroids[unpackBuf[j+5]] +
					qRot[j+6]*centroids[unpackBuf[j+6]] +
					qRot[j+7]*centroids[unpackBuf[j+7]]
			}
			for ; j < wd; j++ {
				dot0 += qRot[j] * centroids[unpackBuf[j]]
			}
			scores[i] = (dot0 + dot1) * qnScale * float64(kNorms[i])
		}
	} else {
		// Unpacked path: direct uint8 access with 8-way unroll.
		allIdx := h.data
		for i := range n {
			idx := allIdx[i*wd : i*wd+wd : i*wd+wd]
			var dot0, dot1 float64
			j := 0
			for ; j <= wd-8; j += 8 {
				dot0 += qRot[j]*centroids[idx[j]] +
					qRot[j+1]*centroids[idx[j+1]] +
					qRot[j+2]*centroids[idx[j+2]] +
					qRot[j+3]*centroids[idx[j+3]]
				dot1 += qRot[j+4]*centroids[idx[j+4]] +
					qRot[j+5]*centroids[idx[j+5]] +
					qRot[j+6]*centroids[idx[j+6]] +
					qRot[j+7]*centroids[idx[j+7]]
			}
			for ; j < wd; j++ {
				dot0 += qRot[j] * centroids[idx[j]]
			}
			scores[i] = (dot0 + dot1) * qnScale * float64(kNorms[i])
		}
	}

	kv.quantizer.PutBuf(qRot)
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

	cv := &tqdb.CompressedVector{
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

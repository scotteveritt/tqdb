package quantize

import (
	"fmt"
	"math"
	"sync"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
	"github.com/scotteveritt/tqdb/internal/mathutil"
)

// TurboQuantMSE is the Stage 1 MSE-optimal quantizer.
// It applies random orthogonal rotation followed by per-coordinate
// Lloyd-Max scalar quantization.
//
// Safe for concurrent use after initialization.
type TurboQuantMSE struct {
	config   tqdb.Config
	codebook *codec.Codebook
	rotation codec.Rotator

	// Pool of reusable float64 buffers (length = Dim) to avoid
	// heap allocation on every Quantize/Dequantize call.
	bufPool sync.Pool
}

// NewMSE creates a new TurboQuantMSE quantizer.
func NewMSE(cfg tqdb.Config) (*TurboQuantMSE, error) {
	cfg = cfg.WithDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	var rot codec.Rotator
	switch cfg.Rotation {
	case tqdb.RotationHadamard:
		rot = codec.NewHadamardRotator(cfg.Dim, cfg.Seed)
	default:
		rot = codec.NewRotationMatrix(cfg.Dim, cfg.Seed)
	}

	// codec.Codebook is built for the working dimension (padded for Hadamard).
	workDim := rot.WorkDim()
	cb := lookupPrecomputed(workDim, cfg.Bits)
	if cb == nil {
		cb = codec.SolveCodebook(workDim, cfg.Bits, cfg.UseExactPDF)
	}

	tq := &TurboQuantMSE{
		config:   cfg,
		codebook: cb,
		rotation: rot,
	}
	tq.bufPool = sync.Pool{
		New: func() any {
			buf := make([]float64, workDim)
			return &buf
		},
	}
	return tq, nil
}

// GetBuf returns a pooled float64 buffer of length WorkDim.
func (tq *TurboQuantMSE) GetBuf() []float64 {
	bp := tq.bufPool.Get().(*[]float64) //nolint:errcheck // pool always returns *[]float64
	return *bp
}

// PutBuf returns a buffer to the pool.
func (tq *TurboQuantMSE) PutBuf(buf []float64) {
	tq.bufPool.Put(&buf)
}

// Codebook returns the quantizer's codebook.
func (tq *TurboQuantMSE) Codebook() *codec.Codebook {
	return tq.codebook
}

// Rotation returns the quantizer's rotation matrix.
func (tq *TurboQuantMSE) Rotation() codec.Rotator {
	return tq.rotation
}

// Quantize compresses a float64 vector into a CompressedVector.
// The input vec must have length equal to Config.Dim.
func (tq *TurboQuantMSE) Quantize(vec []float64) *tqdb.CompressedVector {
	d := tq.config.Dim
	if len(vec) != d {
		panic(fmt.Sprintf("turboquant: Quantize: input length %d != Dim %d", len(vec), d))
	}
	workDim := tq.rotation.WorkDim()

	// 1. Normalize into a temp buffer of length d.
	unitBuf := tq.GetBuf() // length workDim, we only use first d
	norm := mathutil.NormalizeTo(unitBuf[:d], vec)

	// 2. Rotate: input d → output workDim (may pad for Hadamard).
	rotated := tq.GetBuf() // length workDim
	tq.rotation.Rotate(rotated, unitBuf[:d])
	tq.PutBuf(unitBuf)

	// 3. Per-coordinate Lloyd-Max quantization over all workDim coordinates.
	indices := make([]uint8, workDim)
	tq.codebook.QuantizeTo(indices, rotated)
	tq.PutBuf(rotated)

	return &tqdb.CompressedVector{
		Dim:     d,
		Bits:    tq.config.Bits,
		Norm:    float32(norm),
		Indices: indices,
	}
}

// Dequantize reconstructs a float64 vector from a CompressedVector.
func (tq *TurboQuantMSE) Dequantize(cv *tqdb.CompressedVector) []float64 {
	d := tq.config.Dim
	recon := make([]float64, d)
	tq.DequantizeTo(recon, cv)
	return recon
}

// DequantizeTo reconstructs a float64 vector into a pre-allocated dst buffer (length Dim).
func (tq *TurboQuantMSE) DequantizeTo(dst []float64, cv *tqdb.CompressedVector) {
	d := tq.config.Dim

	// 1. Look up centroid values in rotated space (workDim coordinates).
	rotBuf := tq.GetBuf() // length workDim
	tq.codebook.DequantizeTo(rotBuf[:len(cv.Indices)], cv.Indices)

	// 2. Un-rotate: workDim → d.
	tq.rotation.Unrotate(dst[:d], rotBuf[:len(cv.Indices)])
	tq.PutBuf(rotBuf)

	// 3. Rescale by original norm.
	norm := float64(cv.Norm)
	for i := range d {
		dst[i] *= norm
	}
}

// QuantizeFloat32 compresses a float32 vector (convenience wrapper).
func (tq *TurboQuantMSE) QuantizeFloat32(vec []float32) *tqdb.CompressedVector {
	return tq.Quantize(mathutil.Float32ToFloat64(vec))
}

// DequantizeFloat32 reconstructs a float32 vector (convenience wrapper).
func (tq *TurboQuantMSE) DequantizeFloat32(cv *tqdb.CompressedVector) []float32 {
	return mathutil.Float64ToFloat32(tq.Dequantize(cv))
}

// CosineSimilarity computes cosine similarity between a raw query and a compressed vector.
// It decompresses the vector internally using a pooled buffer.
func (tq *TurboQuantMSE) CosineSimilarity(query []float64, cv *tqdb.CompressedVector) float64 {
	d := tq.config.Dim
	recon := make([]float64, d)
	tq.DequantizeTo(recon, cv)
	return mathutil.CosineSimilarity(query, recon)
}

// AsymmetricCosineSimilarity computes cosine similarity between a raw query
// and a compressed vector using the rotated-space trick.
//
// Cost: O(d²) for the query rotation + O(d) for the comparison.
// When comparing one query against many vectors, use AsymmetricCosineSimilarityBatch
// which amortizes the rotation.
//
// The result matches what Collection.Search computes internally.
func (tq *TurboQuantMSE) AsymmetricCosineSimilarity(query []float64, cv *tqdb.CompressedVector) float64 {
	centroids := tq.codebook.Centroids
	centroidsSq := tq.codebook.CentroidsSq

	qRot := tq.GetBuf()
	tq.rotation.Rotate(qRot, query)
	queryNorm := mathutil.Norm(query)

	if queryNorm < 1e-15 {
		tq.PutBuf(qRot)
		return 0
	}

	var dot, cnSq float64
	for j, idx := range cv.Indices {
		dot += qRot[j] * centroids[idx]
		cnSq += centroidsSq[idx]
	}
	tq.PutBuf(qRot)

	cn := math.Sqrt(cnSq)
	if cn < 1e-15 {
		return 0
	}
	return dot / (queryNorm * cn)
}

// AsymmetricCosineSimilarityBatch computes cosine similarity between one query
// and multiple compressed vectors. The query rotation is done once (O(d²)),
// then each comparison is O(d). Total: O(d² + N·d).
func (tq *TurboQuantMSE) AsymmetricCosineSimilarityBatch(
	query []float64, cvs []*tqdb.CompressedVector,
) []float64 {
	centroids := tq.codebook.Centroids
	centroidsSq := tq.codebook.CentroidsSq
	workDim := tq.rotation.WorkDim()

	qRot := tq.GetBuf()
	tq.rotation.Rotate(qRot, query)
	queryNorm := mathutil.Norm(query)

	results := make([]float64, len(cvs))
	if queryNorm < 1e-15 {
		tq.PutBuf(qRot)
		return results
	}
	invQN := 1.0 / queryNorm

	for k, cv := range cvs {
		if len(cv.Indices) != workDim {
			continue
		}
		var dot, cnSq float64
		for j, idx := range cv.Indices {
			dot += qRot[j] * centroids[idx]
			cnSq += centroidsSq[idx]
		}
		cn := math.Sqrt(cnSq)
		if cn > 1e-15 {
			results[k] = dot * invQN / cn
		}
	}

	tq.PutBuf(qRot)
	return results
}

// GetConfig returns the quantizer configuration.
func (tq *TurboQuantMSE) GetConfig() tqdb.Config {
	return tq.config
}

// lookupPrecomputed checks if a precomputed codebook exists for (d, bits).
func lookupPrecomputed(d, bits int) *codec.Codebook {
	key := codec.PrecomputedKey{Dim: d, Bits: bits}
	entry, ok := codec.PrecomputedCodebooks[key]
	if !ok {
		return nil
	}

	numLevels := 1 << bits
	centroidsSq := make([]float64, numLevels)
	for i, c := range entry.Centroids {
		centroidsSq[i] = c * c
	}

	return &codec.Codebook{
		Dim:         d,
		Bits:        bits,
		NumLevels:   numLevels,
		Centroids:   entry.Centroids,
		Boundaries:  entry.Boundaries,
		CentroidsSq: centroidsSq,
	}
}

package tqdb

import (
	"github.com/scotteveritt/tqdb/internal/codec"
	"fmt"
	"math"
	"sync"

	"github.com/scotteveritt/tqdb/internal/mathutil"
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

func (c Config) validate() error {
	if c.Dim < 2 {
		return fmt.Errorf("turboquant: Dim must be >= 2, got %d", c.Dim)
	}
	if c.Bits < 1 || c.Bits > 8 {
		return fmt.Errorf("turboquant: Bits must be 1-8, got %d", c.Bits)
	}
	return nil
}

func (c Config) withDefaults() Config {
	if c.Bits == 0 {
		c.Bits = 4
	}
	if c.Seed == 0 {
		c.Seed = 42
	}
	return c
}

// TurboQuantMSE is the Stage 1 MSE-optimal quantizer.
// It applies random orthogonal rotation followed by per-coordinate
// Lloyd-Max scalar quantization.
//
// Safe for concurrent use after initialization.
type TurboQuantMSE struct {
	config   Config
	codebook *codec.Codebook
	rotation codec.Rotator

	// Pool of reusable float64 buffers (length = Dim) to avoid
	// heap allocation on every Quantize/Dequantize call.
	bufPool sync.Pool
}

// NewMSE creates a new TurboQuantMSE quantizer.
func NewMSE(cfg Config) (*TurboQuantMSE, error) {
	cfg = cfg.withDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	var rot codec.Rotator
	switch cfg.Rotation {
	case RotationHadamard:
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

func (tq *TurboQuantMSE) getBuf() []float64 {
	bp := tq.bufPool.Get().(*[]float64)
	return *bp
}

func (tq *TurboQuantMSE) putBuf(buf []float64) {
	tq.bufPool.Put(&buf)
}

// Quantize compresses a float64 vector into a CompressedVector.
// The input vec must have length equal to Config.Dim.
func (tq *TurboQuantMSE) Quantize(vec []float64) *CompressedVector {
	d := tq.config.Dim
	if len(vec) != d {
		panic(fmt.Sprintf("turboquant: Quantize: input length %d != Dim %d", len(vec), d))
	}
	workDim := tq.rotation.WorkDim()

	// 1. Normalize into a temp buffer of length d.
	unitBuf := tq.getBuf() // length workDim, we only use first d
	norm := mathutil.NormalizeTo(unitBuf[:d], vec)

	// 2. Rotate: input d → output workDim (may pad for Hadamard).
	rotated := tq.getBuf() // length workDim
	tq.rotation.Rotate(rotated, unitBuf[:d])
	tq.putBuf(unitBuf)

	// 3. Per-coordinate Lloyd-Max quantization over all workDim coordinates.
	indices := make([]uint8, workDim)
	tq.codebook.QuantizeTo(indices, rotated)
	tq.putBuf(rotated)

	return &CompressedVector{
		Dim:     d,
		Bits:    tq.config.Bits,
		Norm:    float32(norm),
		Indices: indices,
	}
}

// Dequantize reconstructs a float64 vector from a CompressedVector.
func (tq *TurboQuantMSE) Dequantize(cv *CompressedVector) []float64 {
	d := tq.config.Dim
	recon := make([]float64, d)
	tq.DequantizeTo(recon, cv)
	return recon
}

// DequantizeTo reconstructs a float64 vector into a pre-allocated dst buffer (length Dim).
func (tq *TurboQuantMSE) DequantizeTo(dst []float64, cv *CompressedVector) {
	d := tq.config.Dim

	// 1. Look up centroid values in rotated space (workDim coordinates).
	rotBuf := tq.getBuf() // length workDim
	tq.codebook.DequantizeTo(rotBuf[:len(cv.Indices)], cv.Indices)

	// 2. Un-rotate: workDim → d.
	tq.rotation.Unrotate(dst[:d], rotBuf[:len(cv.Indices)])
	tq.putBuf(rotBuf)

	// 3. Rescale by original norm.
	norm := float64(cv.Norm)
	for i := range d {
		dst[i] *= norm
	}
}

// QuantizeFloat32 compresses a float32 vector (convenience wrapper).
func (tq *TurboQuantMSE) QuantizeFloat32(vec []float32) *CompressedVector {
	return tq.Quantize(mathutil.Float32ToFloat64(vec))
}

// DequantizeFloat32 reconstructs a float32 vector (convenience wrapper).
func (tq *TurboQuantMSE) DequantizeFloat32(cv *CompressedVector) []float32 {
	return mathutil.Float64ToFloat32(tq.Dequantize(cv))
}

// CosineSimilarity computes cosine similarity between a raw query and a compressed vector.
// It decompresses the vector internally using a pooled buffer.
func (tq *TurboQuantMSE) CosineSimilarity(query []float64, cv *CompressedVector) float64 {
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
func (tq *TurboQuantMSE) AsymmetricCosineSimilarity(query []float64, cv *CompressedVector) float64 {
	centroids := tq.codebook.Centroids
	centroidsSq := tq.codebook.CentroidsSq

	qRot := tq.getBuf()
	tq.rotation.Rotate(qRot, query)
	queryNorm := mathutil.Norm(query)

	if queryNorm < 1e-15 {
		tq.putBuf(qRot)
		return 0
	}

	var dot, cnSq float64
	for j, idx := range cv.Indices {
		dot += qRot[j] * centroids[idx]
		cnSq += centroidsSq[idx]
	}
	tq.putBuf(qRot)

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
	query []float64, cvs []*CompressedVector,
) []float64 {
	centroids := tq.codebook.Centroids
	centroidsSq := tq.codebook.CentroidsSq
	workDim := tq.rotation.WorkDim()

	qRot := tq.getBuf()
	tq.rotation.Rotate(qRot, query)
	queryNorm := mathutil.Norm(query)

	results := make([]float64, len(cvs))
	if queryNorm < 1e-15 {
		tq.putBuf(qRot)
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

	tq.putBuf(qRot)
	return results
}

// GetConfig returns the quantizer configuration.
func (tq *TurboQuantMSE) GetConfig() Config {
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

package tqdb

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

// QJL implements the Quantized Johnson-Lindenstrauss transform for
// 1-bit residual sketching. This enables unbiased inner product
// estimation without full decompression.
type QJL struct {
	d     int       // input dimension
	m     int       // projection dimension
	seed  uint64    // seed for reproducibility
	data  []float64 // m×d random Gaussian matrix, row-major
	scale float64   // precomputed √(π/2) / m
}

// NewQJL creates a new QJL transform with projection dimension m.
// If m is 0, it defaults to d.
func NewQJL(d int, m int, seed uint64) *QJL {
	if m <= 0 {
		m = d
	}
	rng := rand.New(rand.NewPCG(seed, 0))

	data := make([]float64, m*d)
	for i := range data {
		data[i] = rng.NormFloat64()
	}

	return &QJL{
		d:     d,
		m:     m,
		seed:  seed,
		data:  data,
		scale: math.Sqrt(math.Pi/2.0) / float64(m),
	}
}

// Sketch computes sign(S · x), returning a slice of {-1, +1} values.
func (q *QJL) Sketch(x []float64) []int8 {
	signs := make([]int8, q.m)
	q.SketchTo(signs, x)
	return signs
}

// SketchTo computes sign(S · x) into a pre-allocated dst slice.
func (q *QJL) SketchTo(dst []int8, x []float64) {
	d := q.d
	m := q.m
	data := q.data

	// BCE hints.
	_ = x[d-1]
	_ = dst[m-1]
	_ = data[m*d-1]

	for i := range m {
		row := data[i*d : i*d+d : i*d+d]
		// Inline dot product to avoid function call overhead per row.
		var s0, s1, s2, s3 float64
		j := 0
		for ; j <= d-4; j += 4 {
			s0 += row[j] * x[j]
			s1 += row[j+1] * x[j+1]
			s2 += row[j+2] * x[j+2]
			s3 += row[j+3] * x[j+3]
		}
		for ; j < d; j++ {
			s0 += row[j] * x[j]
		}
		dot := s0 + s1 + s2 + s3
		if dot >= 0 {
			dst[i] = 1
		} else {
			dst[i] = -1
		}
	}
}

// InnerProduct computes the unbiased inner product estimator:
//
//	⟨q, x⟩ ≈ ⟨q, x_mse⟩ + γ · √(π/2)/m · ⟨S·q, signs⟩
//
// where x_mse is the MSE reconstruction, γ is the residual norm,
// and signs are the QJL sketch of the normalized residual.
func (q *QJL) InnerProduct(query, xMSE []float64, signs []int8, residualNorm float64) float64 {
	d := q.d
	m := q.m
	data := q.data

	// Term 1: dot product with MSE reconstruction
	term1 := mathutil.Dot(query, xMSE)

	// Term 2: QJL correction — ⟨S·query, signs⟩
	_ = data[m*d-1]
	_ = signs[m-1]
	_ = query[d-1]

	sqDot := 0.0
	for i := range m {
		row := data[i*d : i*d+d : i*d+d]
		var s0, s1, s2, s3 float64
		j := 0
		for ; j <= d-4; j += 4 {
			s0 += row[j] * query[j]
			s1 += row[j+1] * query[j+1]
			s2 += row[j+2] * query[j+2]
			s3 += row[j+3] * query[j+3]
		}
		for ; j < d; j++ {
			s0 += row[j] * query[j]
		}
		sqDot += (s0 + s1 + s2 + s3) * float64(signs[i])
	}

	return term1 + residualNorm*q.scale*sqDot
}

// TurboQuantProd is the Stage 1+2 quantizer that provides unbiased
// inner product estimation. It uses (Bits-1) bits for MSE quantization
// and 1 bit for QJL residual correction.
type TurboQuantProd struct {
	config Config
	mse    *TurboQuantMSE
	qjl    *QJL
}

// NewProd creates a new TurboQuantProd quantizer.
// The MSE stage uses (Bits-1) bits, and QJL uses 1 bit.
func NewProd(cfg Config) (*TurboQuantProd, error) {
	cfg = cfg.withDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if cfg.Bits < 2 {
		return nil, fmt.Errorf("turboquant: TurboQuantProd requires Bits >= 2, got %d", cfg.Bits)
	}

	mseCfg := cfg
	mseCfg.Bits = cfg.Bits - 1

	mse, err := NewMSE(mseCfg)
	if err != nil {
		return nil, err
	}

	qjl := NewQJL(cfg.Dim, cfg.Dim, cfg.Seed+1)

	return &TurboQuantProd{
		config: cfg,
		mse:    mse,
		qjl:    qjl,
	}, nil
}

// Quantize compresses a vector using MSE (Stage 1) + QJL (Stage 2).
func (tq *TurboQuantProd) Quantize(vec []float64) *CompressedProdVector {
	d := tq.config.Dim

	// Stage 1: MSE quantization
	cv := tq.mse.Quantize(vec)

	// Reconstruct MSE approximation into pooled buffer
	xMSE := tq.mse.getBuf()
	tq.mse.DequantizeTo(xMSE, cv)

	// Compute residual in-place in xMSE buffer (saves an allocation)
	var normSq float64
	for i := range d {
		r := vec[i] - xMSE[i]
		normSq += r * r
		xMSE[i] = r // reuse buffer for residual
	}
	residualNorm := math.Sqrt(normSq)

	// Normalize residual before sketching
	if residualNorm > 1e-15 {
		invNorm := 1.0 / residualNorm
		for i := range d {
			xMSE[i] *= invNorm
		}
	}

	signs := tq.qjl.Sketch(xMSE)
	tq.mse.putBuf(xMSE)

	return &CompressedProdVector{
		CompressedVector: *cv,
		Signs:            signs,
		ResidualNorm:     float32(residualNorm),
	}
}

// Dequantize reconstructs a vector from MSE + QJL compressed form.
func (tq *TurboQuantProd) Dequantize(cv *CompressedProdVector) []float64 {
	xMSE := tq.mse.Dequantize(&cv.CompressedVector)

	// Add QJL contribution: γ · √(π/2)/m · signs @ S
	scale := float64(cv.ResidualNorm) * tq.qjl.scale
	d := tq.qjl.d
	data := tq.qjl.data

	for i := range tq.qjl.m {
		s := float64(cv.Signs[i]) * scale
		if s == 0 {
			continue
		}
		row := data[i*d : i*d+d : i*d+d]
		for j := range d {
			xMSE[j] += s * row[j]
		}
	}

	return xMSE
}

// InnerProduct computes an unbiased estimate of ⟨query, vec⟩
// without fully decompressing the vector.
func (tq *TurboQuantProd) InnerProduct(query []float64, cv *CompressedProdVector) float64 {
	buf := tq.mse.getBuf()
	tq.mse.DequantizeTo(buf, &cv.CompressedVector)
	ip := tq.qjl.InnerProduct(query, buf, cv.Signs, float64(cv.ResidualNorm))
	tq.mse.putBuf(buf)
	return ip
}

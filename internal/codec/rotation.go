package codec

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// Rotator applies an orthogonal rotation and its inverse.
//
// Dim() returns the original input dimension d.
// WorkDim() returns the dimension of the rotated output (may be > Dim
// for HadamardRotator which pads to the next power of 2).
//
// Rotate: src has length Dim(), dst has length WorkDim().
// Unrotate: src has length WorkDim(), dst has length Dim().
type Rotator interface {
	Rotate(dst, src []float64)
	Unrotate(dst, src []float64)
	Dim() int
	WorkDim() int
}

// RotationMatrix holds a d×d random orthogonal matrix drawn from the
// Haar measure via QR decomposition of a random Gaussian matrix.
type RotationMatrix struct {
	d    int
	seed uint64
	data []float64 // d×d matrix stored row-major
}

// NewRotationMatrix generates a d×d random orthogonal matrix from the
// Haar measure. The matrix is deterministic for a given (d, seed) pair.
func NewRotationMatrix(d int, seed uint64) *RotationMatrix {
	rng := rand.New(rand.NewPCG(seed, 0))

	// Fill d×d with i.i.d. N(0,1)
	data := make([]float64, d*d)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	G := mat.NewDense(d, d, data)

	// QR decomposition
	var qr mat.QR
	qr.Factorize(G)

	var Q mat.Dense
	qr.QTo(&Q)

	var R mat.Dense
	qr.RTo(&R)

	// Sign correction: multiply columns of Q by sign(R_ii)
	// to ensure a unique decomposition from the Haar measure.
	qData := Q.RawMatrix().Data
	for i := range d {
		if R.At(i, i) < 0 {
			for j := range d {
				qData[j*d+i] = -qData[j*d+i]
			}
		}
	}

	return &RotationMatrix{
		d:    d,
		seed: seed,
		data: qData,
	}
}

// Rotate computes dst = Q · src. Both dst and src must have length d.
// Row-major access pattern — cache-friendly.
func (r *RotationMatrix) Rotate(dst, src []float64) {
	d := r.d
	rdata := r.data

	// BCE hints: prove slice bounds to the compiler once.
	_ = dst[d-1]
	_ = src[d-1]
	_ = rdata[d*d-1]

	for i := range d {
		row := rdata[i*d : i*d+d : i*d+d]
		var s0, s1, s2, s3 float64
		j := 0
		for ; j <= d-4; j += 4 {
			s0 += row[j] * src[j]
			s1 += row[j+1] * src[j+1]
			s2 += row[j+2] * src[j+2]
			s3 += row[j+3] * src[j+3]
		}
		for ; j < d; j++ {
			s0 += row[j] * src[j]
		}
		dst[i] = s0 + s1 + s2 + s3
	}
}

// Unrotate computes dst = Qᵀ · src (inverse rotation).
// Both dst and src must have length d.
//
// Uses outer-product formulation: dst += src[j] * Q[j,:] for each j.
// This accesses the matrix row-by-row (cache-friendly), unlike the naive
// column-by-column approach which causes d² cache misses at large d.
func (r *RotationMatrix) Unrotate(dst, src []float64) {
	d := r.d
	rdata := r.data

	// BCE hints.
	_ = dst[d-1]
	_ = src[d-1]
	_ = rdata[d*d-1]

	// Zero destination.
	for i := range d {
		dst[i] = 0
	}

	// Accumulate: dst[i] = Σ_j Q[j,i] * src[j]
	// Rewritten as: for each j, dst[:] += src[j] * Q[j,:]
	for j := range d {
		sj := src[j]
		if sj == 0 {
			continue
		}
		row := rdata[j*d : j*d+d : j*d+d]
		i := 0
		for ; i <= d-4; i += 4 {
			dst[i] += sj * row[i]
			dst[i+1] += sj * row[i+1]
			dst[i+2] += sj * row[i+2]
			dst[i+3] += sj * row[i+3]
		}
		for ; i < d; i++ {
			dst[i] += sj * row[i]
		}
	}
}

// Dim returns the dimension of the rotation matrix.
func (r *RotationMatrix) Dim() int { return r.d }

// WorkDim returns the working dimension (same as Dim for QR rotation).
func (r *RotationMatrix) WorkDim() int { return r.d }

// At returns the element at row i, column j.
func (r *RotationMatrix) At(i, j int) float64 {
	return r.data[i*r.d+j]
}

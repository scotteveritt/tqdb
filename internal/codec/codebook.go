package codec

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/integrate/quad"
	"gonum.org/v1/gonum/stat/distuv"
)

// Codebook holds precomputed Lloyd-Max centroids and decision boundaries
// for a given dimension and bit-width.
type Codebook struct {
	Dim         int       // vector dimension d (determines σ = 1/√d)
	Bits        int       // quantization bits per coordinate
	NumLevels   int       // 2^Bits
	Centroids   []float64 // length NumLevels, sorted ascending
	Boundaries  []float64 // length NumLevels-1, decision thresholds
	CentroidsSq []float64 // Centroids[i]², precomputed for fast norm calculation
	Distortion  float64   // expected MSE distortion per coordinate
}

// SolveCodebook computes the optimal Lloyd-Max codebook for the Gaussian
// distribution N(0, 1/d) arising from random rotation of unit vectors.
//
// For d >= 64, the Gaussian approximation is excellent. Set useExact=true
// to use the exact Beta PDF for smaller dimensions.
func SolveCodebook(d, bits int, useExact bool) *Codebook {
	numLevels := 1 << bits
	sigma := 1.0 / math.Sqrt(float64(d))

	var pdf func(float64) float64
	if useExact && d > 2 {
		pdf = betaPDF(d)
	} else {
		pdf = gaussianPDF(sigma)
	}

	// Initialize centroids at Gaussian quantiles (OmarHory approach — faster convergence)
	norm := distuv.Normal{Mu: 0, Sigma: sigma}
	centroids := make([]float64, numLevels)
	for i := range numLevels {
		p := (2.0*float64(i) + 1.0) / (2.0 * float64(numLevels))
		centroids[i] = norm.Quantile(p)
	}

	clipLo := -6.0 * sigma
	clipHi := 6.0 * sigma

	// Pre-allocate iteration buffers (reused every iteration)
	boundaries := make([]float64, numLevels-1)
	edges := make([]float64, numLevels+1)
	newCentroids := make([]float64, numLevels)

	edges[0] = clipLo
	edges[numLevels] = clipHi

	// Lloyd-Max iteration
	for range 200 {
		// Step 1: boundaries = midpoints between adjacent centroids
		for i := range numLevels - 1 {
			boundaries[i] = (centroids[i] + centroids[i+1]) * 0.5
		}

		// Step 2: update centroids as E[X | X in partition_i]
		copy(edges[1:], boundaries)

		maxShift := 0.0
		for i := range numLevels {
			a, b := edges[i], edges[i+1]

			numerator := quad.Fixed(func(x float64) float64 {
				return x * pdf(x)
			}, a, b, 100, nil, 0)

			denominator := quad.Fixed(func(x float64) float64 {
				return pdf(x)
			}, a, b, 100, nil, 0)

			if denominator > 1e-15 {
				newCentroids[i] = numerator / denominator
			} else {
				newCentroids[i] = (a + b) * 0.5
			}

			if shift := math.Abs(newCentroids[i] - centroids[i]); shift > maxShift {
				maxShift = shift
			}
		}

		centroids, newCentroids = newCentroids, centroids
		if maxShift < 1e-10 {
			break
		}
	}

	// Final boundaries
	for i := range numLevels - 1 {
		boundaries[i] = (centroids[i] + centroids[i+1]) * 0.5
	}

	// Precompute squared centroids
	centroidsSq := make([]float64, numLevels)
	for i, c := range centroids {
		centroidsSq[i] = c * c
	}

	distortion := computeDistortion(centroids, boundaries, pdf, clipLo, clipHi)

	return &Codebook{
		Dim:         d,
		Bits:        bits,
		NumLevels:   numLevels,
		Centroids:   centroids,
		Boundaries:  boundaries,
		CentroidsSq: centroidsSq,
		Distortion:  distortion,
	}
}

// QuantizeTo maps each scalar value to its nearest centroid index,
// writing results into the pre-allocated dst slice.
func (c *Codebook) QuantizeTo(dst []uint8, values []float64) {
	bounds := c.Boundaries
	for i, v := range values {
		dst[i] = uint8(sort.SearchFloat64s(bounds, v))
	}
}

// Quantize maps each scalar value to its nearest centroid index.
// Allocates a new result slice. Prefer QuantizeTo on hot paths.
func (c *Codebook) Quantize(values []float64) []uint8 {
	dst := make([]uint8, len(values))
	c.QuantizeTo(dst, values)
	return dst
}

// DequantizeTo maps indices back to centroid values,
// writing results into the pre-allocated dst slice.
func (c *Codebook) DequantizeTo(dst []float64, indices []uint8) {
	centroids := c.Centroids
	for i, idx := range indices {
		dst[i] = centroids[idx]
	}
}

// Dequantize maps indices back to centroid values.
// Allocates a new result slice. Prefer DequantizeTo on hot paths.
func (c *Codebook) Dequantize(indices []uint8) []float64 {
	dst := make([]float64, len(indices))
	c.DequantizeTo(dst, indices)
	return dst
}

// gaussianPDF returns the PDF of N(0, σ²).
func gaussianPDF(sigma float64) func(float64) float64 {
	sigma2 := sigma * sigma
	coeff := 1.0 / math.Sqrt(2*math.Pi*sigma2)
	inv2sigma2 := -1.0 / (2 * sigma2)
	return func(x float64) float64 {
		return coeff * math.Exp(x*x*inv2sigma2)
	}
}

// betaPDF returns the exact marginal PDF of a rotated unit-vector coordinate
// in d dimensions: f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
// supported on [-1, 1].
func betaPDF(d int) func(float64) float64 {
	df := float64(d)
	lgCoeff, _ := math.Lgamma(df / 2.0)
	lgDenom, _ := math.Lgamma((df - 1.0) / 2.0)
	logCoeff := lgCoeff - lgDenom - 0.5*math.Log(math.Pi)
	exp := (df - 3.0) / 2.0

	return func(x float64) float64 {
		if x <= -1.0 || x >= 1.0 {
			return 0.0
		}
		base := 1.0 - x*x
		if base <= 0 {
			return 0
		}
		return math.Exp(logCoeff + exp*math.Log(base))
	}
}

func computeDistortion(centroids, boundaries []float64, pdf func(float64) float64, clipLo, clipHi float64) float64 {
	numLevels := len(centroids)
	edges := make([]float64, numLevels+1)
	edges[0] = clipLo
	copy(edges[1:], boundaries)
	edges[numLevels] = clipHi

	totalDistortion := 0.0
	for i := range numLevels {
		a, b := edges[i], edges[i+1]
		c := centroids[i]
		dist := quad.Fixed(func(x float64) float64 {
			diff := x - c
			return diff * diff * pdf(x)
		}, a, b, 100, nil, 0)
		totalDistortion += dist
	}
	return totalDistortion
}

package tqdb

import (
	"math/rand/v2"
	"testing"

	"github.com/scotteveritt/tqdb/internal/mathutil"
)

func TestCompressedVectorMarshalRoundtrip(t *testing.T) {
	d := 128
	rng := rand.New(rand.NewPCG(100, 0))

	for _, bits := range []int{1, 2, 3, 4} {
		q, err := NewMSE(Config{Dim: d, Bits: bits, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}

		vec := randomVector(d, rng)
		cv := q.Quantize(vec)

		data, err := cv.MarshalBinary()
		if err != nil {
			t.Fatalf("bits=%d: marshal error: %v", bits, err)
		}

		cv2 := &CompressedVector{}
		err = cv2.UnmarshalBinary(data)
		if err != nil {
			t.Fatalf("bits=%d: unmarshal error: %v", bits, err)
		}

		if cv2.Dim != cv.Dim || cv2.Bits != cv.Bits || cv2.Norm != cv.Norm {
			t.Errorf("bits=%d: header mismatch", bits)
		}
		if len(cv2.Indices) != len(cv.Indices) {
			t.Errorf("bits=%d: indices length %d vs %d", bits, len(cv2.Indices), len(cv.Indices))
			continue
		}
		for i := range cv.Indices {
			if cv2.Indices[i] != cv.Indices[i] {
				t.Errorf("bits=%d idx=%d: got %d, want %d", bits, i, cv2.Indices[i], cv.Indices[i])
				break
			}
		}
	}
}

func TestCompressedVectorDequantizeAfterRoundtrip(t *testing.T) {
	d := 128
	rng := rand.New(rand.NewPCG(200, 0))

	q, err := NewMSE(Config{Dim: d, Bits: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	vec := randomVector(d, rng)
	cv := q.Quantize(vec)

	// Serialize and deserialize
	data, _ := cv.MarshalBinary()
	cv2 := &CompressedVector{}
	_ = cv2.UnmarshalBinary(data)

	// Both should dequantize to the same vector
	recon1 := q.Dequantize(cv)
	recon2 := q.Dequantize(cv2)

	sim := mathutil.CosineSimilarity(recon1, recon2)
	if sim < 0.9999 {
		t.Errorf("dequantized vectors differ after roundtrip: cosine sim = %f", sim)
	}
}

func TestCompressedVectorSize(t *testing.T) {
	tests := []struct {
		dim, bits int
		wantSize  int
	}{
		{3072, 4, 7 + 1536},
		{3072, 2, 7 + 768},
		{3072, 3, 7 + 1152},
		{1536, 4, 7 + 768},
	}

	for _, tt := range tests {
		cv := &CompressedVector{Dim: tt.dim, Bits: tt.bits}
		got := cv.Size()
		if got != tt.wantSize {
			t.Errorf("dim=%d bits=%d: Size()=%d, want %d", tt.dim, tt.bits, got, tt.wantSize)
		}
	}
}

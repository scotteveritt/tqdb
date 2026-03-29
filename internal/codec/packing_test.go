package codec

import (
	"testing"
)

func TestPack4BitRoundtrip(t *testing.T) {
	for _, n := range []int{1, 2, 3, 10, 100, 1001, 3072} {
		indices := make([]uint8, n)
		for i := range indices {
			indices[i] = uint8(i % 16)
		}

		packed := make([]byte, PackedSize(n, 4))
		Pack4BitTo(packed, indices)
		unpacked := make([]uint8, n)
		Unpack4BitTo(unpacked, packed)

		for i := range n {
			if unpacked[i] != indices[i] {
				t.Errorf("n=%d idx=%d: got %d, want %d", n, i, unpacked[i], indices[i])
				break
			}
		}
	}
}

func TestPack2BitRoundtrip(t *testing.T) {
	for _, n := range []int{1, 4, 7, 100, 3072} {
		indices := make([]uint8, n)
		for i := range indices {
			indices[i] = uint8(i % 4)
		}

		packed := make([]byte, PackedSize(n, 2))
		Pack2BitTo(packed, indices)
		unpacked := make([]uint8, n)
		Unpack2BitTo(unpacked, packed)

		for i := range n {
			if unpacked[i] != indices[i] {
				t.Errorf("n=%d idx=%d: got %d, want %d", n, i, unpacked[i], indices[i])
				break
			}
		}
	}
}

func TestPack3BitRoundtrip(t *testing.T) {
	for _, n := range []int{1, 8, 9, 100, 3072} {
		indices := make([]uint8, n)
		for i := range indices {
			indices[i] = uint8(i % 8)
		}

		packed := make([]byte, PackedSize(n, 3))
		Pack3BitTo(packed, indices)
		unpacked := make([]uint8, n)
		Unpack3BitTo(unpacked, packed)

		for i := range n {
			if unpacked[i] != indices[i] {
				t.Errorf("n=%d idx=%d: got %d, want %d", n, i, unpacked[i], indices[i])
				break
			}
		}
	}
}

func TestPack1BitRoundtrip(t *testing.T) {
	for _, n := range []int{1, 8, 9, 100, 3072} {
		indices := make([]uint8, n)
		for i := range indices {
			indices[i] = uint8(i % 2)
		}

		packed := make([]byte, PackedSize(n, 1))
		Pack1BitTo(packed, indices)
		unpacked := make([]uint8, n)
		Unpack1BitTo(unpacked, packed)

		for i := range n {
			if unpacked[i] != indices[i] {
				t.Errorf("n=%d idx=%d: got %d, want %d", n, i, unpacked[i], indices[i])
				break
			}
		}
	}
}

func TestPackedSize(t *testing.T) {
	tests := []struct {
		n, bits, want int
	}{
		{3072, 4, 1536},
		{3072, 2, 768},
		{3072, 3, 1152},
		{3072, 1, 384},
		{1, 4, 1},
		{2, 4, 1},
		{3, 4, 2},
	}
	for _, tt := range tests {
		got := PackedSize(tt.n, tt.bits)
		if got != tt.want {
			t.Errorf("PackedSize(%d, %d) = %d, want %d", tt.n, tt.bits, got, tt.want)
		}
	}
}

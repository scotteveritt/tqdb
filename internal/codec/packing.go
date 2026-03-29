package codec

// PackIndicesTo packs quantization indices into an existing byte slice.
// dst must have length >= PackedSize(len(indices), bits).
func PackIndicesTo(dst []byte, indices []uint8, bits int) {
	switch bits {
	case 4:
		Pack4BitTo(dst, indices)
	case 2:
		Pack2BitTo(dst, indices)
	case 3:
		Pack3BitTo(dst, indices)
	case 1:
		Pack1BitTo(dst, indices)
	default:
		copy(dst, indices)
	}
}

// unpackIndices unpacks indices from a compact byte slice.
func UnpackIndices(packed []byte, n, bits int) []uint8 {
	out := make([]uint8, n)
	switch bits {
	case 4:
		Unpack4BitTo(out, packed)
	case 2:
		Unpack2BitTo(out, packed)
	case 3:
		Unpack3BitTo(out, packed)
	case 1:
		Unpack1BitTo(out, packed)
	default:
		copy(out, packed[:n])
	}
	return out
}

// PackedSize returns the number of bytes needed to pack n indices at the given bit-width.
func PackedSize(n, bits int) int {
	return (n*bits + 7) / 8
}

// --- 4-bit packing: 2 values per byte ---

func Pack4BitTo(dst []byte, indices []uint8) {
	n := len(indices)
	pairs := n / 2
	for i := range pairs {
		dst[i] = (indices[i*2] & 0x0F) | (indices[i*2+1] << 4)
	}
	if n&1 != 0 {
		dst[pairs] = indices[n-1] & 0x0F
	}
}

func Unpack4BitTo(dst []uint8, packed []byte) {
	n := len(dst)
	pairs := n / 2
	for i := range pairs {
		b := packed[i]
		dst[i*2] = b & 0x0F
		dst[i*2+1] = b >> 4
	}
	if n&1 != 0 {
		dst[n-1] = packed[pairs] & 0x0F
	}
}

// --- 2-bit packing: 4 values per byte ---

func Pack2BitTo(dst []byte, indices []uint8) {
	n := len(indices)
	quads := n / 4
	for i := range quads {
		base := i * 4
		dst[i] = (indices[base] & 0x03) |
			((indices[base+1] & 0x03) << 2) |
			((indices[base+2] & 0x03) << 4) |
			((indices[base+3] & 0x03) << 6)
	}
	// Handle remainder
	rem := n - quads*4
	if rem > 0 {
		var b byte
		base := quads * 4
		for k := range rem {
			b |= (indices[base+k] & 0x03) << (uint(k) * 2)
		}
		dst[quads] = b
	}
}

func Unpack2BitTo(dst []uint8, packed []byte) {
	n := len(dst)
	quads := n / 4
	for i := range quads {
		b := packed[i]
		base := i * 4
		dst[base] = b & 0x03
		dst[base+1] = (b >> 2) & 0x03
		dst[base+2] = (b >> 4) & 0x03
		dst[base+3] = (b >> 6) & 0x03
	}
	rem := n - quads*4
	if rem > 0 {
		b := packed[quads]
		base := quads * 4
		for k := range rem {
			dst[base+k] = (b >> (uint(k) * 2)) & 0x03
		}
	}
}

// --- 3-bit packing: 8 values per 3 bytes (24 bits) ---

func Pack3BitTo(dst []byte, indices []uint8) {
	n := len(indices)
	for i := range n {
		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOffset := uint(bitPos % 8)
		val := uint16(indices[i] & 0x07)
		word := uint16(dst[byteIdx])
		if byteIdx+1 < len(dst) {
			word |= uint16(dst[byteIdx+1]) << 8
		}
		word |= val << bitOffset
		dst[byteIdx] = byte(word)
		if byteIdx+1 < len(dst) {
			dst[byteIdx+1] = byte(word >> 8)
		}
	}
}

func Unpack3BitTo(dst []uint8, packed []byte) {
	n := len(dst)
	for i := range n {
		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOffset := uint(bitPos % 8)
		word := uint16(packed[byteIdx])
		if byteIdx+1 < len(packed) {
			word |= uint16(packed[byteIdx+1]) << 8
		}
		dst[i] = uint8((word >> bitOffset) & 0x07)
	}
}

// --- 1-bit packing: 8 values per byte ---

func Pack1BitTo(dst []byte, indices []uint8) {
	n := len(indices)
	// Process 8 at a time for speed.
	octets := n / 8
	for i := range octets {
		base := i * 8
		var b byte
		b |= (indices[base] & 1)
		b |= (indices[base+1] & 1) << 1
		b |= (indices[base+2] & 1) << 2
		b |= (indices[base+3] & 1) << 3
		b |= (indices[base+4] & 1) << 4
		b |= (indices[base+5] & 1) << 5
		b |= (indices[base+6] & 1) << 6
		b |= (indices[base+7] & 1) << 7
		dst[i] = b
	}
	rem := n - octets*8
	if rem > 0 {
		var b byte
		base := octets * 8
		for k := range rem {
			b |= (indices[base+k] & 1) << uint(k)
		}
		dst[octets] = b
	}
}

func Unpack1BitTo(dst []uint8, packed []byte) {
	n := len(dst)
	octets := n / 8
	for i := range octets {
		b := packed[i]
		base := i * 8
		dst[base] = b & 1
		dst[base+1] = (b >> 1) & 1
		dst[base+2] = (b >> 2) & 1
		dst[base+3] = (b >> 3) & 1
		dst[base+4] = (b >> 4) & 1
		dst[base+5] = (b >> 5) & 1
		dst[base+6] = (b >> 6) & 1
		dst[base+7] = (b >> 7) & 1
	}
	rem := n - octets*8
	if rem > 0 {
		b := packed[octets]
		base := octets * 8
		for k := range rem {
			dst[base+k] = (b >> uint(k)) & 1
		}
	}
}

//go:build arm64

package distancer

import (
	"github.com/scotteveritt/tqdb/internal/distancer/asm"
	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.ARM64.HasASIMD {
		dotImpl = asm.DotNeon
		negDotImpl = asm.NegDotNeon
		l2Impl = asm.L2Neon
		dotF64Impl = asm.DotF64
		vecMulF64Impl = asm.VecMulF64
		vecScaleF64Impl = asm.VecScaleF64
	}
}

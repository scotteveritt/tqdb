//go:build tqdb_nommap

package tqdb

import (
	"io"
	"os"
)

// mapFile reads the entire file into memory (mmap disabled via build tag).
func mapFile(f *os.File) ([]byte, func() error, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, nil, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}

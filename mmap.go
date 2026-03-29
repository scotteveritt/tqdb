//go:build !tqdb_nommap

package tqdb

import (
	"io"
	"os"

	mmap "github.com/edsrzf/mmap-go"
)

// mapFile memory-maps the file for read-only access.
// Falls back to bulk read if mmap fails.
func mapFile(f *os.File) ([]byte, func() error, error) {
	m, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		// Transparent fallback to bulk read.
		return bulkRead(f)
	}
	return []byte(m), m.Unmap, nil
}

func bulkRead(f *os.File) ([]byte, func() error, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, nil, err
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}

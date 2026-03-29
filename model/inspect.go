package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"

	"github.com/scotteveritt/tqdb"
)

// OpenTQM opens a .tqm file and returns its header.
func OpenTQM(path string) (*tqdb.TQModelHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck

	var lenBuf [8]byte
	if _, err := f.Read(lenBuf[:]); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen > 100_000_000 {
		return nil, fmt.Errorf("header too large: %d bytes", headerLen)
	}

	headerJSON := make([]byte, headerLen)
	if _, err := f.Read(headerJSON); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var header tqdb.TQModelHeader
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	return &header, nil
}

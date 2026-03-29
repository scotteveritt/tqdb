package model

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
	"github.com/scotteveritt/tqdb/internal/mathutil"
	"github.com/scotteveritt/tqdb/quantize"
)

// Compress quantizes a SafeTensors file and writes a .tqm file.
//
// Row quantization is parallelized across Workers goroutines per tensor.
// Each worker reads rows from a channel, quantizes them, and writes the
// packed result directly to a pre-allocated slot in the output buffer.
// No coordination is needed for writes since each row has a unique slot.
func Compress(sf *tqdb.SafeTensorsFile, outPath string, cfg tqdb.ModelConfig, progress tqdb.CompressModelProgress) (*tqdb.TQModelHeader, error) {
	cfg = withDefaults(cfg)

	// Collect quantizable tensors (2D weight matrices with float dtypes).
	type tensorWork struct {
		name string
		info tqdb.TensorInfo
	}
	var work []tensorWork
	for _, name := range sf.TensorNames() {
		ti, _ := sf.Tensor(name)
		if len(ti.Shape) < 2 {
			continue
		}
		switch ti.DType {
		case "F32", "F16", "BF16", "F64":
			work = append(work, tensorWork{name: name, info: ti})
		}
	}
	sort.Slice(work, func(i, j int) bool {
		return work[i].name < work[j].name
	})

	// Quantizer cache — one per unique row dimension.
	quantizers := make(map[int]*quantize.TurboQuantMSE)
	getQuantizer := func(dim int) (*quantize.TurboQuantMSE, error) {
		if q, ok := quantizers[dim]; ok {
			return q, nil
		}
		q, err := quantize.NewMSE(tqdb.Config{
			Dim:      dim,
			Bits:     cfg.Bits,
			Rotation: cfg.Rotation,
			Seed:     cfg.Seed,
		})
		if err != nil {
			return nil, err
		}
		quantizers[dim] = q
		return q, nil
	}

	tqmTensors := make(map[string]tqdb.TQMTensorInfo)
	type tensorResult struct {
		name string
		data []byte
	}
	var results []tensorResult
	dataOffset := int64(0)
	bits := cfg.Bits
	numWorkers := cfg.Workers

	for ti, tw := range work {
		rows := tw.info.Rows()
		cols := tw.info.Cols()

		q, err := getQuantizer(cols)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: create quantizer: %w", tw.name, err)
		}

		workDim := q.Rotation().WorkDim()
		packedLen := codec.PackedSize(workDim, bits)
		packedRowSize := packedLen + 4 // packed indices + float32 norm
		tensorSize := int64(rows) * int64(packedRowSize)

		// Pre-allocate output buffer. Each worker writes to its own slot
		// at offset row*packedRowSize — no locking needed.
		tensorData := make([]byte, tensorSize)

		// Quality sampling: accumulate cosine similarity from sampled rows.
		var cosSimSum atomic.Int64 // fixed-point: store sim * 1e9 as int64
		var cosSimCount atomic.Int64

		// Progress tracking.
		var rowsDone atomic.Int64

		// --- Fan-out: workers read row indices from channel ---
		rowCh := make(chan int, numWorkers*4)
		var wg sync.WaitGroup

		// Error from any worker (first error wins).
		var firstErr atomic.Value

		for w := range numWorkers {
			_ = w
			wg.Go(func() {

				// Each worker has its own pack buffer to avoid allocation per row.
				localPackBuf := make([]byte, packedLen)

				for row := range rowCh {
					// Check if another worker already failed.
					if firstErr.Load() != nil {
						return
					}

					rowVec, err := sf.ReadRowFloat64(tw.name, row)
					if err != nil {
						firstErr.CompareAndSwap(nil, fmt.Errorf("tensor %s row %d: %w", tw.name, row, err))
						return
					}

					cv := q.Quantize(rowVec)

					// Write packed indices + norm to this row's slot.
					off := row * packedRowSize
					codec.PackIndicesTo(localPackBuf, cv.Indices, bits)
					copy(tensorData[off:], localPackBuf)
					binary.LittleEndian.PutUint32(
						tensorData[off+packedLen:],
						math.Float32bits(cv.Norm),
					)

					// Sample quality every 100th row.
					if row%100 == 0 {
						recon := q.Dequantize(cv)
						sim := mathutil.CosineSimilarity(rowVec, recon)
						cosSimSum.Add(int64(sim * 1e9))
						cosSimCount.Add(1)
					}

					rowsDone.Add(1)
				}
			})
		}

		// --- Producer: feed row indices into the channel ---
		go func() {
			for r := range rows {
				rowCh <- r
			}
			close(rowCh)
		}()

		// --- Progress reporting from main goroutine ---
		// Poll rowsDone periodically while workers are running.
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		lastReported := 0
		for {
			select {
			case <-done:
				// Final progress report.
				if progress != nil {
					progress(tw.name, ti, len(work), rows, rows)
				}
				goto tensorDone
			default:
				current := int(rowsDone.Load())
				if progress != nil && current-lastReported >= 1000 {
					progress(tw.name, ti, len(work), current, rows)
					lastReported = current
				}
				runtime.Gosched() // yield to let workers run
			}
		}
	tensorDone:

		// Check for worker errors.
		if errVal := firstErr.Load(); errVal != nil {
			return nil, errVal.(error) //nolint:errcheck // type guaranteed by CompareAndSwap
		}

		// Compute average cosine similarity.
		avgCosSim := 0.0
		if n := cosSimCount.Load(); n > 0 {
			avgCosSim = float64(cosSimSum.Load()) / (float64(n) * 1e9)
		}

		tqmTensors[tw.name] = tqdb.TQMTensorInfo{
			Shape:     tw.info.Shape,
			OrigDType: tw.info.DType,
			Rows:      rows,
			RowDim:    cols,
			WorkDim:   workDim,
			Offset:    dataOffset,
			Size:      tensorSize,
			AvgCosSim: avgCosSim,
		}

		results = append(results, tensorResult{name: tw.name, data: tensorData})
		dataOffset += tensorSize
	}

	// Build header.
	rotName := "qr"
	if cfg.Rotation == tqdb.RotationHadamard {
		rotName = "hadamard"
	}
	header := &tqdb.TQModelHeader{
		Version:  1,
		Source:   sf.Metadata()["format"],
		Bits:     cfg.Bits,
		Packed:   true,
		Rotation: rotName,
		Seed:     cfg.Seed,
		Tensors:  tqmTensors,
	}

	// --- Write file ---
	headerJSON, err := json.MarshalIndent(header, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal header: %w", err)
	}

	tmpPath := outPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return nil, err
	}
	cleanup := func(err error) (*tqdb.TQModelHeader, error) {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return nil, err
	}

	w := bufio.NewWriterSize(f, 1<<20)

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := w.Write(lenBuf[:]); err != nil {
		return cleanup(fmt.Errorf("write header length: %w", err))
	}
	if _, err := w.Write(headerJSON); err != nil {
		return cleanup(fmt.Errorf("write header JSON: %w", err))
	}
	for _, tr := range results {
		if _, err := w.Write(tr.data); err != nil {
			return cleanup(fmt.Errorf("write tensor %s: %w", tr.name, err))
		}
	}
	if err := w.Flush(); err != nil {
		return cleanup(err)
	}
	if err := f.Sync(); err != nil {
		return cleanup(fmt.Errorf("sync: %w", err))
	}
	_ = f.Close()

	if err := os.Rename(tmpPath, outPath); err != nil {
		_ = os.Remove(tmpPath)
		return nil, err
	}

	return header, nil
}

func withDefaults(c tqdb.ModelConfig) tqdb.ModelConfig {
	if c.Bits == 0 {
		c.Bits = 4
	}
	if c.Seed == 0 {
		c.Seed = 42
	}
	if c.Workers <= 0 {
		c.Workers = runtime.GOMAXPROCS(0)
	}
	return c
}

package store

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/codec"
	"github.com/scotteveritt/tqdb/internal/mathutil"
	"github.com/scotteveritt/tqdb/quantize"
)

type storeMode int

const (
	modeWrite storeMode = iota
	modeRead
)

// Store is a quantized vector store backed by a .tq file.
//
// A Store is either in write mode (created via Create) or
// read mode (opened via Open). Write-mode stores support Add/Flush.
// Read-mode stores support Search. Concurrent Search calls are safe.
type Store struct {
	path      string
	config    tqdb.StoreConfig
	quantizer *quantize.TurboQuantMSE
	mode      storeMode
	workDim   int

	// Write mode: buffer in memory, flush to disk.
	buf *writeBuffer

	// Read mode: mmap'd or loaded file data.
	data       []byte       // full file data (mmap or read)
	release    func() error // cleanup (Unmap or no-op)
	allIndices []byte       // slice into data (indices section, zero-copy)
	norms      []float32    // original L2 norms (for dequantization, not search)
	numVecs    int
	header     fileHeader // cached for lazy loading

	// Lazy-loaded via sync.Once on first Search result access.
	idsOnce  sync.Once
	ids      []string
	dataRaw  [][]byte // raw JSON blobs
	contents []string // document content

	// Parsed data cache for filter-heavy searches (avoids repeated json.Unmarshal).
	dataCacheOnce sync.Once
	dataCache     []map[string]any

	// IVF index (built lazily on first indexed search).
	ivfOnce sync.Once
	ivfIdx  *ivfIndex

	// HNSW index (loaded from .tq file if present).
	hnswIdx  *hnswIndex
	hnswVecs []float32 // decoded float32 vectors for HNSW search
}

// writeBuffer accumulates vectors before flushing to disk.
type writeBuffer struct {
	allIndices []uint8
	norms      []float32
	ids        []string
	data       [][]byte  // raw JSON bytes per vector (map[string]any)
	contents   []string  // document content per vector
	graphData  []byte    // serialized HNSW graph (optional)
}

// Create creates a new store for writing. Call Add() to insert vectors,
// then Flush() to write the .tq file atomically.
func Create(path string, cfg tqdb.StoreConfig) (*Store, error) {
	cfg = cfg.WithDefaults()
	qcfg := cfg.ToConfig()
	if err := qcfg.Validate(); err != nil {
		return nil, err
	}

	q, err := quantize.NewMSE(qcfg)
	if err != nil {
		return nil, fmt.Errorf("tqdb: create quantizer: %w", err)
	}

	return &Store{
		path:      path,
		config:    cfg,
		quantizer: q,
		mode:      modeWrite,
		workDim:   q.Rotation().WorkDim(),
		buf:       &writeBuffer{},
	}, nil
}

// Add quantizes and buffers a document for later flushing.
// Only valid in write mode (created via Create).
// doc.Embedding must be non-nil; Store does not support auto-embedding.
func (s *Store) Add(doc tqdb.Document) error {
	if s.mode != modeWrite {
		return fmt.Errorf("tqdb: Add called on read-only store")
	}

	vec := doc.Embedding
	if len(vec) == 0 {
		return fmt.Errorf("tqdb: Document.Embedding is required for Store")
	}

	cv := s.quantizer.Quantize(vec)
	return s.addBuf(doc.ID, doc.Content, cv.Indices, cv.Norm, doc.Data)
}

// addBuf appends to the write buffer. Shared by Add and AddRaw.
func (s *Store) addBuf(id, content string, indices []uint8, norm float32, data map[string]any) error {
	var dataJSON []byte
	if len(data) > 0 {
		var err error
		dataJSON, err = json.Marshal(data)
		if err != nil {
			return fmt.Errorf("tqdb: marshal data: %w", err)
		}
	}

	s.buf.allIndices = append(s.buf.allIndices, indices...)
	s.buf.norms = append(s.buf.norms, norm)
	s.buf.ids = append(s.buf.ids, id)
	s.buf.data = append(s.buf.data, dataJSON)
	s.buf.contents = append(s.buf.contents, content)
	return nil
}

// Flush writes the store to disk atomically (temp file -> rename).
// Only valid in write mode.
func (s *Store) Flush() error {
	if s.mode != modeWrite {
		return fmt.Errorf("tqdb: Flush called on read-only store")
	}

	data := encodeFile(s.config, s.workDim, s.buf)

	tmpPath := s.path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("tqdb: write temp file: %w", err)
	}

	// Sync to ensure data is on disk before rename.
	if f, err := os.Open(tmpPath); err == nil {
		_ = f.Sync()
		_ = f.Close()
	}

	if err := os.Rename(tmpPath, s.path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("tqdb: rename: %w", err)
	}

	return nil
}

// Len returns the number of vectors in the store.
func (s *Store) Len() int {
	if s.mode == modeWrite {
		return len(s.buf.norms)
	}
	return s.numVecs
}

// Close flushes (if write mode) and releases all resources.
func (s *Store) Close() error {
	if s.mode == modeWrite {
		if err := s.Flush(); err != nil {
			return err
		}
		return nil
	}
	// Read mode: release mmap or free data.
	if s.release != nil {
		return s.release()
	}
	return nil
}

// Open opens a .tq file for reading. The indices section is
// memory-mapped for zero-copy search. Falls back to bulk read
// if mmap is unavailable.
func Open(path string) (*Store, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tqdb: open: %w", err)
	}
	defer f.Close() //nolint:errcheck

	// mmap (or fallback to bulk read).
	data, release, err := mapFile(f)
	if err != nil {
		return nil, fmt.Errorf("tqdb: map file: %w", err)
	}

	if int64(len(data)) < fileHeaderSize {
		_ = release()
		return nil, fmt.Errorf("tqdb: file too small (%d bytes)", len(data))
	}

	// Decode header.
	hdr, err := decodeHeader(data[:fileHeaderSize])
	if err != nil {
		_ = release()
		return nil, err
	}

	numVecs := int(hdr.NumVecs)
	workDim := int(hdr.WorkDim)
	bits := int(hdr.Bits)
	if bits <= 0 || bits > 8 {
		bits = 8
	}

	// Validate section bounds.
	packedRowSize := codec.PackedSize(workDim, bits)
	indicesEnd := fileHeaderSize + numVecs*packedRowSize
	normsEnd := int(hdr.NormsOff) + numVecs*4
	if indicesEnd > len(data) || normsEnd > len(data) {
		_ = release()
		return nil, fmt.Errorf("tqdb: file truncated (size %d, need at least %d)", len(data), normsEnd)
	}

	// Reconstruct quantizer from header config.
	cfg := tqdb.StoreConfig{
		Dim:         int(hdr.Dim),
		Bits:        bits,
		Rotation:    tqdb.RotationType(hdr.Rotation),
		Seed:        hdr.Seed,
		UseExactPDF: hdr.UseExact != 0,
	}
	q, err := quantize.NewMSE(cfg.ToConfig())
	if err != nil {
		_ = release()
		return nil, fmt.Errorf("tqdb: reconstruct quantizer: %w", err)
	}

	// Unpack indices from bit-packed storage to uint8 for search.
	var allIndices []byte
	if bits == 8 {
		// Zero-copy from mmap.
		allIndices = data[fileHeaderSize:indicesEnd]
	} else {
		// Unpack to uint8 for the search loop.
		allIndices = make([]byte, numVecs*workDim)
		for i := range numVecs {
			packed := data[fileHeaderSize+i*packedRowSize : fileHeaderSize+i*packedRowSize+packedRowSize]
			codec.UnpackIndicesTo(packed, allIndices[i*workDim:i*workDim+workDim], workDim, bits)
		}
	}

	// Decode norms into []float32 (for dequantization; not used in search ranking).
	norms := decodeFloat32s(data[hdr.NormsOff:], numVecs)

	st := &Store{
		path:       path,
		config:     cfg,
		quantizer:  q,
		mode:       modeRead,
		workDim:    workDim,
		data:       data,
		release:    release,
		allIndices: allIndices,
		norms:      norms,
		numVecs:    numVecs,
		header:     hdr,
	}

	// Load HNSW graph if present.
	if hdr.GraphOff > 0 && int(hdr.GraphOff) < len(data) {
		h, err := UnmarshalHNSW(data[hdr.GraphOff:])
		if err == nil {
			st.hnswIdx = h
			// Precompute decoded float32 vectors for NEON search.
			centroids32 := q.Codebook().Centroids32
			decoded := make([]float32, numVecs*workDim)
			for i := range numVecs {
				off := i * workDim
				for j := range workDim {
					decoded[off+j] = centroids32[allIndices[off+j]]
				}
			}
			st.hnswVecs = decoded
		}
	}

	return st, nil
}

// ensureIDsLoaded lazily parses the IDs, data, and contents sections.
// Safe for concurrent calls via sync.Once.
func (s *Store) ensureIDsLoaded() {
	s.idsOnce.Do(func() {
		s.ids = decodeIDs(s.data[s.header.IDsOff:], s.numVecs)
		s.dataRaw = decodeDataRaw(s.data[s.header.DataOff:], s.numVecs)
		s.contents = decodeContents(s.data[s.header.ContentsOff:], s.numVecs)
	})
}

// ensureIVF builds the IVF index lazily on first search (if >= 100 vectors).
func (s *Store) ensureIVF() {
	s.ivfOnce.Do(func() {
		if s.numVecs < 100 {
			return
		}
		numPartitions := int(math.Sqrt(float64(s.numVecs)))
		if numPartitions < 4 {
			numPartitions = 4
		}
		nProbe := int(math.Sqrt(float64(numPartitions)))
		if nProbe < 1 {
			nProbe = 1
		}
		noDeleted := make([]bool, s.numVecs) // Store has no deletes
		s.ivfIdx = buildIVF(
			s.allIndices,
			s.quantizer.Codebook().Centroids,
			s.workDim,
			s.numVecs,
			numPartitions,
			nProbe,
			noDeleted,
		)
	})
}

// ensureDataCached parses all JSON data blobs once for efficient filtered searches.
func (s *Store) ensureDataCached() {
	s.dataCacheOnce.Do(func() {
		s.ensureIDsLoaded()
		s.dataCache = make([]map[string]any, s.numVecs)
		for i, raw := range s.dataRaw {
			if raw != nil {
				var m map[string]any
				_ = json.Unmarshal(raw, &m)
				s.dataCache[i] = m
			}
		}
	})
}

func (s *Store) dataAt(i int) map[string]any {
	if s.dataRaw[i] == nil {
		return nil
	}
	var m map[string]any
	_ = json.Unmarshal(s.dataRaw[i], &m)
	return m
}

// Search finds the top-k most similar vectors to the query.
func (s *Store) Search(query []float64, topK int) []tqdb.Result {
	return s.SearchWithOptions(query, tqdb.SearchOptions{TopK: topK})
}

// SearchWithOptions performs a vector similarity search with VS2-aligned options.
func (s *Store) SearchWithOptions(query []float64, opts tqdb.SearchOptions) []tqdb.Result {
	return s.searchInternal(query, opts)
}

// Query performs filter-only retrieval (no vector similarity scoring).
func (s *Store) Query(opts tqdb.QueryOptions) []tqdb.Result {
	if s.mode != modeRead {
		return nil
	}
	if opts.PageSize <= 0 {
		opts.PageSize = 100
	}

	s.ensureDataCached()

	var results []tqdb.Result
	for i := range s.numVecs {
		if opts.Filter != nil && !opts.Filter.Match(s.dataCache[i]) {
			continue
		}
		results = append(results, tqdb.Result{
			ID:   s.ids[i],
			Data: s.dataCache[i],
		})
		if len(results) >= opts.PageSize {
			break
		}
	}
	return results
}

// SearchFloat32 is a convenience wrapper for float32 queries.
func (s *Store) SearchFloat32(query []float32, topK int) []tqdb.Result {
	return s.Search(mathutil.Float32ToFloat64(query), topK)
}

func (s *Store) searchInternal(query []float64, opts tqdb.SearchOptions) []tqdb.Result {
	if s.mode != modeRead {
		return nil
	}

	if opts.TopK <= 0 {
		opts.TopK = 10
	}

	n := s.numVecs
	if n == 0 {
		return nil
	}

	// Build IVF lazily on first search.
	s.ensureIVF()

	// If filter is set, parse all data blobs once up front.
	if opts.Filter != nil {
		s.ensureDataCached()
	}

	d := s.workDim
	centroids32 := s.quantizer.Codebook().Centroids32

	// Normalize query and rotate.
	queryNorm := mathutil.Norm(query)
	if queryNorm < 1e-15 {
		return nil
	}
	invQN := 1.0 / queryNorm

	unitQuery := s.quantizer.GetBuf()
	for i, v := range query {
		unitQuery[i] = v * invQN
	}
	queryRotated := s.quantizer.GetBuf()
	s.quantizer.Rotation().Rotate(queryRotated, unitQuery[:s.quantizer.GetConfig().Dim])
	s.quantizer.PutBuf(unitQuery)

	// Convert rotated query to float32 for scoring.
	qr32 := make([]float32, d)
	for i := range d {
		qr32[i] = float32(queryRotated[i])
	}

	// Determine effective K (account for offset and rescore).
	effectiveK := opts.TopK + opts.Offset
	if opts.Rescore > 0 && opts.Rescore > effectiveK {
		effectiveK = opts.Rescore
	}

	type scored struct {
		idx   int
		score float64
	}
	topBuf := make([]scored, 0, effectiveK+1)
	minScore := -math.MaxFloat64

	allIdx := s.allIndices

	insertTopK := func(i int, dot float64) {
		if opts.MinScore > 0 && dot < opts.MinScore {
			return
		}
		if len(topBuf) >= effectiveK && dot <= minScore {
			return
		}

		// Manual binary search (avoids sort.Search closure alloc).
		lo, hi := 0, len(topBuf)
		for lo < hi {
			mid := int(uint(lo+hi) >> 1)
			if topBuf[mid].score < dot {
				hi = mid
			} else {
				lo = mid + 1
			}
		}
		topBuf = append(topBuf, scored{})
		copy(topBuf[lo+1:], topBuf[lo:])
		topBuf[lo] = scored{idx: i, score: dot}

		if len(topBuf) > effectiveK {
			topBuf = topBuf[:effectiveK]
		}
		if len(topBuf) == effectiveK {
			minScore = topBuf[effectiveK-1].score
		}
	}

	scoreAndInsert := func(i int) {
		if opts.Filter != nil && !opts.Filter.Match(s.dataCache[i]) {
			return
		}

		indices := allIdx[i*d : i*d+d : i*d+d]
		var dot0, dot1 float32
		j := 0
		for ; j <= d-8; j += 8 {
			dot0 += qr32[j]*centroids32[indices[j]] +
				qr32[j+1]*centroids32[indices[j+1]] +
				qr32[j+2]*centroids32[indices[j+2]] +
				qr32[j+3]*centroids32[indices[j+3]]
			dot1 += qr32[j+4]*centroids32[indices[j+4]] +
				qr32[j+5]*centroids32[indices[j+5]] +
				qr32[j+6]*centroids32[indices[j+6]] +
				qr32[j+7]*centroids32[indices[j+7]]
		}
		for ; j < d; j++ {
			dot0 += qr32[j] * centroids32[indices[j]]
		}
		insertTopK(i, float64(dot0+dot1))
	}

	if s.ivfIdx != nil {
		topPartitions := s.ivfIdx.findNearestPartitions(queryRotated, s.ivfIdx.scoreBuf)
		s.ivfIdx.forEachCandidate(topPartitions, func(i int) {
			scoreAndInsert(i)
		})
	} else {
		for i := range n {
			scoreAndInsert(i)
		}
	}

	s.quantizer.PutBuf(queryRotated)

	// Rescore: dequantize top candidates and re-rank with exact cosine similarity.
	// Single reusable buffer for all dequantizations.
	if opts.Rescore > 0 && len(topBuf) > 0 {
		origDim := s.quantizer.GetConfig().Dim
		bits := s.quantizer.GetConfig().Bits
		recon := make([]float64, origDim)
		cv := tqdb.CompressedVector{Dim: origDim, Bits: bits}
		for k := range topBuf {
			cv.Norm = s.norms[topBuf[k].idx]
			cv.Indices = allIdx[topBuf[k].idx*d : topBuf[k].idx*d+d]
			s.quantizer.DequantizeTo(recon, &cv)
			topBuf[k].score = mathutil.CosineSimilarity(query, recon)
		}
		sort.Slice(topBuf, func(i, j int) bool {
			return topBuf[i].score > topBuf[j].score
		})
	}

	// Apply offset.
	if opts.Offset > 0 && opts.Offset < len(topBuf) {
		topBuf = topBuf[opts.Offset:]
	} else if opts.Offset >= len(topBuf) {
		topBuf = nil
	}

	// Trim to TopK.
	if len(topBuf) > opts.TopK {
		topBuf = topBuf[:opts.TopK]
	}

	// Lazy-load IDs for results.
	s.ensureIDsLoaded()

	results := make([]tqdb.Result, len(topBuf))
	for i, sc := range topBuf {
		results[i] = tqdb.Result{
			ID:      s.ids[sc.idx],
			Score:   sc.score,
			Content: s.contents[sc.idx],
			Data:    s.dataAt(sc.idx),
		}
	}

	return results
}

// ForEachCompressed iterates all stored vectors, calling fn with the raw data.
// Used for loading a Store's contents into a Collection.
// The indices slice is a view into mmap'd storage and must not be modified.
func (s *Store) ForEachCompressed(fn func(id string, indices []uint8, norm float32, content string, data map[string]any)) {
	if s.mode != modeRead {
		return
	}
	s.ensureIDsLoaded()
	s.ensureDataCached()
	d := s.workDim
	for i := range s.numVecs {
		fn(s.ids[i], s.allIndices[i*d:i*d+d], s.norms[i], s.contents[i], s.dataCache[i])
	}
}

// SetGraph attaches serialized HNSW graph data to be written on Flush.
func (s *Store) SetGraph(data []byte) {
	if s.mode == modeWrite {
		s.buf.graphData = data
	}
}

// AddRaw adds pre-compressed vector data without re-quantizing.
// Used for persisting a Collection's state to a Store file.
// Only valid in write mode (created via Create).
func (s *Store) AddRaw(id string, indices []uint8, norm float32, content string, data map[string]any) error {
	if s.mode != modeWrite {
		return fmt.Errorf("tqdb: AddRaw called on read-only store")
	}
	return s.addBuf(id, content, indices, norm, data)
}

// Info returns statistics about the store.
func (s *Store) Info() tqdb.StoreInfo {
	var fileSize int64
	if s.mode == modeRead {
		fileSize = int64(len(s.data))
	}

	numVecs := s.Len()
	indexBytes := int64(numVecs) * int64(s.workDim)

	// Compression vs float32: float32 size = numVecs * dim * 4
	float32Size := int64(numVecs) * int64(s.config.Dim) * 4
	compression := 0.0
	if fileSize > 0 {
		compression = float64(float32Size) / float64(fileSize)
	}

	return tqdb.StoreInfo{
		Path:        s.path,
		Dim:         s.config.Dim,
		WorkDim:     s.workDim,
		Bits:        s.config.Bits,
		Rotation:    s.config.Rotation,
		Seed:        s.config.Seed,
		NumVecs:     numVecs,
		FileSize:    fileSize,
		IndexBytes:  indexBytes,
		Compression: compression,
	}
}

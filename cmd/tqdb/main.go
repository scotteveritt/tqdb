package main

import (
	"compress/gzip"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/convert"
	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/model"
	"github.com/scotteveritt/tqdb/store"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "create":
		runCreate(os.Args[2:])
	case "add":
		runAdd(os.Args[2:])
	case "search":
		runSearch(os.Args[2:])
	case "info":
		runInfo(os.Args[2:])
	case "import":
		runImport(os.Args[2:])
	case "compress":
		runCompress(os.Args[2:])
	case "convert":
		runConvert(os.Args[2:])
	case "inspect":
		runInspect(os.Args[2:])
	case "version":
		fmt.Printf("tqdb %s\n", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`tqdb — SQLite for quantized vectors

Usage:
  tqdb create <path> --dim N [--bits B] [--rotation hadamard|qr] [--seed S]
  tqdb add <path> [--from file.jsonl]          reads JSONL from stdin by default
  tqdb import <out.tq> --format chromem --dir <path>   import from chromem-go data dir
  tqdb import <out.tq> --format jsonl [--from file]    import from JSONL (default: stdin)
  tqdb convert <model-dir> -o <out.gguf>                convert to GGUF (Hadamard pre-conditioned)
  tqdb compress <model.safetensors> -o <out.tqm>       quantize model weights (TQ format)
  tqdb inspect <model.tqm>                             inspect quantized model
  tqdb search <path> --query "0.1,..."         [--top K] [--format text|json]
  tqdb info <path>
  tqdb version

`)
}

// extractPath finds the first non-flag argument (the .tq path) from args,
// and returns it plus the remaining flag-only args for flag.Parse.
func extractPath(args []string) (string, []string) {
	var path string
	var flagArgs []string
	for i := 0; i < len(args); i++ {
		if strings.HasPrefix(args[i], "-") {
			flagArgs = append(flagArgs, args[i])
			// If the next arg is the flag's value (not another flag), include it.
			if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
				flagArgs = append(flagArgs, args[i+1])
				i++
			}
		} else if path == "" {
			path = args[i]
		}
	}
	return path, flagArgs
}

// --- create ---

func runCreate(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("create", flag.ExitOnError)
	dim := fs.Int("dim", 0, "embedding dimension (required)")
	bits := fs.Int("bits", 4, "bits per coordinate (1-8)")
	rotation := fs.String("rotation", "hadamard", "rotation type: qr or hadamard")
	seed := fs.Uint64("seed", 42, "rotation seed")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: path argument required")
		fmt.Fprintln(os.Stderr, "usage: tqdb create <path> --dim N")
		os.Exit(1)
	}

	if *dim <= 0 {
		fmt.Fprintln(os.Stderr, "error: --dim is required and must be > 0")
		os.Exit(1)
	}

	var rot tqdb.RotationType
	switch strings.ToLower(*rotation) {
	case "hadamard", "h":
		rot = tqdb.RotationHadamard
	case "qr", "q":
		rot = tqdb.RotationQR
	default:
		fmt.Fprintf(os.Stderr, "error: unknown rotation type %q (use qr or hadamard)\n", *rotation)
		os.Exit(1)
	}

	s, err := store.Create(path, tqdb.StoreConfig{
		Dim:      *dim,
		Bits:     *bits,
		Rotation: rot,
		Seed:     *seed,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if err := s.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("created %s (dim=%d, bits=%d, rotation=%s)\n", path, *dim, *bits, *rotation)
}

// --- add ---

type jsonlRecord struct {
	ID       string         `json:"id"`
	Vector   []float64      `json:"vector"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func runAdd(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("add", flag.ExitOnError)
	from := fs.String("from", "", "JSONL file path (default: stdin)")
	dim := fs.Int("dim", 0, "embedding dimension (required if store doesn't exist)")
	bits := fs.Int("bits", 4, "bits per coordinate")
	rotation := fs.String("rotation", "hadamard", "rotation type")
	seed := fs.Uint64("seed", 42, "rotation seed")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: path argument required")
		fmt.Fprintln(os.Stderr, "usage: tqdb add <path> [--from file.jsonl]")
		os.Exit(1)
	}

	// Try to open existing store first.
	var s *store.Store
	var err error

	if _, statErr := os.Stat(path); statErr == nil {
		// File exists — open, load existing data, add new.
		existing, openErr := store.Open(path)
		if openErr != nil {
			fmt.Fprintf(os.Stderr, "error opening existing store: %v\n", openErr)
			os.Exit(1)
		}
		info := existing.Info()
		_ = existing.Close()

		// Re-create with same config to add vectors.
		s, err = store.Create(path, tqdb.StoreConfig{
			Dim:      info.Dim,
			Bits:     info.Bits,
			Rotation: info.Rotation,
			Seed:     info.Seed,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		// TODO: reload existing vectors. For now, this overwrites.
		// In v2, support true append.
	} else {
		// New store — need dim.
		if *dim <= 0 {
			fmt.Fprintln(os.Stderr, "error: --dim required when creating new store")
			os.Exit(1)
		}
		var rot tqdb.RotationType
		switch strings.ToLower(*rotation) {
		case "hadamard", "h":
			rot = tqdb.RotationHadamard
		default:
			rot = tqdb.RotationQR
		}
		s, err = store.Create(path, tqdb.StoreConfig{
			Dim:      *dim,
			Bits:     *bits,
			Rotation: rot,
			Seed:     *seed,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}

	// Read JSONL.
	var input *os.File
	if *from != "" {
		input, err = os.Open(*from)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		defer input.Close() //nolint:errcheck
	} else {
		input = os.Stdin
	}

	dec := json.NewDecoder(input)
	count := 0
	for dec.More() {
		var rec jsonlRecord
		if err := dec.Decode(&rec); err != nil {
			_ = input.Close()
			fmt.Fprintf(os.Stderr, "error at record %d: %v\n", count+1, err)
			os.Exit(1) //nolint:gocritic // CLI exit after explicit close
		}
		if rec.ID == "" {
			rec.ID = fmt.Sprintf("vec-%d", count)
		}
		if err := s.Add(rec.ID, rec.Vector, rec.Metadata); err != nil {
			fmt.Fprintf(os.Stderr, "error adding %s: %v\n", rec.ID, err)
			os.Exit(1)
		}
		count++
	}

	if err := s.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("added %d vectors to %s\n", count, path)
}

// --- search ---

func runSearch(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("search", flag.ExitOnError)
	queryStr := fs.String("query", "", "query vector as comma-separated floats (required)")
	topK := fs.Int("top", 10, "number of results")
	format := fs.String("format", "text", "output format: text or json")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: path argument required")
		os.Exit(1)
	}

	if *queryStr == "" {
		fmt.Fprintln(os.Stderr, "error: --query is required")
		os.Exit(1)
	}

	// Parse query vector.
	query, err := parseVector(*queryStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error parsing query: %v\n", err)
		os.Exit(1)
	}

	s, err := store.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer s.Close() //nolint:errcheck

	results := s.Search(query, *topK)

	switch *format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(results)
	default:
		for i, r := range results {
			var dataStr strings.Builder
			for k, v := range r.Data {
				dataStr.WriteString(fmt.Sprintf("  %s=%v", k, v))
			}
			fmt.Printf("%d. %-20s %.4f%s\n", i+1, r.ID, r.Score, dataStr.String())
		}
	}
}

// --- info ---

func runInfo(args []string) {
	path, _ := extractPath(args)
	if path == "" {
		fmt.Fprintln(os.Stderr, "error: path argument required")
		os.Exit(1)
	}

	s, err := store.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer s.Close() //nolint:errcheck

	info := s.Info()

	rotName := "qr"
	if info.Rotation == tqdb.RotationHadamard {
		rotName = "hadamard"
	}

	fmt.Printf("tqdb store: %s\n", info.Path)
	fmt.Printf("  vectors:     %d\n", info.NumVecs)
	fmt.Printf("  dimension:   %d", info.Dim)
	if info.WorkDim != info.Dim {
		fmt.Printf(" (work: %d)", info.WorkDim)
	}
	fmt.Println()
	fmt.Printf("  bits:        %d\n", info.Bits)
	fmt.Printf("  rotation:    %s (seed: %d)\n", rotName, info.Seed)
	fmt.Printf("  file size:   %s\n", formatBytes(info.FileSize))
	fmt.Printf("  index size:  %s", formatBytes(info.IndexBytes))
	if info.FileSize > 0 {
		fmt.Printf(" (%.1f%%)", float64(info.IndexBytes)/float64(info.FileSize)*100)
	}
	fmt.Println()
	fmt.Printf("  compression: %.2fx vs float32\n", info.Compression)
}

// --- helpers ---

func parseVector(s string) ([]float64, error) {
	parts := strings.Split(s, ",")
	vec := make([]float64, len(parts))
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("element %d: %w", i, err)
		}
		vec[i] = v
	}
	return vec, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

func formatBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// --- import ---

// chromemDocument matches chromem-go's Document struct for gob decoding.
type chromemDocument struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
	Content   string
}

func runImport(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("import", flag.ExitOnError)
	format := fs.String("format", "chromem", "import format: chromem, jsonl")
	dir := fs.String("dir", "", "source directory (for chromem format)")
	from := fs.String("from", "", "source file (for jsonl format; default: stdin)")
	bits := fs.Int("bits", 4, "bits per coordinate")
	dim := fs.Int("dim", 0, "embedding dimension (auto-detected for chromem)")
	rotation := fs.String("rotation", "hadamard", "rotation type: qr or hadamard")
	seed := fs.Uint64("seed", 42, "rotation seed")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: output .tq path required")
		fmt.Fprintln(os.Stderr, "usage: tqdb import <out.tq> --format chromem --dir <path>")
		os.Exit(1)
	}

	var rot tqdb.RotationType
	switch strings.ToLower(*rotation) {
	case "hadamard", "h":
		rot = tqdb.RotationHadamard
	default:
		rot = tqdb.RotationQR
	}

	switch strings.ToLower(*format) {
	case "chromem":
		importChromem(path, *dir, *bits, rot, *seed)
	case "jsonl":
		importJSONL(path, *from, *dim, *bits, rot, *seed)
	default:
		fmt.Fprintf(os.Stderr, "error: unknown format %q (supported: chromem, jsonl)\n", *format)
		os.Exit(1)
	}
}

func importChromem(outPath, dir string, bits int, rot tqdb.RotationType, seed uint64) {
	if dir == "" {
		fmt.Fprintln(os.Stderr, "error: --dir is required for chromem format")
		os.Exit(1)
	}

	// Find all .gob.gz files recursively.
	var gobFiles []string
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(p, ".gob.gz") {
			gobFiles = append(gobFiles, p)
		}
		return nil
	})

	if len(gobFiles) == 0 {
		fmt.Fprintf(os.Stderr, "error: no .gob.gz files found in %s\n", dir)
		os.Exit(1)
	}

	fmt.Printf("found %d .gob.gz files in %s\n", len(gobFiles), dir)

	// Detect dimension from first file.
	firstDoc, err := decodeGobGz(gobFiles[0])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading first file: %v\n", err)
		os.Exit(1)
	}
	dim := len(firstDoc.Embedding)
	fmt.Printf("detected dimension: %d\n", dim)

	s, err := store.Create(outPath, tqdb.StoreConfig{
		Dim:      dim,
		Bits:     bits,
		Rotation: rot,
		Seed:     seed,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating store: %v\n", err)
		os.Exit(1)
	}

	start := time.Now()
	imported := 0
	skipped := 0
	lastReport := time.Now()

	for _, gf := range gobFiles {
		doc, err := decodeGobGz(gf)
		if err != nil {
			skipped++
			continue
		}
		if len(doc.Embedding) != dim {
			skipped++
			continue
		}

		vec := make([]float64, dim)
		for j, v := range doc.Embedding {
			vec[j] = float64(v)
		}

		// Convert chromem map[string]string to map[string]any.
		var data map[string]any
		if len(doc.Metadata) > 0 {
			data = make(map[string]any, len(doc.Metadata))
			for k, v := range doc.Metadata {
				data[k] = v
			}
		}
		if err := s.Add(doc.ID, vec, data); err != nil {
			skipped++
			continue
		}
		imported++

		if time.Since(lastReport) > 2*time.Second {
			elapsed := time.Since(start)
			rate := float64(imported) / elapsed.Seconds()
			fmt.Printf("  %d/%d imported (%.0f vec/s, %d skipped)\n",
				imported, len(gobFiles), rate, skipped)
			lastReport = time.Now()
		}
	}

	fmt.Printf("flushing %d vectors to %s...\n", imported, outPath)
	if err := s.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	printImportStats(imported, skipped, dim, outPath, start)
}

func importJSONL(outPath, fromFile string, dim, bits int, rot tqdb.RotationType, seed uint64) {
	var input *os.File
	var err error
	if fromFile != "" {
		input, err = os.Open(fromFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		defer input.Close() //nolint:errcheck
	} else {
		input = os.Stdin
	}

	// For JSONL, we need dim upfront (or detect from first record).
	dec := json.NewDecoder(input)
	var records []jsonlRecord
	for dec.More() {
		var rec jsonlRecord
		if err := dec.Decode(&rec); err != nil {
			fmt.Fprintf(os.Stderr, "error decoding JSONL: %v\n", err)
			os.Exit(1) //nolint:gocritic // CLI exit
		}
		if dim == 0 && len(rec.Vector) > 0 {
			dim = len(rec.Vector)
			fmt.Printf("detected dimension: %d\n", dim)
		}
		records = append(records, rec)
	}

	if dim == 0 {
		fmt.Fprintln(os.Stderr, "error: could not detect dimension (no vectors or --dim not set)")
		os.Exit(1)
	}

	s, err := store.Create(outPath, tqdb.StoreConfig{
		Dim:      dim,
		Bits:     bits,
		Rotation: rot,
		Seed:     seed,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating store: %v\n", err)
		os.Exit(1)
	}

	start := time.Now()
	imported := 0
	skipped := 0
	for _, rec := range records {
		if len(rec.Vector) != dim {
			skipped++
			continue
		}
		id := rec.ID
		if id == "" {
			id = fmt.Sprintf("vec-%d", imported)
		}
		if err := s.Add(id, rec.Vector, rec.Metadata); err != nil {
			skipped++
			continue
		}
		imported++
	}

	fmt.Printf("flushing %d vectors to %s...\n", imported, outPath)
	if err := s.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	printImportStats(imported, skipped, dim, outPath, start)
}

func printImportStats(imported, skipped, dim int, outPath string, start time.Time) {
	elapsed := time.Since(start)
	info, _ := os.Stat(outPath)
	fileSize := int64(0)
	if info != nil {
		fileSize = info.Size()
	}

	origSize := int64(imported) * int64(dim) * 4
	ratio := 0.0
	if fileSize > 0 {
		ratio = float64(origSize) / float64(fileSize)
	}

	fmt.Printf("\nimport complete:\n")
	fmt.Printf("  vectors:     %d (%d skipped)\n", imported, skipped)
	fmt.Printf("  dimension:   %d\n", dim)
	fmt.Printf("  elapsed:     %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  throughput:  %.0f vec/s\n", float64(imported)/elapsed.Seconds())
	fmt.Printf("  file size:   %s\n", formatBytes(fileSize))
	fmt.Printf("  orig float32:%s\n", formatBytes(origSize))
	fmt.Printf("  compression: %.2fx\n", ratio)
}

func decodeGobGz(path string) (*chromemDocument, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close() //nolint:errcheck

	var doc chromemDocument
	if err := gob.NewDecoder(gz).Decode(&doc); err != nil {
		return nil, err
	}
	return &doc, nil
}

// --- compress ---

func runCompress(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("compress", flag.ExitOnError)
	output := fs.String("o", "", "output .tqm file path (required)")
	bits := fs.Int("bits", 4, "bits per coordinate")
	rotation := fs.String("rotation", "hadamard", "rotation type: qr or hadamard")
	seed := fs.Uint64("seed", 42, "rotation seed")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: input .safetensors path required")
		fmt.Fprintln(os.Stderr, "usage: tqdb compress <model.safetensors> -o <out.tqm>")
		os.Exit(1)
	}
	if *output == "" {
		fmt.Fprintln(os.Stderr, "error: -o output path required")
		os.Exit(1)
	}

	var rot tqdb.RotationType
	switch strings.ToLower(*rotation) {
	case "hadamard", "h":
		rot = tqdb.RotationHadamard
	default:
		rot = tqdb.RotationQR
	}

	fmt.Printf("Opening %s...\n", path)
	sf, err := tqdb.OpenSafeTensors(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer sf.Close() //nolint:errcheck

	fmt.Printf("  %d tensors found\n\n", sf.NumTensors())

	start := time.Now()
	header, err := model.Compress(sf, *output, tqdb.ModelConfig{
		Bits:     *bits,
		Rotation: rot,
		Seed:     *seed,
	}, func(tensorName string, tensorIdx, totalTensors, rows, totalRows int) {
		fmt.Printf("\r  [%d/%d] %-60s %d/%d rows",
			tensorIdx+1, totalTensors, truncate(tensorName, 60), rows, totalRows)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1) //nolint:gocritic // CLI exit
	}

	elapsed := time.Since(start)
	info, _ := os.Stat(*output)
	srcInfo, _ := os.Stat(path)

	fmt.Printf("\n\nCompression complete:\n")
	fmt.Printf("  tensors:     %d\n", len(header.Tensors))
	fmt.Printf("  elapsed:     %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  source:      %s\n", formatBytes(srcInfo.Size()))
	fmt.Printf("  compressed:  %s\n", formatBytes(info.Size()))
	fmt.Printf("  ratio:       %.2fx\n", float64(srcInfo.Size())/float64(info.Size()))

	// Print per-tensor quality summary.
	fmt.Printf("\n  %-50s %10s %10s\n", "tensor", "shape", "cos_sim")
	names := make([]string, 0, len(header.Tensors))
	for name := range header.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		t := header.Tensors[name]
		shapeStr := fmt.Sprintf("%v", t.Shape)
		fmt.Printf("  %-50s %10s %9.4f%%\n",
			truncate(name, 50), shapeStr, t.AvgCosSim*100)
	}
}

// --- inspect ---

func runInspect(args []string) {
	path, _ := extractPath(args)
	if path == "" {
		fmt.Fprintln(os.Stderr, "error: .tqm path required")
		os.Exit(1)
	}

	header, err := model.OpenTQM(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fileInfo, _ := os.Stat(path)

	fmt.Printf("tqdb model: %s\n", path)
	fmt.Printf("  version:    %d\n", header.Version)
	fmt.Printf("  bits:       %d\n", header.Bits)
	fmt.Printf("  rotation:   %s (seed: %d)\n", header.Rotation, header.Seed)
	fmt.Printf("  tensors:    %d\n", len(header.Tensors))
	if fileInfo != nil {
		fmt.Printf("  file size:  %s\n", formatBytes(fileInfo.Size()))
	}

	fmt.Printf("\n  %-50s %15s %8s %8s %10s\n", "tensor", "shape", "dtype", "workDim", "cos_sim")
	fmt.Printf("  %-50s %15s %8s %8s %10s\n", "---", "---", "---", "---", "---")

	names := make([]string, 0, len(header.Tensors))
	for name := range header.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		t := header.Tensors[name]
		shapeStr := fmt.Sprintf("%v", t.Shape)
		fmt.Printf("  %-50s %15s %8s %8d %9.4f%%\n",
			truncate(name, 50), shapeStr, t.OrigDType, t.WorkDim, t.AvgCosSim*100)
	}
}

// --- convert ---

func runConvert(args []string) {
	path, flagArgs := extractPath(args)

	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	output := fs.String("o", "", "output GGUF file path (required)")
	_ = fs.Parse(flagArgs)

	if path == "" {
		fmt.Fprintln(os.Stderr, "error: model directory path required")
		fmt.Fprintln(os.Stderr, "usage: tqdb convert <model-dir> -o <output.gguf>")
		os.Exit(1)
	}
	if *output == "" {
		fmt.Fprintln(os.Stderr, "error: -o output path required")
		os.Exit(1)
	}

	fmt.Printf("Converting %s → %s\n", path, *output)

	fsys := os.DirFS(path)

	f, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error creating output: %v\n", err)
		os.Exit(1)
	}

	start := time.Now()
	if err := convert.ConvertModel(fsys, f); err != nil {
		_ = f.Close()
		_ = os.Remove(*output)
		fmt.Fprintf(os.Stderr, "error converting: %v\n", err)
		os.Exit(1)
	}

	info, _ := f.Stat()
	_ = f.Close()
	elapsed := time.Since(start)

	fmt.Printf("\nConversion complete:\n")
	fmt.Printf("  output:  %s\n", *output)
	fmt.Printf("  size:    %s\n", formatBytes(info.Size()))
	fmt.Printf("  elapsed: %s\n", elapsed.Round(time.Millisecond))
}

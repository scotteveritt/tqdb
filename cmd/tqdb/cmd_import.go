package main

import (
	"compress/gzip"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/embed"
	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newImportCmd() *cobra.Command {
	var from, fromChromem string
	var bits, dim int
	var doEmbed, force bool

	cmd := &cobra.Command{
		Use:   "import [path.tq]",
		Short: "Import data into a .tq file",
		Long: `Create a .tq file from JSONL data, chromem-go databases, or text files.

Examples:
  tqdb import --from embeddings.jsonl                  # JSONL with vectors
  tqdb import --from docs.jsonl --embed                # JSONL text, auto-embed
  tqdb import --from-chromem ~/.local/share/myapp/db/  # chromem-go migration
  cat data.jsonl | tqdb import                         # from stdin
  tqdb import index.tq --from data.jsonl               # explicit output path`,
		RunE: func(cmd *cobra.Command, args []string) error {
			quiet, _ := cmd.Flags().GetBool("quiet")
			cfgFile, _ := cmd.Flags().GetString("config")
			cfg := loadConfig(cfgFile)
			applyEnvOverrides(&cfg)

			// Resolve output path.
			var outPath string
			if len(args) > 0 {
				outPath = args[0]
			} else {
				p, err := resolveStorePath("")
				if err != nil {
					return fmt.Errorf("no output path specified and no workspace found (run 'tqdb init' or specify a path)")
				}
				outPath = p
			}

			if _, err := os.Stat(outPath); err == nil && !force {
				return fmt.Errorf("%s already exists (use --force to overwrite)", outPath)
			}

			// Determine bits.
			b := bits
			if b <= 0 {
				b = cfg.Defaults.Bits
			}
			if b <= 0 {
				b = 8
			}

			if fromChromem != "" {
				return importChromem(outPath, fromChromem, b, dim, quiet)
			}
			return importJSONL(cmd, outPath, from, b, dim, doEmbed, &cfg, quiet)
		},
	}

	cmd.Flags().StringVar(&from, "from", "", "input JSONL file (default: stdin)")
	cmd.Flags().StringVar(&fromChromem, "from-chromem", "", "chromem-go database directory")
	cmd.Flags().IntVar(&bits, "bits", 0, "quantization bits (default: from config or 8)")
	cmd.Flags().IntVar(&dim, "dim", 0, "embedding dimension (auto-detected if omitted)")
	cmd.Flags().BoolVar(&doEmbed, "embed", false, "embed text content via configured provider")
	cmd.Flags().BoolVar(&force, "force", false, "overwrite existing .tq file")

	return cmd
}

type jsonlRecord struct {
	ID       string         `json:"id"`
	Vector   []float64      `json:"vector"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func importJSONL(_ *cobra.Command, outPath, from string, bits, dim int, doEmbed bool, cfg *Config, quiet bool) error {
	var input *os.File
	var err error
	if from != "" {
		input, err = os.Open(from)
		if err != nil {
			return err
		}
		defer input.Close()
	} else {
		input = os.Stdin
	}

	// Read all records (need dim before creating store).
	var records []jsonlRecord
	dec := json.NewDecoder(input)
	for dec.More() {
		var rec jsonlRecord
		if err := dec.Decode(&rec); err != nil {
			return fmt.Errorf("decoding JSONL at record %d: %w", len(records)+1, err)
		}
		records = append(records, rec)
	}

	if len(records) == 0 {
		return fmt.Errorf("no records found in input")
	}

	// If embedding, set up provider.
	var embedder embed.Provider
	if doEmbed {
		embedder, err = embed.New(cfg.Provider, cfg.Project, cfg.APIKey, cfg.Model, cfg.BaseURL)
		if err != nil {
			return fmt.Errorf("creating embedding provider: %w", err)
		}
	}

	// Detect dimension.
	if dim <= 0 {
		if len(records[0].Vector) > 0 {
			dim = len(records[0].Vector)
		} else if doEmbed && embedder != nil {
			// Embed first record to detect dimension.
			vec, err := embedder.Embed(cmd_context(), records[0].Content)
			if err != nil {
				return fmt.Errorf("embedding first record for dimension detection: %w", err)
			}
			dim = len(vec)
			records[0].Vector = vec
		}
	}
	if dim <= 0 {
		return fmt.Errorf("cannot detect dimension (set --dim or provide vectors in JSONL)")
	}

	if !quiet {
		fmt.Printf("importing %d records (dim=%d, bits=%d)...\n", len(records), dim, bits)
	}

	// Create store.
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	s, err := store.Create(outPath, tqdb.StoreConfig{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard,
	})
	if err != nil {
		return err
	}

	start := time.Now()
	var imported, skipped int
	lastReport := start

	for i, rec := range records {
		// Embed if needed.
		if doEmbed && len(rec.Vector) == 0 && embedder != nil {
			vec, err := embedder.Embed(cmd_context(), rec.Content)
			if err != nil {
				skipped++
				if !quiet {
					fmt.Fprintf(os.Stderr, "  skipping record %d: %v\n", i, err)
				}
				continue
			}
			rec.Vector = vec
		}

		if len(rec.Vector) != dim {
			skipped++
			continue
		}

		id := rec.ID
		if id == "" {
			id = fmt.Sprintf("vec-%d", imported)
		}

		if err := s.Add(tqdb.Document{
			ID: id, Content: rec.Content, Embedding: rec.Vector, Data: rec.Metadata,
		}); err != nil {
			skipped++
			continue
		}
		imported++

		if !quiet && time.Since(lastReport) > 2*time.Second {
			rate := float64(imported) / time.Since(start).Seconds()
			fmt.Printf("  %d/%d imported (%.0f/s)\n", imported, len(records), rate)
			lastReport = time.Now()
		}
	}

	if err := s.Close(); err != nil {
		return err
	}

	if !quiet {
		elapsed := time.Since(start)
		fi, _ := os.Stat(outPath)
		fmt.Printf("imported %d vectors to %s (%s, %s)\n",
			imported, outPath, formatSize(fi.Size()), elapsed.Round(time.Millisecond))
		if skipped > 0 {
			fmt.Printf("  %d records skipped\n", skipped)
		}
	}
	return nil
}

// chromemDoc matches chromem-go's internal Document struct.
type chromemDoc struct {
	ID        string
	Metadata  map[string]string
	Embedding []float32
	Content   string
}

func importChromem(outPath, dir string, bits, dim int, quiet bool) error {
	var gobFiles []string
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() && strings.HasSuffix(p, ".gob.gz") {
			gobFiles = append(gobFiles, p)
		}
		return nil
	})
	if len(gobFiles) == 0 {
		return fmt.Errorf("no .gob.gz files found in %s", dir)
	}

	// Detect dim from first file.
	if dim <= 0 {
		first, err := decodeChromemGob(gobFiles[0])
		if err != nil {
			return fmt.Errorf("reading first file: %w", err)
		}
		dim = len(first.Embedding)
	}
	if dim <= 0 {
		return fmt.Errorf("cannot detect dimension from chromem data")
	}

	if !quiet {
		fmt.Printf("found %d .gob.gz files (dim=%d, bits=%d)\n", len(gobFiles), dim, bits)
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}
	s, err := store.Create(outPath, tqdb.StoreConfig{
		Dim: dim, Bits: bits, Rotation: tqdb.RotationHadamard,
	})
	if err != nil {
		return err
	}

	start := time.Now()
	var imported, skipped int
	lastReport := start

	for _, gf := range gobFiles {
		doc, err := decodeChromemGob(gf)
		if err != nil || len(doc.Embedding) != dim {
			skipped++
			continue
		}

		vec := make([]float64, dim)
		for j, v := range doc.Embedding {
			vec[j] = float64(v)
		}
		data := make(map[string]any, len(doc.Metadata))
		for k, v := range doc.Metadata {
			data[k] = v
		}

		if err := s.Add(tqdb.Document{
			ID: doc.ID, Content: doc.Content, Embedding: vec, Data: data,
		}); err != nil {
			skipped++
			continue
		}
		imported++

		if !quiet && time.Since(lastReport) > 2*time.Second {
			rate := float64(imported) / time.Since(start).Seconds()
			fmt.Printf("  %d/%d imported (%.0f/s, %d skipped)\n", imported, len(gobFiles), rate, skipped)
			lastReport = time.Now()
		}
	}

	if err := s.Close(); err != nil {
		return err
	}

	if !quiet {
		elapsed := time.Since(start)
		fi, _ := os.Stat(outPath)
		fmt.Printf("imported %d vectors to %s (%s, %s)\n",
			imported, outPath, formatSize(fi.Size()), elapsed.Round(time.Millisecond))
		if skipped > 0 {
			fmt.Printf("  %d records skipped\n", skipped)
		}
	}
	return nil
}

func decodeChromemGob(path string) (*chromemDoc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	var doc chromemDoc
	if err := gob.NewDecoder(gz).Decode(&doc); err != nil {
		return nil, err
	}
	return &doc, nil
}

package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newExportCmd() *cobra.Command {
	var to string
	var noVectors bool
	var limit int

	cmd := &cobra.Command{
		Use:   "export [path.tq]",
		Short: "Export vectors to JSONL",
		Long: `Dump the contents of a .tq file to JSONL format.
Vectors are dequantized (reconstructed, lossy).

Examples:
  tqdb export --to dump.jsonl
  tqdb export | head -5
  tqdb export --no-vectors | jq '.metadata.repo' | sort | uniq -c
  tqdb export index.tq --limit 100`,
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			path := ""
			if len(args) > 0 {
				path = args[0]
			}
			p, err := resolveStorePath(path)
			if err != nil {
				return fmt.Errorf("no store path and no workspace found")
			}

			s, err := store.Open(p)
			if err != nil {
				return err
			}
			defer func() { _ = s.Close() }()

			var output *os.File
			if to != "" {
				output, err = os.Create(to)
				if err != nil {
					return err
				}
				defer func() { _ = output.Close() }()
			} else {
				output = os.Stdout
			}

			enc := json.NewEncoder(output)
			count := 0
			s.ForEachCompressed(func(id string, indices []uint8, norm float32, content string, data map[string]any) {
				if limit > 0 && count >= limit {
					return
				}

				rec := map[string]any{
					"id":       id,
					"content":  content,
					"metadata": data,
				}
				if !noVectors {
					// TODO: dequantize indices back to float64 vector
					// For now, export the raw indices + norm.
					rec["norm"] = norm
					rec["indices"] = indices
				}

				_ = enc.Encode(rec)
				count++
			})

			quiet, _ := cmd.Flags().GetBool("quiet")
			if !quiet && to != "" {
				fmt.Fprintf(os.Stderr, "exported %d records to %s\n", count, to)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&to, "to", "", "output file (default: stdout)")
	cmd.Flags().BoolVar(&noVectors, "no-vectors", false, "omit vectors (export only id/content/metadata)")
	cmd.Flags().IntVar(&limit, "limit", 0, "max records to export (0 = all)")

	return cmd
}

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/internal/embed"
	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newSearchCmd() *cobra.Command {
	var vectorFile, format, provider, project, apiKey, model string
	var topK, rescore int
	var filters []string

	cmd := &cobra.Command{
		Use:   "search [path.tq] <query>",
		Short: "Search by text or vector",
		Long: `Search the vector index by text query (requires embedding provider) or
raw vector file.

Examples:
  tqdb search "how does authentication work"
  tqdb search "error handling" --top 5 --filter repo=csgda-kit
  tqdb search index.tq "grpc middleware" --provider vertex --project my-proj
  tqdb search --vector query.json --top 10
  tqdb search "auth" --format json | jq '.[0].content'`,
		Args: cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			quiet, _ := cmd.Flags().GetBool("quiet")
			cfgFile, _ := cmd.Flags().GetString("config")
			cfg := loadConfig(cfgFile)
			applyEnvOverrides(&cfg)

			// Parse args: either (path, query) or just (query) with workspace.
			var storePath, queryText string
			if len(args) == 2 {
				storePath = args[0]
				queryText = args[1]
			} else {
				queryText = args[0]
			}

			p, err := resolveStorePath(storePath)
			if err != nil {
				return fmt.Errorf("no store path specified and no workspace found")
			}
			storePath = p

			// Open store.
			s, err := store.Open(storePath)
			if err != nil {
				return fmt.Errorf("opening %s: %w", storePath, err)
			}
			defer func() { _ = s.Close() }()

			// Get query vector.
			var query []float64
			if vectorFile != "" {
				// Raw vector from file.
				data, err := os.ReadFile(vectorFile)
				if err != nil {
					return err
				}
				if err := json.Unmarshal(data, &query); err != nil {
					return fmt.Errorf("parsing vector file: %w", err)
				}
			} else {
				// Text search: embed via provider.
				prov := provider
				if prov == "" {
					prov = cfg.Provider
				}
				proj := project
				if proj == "" {
					proj = cfg.Project
				}
				key := apiKey
				if key == "" {
					key = cfg.APIKey
				}
				mdl := model
				if mdl == "" {
					mdl = cfg.Model
				}

				embedder, err := embed.New(prov, proj, key, mdl, cfg.BaseURL)
				if err != nil {
					return fmt.Errorf("creating embedding provider: %w", err)
				}

				query, err = embedder.Embed(cmd_context(), queryText)
				if err != nil {
					return fmt.Errorf("embedding query: %w", err)
				}
			}

			// Build search options.
			opts := tqdb.SearchOptions{TopK: topK, Rescore: rescore}
			if len(filters) > 0 {
				filterList := make([]tqdb.Filter, 0, len(filters))
				for _, f := range filters {
					k, v, ok := strings.Cut(f, "=")
					if !ok {
						return fmt.Errorf("invalid filter %q (expected key=value)", f)
					}
					filterList = append(filterList, tqdb.Eq(k, v))
				}
				if len(filterList) == 1 {
					opts.Filter = filterList[0]
				} else {
					opts.Filter = tqdb.And(filterList...)
				}
			}

			results := s.SearchWithOptions(query, opts)

			// Output.
			switch format {
			case "json":
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(results)
			default:
				if !quiet {
					fmt.Printf("Results for %q (%d vectors, %d-bit):\n\n", queryText, s.Len(), s.Info().Bits)
				}
				for i, r := range results {
					fmt.Printf(" %d. (%.3f) %s\n", i+1, r.Score, r.ID)
					// Show key metadata inline.
					var meta []string
					for _, k := range []string{"repo", "file_path", "language"} {
						if v, ok := r.Data[k]; ok {
							meta = append(meta, fmt.Sprintf("%s=%v", k, v))
						}
					}
					if len(meta) > 0 {
						fmt.Printf("    %s\n", strings.Join(meta, " | "))
					}
					if r.Content != "" {
						c := r.Content
						if len(c) > 200 {
							c = c[:200] + "..."
						}
						fmt.Printf("    %s\n", c)
					}
					fmt.Println()
				}
				return nil
			}
		},
	}

	cmd.Flags().StringVar(&vectorFile, "vector", "", "query vector file (JSON array)")
	cmd.Flags().IntVar(&topK, "top", 10, "number of results")
	cmd.Flags().IntVar(&rescore, "rescore", 0, "rescore top-N with exact distance")
	cmd.Flags().StringSliceVar(&filters, "filter", nil, "metadata filter key=value (repeatable)")
	cmd.Flags().StringVar(&format, "format", "text", "output format: text, json")
	cmd.Flags().StringVar(&provider, "provider", "", "override embedding provider")
	cmd.Flags().StringVar(&project, "project", "", "override GCP project")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "override API key")
	cmd.Flags().StringVar(&model, "model", "", "override embedding model")

	return cmd
}

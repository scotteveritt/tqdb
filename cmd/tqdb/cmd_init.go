package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

func newInitCmd() *cobra.Command {
	var provider, project, apiKey, model, baseURL string
	var bits, dim int

	cmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize a tqdb workspace in the current directory",
		Long: `Creates a .tqdb/ directory with a config file. After init, commands
like 'tqdb import' and 'tqdb search' work without specifying paths or providers.

Examples:
  tqdb init --provider vertex --project my-gcp-project
  tqdb init --provider ollama
  tqdb init --provider openai --api-key sk-...`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			dir := filepath.Join(".", ".tqdb")
			if _, err := os.Stat(dir); err == nil {
				return fmt.Errorf("workspace already exists at %s (use --force to reinitialize)", dir)
			}

			if err := os.MkdirAll(dir, 0o755); err != nil {
				return fmt.Errorf("creating .tqdb directory: %w", err)
			}

			cfg := Config{
				Provider: provider,
				Project:  project,
				APIKey:   apiKey,
				Model:    model,
				BaseURL:  baseURL,
			}
			if bits > 0 {
				cfg.Defaults.Bits = bits
			}
			if dim > 0 {
				cfg.Defaults.Dim = dim
			}

			data, err := yaml.Marshal(cfg)
			if err != nil {
				return fmt.Errorf("marshaling config: %w", err)
			}

			cfgPath := filepath.Join(dir, "config.yaml")
			if err := os.WriteFile(cfgPath, data, 0o644); err != nil {
				return fmt.Errorf("writing config: %w", err)
			}

			fmt.Printf("Initialized tqdb workspace at .tqdb/\n")
			fmt.Printf("  provider: %s\n", provider)
			if model != "" {
				fmt.Printf("  model:    %s\n", model)
			}
			fmt.Printf("\nNext: tqdb import --from <data.jsonl>\n")
			return nil
		},
	}

	cmd.Flags().StringVar(&provider, "provider", "ollama", "embedding provider: vertex, openai, ollama")
	cmd.Flags().StringVar(&project, "project", "", "GCP project ID (for vertex)")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "API key (for openai)")
	cmd.Flags().StringVar(&model, "model", "", "embedding model name")
	cmd.Flags().StringVar(&baseURL, "base-url", "", "API base URL (for ollama)")
	cmd.Flags().IntVar(&bits, "bits", 8, "default quantization bits")
	cmd.Flags().IntVar(&dim, "dim", 0, "embedding dimension (auto-detected if omitted)")

	return cmd
}

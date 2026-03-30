package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var version = "dev"

func main() {
	root := &cobra.Command{
		Use:   "tqdb",
		Short: "The quantization-native vector database",
		Long: `tqdb is a vector database that stores and searches vectors in quantized form.
Import your data, search by text or vector, and export results.

Quick start:
  tqdb init --provider ollama
  tqdb import --from data.jsonl --embed
  tqdb search "how does authentication work"`,
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	root.AddCommand(
		newInitCmd(),
		newImportCmd(),
		newSearchCmd(),
		newInfoCmd(),
		newCountCmd(),
		newExportCmd(),
		newBenchCmd(),
		newVersionCmd(),
	)

	// Global flags
	root.PersistentFlags().StringP("config", "c", "", "config file (default: .tqdb/config.yaml or ~/.config/tqdb/config.yaml)")
	root.PersistentFlags().BoolP("quiet", "q", false, "suppress progress output")

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

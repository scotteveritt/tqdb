package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/scotteveritt/tqdb"
	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newInfoCmd() *cobra.Command {
	var format string

	cmd := &cobra.Command{
		Use:   "info [path.tq]",
		Short: "Display .tq file metadata",
		Args:  cobra.MaximumNArgs(1),
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

			info := s.Info()

			if format == "json" {
				return json.NewEncoder(os.Stdout).Encode(info)
			}

			rot := "qr"
			if info.Rotation == tqdb.RotationHadamard {
				rot = "hadamard"
			}

			fmt.Printf("tqdb: %s\n\n", info.Path)
			fmt.Printf("  vectors:     %d\n", info.NumVecs)
			fmt.Printf("  dimension:   %d", info.Dim)
			if info.WorkDim != info.Dim {
				fmt.Printf(" (padded: %d)", info.WorkDim)
			}
			fmt.Println()
			fmt.Printf("  bits:        %d\n", info.Bits)
			fmt.Printf("  rotation:    %s (seed: %d)\n", rot, info.Seed)
			fmt.Printf("  file size:   %s\n", formatSize(info.FileSize))
			fmt.Printf("  compression: %.1fx vs float32\n", info.Compression)

			return nil
		},
	}

	cmd.Flags().StringVar(&format, "format", "text", "output format: text, json")
	return cmd
}

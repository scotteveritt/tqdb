package main

import (
	"fmt"

	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newCountCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "count [path.tq]",
		Short: "Print the number of vectors",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
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
			fmt.Println(s.Len())
			return nil
		},
	}
}

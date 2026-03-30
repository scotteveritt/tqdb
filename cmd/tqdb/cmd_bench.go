package main

import (
	"fmt"
	"math/rand/v2"
	"sort"
	"time"

	"github.com/scotteveritt/tqdb/store"
	"github.com/spf13/cobra"
)

func newBenchCmd() *cobra.Command {
	var nQueries int

	cmd := &cobra.Command{
		Use:   "bench [path.tq]",
		Short: "Benchmark search performance",
		Long: `Measure search latency using random query vectors sampled from the index.

Examples:
  tqdb bench
  tqdb bench index.tq --queries 500`,
		Args: cobra.MaximumNArgs(1),
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

			info := s.Info()
			fmt.Printf("benchmark: %s\n", info.Path)
			fmt.Printf("  vectors: %d  dim: %d  bits: %d\n\n", info.NumVecs, info.Dim, info.Bits)

			// Generate random queries.
			rng := rand.New(rand.NewPCG(42, 0))
			queries := make([][]float64, nQueries)
			for i := range queries {
				q := make([]float64, info.Dim)
				for j := range q {
					q[j] = rng.NormFloat64()
				}
				queries[i] = q
			}

			// Warm up.
			for i := range min(5, nQueries) {
				_ = s.Search(queries[i], 10)
			}

			// Benchmark.
			latencies := make([]time.Duration, nQueries)
			for i, q := range queries {
				start := time.Now()
				_ = s.Search(q, 10)
				latencies[i] = time.Since(start)
			}

			sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })

			var total time.Duration
			for _, d := range latencies {
				total += d
			}

			p50 := latencies[nQueries/2]
			p95 := latencies[nQueries*95/100]
			p99 := latencies[nQueries*99/100]
			qps := float64(nQueries) / total.Seconds()

			fmt.Printf("  brute-force (%d queries):\n", nQueries)
			fmt.Printf("    p50: %s  p95: %s  p99: %s\n",
				p50.Round(100*time.Microsecond),
				p95.Round(100*time.Microsecond),
				p99.Round(100*time.Microsecond))
			fmt.Printf("    QPS: %.0f\n", qps)
			fmt.Printf("    memory: %s\n", formatSize(info.FileSize))

			return nil
		},
	}

	cmd.Flags().IntVar(&nQueries, "queries", 100, "number of test queries")
	return cmd
}

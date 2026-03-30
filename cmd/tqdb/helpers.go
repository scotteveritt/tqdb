package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
)

// cmd_context returns a context that cancels on SIGINT/SIGTERM.
func cmd_context() context.Context {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	_ = cancel // caller doesn't need cancel; process exits on signal
	return ctx
}

// formatSize formats a byte count as a human-readable string.
func formatSize(b int64) string {
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

// Package embed provides lightweight embedding provider clients.
// Each provider is a simple HTTP client with no external SDK dependencies.
package embed

import (
	"context"
	"fmt"
)

// Provider embeds text into a float64 vector.
type Provider interface {
	Embed(ctx context.Context, text string) ([]float64, error)
	Name() string
}

// New creates an embedding provider from a name and config.
func New(provider, project, apiKey, model, baseURL string) (Provider, error) {
	switch provider {
	case "vertex", "":
		if project == "" {
			return nil, fmt.Errorf("vertex provider requires project (set --project or TQDB_PROJECT)")
		}
		if model == "" {
			model = "gemini-embedding-001"
		}
		return newVertex(project, model), nil

	case "openai":
		if apiKey == "" {
			return nil, fmt.Errorf("openai provider requires api_key (set --api-key or TQDB_API_KEY)")
		}
		if model == "" {
			model = "text-embedding-3-small"
		}
		return newOpenAI(apiKey, model), nil

	case "ollama":
		if baseURL == "" {
			baseURL = "http://localhost:11434"
		}
		if model == "" {
			model = "nomic-embed-text"
		}
		return newOllama(baseURL, model), nil

	default:
		return nil, fmt.Errorf("unknown provider %q (supported: vertex, openai, ollama)", provider)
	}
}

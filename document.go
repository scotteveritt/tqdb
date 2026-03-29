package tqdb

import "context"

// Document is a data object in a Collection.
// Data fields are typed (string, float64, bool, []string) matching VS2 DataObject.
type Document struct {
	ID        string         // unique identifier
	Content   string         // original text for RAG (optional)
	Data      map[string]any // typed fields (VS2: data)
	Embedding []float64      // raw embedding (nil = auto-embed via EmbeddingFunc)
}

// EmbeddingFunc converts text to an embedding vector.
// Same signature as chromem-go for drop-in compatibility.
type EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)

package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"golang.org/x/oauth2/google"
)

type vertexProvider struct {
	project string
	model   string
	client  *http.Client
}

func newVertex(project, model string) *vertexProvider {
	return &vertexProvider{
		project: project,
		model:   model,
		client:  &http.Client{},
	}
}

func (v *vertexProvider) Name() string { return "vertex" }

func (v *vertexProvider) Embed(ctx context.Context, text string) ([]float64, error) {
	creds, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("vertex: finding credentials: %w", err)
	}
	token, err := creds.TokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("vertex: getting token: %w", err)
	}

	body, _ := json.Marshal(map[string]any{
		"instances":  []map[string]any{{"content": text}},
		"parameters": map[string]any{"autoTruncate": true},
	})

	url := fmt.Sprintf("https://us-central1-aiplatform.googleapis.com/v1/projects/%s/locations/us-central1/publishers/google/models/%s:predict",
		v.project, v.model)

	req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+token.AccessToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := v.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vertex: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vertex: API error %d: %s", resp.StatusCode, string(b))
	}

	var result struct {
		Predictions []struct {
			Embeddings struct {
				Values []float64 `json:"values"`
			} `json:"embeddings"`
		} `json:"predictions"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("vertex: decode response: %w", err)
	}
	if len(result.Predictions) == 0 {
		return nil, fmt.Errorf("vertex: empty response")
	}
	return result.Predictions[0].Embeddings.Values, nil
}

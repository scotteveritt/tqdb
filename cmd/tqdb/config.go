package main

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config holds CLI configuration, loaded from files and flags.
type Config struct {
	Provider string `yaml:"provider"`
	Project  string `yaml:"project"`
	APIKey   string `yaml:"api_key"`
	Model    string `yaml:"model"`
	BaseURL  string `yaml:"base_url"`

	Defaults struct {
		Bits     int    `yaml:"bits"`
		Rotation string `yaml:"rotation"`
		Seed     uint64 `yaml:"seed"`
		TopK     int    `yaml:"top_k"`
		Dim      int    `yaml:"dim"`
	} `yaml:"defaults"`
}

// loadConfig loads config from the hierarchy:
// 1. Explicit --config flag
// 2. .tqdb/config.yaml (workspace, walk up from CWD)
// 3. ~/.config/tqdb/config.yaml (user)
func loadConfig(explicit string) Config {
	var cfg Config

	// Try explicit path first.
	if explicit != "" {
		if data, err := os.ReadFile(explicit); err == nil {
			_ = yaml.Unmarshal(data, &cfg)
			return cfg
		}
	}

	// Try workspace config (walk up from CWD).
	if dir, err := findWorkspace(); err == nil {
		cfgPath := filepath.Join(dir, ".tqdb", "config.yaml")
		if data, err := os.ReadFile(cfgPath); err == nil {
			_ = yaml.Unmarshal(data, &cfg)
			return cfg
		}
	}

	// Try user config.
	home, _ := os.UserHomeDir()
	if home != "" {
		cfgPath := filepath.Join(home, ".config", "tqdb", "config.yaml")
		if data, err := os.ReadFile(cfgPath); err == nil {
			_ = yaml.Unmarshal(data, &cfg)
		}
	}

	return cfg
}

// findWorkspace walks up from CWD looking for a .tqdb directory.
func findWorkspace() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if info, err := os.Stat(filepath.Join(dir, ".tqdb")); err == nil && info.IsDir() {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", os.ErrNotExist
}

// resolveStorePath returns the .tq file path.
// If explicit path given, use it. Otherwise look for workspace default.tq.
func resolveStorePath(explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	dir, err := findWorkspace()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, ".tqdb", "default.tq"), nil
}

// applyEnvOverrides applies environment variable overrides to config.
func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("TQDB_PROVIDER"); v != "" {
		cfg.Provider = v
	}
	if v := os.Getenv("TQDB_PROJECT"); v != "" {
		cfg.Project = v
	}
	if v := os.Getenv("TQDB_API_KEY"); v != "" {
		cfg.APIKey = v
	}
	if v := os.Getenv("TQDB_MODEL"); v != "" {
		cfg.Model = v
	}
	if v := os.Getenv("TQDB_BASE_URL"); v != "" {
		cfg.BaseURL = v
	}
}

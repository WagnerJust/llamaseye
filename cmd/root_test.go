package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/WagnerJust/llamaseye/config"
)

func TestParseOptInt(t *testing.T) {
	tests := []struct {
		input   string
		want    *int
		wantErr bool
	}{
		{"", nil, false},
		{"  ", nil, false},
		{"42", intPtr(42), false},
		{"0", intPtr(0), false},
		{"abc", nil, true},
		{" 8 ", intPtr(8), false},
	}
	for _, tt := range tests {
		got, err := parseOptInt(tt.input)
		if tt.wantErr {
			if err == nil {
				t.Errorf("parseOptInt(%q) expected error, got nil", tt.input)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseOptInt(%q) unexpected error: %v", tt.input, err)
			continue
		}
		if tt.want == nil {
			if got != nil {
				t.Errorf("parseOptInt(%q) = %d, want nil", tt.input, *got)
			}
		} else {
			if got == nil || *got != *tt.want {
				t.Errorf("parseOptInt(%q) = %v, want %d", tt.input, got, *tt.want)
			}
		}
	}
}

func TestResolveModels_SingleModel(t *testing.T) {
	// Create a temp .gguf file
	dir := t.TempDir()
	model := filepath.Join(dir, "test.gguf")
	if err := os.WriteFile(model, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := testConfig()
	models, err := ResolveModels(cfg, []string{model})
	if err != nil {
		t.Fatalf("ResolveModels: %v", err)
	}
	if len(models) != 1 || models[0] != model {
		t.Errorf("got %v, want [%s]", models, model)
	}
}

func TestResolveModels_ModelNotFound(t *testing.T) {
	cfg := testConfig()
	_, err := ResolveModels(cfg, []string{"/nonexistent/model.gguf"})
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestResolveModels_ModelsDir(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.gguf", "b.gguf", "not-a-model.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	cfg := testConfig()
	cfg.ModelsDir = dir
	models, err := ResolveModels(cfg, nil)
	if err != nil {
		t.Fatalf("ResolveModels: %v", err)
	}
	if len(models) != 2 {
		t.Errorf("expected 2 .gguf files, got %d: %v", len(models), models)
	}
}

func TestResolveModels_ModelList(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(model, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	listFile := filepath.Join(dir, "list.txt")
	if err := os.WriteFile(listFile, []byte("# comment\nm.gguf\n\n"), 0644); err != nil {
		t.Fatal(err)
	}

	cfg := testConfig()
	cfg.ModelListFile = listFile
	cfg.ModelsDir = dir
	models, err := ResolveModels(cfg, nil)
	if err != nil {
		t.Fatalf("ResolveModels: %v", err)
	}
	if len(models) != 1 {
		t.Errorf("expected 1 model from list, got %d", len(models))
	}
}

func TestResolveModels_NoModels(t *testing.T) {
	cfg := testConfig()
	_, err := ResolveModels(cfg, nil)
	if err == nil {
		t.Error("expected error when no models specified")
	}
}

func TestParse_Version(t *testing.T) {
	cfg, _, err := Parse([]string{"--debug"}, "v1.0.0-test")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !cfg.Debug {
		t.Error("expected --debug to be true")
	}
}

func intPtr(n int) *int { return &n }

func testConfig() *config.Config {
	return config.Defaults()
}

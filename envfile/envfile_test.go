package envfile

import (
	"os"
	"path/filepath"
	"testing"
)

func writeEnv(t *testing.T, content string) string {
	t.Helper()
	f := filepath.Join(t.TempDir(), ".env")
	if err := os.WriteFile(f, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestLoad_Basic(t *testing.T) {
	path := writeEnv(t, "LLAMASEYE_TEST_KEY=hello\nLLAMASEYE_TEST_NUM=42\n")
	t.Cleanup(func() {
		os.Unsetenv("LLAMASEYE_TEST_KEY")
		os.Unsetenv("LLAMASEYE_TEST_NUM")
	})

	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_TEST_KEY"); v != "hello" {
		t.Errorf("KEY = %q, want hello", v)
	}
	if v := os.Getenv("LLAMASEYE_TEST_NUM"); v != "42" {
		t.Errorf("NUM = %q, want 42", v)
	}
}

func TestLoad_ProcessEnvWins(t *testing.T) {
	// Pre-set the var — file value must NOT override it
	os.Setenv("LLAMASEYE_PRESET", "process_value")
	t.Cleanup(func() { os.Unsetenv("LLAMASEYE_PRESET") })

	path := writeEnv(t, "LLAMASEYE_PRESET=file_value\n")
	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_PRESET"); v != "process_value" {
		t.Errorf("env = %q, want process_value (process env should win)", v)
	}
}

func TestLoad_SkipsCommentsAndBlanks(t *testing.T) {
	path := writeEnv(t, `
# This is a comment
LLAMASEYE_REAL=yes

# Another comment
`)
	t.Cleanup(func() { os.Unsetenv("LLAMASEYE_REAL") })

	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_REAL"); v != "yes" {
		t.Errorf("REAL = %q, want yes", v)
	}
}

func TestLoad_InlineComment(t *testing.T) {
	path := writeEnv(t, "LLAMASEYE_IC=value # inline comment\n")
	t.Cleanup(func() { os.Unsetenv("LLAMASEYE_IC") })

	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_IC"); v != "value" {
		t.Errorf("IC = %q, want value (inline comment stripped)", v)
	}
}

func TestLoad_QuotedValues(t *testing.T) {
	path := writeEnv(t, `LLAMASEYE_DQ="double quoted"` + "\n" + `LLAMASEYE_SQ='single quoted'` + "\n")
	t.Cleanup(func() {
		os.Unsetenv("LLAMASEYE_DQ")
		os.Unsetenv("LLAMASEYE_SQ")
	})

	if err := Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_DQ"); v != "double quoted" {
		t.Errorf("DQ = %q, want 'double quoted'", v)
	}
	if v := os.Getenv("LLAMASEYE_SQ"); v != "single quoted" {
		t.Errorf("SQ = %q, want 'single quoted'", v)
	}
}

func TestLoad_MissingFile(t *testing.T) {
	err := Load("/nonexistent/path/.env")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestLoadIfExists_Missing(t *testing.T) {
	// Should not error for missing file
	if err := LoadIfExists("/nonexistent/.env"); err != nil {
		t.Errorf("LoadIfExists missing: %v", err)
	}
}

func TestLoadIfExists_Present(t *testing.T) {
	path := writeEnv(t, "LLAMASEYE_LIE=exists\n")
	t.Cleanup(func() { os.Unsetenv("LLAMASEYE_LIE") })

	if err := LoadIfExists(path); err != nil {
		t.Fatalf("LoadIfExists: %v", err)
	}
	if v := os.Getenv("LLAMASEYE_LIE"); v != "exists" {
		t.Errorf("LIE = %q, want exists", v)
	}
}

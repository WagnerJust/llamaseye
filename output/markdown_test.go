package output

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/WagnerJust/llamaseye/hardware"
)

const sampleJSONL = `{"run_id":"r1","timestamp":"2024-01-01T00:00:00Z","model_path":"/m.gguf","model_stem":"m","phase":1,"phase_label":"ngl_sweep","binary":"standard","status":"ok","viable":true,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":512,"n_gen":128,"repetitions":3},"results":[{"test":"pp","n_prompt":512,"avg_ts":100.0,"stddev_ts":1.0},{"test":"tg","n_gen":128,"avg_ts":25.0,"stddev_ts":0.5}],"wall_time_sec":10.0,"raw_output_file":"raw/r1.txt","error_snippet":null}
{"run_id":"r2","timestamp":"2024-01-01T00:00:01Z","model_path":"/m.gguf","model_stem":"m","phase":1,"phase_label":"ngl_sweep","binary":"standard","status":"oom","viable":null,"params":{"ngl":99,"fa":0,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":512,"n_gen":128,"repetitions":3},"results":[],"wall_time_sec":null,"raw_output_file":null,"error_snippet":"CUDA out of memory"}
{"run_id":"r3","timestamp":"2024-01-01T00:00:02Z","model_path":"/m.gguf","model_stem":"m","phase":6,"phase_label":"ctx_sweep","binary":"standard","status":"timeout","viable":null,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":65536,"n_gen":0,"repetitions":1},"results":[],"wall_time_sec":600.0,"raw_output_file":null,"error_snippet":null}
`

func TestGenerateMarkdown_Basic(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "sweep.jsonl"), []byte(sampleJSONL), 0644); err != nil {
		t.Fatal(err)
	}

	if err := GenerateMarkdown(dir, "m", "", 600); err != nil {
		t.Fatalf("GenerateMarkdown: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "sweep.md"))
	if err != nil {
		t.Fatal(err)
	}
	md := string(data)

	// Should have top-N section
	if !strings.Contains(md, "Best Configurations") {
		t.Error("missing Best Configurations section")
	}
	// Should have phase 1 section
	if !strings.Contains(md, "Phase 1") {
		t.Error("missing Phase 1 section")
	}
	// Phase 6 timeout section
	if !strings.Contains(md, "timed out") {
		t.Error("missing phase 6 timeout section")
	}
	// Winner callout for phase 1
	if !strings.Contains(md, "Winner") {
		t.Error("missing Winner callout")
	}
}

func TestGenerateMarkdown_EmptyJSONL(t *testing.T) {
	dir := t.TempDir()
	// Empty file
	if err := os.WriteFile(filepath.Join(dir, "sweep.jsonl"), []byte{}, 0644); err != nil {
		t.Fatal(err)
	}
	if err := GenerateMarkdown(dir, "m", "", 600); err != nil {
		t.Fatalf("GenerateMarkdown on empty: %v", err)
	}
}

func TestGenerateMarkdown_NoFile(t *testing.T) {
	dir := t.TempDir()
	// No sweep.jsonl — should be no-op
	if err := GenerateMarkdown(dir, "m", "", 600); err != nil {
		t.Fatalf("GenerateMarkdown on no file: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sweep.md")); err == nil {
		t.Error("sweep.md should not be created when no sweep.jsonl")
	}
}

func TestGenerateCrossModelSummary(t *testing.T) {
	dir := t.TempDir()
	for _, stem := range []string{"model-a", "model-b"} {
		subdir := filepath.Join(dir, stem)
		if err := os.MkdirAll(subdir, 0755); err != nil {
			t.Fatal(err)
		}
		jsonl := `{"run_id":"x","timestamp":"T","model_path":"/x.gguf","model_stem":"` + stem + `","phase":1,"phase_label":"ngl_sweep","binary":"standard","status":"ok","viable":true,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":512,"n_gen":128,"repetitions":3},"results":[{"test":"tg","n_gen":128,"avg_ts":25.0,"stddev_ts":0.5}],"wall_time_sec":null,"raw_output_file":null,"error_snippet":null}` + "\n"
		if err := os.WriteFile(filepath.Join(subdir, "sweep.jsonl"), []byte(jsonl), 0644); err != nil {
			t.Fatal(err)
		}
	}

	if err := GenerateCrossModelSummary(dir, []string{"model-a", "model-b"}); err != nil {
		t.Fatalf("GenerateCrossModelSummary: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "summary.md"))
	if err != nil {
		t.Fatal(err)
	}
	md := string(data)
	if !strings.Contains(md, "model-a") || !strings.Contains(md, "model-b") {
		t.Error("summary.md should contain both model names")
	}
	if !strings.Contains(md, "Multi-Model") {
		t.Error("summary.md should have Multi-Model header")
	}
}

func TestGenerateCrossModelSummary_LessThanTwo(t *testing.T) {
	dir := t.TempDir()
	// Only one model — should skip
	if err := GenerateCrossModelSummary(dir, []string{"solo"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "summary.md")); err == nil {
		t.Error("summary.md should not be created for <2 models")
	}
}

func TestLogger_Basic(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "test.log")
	logger, err := NewLogger(logPath)
	if err != nil {
		t.Fatalf("NewLogger: %v", err)
	}
	logger.Log("hello %s", "world")
	logger.Warn("something went wrong: %d", 42)
	logger.Close()

	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "hello world") {
		t.Error("expected 'hello world' in log")
	}
	if !strings.Contains(content, "[WARN]") {
		t.Error("expected [WARN] in log")
	}
}

func TestLogger_NoFile(t *testing.T) {
	// Logger with no file should not error and not panic
	logger, err := NewLogger("")
	if err != nil {
		t.Fatalf("NewLogger(empty): %v", err)
	}
	logger.Log("test message")
	logger.Close() // should be no-op
}

func TestWriteHardwareJSON(t *testing.T) {
	dir := t.TempDir()
	hw := &hardware.HardwareInfo{
		CPUModel:    "Test CPU",
		CPUPhysical: 8,
		CPULogical:  16,
		RAMGiB:      32,
		GPUModel:    "Test GPU",
		GPUVRAMGiB:  24,
	}
	if err := WriteHardwareJSON(dir, hw); err != nil {
		t.Fatalf("WriteHardwareJSON: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "hardware.json"))
	if err != nil {
		t.Fatalf("read hardware.json: %v", err)
	}
	if !strings.Contains(string(data), "Test CPU") {
		t.Error("expected CPU model in hardware.json")
	}
}

func TestGenerateMarkdown_WithGoal(t *testing.T) {
	dir := t.TempDir()
	// Phase 7 record that meets goal (ctx>=4096, tg>=20)
	jsonl := `{"run_id":"g1","timestamp":"2024-01-01T00:00:00Z","model_path":"/m.gguf","model_stem":"m","phase":7,"phase_label":"combo_matrix","binary":"standard","status":"ok","viable":true,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":4096,"n_gen":128,"repetitions":3},"results":[{"test":"tg","n_gen":128,"avg_ts":25.0,"stddev_ts":0.5}],"wall_time_sec":5.0,"raw_output_file":null,"error_snippet":null}` + "\n"
	if err := os.WriteFile(filepath.Join(dir, "sweep.jsonl"), []byte(jsonl), 0644); err != nil {
		t.Fatal(err)
	}
	if err := GenerateMarkdown(dir, "m", "ctx=4096,tg=20.0", 600); err != nil {
		t.Fatalf("GenerateMarkdown with goal: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "sweep.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "Goal") {
		t.Error("expected Goal section in markdown when goal spec provided")
	}
}

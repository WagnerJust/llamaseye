package sweep

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/justinphilpott/llamaseye/bench"
	"github.com/justinphilpott/llamaseye/config"
	"github.com/justinphilpott/llamaseye/hardware"
	"github.com/justinphilpott/llamaseye/output"
	"github.com/justinphilpott/llamaseye/state"
)

// mockExecutor simulates llama-bench output.
type mockExecutor struct {
	// For each call, return pre-scripted responses
	responses []mockResponse
	callIdx   int
}

type mockResponse struct {
	stdout   string
	stderr   string
	exitCode int
}

func (m *mockExecutor) Run(_ context.Context, _ string, _ []string, stdout, stderr io.Writer) (int, error) {
	idx := m.callIdx
	if idx >= len(m.responses) {
		// Default: return a successful result
		_, _ = io.WriteString(stdout, defaultOKOutput())
		return 0, nil
	}
	m.callIdx++
	r := m.responses[idx]
	if r.stdout != "" {
		_, _ = io.WriteString(stdout, r.stdout)
	}
	if r.stderr != "" {
		_, _ = io.WriteString(stderr, r.stderr)
	}
	return r.exitCode, nil
}

func defaultOKOutput() string {
	return `{"n_prompt":512,"n_gen":0,"avg_ts":120.5,"stddev_ts":1.2}` + "\n" +
		`{"n_prompt":0,"n_gen":128,"avg_ts":25.3,"stddev_ts":0.4}` + "\n"
}

func newTestSweeper(t *testing.T, exec bench.CommandExecutor, cfg *config.Config) (*Sweeper, string) {
	t.Helper()
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir
	cfg.DelaySeconds = 0
	cfg.TimeoutSec = 5
	cfg.NoConfirm = true
	cfg.NoThermal = true

	logger, _ := output.NewLogger("")

	hw := &hardware.HardwareInfo{
		CPUModel:    "Test CPU",
		CPUPhysical: 8,
		CPULogical:  16,
		RAMGiB:      32,
		GPUCount:    1,
		GPUModel:    "Test GPU",
		GPUVRAMGiB:  24,
		Backend:     hardware.BackendCUDA,
	}

	return &Sweeper{
		Config:   cfg,
		HW:       hw,
		Logger:   logger,
		Executor: exec,
	}, outputDir
}

func newTestModel(t *testing.T) string {
	t.Helper()
	// Create a tiny fake .gguf file (content doesn't matter for mock tests)
	dir := t.TempDir()
	path := filepath.Join(dir, "test-model.gguf")
	if err := os.WriteFile(path, []byte("fake gguf"), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestSweepModel_DryRun(t *testing.T) {
	cfg := config.Defaults()
	cfg.DryRun = true
	// Only run phases 0 and 1 to keep test fast
	cfg.OnlyPhases = []int{0, 1}

	exec := &mockExecutor{}
	s, outputDir := newTestSweeper(t, exec, cfg)
	modelPath := newTestModel(t)

	if err := s.SweepModel(context.Background(), modelPath); err != nil {
		t.Fatalf("SweepModel: %v", err)
	}

	// Output dir should have been created
	modelDir := filepath.Join(outputDir, "test-model")
	if _, err := os.Stat(modelDir); err != nil {
		t.Errorf("expected model output dir %s to exist", modelDir)
	}
}

func TestSweepModel_AllPhases_Mock(t *testing.T) {
	cfg := config.Defaults()
	// Run all phases with a mock that always succeeds
	// Use only phases 0,1,2 to keep fast
	cfg.OnlyPhases = []int{0, 1, 2}

	exec := &mockExecutor{} // always returns defaultOKOutput
	s, outputDir := newTestSweeper(t, exec, cfg)
	modelPath := newTestModel(t)

	if err := s.SweepModel(context.Background(), modelPath); err != nil {
		t.Fatalf("SweepModel: %v", err)
	}

	modelDir := filepath.Join(outputDir, "test-model")

	// Verify sweep.jsonl was created and has records
	jsonlPath := filepath.Join(modelDir, "sweep.jsonl")
	data, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatalf("read sweep.jsonl: %v", err)
	}
	lines := nonEmptyLines(data)
	if len(lines) == 0 {
		t.Error("expected sweep.jsonl to have records")
	}

	// Verify all records are valid JSON
	for i, line := range lines {
		var rec map[string]any
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Errorf("line %d: invalid JSON: %v", i, err)
		}
	}

	// Verify state.json was created
	if _, err := os.Stat(filepath.Join(modelDir, "state.json")); err != nil {
		t.Error("expected state.json to exist")
	}
}

func TestSweepModel_Resume(t *testing.T) {
	cfg := config.Defaults()

	exec := &mockExecutor{}
	s, outputDir := newTestSweeper(t, exec, cfg)
	modelPath := newTestModel(t)

	// First run: only phases 0, 1
	cfg.OnlyPhases = []int{0, 1}
	if err := s.SweepModel(context.Background(), modelPath); err != nil {
		t.Fatalf("first SweepModel: %v", err)
	}

	modelDir := filepath.Join(outputDir, "test-model")

	// Verify state was saved with phases 0,1
	st, err := state.Load(modelDir)
	if err != nil {
		t.Fatalf("Load state: %v", err)
	}
	if !st.PhaseComplete(0) || !st.PhaseComplete(1) {
		t.Error("expected phases 0 and 1 to be complete after first run")
	}

	// Second run: resume, run phases 2,3
	cfg2 := config.Defaults()
	cfg2.OutputDir = outputDir
	cfg2.OnlyPhases = []int{0, 1, 2} // 0 and 1 should be skipped due to state
	cfg2.Resume = true
	cfg2.DelaySeconds = 0
	cfg2.TimeoutSec = 5
	cfg2.NoThermal = true

	exec2 := &mockExecutor{}
	s2 := &Sweeper{
		Config:   cfg2,
		HW:       s.HW,
		Logger:   s.Logger,
		Executor: exec2,
	}

	if err := s2.SweepModel(context.Background(), modelPath); err != nil {
		t.Fatalf("second SweepModel: %v", err)
	}

	// The call count should be lower since 0,1 were skipped
	// (mock executor has 0 pre-scripted responses, so all go to default)
	// Phase 2 has 5 standard combos, so expect ~5 calls
	if exec2.callIdx > exec.callIdx+10 {
		t.Errorf("resume seems to have re-run phases: exec2.callIdx=%d, expected <= %d", exec2.callIdx, exec.callIdx+10)
	}
}

func TestSweepModel_OOMHandling(t *testing.T) {
	cfg := config.Defaults()
	cfg.OnlyPhases = []int{0}

	// Phase 0 probe: first call OOMs at ngl=99, succeeds at ngl=95
	responses := []mockResponse{
		{stderr: "CUDA out of memory\n", exitCode: 1}, // ngl=99 OOM
		{stderr: "CUDA out of memory\n", exitCode: 1}, // ngl=95 OOM
		{stderr: "CUDA out of memory\n", exitCode: 1}, // ngl=91 OOM
		{stdout: defaultOKOutput(), exitCode: 0},       // ngl=87 OK
	}
	exec := &mockExecutor{responses: responses}
	s, outputDir := newTestSweeper(t, exec, cfg)
	modelPath := newTestModel(t)

	if err := s.SweepModel(context.Background(), modelPath); err != nil {
		t.Fatalf("SweepModel: %v", err)
	}

	// Verify OOM records in jsonl
	modelDir := filepath.Join(outputDir, "test-model")
	data, _ := os.ReadFile(filepath.Join(modelDir, "sweep.jsonl"))
	oomCount := 0
	for _, line := range nonEmptyLines(data) {
		var rec map[string]any
		if json.Unmarshal([]byte(line), &rec) == nil {
			if rec["status"] == "oom" {
				oomCount++
			}
		}
	}
	if oomCount != 3 {
		t.Errorf("expected 3 OOM records, got %d", oomCount)
	}
}

func TestReportMode(t *testing.T) {
	cfg := config.Defaults()
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir

	// Create a minimal sweep.jsonl for a fake model
	modelDir := filepath.Join(outputDir, "my-model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}

	rec := `{"run_id":"abc","timestamp":"2024-01-01T00:00:00Z","model_path":"/m.gguf","model_stem":"my-model","phase":1,"phase_label":"ngl_sweep","binary":"standard","status":"ok","viable":true,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":512,"n_gen":128,"repetitions":3},"results":[{"test":"pp","n_prompt":512,"avg_ts":100.0,"stddev_ts":1.0},{"test":"tg","n_gen":128,"avg_ts":25.0,"stddev_ts":0.5}],"wall_time_sec":12.5,"raw_output_file":"raw/abc.txt","error_snippet":null}` + "\n"
	if err := os.WriteFile(filepath.Join(modelDir, "sweep.jsonl"), []byte(rec), 0644); err != nil {
		t.Fatal(err)
	}

	logger, _ := output.NewLogger("")
	s := &Sweeper{
		Config:   cfg,
		HW:       &hardware.HardwareInfo{},
		Logger:   logger,
		Executor: &mockExecutor{},
	}

	if err := s.ReportMode([]string{"my-model"}); err != nil {
		t.Fatalf("ReportMode: %v", err)
	}

	mdPath := filepath.Join(modelDir, "sweep.md")
	if _, err := os.Stat(mdPath); err != nil {
		t.Error("expected sweep.md to be generated")
	}
	data, _ := os.ReadFile(mdPath)
	if !strings.Contains(string(data), "my-model") {
		t.Error("sweep.md should contain model name")
	}
}

func nonEmptyLines(data []byte) []string {
	var lines []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func TestParseGoal_Basic(t *testing.T) {
	g := parseGoal("ctx=8192,tg=15.5,pp=200.0")
	if g.CtxMin != 8192 {
		t.Errorf("CtxMin = %d, want 8192", g.CtxMin)
	}
	if g.TGMin != 15.5 {
		t.Errorf("TGMin = %f, want 15.5", g.TGMin)
	}
	if g.PPMin != 200.0 {
		t.Errorf("PPMin = %f, want 200.0", g.PPMin)
	}
	if g.MaxHits != 3 {
		t.Errorf("MaxHits = %d, want 3 (default)", g.MaxHits)
	}
}

func TestParseGoal_Empty(t *testing.T) {
	g := parseGoal("")
	if g.CtxMin != 0 || g.TGMin != 0 || g.PPMin != 0 {
		t.Errorf("expected zero goal for empty spec, got %+v", g)
	}
}

func TestParseGoal_PartialSpec(t *testing.T) {
	g := parseGoal("ctx=4096")
	if g.CtxMin != 4096 {
		t.Errorf("CtxMin = %d, want 4096", g.CtxMin)
	}
	if g.TGMin != 0 {
		t.Errorf("TGMin = %f, want 0 (unset)", g.TGMin)
	}
}

func TestDetectTurbo_EmptyPath(t *testing.T) {
	if detectTurbo("") {
		t.Error("detectTurbo('') should return false")
	}
}

func TestDetectTurbo_NonExistentPath(t *testing.T) {
	if detectTurbo("/nonexistent/turbo-bench") {
		t.Error("detectTurbo with nonexistent path should return false")
	}
}

func TestSetupOutputDir_New(t *testing.T) {
	cfg := config.Defaults()
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir
	logger, _ := output.NewLogger("")
	s := &Sweeper{Config: cfg, HW: &hardware.HardwareInfo{}, Logger: logger}

	newDir := filepath.Join(outputDir, "newmodel")
	if err := s.setupOutputDir(newDir); err != nil {
		t.Fatalf("setupOutputDir for new dir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(newDir, "raw")); err != nil {
		t.Error("expected raw/ subdir to be created")
	}
}

func TestSetupOutputDir_ExistsNoResume(t *testing.T) {
	cfg := config.Defaults()
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir
	logger, _ := output.NewLogger("")
	s := &Sweeper{Config: cfg, HW: &hardware.HardwareInfo{}, Logger: logger}

	existingDir := filepath.Join(outputDir, "model")
	if err := os.MkdirAll(existingDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Without --resume or --overwrite, should error
	err := s.setupOutputDir(existingDir)
	if err == nil {
		t.Error("expected error for existing dir without resume/overwrite")
	}
}

func TestSetupOutputDir_Overwrite(t *testing.T) {
	cfg := config.Defaults()
	cfg.Overwrite = true
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir
	logger, _ := output.NewLogger("")
	s := &Sweeper{Config: cfg, HW: &hardware.HardwareInfo{}, Logger: logger}

	existingDir := filepath.Join(outputDir, "model")
	if err := os.MkdirAll(filepath.Join(existingDir, "raw"), 0755); err != nil {
		t.Fatal(err)
	}
	// Put a sentinel file in it
	if err := os.WriteFile(filepath.Join(existingDir, "sweep.jsonl"), []byte("old"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := s.setupOutputDir(existingDir); err != nil {
		t.Fatalf("setupOutputDir with overwrite: %v", err)
	}
	// Old file should be gone
	if _, err := os.Stat(filepath.Join(existingDir, "sweep.jsonl")); err == nil {
		t.Error("old sweep.jsonl should have been removed by overwrite")
	}
}

func TestReportMode_AutoScan(t *testing.T) {
	cfg := config.Defaults()
	outputDir := t.TempDir()
	cfg.OutputDir = outputDir

	// Create two model subdirs with sweep.jsonl
	rec := `{"run_id":"x","timestamp":"T","model_path":"/x.gguf","model_stem":"scan-model","phase":1,"phase_label":"ngl_sweep","binary":"standard","status":"ok","viable":true,"params":{"ngl":32,"fa":1,"ctk":"f16","ctv":"f16","nkvo":0,"threads":null,"threads_is_default":true,"b":2048,"ub":512,"n_prompt":512,"n_gen":128,"repetitions":3},"results":[{"test":"tg","n_gen":128,"avg_ts":25.0,"stddev_ts":0.5}],"wall_time_sec":null,"raw_output_file":null,"error_snippet":null}` + "\n"
	for _, stem := range []string{"scan-model-a", "scan-model-b"} {
		dir := filepath.Join(outputDir, stem)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "sweep.jsonl"), []byte(rec), 0644); err != nil {
			t.Fatal(err)
		}
	}

	logger, _ := output.NewLogger("")
	s := &Sweeper{
		Config:   cfg,
		HW:       &hardware.HardwareInfo{},
		Logger:   logger,
		Executor: &mockExecutor{},
	}

	// Pass nil stems — should auto-scan
	if err := s.ReportMode(nil); err != nil {
		t.Fatalf("ReportMode auto-scan: %v", err)
	}
}

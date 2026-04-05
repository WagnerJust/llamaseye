package bench

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/justinphilpott/llamaseye/config"
)

// mockExecutor implements CommandExecutor for tests.
type mockExecutor struct {
	stdout   string
	stderr   string
	exitCode int
	// If slowTimeout is true, the executor hangs until ctx is cancelled.
	slowTimeout bool
}

func (m *mockExecutor) Run(ctx context.Context, binary string, args []string, stdout, stderr io.Writer) (int, error) {
	if m.slowTimeout {
		<-ctx.Done()
		return 1, ctx.Err()
	}
	if m.stdout != "" {
		_, _ = io.WriteString(stdout, m.stdout)
	}
	if m.stderr != "" {
		_, _ = io.WriteString(stderr, m.stderr)
	}
	return m.exitCode, nil
}

func newTestRunner(t *testing.T, exec CommandExecutor) *BenchRunner {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "raw"), 0755); err != nil {
		t.Fatal(err)
	}
	cfg := config.Defaults()
	cfg.DelaySeconds = 0
	cfg.TimeoutSec = 2
	return &BenchRunner{
		Config: cfg,
		Selector: &BinarySelector{
			StandardBin:    "/usr/bin/echo",
			TurboBin:       "",
			TurboAvailable: false,
		},
		Executor:  exec,
		OutputDir: dir,
		ModelPath: "/fake/model.gguf",
		ModelStem: "model",
	}
}

func TestRunBench_OK(t *testing.T) {
	stdout := `{"n_prompt":512,"n_gen":0,"avg_ts":123.4,"stddev_ts":1.2}` + "\n" +
		`{"n_prompt":0,"n_gen":128,"avg_ts":45.6,"stddev_ts":0.5}` + "\n"
	exec := &mockExecutor{stdout: stdout}
	r := newTestRunner(t, exec)

	res, err := r.RunBench("test", RunParams{
		NGL: 20, FA: 1, CTK: "f16", CTV: "f16", NKVO: 0,
		B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 3,
		Phase: 1, PhaseLabel: "ngl_sweep",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != StatusOK {
		t.Errorf("status = %q, want ok", res.Status)
	}
	if PPSpeed(res.Results) != 123.4 {
		t.Errorf("PP t/s = %f, want 123.4", PPSpeed(res.Results))
	}
	if TGSpeed(res.Results) != 45.6 {
		t.Errorf("TG t/s = %f, want 45.6", TGSpeed(res.Results))
	}
}

func TestRunBench_OOM(t *testing.T) {
	exec := &mockExecutor{
		stderr:   "CUDA out of memory\n",
		exitCode: 1,
	}
	r := newTestRunner(t, exec)

	res, err := r.RunBench("test", RunParams{
		NGL: 99, FA: 0, CTK: "f16", CTV: "f16",
		B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != StatusOOM {
		t.Errorf("status = %q, want oom", res.Status)
	}
}

func TestRunBench_Timeout(t *testing.T) {
	exec := &mockExecutor{slowTimeout: true}
	r := newTestRunner(t, exec)

	res, err := r.RunBench("test", RunParams{
		NGL: 20, FA: 0, CTK: "f16", CTV: "f16",
		B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != StatusTimeout {
		t.Errorf("status = %q, want timeout", res.Status)
	}
}

func TestRunBench_Error(t *testing.T) {
	exec := &mockExecutor{
		stdout:   "",
		stderr:   "some random error\n",
		exitCode: 1,
	}
	r := newTestRunner(t, exec)

	res, err := r.RunBench("test", RunParams{
		NGL: 20, FA: 0, CTK: "f16", CTV: "f16",
		B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 1,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != StatusError {
		t.Errorf("status = %q, want error", res.Status)
	}
}

func TestRunBench_DryRun(t *testing.T) {
	exec := &mockExecutor{}
	r := newTestRunner(t, exec)
	r.Config.DryRun = true

	res, err := r.RunBench("test", RunParams{NGL: 20, CTK: "f16", CTV: "f16", B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Status != StatusDryRun {
		t.Errorf("status = %q, want dry-run", res.Status)
	}
}

func TestParseLlamaBenchOutput(t *testing.T) {
	input := `{"n_prompt":512,"n_gen":0,"avg_ts":100.0,"stddev_ts":2.0}` + "\n" +
		`not json` + "\n" +
		`{"n_prompt":0,"n_gen":128,"avg_ts":20.5,"stddev_ts":0.3}` + "\n"

	results, err := parseLlamaBenchOutput([]byte(input))
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if results[0].Test != "pp" || results[0].AvgTS != 100.0 {
		t.Errorf("results[0] = %+v, want {Test:pp AvgTS:100.0}", results[0])
	}
	if results[1].Test != "tg" || results[1].AvgTS != 20.5 {
		t.Errorf("results[1] = %+v, want {Test:tg AvgTS:20.5}", results[1])
	}
}

func TestBinarySelector_Standard(t *testing.T) {
	sel := &BinarySelector{StandardBin: "/std/llama-bench", TurboAvailable: false}
	path, label, err := sel.Select("f16")
	if err != nil {
		t.Fatalf("Select f16: %v", err)
	}
	if path != "/std/llama-bench" {
		t.Errorf("path = %q, want /std/llama-bench", path)
	}
	if label != "standard" {
		t.Errorf("label = %q, want standard", label)
	}
}

func TestBinarySelector_TurboUnavailable(t *testing.T) {
	sel := &BinarySelector{StandardBin: "/std/llama-bench", TurboAvailable: false}
	_, _, err := sel.Select("turbo3")
	if err == nil {
		t.Error("expected error for turbo type when TurboAvailable=false")
	}
}

func TestBinarySelector_TurboAvailable(t *testing.T) {
	sel := &BinarySelector{
		StandardBin:    "/std/llama-bench",
		TurboBin:       "/turbo/llama-bench",
		TurboAvailable: true,
	}
	path, label, err := sel.Select("turbo3")
	if err != nil {
		t.Fatalf("Select turbo3: %v", err)
	}
	if path != "/turbo/llama-bench" {
		t.Errorf("path = %q, want /turbo/llama-bench", path)
	}
	if label != "turboquant" {
		t.Errorf("label = %q, want turboquant", label)
	}
}

func TestRunBench_TurboUnavailable(t *testing.T) {
	r := newTestRunner(t, &mockExecutor{})
	r.Selector.TurboAvailable = false

	// turbo type with unavailable turbo binary should error or return error status
	res, err := r.RunBench("test-turbo", RunParams{CTK: "turbo3", CTV: "f16", NGL: 20, B: 2048, UB: 512, NPrompt: 512, NGen: 128, Reps: 1})
	if err == nil && (res == nil || res.Status != StatusError) {
		t.Error("expected error or error status for turbo with unavailable binary")
	}
}

func TestSpeedHelpers(t *testing.T) {
	results := []TestResult{
		{Test: "pp", NPrompt: 512, AvgTS: 100.0},
		{Test: "tg", NGen: 128, AvgTS: 25.0},
	}
	if TGSpeed(results) != 25.0 {
		t.Errorf("TGSpeed = %f, want 25.0", TGSpeed(results))
	}
	if PPSpeed(results) != 100.0 {
		t.Errorf("PPSpeed = %f, want 100.0", PPSpeed(results))
	}

	// No results
	if TGSpeed(nil) != 0 {
		t.Errorf("TGSpeed(nil) = %f, want 0", TGSpeed(nil))
	}
	if PPSpeed(nil) != 0 {
		t.Errorf("PPSpeed(nil) = %f, want 0", PPSpeed(nil))
	}
}

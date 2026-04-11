package phase

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/config"
	"github.com/WagnerJust/llamaseye/hardware"
	"github.com/WagnerJust/llamaseye/output"
	"github.com/WagnerJust/llamaseye/state"
)

// mockExecutor for phase tests.
type mockPhaseExecutor struct {
	responses []phaseResponse
	idx       int
}

type phaseResponse struct {
	stdout   string
	stderr   string
	exitCode int
}

func (m *mockPhaseExecutor) Run(_ context.Context, _ string, _ []string, stdout, stderr io.Writer) (int, error) {
	i := m.idx
	if i >= len(m.responses) {
		_, _ = io.WriteString(stdout, okOutput())
		return 0, nil
	}
	m.idx++
	r := m.responses[i]
	if r.stdout != "" {
		_, _ = io.WriteString(stdout, r.stdout)
	}
	if r.stderr != "" {
		_, _ = io.WriteString(stderr, r.stderr)
	}
	return r.exitCode, nil
}

func okOutput() string {
	return `{"n_prompt":512,"n_gen":0,"avg_ts":100.0,"stddev_ts":1.0}` + "\n" +
		`{"n_prompt":0,"n_gen":128,"avg_ts":20.0,"stddev_ts":0.5}` + "\n"
}

func oomResponse() phaseResponse {
	return phaseResponse{stderr: "CUDA out of memory\n", exitCode: 1}
}

func newTestEnv(t *testing.T, exec bench.CommandExecutor) *PhaseEnv {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "raw"), 0755); err != nil {
		t.Fatal(err)
	}

	cfg := config.Defaults()
	cfg.DelaySeconds = 0
	cfg.TimeoutSec = 5
	cfg.NoThermal = true

	hw := &hardware.HardwareInfo{
		CPUPhysical: 8,
		CPULogical:  16,
		GPUCount:    1,
		GPUVRAMGiB:  24,
		Backend:     hardware.BackendCUDA,
	}

	logger, _ := output.NewLogger("")
	sel := &bench.BinarySelector{
		StandardBin:    "/fake/llama-bench",
		TurboAvailable: false,
	}
	runner := &bench.BenchRunner{
		Config:    cfg,
		Selector:  sel,
		Executor:  exec,
		OutputDir: dir,
		ModelPath: "/fake/model.gguf",
		ModelStem: "model",
	}

	env := NewPhaseEnv(cfg, hw, runner, nil, logger, dir, "/fake/model.gguf", "model")
	env.MaxNGL = 32
	env.Best.NGL = 32
	return env
}

func TestP0NGLProbe_NoGPU(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)
	env.HW.GPUCount = 0

	p := P0NGLProbe{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P0: %v", err)
	}
	if env.MaxNGL != 0 {
		t.Errorf("MaxNGL = %d, want 0 for no-GPU", env.MaxNGL)
	}
}

func TestP0NGLProbe_CapsAtNumLayers(t *testing.T) {
	// Model has 36 layers — Phase 0 should probe at 36, not 99.
	responses := []phaseResponse{
		{stdout: okOutput(), exitCode: 0}, // first probe (ngl=36) succeeds
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.NumLayers = 36

	p := P0NGLProbe{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P0: %v", err)
	}
	// Should succeed at 36 on the first try (no wasted probes at 99, 95, ...)
	if env.MaxNGL != 36 {
		t.Errorf("MaxNGL = %d, want 36 (capped at NumLayers)", env.MaxNGL)
	}
	// Only 1 call should have been made
	if exec.idx != 1 {
		t.Errorf("expected 1 probe call, got %d", exec.idx)
	}
}

func TestP0NGLProbe_FindsMax(t *testing.T) {
	// OOM at 99, 95, then success at 91
	responses := []phaseResponse{
		oomResponse(),
		oomResponse(),
		{stdout: okOutput(), exitCode: 0},
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)

	p := P0NGLProbe{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P0: %v", err)
	}
	if env.MaxNGL != 91 {
		t.Errorf("MaxNGL = %d, want 91", env.MaxNGL)
	}
	if env.Best.NGL != 91 {
		t.Errorf("Best.NGL = %d, want 91", env.Best.NGL)
	}
}

func TestP1NGLSweep_BuildsWorkingSet(t *testing.T) {
	exec := &mockPhaseExecutor{} // all OK
	env := newTestEnv(t, exec)
	env.MaxNGL = 8
	env.Config.NGLStep = 4
	env.Config.StartNGL = intPtr(0) // full sweep from 0

	p := P1NGLSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P1: %v", err)
	}
	if len(env.WS.NGL) == 0 {
		t.Error("expected non-empty WS.NGL")
	}
}

func TestP1NGLSweep_CapsAtNumLayers(t *testing.T) {
	// MaxNGL=32 (from Phase 0) but model only has 10 layers.
	// Phase 1 sweep should stop at 10, not 32.
	exec := &mockPhaseExecutor{} // all OK
	env := newTestEnv(t, exec)
	env.MaxNGL = 32
	env.NumLayers = 10
	env.Config.NGLStep = 4
	env.Config.StartNGL = intPtr(0) // full sweep from 0

	p := P1NGLSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P1: %v", err)
	}
	// No NGL value in the working set should exceed NumLayers
	for _, ngl := range env.WS.NGL {
		if ngl > env.NumLayers {
			t.Errorf("WS.NGL contains %d which exceeds NumLayers=%d", ngl, env.NumLayers)
		}
	}
	// Call count should be for [0,4,8,10] = 4 runs, not [0,4,...,32] = 9 runs
	if exec.idx > 5 {
		t.Errorf("expected ≤5 bench calls (capped at NumLayers=10), got %d", exec.idx)
	}
}

func TestP2FAKVSweep_BuildsWorkingSet(t *testing.T) {
	exec := &mockPhaseExecutor{} // all OK
	env := newTestEnv(t, exec)

	p := P2FAKVSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P2: %v", err)
	}
	if len(env.WS.FACTK) == 0 {
		t.Error("expected non-empty WS.FACTK")
	}
	// Standard combos without turbo = 5 candidates, but fa=0+q4_0 is invalid = 4 actually tested
	// All should succeed with mock
	if len(env.WS.FACTK) < 4 {
		t.Errorf("expected ≥4 FACTK combos, got %d", len(env.WS.FACTK))
	}
	// Independent CTK/CTV working sets must be populated
	if len(env.WS.CTKValues) == 0 {
		t.Error("expected non-empty WS.CTKValues after Phase 2")
	}
	if len(env.WS.CTVValues) == 0 {
		t.Error("expected non-empty WS.CTVValues after Phase 2")
	}
}

func TestP7CombinationMatrix_PrecisionFilter(t *testing.T) {
	// Phase 7 should skip combos where V is more precise than K.
	// With ctk=[q8_0] and ctv=[q8_0, f16], only (q8_0, q8_0) is valid;
	// (q8_0, f16) is invalid (V=f16 more precise than K=q8_0).
	// Provide exactly 1 response — if 2 runs happen the mock would default ok, but idx stays 1.
	exec := &mockPhaseExecutor{responses: []phaseResponse{{stdout: okOutput()}}}
	env := newTestEnv(t, exec)
	env.WS.NGL = []int{32}
	env.WS.CTKValues = []string{"q8_0"}
	env.WS.CTVValues = []string{"q8_0", "f16"} // f16 more precise than q8_0 — should be filtered
	env.WS.FACTK = []state.FACTKCombo{{FA: 1, CTK: "q8_0", CTV: "q8_0"}}
	env.WS.NKVO = []int{0}
	env.WS.Threads = state.ThreadValues{nil}
	env.WS.BUB = []state.BUBCombo{{B: 2048, UB: 512}}
	env.WS.CTX = []int{8192}
	env.Config.MinCTK = "q8_0"
	env.Config.MinCtx = intPtr(8192)

	p := P7CombinationMatrix{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P7: %v", err)
	}
	// Only 1 valid combo (q8_0, q8_0) — idx must be exactly 1
	if exec.idx != 1 {
		t.Errorf("expected 1 run (precision filter should drop (q8_0,f16)), got %d", exec.idx)
	}
}

func TestP7CombinationMatrix_IndependentKV(t *testing.T) {
	// With ctk=[f16, q8_0] and ctv=[f16, q8_0], valid combos are:
	// (f16,f16), (f16,q8_0), (q8_0,q8_0) — 3 total; (q8_0,f16) filtered (V more precise).
	// Provide exactly 3 responses so idx ends at 3 if the right count is run.
	responses := []phaseResponse{
		{stdout: okOutput()},
		{stdout: okOutput()},
		{stdout: okOutput()},
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.WS.NGL = []int{32}
	env.WS.CTKValues = []string{"f16", "q8_0"}
	env.WS.CTVValues = []string{"f16", "q8_0"}
	env.WS.FACTK = []state.FACTKCombo{
		{FA: 1, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "q8_0", CTV: "q8_0"},
	}
	env.WS.NKVO = []int{0}
	env.WS.Threads = state.ThreadValues{nil}
	env.WS.BUB = []state.BUBCombo{{B: 2048, UB: 512}}
	env.WS.CTX = []int{8192}
	env.Config.MinCTK = "q8_0"
	env.Config.MinCtx = intPtr(8192)

	p := P7CombinationMatrix{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P7: %v", err)
	}
	// (f16,f16), (f16,q8_0), (q8_0,q8_0) = 3 valid combos; idx must be 3
	if exec.idx != 3 {
		t.Errorf("expected 3 runs (independent KV cartesian product), got %d", exec.idx)
	}
}

func TestP3ThreadSweep_BuildsWorkingSet(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)

	p := P3ThreadSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P3: %v", err)
	}
	if len(env.WS.Threads) == 0 {
		t.Error("expected non-empty WS.Threads")
	}
}

func TestP4NKVOSweep_BuildsWorkingSet(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)

	p := P4NKVOSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P4: %v", err)
	}
	// Should have 0 and 1 in working set (plus extra NGL probes)
	if len(env.WS.NKVO) == 0 {
		t.Error("expected non-empty WS.NKVO")
	}
}

func TestP5BatchSweep_BuildsWorkingSet(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)

	p := P5BatchSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P5: %v", err)
	}
	if len(env.WS.BUB) == 0 {
		t.Error("expected non-empty WS.BUB")
	}
}

func TestP6CtxSweep_StopsOnOOM(t *testing.T) {
	// First few ctx sizes succeed, then OOM
	responses := []phaseResponse{
		{stdout: okOutput()},           // ctx=512
		{stdout: okOutput()},           // ctx=1024
		{stderr: "failed to allocate"}, // ctx=2048 OOM
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)

	// Minimal working sets needed for fallback logic
	env.WS.NKVO = []int{0}
	env.WS.FACTK = nil // no fallbacks

	p := P6CtxSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6: %v", err)
	}
	if len(env.WS.CTX) == 0 {
		t.Error("expected WS.CTX to have entries")
	}
	// Best CTX should be 1024 (last successful)
	if env.Best.CTX != 1024 {
		t.Errorf("Best.CTX = %d, want 1024", env.Best.CTX)
	}
}

func TestP6CtxSweep_VFirstFallback(t *testing.T) {
	// Primary fails at ctx=512; V-first fallback (same CTK, more-compressed CTV) succeeds.
	// ctx=1024 primary + its V-first fallback both OOM to stop the sweep at 512.
	// WS.FACTK has (f16,f16) and (f16,turbo3); no K+V fallbacks because CTK=f16 has no
	// more-compressed CTK entries in WS.FACTK.
	responses := []phaseResponse{
		{stderr: "failed to allocate"}, // ctx=512 primary OOM
		{stdout: okOutput()},           // V-first: ctk=f16, ctv=turbo3, nkvo=0 ok
		{stderr: "failed to allocate"}, // ctx=1024 primary OOM
		{stderr: "failed to allocate"}, // ctx=1024 V-first: ctk=f16, ctv=turbo3, nkvo=0 OOM
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)
	env.Best.CTK = "f16"
	env.Best.CTV = "f16"
	env.WS.NKVO = []int{0}
	// Phase 2 validated (f16, turbo3) as an asymmetric combo
	env.WS.FACTK = []state.FACTKCombo{
		{FA: 1, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "f16", CTV: "turbo3"},
	}
	env.Runner.Selector.TurboAvailable = true

	p := P6CtxSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6: %v", err)
	}
	if env.Best.CTX != 512 {
		t.Errorf("Best.CTX = %d, want 512 (V-first fallback should have succeeded)", env.Best.CTX)
	}
	if !containsInt(env.WS.CTX, 512) {
		t.Error("expected 512 in WS.CTX")
	}
}

func TestP6CtxSweep_KVFallbackAfterVExhausted(t *testing.T) {
	// V-first fallback OOMs; K+V fallback then succeeds at ctx=512.
	// ctx=1024 primary + all its fallbacks (V-first + K+V) OOM to stop sweep.
	// WS.FACTK has (f16,f16), (f16,turbo3), (q8_0,q8_0):
	//   V-first at 1024: (f16,turbo3) — 1 attempt
	//   K+V at 1024: (q8_0,q8_0) — 1 attempt
	responses := []phaseResponse{
		{stderr: "failed to allocate"}, // ctx=512 primary OOM
		{stderr: "failed to allocate"}, // ctx=512 V-first: ctk=f16, ctv=turbo3, nkvo=0 OOM
		{stdout: okOutput()},           // ctx=512 K+V: ctk=q8_0, ctv=q8_0, nkvo=0 ok
		{stderr: "failed to allocate"}, // ctx=1024 primary OOM
		{stderr: "failed to allocate"}, // ctx=1024 V-first: ctk=f16, ctv=turbo3, nkvo=0 OOM
		{stderr: "failed to allocate"}, // ctx=1024 K+V: ctk=q8_0, ctv=q8_0, nkvo=0 OOM
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)
	env.Best.CTK = "f16"
	env.Best.CTV = "f16"
	env.WS.NKVO = []int{0}
	env.WS.FACTK = []state.FACTKCombo{
		{FA: 1, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "f16", CTV: "turbo3"},
		{FA: 1, CTK: "q8_0", CTV: "q8_0"},
	}
	env.Runner.Selector.TurboAvailable = true

	p := P6CtxSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6: %v", err)
	}
	if env.Best.CTX != 512 {
		t.Errorf("Best.CTX = %d, want 512 (K+V fallback should have succeeded)", env.Best.CTX)
	}
	if !containsInt(env.WS.CTX, 512) {
		t.Error("expected 512 in WS.CTX")
	}
}

func TestP7CombinationMatrix_GoalEarlyExit(t *testing.T) {
	exec := &mockPhaseExecutor{} // all OK
	env := newTestEnv(t, exec)
	env.WS.NGL = []int{32}
	env.WS.FACTK = []state.FACTKCombo{{FA: 1, CTK: "f16", CTV: "f16"}}
	env.WS.NKVO = []int{0}
	env.WS.Threads = state.ThreadValues{nil}
	env.WS.BUB = []state.BUBCombo{{B: 2048, UB: 512}}
	env.WS.CTX = []int{4096, 8192, 16384, 32768, 65536}

	goal := &GoalConfig{
		CtxMin: 8192,
		TGMin:  10.0, // mock returns 20 t/s, so all qualify
		MaxHits: 2,
	}

	p := P7CombinationMatrix{Goal: goal}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P7: %v", err)
	}
	// With MaxHits=2 and 4 qualifying ctx values (8192, 16384, 32768, 65536),
	// Phase 7 should stop after 2 combos — not run all 4.
	if exec.idx > 2 {
		t.Errorf("goal early-exit: expected at most 2 runs, got %d", exec.idx)
	}
}

func intPtr(n int) *int {
	return &n
}

func TestPhaseIDAndLabel(t *testing.T) {
	phases := []Phase{
		P0NGLProbe{},
		P1NGLSweep{},
		P2FAKVSweep{},
		P3ThreadSweep{},
		P4NKVOSweep{},
		P5BatchSweep{},
		P6CtxSweep{},
		P7CombinationMatrix{},
	}
	for i, p := range phases {
		if p.ID() != i {
			t.Errorf("phase[%d].ID() = %d, want %d", i, p.ID(), i)
		}
		if p.Label() == "" {
			t.Errorf("phase[%d].Label() is empty", i)
		}
	}
}

func TestLoadFromState_ToState_RoundTrip(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)
	threads := 8
	env.Best.Threads = &threads
	env.Best.CTK = "q8_0"
	env.Best.CTX = 16384
	env.MaxNGL = 28
	env.WS.NGL = []int{24, 28, 32}
	env.WS.NKVO = []int{0, 1}
	env.WS.CTX = []int{8192, 16384}

	// Convert to state and back
	st := env.ToState([]int{0, 1, 2})
	if st.MaxNGL != 28 {
		t.Errorf("ToState MaxNGL = %d, want 28", st.MaxNGL)
	}
	if st.Best.CTK != "q8_0" {
		t.Errorf("ToState CTK = %q, want q8_0", st.Best.CTK)
	}
	if st.Best.Threads == nil || *st.Best.Threads != 8 {
		t.Error("ToState Threads not preserved")
	}
	if len(st.PhasesComplete) != 3 {
		t.Errorf("ToState PhasesComplete = %v, want [0,1,2]", st.PhasesComplete)
	}

	// Restore into a fresh env
	env2 := newTestEnv(t, exec)
	env2.LoadFromState(st)
	if env2.MaxNGL != 28 {
		t.Errorf("LoadFromState MaxNGL = %d, want 28", env2.MaxNGL)
	}
	if env2.Best.CTK != "q8_0" {
		t.Errorf("LoadFromState CTK = %q, want q8_0", env2.Best.CTK)
	}
	if env2.Best.Threads == nil || *env2.Best.Threads != 8 {
		t.Error("LoadFromState Threads not preserved")
	}
	if len(env2.WS.NGL) != 3 {
		t.Errorf("LoadFromState WS.NGL = %v, want [24,28,32]", env2.WS.NGL)
	}
}

func TestLoadFromState_NilThreads(t *testing.T) {
	exec := &mockPhaseExecutor{}
	env := newTestEnv(t, exec)
	// state with nil Threads
	st := env.ToState([]int{0})
	if st.Best.Threads != nil {
		t.Error("expected nil Threads in state when env.Best.Threads is nil")
	}
	env2 := newTestEnv(t, exec)
	env2.LoadFromState(st)
	if env2.Best.Threads != nil {
		t.Error("expected nil Threads after LoadFromState")
	}
}

func TestHelpers_SortIntsDesc(t *testing.T) {
	vals := []int{1, 5, 3, 8, 2}
	sortIntsDesc(vals)
	for i := 1; i < len(vals); i++ {
		if vals[i] > vals[i-1] {
			t.Errorf("sortIntsDesc: not sorted at index %d: %v", i, vals)
		}
	}
}


func TestHelpers_ContainsInt(t *testing.T) {
	if !containsInt([]int{1, 2, 3}, 2) {
		t.Error("containsInt should find 2")
	}
	if containsInt([]int{1, 2, 3}, 4) {
		t.Error("containsInt should not find 4")
	}
}

func TestP6CtxSweep_FineCTX(t *testing.T) {
	// First response succeeds, second OOMs — bisection should find midpoint
	responses := []phaseResponse{
		{stdout: okOutput()},            // ctx=512 ok
		{stdout: okOutput()},            // ctx=1024 ok
		{stderr: "failed to allocate"},  // ctx=2048 OOM
		{stdout: okOutput()},            // ctx=1536 (bisect) ok
		{stderr: "failed to allocate"},  // ctx=1792 (bisect) OOM
	}
	exec := &mockPhaseExecutor{responses: responses}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)
	env.Config.FineCtx = true
	env.WS.NKVO = []int{0}
	env.WS.FACTK = nil

	p := P6CtxSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6 fine ctx: %v", err)
	}
	if len(env.WS.CTX) == 0 {
		t.Error("expected WS.CTX non-empty after fine-ctx sweep")
	}
}

func TestP6CtxSweep_P6CTKOverride(t *testing.T) {
	// Verify that --p6-ctk pins Phase 6 to use the specified CTK instead of env.Best.CTK.
	// env.Best.CTK is f16 but we override to q8_0.
	exec := &mockPhaseExecutor{responses: []phaseResponse{
		{stdout: okOutput()}, // ctx=512
		{stdout: okOutput()}, // ctx=1024
	}}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)
	env.Config.P6CTK = "q8_0"
	env.Config.P6CTV = "turbo2"
	env.Best.CTK = "f16"
	env.Best.CTV = "f16"
	env.WS.NKVO = []int{0}
	env.WS.FACTK = []state.FACTKCombo{
		{FA: 1, CTK: "q8_0", CTV: "turbo2"},
	}
	env.Runner.Selector.TurboAvailable = true

	p := P6CtxSweep{}
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6 with override: %v", err)
	}
	// Phase 6 should have succeeded and populated WS.CTX
	if len(env.WS.CTX) == 0 {
		t.Error("expected WS.CTX non-empty with P6CTK override")
	}
	// env.Best.CTK should NOT be mutated by Phase 6's override
	if env.Best.CTK != "f16" {
		t.Errorf("Best.CTK mutated to %q, expected f16 (override should be local to Phase 6)", env.Best.CTK)
	}
}

func TestP6CtxSweep_P6CTKOverrideUnknownType(t *testing.T) {
	// An unknown CTK type should warn and fall back to env.Best.CTK without panicking.
	exec := &mockPhaseExecutor{responses: []phaseResponse{
		{stdout: okOutput()},
	}}
	env := newTestEnv(t, exec)
	env.Config.StartCtx = intPtr(512)
	env.Config.P6CTK = "bogustype"
	env.Best.CTK = "f16"
	env.Best.CTV = "f16"
	env.WS.NKVO = []int{0}
	env.WS.FACTK = nil

	p := P6CtxSweep{}
	// Should not panic or error — just warn and continue with f16
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P6 with unknown override type: %v", err)
	}
}

func TestP7CombinationMatrix_NoGoal(t *testing.T) {
	exec := &mockPhaseExecutor{} // all OK
	env := newTestEnv(t, exec)
	env.WS.NGL = []int{32}
	env.WS.FACTK = []state.FACTKCombo{{FA: 1, CTK: "f16", CTV: "f16"}}
	env.WS.NKVO = []int{0}
	env.WS.Threads = state.ThreadValues{nil}
	env.WS.BUB = []state.BUBCombo{{B: 2048, UB: 512}}
	env.WS.CTX = []int{4096}

	p := P7CombinationMatrix{} // no goal
	if err := p.Run(context.Background(), env); err != nil {
		t.Fatalf("P7 no goal: %v", err)
	}
}

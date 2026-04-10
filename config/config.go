// Package config defines the Config struct and handles env-var/CLI merge.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds all runtime configuration for a sweep run.
// Every field has a corresponding SWEEP_* env var and a --flag CLI counterpart.
type Config struct {
	// Binary paths
	LlamaBenchBin  string
	TurboBenchBin  string
	RotorBenchBin  string

	// Model selection
	ModelPath      string   // --model
	ModelsDir      string   // --models-dir
	ModelListFile  string   // --model-list

	// Output
	OutputDir string

	// Sweep tuning
	NGLStep      int
	Repetitions  int
	ProbeReps    int
	TimeoutSec   int
	MinTGTS      float64
	DelaySeconds int
	Priority     int

	// Thermal guard
	CPUTempLimit  int
	GPUTempLimit  int
	CoolPollSec   int
	NoThermal     bool

	// Execution control
	Resume      bool
	Overwrite   bool
	DryRun      bool
	NoConfirm   bool
	Report      bool
	OnlyPhases  []int
	SkipPhases  []int

	// Axis start points
	StartNGL     *int
	StartThreads *int
	StartCtx     *int
	StartCTK     string
	StartCTV     string
	StartB       *int
	StartUB      *int
	StartFA      *int

	// Axis directions ("up" or "down")
	DirNGL     string
	DirThreads string
	DirCtx     string
	DirCTK     string
	DirCTV     string
	DirB       string
	DirUB      string
	DirFA      string

	// Explicit CTV value list (comma-separated); takes precedence over StartCTV/DirCTV
	CTV string

	// Phase 7 minimums
	MinNGL     *int
	MinThreads *int
	MinCtx     *int
	MinCTK     string
	MinB       *int
	MinUB      *int

	// Goal-directed Phase 7
	Goal            string
	GoalTargetCount int
	GoalSort        string // sort key for Goal Results table: "tg" | "ctx" | "ngl" | "pp"

	// Fine-grained context sweep
	FineCtx      bool
	CtxStepMin   int

	// Optimized sweep
	OptimizedSweep bool

	// Asymmetric K/V quant combos in Phase 2
	AsymmetricKV bool

	// Focused mode: only run combos not already in sweep.jsonl
	Focused bool

	// Debug mode
	Debug bool
}

// envStr reads an env var with a fallback default.
func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envInt reads an env var as int with a fallback default.
func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return def
}

// envFloat reads an env var as float64 with a fallback default.
func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

// envBool reads an env var as bool with a fallback default.
func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		b, err := strconv.ParseBool(v)
		if err == nil {
			return b
		}
	}
	return def
}

// Defaults returns a Config populated from environment variables.
// CLI flags then override individual fields.
func Defaults() *Config {
	home := os.Getenv("HOME")
	return &Config{
		LlamaBenchBin:   envStr("LLAMA_BENCH_BIN", home+"/llama.cpp/build/bin/llama-bench"),
		TurboBenchBin:   envStr("SWEEP_TURBO_BENCH_BIN", ""),
		RotorBenchBin:   envStr("SWEEP_ROTOR_BENCH_BIN", ""),
		ModelsDir:       envStr("SWEEP_MODELS_DIR", ""),
		OutputDir:       envStr("SWEEP_OUTPUT_DIR", home+"/Models/bench/sweep"),
		NGLStep:         envInt("SWEEP_NGL_STEP", 4),
		Repetitions:     envInt("SWEEP_REPETITIONS", 3),
		ProbeReps:       envInt("SWEEP_PROBE_REPS", 1),
		TimeoutSec:      envInt("SWEEP_TIMEOUT_SEC", 600),
		MinTGTS:         envFloat("SWEEP_MIN_TG_TS", 2.0),
		CPUTempLimit:    envInt("SWEEP_CPU_TEMP_LIMIT", 88),
		GPUTempLimit:    envInt("SWEEP_GPU_TEMP_LIMIT", 81),
		CoolPollSec:     envInt("SWEEP_COOL_POLL_SEC", 20),
		DelaySeconds:    envInt("SWEEP_DELAY_SEC", 5),
		Priority:        envInt("SWEEP_PRIO", 2),
		Resume:          envBool("SWEEP_RESUME", false),
		Overwrite:       envBool("SWEEP_OVERWRITE", false),
		DryRun:          envBool("SWEEP_DRY_RUN", false),
		NoConfirm:       envBool("SWEEP_NO_CONFIRM", false),
		NoThermal:       envBool("SWEEP_NO_THERMAL", false),
		Report:          envBool("SWEEP_REPORT", false),
		ModelListFile:   envStr("SWEEP_MODEL_LIST", ""),
		OnlyPhases:      ParsePhaseList(os.Getenv("SWEEP_ONLY_PHASES")),
		SkipPhases:      ParsePhaseList(os.Getenv("SWEEP_SKIP_PHASES")),
		DirNGL:          envStr("SWEEP_NGL_DIR", "up"),
		DirThreads:      envStr("SWEEP_THREADS_DIR", "up"),
		DirCtx:          envStr("SWEEP_CTX_DIR", "up"),
		DirCTK:          envStr("SWEEP_CTK_DIR", "up"),
		DirCTV:          envStr("SWEEP_CTV_DIR", "up"),
		DirB:            envStr("SWEEP_B_DIR", "up"),
		DirUB:           envStr("SWEEP_UB_DIR", "up"),
		DirFA:           envStr("SWEEP_FA_DIR", "up"),
		MinCTK:          envStr("SWEEP_MIN_CTK", ""),
		Goal:            envStr("SWEEP_GOAL", ""),
		GoalTargetCount: envInt("SWEEP_GOAL_HITS", 3),
		GoalSort:        envStr("SWEEP_GOAL_SORT", "tg"),
		FineCtx:         envBool("SWEEP_FINE_CTX", false),
		CtxStepMin:      envInt("SWEEP_CTX_STEP_MIN", 8192),
		OptimizedSweep:  envBool("SWEEP_OPTIMIZED_SWEEP", false),
		CTV:             envStr("SWEEP_CTV", ""),
		AsymmetricKV:   envBool("SWEEP_ASYMMETRIC_KV", true),
		Focused:         envBool("SWEEP_FOCUSED", false),
		Debug:           envBool("SWEEP_DEBUG", false),
	}
}

// ParsePhaseList parses a comma-separated list of phase numbers (e.g. "0,2,5").
func ParsePhaseList(s string) []int {
	if s == "" {
		return nil
	}
	var result []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if n, err := strconv.Atoi(p); err == nil {
			result = append(result, n)
		}
	}
	return result
}

// PhaseInList returns true if phase is in the given list.
func PhaseInList(phase int, list []int) bool {
	for _, p := range list {
		if p == phase {
			return true
		}
	}
	return false
}

// Validate checks for invalid or conflicting flag combinations.
func (c *Config) Validate() error {
	if c.Resume && c.Overwrite {
		return fmt.Errorf("--resume and --overwrite are mutually exclusive")
	}
	if c.Focused && len(c.OnlyPhases) == 0 {
		return fmt.Errorf("--focused requires --only-phases")
	}
	if len(c.OnlyPhases) > 0 && len(c.SkipPhases) > 0 {
		return fmt.Errorf("--only-phases and --skip-phases are mutually exclusive")
	}
	if c.OptimizedSweep {
		var conflicts []string
		if c.StartNGL != nil {
			conflicts = append(conflicts, "--start-ngl")
		}
		if c.StartCtx != nil {
			conflicts = append(conflicts, "--start-ctx")
		}
		if c.StartCTK != "" {
			conflicts = append(conflicts, "--start-ctk")
		}
		if c.StartThreads != nil {
			conflicts = append(conflicts, "--start-threads")
		}
		if c.StartB != nil {
			conflicts = append(conflicts, "--start-b")
		}
		if c.StartUB != nil {
			conflicts = append(conflicts, "--start-ub")
		}
		if c.StartFA != nil {
			conflicts = append(conflicts, "--start-fa")
		}
		if c.MinNGL != nil {
			conflicts = append(conflicts, "--min-ngl")
		}
		if c.MinCtx != nil {
			conflicts = append(conflicts, "--min-ctx")
		}
		if c.MinCTK != "" {
			conflicts = append(conflicts, "--min-ctk")
		}
		if c.MinThreads != nil {
			conflicts = append(conflicts, "--min-threads")
		}
		if c.MinB != nil {
			conflicts = append(conflicts, "--min-b")
		}
		if c.MinUB != nil {
			conflicts = append(conflicts, "--min-ub")
		}
		if len(conflicts) > 0 {
			return fmt.Errorf("--optimized-sweep derives axis flags automatically and cannot be combined with: %s",
				strings.Join(conflicts, ", "))
		}
	}
	for _, d := range []string{c.DirNGL, c.DirThreads, c.DirCtx, c.DirCTK, c.DirCTV, c.DirB, c.DirUB, c.DirFA} {
		if d != "up" && d != "down" {
			return fmt.Errorf("direction flags must be 'up' or 'down', got %q", d)
		}
	}
	switch c.GoalSort {
	case "tg", "ctx", "ngl", "pp":
		// valid
	default:
		return fmt.Errorf("--goal-sort must be one of: tg, ctx, ngl, pp (got %q)", c.GoalSort)
	}
	return nil
}

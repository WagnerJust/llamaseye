// Package cmd implements the CLI flag definitions for llamaseye.
package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/WagnerJust/llamaseye/config"
	flag "github.com/spf13/pflag"
)

// Parse parses command-line arguments into a Config.
func Parse(args []string) (*config.Config, []string, error) {
	cfg := config.Defaults()
	fs := flag.NewFlagSet("llamaseye", flag.ContinueOnError)

	// Model selection
	var model, modelsDir, modelList string
	fs.StringVar(&model, "model", "", "Benchmark a single .gguf file")
	fs.StringVar(&modelsDir, "models-dir", cfg.ModelsDir, "Benchmark all .gguf files in DIR")
	fs.StringVar(&modelList, "model-list", cfg.ModelListFile, "Benchmark models listed in FILE")

	// Output
	fs.StringVar(&cfg.OutputDir, "output-dir", cfg.OutputDir, "Root directory for results")

	// Binary paths
	fs.StringVar(&cfg.LlamaBenchBin, "llama-bench", cfg.LlamaBenchBin, "llama-bench binary path")
	fs.StringVar(&cfg.TurboBenchBin, "turbo-bench", cfg.TurboBenchBin, "turbo-bench binary path (optional)")

	// Sweep tuning
	fs.IntVar(&cfg.NGLStep, "ngl-step", cfg.NGLStep, "NGL step size for phase 1")
	fs.IntVar(&cfg.Repetitions, "repetitions", cfg.Repetitions, "Bench reps per data point")
	fs.IntVar(&cfg.TimeoutSec, "timeout", cfg.TimeoutSec, "Per-run timeout (seconds)")

	// Thermal
	fs.IntVar(&cfg.CPUTempLimit, "cpu-temp-limit", cfg.CPUTempLimit, "CPU °C ceiling")
	fs.IntVar(&cfg.GPUTempLimit, "gpu-temp-limit", cfg.GPUTempLimit, "GPU °C ceiling")
	fs.BoolVar(&cfg.NoThermal, "no-thermal-guard", cfg.NoThermal, "Disable thermal polling")

	// Execution control
	fs.BoolVar(&cfg.Resume, "resume", cfg.Resume, "Skip already-completed phases")
	fs.BoolVar(&cfg.Overwrite, "overwrite", cfg.Overwrite, "Delete existing output dir before starting")
	fs.BoolVar(&cfg.DryRun, "dry-run", cfg.DryRun, "Print bench commands without executing")
	fs.BoolVar(&cfg.NoConfirm, "no-confirm", cfg.NoConfirm, "Skip pre-sweep confirmation")
	fs.BoolVar(&cfg.Report, "report", cfg.Report, "Regenerate sweep.md from existing .jsonl")

	var onlyPhases, skipPhases string
	fs.StringVar(&onlyPhases, "only-phases", "", "Run only these phases (comma-separated)")
	fs.StringVar(&skipPhases, "skip-phases", "", "Skip these phases (comma-separated)")

	// Axis start points
	var startNGL, startThreads, startCtx, startB, startUB, startFA string
	fs.StringVar(&startNGL, "start-ngl", "", "Begin NGL sweep at this value")
	fs.StringVar(&startThreads, "start-threads", "", "Begin thread sweep at this value")
	fs.StringVar(&startCtx, "start-ctx", "", "Begin context sweep at this size")
	fs.StringVar(&cfg.StartCTK, "start-ctk", cfg.StartCTK, "Begin KV quant sweep at this type")
	fs.StringVar(&startB, "start-b", "", "Begin batch size sweep at this value")
	fs.StringVar(&startUB, "start-ub", "", "Begin ubatch size sweep at this value")
	fs.StringVar(&startFA, "start-fa", "", "Begin FA sweep at this value (0 or 1)")

	// Axis directions
	fs.StringVar(&cfg.DirNGL, "ngl-dir", cfg.DirNGL, "NGL sweep direction (up|down)")
	fs.StringVar(&cfg.DirThreads, "threads-dir", cfg.DirThreads, "Thread sweep direction (up|down)")
	fs.StringVar(&cfg.DirCtx, "ctx-dir", cfg.DirCtx, "Context sweep direction (up|down)")
	fs.StringVar(&cfg.DirCTK, "ctk-dir", cfg.DirCTK, "KV type sweep direction (up|down)")
	fs.StringVar(&cfg.DirB, "b-dir", cfg.DirB, "Batch size sweep direction (up|down)")
	fs.StringVar(&cfg.DirUB, "ub-dir", cfg.DirUB, "Ubatch size sweep direction (up|down)")
	fs.StringVar(&cfg.DirFA, "fa-dir", cfg.DirFA, "FA sweep direction (up|down)")

	// Phase 7 minimums
	var minNGL, minThreads, minCtx, minB, minUB string
	fs.StringVar(&minNGL, "min-ngl", "", "Exclude NGL values below N from Phase 7")
	fs.StringVar(&minThreads, "min-threads", "", "Exclude thread counts below N from Phase 7")
	fs.StringVar(&minCtx, "min-ctx", "", "Exclude context sizes below N from Phase 7")
	fs.StringVar(&cfg.MinCTK, "min-ctk", cfg.MinCTK, "Exclude KV types below TYPE from Phase 7")
	fs.StringVar(&minB, "min-b", "", "Exclude batch sizes below N from Phase 7")
	fs.StringVar(&minUB, "min-ub", "", "Exclude ubatch sizes below N from Phase 7")

	// Goal and fine-ctx
	fs.StringVar(&cfg.Goal, "goal", cfg.Goal, `Goal-directed Phase 7 (e.g. "ctx=32768,tg=5")`)
	fs.BoolVar(&cfg.FineCtx, "fine-ctx", cfg.FineCtx, "Enable midpoint bisection in Phase 6")
	fs.IntVar(&cfg.CtxStepMin, "ctx-step-min", cfg.CtxStepMin, "Min bisection step for --fine-ctx")
	fs.BoolVar(&cfg.OptimizedSweep, "optimized-sweep", cfg.OptimizedSweep,
		"Parse GGUF metadata to derive start flags automatically")

	// --env-file is pre-consumed in main.go before Parse is called.
	// Register it here so it appears in --help output.
	fs.String("env-file", "", `Load environment variables from FILE before flag parsing (default: auto-load ".env" if present)`)

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "llamaseye v0.1.0 — exhaustive llama-bench parameter sweep harness\n\n")
		fmt.Fprintf(os.Stderr, "Usage: llamaseye [options]\n\n")
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return nil, nil, err
	}

	// Apply string → typed optionals
	cfg.ModelsDir = modelsDir
	cfg.ModelListFile = modelList
	if onlyPhases != "" {
		cfg.OnlyPhases = config.ParsePhaseList(onlyPhases)
	}
	if skipPhases != "" {
		cfg.SkipPhases = config.ParsePhaseList(skipPhases)
	}
	cfg.StartNGL = parseOptInt(startNGL)
	cfg.StartThreads = parseOptInt(startThreads)
	cfg.StartCtx = parseOptInt(startCtx)
	cfg.StartB = parseOptInt(startB)
	cfg.StartUB = parseOptInt(startUB)
	cfg.StartFA = parseOptInt(startFA)
	cfg.MinNGL = parseOptInt(minNGL)
	cfg.MinThreads = parseOptInt(minThreads)
	cfg.MinCtx = parseOptInt(minCtx)
	cfg.MinB = parseOptInt(minB)
	cfg.MinUB = parseOptInt(minUB)

	// Build model list from --model flag
	var models []string
	if model != "" {
		models = []string{model}
		cfg.ModelPath = model
	}

	return cfg, models, nil
}

// ResolveModels builds the final list of model paths to sweep.
func ResolveModels(cfg *config.Config, explicitModels []string) ([]string, error) {
	if len(explicitModels) > 0 {
		for _, m := range explicitModels {
			if err := validateModel(m); err != nil {
				return nil, err
			}
		}
		return explicitModels, nil
	}

	if cfg.ModelListFile != "" {
		return resolveModelList(cfg.ModelListFile, cfg.ModelsDir)
	}

	if cfg.ModelsDir != "" {
		return resolveModelsDir(cfg.ModelsDir)
	}

	return nil, fmt.Errorf("no models found. Use --model, --models-dir, or --model-list")
}

func resolveModelList(file, basedir string) ([]string, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("model list file: %w", err)
	}
	var models []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if !strings.HasPrefix(line, "/") && basedir != "" {
			line = strings.TrimRight(basedir, "/") + "/" + line
		}
		if err := validateModel(line); err != nil {
			return nil, err
		}
		models = append(models, line)
	}
	return models, nil
}

func resolveModelsDir(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("models dir: %w", err)
	}
	var models []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") {
			path := filepath.Join(dir, name)
			models = append(models, path)
		}
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("no .gguf files found in %s", dir)
	}
	return models, nil
}

func validateModel(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("model not found: %s", path)
	}
	if info.IsDir() {
		return fmt.Errorf("model path is a directory: %s", path)
	}
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return fmt.Errorf("model does not have .gguf extension: %s", path)
	}
	return nil
}

func parseOptInt(s string) *int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return nil
	}
	return &n
}

// ParsePhaseList is exported for use from config package.
func ParsePhaseList(s string) []int {
	return config.ParsePhaseList(s)
}


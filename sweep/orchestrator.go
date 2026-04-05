// Package sweep orchestrates the full benchmark sweep for one or more models.
package sweep

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/justinphilpott/llamaseye/bench"
	"github.com/justinphilpott/llamaseye/config"
	"github.com/justinphilpott/llamaseye/gguf"
	"github.com/justinphilpott/llamaseye/hardware"
	"github.com/justinphilpott/llamaseye/output"
	"github.com/justinphilpott/llamaseye/phase"
	"github.com/justinphilpott/llamaseye/state"
)

// Sweeper runs the full sweep pipeline.
type Sweeper struct {
	Config   *config.Config
	HW       *hardware.HardwareInfo
	Logger   *output.Logger
	Executor bench.CommandExecutor
}

// SweepModel runs all phases for a single model file.
func (s *Sweeper) SweepModel(ctx context.Context, modelPath string) error {
	modelStem := stemOf(modelPath)
	outputDir := filepath.Join(s.Config.OutputDir, modelStem)

	s.Logger.Log("===== Starting sweep: %s =====", modelStem)

	// Setup output directory
	if err := s.setupOutputDir(outputDir); err != nil {
		return err
	}

	// Logger now knows the output dir — re-open with log file
	logPath := filepath.Join(outputDir, "sweep.log")
	logger, err := output.NewLogger(logPath)
	if err != nil {
		return fmt.Errorf("open sweep.log: %w", err)
	}
	defer logger.Close()
	s.Logger = logger

	// Write hardware.json
	if err := output.WriteHardwareJSON(outputDir, s.HW); err != nil {
		s.Logger.Warn("write hardware.json: %v", err)
	}

	// Build runner
	sel := &bench.BinarySelector{
		StandardBin:    s.Config.LlamaBenchBin,
		TurboBin:       s.Config.TurboBenchBin,
		TurboAvailable: s.Config.TurboBenchBin != "" && detectTurbo(s.Config.TurboBenchBin),
	}
	runner := &bench.BenchRunner{
		Config:    s.Config,
		Selector:  sel,
		Executor:  s.Executor,
		OutputDir: outputDir,
		ModelPath: modelPath,
		ModelStem: modelStem,
	}

	// Build thermal monitor
	thermal := &hardware.ThermalMonitor{
		HW:          s.HW,
		CPULimit:    s.Config.CPUTempLimit,
		GPULimit:    s.Config.GPUTempLimit,
		PollSeconds: s.Config.CoolPollSec,
		Disabled:    s.Config.NoThermal,
		Log:         s.Logger.Log,
	}

	// Build PhaseEnv
	env := phase.NewPhaseEnv(s.Config, s.HW, runner, thermal, s.Logger,
		outputDir, modelPath, modelStem)

	// Parse GGUF metadata to cap NGL at the model's actual layer count.
	// Values above NumLayers are functionally identical (llama.cpp clamps silently).
	if meta, err := gguf.Parse(modelPath); err == nil && meta.NumLayers > 0 {
		env.NumLayers = meta.NumLayers
		s.Logger.Log("[GGUF] %d layers, %s architecture", meta.NumLayers, meta.Architecture)
	} else if err != nil {
		s.Logger.Log("[GGUF] Could not parse metadata (%v) — NGL ceiling defaults to 99", err)
	}

	// Load existing state if available
	var phasesComplete []int
	savedState, err := state.Load(outputDir)
	if err != nil {
		s.Logger.Warn("load state.json: %v", err)
	}
	if savedState != nil {
		env.LoadFromState(savedState)
		if s.Config.Resume {
			phasesComplete = savedState.PhasesComplete
			s.Logger.Log("[STATE] Resuming — phases complete: %v", phasesComplete)
		} else {
			s.Logger.Log("[STATE] Loaded prior working sets from state.json")
		}
	}

	// Build goal config
	var goal *phase.GoalConfig
	if s.Config.Goal != "" {
		goal = parseGoal(s.Config.Goal)
	}

	// Define all phases
	phases := []phase.Phase{
		phase.P0NGLProbe{},
		phase.P1NGLSweep{},
		phase.P2FAKVSweep{},
		phase.P3ThreadSweep{},
		phase.P4NKVOSweep{},
		phase.P5BatchSweep{},
		phase.P6CtxSweep{},
		phase.P7CombinationMatrix{Goal: goal},
	}

	for _, p := range phases {
		phaseID := p.ID()

		// --only-phases filter
		if len(s.Config.OnlyPhases) > 0 && !config.PhaseInList(phaseID, s.Config.OnlyPhases) {
			s.Logger.Log("[Phase %d] Skipped (not in --only-phases)", phaseID)
			continue
		}
		// --skip-phases filter
		if config.PhaseInList(phaseID, s.Config.SkipPhases) {
			s.Logger.Log("[Phase %d] Skipped (in --skip-phases)", phaseID)
			continue
		}
		// Resume: skip already-completed phases unless explicitly requested
		if config.PhaseInList(phaseID, phasesComplete) {
			if len(s.Config.OnlyPhases) > 0 && config.PhaseInList(phaseID, s.Config.OnlyPhases) {
				s.Logger.Log("[Phase %d] Already complete — re-running (explicitly requested via --only-phases)", phaseID)
			} else {
				s.Logger.Log("[Phase %d] Already complete — skipping (--resume)", phaseID)
				continue
			}
		}

		if err := p.Run(ctx, env); err != nil {
			return fmt.Errorf("phase %d: %w", phaseID, err)
		}

		// Save state
		phasesComplete = appendUnique(phasesComplete, phaseID)
		if err := state.Save(outputDir, env.ToState(phasesComplete)); err != nil {
			s.Logger.Warn("save state.json: %v", err)
		}
	}

	// Generate markdown
	if err := output.GenerateMarkdown(outputDir, modelStem, s.Config.Goal, s.Config.TimeoutSec); err != nil {
		s.Logger.Warn("generate markdown: %v", err)
	}

	printSummary(s.Logger, env)
	s.Logger.Log("===== Sweep complete: %s =====", modelStem)
	return nil
}

// ReportMode regenerates markdown files without running benchmarks.
func (s *Sweeper) ReportMode(stems []string) error {
	if len(stems) == 0 {
		// Scan output dir for subdirs with sweep.jsonl
		entries, err := os.ReadDir(s.Config.OutputDir)
		if err != nil {
			return fmt.Errorf("--report: read output dir: %w", err)
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			if _, err := os.Stat(filepath.Join(s.Config.OutputDir, e.Name(), "sweep.jsonl")); err == nil {
				stems = append(stems, e.Name())
			}
		}
	}
	if len(stems) == 0 {
		return fmt.Errorf("--report: no sweep.jsonl files found under %s", s.Config.OutputDir)
	}
	for _, stem := range stems {
		dir := filepath.Join(s.Config.OutputDir, stem)
		if err := output.GenerateMarkdown(dir, stem, s.Config.Goal, s.Config.TimeoutSec); err != nil {
			s.Logger.Warn("generate markdown for %s: %v", stem, err)
			continue
		}
		fmt.Printf("Regenerated: %s/sweep.md\n", dir)
	}
	return output.GenerateCrossModelSummary(s.Config.OutputDir, stems)
}

func (s *Sweeper) setupOutputDir(outputDir string) error {
	if _, err := os.Stat(outputDir); err == nil {
		if s.Config.Overwrite {
			if err := os.RemoveAll(outputDir); err != nil {
				return err
			}
			s.Logger.Log("Overwrite: removed existing output dir")
		} else if !s.Config.Resume && len(s.Config.OnlyPhases) == 0 {
			return fmt.Errorf("output dir already exists: %s\n  Use --resume to continue, --only-phases to re-run specific phases, or --overwrite to start fresh", outputDir)
		}
	}
	return os.MkdirAll(filepath.Join(outputDir, "raw"), 0755)
}

func stemOf(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, ".gguf")
}

func appendUnique(s []int, v int) []int {
	for _, e := range s {
		if e == v {
			return s
		}
	}
	return append(s, v)
}

func parseGoal(spec string) *phase.GoalConfig {
	g := &phase.GoalConfig{MaxHits: 3}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			continue
		}
		switch kv[0] {
		case "ctx":
			fmt.Sscanf(kv[1], "%d", &g.CtxMin)
		case "tg":
			fmt.Sscanf(kv[1], "%f", &g.TGMin)
		case "pp":
			fmt.Sscanf(kv[1], "%f", &g.PPMin)
		}
	}
	return g
}

func detectTurbo(path string) bool {
	if path == "" {
		return false
	}
	if _, err := os.Stat(path); err != nil {
		return false
	}
	// Check executable
	info, err := os.Stat(path)
	if err != nil || info.Mode()&0111 == 0 {
		return false
	}
	return true
}

func printSummary(logger *output.Logger, env *phase.PhaseEnv) {
	logger.Log("Sweep complete: %s", env.ModelStem)
	logger.Log("Best config: ngl=%d fa=%d ctk=%s nkvo=%d",
		env.Best.NGL, env.Best.FA, env.Best.CTK, env.Best.NKVO)
	logger.Log("Best context: %d tokens", env.Best.CTX)
	logger.Log("Output dir: %s", env.OutputDir)
}

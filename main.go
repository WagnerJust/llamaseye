package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/justinphilpott/llamaseye/bench"
	"github.com/justinphilpott/llamaseye/cmd"
	"github.com/justinphilpott/llamaseye/hardware"
	"github.com/justinphilpott/llamaseye/output"
	"github.com/justinphilpott/llamaseye/sweep"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "llamaseye: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	cfg, models, err := cmd.Parse(args)
	if err != nil {
		return err
	}

	if err := cfg.Validate(); err != nil {
		return err
	}

	logger, err := output.NewLogger("") // no log file yet (before output dir is known)
	if err != nil {
		return err
	}
	defer logger.Close()

	// --report mode: no benchmarks, just regenerate markdown
	if cfg.Report {
		hw := &hardware.HardwareInfo{} // not needed for report mode
		s := &sweep.Sweeper{
			Config:   cfg,
			HW:       hw,
			Logger:   logger,
			Executor: bench.OSExecutor{},
		}
		var stems []string
		if len(models) > 0 {
			for _, m := range models {
				stems = append(stems, stemOf(m))
			}
		} else if cfg.ModelsDir != "" {
			resolved, _ := cmd.ResolveModels(cfg, nil)
			for _, m := range resolved {
				stems = append(stems, stemOf(m))
			}
		}
		return s.ReportMode(stems)
	}

	// Resolve models
	if len(models) == 0 {
		models, err = cmd.ResolveModels(cfg, nil)
		if err != nil {
			return err
		}
	}

	// Detect hardware
	hw, err := hardware.Detect()
	if err != nil {
		return fmt.Errorf("hardware detection: %w", err)
	}
	logger.Log("[HW] CPU: %s (%dP/%dL)  RAM: %d GiB",
		hw.CPUModel, hw.CPUPhysical, hw.CPULogical, hw.RAMGiB)
	logger.Log("[HW] GPU: %s  VRAM: %d GiB  Backend: %s",
		hw.GPUModel, hw.GPUVRAMGiB, hw.Backend)

	// Validate binary
	if _, err := os.Stat(cfg.LlamaBenchBin); err != nil {
		return fmt.Errorf("llama-bench not found: %s", cfg.LlamaBenchBin)
	}

	printHardwareSummary(hw, cfg)

	// Pre-sweep confirmation
	if !cfg.NoConfirm && !cfg.DryRun {
		fmt.Printf("\nReady to sweep %d model(s). Output -> %s\n", len(models), cfg.OutputDir)
		fmt.Print("Continue? [y/N] ")
		var reply string
		fmt.Scan(&reply)
		if !strings.EqualFold(reply, "y") {
			return fmt.Errorf("aborted by user")
		}
	}

	s := &sweep.Sweeper{
		Config:   cfg,
		HW:       hw,
		Logger:   logger,
		Executor: bench.OSExecutor{},
	}

	ctx := context.Background()
	var swept []string
	for _, modelPath := range models {
		if err := s.SweepModel(ctx, modelPath); err != nil {
			logger.Warn("sweep failed for %s: %v", modelPath, err)
			continue
		}
		swept = append(swept, stemOf(modelPath))
	}

	// Cross-model summary
	if len(swept) > 1 {
		_ = output.GenerateCrossModelSummary(cfg.OutputDir, swept)
	}

	logger.Log("All sweeps complete.")
	return nil
}

func stemOf(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, ".gguf")
}

func printHardwareSummary(hw *hardware.HardwareInfo, _ interface{}) {
	fmt.Println()
	fmt.Println("┌─────────────────────────────────────────────────────┐")
	fmt.Println("│  Hardware Summary                                   │")
	fmt.Println("├────────────────────┬────────────────────────────────┤")
	fmt.Printf("│ %-18s │ %-30s │\n", "CPU", truncStr(hw.CPUModel, 30))
	fmt.Printf("│ %-18s │ %-30s │\n", "Cores", fmt.Sprintf("%dP / %dL", hw.CPUPhysical, hw.CPULogical))
	fmt.Printf("│ %-18s │ %-30s │\n", "RAM", fmt.Sprintf("%d GiB (%d GiB free)", hw.RAMGiB, hw.RAMFreeGiB))
	fmt.Printf("│ %-18s │ %-30s │\n", "GPU", truncStr(hw.GPUModel, 30))
	fmt.Printf("│ %-18s │ %-30s │\n", "VRAM", fmt.Sprintf("%d GiB (%d GiB free)", hw.GPUVRAMGiB, hw.GPUVRAMFreeGiB))
	fmt.Printf("│ %-18s │ %-30s │\n", "Backend", string(hw.Backend))
	fmt.Println("└────────────────────┴────────────────────────────────┘")
	fmt.Println()
}

func truncStr(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

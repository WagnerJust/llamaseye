package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/cmd"
	"github.com/WagnerJust/llamaseye/envfile"
	"github.com/WagnerJust/llamaseye/hardware"
	"github.com/WagnerJust/llamaseye/output"
	"github.com/WagnerJust/llamaseye/sweep"
)

// version is set at build time via -ldflags "-X main.version=vX.Y.Z"
var version = "dev"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "llamaseye: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	// Load .env before parsing flags so SWEEP_* vars are visible to config.Defaults().
	// --env-file is pre-scanned from args; full flag parsing happens in cmd.Parse below.
	envPath, args := extractEnvFileFlag(args)
	if envPath != "" {
		if err := envfile.Load(envPath); err != nil {
			return fmt.Errorf("--env-file %s: %w", envPath, err)
		}
	} else {
		// Auto-load .env from working directory if present
		if err := envfile.LoadIfExists(".env"); err != nil {
			return fmt.Errorf(".env: %w", err)
		}
	}

	// Handle --version before full flag parsing
	for _, a := range args {
		if a == "--version" || a == "-version" {
			fmt.Printf("llamaseye %s\n", version)
			return nil
		}
	}

	// Subcommand: install-skill (writes skills/llamaseye.md into the
	// directories that coding-agent tools read). Dispatched before the
	// main flag parser since it has its own flag set.
	if cmd.IsInstallSkillSubcommand(args) {
		return cmd.RunInstallSkill(args[1:], os.Stdout, os.Stderr)
	}

	cfg, models, err := cmd.Parse(args, version)
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
	logger.Debug = cfg.Debug

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

	printHardwareSummary(hw)

	// Pre-sweep confirmation
	if !cfg.NoConfirm && !cfg.DryRun {
		fmt.Printf("\nReady to sweep %d model(s). Output -> %s\n", len(models), cfg.OutputDir)
		fmt.Print("Continue? [y/N] ")
		var reply string
		_, _ = fmt.Scan(&reply)
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

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var swept []string
	for _, modelPath := range models {
		if err := s.SweepModel(ctx, modelPath); err != nil {
			if ctx.Err() != nil {
				logger.Log("Interrupted — saving state and exiting")
				break
			}
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

// extractEnvFileFlag scans args for --env-file <path> or --env-file=<path>,
// removes those tokens from the slice, and returns the path and remaining args.
func extractEnvFileFlag(args []string) (path string, remaining []string) {
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--env-file" && i+1 < len(args) {
			path = args[i+1]
			remaining = append(remaining, args[:i]...)
			remaining = append(remaining, args[i+2:]...)
			return
		}
		if strings.HasPrefix(arg, "--env-file=") {
			path = strings.TrimPrefix(arg, "--env-file=")
			remaining = append(remaining, args[:i]...)
			remaining = append(remaining, args[i+1:]...)
			return
		}
	}
	return "", args
}

func stemOf(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, ".gguf")
}

func printHardwareSummary(hw *hardware.HardwareInfo) {
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

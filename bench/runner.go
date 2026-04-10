package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/WagnerJust/llamaseye/config"
)

// CommandExecutor is the testability seam for running external processes.
type CommandExecutor interface {
	Run(ctx context.Context, binary string, args []string, stdout, stderr io.Writer) (exitCode int, err error)
}

// DebugLogger is satisfied by any logger that supports formatted debug output.
type DebugLogger interface {
	Debugf(string, ...any)
}

// RunParams holds all parameters for a single bench run.
type RunParams struct {
	NGL     int
	FA      int
	CTK     string
	CTV     string
	NKVO    int
	Threads *int   // nil = system default (no -t flag)
	B       int
	UB      int
	NPrompt int
	NGen    int
	Reps    int
	Phase   int
	PhaseLabel string
}

// BenchRunner wraps CommandExecutor with all the sweep-specific logic.
type BenchRunner struct {
	Config   *config.Config
	Selector *BinarySelector
	Executor CommandExecutor
	OutputDir string // per-model output directory (must have "raw/" subdir)
	ModelPath string
	ModelStem string
	Logger    DebugLogger // optional; nil = no debug output
}

// llamaBenchLine is a single JSON record emitted by llama-bench -o jsonl.
type llamaBenchLine struct {
	NPrompt  int     `json:"n_prompt"`
	NGen     int     `json:"n_gen"`
	AvgTS    float64 `json:"avg_ts"`
	StddevTS float64 `json:"stddev_ts"`
}

// RunBench executes a single llama-bench invocation and returns the result.
// The provided ctx is used as the parent for the per-run timeout, allowing
// callers to cancel in-flight benchmarks (e.g. on SIGINT).
func (r *BenchRunner) RunBench(ctx context.Context, label string, p RunParams) (*RunResult, error) {
	binary, _, err := r.Selector.Select(p.CTK, p.CTV)
	if err != nil {
		return nil, err
	}

	runID := genRunID()
	rawDir := filepath.Join(r.OutputDir, "raw")
	rawFile := filepath.Join(rawDir, runID+".txt")
	stderrFile := filepath.Join(rawDir, runID+".err")

	// Build args
	threadsDisplay := "sys"
	args := []string{
		"-m", r.ModelPath,
		"-ngl", strconv.Itoa(p.NGL),
		"-fa", strconv.Itoa(p.FA),
		"-ctk", p.CTK, "-ctv", p.CTV,
		"-nkvo", strconv.Itoa(p.NKVO),
	}
	if p.Threads != nil {
		args = append(args, "-t", strconv.Itoa(*p.Threads))
		threadsDisplay = strconv.Itoa(*p.Threads)
	}
	args = append(args,
		"-b", strconv.Itoa(p.B),
		"-ub", strconv.Itoa(p.UB),
		"-p", strconv.Itoa(p.NPrompt),
		"-n", strconv.Itoa(p.NGen),
		"-r", strconv.Itoa(p.Reps),
		"-o", "jsonl",
		"--prio", strconv.Itoa(r.Config.Priority),
	)

	if r.Logger != nil {
		r.Logger.Debugf("cmd: %s %s (threads=%s)", binary, strings.Join(args, " "), threadsDisplay)
	}

	if r.Config.DryRun {
		return &RunResult{RunID: runID, Status: StatusDryRun}, nil
	}

	// Thermal wait happens in the orchestrator before calling RunBench

	// Delay between runs
	if r.Config.DelaySeconds > 0 {
		time.Sleep(time.Duration(r.Config.DelaySeconds) * time.Second)
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	ctx, cancel := context.WithTimeout(ctx,
		time.Duration(r.Config.TimeoutSec)*time.Second)
	defer cancel()

	start := time.Now()
	exitCode, execErr := r.Executor.Run(ctx, binary, args, &stdoutBuf, &stderrBuf)
	wallTime := time.Since(start).Seconds()

	// Write raw files
	_ = os.WriteFile(rawFile, stdoutBuf.Bytes(), 0644)
	_ = os.WriteFile(stderrFile, stderrBuf.Bytes(), 0644)

	if r.Logger != nil {
		if stdoutBuf.Len() > 0 {
			r.Logger.Debugf("stdout (%d bytes): %s", stdoutBuf.Len(), string(truncate(stdoutBuf.Bytes(), 500)))
		}
		if stderrBuf.Len() > 0 {
			r.Logger.Debugf("stderr (%d bytes): %s", stderrBuf.Len(), string(truncate(stderrBuf.Bytes(), 500)))
		}
	}

	// Check for timeout (context deadline exceeded => exitCode 124 in bash equiv)
	if execErr != nil && ctx.Err() == context.DeadlineExceeded {
		errSnip := truncate(stderrBuf.Bytes(), 400)
		return &RunResult{
			RunID:        runID,
			Status:       StatusTimeout,
			WallTimeSec:  wallTime,
			ErrorSnippet: string(errSnip),
		}, nil
	}

	// OOM detection — combine stdout+stderr
	combined := append(stdoutBuf.Bytes(), stderrBuf.Bytes()...)
	if DetectOOMBytes(combined) {
		errSnip := extractOOMSnippet(combined)
		if r.Logger != nil {
			r.Logger.Debugf("OOM detected — matched: %s", errSnip)
		}
		return &RunResult{
			RunID:        runID,
			Status:       StatusOOM,
			WallTimeSec:  wallTime,
			ErrorSnippet: errSnip,
		}, nil
	}

	// Parse JSON lines from stdout
	results, parseErr := parseLlamaBenchOutput(stdoutBuf.Bytes())

	// Check for empty output + non-zero exit
	if len(results) == 0 && exitCode != 0 {
		errSnip := truncate(stderrBuf.Bytes(), 400)
		return &RunResult{
			RunID:        runID,
			Status:       StatusError,
			WallTimeSec:  wallTime,
			ErrorSnippet: string(errSnip),
		}, nil
	}

	// Empty results with zero exit = parse failure (malformed output)
	if parseErr != nil || (len(results) == 0 && exitCode == 0) {
		errMsg := "no results parsed from llama-bench output"
		if parseErr != nil {
			errMsg = parseErr.Error()
		}
		if r.Logger != nil {
			r.Logger.Debugf("parse issue: %s", errMsg)
		}
		if len(results) == 0 {
			return &RunResult{
				RunID:        runID,
				Status:       StatusError,
				WallTimeSec:  wallTime,
				ErrorSnippet: errMsg,
			}, nil
		}
	}

	return &RunResult{
		RunID:         runID,
		Status:        StatusOK,
		WallTimeSec:   wallTime,
		Results:       results,
		RawOutputFile: "raw/" + runID + ".txt",
	}, nil
}

// parseLlamaBenchOutput parses llama-bench -o jsonl output lines into TestResult records.
func parseLlamaBenchOutput(data []byte) ([]TestResult, error) {
	var results []TestResult
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec llamaBenchLine
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue // skip non-JSON lines (headers, etc.)
		}
		if rec.NGen == 0 && rec.NPrompt > 0 {
			results = append(results, TestResult{
				Test:     "pp",
				NPrompt:  rec.NPrompt,
				AvgTS:    rec.AvgTS,
				StddevTS: rec.StddevTS,
			})
		} else if rec.NGen > 0 {
			results = append(results, TestResult{
				Test:     "tg",
				NGen:     rec.NGen,
				AvgTS:    rec.AvgTS,
				StddevTS: rec.StddevTS,
			})
		}
	}
	return results, scanner.Err()
}

// TGSpeed returns the TG avg_ts from results, or 0 if not present.
func TGSpeed(results []TestResult) float64 {
	for _, r := range results {
		if r.Test == "tg" {
			return r.AvgTS
		}
	}
	return 0
}

// PPSpeed returns the PP avg_ts from results, or 0 if not present.
func PPSpeed(results []TestResult) float64 {
	for _, r := range results {
		if r.Test == "pp" {
			return r.AvgTS
		}
	}
	return 0
}

// genRunID returns an 8-hex-char random run identifier.
func genRunID() string {
	b := rand.Uint32()
	return fmt.Sprintf("%08x", b)
}

// truncate returns at most n bytes from data.
func truncate(data []byte, n int) []byte {
	if len(data) <= n {
		return data
	}
	return data[:n]
}

// extractOOMSnippet returns the first matching OOM line, truncated.
func extractOOMSnippet(data []byte) string {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		lower := strings.ToLower(line)
		if strings.Contains(lower, "out of memory") ||
			strings.Contains(lower, "cuda error") ||
			strings.Contains(lower, "failed to allocate") ||
			strings.Contains(lower, "killed") {
			if len(line) > 400 {
				line = line[:400]
			}
			return line
		}
	}
	// fallback: first 400 bytes
	return string(truncate(data, 400))
}

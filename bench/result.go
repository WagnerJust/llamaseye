// Package bench handles llama-bench invocation, result parsing, and OOM detection.
package bench

// Status represents the outcome of a single llama-bench run.
type Status string

const (
	StatusOK      Status = "ok"
	StatusOOM     Status = "oom"
	StatusTimeout Status = "timeout"
	StatusError   Status = "error"
	StatusDryRun  Status = "dry-run"
)

// TestResult holds parsed throughput numbers for a single test type (pp or tg).
type TestResult struct {
	Test    string  // "pp" or "tg"
	NPrompt int
	NGen    int
	AvgTS   float64
	StddevTS float64
}

// RunResult is the outcome of a single run_bench invocation.
type RunResult struct {
	RunID         string
	Status        Status
	WallTimeSec   float64
	Results       []TestResult
	ErrorSnippet  string
	RawOutputFile string // relative path like "raw/<run_id>.txt"
}

package output

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/justinphilpott/llamaseye/bench"
)

func TestAppendRecord_OK(t *testing.T) {
	dir := t.TempDir()
	threads := 8
	p := JSONLParams{
		NGL:     32,
		FA:      1,
		CTK:     "q8_0",
		CTV:     "q8_0",
		NKVO:    0,
		Threads: &threads,
		B:       2048,
		UB:      512,
		NPrompt: 512,
		NGen:    128,
		Repetitions: 3,
	}
	res := &bench.RunResult{
		RunID:  "abcd1234",
		Status: bench.StatusOK,
		Results: []bench.TestResult{
			{Test: "pp", NPrompt: 512, AvgTS: 100.0, StddevTS: 1.0},
			{Test: "tg", NGen: 128, AvgTS: 25.0, StddevTS: 0.5},
		},
		RawOutputFile: "raw/abcd1234.txt",
		WallTimeSec:   12.5,
	}
	if err := AppendRecord(dir, "/models/m.gguf", "m", p, res, 1, "ngl_sweep", "standard"); err != nil {
		t.Fatalf("AppendRecord: %v", err)
	}

	// Read back and verify
	data, err := os.ReadFile(filepath.Join(dir, "sweep.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var rec JSONLRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if rec.RunID != "abcd1234" {
		t.Errorf("RunID = %q, want abcd1234", rec.RunID)
	}
	if rec.Phase != 1 {
		t.Errorf("Phase = %d, want 1", rec.Phase)
	}
	if rec.PhaseLabel != "ngl_sweep" {
		t.Errorf("PhaseLabel = %q, want ngl_sweep", rec.PhaseLabel)
	}
	if rec.Status != "ok" {
		t.Errorf("Status = %q, want ok", rec.Status)
	}
	if rec.Params.NGL != 32 {
		t.Errorf("Params.NGL = %d, want 32", rec.Params.NGL)
	}
	if rec.Params.Threads == nil || *rec.Params.Threads != 8 {
		t.Errorf("Params.Threads = %v, want &8", rec.Params.Threads)
	}
	if len(rec.Results) != 2 {
		t.Fatalf("Results len = %d, want 2", len(rec.Results))
	}
	if rec.Results[0].Test != "pp" || rec.Results[0].AvgTS != 100.0 {
		t.Errorf("Results[0] = %+v", rec.Results[0])
	}
}

func TestAppendRecord_OOM(t *testing.T) {
	dir := t.TempDir()
	p := JSONLParams{NGL: 99, FA: 0, CTK: "f16", CTV: "f16", B: 2048, UB: 512, NPrompt: 512, NGen: 128, Repetitions: 1}
	res := &bench.RunResult{
		RunID:        "deadbeef",
		Status:       bench.StatusOOM,
		ErrorSnippet: "CUDA out of memory",
	}
	if err := AppendRecord(dir, "/m.gguf", "m", p, res, 0, "ngl_probe", "standard"); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "sweep.jsonl"))
	var rec JSONLRecord
	_ = json.Unmarshal(data, &rec)
	if rec.Status != "oom" {
		t.Errorf("Status = %q, want oom", rec.Status)
	}
	if rec.ErrorSnippet == nil || *rec.ErrorSnippet != "CUDA out of memory" {
		t.Errorf("ErrorSnippet = %v", rec.ErrorSnippet)
	}
	// Results should be empty array, not null
	if rec.Results == nil {
		t.Error("Results should be non-nil (empty array)")
	}
}

func TestAppendRecord_MultipleLines(t *testing.T) {
	dir := t.TempDir()
	p := JSONLParams{NGL: 10, FA: 0, CTK: "f16", CTV: "f16", B: 2048, UB: 512, NPrompt: 512, NGen: 128, Repetitions: 1}
	for i := 0; i < 3; i++ {
		res := &bench.RunResult{RunID: "run000" + string(rune('0'+i)), Status: bench.StatusOK}
		_ = AppendRecord(dir, "/m.gguf", "m", p, res, 1, "ngl_sweep", "standard")
	}
	data, _ := os.ReadFile(filepath.Join(dir, "sweep.jsonl"))
	lines := splitLines(data)
	if len(lines) != 3 {
		t.Errorf("expected 3 lines, got %d", len(lines))
	}
}

func splitLines(data []byte) []string {
	var lines []string
	start := 0
	for i, b := range data {
		if b == '\n' {
			if i > start {
				lines = append(lines, string(data[start:i]))
			}
			start = i + 1
		}
	}
	return lines
}

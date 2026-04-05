package output

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/justinphilpott/llamaseye/bench"
)

// JSONLParams holds all benchmark parameters to be recorded.
type JSONLParams struct {
	NGL              int
	FA               int
	CTK              string
	CTV              string
	NKVO             int
	Threads          *int
	ThreadsIsDefault bool
	B                int
	UB               int
	NPrompt          int
	NGen             int
	Repetitions      int
}

// JSONLRecord is the exact schema written to sweep.jsonl.
type JSONLRecord struct {
	RunID         string          `json:"run_id"`
	Timestamp     string          `json:"timestamp"`
	ModelPath     string          `json:"model_path"`
	ModelStem     string          `json:"model_stem"`
	Phase         int             `json:"phase"`
	PhaseLabel    string          `json:"phase_label"`
	Binary        string          `json:"binary"`
	Status        string          `json:"status"`
	Viable        *bool           `json:"viable"`
	Params        jsonlParamsJSON `json:"params"`
	Results       []jsonlResult   `json:"results"`
	WallTimeSec   *float64        `json:"wall_time_sec"`
	RawOutputFile *string         `json:"raw_output_file"`
	ErrorSnippet  *string         `json:"error_snippet"`
}

type jsonlParamsJSON struct {
	NGL              int    `json:"ngl"`
	FA               int    `json:"fa"`
	CTK              string `json:"ctk"`
	CTV              string `json:"ctv"`
	NKVO             int    `json:"nkvo"`
	Threads          *int   `json:"threads"`
	ThreadsIsDefault bool   `json:"threads_is_default"`
	B                int    `json:"b"`
	UB               int    `json:"ub"`
	NPrompt          int    `json:"n_prompt"`
	NGen             int    `json:"n_gen"`
	Repetitions      int    `json:"repetitions"`
}

type jsonlResult struct {
	Test    string  `json:"test"`
	NPrompt int     `json:"n_prompt,omitempty"`
	NGen    int     `json:"n_gen,omitempty"`
	AvgTS   float64 `json:"avg_ts"`
	StddevTS float64 `json:"stddev_ts"`
}

// AppendRecord constructs and appends one JSON line to outputDir/sweep.jsonl.
func AppendRecord(outputDir, modelPath, modelStem string,
	p JSONLParams, result *bench.RunResult,
	phase int, phaseLabel, binaryLabel string) error {

	rec := JSONLRecord{
		RunID:      result.RunID,
		Timestamp:  time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		ModelPath:  modelPath,
		ModelStem:  modelStem,
		Phase:      phase,
		PhaseLabel: phaseLabel,
		Binary:     binaryLabel,
		Status:     string(result.Status),
		Params: jsonlParamsJSON{
			NGL:              p.NGL,
			FA:               p.FA,
			CTK:              p.CTK,
			CTV:              p.CTV,
			NKVO:             p.NKVO,
			Threads:          p.Threads,
			ThreadsIsDefault: p.ThreadsIsDefault,
			B:                p.B,
			UB:               p.UB,
			NPrompt:          p.NPrompt,
			NGen:             p.NGen,
			Repetitions:      p.Repetitions,
		},
	}

	// Viable
	if result.Status == bench.StatusOK && p.NGen > 0 {
		v := bench.TGSpeed(result.Results) >= 2.0 // will be overridden by caller if needed
		rec.Viable = &v
	}

	// Results array
	for _, r := range result.Results {
		jr := jsonlResult{
			Test:     r.Test,
			AvgTS:    r.AvgTS,
			StddevTS: r.StddevTS,
		}
		if r.Test == "pp" {
			jr.NPrompt = r.NPrompt
		} else {
			jr.NGen = r.NGen
		}
		rec.Results = append(rec.Results, jr)
	}
	if rec.Results == nil {
		rec.Results = []jsonlResult{} // always emit array, not null
	}

	// Wall time
	if result.WallTimeSec > 0 {
		w := result.WallTimeSec
		rec.WallTimeSec = &w
	}

	// Raw output file
	if result.RawOutputFile != "" {
		r := result.RawOutputFile
		rec.RawOutputFile = &r
	}

	// Error snippet
	if result.ErrorSnippet != "" {
		e := result.ErrorSnippet
		rec.ErrorSnippet = &e
	}

	// Append to sweep.jsonl
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	line = append(line, '\n')

	f, err := os.OpenFile(filepath.Join(outputDir, "sweep.jsonl"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(line)
	return err
}

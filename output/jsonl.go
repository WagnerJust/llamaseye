package output

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/WagnerJust/llamaseye/bench"
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

// ExistingCombo holds performance data for a previously-tested combo.
type ExistingCombo struct {
	TG float64
	PP float64
}

// LoadExistingCombos reads sweep.jsonl and returns combo keys for all status="ok"
// records, indexed by phase ID. Each phase uses a different combo key format:
//
//	P0,P1: "ngl"
//	P2:    "fa_ctk_ctv"
//	P3:    "threads" or "sys"
//	P4:    "nkvo"
//	P5:    "b_ub"
//	P6:    "ctx"
//	P7:    "ngl_fa_ctk_ctv_nkvo_threads_b_ub_ctx"
func LoadExistingCombos(outputDir string) (map[int]map[string]ExistingCombo, error) {
	path := filepath.Join(outputDir, "sweep.jsonl")
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	result := make(map[int]map[string]ExistingCombo)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec JSONLRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		if rec.Status != "ok" {
			continue
		}

		key := comboKey(rec.Phase, rec.Params)
		if key == "" {
			continue
		}

		if result[rec.Phase] == nil {
			result[rec.Phase] = make(map[string]ExistingCombo)
		}

		// Extract TG/PP from results
		var tg, pp float64
		for _, r := range rec.Results {
			switch r.Test {
			case "tg":
				tg = r.AvgTS
			case "pp":
				pp = r.AvgTS
			}
		}

		// Keep the best TG for each combo key
		if existing, ok := result[rec.Phase][key]; !ok || tg > existing.TG {
			result[rec.Phase][key] = ExistingCombo{TG: tg, PP: pp}
		}
	}
	return result, scanner.Err()
}

// ComboKey builds the canonical dedup key for --focused mode.
// Thread value is encoded as "sys" for system default (nil), or the integer value.
func ComboKey(phase int, ngl, fa int, ctk, ctv string, nkvo int, threads *int, b, ub, ctx int) string {
	switch phase {
	case 0, 1:
		return fmt.Sprintf("%d", ngl)
	case 2:
		return fmt.Sprintf("%d_%s_%s", fa, ctk, ctv)
	case 3:
		if threads == nil {
			return "sys"
		}
		return fmt.Sprintf("%d", *threads)
	case 4:
		return fmt.Sprintf("%d", nkvo)
	case 5:
		return fmt.Sprintf("%d_%d", b, ub)
	case 6:
		return fmt.Sprintf("%d", ctx)
	case 7:
		thr := "sys"
		if threads != nil {
			thr = fmt.Sprintf("%d", *threads)
		}
		return fmt.Sprintf("%d_%d_%s_%s_%d_%s_%d_%d_%d", ngl, fa, ctk, ctv, nkvo, thr, b, ub, ctx)
	default:
		return ""
	}
}

// comboKey unpacks JSONL params and delegates to ComboKey.
func comboKey(phase int, p jsonlParamsJSON) string {
	var threads *int
	if !p.ThreadsIsDefault && p.Threads != nil {
		threads = p.Threads
	}
	return ComboKey(phase, p.NGL, p.FA, p.CTK, p.CTV, p.NKVO, threads, p.B, p.UB, p.NPrompt)
}

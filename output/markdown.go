package output

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// record is an internal struct for reading sweep.jsonl rows.
type record struct {
	RunID      string `json:"run_id"`
	Phase      int    `json:"phase"`
	PhaseLabel string `json:"phase_label"`
	Status     string `json:"status"`
	Viable     *bool  `json:"viable"`
	Params     struct {
		NGL              int     `json:"ngl"`
		FA               int     `json:"fa"`
		CTK              string  `json:"ctk"`
		CTV              string  `json:"ctv"`
		NKVO             int     `json:"nkvo"`
		Threads          *int    `json:"threads"`
		ThreadsIsDefault bool    `json:"threads_is_default"`
		B                int     `json:"b"`
		UB               int     `json:"ub"`
		NPrompt          int     `json:"n_prompt"`
		NGen             int     `json:"n_gen"`
		Repetitions      int     `json:"repetitions"`
	} `json:"params"`
	Results []struct {
		Test     string  `json:"test"`
		NPrompt  int     `json:"n_prompt"`
		NGen     int     `json:"n_gen"`
		AvgTS    float64 `json:"avg_ts"`
		StddevTS float64 `json:"stddev_ts"`
	} `json:"results"`
	WallTimeSec   *float64 `json:"wall_time_sec"`
	RawOutputFile *string  `json:"raw_output_file"`
	ErrorSnippet  *string  `json:"error_snippet"`
}

func (r *record) tgTS() float64 {
	for _, res := range r.Results {
		if res.Test == "tg" {
			return res.AvgTS
		}
	}
	return 0
}

func (r *record) ppTS() float64 {
	for _, res := range r.Results {
		if res.Test == "pp" {
			return res.AvgTS
		}
	}
	return 0
}

func (r *record) threadsDisplay() string {
	if r.Params.ThreadsIsDefault || r.Params.Threads == nil {
		return "sys"
	}
	return fmt.Sprintf("%d", *r.Params.Threads)
}

// loadJSONL reads all records from sweep.jsonl.
func loadJSONL(path string) ([]*record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var records []*record
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20) // 1MB per line
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec record
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue // skip malformed lines
		}
		records = append(records, &rec)
	}
	return records, scanner.Err()
}

// GenerateMarkdown writes sweep.md from sweep.jsonl in outputDir.
// goalSpec is the raw --goal string (may be empty).
func GenerateMarkdown(outputDir, modelStem, goalSpec string, timeoutSec int) error {
	jsonlPath := filepath.Join(outputDir, "sweep.jsonl")
	if _, err := os.Stat(jsonlPath); os.IsNotExist(err) {
		return nil
	}

	records, err := loadJSONL(jsonlPath)
	if err != nil {
		return err
	}

	mdPath := filepath.Join(outputDir, "sweep.md")
	f, err := os.Create(mdPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	fmt.Fprintf(w, "# Sweep Results: %s\n\n", modelStem)
	fmt.Fprintf(w, "Generated: %s\n\n", time.Now().UTC().Format("2006-01-02T15:04:05Z"))

	// Top-N by TG
	okRecords := filterByStatus(records, "ok")
	topN := 10
	if len(okRecords) > 0 {
		sort.Slice(okRecords, func(i, j int) bool {
			return okRecords[i].tgTS() > okRecords[j].tgTS()
		})
		fmt.Fprintf(w, "## Best Configurations (Top %d by TG t/s)\n\n", topN)
		fmt.Fprintln(w, "Best results across all phases, ranked by token generation speed.")
		fmt.Fprintln(w)
		fmt.Fprintln(w, "| # | ph | ngl | fa | ctk | threads | nkvo | b | ub | n_prompt | PP t/s | TG t/s |")
		fmt.Fprintln(w, "|--:|---:|-----|-----|-----|---------|------|---|----|---------:|-------:|-------:|")
		for i, rec := range okRecords {
			if i >= topN {
				break
			}
			ppStr := fmtTS(rec.ppTS())
			tgStr := fmtTS(rec.tgTS())
			fmt.Fprintf(w, "| %d | %d | %d | %d | %s | %s | %d | %d | %d | %d | %s | %s |\n",
				i+1, rec.Phase, rec.Params.NGL, rec.Params.FA, rec.Params.CTK,
				rec.threadsDisplay(), rec.Params.NKVO, rec.Params.B, rec.Params.UB,
				rec.Params.NPrompt, ppStr, tgStr)
		}
		fmt.Fprintln(w)
	}

	// Goal results section (when --goal was active and phase 7 ran)
	if goalSpec != "" {
		goalCtx, goalTG, goalPP := parseGoalSpec(goalSpec)
		phase7OK := filterByPhaseStatus(records, 7, "ok")
		if len(phase7OK) > 0 {
			fmt.Fprintf(w, "## Goal Results — %s\n\n", goalSpecDesc(goalCtx, goalTG, goalPP))
			fmt.Fprintln(w, "One winner per distinct (ngl, ctk, nkvo, ctx) combination, ranked by TG t/s.")
			fmt.Fprintln(w, "Each row is a meaningful trade-off decision; tuning variants (threads/b/ub) are collapsed to the best result.")
			fmt.Fprintln(w)
			fmt.Fprintln(w, "| ngl | ctk | nkvo | ctx | TG t/s | PP t/s | fa | threads | b | ub |")
			fmt.Fprintln(w, "|-----|-----|------|----:|-------:|-------:|-----|---------|---|----|")

			// Deduplicate: one winner per (ngl, ctk, nkvo, ctx) tuple — best TG wins
			type tupleKey struct{ NGL int; CTK string; NKVO, Ctx int }
			best := make(map[tupleKey]*record)
			for _, rec := range phase7OK {
				if !meetsGoal(rec, goalCtx, goalTG, goalPP) {
					continue
				}
				k := tupleKey{rec.Params.NGL, rec.Params.CTK, rec.Params.NKVO, rec.Params.NPrompt}
				if existing, found := best[k]; !found || rec.tgTS() > existing.tgTS() {
					best[k] = rec
				}
			}
			winners := make([]*record, 0, len(best))
			for _, rec := range best {
				winners = append(winners, rec)
			}
			sort.Slice(winners, func(i, j int) bool {
				return winners[i].tgTS() > winners[j].tgTS()
			})
			for _, rec := range winners {
				fmt.Fprintf(w, "| %d | %s | %d | %d | %s | %s | %d | %s | %d | %d |\n",
					rec.Params.NGL, rec.Params.CTK, rec.Params.NKVO, rec.Params.NPrompt,
					fmtTS(rec.tgTS()), fmtTS(rec.ppTS()),
					rec.Params.FA, rec.threadsDisplay(), rec.Params.B, rec.Params.UB)
			}
			fmt.Fprintln(w)
		}
	}

	// Per-phase tables
	for phaseID := 0; phaseID <= 7; phaseID++ {
		phaseRecs := filterByPhase(records, phaseID)
		if len(phaseRecs) == 0 {
			continue
		}
		phaseLabel := phaseRecs[0].PhaseLabel
		fmt.Fprintf(w, "## Phase %d — %s\n\n", phaseID, phaseLabel)
		fmt.Fprintln(w, "| ngl | fa | ctk | threads | nkvo | b | ub | n_prompt | PP t/s | TG t/s | viable | status |")
		fmt.Fprintln(w, "|-----|-----|-----|---------|------|---|----|---------:|-------:|-------:|--------|--------|")

		// Sort: ok rows by TG desc, then other statuses
		sort.Slice(phaseRecs, func(i, j int) bool {
			if phaseRecs[i].Status == "ok" && phaseRecs[j].Status != "ok" {
				return true
			}
			if phaseRecs[i].Status != "ok" && phaseRecs[j].Status == "ok" {
				return false
			}
			return phaseRecs[i].tgTS() > phaseRecs[j].tgTS()
		})

		for _, rec := range phaseRecs {
			viableStr := "-"
			if rec.Viable != nil {
				if *rec.Viable {
					viableStr = "true"
				} else {
					viableStr = "false"
				}
			}
			fmt.Fprintf(w, "| %d | %d | %s | %s | %d | %d | %d | %d | %s | %s | %s | %s |\n",
				rec.Params.NGL, rec.Params.FA, rec.Params.CTK,
				rec.threadsDisplay(), rec.Params.NKVO, rec.Params.B, rec.Params.UB,
				rec.Params.NPrompt, fmtTS(rec.ppTS()), fmtTS(rec.tgTS()), viableStr, rec.Status)
		}

		// Winner callout
		okInPhase := filterByStatus(phaseRecs, "ok")
		if len(okInPhase) > 0 {
			sort.Slice(okInPhase, func(i, j int) bool {
				return okInPhase[i].tgTS() > okInPhase[j].tgTS()
			})
			winner := okInPhase[0]
			fmt.Fprintf(w, "> **Winner:** ngl=%d fa=%d ctk=%s threads=%s nkvo=%d b=%d ub=%d n_prompt=%d → TG %s t/s  PP %s t/s\n",
				winner.Params.NGL, winner.Params.FA, winner.Params.CTK,
				winner.threadsDisplay(), winner.Params.NKVO, winner.Params.B, winner.Params.UB,
				winner.Params.NPrompt, fmtTS(winner.tgTS()), fmtTS(winner.ppTS()))
		}
		fmt.Fprintln(w)
	}

	// Phase 6 timeout ctx sizes
	p6Timeout := filterByPhaseStatus(records, 6, "timeout")
	if len(p6Timeout) > 0 {
		fmt.Fprintln(w, "## Phase 6 — Context sizes that timed out (achievable but slow)")
		fmt.Fprintln(w)
		fmt.Fprintf(w, "These context sizes were not OOM — they timed out after `%ds`.\n", timeoutSec)
		fmt.Fprintln(w, "They are feasible for overnight batch jobs but impractical for interactive use.")
		fmt.Fprintln(w)
		fmt.Fprintln(w, "| ctx | wall time (s) | ngl | ctk | nkvo |")
		fmt.Fprintln(w, "|----:|--------------:|-----|-----|------|")
		sort.Slice(p6Timeout, func(i, j int) bool {
			return p6Timeout[i].Params.NPrompt < p6Timeout[j].Params.NPrompt
		})
		for _, rec := range p6Timeout {
			wallStr := "-"
			if rec.WallTimeSec != nil {
				wallStr = fmt.Sprintf("%.0f", *rec.WallTimeSec)
			}
			fmt.Fprintf(w, "| %d | %s | %d | %s | %d |\n",
				rec.Params.NPrompt, wallStr, rec.Params.NGL, rec.Params.CTK, rec.Params.NKVO)
		}
		fmt.Fprintln(w)
	}

	// Context frontier (from Phase 7)
	p7OK := filterByPhaseStatus(records, 7, "ok")
	if len(p7OK) > 0 {
		fmt.Fprintln(w, "## Context Frontier")
		fmt.Fprintln(w)
		fmt.Fprintln(w, "| ngl | ctk | nkvo | Max Context | PP t/s |")
		fmt.Fprintln(w, "|-----|-----|------|------------:|-------:|")

		type frontierKey struct{ NGL int; CTK string; NKVO int }
		frontier := make(map[frontierKey]*record)
		for _, rec := range p7OK {
			key := frontierKey{rec.Params.NGL, rec.Params.CTK, rec.Params.NKVO}
			if existing, found := frontier[key]; !found || rec.Params.NPrompt > existing.Params.NPrompt {
				frontier[key] = rec
			}
		}
		var frontierKeys []frontierKey
		for k := range frontier {
			frontierKeys = append(frontierKeys, k)
		}
		sort.Slice(frontierKeys, func(i, j int) bool {
			ri := frontier[frontierKeys[i]]
			rj := frontier[frontierKeys[j]]
			return ri.Params.NPrompt > rj.Params.NPrompt
		})
		for _, k := range frontierKeys {
			rec := frontier[k]
			fmt.Fprintf(w, "| %d | %s | %d | %d | %s |\n",
				k.NGL, k.CTK, k.NKVO, rec.Params.NPrompt, fmtTS(rec.ppTS()))
		}
		fmt.Fprintln(w)
	}

	return nil
}

// GenerateCrossModelSummary writes summary.md comparing results across model subdirs.
func GenerateCrossModelSummary(outputDir string, stems []string) error {
	if len(stems) < 2 {
		return nil
	}
	type modelRow struct {
		Stem     string
		TG       float64
		PP       float64
		NGL      int
		FA       int
		CTK      string
		NKVO     int
		B        int
		UB       int
		Threads  string
		NPrompt  int
	}

	var rows []modelRow
	for _, stem := range stems {
		jsonlPath := filepath.Join(outputDir, stem, "sweep.jsonl")
		recs, err := loadJSONL(jsonlPath)
		if err != nil {
			continue
		}
		okRecs := filterByStatus(recs, "ok")
		if len(okRecs) == 0 {
			continue
		}
		sort.Slice(okRecs, func(i, j int) bool {
			return okRecs[i].tgTS() > okRecs[j].tgTS()
		})
		best := okRecs[0]
		rows = append(rows, modelRow{
			Stem:    stem,
			TG:      best.tgTS(),
			PP:      best.ppTS(),
			NGL:     best.Params.NGL,
			FA:      best.Params.FA,
			CTK:     best.Params.CTK,
			NKVO:    best.Params.NKVO,
			B:       best.Params.B,
			UB:      best.Params.UB,
			Threads: best.threadsDisplay(),
			NPrompt: best.Params.NPrompt,
		})
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].TG > rows[j].TG })

	summaryPath := filepath.Join(outputDir, "summary.md")
	f, err := os.Create(summaryPath)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	fmt.Fprintln(w, "# Multi-Model Sweep Summary")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Generated: %s\n\n", time.Now().UTC().Format("2006-01-02T15:04:05Z"))
	fmt.Fprintf(w, "%d models benchmarked. Sorted by best TG t/s.\n\n", len(rows))
	fmt.Fprintln(w, "| Model | Best TG t/s | PP t/s | ngl | fa | ctk | nkvo | b | ub | threads | n_prompt |")
	fmt.Fprintln(w, "|-------|------------:|-------:|-----|-----|-----|------|---|----|---------:|---------:|")
	for _, row := range rows {
		fmt.Fprintf(w, "| %s | %s | %s | %d | %d | %s | %d | %d | %d | %s | %d |\n",
			row.Stem, fmtTS(row.TG), fmtTS(row.PP),
			row.NGL, row.FA, row.CTK, row.NKVO, row.B, row.UB, row.Threads, row.NPrompt)
	}
	fmt.Fprintln(w)
	return nil
}

// helpers

func filterByStatus(records []*record, status string) []*record {
	var result []*record
	for _, r := range records {
		if r.Status == status {
			result = append(result, r)
		}
	}
	return result
}

func filterByPhase(records []*record, phase int) []*record {
	var result []*record
	for _, r := range records {
		if r.Phase == phase {
			result = append(result, r)
		}
	}
	return result
}

func filterByPhaseStatus(records []*record, phase int, status string) []*record {
	var result []*record
	for _, r := range records {
		if r.Phase == phase && r.Status == status {
			result = append(result, r)
		}
	}
	return result
}

func fmtTS(v float64) string {
	if v == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", v)
}

func parseGoalSpec(spec string) (ctx int, tg, pp float64) {
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			continue
		}
		switch kv[0] {
		case "ctx":
			fmt.Sscanf(kv[1], "%d", &ctx)
		case "tg":
			fmt.Sscanf(kv[1], "%f", &tg)
		case "pp":
			fmt.Sscanf(kv[1], "%f", &pp)
		}
	}
	return
}

func goalSpecDesc(ctx int, tg, pp float64) string {
	var parts []string
	if ctx > 0 {
		parts = append(parts, fmt.Sprintf("ctx≥%d", ctx))
	}
	if tg > 0 {
		parts = append(parts, fmt.Sprintf("tg≥%.1f t/s", tg))
	}
	if pp > 0 {
		parts = append(parts, fmt.Sprintf("pp≥%.1f t/s", pp))
	}
	return strings.Join(parts, " ")
}

func meetsGoal(rec *record, goalCtx int, goalTG, goalPP float64) bool {
	if goalCtx > 0 && rec.Params.NPrompt < goalCtx {
		return false
	}
	if goalTG > 0 && rec.tgTS() < goalTG {
		return false
	}
	if goalPP > 0 && rec.ppTS() < goalPP {
		return false
	}
	return true
}

package phase

import (
	"context"

	"github.com/justinphilpott/llamaseye/bench"
	"github.com/justinphilpott/llamaseye/output"
	"github.com/justinphilpott/llamaseye/state"
)

// CTKQualityOrder defines KV type ordering from least to most compressed.
// Quality order: turbo2 < turbo3 < turbo4 < q4_0 < q8_0 < f16
var CTKQualityOrder = []string{"turbo2", "turbo3", "turbo4", "q4_0", "q8_0", "f16"}

// CTKQualityIndex returns the index of ctk in CTKQualityOrder, or -1 if not found.
func CTKQualityIndex(ctk string) int {
	for i, v := range CTKQualityOrder {
		if v == ctk {
			return i
		}
	}
	return -1
}

// ApplyAxisOpts slices fullList starting at startValue in the given direction.
// direction must be "up" or "down".
// If startValue is empty, returns the full ordered list.
// If startValue is not found, returns the full ordered list (with a warn).
func ApplyAxisOpts(fullList []string, startValue, direction string, warn func(string, ...any)) []string {
	ordered := make([]string, len(fullList))
	copy(ordered, fullList)
	if direction == "down" {
		for i, j := 0, len(ordered)-1; i < j; i, j = i+1, j-1 {
			ordered[i], ordered[j] = ordered[j], ordered[i]
		}
	}
	if startValue == "" {
		return ordered
	}
	for i, v := range ordered {
		if v == startValue {
			return ordered[i:]
		}
	}
	if warn != nil {
		warn("Start value %q not found in axis list -- using full list", startValue)
	}
	return ordered
}

// ApplyAxisOptsInt is like ApplyAxisOpts but for integer lists.
func ApplyAxisOptsInt(fullList []int, startValue *int, direction string, warn func(string, ...any)) []int {
	ordered := make([]int, len(fullList))
	copy(ordered, fullList)
	if direction == "down" {
		for i, j := 0, len(ordered)-1; i < j; i, j = i+1, j-1 {
			ordered[i], ordered[j] = ordered[j], ordered[i]
		}
	}
	if startValue == nil {
		return ordered
	}
	for i, v := range ordered {
		if v == *startValue {
			return ordered[i:]
		}
	}
	if warn != nil {
		warn("Start value %d not found in axis list -- using full list", *startValue)
	}
	return ordered
}

// ApplyPhase7Mins filters a list of values, removing entries below minValue.
// axis: "ngl" | "threads" | "ctx" | "b" | "ub" | "ctk"
// For numeric axes: removes values strictly < minValue.
// For "ctk": removes types with lower quality index than minValue.
// Returns the full list unchanged if minValue is nil/empty.
func ApplyPhase7MinsInt(axis string, values []int, minValue *int, warn func(string, ...any)) []int {
	if minValue == nil {
		return values
	}
	min := *minValue
	var result []int
	for _, v := range values {
		if v >= min {
			result = append(result, v)
		}
	}
	return result
}

// ApplyPhase7MinsCTK filters CTK types to those at or above minCTK in quality order.
func ApplyPhase7MinsCTK(values []string, minCTK string, warn func(string, ...any)) []string {
	if minCTK == "" {
		return values
	}
	minIdx := CTKQualityIndex(minCTK)
	if minIdx == -1 {
		if warn != nil {
			warn("Unknown --min-ctk value %q — no filtering applied", minCTK)
		}
		return values
	}
	var result []string
	for _, v := range values {
		idx := CTKQualityIndex(v)
		if idx >= minIdx {
			result = append(result, v)
		}
	}
	return result
}

// BestFACTVForCTK scans ws for the best (fa=1 preferred) ctv for the given ctk.
func BestFACTVForCTK(ws []state.FACTKCombo, targetCTK string) (fa int, ctv string, found bool) {
	for _, combo := range ws {
		if combo.CTK == targetCTK {
			if !found || combo.FA == 1 {
				fa = combo.FA
				ctv = combo.CTV
				found = true
			}
		}
	}
	return
}

// RecordAndTrack runs a bench and writes the JSONL record.
// Returns the status and the TG t/s (0 if not available).
func RecordAndTrack(env *PhaseEnv, label string, p bench.RunParams) (bench.Status, float64, float64) {
	if env.Thermal != nil {
		env.Thermal.WaitCool(context.Background())
	}

	res, err := env.Runner.RunBench(label, p)
	if err != nil {
		env.Logger.Warn("RunBench error for %s: %v", label, err)
		return bench.StatusError, 0, 0
	}

	// Build JSONLParams from RunParams
	jp := output.JSONLParams{
		NGL:         p.NGL,
		FA:          p.FA,
		CTK:         p.CTK,
		CTV:         p.CTV,
		NKVO:        p.NKVO,
		Threads:     p.Threads,
		B:           p.B,
		UB:          p.UB,
		NPrompt:     p.NPrompt,
		NGen:        p.NGen,
		Repetitions: p.Reps,
	}
	jp.ThreadsIsDefault = p.Threads == nil

	// Viability
	if res.Status == bench.StatusOK && p.NGen > 0 {
		tg := bench.TGSpeed(res.Results)
		v := tg >= env.Config.MinTGTS
		_ = v
	}

	binaryLabel := "standard"
	if p.CTK != "" {
		_, bl, _ := env.Runner.Selector.Select(p.CTK)
		binaryLabel = bl
	}

	_ = output.AppendRecord(env.OutputDir, env.ModelPath, env.ModelStem,
		jp, res, p.Phase, p.PhaseLabel, binaryLabel)

	return res.Status, bench.TGSpeed(res.Results), bench.PPSpeed(res.Results)
}

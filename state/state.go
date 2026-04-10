// Package state handles load/save of state.json, compatible with bash schema.
package state

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Best holds the best observed parameter values across phases.
type Best struct {
	NGL     int      `json:"ngl"`
	FA      int      `json:"fa"`
	CTK     string   `json:"ctk"`
	CTV     string   `json:"ctv"`
	Threads *int     `json:"threads"` // null = system default
	NKVO    int      `json:"nkvo"`
	B       int      `json:"b"`
	UB      int      `json:"ub"`
	CTX     int      `json:"ctx"`
}

// FACTKCombo represents one row in the fa_ctk_combos working set.
type FACTKCombo struct {
	FA  int    `json:"fa"`
	CTK string `json:"ctk"`
	CTV string `json:"ctv"`
}

// BUBCombo represents one batch/ubatch pair.
type BUBCombo struct {
	B  int `json:"b"`
	UB int `json:"ub"`
}

// WorkingSets holds the accumulated output of each sweep phase.
type WorkingSets struct {
	NGL          []int        `json:"ngl"`
	FACTKCombos  []FACTKCombo `json:"fa_ctk_combos"`
	CTKValues    []string     `json:"ctk_values"`    // independent CTK axis for Phase 7
	CTVValues    []string     `json:"ctv_values"`    // independent CTV axis for Phase 7
	ThreadValues []any        `json:"thread_values"` // int or "system_default"
	NKVOValues   []int        `json:"nkvo_values"`
	BUBCombos    []BUBCombo   `json:"b_ub_combos"`
	CTXValues    []int        `json:"ctx_values"`
}

// State is the full state.json schema — compatible with bash-written state files.
type State struct {
	ModelPath       string      `json:"model_path"`
	ModelStem       string      `json:"model_stem"`
	MaxNGL          int         `json:"max_ngl"`
	PhasesComplete  []int       `json:"phases_complete"`
	Best            Best        `json:"best"`
	WorkingSets     WorkingSets `json:"working_sets"`
}

// DefaultBest returns the initial best values matching the bash defaults.
func DefaultBest() Best {
	return Best{
		NGL:     99,
		FA:      0,
		CTK:     "f16",
		CTV:     "f16",
		Threads: nil,
		NKVO:    0,
		B:       2048,
		UB:      512,
		CTX:     512,
	}
}

// Load reads state.json from outputDir. Returns default state if file doesn't exist.
func Load(outputDir string) (*State, error) {
	path := filepath.Join(outputDir, "state.json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// Save writes the state to outputDir/state.json atomically.
func Save(outputDir string, s *State) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(outputDir, "state.json")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// PhaseComplete returns true if phase is in s.PhasesComplete.
func (s *State) PhaseComplete(phase int) bool {
	for _, p := range s.PhasesComplete {
		if p == phase {
			return true
		}
	}
	return false
}

// MarkPhaseComplete adds phase to PhasesComplete if not already present.
func (s *State) MarkPhaseComplete(phase int) {
	if !s.PhaseComplete(phase) {
		s.PhasesComplete = append(s.PhasesComplete, phase)
	}
}


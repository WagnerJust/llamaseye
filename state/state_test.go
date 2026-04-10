package state

import (
	"os"
	"path/filepath"
	"testing"
)

func intPtr(n int) *int { return &n }

func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	threads := 8
	orig := &State{
		ModelPath:      "/models/foo.gguf",
		ModelStem:      "foo",
		MaxNGL:         32,
		PhasesComplete: []int{0, 1, 2},
		Best: Best{
			NGL:     32,
			FA:      1,
			CTK:     "q8_0",
			CTV:     "q8_0",
			Threads: &threads,
			NKVO:    0,
			B:       2048,
			UB:      512,
			CTX:     8192,
		},
		WorkingSets: WorkingSets{
			NGL:        []int{28, 32},
			NKVOValues: []int{0, 1},
			CTXValues:  []int{512, 1024, 8192},
			FACTKCombos: []FACTKCombo{
				{FA: 0, CTK: "f16", CTV: "f16"},
				{FA: 1, CTK: "q8_0", CTV: "q8_0"},
			},
			BUBCombos: []BUBCombo{
				{B: 2048, UB: 512},
				{B: 1024, UB: 256},
			},
			ThreadValues: ThreadValues{intPtr(4), intPtr(8), nil},
		},
	}

	if err := Save(dir, orig); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if loaded.ModelStem != orig.ModelStem {
		t.Errorf("ModelStem = %q, want %q", loaded.ModelStem, orig.ModelStem)
	}
	if loaded.MaxNGL != orig.MaxNGL {
		t.Errorf("MaxNGL = %d, want %d", loaded.MaxNGL, orig.MaxNGL)
	}
	if loaded.Best.CTK != orig.Best.CTK {
		t.Errorf("Best.CTK = %q, want %q", loaded.Best.CTK, orig.Best.CTK)
	}
	if loaded.Best.Threads == nil || *loaded.Best.Threads != 8 {
		t.Errorf("Best.Threads = %v, want &8", loaded.Best.Threads)
	}
	if len(loaded.PhasesComplete) != 3 {
		t.Errorf("PhasesComplete len = %d, want 3", len(loaded.PhasesComplete))
	}
	if len(loaded.WorkingSets.NGL) != 2 {
		t.Errorf("WS.NGL len = %d, want 2", len(loaded.WorkingSets.NGL))
	}
}

func TestLoad_NonExistent(t *testing.T) {
	dir := t.TempDir()
	s, err := Load(dir)
	if err != nil {
		t.Fatalf("Load on missing file: %v", err)
	}
	if s != nil {
		t.Error("expected nil for missing state.json, got non-nil")
	}
}

func TestMarkPhaseComplete(t *testing.T) {
	s := &State{}
	s.MarkPhaseComplete(0)
	s.MarkPhaseComplete(1)
	s.MarkPhaseComplete(1) // duplicate — should not add
	if len(s.PhasesComplete) != 2 {
		t.Errorf("PhasesComplete len = %d, want 2", len(s.PhasesComplete))
	}
}

func TestPhaseComplete(t *testing.T) {
	s := &State{PhasesComplete: []int{0, 2, 5}}
	if !s.PhaseComplete(0) {
		t.Error("expected phase 0 complete")
	}
	if s.PhaseComplete(1) {
		t.Error("expected phase 1 not complete")
	}
}

func TestLoad_BashCompatible(t *testing.T) {
	// A state.json written by the bash script — load it and verify fields.
	bashJSON := `{
  "model_path": "/models/test.gguf",
  "model_stem": "test",
  "max_ngl": 40,
  "phases_complete": [0,1,2,3],
  "best": {
    "ngl": 40,
    "fa": 1,
    "ctk": "q8_0",
    "ctv": "q8_0",
    "threads": null,
    "nkvo": 0,
    "b": 2048,
    "ub": 512,
    "ctx": 8192
  },
  "working_sets": {
    "ngl": [36, 40],
    "fa_ctk_combos": [{"fa":0,"ctk":"f16","ctv":"f16"},{"fa":1,"ctk":"q8_0","ctv":"q8_0"}],
    "thread_values": [4, 8, "system_default"],
    "nkvo_values": [0, 1],
    "b_ub_combos": [{"b":2048,"ub":512}],
    "ctx_values": [512, 4096, 8192]
  }
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	if err := os.WriteFile(path, []byte(bashJSON), 0644); err != nil {
		t.Fatal(err)
	}

	s, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if s.MaxNGL != 40 {
		t.Errorf("MaxNGL = %d, want 40", s.MaxNGL)
	}
	if s.Best.CTK != "q8_0" {
		t.Errorf("Best.CTK = %q, want q8_0", s.Best.CTK)
	}
	if s.Best.Threads != nil {
		t.Errorf("Best.Threads = %v, want nil", s.Best.Threads)
	}
	if len(s.WorkingSets.FACTKCombos) != 2 {
		t.Errorf("fa_ctk_combos len = %d, want 2", len(s.WorkingSets.FACTKCombos))
	}
	if len(s.WorkingSets.ThreadValues) != 3 {
		t.Errorf("thread_values len = %d, want 3", len(s.WorkingSets.ThreadValues))
	}
}

func TestRoundTrip_CTKCTVValues(t *testing.T) {
	dir := t.TempDir()
	orig := &State{
		ModelPath:      "/models/foo.gguf",
		ModelStem:      "foo",
		MaxNGL:         32,
		PhasesComplete: []int{0, 1, 2},
		Best:           DefaultBest(),
		WorkingSets: WorkingSets{
			NGL:       []int{28, 32},
			CTKValues: []string{"f16", "q8_0"},
			CTVValues: []string{"f16", "q8_0", "turbo3"},
			FACTKCombos: []FACTKCombo{
				{FA: 1, CTK: "f16", CTV: "f16"},
				{FA: 1, CTK: "q8_0", CTV: "q8_0"},
				{FA: 1, CTK: "f16", CTV: "turbo3"},
			},
		},
	}

	if err := Save(dir, orig); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	wantCTK := []string{"f16", "q8_0"}
	if len(loaded.WorkingSets.CTKValues) != len(wantCTK) {
		t.Errorf("CTKValues = %v, want %v", loaded.WorkingSets.CTKValues, wantCTK)
	} else {
		for i, v := range wantCTK {
			if loaded.WorkingSets.CTKValues[i] != v {
				t.Errorf("CTKValues[%d] = %q, want %q", i, loaded.WorkingSets.CTKValues[i], v)
			}
		}
	}
	wantCTV := []string{"f16", "q8_0", "turbo3"}
	if len(loaded.WorkingSets.CTVValues) != len(wantCTV) {
		t.Errorf("CTVValues = %v, want %v", loaded.WorkingSets.CTVValues, wantCTV)
	} else {
		for i, v := range wantCTV {
			if loaded.WorkingSets.CTVValues[i] != v {
				t.Errorf("CTVValues[%d] = %q, want %q", i, loaded.WorkingSets.CTVValues[i], v)
			}
		}
	}
}

func TestDefaultBest(t *testing.T) {
	b := DefaultBest()
	if b.NGL != 99 {
		t.Errorf("DefaultBest NGL = %d, want 99", b.NGL)
	}
	if b.CTK != "f16" {
		t.Errorf("DefaultBest CTK = %q, want f16", b.CTK)
	}
	if b.Threads != nil {
		t.Errorf("DefaultBest Threads = %v, want nil", b.Threads)
	}
}

func TestThreadValues_JSON(t *testing.T) {
	orig := ThreadValues{intPtr(4), nil, intPtr(8)}
	data, err := orig.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}
	// Should produce [4,"system_default",8]
	want := `[4,"system_default",8]`
	if string(data) != want {
		t.Errorf("MarshalJSON = %s, want %s", data, want)
	}

	var loaded ThreadValues
	if err := loaded.UnmarshalJSON(data); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if len(loaded) != 3 {
		t.Fatalf("len = %d, want 3", len(loaded))
	}
	if loaded[0] == nil || *loaded[0] != 4 {
		t.Errorf("[0] = %v, want 4", loaded[0])
	}
	if loaded[1] != nil {
		t.Errorf("[1] = %v, want nil (system_default)", loaded[1])
	}
	if loaded[2] == nil || *loaded[2] != 8 {
		t.Errorf("[2] = %v, want 8", loaded[2])
	}
}

func TestLoad_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "state.json"), []byte("not json{"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(dir)
	if err == nil {
		t.Error("expected error for invalid JSON in state.json")
	}
}

func TestSave_CreatesFile(t *testing.T) {
	dir := t.TempDir()
	s := &State{
		ModelPath:      "/m.gguf",
		ModelStem:      "m",
		MaxNGL:         32,
		PhasesComplete: []int{0, 1},
		Best:           DefaultBest(),
	}
	if err := Save(dir, s); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := Load(dir)
	if err != nil {
		t.Fatalf("Load after Save: %v", err)
	}
	if loaded.MaxNGL != 32 {
		t.Errorf("MaxNGL = %d, want 32", loaded.MaxNGL)
	}
	if !loaded.PhaseComplete(1) {
		t.Error("expected phase 1 complete after save/load")
	}
}

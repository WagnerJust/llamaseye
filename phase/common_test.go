package phase

import (
	"testing"

	"github.com/WagnerJust/llamaseye/state"
)

func TestApplyAxisOpts_Up(t *testing.T) {
	list := []string{"0", "4", "8", "12", "16", "20"}
	result := ApplyAxisOpts(list, "8", "up", nil)
	want := []string{"8", "12", "16", "20"}
	if len(result) != len(want) {
		t.Fatalf("len=%d want %d", len(result), len(want))
	}
	for i, v := range result {
		if v != want[i] {
			t.Errorf("[%d] = %q, want %q", i, v, want[i])
		}
	}
}

func TestApplyAxisOpts_Down(t *testing.T) {
	list := []string{"0", "4", "8", "12", "16", "20"}
	result := ApplyAxisOpts(list, "8", "down", nil)
	want := []string{"8", "4", "0"}
	if len(result) != len(want) {
		t.Fatalf("len=%d want %d", len(result), len(want))
	}
	for i, v := range result {
		if v != want[i] {
			t.Errorf("[%d] = %q, want %q", i, v, want[i])
		}
	}
}

func TestApplyAxisOpts_EmptyStart(t *testing.T) {
	list := []string{"a", "b", "c"}
	result := ApplyAxisOpts(list, "", "up", nil)
	if len(result) != 3 {
		t.Errorf("expected full list, got len=%d", len(result))
	}
}

func TestApplyAxisOpts_NotFound(t *testing.T) {
	var warned bool
	list := []string{"0", "4", "8"}
	result := ApplyAxisOpts(list, "5", "up", func(f string, a ...any) { warned = true })
	if !warned {
		t.Error("expected warning for missing start value")
	}
	if len(result) != 3 {
		t.Errorf("expected full list on not-found, got len=%d", len(result))
	}
}

func TestApplyAxisOptsInt_Up(t *testing.T) {
	list := []int{0, 4, 8, 12, 16}
	start := 8
	result := ApplyAxisOptsInt(list, &start, "up", nil)
	want := []int{8, 12, 16}
	if len(result) != len(want) {
		t.Fatalf("len=%d want %d", len(result), len(want))
	}
	for i, v := range result {
		if v != want[i] {
			t.Errorf("[%d] = %d, want %d", i, v, want[i])
		}
	}
}

func TestApplyAxisOptsInt_Nil(t *testing.T) {
	list := []int{1, 2, 3}
	result := ApplyAxisOptsInt(list, nil, "up", nil)
	if len(result) != 3 {
		t.Errorf("nil start: expected 3 items, got %d", len(result))
	}
}

func TestApplyPhase7MinsInt(t *testing.T) {
	values := []int{0, 4, 8, 12, 16, 20}
	min := 8
	result := ApplyPhase7MinsInt(values, &min, nil)
	for _, v := range result {
		if v < 8 {
			t.Errorf("value %d below min=8", v)
		}
	}
	if len(result) != 4 {
		t.Errorf("expected 4 values >= 8, got %d", len(result))
	}
}

func TestApplyPhase7MinsInt_Nil(t *testing.T) {
	values := []int{1, 2, 3}
	result := ApplyPhase7MinsInt(values, nil, nil)
	if len(result) != 3 {
		t.Errorf("nil min: expected 3, got %d", len(result))
	}
}

func TestApplyPhase7MinsCTK(t *testing.T) {
	// min-ctk = q8_0 should keep q8_0 and f16 but drop q4_0, turbo*
	values := []string{"f16", "q8_0", "q4_0", "turbo4", "turbo3", "turbo2"}
	result := ApplyPhase7MinsCTK(values, "q8_0", nil)
	for _, v := range result {
		idx := CTKQualityIndex(v)
		minIdx := CTKQualityIndex("q8_0")
		if idx < minIdx {
			t.Errorf("%q below min-ctk=q8_0", v)
		}
	}
	if len(result) != 2 { // f16, q8_0
		t.Errorf("expected 2, got %d: %v", len(result), result)
	}
}

func TestApplyPhase7MinsCTK_Empty(t *testing.T) {
	values := []string{"f16", "q8_0"}
	result := ApplyPhase7MinsCTK(values, "", nil)
	if len(result) != 2 {
		t.Errorf("empty min: expected 2, got %d", len(result))
	}
}

func TestApplyPhase7MinsCTK_Unknown(t *testing.T) {
	var warned bool
	values := []string{"f16", "q8_0"}
	result := ApplyPhase7MinsCTK(values, "unknown_type", func(f string, a ...any) { warned = true })
	if !warned {
		t.Error("expected warning for unknown ctk type")
	}
	if len(result) != 2 {
		t.Errorf("unknown: expected all 2, got %d", len(result))
	}
}

func TestBestFACTVForCTK(t *testing.T) {
	ws := []state.FACTKCombo{
		{FA: 0, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "q8_0", CTV: "q8_0"},
	}
	fa, ctv, found := BestFACTVForCTK(ws, "f16")
	if !found {
		t.Fatal("expected to find f16")
	}
	if fa != 1 {
		t.Errorf("fa = %d, want 1 (prefer fa=1)", fa)
	}
	if ctv != "f16" {
		t.Errorf("ctv = %q, want f16", ctv)
	}

	_, _, found2 := BestFACTVForCTK(ws, "turbo3")
	if found2 {
		t.Error("should not find turbo3")
	}
}

func TestFindFACTKByKV(t *testing.T) {
	ws := []state.FACTKCombo{
		{FA: 0, CTK: "f16", CTV: "turbo3"},
		{FA: 1, CTK: "f16", CTV: "turbo3"},
		{FA: 1, CTK: "f16", CTV: "f16"},
	}
	fa, found := FindFACTKByKV(ws, "f16", "turbo3")
	if !found {
		t.Fatal("expected to find (f16, turbo3)")
	}
	if fa != 1 {
		t.Errorf("fa = %d, want 1 (prefer fa=1)", fa)
	}

	_, found2 := FindFACTKByKV(ws, "q8_0", "turbo3")
	if found2 {
		t.Error("should not find (q8_0, turbo3)")
	}
}

func TestKVPrecisionValid(t *testing.T) {
	// K more precise than V — valid
	if !KVPrecisionValid("f16", "turbo2") {
		t.Error("f16 K + turbo2 V should be valid (K more precise)")
	}
	if !KVPrecisionValid("q8_0", "q8_0") {
		t.Error("equal precision should be valid")
	}
	// V more precise than K — invalid (wasteful)
	if KVPrecisionValid("turbo2", "f16") {
		t.Error("turbo2 K + f16 V should be invalid (V more precise than K)")
	}
	if KVPrecisionValid("turbo3", "q4_0") {
		t.Error("turbo3 K + q4_0 V should be invalid (V more precise than K)")
	}
}

func TestBestFAForCTK(t *testing.T) {
	ws := []state.FACTKCombo{
		{FA: 0, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "q8_0", CTV: "q8_0"},
	}
	if BestFAForCTK(ws, "f16") != 1 {
		t.Error("BestFAForCTK f16: want 1 (prefer fa=1)")
	}
	if BestFAForCTK(ws, "q8_0") != 1 {
		t.Error("BestFAForCTK q8_0: want 1")
	}
	if BestFAForCTK(ws, "q4_0") != 0 {
		t.Error("BestFAForCTK q4_0 (not in ws): want 0 (default)")
	}
}

func TestUniqueCTKCTVValues(t *testing.T) {
	ws := []state.FACTKCombo{
		{FA: 0, CTK: "f16", CTV: "f16"},
		{FA: 1, CTK: "f16", CTV: "turbo3"},
		{FA: 1, CTK: "q8_0", CTV: "q8_0"},
	}
	ctks := UniqueCTKValues(ws)
	if len(ctks) != 2 {
		t.Errorf("UniqueCTKValues: got %v, want [f16 q8_0]", ctks)
	}
	ctvs := UniqueCTVValues(ws)
	if len(ctvs) != 3 {
		t.Errorf("UniqueCTVValues: got %v, want [f16 turbo3 q8_0]", ctvs)
	}
}

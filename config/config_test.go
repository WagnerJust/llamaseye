package config

import (
	"os"
	"testing"
)

func TestDefaults(t *testing.T) {
	c := Defaults()
	if c.NGLStep != 4 {
		t.Errorf("NGLStep default = %d, want 4", c.NGLStep)
	}
	if c.Repetitions != 3 {
		t.Errorf("Repetitions default = %d, want 3", c.Repetitions)
	}
	if c.TimeoutSec != 600 {
		t.Errorf("TimeoutSec default = %d, want 600", c.TimeoutSec)
	}
	if c.GoalTargetCount != 3 {
		t.Errorf("GoalTargetCount default = %d, want 3", c.GoalTargetCount)
	}
	if c.CtxStepMin != 8192 {
		t.Errorf("CtxStepMin default = %d, want 8192", c.CtxStepMin)
	}
}

func TestEnvOverride(t *testing.T) {
	t.Setenv("SWEEP_NGL_STEP", "8")
	t.Setenv("SWEEP_REPETITIONS", "5")
	c := Defaults()
	if c.NGLStep != 8 {
		t.Errorf("NGLStep from env = %d, want 8", c.NGLStep)
	}
	if c.Repetitions != 5 {
		t.Errorf("Repetitions from env = %d, want 5", c.Repetitions)
	}
}

func TestValidate_ResumeOverwrite(t *testing.T) {
	c := Defaults()
	c.Resume = true
	c.Overwrite = true
	if err := c.Validate(); err == nil {
		t.Error("expected error for resume+overwrite, got nil")
	}
}

func TestValidate_OnlySkipPhases(t *testing.T) {
	c := Defaults()
	c.OnlyPhases = []int{0, 1}
	c.SkipPhases = []int{2}
	if err := c.Validate(); err == nil {
		t.Error("expected error for only-phases+skip-phases, got nil")
	}
}

func TestValidate_OptimizedSweepConflict(t *testing.T) {
	c := Defaults()
	c.OptimizedSweep = true
	ngl := 10
	c.StartNGL = &ngl
	if err := c.Validate(); err == nil {
		t.Error("expected error for optimized-sweep + start-ngl, got nil")
	}
}

func TestValidate_OptimizedSweepNoConflict(t *testing.T) {
	c := Defaults()
	c.OptimizedSweep = true
	if err := c.Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_BadDirection(t *testing.T) {
	c := Defaults()
	c.DirNGL = "sideways"
	if err := c.Validate(); err == nil {
		t.Error("expected error for bad direction, got nil")
	}
}

func TestParsePhaseList(t *testing.T) {
	cases := []struct {
		in  string
		out []int
	}{
		{"", nil},
		{"0,2,5", []int{0, 2, 5}},
		{"7", []int{7}},
		{"0, 1, 2", []int{0, 1, 2}},
	}
	for _, tc := range cases {
		got := ParsePhaseList(tc.in)
		if len(got) != len(tc.out) {
			t.Errorf("ParsePhaseList(%q) len = %d, want %d", tc.in, len(got), len(tc.out))
			continue
		}
		for i := range got {
			if got[i] != tc.out[i] {
				t.Errorf("ParsePhaseList(%q)[%d] = %d, want %d", tc.in, i, got[i], tc.out[i])
			}
		}
	}
}

func TestPhaseInList(t *testing.T) {
	list := []int{0, 2, 5}
	if !PhaseInList(0, list) {
		t.Error("expected 0 in list")
	}
	if PhaseInList(1, list) {
		t.Error("expected 1 not in list")
	}
	if !PhaseInList(5, list) {
		t.Error("expected 5 in list")
	}
	if PhaseInList(3, nil) {
		t.Error("expected 3 not in nil list")
	}
}

func TestEnvBoolTrueVariants(t *testing.T) {
	for _, v := range []string{"true", "1", "TRUE", "True"} {
		os.Setenv("SWEEP_RESUME", v)
		c := Defaults()
		if !c.Resume {
			t.Errorf("SWEEP_RESUME=%q should set Resume=true", v)
		}
	}
	os.Unsetenv("SWEEP_RESUME")
}

func TestEnvFloat_Override(t *testing.T) {
	t.Setenv("SWEEP_MIN_TG_TS", "5.5")
	c := Defaults()
	if c.MinTGTS != 5.5 {
		t.Errorf("MinTGTS from env = %f, want 5.5", c.MinTGTS)
	}
}

func TestEnvStr_FallsBackToDefault(t *testing.T) {
	os.Unsetenv("SWEEP_OUTPUT_DIR")
	c := Defaults()
	if c.OutputDir == "" {
		t.Error("expected non-empty OutputDir default")
	}
}

func TestValidate_OptimizedSweepConflict_AllFlags(t *testing.T) {
	n := 4
	c := Defaults()
	c.OptimizedSweep = true
	c.MinNGL = &n
	if err := c.Validate(); err == nil {
		t.Error("expected error: optimized-sweep + --min-ngl")
	}
}

func TestValidate_InvalidDirectionStr(t *testing.T) {
	c := Defaults()
	c.DirNGL = "sideways"
	if err := c.Validate(); err == nil {
		t.Error("expected error for invalid direction string")
	}
}

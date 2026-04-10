// Package phase implements each of the 8 sweep phases.
package phase

import (
	"context"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/config"
	"github.com/WagnerJust/llamaseye/hardware"
	"github.com/WagnerJust/llamaseye/output"
	"github.com/WagnerJust/llamaseye/state"
)

// Phase is implemented by each of the 8 sweep phases.
type Phase interface {
	ID() int
	Label() string
	Run(ctx context.Context, env *PhaseEnv) error
}

// BestConfig tracks the current best parameter values across phases.
type BestConfig struct {
	NGL     int
	FA      int
	CTK     string
	CTV     string
	Threads *int // nil = system default
	NKVO    int
	B       int
	UB      int
	CTX     int
}

// WorkingSets holds the accumulated ok values from each phase.
type WorkingSets struct {
	NGL       []int
	FACTK     []state.FACTKCombo
	CTKValues []string // independent CTK axis populated by Phase 2, used by Phase 7
	CTVValues []string // independent CTV axis populated by Phase 2, used by Phase 7
	Threads   state.ThreadValues // nil entry = system_default
	NKVO      []int
	BUB       []state.BUBCombo
	CTX       []int
}

// PhaseEnv is the shared mutable state passed to every phase.
type PhaseEnv struct {
	Config      *config.Config
	HW          *hardware.HardwareInfo
	Runner      *bench.BenchRunner
	Thermal     *hardware.ThermalMonitor
	Logger      *output.Logger
	JSONLWriter *output.JSONLWriter // opened once per SweepModel, closed by caller
	MaxNGL      int
	NumLayers   int // model layer count from GGUF metadata; 0 = unknown
	Best        BestConfig
	WS          WorkingSets
	OutputDir   string
	ModelPath   string
	ModelStem   string

	// SkipCombos holds combo keys from sweep.jsonl when --focused is active.
	// Keyed by phase ID → combo key string → performance data.
	SkipCombos map[int]map[string]output.ExistingCombo
}

// NewPhaseEnv creates a PhaseEnv with default best values.
func NewPhaseEnv(cfg *config.Config, hw *hardware.HardwareInfo,
	runner *bench.BenchRunner, thermal *hardware.ThermalMonitor,
	logger *output.Logger, outputDir, modelPath, modelStem string) *PhaseEnv {
	return &PhaseEnv{
		Config:    cfg,
		HW:        hw,
		Runner:    runner,
		Thermal:   thermal,
		Logger:    logger,
		MaxNGL:    99,
		Best:      defaultBest(),
		OutputDir: outputDir,
		ModelPath: modelPath,
		ModelStem: modelStem,
	}
}

func defaultBest() BestConfig {
	return BestConfig{
		NGL:  99,
		FA:   0,
		CTK:  "f16",
		CTV:  "f16",
		NKVO: 0,
		B:    2048,
		UB:   512,
		CTX:  512,
	}
}

// LoadFromState restores MaxNGL, Best, and WS from a saved state.
func (env *PhaseEnv) LoadFromState(s *state.State) {
	env.MaxNGL = s.MaxNGL
	env.Best = BestConfig{
		NGL:  s.Best.NGL,
		FA:   s.Best.FA,
		CTK:  s.Best.CTK,
		CTV:  s.Best.CTV,
		NKVO: s.Best.NKVO,
		B:    s.Best.B,
		UB:   s.Best.UB,
		CTX:  s.Best.CTX,
	}
	if s.Best.Threads != nil {
		t := *s.Best.Threads
		env.Best.Threads = &t
	}
	env.WS = WorkingSets{
		NGL:       s.WorkingSets.NGL,
		FACTK:     s.WorkingSets.FACTKCombos,
		CTKValues: s.WorkingSets.CTKValues,
		CTVValues: s.WorkingSets.CTVValues,
		NKVO:      s.WorkingSets.NKVOValues,
		BUB:       s.WorkingSets.BUBCombos,
		CTX:       s.WorkingSets.CTXValues,
		Threads:   s.WorkingSets.ThreadValues,
	}
	// Migration: old state.json files lack ctk_values/ctv_values — derive from fa_ctk_combos.
	if len(env.WS.CTKValues) == 0 {
		env.WS.CTKValues = UniqueCTKValues(env.WS.FACTK)
	}
	if len(env.WS.CTVValues) == 0 {
		env.WS.CTVValues = UniqueCTVValues(env.WS.FACTK)
	}
}

// ToState converts the current env into a state.State for persistence.
func (env *PhaseEnv) ToState(phasesComplete []int) *state.State {
	s := &state.State{
		ModelPath:      env.ModelPath,
		ModelStem:      env.ModelStem,
		MaxNGL:         env.MaxNGL,
		PhasesComplete: phasesComplete,
		Best: state.Best{
			NGL:  env.Best.NGL,
			FA:   env.Best.FA,
			CTK:  env.Best.CTK,
			CTV:  env.Best.CTV,
			NKVO: env.Best.NKVO,
			B:    env.Best.B,
			UB:   env.Best.UB,
			CTX:  env.Best.CTX,
		},
		WorkingSets: state.WorkingSets{
			NGL:          env.WS.NGL,
			FACTKCombos:  env.WS.FACTK,
			CTKValues:    env.WS.CTKValues,
			CTVValues:    env.WS.CTVValues,
			NKVOValues:   env.WS.NKVO,
			BUBCombos:    env.WS.BUB,
			CTXValues:    env.WS.CTX,
			ThreadValues: env.WS.Threads,
		},
	}
	if env.Best.Threads != nil {
		t := *env.Best.Threads
		s.Best.Threads = &t
	}
	return s
}

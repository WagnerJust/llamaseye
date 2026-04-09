package phase

import (
	"context"
	"fmt"
	"strings"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/state"
)

// faCombo represents a flash-attention × KV-quant combination to test.
type faCombo struct{ FA int; CTK, CTV string }

// P2FAKVSweep sweeps flash-attention × KV quant combinations.
type P2FAKVSweep struct{}

func (P2FAKVSweep) ID() int       { return 2 }
func (P2FAKVSweep) Label() string { return "fa_kv_sweep" }

func (P2FAKVSweep) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 2] Flash attention × KV quant sweep")

	// Standard combos
	combos := []faCombo{
		{0, "f16", "f16"},
		{1, "f16", "f16"},
		{0, "q8_0", "q8_0"},
		{1, "q8_0", "q8_0"},
		{1, "q4_0", "q4_0"},
	}
	// Turbo combos (only when TurboQuant binary available)
	if env.Runner.Selector.TurboAvailable {
		combos = append(combos,
			faCombo{1, "turbo4", "turbo4"},
			faCombo{1, "turbo3", "turbo3"},
			faCombo{0, "turbo2", "turbo2"},
			faCombo{1, "turbo2", "turbo2"},
		)
		// Asymmetric K/V combos: V compression is effectively free; only K affects quality.
		// These are omitted when --no-asymmetric-kv is set.
		if env.Config.AsymmetricKV {
			combos = append(combos,
				faCombo{1, "q8_0", "turbo4"},   // High K precision, meaningful V compression
				faCombo{1, "q8_0", "turbo3"},   // High K precision, max V compression
				faCombo{1, "q8_0", "turbo2"},   // High K precision, extreme V compression
				faCombo{1, "f16", "turbo3"},    // Full K precision, strong V compression
				faCombo{1, "turbo4", "turbo2"}, // Moderate K, aggressive V
			)
		}
	}
	// RotorQuant combos (only when RotorQuant binary available)
	if env.Runner.Selector.RotorAvailable {
		combos = append(combos,
			faCombo{1, "iso4", "iso4"},
			faCombo{1, "planar4", "planar4"},
			faCombo{1, "iso3", "iso3"},
			faCombo{1, "planar3", "planar3"},
		)
		if env.Config.AsymmetricKV {
			combos = append(combos,
				faCombo{1, "q8_0", "iso3"},    // High K precision, strong V compression
				faCombo{1, "q8_0", "planar3"}, // High K precision, strong V compression
				faCombo{1, "f16", "iso3"},     // Full K precision, strong V compression
			)
		}
	}

	// Apply FA direction filter
	faOrder := ApplyAxisOpts([]string{"0", "1"}, intPtrStr(env.Config.StartFA), env.Config.DirFA,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	faSet := make(map[string]bool)
	for _, v := range faOrder {
		faSet[v] = true
	}

	// Apply CTK direction filter using the shared quality ordering (desc = toward more compression).
	// CTKQualityOrder is low→high quality; reverse it for the "up" direction (f16 first → turbo2 last).
	ctkFullOrder := make([]string, len(CTKQualityOrder))
	for i, v := range CTKQualityOrder {
		ctkFullOrder[len(CTKQualityOrder)-1-i] = v
	}
	ctkFiltered := ApplyAxisOpts(ctkFullOrder, env.Config.StartCTK, env.Config.DirCTK,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	ctkSet := make(map[string]bool)
	for _, v := range ctkFiltered {
		ctkSet[v] = true
	}

	bestTG := -1.0
	env.WS.FACTK = nil

	for _, combo := range combos {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		faStr := itoa(combo.FA)
		if !faSet[faStr] {
			continue
		}
		if !ctkSet[combo.CTK] {
			continue
		}
		// fa=0 + q4_0 is invalid
		if combo.FA == 0 && combo.CTK == "q4_0" {
			continue
		}
		// turbo/rotor types only work when their binary is available
		if strings.HasPrefix(combo.CTK, "turbo") && !env.Runner.Selector.TurboAvailable {
			continue
		}
		if (strings.HasPrefix(combo.CTK, "planar") || strings.HasPrefix(combo.CTK, "iso")) && !env.Runner.Selector.RotorAvailable {
			continue
		}

		label := fmt.Sprintf("phase2/fa=%d_ctk=%s_ctv=%s", combo.FA, combo.CTK, combo.CTV)
		status, tg, _ := RecordAndTrack(env, label, bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         combo.FA,
			CTK:        combo.CTK,
			CTV:        combo.CTV,
			NKVO:       0,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    512,
			NGen:       128,
			Reps:       env.Config.Repetitions,
			Phase:      2,
			PhaseLabel: "fa_kv_sweep",
		})

		if status == bench.StatusOK {
			env.WS.FACTK = append(env.WS.FACTK, state.FACTKCombo{
				FA:  combo.FA,
				CTK: combo.CTK,
				CTV: combo.CTV,
			})
			if tg > bestTG {
				bestTG = tg
				env.Best.FA = combo.FA
				env.Best.CTK = combo.CTK
				env.Best.CTV = combo.CTV
			}
		}
	}

	if len(env.WS.FACTK) == 0 {
		env.WS.FACTK = []state.FACTKCombo{{FA: 0, CTK: "f16", CTV: "f16"}}
	}
	// Derive independent CTK/CTV working sets for Phase 7.
	env.WS.CTKValues = UniqueCTKValues(env.WS.FACTK)
	env.WS.CTVValues = UniqueCTVValues(env.WS.FACTK)

	env.Logger.Log("[Phase 2] Best: fa=%d ctk=%s (TG=%.2f t/s)",
		env.Best.FA, env.Best.CTK, bestTG)
	return nil
}

// intPtrStr converts *int to string representation, or "" if nil.
func intPtrStr(p *int) string {
	if p == nil {
		return ""
	}
	return itoa(*p)
}

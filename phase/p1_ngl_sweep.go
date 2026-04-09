package phase

import (
	"context"
	"fmt"

	"github.com/WagnerJust/llamaseye/bench"
)

// P1NGLSweep sweeps NGL near MAX_NGL to find the best GPU layer count.
type P1NGLSweep struct{}

func (P1NGLSweep) ID() int       { return 1 }
func (P1NGLSweep) Label() string { return "ngl_sweep" }

func (P1NGLSweep) Run(ctx context.Context, env *PhaseEnv) error {
	step := env.Config.NGLStep
	env.Logger.Log("[Phase 1] NGL axis sweep (step=%d)", step)

	// Cap the sweep ceiling: NGL values above NumLayers are functionally identical
	// because llama.cpp silently clamps NGL to the model's actual layer count.
	nglCeiling := env.MaxNGL
	if env.NumLayers > 0 && env.NumLayers < env.MaxNGL {
		nglCeiling = env.NumLayers
		env.Logger.Log("[Phase 1] Model has %d layers — capping NGL sweep at %d (values above are identical)",
			env.NumLayers, nglCeiling)
	}

	// Build full NGL list: 0, step, 2*step, ..., nglCeiling
	var fullList []int
	fullList = append(fullList, 0)
	for n := step; n < nglCeiling; n += step {
		fullList = append(fullList, n)
	}
	fullList = append(fullList, nglCeiling)
	fullList = dedupeInts(fullList)

	// Smart default start: 2 steps below MAX_NGL
	effectiveStart := env.Config.StartNGL
	if effectiveStart == nil {
		stepsFromMax := 2
		smartStart := (env.MaxNGL/step - stepsFromMax) * step
		if smartStart < 0 {
			smartStart = 0
		}
		effectiveStart = &smartStart
		env.Logger.Log("[Phase 1] Auto start-ngl=%d (max_ngl=%d − %d×step); use --start-ngl 0 for full sweep",
			smartStart, env.MaxNGL, stepsFromMax)
	}

	nglList := ApplyAxisOptsInt(fullList, effectiveStart, env.Config.DirNGL,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })

	bestTG := -1.0
	env.WS.NGL = nil

	for _, ngl := range nglList {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		comboKey := fmt.Sprintf("%d", ngl)
		if existing, skip := ShouldSkip(env, 1, comboKey); skip {
			env.WS.NGL = append(env.WS.NGL, ngl)
			if existing.TG > bestTG {
				bestTG = existing.TG
				env.Best.NGL = ngl
			}
			continue
		}

		status, tg, _ := RecordAndTrack(env, fmt.Sprintf("phase1/ngl=%d", ngl), bench.RunParams{
			NGL:        ngl,
			FA:         0,
			CTK:        "f16",
			CTV:        "f16",
			NKVO:       0,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    512,
			NGen:       128,
			Reps:       env.Config.Repetitions,
			Phase:      1,
			PhaseLabel: "ngl_sweep",
		})

		if status == bench.StatusOK {
			env.WS.NGL = append(env.WS.NGL, ngl)
			if tg > bestTG {
				bestTG = tg
				env.Best.NGL = ngl
			}
		}
	}

	if len(env.WS.NGL) == 0 {
		env.WS.NGL = []int{env.MaxNGL}
	}
	env.Logger.Log("[Phase 1] Best NGL: %d (TG=%.2f t/s)  Working set: %v",
		env.Best.NGL, bestTG, env.WS.NGL)
	return nil
}

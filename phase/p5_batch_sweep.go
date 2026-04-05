package phase

import (
	"context"
	"fmt"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/state"
)

// P5BatchSweep sweeps batch and micro-batch size pairs.
type P5BatchSweep struct{}

func (P5BatchSweep) ID() int       { return 5 }
func (P5BatchSweep) Label() string { return "batch_sweep" }

func (P5BatchSweep) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 5] Batch / ubatch sweep")

	// All candidate pairs (b, ub) — ub must be <= b
	allPairs := []state.BUBCombo{
		{B: 2048, UB: 512},
		{B: 2048, UB: 256},
		{B: 2048, UB: 128},
		{B: 1024, UB: 512},
		{B: 1024, UB: 256},
		{B: 1024, UB: 128},
		{B: 512, UB: 256},
		{B: 512, UB: 128},
	}

	bFiltered := ApplyAxisOptsInt([]int{512, 1024, 2048}, env.Config.StartB, env.Config.DirB,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	ubFiltered := ApplyAxisOptsInt([]int{128, 256, 512}, env.Config.StartUB, env.Config.DirUB,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })

	bSet := make(map[int]bool)
	for _, v := range bFiltered {
		bSet[v] = true
	}
	ubSet := make(map[int]bool)
	for _, v := range ubFiltered {
		ubSet[v] = true
	}

	bestPP := -1.0
	env.WS.BUB = nil

	for _, pair := range allPairs {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if !bSet[pair.B] || !ubSet[pair.UB] {
			continue
		}
		if pair.UB > pair.B {
			continue
		}

		label := fmt.Sprintf("phase5/b=%d_ub=%d", pair.B, pair.UB)
		status, _, pp := RecordAndTrack(env, label, bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         env.Best.FA,
			CTK:        env.Best.CTK,
			CTV:        env.Best.CTV,
			Threads:    env.Best.Threads,
			NKVO:       env.Best.NKVO,
			B:          pair.B,
			UB:         pair.UB,
			NPrompt:    512,
			NGen:       0,
			Reps:       env.Config.Repetitions,
			Phase:      5,
			PhaseLabel: "batch_sweep",
		})
		if status == bench.StatusOK {
			env.WS.BUB = append(env.WS.BUB, pair)
			if pp > bestPP {
				bestPP = pp
				env.Best.B = pair.B
				env.Best.UB = pair.UB
			}
		}
	}

	if len(env.WS.BUB) == 0 {
		env.WS.BUB = []state.BUBCombo{{B: 2048, UB: 512}}
	}
	env.Logger.Log("[Phase 5] Best batch: b=%d ub=%d (PP=%.2f t/s)",
		env.Best.B, env.Best.UB, bestPP)
	return nil
}

package phase

import (
	"context"
	"fmt"
	"sort"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/state"
)

// P3ThreadSweep sweeps CPU thread counts.
type P3ThreadSweep struct{}

func (P3ThreadSweep) ID() int       { return 3 }
func (P3ThreadSweep) Label() string { return "thread_sweep" }

func (P3ThreadSweep) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 3] CPU thread count sweep")

	physical := env.HW.CPUPhysical
	logical := env.HW.CPULogical

	// Build candidate list
	candidates := []int{1}
	if half := physical / 2; half > 1 {
		candidates = append(candidates, half)
	}
	candidates = append(candidates, physical)
	if threeQtr := (physical + logical) / 2; threeQtr > physical && threeQtr < logical {
		candidates = append(candidates, threeQtr)
	}
	if logical > physical {
		candidates = append(candidates, logical)
	}

	// Sort and deduplicate
	sort.Ints(candidates)
	candidates = dedupeInts(candidates)

	threadList := ApplyAxisOptsInt(candidates, env.Config.StartThreads, env.Config.DirThreads,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })

	bestTG := -1.0
	env.WS.Threads = nil

	// System default run first (no -t flag)
	if existing, skip := ShouldSkip(env, 3, "sys"); skip {
		env.WS.Threads = append(env.WS.Threads, nil)
		if existing.TG > bestTG {
			bestTG = existing.TG
		}
	} else {
		status, tg, _ := RecordAndTrack(ctx, env, "phase3/threads=system_default", bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         env.Best.FA,
			CTK:        env.Best.CTK,
			CTV:        env.Best.CTV,
			NKVO:       0,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    512,
			NGen:       128,
			Reps:       env.Config.Repetitions,
			Phase:      3,
			PhaseLabel: "thread_sweep",
		})
		if status == bench.StatusOK {
			env.WS.Threads = append(env.WS.Threads, nil)
			if tg > bestTG {
				bestTG = tg
				// system default → leave env.Best.Threads nil
			}
		}
	}

	for _, t := range threadList {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		comboKey := fmt.Sprintf("%d", t)
		if existing, skip := ShouldSkip(env, 3, comboKey); skip {
			tc := t; env.WS.Threads = append(env.WS.Threads, &tc)
			if existing.TG > bestTG {
				bestTG = existing.TG
				tc := t
				env.Best.Threads = &tc
			}
			continue
		}

		tc := t // capture
		status, tg, _ := RecordAndTrack(ctx, env, fmt.Sprintf("phase3/threads=%d", t), bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         env.Best.FA,
			CTK:        env.Best.CTK,
			CTV:        env.Best.CTV,
			NKVO:       0,
			Threads:    &tc,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    512,
			NGen:       128,
			Reps:       env.Config.Repetitions,
			Phase:      3,
			PhaseLabel: "thread_sweep",
		})
		if status == bench.StatusOK {
			tc := t; env.WS.Threads = append(env.WS.Threads, &tc)
			if tg > bestTG {
				bestTG = tg
				tc2 := t
				env.Best.Threads = &tc2
			}
		}
	}

	if len(env.WS.Threads) == 0 {
		env.WS.Threads = state.ThreadValues{nil}
	}
	bestThreadsStr := "system default"
	if env.Best.Threads != nil {
		bestThreadsStr = itoa(*env.Best.Threads)
	}
	env.Logger.Log("[Phase 3] Best threads: %s (TG=%.2f t/s)", bestThreadsStr, bestTG)
	return nil
}

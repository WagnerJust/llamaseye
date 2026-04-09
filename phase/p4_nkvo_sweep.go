package phase

import (
	"context"
	"fmt"

	"github.com/WagnerJust/llamaseye/bench"
)

// P4NKVOSweep tests KV offload combinations.
type P4NKVOSweep struct{}

func (P4NKVOSweep) ID() int       { return 4 }
func (P4NKVOSweep) Label() string { return "nkvo_sweep" }

func (P4NKVOSweep) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 4] KV offload sweep (nkvo)")
	bestTG := -1.0
	env.WS.NKVO = nil

	for _, nkvo := range []int{0, 1} {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		comboKey := fmt.Sprintf("%d", nkvo)
		if existing, skip := ShouldSkip(env, 4, comboKey); skip {
			env.WS.NKVO = append(env.WS.NKVO, nkvo)
			if existing.TG > bestTG {
				bestTG = existing.TG
				env.Best.NKVO = nkvo
			}
			continue
		}

		status, tg, _ := RecordAndTrack(env, fmt.Sprintf("phase4/nkvo=%d", nkvo), bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         env.Best.FA,
			CTK:        env.Best.CTK,
			CTV:        env.Best.CTV,
			Threads:    env.Best.Threads,
			NKVO:       nkvo,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    512,
			NGen:       128,
			Reps:       env.Config.Repetitions,
			Phase:      4,
			PhaseLabel: "nkvo_sweep",
		})
		if status == bench.StatusOK {
			env.WS.NKVO = append(env.WS.NKVO, nkvo)
			if tg > bestTG {
				bestTG = tg
				env.Best.NKVO = nkvo
			}
		}
	}

	// Also test nkvo=1 at higher NGL values if max was reached before 99
	if env.MaxNGL < 99 {
		for _, extra := range []int{env.MaxNGL + 4, env.MaxNGL + 8, env.MaxNGL + 12} {
			if extra > 99 {
				break
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}
			status, _, _ := RecordAndTrack(env, fmt.Sprintf("phase4/nkvo=1_ngl=%d", extra), bench.RunParams{
				NGL:        extra,
				FA:         env.Best.FA,
				CTK:        env.Best.CTK,
				CTV:        env.Best.CTV,
				Threads:    env.Best.Threads,
				NKVO:       1,
				B:          env.Best.B,
				UB:         env.Best.UB,
				NPrompt:    512,
				NGen:       128,
				Reps:       env.Config.Repetitions,
				Phase:      4,
				PhaseLabel: "nkvo_sweep",
			})
			if status == bench.StatusOOM || status == bench.StatusError || status == bench.StatusTimeout {
				break
			}
		}
	}

	if len(env.WS.NKVO) == 0 {
		env.WS.NKVO = []int{0}
	}
	env.Logger.Log("[Phase 4] Best nkvo: %d (TG=%.2f t/s)", env.Best.NKVO, bestTG)
	return nil
}

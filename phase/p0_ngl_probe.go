package phase

import (
	"context"
	"fmt"

	"github.com/justinphilpott/llamaseye/bench"
)

// P0NGLProbe implements Phase 0: binary-step probe to find max stable NGL.
type P0NGLProbe struct{}

func (P0NGLProbe) ID() int    { return 0 }
func (P0NGLProbe) Label() string { return "ngl_probe" }

func (P0NGLProbe) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 0] NGL probe — finding max stable NGL")

	if env.HW.GPUCount == 0 {
		env.Logger.Log("[Phase 0] No GPU detected — skipping probe, setting MAX_NGL=0")
		env.MaxNGL = 0
		env.Best.NGL = 0
		return nil
	}

	for ngl := 99; ngl >= 0; ngl -= 4 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		env.Logger.Log("[Phase 0] Probing ngl=%d", ngl)
		status, _, _ := RecordAndTrack(env, "phase0/ngl="+itoa(ngl), bench.RunParams{
			NGL:        ngl,
			FA:         0,
			CTK:        "f16",
			CTV:        "f16",
			NKVO:       0,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    64,
			NGen:       0,
			Reps:       env.Config.ProbeReps,
			Phase:      0,
			PhaseLabel: "ngl_probe",
		})

		if status == bench.StatusOK || status == bench.StatusDryRun {
			env.MaxNGL = ngl
			env.Best.NGL = ngl
			env.Logger.Log("[Phase 0] max_ngl=%d", ngl)
			return nil
		}
	}

	return fmt.Errorf("model cannot be loaded at any NGL value on this hardware")
}

package phase

import (
	"context"
	"fmt"
	"strings"

	"github.com/WagnerJust/llamaseye/bench"
)

// P6CtxSweep sweeps context window sizes with OOM fallback logic.
type P6CtxSweep struct{}

func (P6CtxSweep) ID() int       { return 6 }
func (P6CtxSweep) Label() string { return "ctx_sweep" }

var fullCTXList = []int{128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}

func (P6CtxSweep) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 6] Context size sweep")

	ctxList := applyCtxAxisOpts(env)

	// KV quality order (index 0 = most compressed / lowest quality, last = f16 / highest quality).
	// Shared with common.CTKQualityOrder — do not duplicate.
	kvQualityOrder := CTKQualityOrder

	// Resolve effective starting CTK/CTV for Phase 6.
	// --p6-ctk / --p6-ctv override env.Best when non-empty.
	effectiveCTK := env.Best.CTK
	if env.Config.P6CTK != "" {
		if CTKQualityIndex(env.Config.P6CTK) == -1 {
			env.Logger.Warn("[Phase 6] --p6-ctk value %q is not a known KV type — falling back to best (%s)",
				env.Config.P6CTK, env.Best.CTK)
		} else {
			effectiveCTK = env.Config.P6CTK
			env.Logger.Log("[Phase 6] CTK override: %s (best from prior phases was %s)", effectiveCTK, env.Best.CTK)
		}
	}
	effectiveCTV := env.Best.CTV
	if env.Config.P6CTV != "" {
		if CTKQualityIndex(env.Config.P6CTV) == -1 {
			env.Logger.Warn("[Phase 6] --p6-ctv value %q is not a known KV type — falling back to best (%s)",
				env.Config.P6CTV, env.Best.CTV)
		} else {
			effectiveCTV = env.Config.P6CTV
			env.Logger.Log("[Phase 6] CTV override: %s (best from prior phases was %s)", effectiveCTV, env.Best.CTV)
		}
	}

	bestCTKIdx := len(kvQualityOrder) - 1 // default = f16
	for i, v := range kvQualityOrder {
		if v == effectiveCTK {
			bestCTKIdx = i
			break
		}
	}
	bestCTVIdx := len(kvQualityOrder) - 1 // default = f16
	for i, v := range kvQualityOrder {
		if v == effectiveCTV {
			bestCTVIdx = i
			break
		}
	}

	env.WS.CTX = nil
	env.Best.CTX = 128

	for _, ctxVal := range ctxList {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		comboKey := fmt.Sprintf("%d", ctxVal)
		if _, skip := ShouldSkip(env, 6, comboKey); skip {
			env.WS.CTX = appendIfMissing(env.WS.CTX, ctxVal)
			env.Best.CTX = ctxVal
			continue
		}

		result := p6TryCtx(ctx, env, ctxVal, kvQualityOrder, bestCTKIdx, bestCTVIdx, effectiveCTK, effectiveCTV)

		if result == "ok" {
			continue
		}
		if result == "timeout" {
			env.Logger.Log("[Phase 6] Stopping at ctx=%d (timeout)", ctxVal)
			break
		}

		// All fallbacks failed — optionally bisect
		if env.Config.FineCtx && env.Best.CTX > 0 {
			hi := ctxVal
			minStep := env.Config.CtxStepMin
			env.Logger.Log("[Phase 6] Bisecting between ctx=%d and ctx=%d (step-min=%d)",
				env.Best.CTX, hi, minStep)

		bisectLoop:
			for hi-env.Best.CTX > minStep {
				mid := ((env.Best.CTX + hi) / 2)
				// Round to nearest 512
				mid = (mid+256)/512*512
				if mid <= env.Best.CTX || mid >= hi {
					break
				}
				env.Logger.Log("[Phase 6] Bisect probe: ctx=%d (range %d–%d)", mid, env.Best.CTX, hi)
				bisectResult := p6TryCtx(ctx, env, mid, kvQualityOrder, bestCTKIdx, bestCTVIdx, effectiveCTK, effectiveCTV)
				switch bisectResult {
				case "ok":
					// BEST_CTX updated by p6TryCtx
				case "timeout":
					env.Logger.Log("[Phase 6] Bisect: timeout at ctx=%d — stopping", mid)
					break bisectLoop
				default:
					hi = mid
				}
			}
			env.Logger.Log("[Phase 6] Bisect complete — best ctx: %d", env.Best.CTX)
		}
		break
	}

	if len(env.WS.CTX) == 0 {
		if env.Config.StartCtx != nil {
			env.Logger.Warn("[Phase 6] No context ≥ %d succeeded. WS_CTX will be empty — Phase 7 will produce 0 combinations for ctx.",
				*env.Config.StartCtx)
			env.Logger.Warn("[Phase 6] To include smaller contexts, re-run with --start-ctx lowered or omitted.")
		} else {
			env.WS.CTX = []int{512}
		}
	}
	env.Logger.Log("[Phase 6] Best (max) context: %d  Working set: %v",
		env.Best.CTX, env.WS.CTX)
	return nil
}

// p6TryCtx tries the primary config at ctx, then fallbacks if it fails.
// Returns "ok", "timeout", or "fail".
// effectiveCTK/CTV are the resolved starting types (may be overridden via --p6-ctk/--p6-ctv).
func p6TryCtx(ctx context.Context, env *PhaseEnv, ctxVal int,
	kvOrder []string, bestCTKIdx, bestCTVIdx int, effectiveCTK, effectiveCTV string) string {

	// Primary config
	status, _, _ := RecordAndTrack(ctx, env, fmt.Sprintf("phase6/ctx=%d", ctxVal), bench.RunParams{
		NGL:        env.Best.NGL,
		FA:         env.Best.FA,
		CTK:        effectiveCTK,
		CTV:        effectiveCTV,
		Threads:    env.Best.Threads,
		NKVO:       env.Best.NKVO,
		B:          env.Best.B,
		UB:         env.Best.UB,
		NPrompt:    ctxVal,
		NGen:       0,
		Reps:       2,
		Phase:      6,
		PhaseLabel: "ctx_sweep",
	})

	if status == bench.StatusOK || status == bench.StatusDryRun {
		env.WS.CTX = appendIfMissing(env.WS.CTX, ctxVal)
		env.Best.CTX = ctxVal
		return "ok"
	}
	if status == bench.StatusTimeout {
		return "timeout"
	}

	// OOM or error — try fallbacks
	env.Logger.Log("[Phase 6] ctx=%d failed (%s) with best config — trying fallbacks", ctxVal, status)

	type fallback struct {
		FA   int
		CTK  string
		CTV  string
		NKVO int
	}

	var fallbacks []fallback

	// Part 1: nkvo flip (keep current CTK/CTV)
	altNKVO := 1 - env.Best.NKVO
	if containsInt(env.WS.NKVO, altNKVO) {
		fallbacks = append(fallbacks, fallback{
			FA:   env.Best.FA,
			CTK:  env.Best.CTK,
			CTV:  env.Best.CTV,
			NKVO: altNKVO,
		})
	}

	// Part 2: V-first — keep CTK fixed, try more-compressed CTV types.
	// V compression is effectively free quality-wise; exhaust V before touching K.
	for i := 0; i < bestCTVIdx; i++ {
		fbCTV := kvOrder[i]
		if !kvTypeAvailable(fbCTV, env.Runner.Selector.TurboAvailable, env.Runner.Selector.RotorAvailable) {
			continue
		}
		fa, found := FindFACTKByKV(env.WS.FACTK, env.Best.CTK, fbCTV)
		if !found {
			continue
		}
		for _, nkvoFB := range []int{0, 1} {
			if !containsInt(env.WS.NKVO, nkvoFB) {
				continue
			}
			fallbacks = append(fallbacks, fallback{
				FA:   fa,
				CTK:  env.Best.CTK,
				CTV:  fbCTV,
				NKVO: nkvoFB,
			})
		}
	}

	// Part 3: K+V — try more-compressed CTK types with their paired CTV.
	// Only reached when V-first fallbacks are exhausted.
	for i := 0; i < bestCTKIdx; i++ {
		fbCTK := kvOrder[i]
		if !kvTypeAvailable(fbCTK, env.Runner.Selector.TurboAvailable, env.Runner.Selector.RotorAvailable) {
			continue
		}
		fa, ctv, found := BestFACTVForCTK(env.WS.FACTK, fbCTK)
		if !found {
			continue
		}
		for _, nkvoFB := range []int{0, 1} {
			if !containsInt(env.WS.NKVO, nkvoFB) {
				continue
			}
			fallbacks = append(fallbacks, fallback{
				FA:   fa,
				CTK:  fbCTK,
				CTV:  ctv,
				NKVO: nkvoFB,
			})
		}
	}

	for _, fb := range fallbacks {
		select {
		case <-ctx.Done():
			return "timeout"
		default:
		}

		label := fmt.Sprintf("phase6/ctx=%d/nkvo=%d_ctk=%s_ctv=%s", ctxVal, fb.NKVO, fb.CTK, fb.CTV)
		fbStatus, _, _ := RecordAndTrack(ctx, env, label, bench.RunParams{
			NGL:        env.Best.NGL,
			FA:         fb.FA,
			CTK:        fb.CTK,
			CTV:        fb.CTV,
			Threads:    env.Best.Threads,
			NKVO:       fb.NKVO,
			B:          env.Best.B,
			UB:         env.Best.UB,
			NPrompt:    ctxVal,
			NGen:       0,
			Reps:       2,
			Phase:      6,
			PhaseLabel: "ctx_sweep",
		})
		if fbStatus == bench.StatusOK {
			env.Logger.Log("[Phase 6] ctx=%d succeeded with fallback nkvo=%d ctk=%s ctv=%s",
				ctxVal, fb.NKVO, fb.CTK, fb.CTV)
			env.WS.CTX = appendIfMissing(env.WS.CTX, ctxVal)
			env.Best.CTX = ctxVal
			return "ok"
		}
		// continue through all fallbacks (timeout on one doesn't predict others)
	}

	return "fail"
}

// kvTypeAvailable reports whether the binary for the given KV type is available.
func kvTypeAvailable(t string, turboAvail, rotorAvail bool) bool {
	if strings.HasPrefix(t, "turbo") {
		return turboAvail
	}
	if strings.HasPrefix(t, "planar") || strings.HasPrefix(t, "iso") {
		return rotorAvail
	}
	return true
}

func applyCtxAxisOpts(env *PhaseEnv) []int {
	var startStr string
	if env.Config.StartCtx != nil {
		startStr = itoa(*env.Config.StartCtx)
	}
	strList := intSliceToStrings(fullCTXList)
	filtered := ApplyAxisOpts(strList, startStr, env.Config.DirCtx,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	return stringsToIntSlice(filtered)
}

func appendIfMissing(slice []int, val int) []int {
	for _, v := range slice {
		if v == val {
			return slice
		}
	}
	return append(slice, val)
}

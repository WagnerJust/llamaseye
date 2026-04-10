package phase

import (
	"context"
	"fmt"

	"github.com/WagnerJust/llamaseye/bench"
	"github.com/WagnerJust/llamaseye/output"
	"github.com/WagnerJust/llamaseye/state"
)

// GoalConfig holds the parsed goal specification for phase 7.
type GoalConfig struct {
	CtxMin  int
	TGMin   float64
	PPMin   float64
	MaxHits int // stop after this many goal-satisfying configs (default 3)
}

// P7CombinationMatrix runs the cartesian product of all working sets.
type P7CombinationMatrix struct {
	Goal *GoalConfig // nil = exhaustive mode
}

func (P7CombinationMatrix) ID() int       { return 7 }
func (P7CombinationMatrix) Label() string { return "combination_matrix" }

func (p P7CombinationMatrix) Run(ctx context.Context, env *PhaseEnv) error {
	env.Logger.Log("[Phase 7] Full combination matrix")

	// Auto-derive Phase 7 minimums
	effMinNGL := derivedMinNGL(env)
	effMinThreads := derivedMinThreads(env)
	effMinCtx := derivedMinCtx(env, p.Goal)
	effMinB := derivedMinB(env)
	effMinCTK := derivedMinCTK(env)

	// Apply minimums to working sets
	nglP7 := ApplyPhase7MinsInt(env.WS.NGL, effMinNGL,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	ctxP7 := ApplyPhase7MinsInt(env.WS.CTX, effMinCtx,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	bubP7 := filterBUBByMinB(env.WS.BUB, effMinB)
	if len(bubP7) == 0 {
		bubP7 = env.WS.BUB
	}

	// Thread working set
	threadP7 := filterThreadsByMin(env.WS.Threads, effMinThreads)

	// CTK values: from independent working set, filtered by min-ctk.
	ctkP7 := ApplyPhase7MinsCTK(env.WS.CTKValues, effMinCTK,
		func(f string, a ...any) { env.Logger.Warn(f, a...) })
	// CTV values: all types discovered by Phase 2 (no min filter — that's #32).
	ctvP7 := env.WS.CTVValues

	nkvoP7 := env.WS.NKVO

	// Bail if nothing to test
	if len(ctxP7) == 0 {
		env.Logger.Warn("[Phase 7] No ctx values passed the minimum filter (min-ctx=%v). Phase 7 skipped.", effMinCtx)
		env.Logger.Warn("[Phase 7] To run Phase 7, lower or remove --min-ctx / --start-ctx, or re-run Phase 6 with a lower --start-ctx.")
		return nil
	}

	// Goal mode: sort NGL descending so best offload configs surface first
	if p.Goal != nil {
		sortIntsDesc(nglP7)
		env.Logger.Log("[Phase 7] Goal mode: ctx≥%d tg≥%.1f pp≥%.1f — stopping after %d distinct (ngl,ctk,ctv,nkvo,ctx) configs",
			p.Goal.CtxMin, p.Goal.TGMin, p.Goal.PPMin, p.Goal.MaxHits)
	}

	// Count valid (ctk, ctv) pairs after precision filter (K must be at least as precise as V).
	validKVPairs := 0
	for _, ctk := range ctkP7 {
		for _, ctv := range ctvP7 {
			if KVPrecisionValid(ctk, ctv) {
				validKVPairs++
			}
		}
	}
	if validKVPairs == 0 {
		env.Logger.Warn("[Phase 7] No valid (ctk, ctv) pairs after precision filter — check ctk_values/ctv_values in working set. Phase 7 skipped.")
		return nil
	}
	total := len(nglP7) * validKVPairs * len(threadP7) * len(nkvoP7) * len(bubP7) * len(ctxP7)
	env.Logger.Log("[Phase 7] Estimated combinations: %d (ngl×%d kv_pairs×%d threads×%d nkvo×%d b_ub×%d ctx×%d)",
		total, len(nglP7), validKVPairs, len(threadP7), len(nkvoP7), len(bubP7), len(ctxP7))

	// Context OOM ceiling cache: key = "ngl_ctk_ctv_nkvo"
	ctxCeil := make(map[string]int)

	runCount := 0
	skipCount := 0
	goalHits := 0
	goalDone := false
	// goalTuples tracks best TG seen per (ngl,ctk,nkvo,ctx) key.
	// A new hit is only counted when a key is seen for the first time.
	goalTuples := make(map[string]float64)

	for _, ngl := range nglP7 {
		if goalDone {
			break
		}
		for _, ctk := range ctkP7 {
			if goalDone {
				break
			}
			for _, ctv := range ctvP7 {
				if goalDone {
					break
				}
				// Skip combos where V is more precise than K — wasteful and not meaningful.
				if !KVPrecisionValid(ctk, ctv) {
					continue
				}
				fa := BestFAForCTK(env.WS.FACTK, ctk)
				for _, nkvo := range nkvoP7 {
					if goalDone {
						break
					}
					for _, threadVal := range threadP7 {
						if goalDone {
							break
						}
						for _, bub := range bubP7 {
							if goalDone {
								break
							}
							if bub.UB > bub.B {
								continue
							}
							for _, ctxVal := range ctxP7 {
								if goalDone {
									break
								}

								// Context ceiling pruning
								ceilKey := fmt.Sprintf("%d_%s_%s_%d", ngl, ctk, ctv, nkvo)
								if maxOK, found := ctxCeil[ceilKey]; found && ctxVal > maxOK {
									env.Logger.Log("[Phase 7] Skip ctx=%d for ngl=%d/ctk=%s/ctv=%s/nkvo=%d (OOM ceiling: %d)",
										ctxVal, ngl, ctk, ctv, nkvo, maxOK)
									continue
								}

								select {
								case <-ctx.Done():
									return ctx.Err()
								default:
								}

								p7Key := output.ComboKey(7, ngl, fa, ctk, ctv, nkvo, threadVal, bub.B, bub.UB, ctxVal)
								if _, skip := ShouldSkip(env, 7, p7Key); skip {
									skipCount++
									continue
								}

								label := fmt.Sprintf("p7/ngl=%d_fa=%d_ctk=%s_ctv=%s_nkvo=%d_b=%d_ub=%d_ctx=%d",
									ngl, fa, ctk, ctv, nkvo, bub.B, bub.UB, ctxVal)

								status, tg, pp := RecordAndTrack(ctx, env, label, bench.RunParams{
									NGL:        ngl,
									FA:         fa,
									CTK:        ctk,
									CTV:        ctv,
									NKVO:       nkvo,
									Threads:    threadVal,
									B:          bub.B,
									UB:         bub.UB,
									NPrompt:    ctxVal,
									NGen:       128,
									Reps:       env.Config.Repetitions,
									Phase:      7,
									PhaseLabel: "combination_matrix",
								})

								runCount++
								if runCount%10 == 0 {
									if skipCount > 0 {
										env.Logger.Log("[Phase 7] %d/%d combinations run (%d skipped via --focused)", runCount, total, skipCount)
									} else {
										env.Logger.Log("[Phase 7] %d/%d combinations run", runCount, total)
									}
								}

								if status == bench.StatusOOM || status == bench.StatusTimeout {
									// Record ceiling
									ceil := ctxVal / 2
									if ceil < 128 {
										ceil = 0
									}
									if existing, found := ctxCeil[ceilKey]; !found || ctxVal < existing {
										ctxCeil[ceilKey] = ceil
									}
								}

								// Goal mode check
								if p.Goal != nil && status == bench.StatusOK {
									meetsGoal := true
									if p.Goal.CtxMin > 0 && ctxVal < p.Goal.CtxMin {
										meetsGoal = false
									}
									if meetsGoal && p.Goal.TGMin > 0 && tg < p.Goal.TGMin {
										meetsGoal = false
									}
									if meetsGoal && p.Goal.PPMin > 0 && pp < p.Goal.PPMin {
										meetsGoal = false
									}
									if meetsGoal {
										tupleKey := fmt.Sprintf("%d_%s_%s_%d_%d", ngl, ctk, ctv, nkvo, ctxVal)
										if _, seen := goalTuples[tupleKey]; !seen {
											// New distinct (ngl,ctk,ctv,nkvo,ctx) combo
											goalHits++
											goalTuples[tupleKey] = tg
											env.Logger.Log("[Phase 7] Goal hit (%d/%d): ngl=%d ctk=%s ctv=%s nkvo=%d ctx=%d tg=%.2f t/s",
												goalHits, p.Goal.MaxHits, ngl, ctk, ctv, nkvo, ctxVal, tg)
											if goalHits >= p.Goal.MaxHits {
												env.Logger.Log("[Phase 7] Goal satisfied — stopping early after %d combinations", runCount)
												goalDone = true
											}
										} else if tg > goalTuples[tupleKey] {
											prevTG := goalTuples[tupleKey]
											goalTuples[tupleKey] = tg
											env.Logger.Debugf("[Phase 7] Goal tuple ngl=%d ctk=%s ctv=%s nkvo=%d ctx=%d improved: %.2f→%.2f t/s",
												ngl, ctk, ctv, nkvo, ctxVal, prevTG, tg)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	if p.Goal != nil {
		if skipCount > 0 {
			env.Logger.Log("[Phase 7] Complete — %d combinations run, %d skipped via --focused, %d/%d distinct goal configs found",
				runCount, skipCount, goalHits, p.Goal.MaxHits)
		} else {
			env.Logger.Log("[Phase 7] Complete — %d combinations run, %d/%d distinct goal configs found",
				runCount, goalHits, p.Goal.MaxHits)
		}
	} else {
		if skipCount > 0 {
			env.Logger.Log("[Phase 7] Complete — %d combinations run, %d skipped via --focused", runCount, skipCount)
		} else {
			env.Logger.Log("[Phase 7] Complete — %d combinations run", runCount)
		}
	}
	return nil
}

// derivedMinNGL returns effective min-ngl for Phase 7.
func derivedMinNGL(env *PhaseEnv) *int {
	if env.Config.MinNGL != nil {
		return env.Config.MinNGL
	}
	step := env.Config.NGLStep
	min := env.MaxNGL - step
	if min < 0 {
		min = 0
	}
	env.Logger.Log("[Phase 7] Auto min-ngl=%d (max_ngl=%d − 1 step); override with --min-ngl", min, env.MaxNGL)
	return &min
}

func derivedMinThreads(env *PhaseEnv) *int {
	if env.Config.MinThreads != nil {
		return env.Config.MinThreads
	}
	if env.HW.CPUPhysical <= 1 {
		return nil
	}
	min := env.HW.CPUPhysical
	env.Logger.Log("[Phase 7] Auto min-threads=%d (physical cores); override with --min-threads", min)
	return &min
}

func derivedMinCtx(env *PhaseEnv, goal *GoalConfig) *int {
	if env.Config.MinCtx != nil {
		return env.Config.MinCtx
	}
	var min int
	if env.Config.StartCtx != nil {
		min = *env.Config.StartCtx
		env.Logger.Log("[Phase 7] Auto min-ctx=%d (inherited from --start-ctx); override with --min-ctx", min)
	} else {
		min = 8192
		env.Logger.Log("[Phase 7] Auto min-ctx=%d (default minimum; override with --min-ctx N or SWEEP_MIN_CTX=N)", min)
	}
	// Goal ctx acts as a floor
	if goal != nil && goal.CtxMin > min {
		min = goal.CtxMin
		env.Logger.Log("[Phase 7] Goal ctx≥%d applied as min-ctx floor", min)
	}
	return &min
}

func derivedMinB(env *PhaseEnv) *int {
	if env.Config.MinB != nil {
		return env.Config.MinB
	}
	if env.Best.B <= 0 {
		return nil
	}
	min := env.Best.B / 2
	if min < 512 {
		min = 512
	}
	env.Logger.Log("[Phase 7] Auto min-b=%d (best_b=%d / 2); override with --min-b", min, env.Best.B)
	return &min
}

func derivedMinCTK(env *PhaseEnv) string {
	if env.Config.MinCTK != "" {
		return env.Config.MinCTK
	}
	// TODO: would need to read sweep.jsonl for phase 6 OOM count
	// For now use q8_0 as default (bash behavior for timeout/clean)
	ctk := "q8_0"
	env.Logger.Log("[Phase 7] Auto min-ctk=%s (default); override with --min-ctk", ctk)
	return ctk
}

func filterBUBByMinB(bubs []state.BUBCombo, minB *int) []state.BUBCombo {
	if minB == nil {
		return bubs
	}
	var result []state.BUBCombo
	for _, v := range bubs {
		if v.B >= *minB {
			result = append(result, v)
		}
	}
	return result
}

func filterThreadsByMin(threads state.ThreadValues, minThreads *int) state.ThreadValues {
	if minThreads == nil {
		return threads
	}
	var result state.ThreadValues
	for _, v := range threads {
		if v == nil || *v >= *minThreads {
			result = append(result, v)
		}
	}
	return result
}

func sortIntsDesc(s []int) {
	for i := 0; i < len(s)-1; i++ {
		for j := i + 1; j < len(s); j++ {
			if s[j] > s[i] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
}


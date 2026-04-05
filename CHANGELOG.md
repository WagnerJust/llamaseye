# Changelog

All notable changes to llamaseye are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.2.0] — 2026-04-05

### Changed
- `--goal` mode now counts hits by distinct **(ngl, ctk, nkvo, ctx)** tuples instead of raw run count. Tuning variants (threads/b/ub) that differ only in performance knobs no longer inflate the hit count — each unique trade-off decision counts as one hit.
- Goal Results section in `sweep.md` deduplicated to one row per tuple (best TG wins), with the key trade-off axes leading the columns.

### Added
- `--goal-hits N` flag and `SWEEP_GOAL_HITS` env var to control how many distinct goal configs trigger early exit (default: 3, was previously hardcoded).
- `--goal-sort` flag and `SWEEP_GOAL_SORT` env var to control Goal Results table sort order: `tg` (default), `ctx`, `ngl`, `pp` — all descending, TG used as tiebreaker.

---

## [1.1.0] — 2026-04-05

### Added
- `--debug` flag and `SWEEP_DEBUG=true` env var: enables verbose `[DEBUG]` lines in the sweep log
  - Full `llama-bench` command line logged before each run
  - Raw stdout/stderr (up to 500 bytes) surfaced in `sweep.log` after each run
  - OOM regex match string logged when OOM is detected
  - Thermal polling results logged on every poll, not just when pausing
  - GGUF metadata fields (file size, heads, KV dims, hybrid attention layout) logged at sweep start
  - `gguf.Predict()` output (predicted NGL, context ceilings) logged when `--optimized-sweep` is active

---

## [1.0.1] — 2026-04-05

### Removed
- `llamaseye.sh` — the legacy Bash script has been deleted now that the Go binary is the sole entry point

### Changed
- `CLAUDE.md` updated to reflect Go package layout, build instructions, and key types

---

## [1.0.0] — 2026-04-05

Initial stable release of the Go rewrite.

### Added
- Full Go binary rewrite — single self-contained executable, no shell dependencies
- Eight-phase sweep pipeline (NGL probe → NGL axis → FA/KV quant → threads → KV offload → batch/ubatch → context ceiling → combination matrix)
- `--optimized-sweep`: parses GGUF metadata to auto-derive `--start-ngl` and `--start-ctx`
- `--goal` flag: targeted Phase 7 with ranked early-exit
- `--fine-ctx`: midpoint bisection for Phase 6 context ceiling
- `--report` flag: regenerate `sweep.md` from existing `sweep.jsonl` without re-running benchmarks
- `--resume`: skip already-completed phases via `state.json`
- `--only-phases` / `--skip-phases`: run or skip specific phase numbers
- `--model-list`: filter sweeps from a text file of model filenames
- `--env-file`: load environment variables from a custom path; auto-loads `.env` from CWD
- TurboQuant support: `turbo2`/`turbo3`/`turbo4` KV types via a second binary (`--turbo-bench`)
- Thermal guard: pauses sweep when CPU/GPU temperature exceeds configurable limits
- Per-run wall-time recording for timed-out context sizes
- Cross-model summary when sweeping multiple models
- `--version` flag

### Changed
- `LLAMA_BENCH_BIN`: no longer has a default path — must be set explicitly
- `SWEEP_OUTPUT_DIR`: default changed from `~/Models/bench/sweep` to `./results`

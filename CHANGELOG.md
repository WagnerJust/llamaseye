# Changelog

All notable changes to llamaseye are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.7.2] — 2026-04-09

### Added
- CI pipeline (`ci.yml`): runs `go vet` and `go test -race` on every push/PR, plus `golangci-lint` for static analysis.
- Release workflow now runs `go test` before building binaries.
- `CONTRIBUTING.md` with development setup, PR requirements, and code style guidance.
- `.github/ISSUE_TEMPLATE/` with bug report and feature request templates.
- `.github/pull_request_template.md` with doc-update checklist.
- `Makefile` with `build`, `test`, `vet`, `lint`, and `clean` targets.
- **Graceful shutdown on SIGINT/SIGTERM** — pressing Ctrl-C now cancels the in-flight benchmark, saves `state.json`, and exits cleanly. Use `--resume` to continue from where it stopped.
- Context propagation through the full call chain: `main.go` → `SweepModel` → `Phase.Run` → `RecordAndTrack` → `WaitCool` + `RunBench`. All `ctx.Done()` checks in phase loops are now functional.
- `RunBench` accepts a parent `context.Context`, allowing in-flight subprocesses to be cancelled on signal.
- `WaitCool` now respects the caller's context instead of using a detached `context.Background()`.

### Fixed
- Thermal monitor on Linux: shell pipeline commands (containing `|`, `>`, `<`) are now dispatched via `sh -c` instead of `strings.Fields` splitting. Previously, `sensors ... | awk ...` was passed as literal arguments to `sensors`, making the thermal guard a silent no-op on Linux AMD/Intel hardware.
- Sysfs fallback (`/sys/class/thermal/thermal_zone0/temp`) now divides by 1000 to convert millidegrees to degrees Celsius.
- GGUF parser: added upper-bound checks on key length (64 KiB), string length (1 MiB), and array length (1M elements) before allocating. A malformed `.gguf` file can no longer exhaust RAM via oversized `make()` calls.
- README.md: corrected stale `bash llamaseye.sh --report` reference to `./llamaseye --report`.
- `go.mod`: removed incorrect `// indirect` comment on `pflag` (direct dependency).
- `AppendRecord` errors are now logged as warnings instead of silently discarded — prevents silent data loss when disk is full or output dir becomes unwritable.
- `parseLlamaBenchOutput` errors are logged at debug level; empty results with zero exit code now returns `StatusError` instead of `StatusOK` with no throughput data.
- `bufio.Writer.Flush()` errors in `GenerateMarkdown` and `GenerateCrossModelSummary` are now captured via named return values instead of silently dropped by `defer w.Flush()`.

---

## [1.7.1] — 2026-04-09

### Fixed
- Deduplicated combo key logic between `output.ComboKey` and the removed `phase.FocusedComboKey` to prevent drift.
- Used `errors.Is(err, os.ErrNotExist)` instead of `os.IsNotExist` for robust missing-file detection in `--focused` mode.
- Phase 7 now logs skipped combo counts separately from executed run counts when `--focused` is active.

---

## [1.7.0] — 2026-04-09

### Added
- **Focused phase re-run** (`--focused`) — when combined with `--only-phases`, each phase diffs its planned combos against existing `status: "ok"` entries in `sweep.jsonl` and only runs combos not yet present. Skipped combos still populate working sets so downstream phases see the full picture. Env var: `SWEEP_FOCUSED` (default: `false`). Closes #33.

---

## [1.6.0] — 2026-04-08

### Added
- **CTV filter axis** — Phase 2 now supports independent V-cache type filtering via three new flags:
  - `--ctv <list>` (env: `SWEEP_CTV`) — comma-separated explicit CTV values (e.g. `turbo3,turbo2`); takes precedence over the other two flags
  - `--start-ctv <type>` (env: `SWEEP_START_CTV`) — begin V-cache sweep at this type, same quality ordering as CTK
  - `--ctv-dir up|down` (env: `SWEEP_CTV_DIR`, default: `up`) — V-cache sweep direction
  Mirrors the existing `--start-ctk` / `--ctk-dir` pattern. Closes #32.

---

## [1.5.0] — 2026-04-09

### Added
- **Independent CTK/CTV working sets** — Phase 2 now populates `WS.CTKValues` and `WS.CTVValues` as independent axes alongside the existing paired `WS.FACTK`. Phase 7 uses the cartesian product `CTK × CTV` instead of replaying only the exact pairs tested in Phase 2.
- **Precision filter in Phase 7** — `(ctk, ctv)` combos where V is more precise than K are skipped as wasteful. Filter rule: `CTK_quality >= CTV_quality`. Quality ordering: `f16 > q8_0 > q4_0 > iso4 > planar4 > turbo4 > iso3 > planar3 > turbo3 > turbo2`.
- **RotorQuant combos in Phase 2** — `iso3`/`iso4`/`planar3`/`planar4` symmetric and asymmetric combos added when `--rotor-bench` is available.
- **State migration for `--resume`** — old `state.json` files without `ctk_values`/`ctv_values` are automatically migrated by deriving them from `fa_ctk_combos`.
- `KVPrecisionValid`, `BestFAForCTK`, `UniqueCTKValues`, `UniqueCTVValues` helpers in `phase/common.go`.
- Updated `CTKQualityOrder` in `phase/common.go` to include rotor types; Phase 6 fallback now uses this shared ordering.

### Changed
- Phase 7 log line updated: `fa_ctk×N` → `kv_pairs×N` (reflecting the new CTK × CTV cartesian count).
- Phase 7 goal tuple key now includes CTV: `(ngl, ctk, ctv, nkvo, ctx)`.
- State schema: `working_sets` gains `"ctk_values"` and `"ctv_values"` string arrays (backward-compatible — old files without these fields are migrated).

---

## [1.4.0] — 2026-04-09

### Added
- **V-first OOM fallback in Phase 6** — when the primary config OOMs at a context size, the fallback sequence now first tries more-compressed `ctv` types while keeping `ctk` fixed (exhaust V before touching K), then falls through to more-compressed `ctk+ctv` pairs. Aligns with TurboQuant research that V compression is effectively quality-free.
- **RotorQuant binary dispatch** (`--rotor-bench <path>` / `SWEEP_ROTOR_BENCH_BIN`) — enables `planar3`, `planar4`, `iso3`, `iso4` KV cache types from the [johndpope/llama-cpp-turboquant](https://github.com/johndpope/llama-cpp-turboquant) fork (branch `feature/planarquant-kv-cache`). RotorQuant types slot into the quality ordering between `q4_0` and TurboQuant types.
- **Unified KV quality ordering** — Phase 6 fallback now uses the full ordering: `f16 > q8_0 > q4_0 > iso4 > planar4 > turbo4 > iso3 > planar3 > turbo3 > turbo2`.
- `FindFACTKByKV` helper in `phase/common.go` to look up the best FA value for an exact (ctk, ctv) pair.

---

## [1.3.0] — 2026-04-08

### Fixed
- `.env` values containing `$VAR` or `${VAR}` references are now expanded against the process environment (`envfile/envfile.go`). Single-quoted values remain literal, matching standard shell behaviour. Previously, paths like `SWEEP_OUTPUT_DIR=${HOME}/Models/bench/sweep` were stored literally, causing a `${HOME}` directory to be created in the working directory.

### Added
- **Asymmetric K/V quant combos in Phase 2** — when `--turbo-bench` is available, Phase 2 now tests five asymmetric `(ctk, ctv)` pairs (e.g. `ctk=q8_0, ctv=turbo3`) in addition to the existing symmetric combos. TurboQuant research shows V cache compression is effectively free; these combos capture high K precision with aggressive V compression.
- `--asymmetric-kv` / `--no-asymmetric-kv` flag (`SWEEP_ASYMMETRIC_KV`, default: `true`) to opt out of asymmetric combos when needed.

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

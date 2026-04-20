# Changelog

All notable changes to llamaseye are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.8.1] — 2026-04-29

### Added
- AMD GPU detection on Linux via `rocm-smi`: populates GPU model, VRAM total/free, and GPU temperature monitoring. `hardware.json` now records `backend = "rocm"` for AMD systems. Requires ROCm userspace tools (`rocm-smi`) to be installed; falls back to `cpu` if absent.

---

## [1.8.0] — 2026-04-29

### Added
- `llamaseye install-skill` subcommand: writes the embedded operational skill to `~/.claude/skills/llamaseye/SKILL.md` and/or `~/.agents/skills/llamaseye/SKILL.md` on demand. Defaults to dry-run + both targets + `$HOME` scope. Flags: `--target {claude,agents}`, `--local`, `--apply`, `--force`, `--list`.
- New top-level `skills/` Go package containing `llamaseye.md` — the canonical operational guide (plain markdown with YAML frontmatter), embedded into the binary via `go:embed`.
- New `internal/skill` package with the install-target registry and the `Install` function, covered by `internal/skill/skill_test.go` (dry-run, apply, target selection, force/no-force, unknown target).
- `codecov/codecov-action@v5` step in `ci.yml` uploads `coverage.out` to Codecov on every PR, making the existing `codecov.yml` load-bearing (per-PR coverage diffs in PR comments).
- `README.md` gains a "For coding-agent users" section explaining `llamaseye install-skill --apply`.

### Changed
- Doc-update rule in `AGENTS.md`, `CONTRIBUTING.md`, and `.github/pull_request_template.md` now points at `skills/llamaseye.md` (the canonical embedded source). PRs that change behaviour update the canonical file; the binary picks up the change automatically.
- `CLAUDE.md` rewritten: agent-tool skill folders (`.claude/skills/`, `.agents/skills/`) are user-local and gitignored; the source of truth is `skills/llamaseye.md`.

### Removed
- `.agents/skills/` directory (both `llamaseye/SKILL.md` and `build/SKILL.md`) introduced in 1.7.7. The build cheatsheet was redundant with AGENTS.md's "Tech Stack" section; the operational skill moved into `skills/` and is distributed via the new install-skill subcommand.
- `.claude/skills/` no longer allow-listed in `.gitignore`; tool-specific skill folders are user-local across the board.

### Why
1.7.7 promoted the operational skill into the repo at `.agents/skills/llamaseye/SKILL.md` to make the Doc-Update Rule honest. Reviewing for public-OSS readiness surfaced two follow-ups: the previously-tracked `.claude/skills/llamaseye/SKILL.md` leaked private hostnames into a public repo, and `codecov.yml` was scaffolded by plumbline but never wired to upload anything. This release fixes both, and goes one step further: instead of shipping the skill as a static checked-in file that contributors and end users have to find, the binary itself ships and installs it (same pattern as `plumbline install-skill --apply`). End users who `go install github.com/WagnerJust/llamaseye@latest` get the operational guide for their coding agent with one command, without cloning.

---

## [1.7.7] — 2026-04-29

### Added
- Canonical agent instructions in `AGENTS.md` (single source of truth for AI agents in this repo). `CLAUDE.md`, `.cursorrules`, `.claudeignore`, `.github/copilot-instructions.md`, and `.gitmessage` point to or extend this canonical doc.
- Cross-agent skills directory `.agents/skills/` for shared agent workflows.
- `docs/llamaseye-vs-llama-benchy.md` — comparison/positioning doc against the sister tool.
- CI now enforces a 60% coverage floor on every push/PR (`go test -coverprofile` + threshold check). Existing `lint` and new `go build ./...` step both gate merges.
- New scheduled workflows: `coverage.yml` (weekly full coverage HTML artifact, Sun 07:00 UTC), `nightly-compliance.yml` (nightly `govulncheck`), `flaky-analysis.yml` (weekly `go test -count=10 -race` to surface intermittents, Mon 08:00 UTC).
- `codecov.yml` declaring the 60% coverage target for project + patch.
- `.github/ISSUE_TEMPLATE/feedback.md` for user feedback on tool behavior and output.

### Changed
- `AGENTS.md` Rules of Engagement: implementation plans and design notes go under `.local/plans/` or `.local/specs/` (gitignored), not `docs/`. `docs/spec.md` remains the canonical committed engineering doc.

### Why
Brings the repo to ACMM Level 3 readiness for AI-assisted development: a single canonical source of agent instructions, PR-time build/lint/coverage gates, scheduled compliance + coverage drift detection, and an explicit feedback channel. Codifies that ephemeral planning artifacts stay out of git so they don't accumulate in `docs/`.

---

## [1.7.6] — 2026-04-11

### Fixed
- Phase 6 fallback Parts 1 and 2 now use `effectiveCTK`/`effectiveCTV` (the `--p6-ctk`/`--p6-ctv` override values) instead of `env.Best.CTK`/`env.Best.CTV`. Previously, when `--p6-ctk q8_0 --p6-ctv turbo2` was used and the primary config OOMed, the nkvo-flip fallback silently reverted to `f16/f16` — causing `q8_0/turbo2/nkvo=0` to never be tested.

---

## [1.7.5] — 2026-04-11

### Added
- `--p6-ctk` / `SWEEP_P6_CTK` and `--p6-ctv` / `SWEEP_P6_CTV` flags — override the CTK/CTV Phase 6 uses as its starting config, independent of what prior phases determined as best. Useful for re-running the context ceiling sweep with a specific quant type (e.g. `--p6-ctk q8_0 --p6-ctv turbo2`) without re-running earlier phases or patching `state.json`. Unknown type strings warn and fall back to best.

---

## [1.7.4] — 2026-04-11

### Changed
- `--rotor-bench` flag and `SWEEP_ROTOR_BENCH_BIN` env var now marked as **experimental / currently broken** in all user-facing docs (flag help, `example.env`, `README.md`, `docs/spec.md`). The flag still exists and the code path is unchanged; it is simply documented as non-functional until the issue is resolved.

---

## [1.7.3] — 2026-04-09

### Fixed
- `validateBenchBinary` now falls back to scanning the binary's string table when `--help` output does not enumerate the marker string. This fixes turbo-bench detection for builds of `turbo-llama-bench` that compile in `turbo3`/`turbo2`/`turbo4` KV cache support but do not list valid cache type names in their help text — such builds were previously silently treated as unavailable, causing all turbo KV runs to be skipped.

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
- Fixed unchecked `fmt.Sscanf` and `fmt.Scan` return values flagged by `errcheck` linter.
- Simplified `jsonlParamsJSON` struct literal to direct type conversion (`gosimple S1016`).
- `parseOptInt` now returns an error on invalid integer values (e.g., `--start-ngl abc`) instead of silently treating them as unset. The sweep fails fast with a clear error message.
- `detectTurbo` renamed to `validateBenchBinary` — now actually runs `<binary> --help` and checks for the expected marker string (`turbo3` for TurboQuant, `planar3` for RotorQuant) instead of only checking that the file exists with the execute bit. A standard llama-bench at the wrong path will no longer be silently accepted.
- Eliminated double `os.Stat` call (TOCTOU race) in binary validation.
- `SweepModel` no longer mutates `s.Logger` — per-model logger is now a local variable, preventing shared state corruption across sequential model sweeps and eliminating a potential data race if models were ever swept concurrently.
- `state.Load` now uses `errors.Is(err, os.ErrNotExist)` instead of deprecated `os.IsNotExist`.
- Goal spec parsing deduplicated — `config.ParseGoalSpec` is the single source of truth, replacing duplicate logic in `sweep/orchestrator.go` and `output/markdown.go`.
- Inline anonymous `Logger` interface in `bench/runner.go` replaced with named `DebugLogger` interface for discoverability.
- Fixed remaining unchecked `fmt.Sscanf` return values in `config/config.go` and `cmd/root_test.go` flagged by `errcheck` linter.
- Removed empty `if err != nil` branch in `validateBenchBinary` (SA9003 staticcheck).

### Changed
- `sweep.jsonl` file handle is now opened once per model sweep and reused for all records, instead of opening and closing the file for every benchmark run. Reduces syscall overhead from O(n) opens to O(1) during Phase 7's combination matrix.
- Thread working set replaced from `[]any` (requiring type switches at every consumer) with typed `state.ThreadValues` (`[]*int`), where `nil` means "system_default". JSON format is backward-compatible — existing `state.json` files with `"system_default"` strings unmarshal correctly.

### Removed
- Dead code cleanup: removed `ParseThreadValues`, `ThreadValuesToAny`, `maxFloat`, `formatError`, `containsStr`, `binaryLabel` suppression, unused `axis` parameter from `ApplyPhase7MinsInt`, unused second parameter from `printHardwareSummary`, and `cmd.ParsePhaseList` re-export.
- Removed suppressed viability computation (`MinTGTS` check that was computed and discarded with `_ = v`).

### Tests
- Fixed inverted assertion in `TestRunBench_TurboUnavailable` — test was passing vacuously due to `&&` condition that was always false.
- Added actual assertion on execution count in `TestP7CombinationMatrix_GoalEarlyExit` — goal early-exit behavior was previously untested.
- `cmd/root_test.go` with tests for `parseOptInt`, `ResolveModels`, `resolveModelsDir`, and `resolveModelList`.
- `hardware/detect_test.go` with smoke test for `Detect()` asserting positive CPU/RAM values.

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

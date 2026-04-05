# Changelog

All notable changes to llamaseye are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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

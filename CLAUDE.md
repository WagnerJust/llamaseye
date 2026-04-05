# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**llamaseye** is a Go binary that exhaustively benchmarks every meaningful llama-bench parameter combination for GGUF models. Build with `go build .` or `make build`.

## Building and running

```bash
# Build
go build -o llamaseye .

# Single model
./llamaseye --model ~/Models/Qwen3-14B-Q4_K_M.gguf --output-dir ./results

# All models in a directory
./llamaseye --models-dir ~/Models --output-dir ./results

# From a model list file (one filename per line, # comments supported)
./llamaseye --models-dir ~/Models --model-list my_models.txt --output-dir ./results

# Auto-derive start-ngl and start-ctx from GGUF metadata
./llamaseye --model ~/Models/model.gguf --optimized-sweep

# Resume an interrupted sweep
./llamaseye --model ~/Models/model.gguf --resume

# Run only specific phases (e.g. re-run context and combo matrix)
./llamaseye --model ~/Models/model.gguf --only-phases 6,7

# Dry run (print commands without executing)
./llamaseye --model ~/Models/model.gguf --dry-run
```

**Environment / .env file:**
```bash
cp example.env .env
# Edit .env, then:
./llamaseye --models-dir ~/Models
```
`.env` is auto-loaded from the working directory if present. `.env` is gitignored. `example.env` documents every available variable with defaults.

## Package layout

```
llamaseye/
  main.go          # Entry point: env loading, hardware detection, model loop
  cmd/root.go      # CLI flag definitions (pflag), env/CLI merge, model resolution
  config/          # Config struct, defaults, validation
  hardware/        # HardwareInfo detection (darwin/linux), thermal polling
  gguf/            # Pure-Go GGUF metadata reader + NGL/context ceiling prediction
  bench/           # BenchRunner, OOM detection, binary selection (turbo vs standard)
  phase/           # One file per phase (p0–p7), shared PhaseEnv state
  sweep/           # Sweeper orchestrator: SweepModel(), ReportMode()
  output/          # Logger, JSONL append, Markdown summary generation
  state/           # state.json serialization for --resume
  envfile/         # .env file loader
```

## Architecture

### Sweep phases (0–7)

Each phase sweeps **exactly one axis** while holding all other parameters at fixed defaults. The only cross-phase dependency is `MAX_NGL` (discovered by Phase 0), which becomes the ceiling for all subsequent NGL values.

| Phase | Name | Axis swept |
|-------|------|------------|
| 0 | NGL probe | Binary search for max stable GPU layers |
| 1 | NGL axis | GPU layer count 0 → `MAX_NGL` |
| 2 | FA + KV quant | Flash attention × KV cache type (f16, q8_0, q4_0, turbo2–4) |
| 3 | Thread count | CPU threads 1 → `HW_CPU_LOGICAL` |
| 4 | KV offload | KV cache in VRAM vs RAM (`nkvo`) |
| 5 | Batch/ubatch | Batch and micro-batch size pairs |
| 6 | Context ceiling | Prompt size 128 → 131072, stops at OOM; on OOM auto-retries with nkvo flip then more-compressed ctk types |
| 7 | Combo matrix | Cartesian product of all working values from phases 1–6 |

Phases 1–6 each populate a working set in `PhaseEnv`. Phase 7 takes the cartesian product of all of them.

Phase 6 fallback order on OOM: (1) flip `nkvo`, (2) try progressively more-compressed `ctk` types × both `nkvo` values. Only ctk/nkvo values already validated by Phases 2 and 4 are tried as fallbacks.

### Key packages and types

- `hardware.Detect()` — probes CPU cores, RAM, VRAM, backend (`cuda`/`metal`/`cpu`), thermal sensors. Returns `*HardwareInfo`.
- `gguf.Predict()` — parses GGUF metadata to predict max NGL and best context ceiling when `--optimized-sweep` is active.
- `bench.BenchRunner` — core invocation: calls `timeout + llama-bench`, captures output, appends JSONL to `sweep.jsonl`, detects OOM/timeout.
- `bench.DetectOOM()` — regex matches output for OOM strings ("CUDA out of memory", "failed to allocate", etc.).
- `state.Save()` / `state.Load()` — serialize/deserialize `state.json` for `--resume`.
- `hardware.ThermalMonitor` — polls CPU/GPU temperature; pauses the sweep if thermal limits are exceeded.
- `phase.P0` … `phase.P7` — one struct per phase implementing the `Phase` interface.

### Output files (per model)

```
results/<model-stem>/
├── sweep.jsonl    # Append-only, one JSON record per run (source of truth)
├── sweep.md       # Human-readable Markdown summary table per phase
├── sweep.log      # Full timestamped execution log
├── hardware.json  # Hardware snapshot from hardware.Detect()
├── state.json     # Resume state: completed phases + best values + working sets
└── raw/<run-id>.txt  # Raw llama-bench stdout per run
```

### TurboQuant binary

When `--turbo-bench <path>` is passed, runs using `turbo2`/`turbo3`/`turbo4` KV types are dispatched to that binary; all other runs use the standard binary. The turbo binary is built from the `feature/turboquant-kv-cache` branch of `github.com/TheTom/llama-cpp-turboquant`. Verified at startup by checking `--help` output contains "turbo3". Invalid path = silently disabled.

### Flag/env var convention

Every CLI flag has a matching `SWEEP_*` environment variable. CLI flags always override env vars. Defaults are set in `config.Defaults()` using `os.Getenv`.

## Documentation update rule

**Every PR that changes behaviour must also update:**
1. `README.md` — user-facing description of affected phases, flags, or output
2. `docs/spec.md` — engineering spec (JSONL schema, phase behaviour, output format)
3. `.claude/skills/llamaseye/SKILL.md` — skill doc used by the Claude Code agent

Update all three in the same branch/PR as the code change. Do not merge a code PR without the doc updates.

## Changelog and versioning rule

**Every commit must include a `CHANGELOG.md` entry with a semver bump.** Follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and [Semantic Versioning](https://semver.org/):

- **Patch** (`x.y.Z`) — removals, cleanup, docs, bug fixes
- **Minor** (`x.Y.0`) — new features, new flags, new phases
- **Major** (`X.0.0`) — breaking changes to CLI flags, output format, or JSONL schema

No commit should land on `main` without a corresponding version entry.

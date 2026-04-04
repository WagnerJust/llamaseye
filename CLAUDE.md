# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**llamaseye** is a single-file Bash script (`llamaseye.sh`) that exhaustively benchmarks every meaningful llama-bench parameter combination for GGUF models. There is no build step — the entry point is the script itself.

## Running the script

```bash
# Single model
bash llamaseye.sh --model ~/Models/Qwen3-14B-Q4_K_M.gguf --output-dir ./results

# All models in a directory
bash llamaseye.sh --models-dir ~/Models --output-dir ./results

# From a model list file (one filename per line, # comments supported)
bash llamaseye.sh --models-dir ~/Models --model-list my_models.txt --output-dir ./results

# Auto-derive start-ngl and start-ctx from GGUF metadata (requires python3)
bash llamaseye.sh --model ~/Models/model.gguf --optimized-sweep

# Resume an interrupted sweep
bash llamaseye.sh --model ~/Models/model.gguf --resume

# Run only specific phases (e.g. re-run context and combo matrix)
bash llamaseye.sh --model ~/Models/model.gguf --only-phases 6,7

# Dry run (print commands without executing)
bash llamaseye.sh --model ~/Models/model.gguf --dry-run
```

**Environment / .env file:**
```bash
cp example.env .env
# Edit .env, then:
source .env && bash llamaseye.sh --models-dir ~/Models
```
`.env` is gitignored. `example.env` documents every available variable with defaults.

## Architecture

The script is structured in three logical sections at the top of the file:

1. **Configuration variables** — all `SWEEP_*` / `LLAMA_BENCH_BIN` defaults, each overridable by env var or CLI flag
2. **Runtime state** — `HW_*` hardware inventory, `BEST_*` current best-per-axis values, `WS_*` per-phase working sets
3. **Functions** — all logic is in functions; `main()` at the bottom orchestrates everything

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

Phases 1–6 each populate a `WS_*` working set variable. Phase 7 takes the cartesian product of all of them.

Phase 6 fallback order on OOM: (1) flip `nkvo`, (2) try progressively more-compressed `ctk` types × both `nkvo` values. Only ctk/nkvo values already validated by Phases 2 and 4 are tried as fallbacks.

### Key functions to know

- `detect_hardware()` — probes CPU cores, RAM, VRAM, backend (`cuda`/`metal`/`cpu`), thermal sensors. Sets all `HW_*` vars. Runs once per model.
- `analyze_model()` — when `--optimized-sweep` is active, parses GGUF metadata via an inline Python3 script to predict max NGL and best context ceiling, then sets `OPT_START_NGL` and `OPT_START_CTX`. Architecture-agnostic via `general.architecture`. Cannot be combined with any `--start-*` or `--min-*` flag.
- `detect_turbo_binary()` — validates the optional TurboQuant llama-bench binary.
- `run_bench()` — the core invocation: calls `timeout + llama-bench`, captures output, appends JSONL to `sweep.jsonl`, detects OOM/timeout.
- `detect_oom()` — pattern matches stderr for OOM strings ("CUDA out of memory", "failed to allocate", etc.).
- `save_state()` / `load_state()` — serialize/deserialize `state.json` for `--resume`.
- `wait_cool()` — polls CPU/GPU temperature; pauses the sweep if thermal limits are exceeded.
- `phase_N_*()` — one function per phase, e.g. `phase_1_ngl_sweep()`, `phase_7_combination_matrix()`.

### Output files (per model)

```
results/<model-stem>/
├── sweep.jsonl    # Append-only, one JSON record per run (source of truth)
├── sweep.md       # Human-readable Markdown summary table per phase
├── sweep.log      # Full timestamped execution log
├── hardware.json  # Hardware snapshot from detect_hardware()
├── state.json     # Resume state: completed phases + best values + working sets
└── raw/<run-id>.txt  # Raw llama-bench stdout per run
```

### TurboQuant binary

When `--turbo-bench <path>` is passed, runs using `turbo2`/`turbo3`/`turbo4` KV types are dispatched to that binary; all other runs use the standard binary. The turbo binary is built from the `feature/turboquant-kv-cache` branch of `github.com/TheTom/llama-cpp-turboquant`. Verified at startup by checking `--help` output contains "turbo3". Invalid path = silently disabled.

### Flag/env var convention

Every CLI flag has a matching `SWEEP_*` environment variable. CLI flags always override env vars. The pattern throughout the script:

```bash
OPT_RESUME="${SWEEP_RESUME:-false}"   # env var sets default; CLI overrides
```

### Bash patterns to be aware of

- `set -euo pipefail` is active — all subshell errors propagate
- `(( expr )) || true` is used for arithmetic that might evaluate to zero (which would cause exit under `set -e`)
- Pure-bash array builders are used in `save_state()` instead of jq pipelines (a previous bug source — jq arrays from bash loops had quoting issues)
- Phase functions append to working sets with `WS_NGL+=" $val"` (space-separated strings, not arrays) to survive subshell boundaries
- OOM detection uses `detect_oom()` called on the raw output file, not stderr trapping, because `timeout` complicates signal propagation

## Documentation update rule

**Every PR that changes behaviour in `llamaseye.sh` must also update:**
1. `README.md` — user-facing description of affected phases, flags, or output
2. `docs/spec.md` — engineering spec (JSONL schema, phase behaviour, output format)
3. `skills/llamaseye/SKILL.md` — skill doc used by the Claude Code agent

Update all three in the same branch/PR as the code change. Do not merge a code PR without the doc updates.

# llamaseye — Exhaustive llama-bench parameter sweep harness

Systematically sweep every meaningful llama-bench parameter combination for any GGUF model, record every result as JSONL, and surface the optimal configuration for your hardware.

---

## What it does

**llamaseye** runs llama-bench across every meaningful parameter combination for any GGUF model. It sweeps each axis independently — GPU layer offload (ngl), flash attention, KV cache quantisation type, thread count, KV offload ratio, batch size, and context size — then runs a full combination matrix (Phase 7) to confirm which configs work together and find the true performance ceiling.

Every result is recorded as JSONL in a per-model output directory, alongside a human-readable Markdown summary, a raw log, a hardware snapshot, and a resume-state file. Runs that trigger an OOM or timeout are caught and logged — the sweep never hangs. OOM and timeout are distinguished: OOM means a context size is impossible at that memory budget; timeout means it is achievable but slow. Timeout runs write a `"status": "timeout"` record with `wall_time_sec` to `sweep.jsonl` and appear in a dedicated section of `sweep.md`.

The script is fully portable: it detects CPU core count, available RAM, GPU VRAM, the active compute backend (cuda / metal / cpu), and the correct thermal-sensor commands at runtime. There are no hardcoded machine values. Optionally, pass a TurboQuant build of llama-bench via `--turbo-bench` to unlock turbo2/turbo3/turbo4 KV cache types from the llama-cpp-turboquant fork, which compress the KV cache 3–6× and enable much longer contexts on the same hardware.

---

## Quick start

**Build the binary (Go 1.22+):**
```bash
go build -o llamaseye .
```

**Single model:**
```bash
./llamaseye --model ~/Models/Qwen3-14B-Q4_K_M.gguf --output-dir ./results
```

**All models in a directory:**
```bash
./llamaseye --models-dir ~/Models --output-dir ./results
```

**From a model list file:**
```bash
./llamaseye --models-dir ~/Models --model-list my_models.txt --output-dir ./results
```

**With TurboQuant KV types:**
```bash
./llamaseye --model ~/Models/model.gguf --turbo-bench ~/llama-cpp-turboquant/build/bin/llama-bench
```

**Unattended overnight run:**
```bash
nohup ./llamaseye --models-dir ~/Models --output-dir ./results > /dev/null 2>&1 &
```


### Using a .env file

All environment variables can be set in a `.env` file instead of passing flags every time. The binary **auto-loads `.env` from the working directory** if it exists — no `source` step needed:

```sh
cp example.env .env
# Edit .env to match your paths and preferences
./llamaseye --models-dir ~/Models   # .env is loaded automatically
```

To load a file at a different path use `--env-file`:

```sh
./llamaseye --env-file ~/my-config.env --model ~/Models/model.gguf
```

**Load order:** `.env` file → process environment → CLI flags. Process env vars always override file values; CLI flags override everything.

Every CLI flag has a corresponding environment variable — env vars set the default value, and CLI flags override them when both are provided. `example.env` in the repo root documents every available variable with its default value and a description. The most important ones to set are `LLAMA_BENCH_BIN` (path to your llama-bench binary) and `SWEEP_OUTPUT_DIR` (where results are written).

`.env` is gitignored — your local paths and configuration will not be committed.

---

## Building

llamaseye is a single Go binary with no runtime dependencies beyond the OS. Requires Go 1.22 or later.

```bash
# Build
go build -o llamaseye .

# Optionally install into your PATH
go install github.com/WagnerJust/llamaseye@latest
```

The binary statically links all dependencies. No external tools are required at runtime except `llama-bench` itself.

---

## Dependencies

### llama-bench (required)

llamaseye does not install or build llama-bench for you. You must build it yourself from [llama.cpp](https://github.com/ggml-org/llama.cpp) with whatever backend flags suit your hardware **before** running llamaseye.

```sh
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build for CUDA (NVIDIA GPU)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench -j$(nproc)

# Build for Metal (macOS — Apple Silicon or Intel Mac)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench -j$(sysctl -n hw.logicalcpu)

# Build CPU-only (any platform, no GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench -j$(nproc)
```

The binary will be at `build/bin/llama-bench`. Pass its path to llamaseye via `--llama-bench <path>` or set the `LLAMA_BENCH_BIN` environment variable. There is no default — llamaseye will exit with an error if the binary is not specified.

> The build flags you choose determine which backends and features are available during the sweep. llamaseye works with any valid llama-bench binary — it does not require any specific build flags itself.

### Optional: TurboQuant llama-bench

To enable `turbo2`/`turbo3`/`turbo4` KV cache types, build a second binary from the [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork (branch `feature/turboquant-kv-cache`) and pass it via `--turbo-bench <path>`. The fork is otherwise identical to llama.cpp — same build flags apply.

```sh
git clone https://github.com/TheTom/llama-cpp-turboquant \
  --branch feature/turboquant-kv-cache --depth=1
cd llama-cpp-turboquant
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench -j$(nproc)
```

llamaseye verifies the binary at startup by probing it with `-ctk turbo3`. If the flag is accepted, turbo types are enabled. If the path is missing or the flag is rejected, turbo types are silently omitted and the sweep continues with the standard KV type set. It is safe to always pass `--turbo-bench` — the script handles an invalid path gracefully.

### Optional system tools

The Go binary uses native OS APIs for hardware detection and thermal monitoring. A few optional tools extend these capabilities:

| Tool | Purpose | Install |
|------|---------|---------|
| `nvidia-smi` | NVIDIA GPU VRAM/temp detection | Included with NVIDIA drivers |
| `sensors` | Linux CPU temperature reading | `apt install lm-sensors` |
| `osx-cpu-temp` | macOS CPU temperature reading (optional) | `brew install osx-cpu-temp` |

If these are absent, llamaseye disables the corresponding thermal guard and logs a warning — the sweep still runs.

---

## Key flags

### Core

| Flag | Description |
|------|-------------|
| `--model <path>` | Single GGUF model to benchmark |
| `--models-dir <dir>` | Directory to scan for GGUF models |
| `--model-list <file>` | Text file listing model filenames (one per line) |
| `--output-dir <dir>` | Root directory for all results (default: `./results`) |
| `--llama-bench <path>` | Path to standard llama-bench binary |
| `--turbo-bench <path>` | Path to TurboQuant llama-bench binary (enables turbo2/3/4 KV types) |
| `--ngl-step <n>` | Step size for NGL axis sweep (default: 4) |
| `--repetitions <n>` | Repetitions per benchmark run (default: 3) |
| `--timeout <s>` | Per-run timeout in seconds (default: 600) |
| `--goal <spec>` | Goal-directed Phase 7: stop after 3 validated configs meeting the spec. Format: `ctx=N,tg=N,pp=N` (all optional). Example: `--goal "ctx=32768,tg=5"` |
| `--resume` | Resume a previous sweep, skipping completed phases |
| `--overwrite` | Delete existing output dir and re-run everything |
| `--only-phases <list>` | Comma-separated list of phase numbers to run (e.g. `0,1,7`) |
| `--skip-phases <list>` | Comma-separated list of phase numbers to skip |
| `--report` | Read-only: regenerate `sweep.md` from existing `sweep.jsonl` files without running any benchmarks. Also generates `summary.md` when multiple models are found. Combine with `--model`/`--models-dir` to target a subset; omit both to scan all subdirs of `--output-dir`. |
| `--dry-run` | Print what would run without executing |
| `--no-confirm` | Skip the pre-run confirmation prompt |
| `--debug` | Enable verbose `[DEBUG]` lines in the log: full command lines, raw stdout/stderr, OOM matches, thermal polls, GGUF metadata |
| `--goal-hits N` | Stop goal mode after N distinct (ngl, ctk, nkvo, ctx) configs are found (default: 3) |
| `--cpu-temp-limit <°C>` | Pause if CPU exceeds this temperature (default: 88) |
| `--gpu-temp-limit <°C>` | Pause if GPU exceeds this temperature (default: 81) |
| `--no-thermal-guard` | Disable thermal polling entirely |

### Axis start and direction

These control where each phase begins its sweep and which direction it moves. Direction flags accept `up` or `down`.

| Flag | Description |
|------|-------------|
| `--start-ngl <n>` | Begin NGL sweep at this value (default: `MAX_NGL − 2×step`) |
| `--ngl-dir up\|down` | NGL sweep direction (default: `up` = toward MAX_NGL) |
| `--start-threads <n>` | Begin thread count sweep at this value |
| `--threads-dir up\|down` | Thread sweep direction (default: `up`) |
| `--start-ctx <n>` | Begin context sweep at this prompt size; also sets Phase 7 min-ctx |
| `--ctx-dir up\|down` | Context sweep direction (default: `up` = toward 131072) |
| `--fine-ctx` | Enable midpoint bisection in Phase 6 (see [Fine-grained context sweep](#fine-grained-context-sweep)) |
| `--ctx-step-min <n>` | Minimum bisection step for `--fine-ctx` (default: `8192`) |
| `--start-ctk <type>` | Begin KV quant sweep at this type |
| `--ctk-dir up\|down` | KV type sweep direction (default: `up` = toward more compression) |
| `--start-b <n>` | Begin batch size sweep at this value |
| `--b-dir up\|down` | Batch sweep direction (default: `up`) |
| `--start-ub <n>` | Begin ubatch size sweep at this value |
| `--ub-dir up\|down` | Ubatch sweep direction (default: `up`) |
| `--start-fa 0\|1` | Begin FA sweep at this value (default: `0`) |
| `--fa-dir up\|down` | FA sweep direction (default: `up` = 0→1) |

### Phase 7 minimum filters

These filter the Phase 7 combination matrix without affecting phases 1–6. When not set, smart defaults are derived automatically (see [Smart defaults](#smart-defaults)).

| Flag | Description |
|------|-------------|
| `--min-ngl <n>` | Exclude NGL values below N from Phase 7 |
| `--min-threads <n>` | Exclude thread counts below N from Phase 7 |
| `--min-ctx <n>` | Exclude context sizes below N from Phase 7 |
| `--min-ctk <type>` | Exclude KV types below TYPE (by quality) from Phase 7 |
| `--min-b <n>` | Exclude batch sizes below N from Phase 7 |
| `--min-ub <n>` | Exclude ubatch sizes below N from Phase 7 |

---

## Environment variables

Every CLI flag can also be set via environment variable — useful for `.env` files so you don't repeat flags on every invocation. CLI flags always override env vars when both are set. See `example.env` for the full list with defaults and descriptions.

| Variable | Equivalent flag | Example |
|----------|----------------|---------|
| `SWEEP_RESUME` | `--resume` | `SWEEP_RESUME=true` |
| `SWEEP_OVERWRITE` | `--overwrite` | `SWEEP_OVERWRITE=true` |
| `SWEEP_SKIP_PHASES` | `--skip-phases` | `SWEEP_SKIP_PHASES=7` |
| `SWEEP_ONLY_PHASES` | `--only-phases` | `SWEEP_ONLY_PHASES=0,1,6` |
| `SWEEP_NGL_STEP` | `--ngl-step` | `SWEEP_NGL_STEP=2` |
| `SWEEP_START_NGL` | `--start-ngl` | `SWEEP_START_NGL=40` |
| `SWEEP_NGL_DIR` | `--ngl-dir` | `SWEEP_NGL_DIR=down` |
| `SWEEP_START_CTX` | `--start-ctx` | `SWEEP_START_CTX=32768` |
| `SWEEP_MIN_CTX` | `--min-ctx` | `SWEEP_MIN_CTX=32768` |
| `SWEEP_MIN_NGL` | `--min-ngl` | `SWEEP_MIN_NGL=16` |
| `SWEEP_MIN_CTK` | `--min-ctk` | `SWEEP_MIN_CTK=q8_0` |
| `SWEEP_MIN_THREADS` | `--min-threads` | `SWEEP_MIN_THREADS=8` |
| `SWEEP_MIN_B` | `--min-b` | `SWEEP_MIN_B=1024` |
| `SWEEP_MODEL_LIST` | `--model-list` | `SWEEP_MODEL_LIST=~/list.txt` |
| `SWEEP_NO_CONFIRM` | `--no-confirm` | `SWEEP_NO_CONFIRM=true` |
| `SWEEP_DRY_RUN` | `--dry-run` | `SWEEP_DRY_RUN=true` |
| `SWEEP_DEBUG` | `--debug` | `SWEEP_DEBUG=true` |
| `SWEEP_GOAL_HITS` | `--goal-hits` | `SWEEP_GOAL_HITS=5` |

---

## Sweep phases

| Phase | Name | What varies | Everything else |
|-------|------|-------------|-----------------|
| 0 | **NGL Probe** | Binary search for max stable GPU layers — starts at model's layer count (from GGUF metadata), falls back to 99 | Defaults — establishes MAX_NGL |
| 1 | **NGL Axis** | NGL values up to model's layer count (capped there since higher values are identical); near MAX_NGL by default (use `--start-ngl 0` for full sweep) | Defaults |
| 2 | **FA + KV Quant Axis** | Flash attention on/off × KV cache type | Best NGL from Phase 1 |
| 3 | **Thread Count** | CPU thread count variants | Best NGL, best FA/KV |
| 4 | **KV Offload** | KV cache in VRAM (nkvo=0) vs RAM (nkvo=1) | Best NGL, best FA/KV, best threads |
| 5 | **Batch Size** | ubatch and batch size variants | Best values so far |
| 6 | **Context Ceiling** | Prompt size scaled up to OOM/timeout, with fallback configs on OOM; timeout runs are recorded with wall time | Best values so far |
| 7 | **Full Combination Matrix** | Cartesian product of all best-per-axis working sets; with `--goal`, runs ranked and exits early once the goal is satisfied | — |

---

## Smart defaults

Common use cases work without any extra flags. The key smart behaviors:

### Phase 0/1 — NGL ceiling capped at model layer count

At sweep start, llamaseye reads the model's layer count from its GGUF metadata. NGL values above that count are functionally identical (llama.cpp silently clamps NGL to the layer count), so:

- **Phase 0** starts its probe at `NumLayers` instead of 99 — eliminating up to 15 wasted probe runs for small models
- **Phase 1** caps its sweep list at `NumLayers` — shrinking the NGL working set and keeping Phase 7's cartesian product manageable

If GGUF parsing fails (non-standard file), both phases fall back to the 99 ceiling.

### Phase 1 — NGL start

Phase 1 starts at `MAX_NGL − 2×step` by default, testing only the top ~3 NGL values near the VRAM ceiling. Low-NGL configs rarely matter for performance. Use `--start-ngl 0` for a full 0→MAX_NGL sweep, or `--start-ngl 40 --ngl-dir down` to sweep downward from a specific cap.

### Phase 6 — Context fallbacks

When the primary config OOMs at a given context size, Phase 6 automatically tries progressively more memory-friendly alternatives before giving up:
1. Flip nkvo (move KV cache from VRAM → RAM)
2. More-compressed ctk types (q4_0, turbo types) × both nkvo values

Only ctk/nkvo values already validated by Phases 2 and 4 are tried.

### Fine-grained context sweep

By default Phase 6 sweeps context as powers of two (512 → 1024 → … → 65536 → 131072). This means a more-compressed KV type might unlock an intermediate context size that the sweep never discovers.

Use `--fine-ctx` to enable midpoint bisection: when all fallbacks fail at a ctx size, the sweep bisects between the last successful ctx and the failed ctx, probing the midpoint and narrowing until the gap is ≤ `--ctx-step-min` (default 8192).

```
# example: turbo3 unlocks 98304 on a card where q4_0 maxes at 65536
bash llamaseye.sh --model my.gguf --fine-ctx
```

Each bisection probe runs the full primary + fallback sequence, so `--fine-ctx` is off by default — it adds real runtime cost at large context sizes.

### Phase 7 — Auto-derived minimum filters

When `--min-*` flags are not set, Phase 7 auto-applies minimum filters so the combination matrix stays focused on high-value configs:

| Axis | Auto default | Override to disable |
|------|-------------|---------------------|
| NGL | `MAX_NGL − 1 step` (top 2 values) | `--min-ngl 0` |
| Threads | `HW_CPU_PHYSICAL` (physical core count) | `--min-threads 1` |
| Context | `--start-ctx` value, else `8192` | `--min-ctx 0` |
| KV type | `q8_0` normally; auto-lowered to most-compressed Phase-2-validated type when Phase 6 hit OOM | `--min-ctk q4_0` |
| Batch | `BEST_B / 2` | `--min-b 512` |

If `--start-ctx` is set and no context at or above that size succeeds in Phase 6, Phase 7 is skipped with a clear warning rather than silently running a useless matrix at a tiny fallback context.

**KV quality order** (low → high): `turbo2 turbo3 turbo4 q4_0 q8_0 f16`
- `--min-ctk turbo3` keeps turbo3, turbo4, q4_0, q8_0, f16 (excludes only turbo2)
- `--min-ctk q8_0` keeps q8_0 and f16 only (the default)
- `--min-ctk q4_0` includes all types (effectively disables the ctk filter)

---

## TurboQuant KV cache types

Passing `--turbo-bench <path>` enables three additional KV cache quantisation types: **turbo2**, **turbo3**, and **turbo4**, sourced from the [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork. These compress the KV cache 3–6× compared to f16, freeing VRAM for more layers or larger contexts without a significant quality penalty.

| Type | Compression vs f16 | Flash attn required |
|------|--------------------|---------------------|
| `turbo2` | ~6.4× | No |
| `turbo3` | ~4.3× | Yes (auto-enabled) |
| `turbo4` | ~3.2× | Yes (auto-enabled) |

The TurboQuant binary is verified at startup by probing it with `-ctk turbo3`. If the flag is accepted, turbo types are enabled. If the path is missing or the flag is rejected, turbo types are silently omitted and the sweep continues with the standard KV type set. It is safe to always pass `--turbo-bench` — the script handles an invalid path gracefully.

---

## Output structure

Results are written to `<output-dir>/<model-stem>/`:

```
results/
├── summary.md                        # Cross-model winner table (multi-model runs only)
└── Qwen3-14B-Q4_K_M/
    ├── sweep.jsonl       # One JSON object per completed run (source of truth)
    ├── sweep.md          # Human-readable Markdown summary (regenerable with --report)
    ├── sweep.log         # Full execution log
    ├── hardware.json     # Hardware snapshot captured at start
    ├── state.json        # Resume state (completed phases + best values + working sets)
    └── raw/
        └── <run-id>.txt  # Raw llama-bench stdout for each run
```

`sweep.jsonl` is append-only and is the source of truth. `state.json` tracks which phases are complete and the best parameter values discovered so far, enabling `--resume` to pick up exactly where it left off.

**`sweep.md` sections:**
- **Best Configurations** — top 10 results across all phases ranked by TG t/s
- **Per-phase tables** — all runs for each phase, sorted by TG t/s, with a `> **Winner:**` callout line showing the best config for that axis
- **Goal Results** — when `--goal` was used, Phase 7 rows that met the target
- **Context Frontier** — max stable context per (ngl, ctk, nkvo) combo from Phase 7
- **Slow context** — Phase 6 sizes that timed out (achievable but impractical for interactive use)

`sweep.md` can be regenerated at any time from `sweep.jsonl` without re-running benchmarks:
```bash
bash llamaseye.sh --report --output-dir ./results
```

---

## Model list file format

`--model-list` accepts a plain text file with one model filename per line. Lines beginning with `#` and blank lines are ignored.

```
# my_models.txt — models to sweep

Qwen3-14B-Q4_K_M.gguf
Qwen3-14B-Q6_K.gguf
Llama-3.1-8B-Instruct-Q8_0.gguf

# WIP — not ready yet
# Mixtral-8x7B-Q4_K_M.gguf
```

Paths are resolved relative to `--models-dir`. If `--models-dir` is not set, filenames are treated as absolute or relative to the working directory.

---

## Hardware portability

At startup the script probes:

- **CPU cores** — via `nproc` (Linux) or `sysctl -n hw.logicalcpu` (macOS)
- **System RAM** — to compute safe context-size upper bounds
- **GPU VRAM** — via `nvidia-smi` or `system_profiler` (Apple Silicon)
- **Compute backend** — cuda / metal / cpu, inferred from the llama-bench binary
- **Thermal sensors** — `nvidia-smi` for GPU, `sensors` or `/sys/class/thermal` for CPU

All sweep parameters are derived from these detected values. The script contains no hardcoded machine-specific constants.

---

## Design philosophy

Each phase sweeps exactly one axis while holding everything else at sane defaults. The only cross-phase dependency is `MAX_NGL`, which is established by Phase 0 and used as the ceiling for all subsequent phases. Phases 1–6 each produce a working set of values for their axis. Phase 7 takes the Cartesian product of all working sets and runs the full combination matrix, confirming which configs compose well and revealing the true peak configuration. This one-variable-at-a-time discipline keeps results interpretable and makes it straightforward to re-run individual phases in isolation with `--only-phases`.

---

## Docs

See `docs/spec.md` for the full engineering specification.

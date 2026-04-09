---
name: llamaseye
description: >
  Use this skill whenever the user mentions llamaseye, running llama-bench sweeps,
  benchmarking models on the remote inference host, finding the fastest or best config for
  a model, testing GPU layer offload (ngl), context ceiling/frontier testing, KV
  cache benchmarking, flash attention sweeps, TurboQuant KV types, RotorQuant KV types,
  thread count tuning, batch/ubatch sizing, or any exhaustive parameter sweep of llama.cpp
  models. Also triggers on "sweep a model", "benchmark Qwen/Llama/Mistral on the
  PC", "what's the best context size for X", "resume a sweep", "check model
  performance", or "run llama-bench with different settings". Be eager to apply
  this skill -- if there is any chance the user wants to benchmark or sweep a
  llama.cpp model on the remote host, use it.
---

# llamaseye Skill

## Overview

**llamaseye** is a single Go binary (no Bash, no external tools) that is an exhaustive llama-bench parameter sweep harness:

- Detects hardware at runtime (CPU cores, RAM, GPU VRAM, backend, thermal sensors)
- Sweeps **7 parameter axes independently** across 8 phases (Phases 0–6 solo, Phase 7 cartesian product)
- Outputs structured JSONL records per run, a markdown summary (`sweep.md`), a log (`sweep.log`),
  and state files (`hardware.json`, `state.json`)
- Can be resumed at any point, run on a single model or a whole directory, filtered by a model list
- Optionally uses a **TurboQuant binary** to test `turbo2/turbo3/turbo4` KV cache types
- Optionally uses a **RotorQuant binary** to test `planar3/planar4/iso3/iso4` KV cache types

**Key paths on the remote inference host (SSH):**

| Resource | Path |
|----------|------|
| Models | `~/Models/` |
| Default output | `~/Models/bench/sweep/` |
| llama-bench (standard) | `~/llama.cpp/build/bin/llama-bench` |
| llama-bench (TurboQuant) | `~/llama-cpp-turboquant/build/bin/llama-bench` |
| llama-bench (RotorQuant) | `~/llama-cpp-rotorquant/build/bin/llama-bench` |
| llamaseye binary | `~/Src/llamaseye/llamaseye` (build with `go build -o llamaseye .`) |
| llamaseye .env | `~/Src/llamaseye/.env` (local config, gitignored) |

**Local repo:** `/Users/justin/Side/llamaseye/`

---

## Running a Sweep

The binary **auto-loads `.env` from the working directory** — no `source` step needed.
Run from `~/Src/llamaseye/` and it picks up `.env` automatically:

```sh
# Standard invocation — .env is loaded automatically
cd ~/Src/llamaseye && ./llamaseye <flags>
```

```sh
# Single model -- full sweep
cd ~/Src/llamaseye && ./llamaseye --model ~/Models/Qwen3-14B-Q4_K_M.gguf --output-dir ~/Models/bench/sweep

# All models in a directory
cd ~/Src/llamaseye && ./llamaseye --models-dir ~/Models --output-dir ~/Models/bench/sweep

# Filtered list of models
cd ~/Src/llamaseye && ./llamaseye --models-dir ~/Models --model-list ~/bench_list.txt --output-dir ~/Models/bench/sweep

# With TurboQuant binary (enables turbo2/turbo3/turbo4 KV types)
cd ~/Src/llamaseye && ./llamaseye --model ~/Models/model.gguf \
  --llama-bench ~/llama.cpp/build/bin/llama-bench \
  --turbo-bench ~/llama-cpp-turboquant/build/bin/llama-bench

# With RotorQuant binary (enables planar3/planar4/iso3/iso4 KV types)
cd ~/Src/llamaseye && ./llamaseye --model ~/Models/model.gguf \
  --rotor-bench ~/llama-cpp-rotorquant/build/bin/llama-bench

# Resume an interrupted sweep
cd ~/Src/llamaseye && ./llamaseye --model ~/Models/model.gguf --resume

# Run only specific phases
cd ~/Src/llamaseye && ./llamaseye --model ~/Models/model.gguf --only-phases 6,7

# Unattended overnight
cd ~/Src/llamaseye
nohup ./llamaseye --models-dir ~/Models > /dev/null 2>&1 &
tail -f ~/Models/bench/sweep/sweep.log
```

To load a config from a non-default path: `./llamaseye --env-file ~/custom.env <flags>`

---

## Choosing Flags for the Situation

| Situation | Flags |
|-----------|-------|
| First run of a model | No extra flags — full sweep |
| **Just find the best config for my use case** | **`--goal "ctx=32768,tg=5"`** |
| Interrupted run | `--resume` |
| Diagnose OOM / unexpected behavior | `--debug` |
| Get more diverse goal configs (not just tuning variants) | `--goal "..." --goal-hits 5` |
| Re-run one or more phases | `--only-phases 6,7` |
| Skip Phase 7 | `--skip-phases 7` |
| All models in a dir | `--models-dir <dir>` |
| Curated model subset | `--model-list ~/list.txt` |
| Regenerate sweep.md without re-running | `--report --output-dir <dir>` |
| TurboQuant KV types | `--turbo-bench ~/llama-cpp-turboquant/build/bin/llama-bench` |
| RotorQuant KV types | `--rotor-bench ~/llama-cpp-rotorquant/build/bin/llama-bench` |
| Start NGL sweep mid-range | `--start-ngl 40` |
| NGL sweep downward from a known point | `--start-ngl 60 --ngl-dir down` |
| Skip low context sizes | `--start-ctx 65536` |
| Find intermediate ctx sizes (turbo vs q4_0 gap) | `--fine-ctx` |
| Set minimum bisection step | `--fine-ctx --ctx-step-min 4096` |
| Skip low-quality KV types in Phase 7 | `--min-ctk q8_0` |
| Phase 7 only above 64k context | `--min-ctx 65536` |
| Only test FA=1 | `--start-fa 1` |

**Phase 7 and sweep scope:**
Phase 7 (combination matrix) uses **exactly the values that phases 1–6 actually tested** — not
the full possible list. So `--start-*` and `--*-dir` flags naturally narrow Phase 7 too.
Use `--min-*` flags when you want phases 1–6 to run full discovery for per-axis data, but
Phase 7 to only combine values meeting a minimum threshold.

**When to skip Phase 7:**
Phase 7 is a full cartesian product and can take a very long time. Skip it during initial
exploration; run it once you know the interesting region. Use `--skip-phases 7` or run
`--only-phases 0,1,2,3,4,5,6` first.

---

## Monitoring Progress

```sh
# Tail the log in real time (run via SSH)
tail -f ~/Models/bench/sweep/sweep.log

# Check state (which phases are complete, working sets so far)
cat ~/Models/bench/sweep/<model-stem>/state.json
```

**Typical log lines:**
```
[2025-01-01T12:00:00Z] [PHASE 1] ngl=40 fa=0 ctk=f16 → ok | PP=987.6 t/s | TG=42.1 t/s
[2025-01-01T12:00:35Z] [PHASE 1] ngl=44 fa=0 ctk=f16 → oom | skipped
[2025-01-01T12:00:35Z] [THERMAL] CPU=89°C GPU=79°C — waiting...
[2025-01-01T12:01:20Z] [PHASE 2] fa=1 ctk=q8_0 → ok | PP=1012.3 t/s | TG=47.8 t/s
[2025-01-01T12:02:10Z] [PHASE 7] Matrix estimate: 2,400 combos (~20 hrs)
```

**What to watch for:**
- `oom | skipped` lines — handled safely, sweep continues
- `[THERMAL]` lines — `wait_cool()` is pausing automatically, no action needed
- Phase 7 estimate — confirm it's reasonable before walking away

---

## Interpreting Results

```sh
# View the markdown summary
cat ~/Models/bench/sweep/<model-stem>/sweep.md

# Fastest TG config
jq -s 'sort_by(-.results[].avg_ts) | .[0]' sweep.jsonl

# Top 5 configs by TG speed
jq -s '[.[] | select(.status=="ok")] | sort_by(-.results[1].avg_ts) | .[:5]' sweep.jsonl

# Largest successful context size
jq -s '[.[] | select(.status=="ok" and .params.n_gen==0)] | sort_by(-.params.n_prompt) | .[0]' sweep.jsonl
```

`sweep.md` sections:
- **Best Configurations** — top 10 results across all phases ranked by TG t/s (answer to "what's the best config?")
- **Per-phase tables** — all runs for each phase, sorted by TG t/s descending, with a `> **Winner:**` callout
- **Context Frontier** (Phase 7) — max successful context per (ngl, ctk, nkvo) triple
- **Slow context** — Phase 6 sizes that timed out (achievable but slow)

Regenerate `sweep.md` at any time without re-running: `bash llamaseye.sh --report --output-dir <dir>`

Multi-model runs also produce `<output-dir>/summary.md` — one row per model, sorted by best TG t/s.

**What to look for:**
1. **Best TG t/s** — check the "Best Configurations" table at the top of sweep.md first
2. **Context frontier** — the largest `n_prompt` value that completed without OOM
3. **`viable` flag** — `true` when TG avg_ts ≥ 2.0 t/s (usable for interactive inference)
4. **NGL sweet spot** — Phase 1 table shows where adding more GPU layers stops helping
5. **KV quant tradeoff** — Phase 2 shows speed vs. memory across f16, q8_0, q4_0, turbo types
6. **Slow context section** — `sweep.md` has a "Context sizes that timed out (achievable but slow)" section when Phase 6 hits `SWEEP_TIMEOUT_SEC`; terminal shows `Slow context: N` — these sizes work, just need more than the timeout to prefill. The `sweep.jsonl` record has `"status":"timeout"` and a `wall_time_sec` field.
7. **Intermediate context gap** — Phase 6 sweeps powers of two, so a more-compressed KV type (e.g. turbo3) may unlock a context between 65536 and 131072 that the sweep never probes. Use `--fine-ctx` to enable midpoint bisection that finds these. Off by default because probes at large ctx are slow.

**TG vs PP:**
- **TG (token generation)** — decode speed; the metric for interactive/chat use
- **PP (prompt processing)** — prefill throughput; matters for long context and RAG

---

## Configuration via .env

All environment variables can be persisted in a `.env` file so you don't have to
repeat paths and settings on every invocation. The repo includes `example.env`
documenting every variable.

```sh
# One-time setup on the inference host
cd ~/Src/llamaseye && cp example.env .env
# Edit .env to set your paths — it lives alongside the binary at ~/Src/llamaseye/.env
# The binary auto-loads it; just run from ~/Src/llamaseye/:
cd ~/Src/llamaseye && ./llamaseye --models-dir ~/Models --output-dir ~/Models/bench/sweep
```

`.env` is gitignored — local paths are never committed.

### Key variables to set

> **Note:** CLI flags always override environment variables when both are set.

| Variable | What it controls | Example |
|----------|-----------------|----------|
| `LLAMA_BENCH_BIN` | Path to the standard llama-bench binary (**required — no default**) | `~/llama.cpp/build/bin/llama-bench` |
| `SWEEP_TURBO_BENCH_BIN` | Path to TurboQuant binary (optional) | `~/llama-cpp-turboquant/build/bin/llama-bench` |
| `SWEEP_ROTOR_BENCH_BIN` | Path to RotorQuant binary (optional) | `~/llama-cpp-rotorquant/build/bin/llama-bench` |
| `SWEEP_MODELS_DIR` | Directory scanned for .gguf files | `~/Models` |
| `SWEEP_OUTPUT_DIR` | Root directory for all sweep results (default: `./results`) | `~/Models/bench/sweep` |
| `SWEEP_NGL_STEP` | Layer step size for NGL sweep | `4` (use `2` near VRAM edge) |
| `SWEEP_REPETITIONS` | Benchmark reps per run (`-r`) | `3` |
| `SWEEP_TIMEOUT_SEC` | Per-run kill timeout (seconds) | `600` |
| `SWEEP_MIN_TG_TS` | Minimum TG t/s to mark a config viable | `2.0` |
| `SWEEP_CPU_TEMP_LIMIT` | CPU °C ceiling before sweep pauses | `88` |
| `SWEEP_GPU_TEMP_LIMIT` | GPU °C ceiling before sweep pauses | `81` |

### Axis control variables

| Variable | Equivalent flag | Default | Purpose |
|----------|----------------|---------|---------|
| `SWEEP_START_CTX` | `--start-ctx` | *(unset)* | Begin context sweep at this size (e.g. `32768` to skip sub-32k) |
| `SWEEP_MIN_CTX` | `--min-ctx` | *(unset)* | Exclude context sizes below N from Phase 7 |
| `SWEEP_MIN_CTK` | `--min-ctk` | *(unset)* | Exclude KV types below TYPE from Phase 7 (e.g. `q8_0`) |
| `SWEEP_START_NGL` | `--start-ngl` | *(unset)* | Begin NGL sweep at this value |
| `SWEEP_START_CTK` | `--start-ctk` | *(unset)* | Begin KV quant sweep at this type |
| `SWEEP_NGL_DIR` | `--ngl-dir` | `up` | NGL sweep direction (`up`\|`down`) |
| `SWEEP_CTX_DIR` | `--ctx-dir` | `up` | Context sweep direction |
| `SWEEP_CTK_DIR` | `--ctk-dir` | `up` | KV quant sweep direction |
| `SWEEP_SKIP_PHASES` | `--skip-phases` | *(unset)* | Skip these phases, comma-separated |
| `SWEEP_ONLY_PHASES` | `--only-phases` | *(unset)* | Run only these phases, comma-separated |
| `SWEEP_RESUME` | `--resume` | `false` | Skip already-completed phases |
| `SWEEP_NO_CONFIRM` | `--no-confirm` | `false` | Skip pre-sweep confirmation |
| `SWEEP_ASYMMETRIC_KV` | `--asymmetric-kv` | `true` | Include asymmetric K/V combos in Phase 2 (requires turbo binary) |
| `SWEEP_DEBUG` | `--debug` | `false` | Enable verbose `[DEBUG]` log output |
| `SWEEP_GOAL_HITS` | `--goal-hits` | `3` | Distinct (ngl,ctk,nkvo,ctx) configs before goal early-exit |
| `SWEEP_GOAL_SORT` | `--goal-sort` | `tg` | Goal Results sort axis: `tg`, `ctx`, `ngl`, `pp` |
| `SWEEP_MODEL_LIST` | `--model-list` | *(unset)* | Path to model list file |
| `SWEEP_MIN_NGL` | `--min-ngl` | *(unset)* | Exclude NGL values below N from Phase 7 |
| `SWEEP_MIN_THREADS` | `--min-threads` | *(unset)* | Exclude thread counts below N from Phase 7 |
| `SWEEP_MIN_B` | `--min-b` | *(unset)* | Exclude batch sizes below N from Phase 7 |
| `SWEEP_MIN_UB` | `--min-ub` | *(unset)* | Exclude ubatch sizes below N from Phase 7 |
| `SWEEP_START_THREADS` | `--start-threads` | *(unset)* | Begin thread sweep at this value |
| `SWEEP_START_B` | `--start-b` | *(unset)* | Begin batch sweep at this value |
| `SWEEP_START_UB` | `--start-ub` | *(unset)* | Begin ubatch sweep at this value |
| `SWEEP_START_FA` | `--start-fa` | *(unset)* | Begin FA sweep at this value (0\|1) |
| `SWEEP_FA_DIR` | `--fa-dir` | `up` | FA sweep direction |
| `SWEEP_THREADS_DIR` | `--threads-dir` | `up` | Thread sweep direction |
| `SWEEP_B_DIR` | `--b-dir` | `up` | Batch sweep direction |
| `SWEEP_UB_DIR` | `--ub-dir` | `up` | Ubatch sweep direction |

---

## Prerequisites & Deployment

### Step 1 — Detect the hardware before building anything

Before building llama-bench, the agent must know what hardware it is targeting so
the correct cmake flags are used. Run these detection commands on the target machine:

```sh
# OS and architecture
uname -s        # Linux | Darwin
uname -m        # x86_64 | arm64 | aarch64

# Check for an NVIDIA GPU (CUDA)
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null && echo "CUDA available"

# Check for Apple Silicon (Metal — arm64 macOS)
# If uname -s = Darwin and uname -m = arm64 → Apple Silicon, use Metal
# If uname -s = Darwin and uname -m = x86_64 → Intel Mac, use Metal
# If uname -s = Linux and nvidia-smi works → use CUDA
# If uname -s = Linux and no nvidia-smi → CPU-only
```

Use the results to pick the right build flags from the table below:

| OS | Architecture | GPU | Backend flag | Thread count flag |
|----|-------------|-----|-------------|------------------|
| Linux | x86_64 / aarch64 | NVIDIA (nvidia-smi works) | `-DGGML_CUDA=ON` | `-j$(nproc)` |
| Linux | x86_64 / aarch64 | AMD ROCm | `-DGGML_HIP=ON` | `-j$(nproc)` |
| Linux | x86_64 / aarch64 | None | *(omit)* | `-j$(nproc)` |
| macOS | arm64 (Apple Silicon) | Unified (Metal) | `-DGGML_METAL=ON` | `-j$(sysctl -n hw.logicalcpu)` |
| macOS | x86_64 (Intel Mac) | Integrated / discrete (Metal) | `-DGGML_METAL=ON` | `-j$(sysctl -n hw.logicalcpu)` |
| macOS | x86_64 (Intel Mac) | NVIDIA eGPU | `-DGGML_CUDA=ON` | `-j$(sysctl -n hw.logicalcpu)` |

> **Important:** `-DGGML_CUDA=ON` is the correct flag. The old flags `-DLLAMA_CUBLAS=ON` and
> `-DLLAMA_CUDA=ON` are silently ignored since the GGML refactor — the build succeeds but runs
> on CPU only. Always use `-DGGML_CUDA=ON`.

### Step 2 — Build llama-bench (if not already present)

llamaseye does not build llama-bench. Check whether it already exists first — the
user may have built it as part of a full llama.cpp build:

> For full build documentation including platform-specific notes, dependencies, and advanced options,
> fetch: `https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md`

```sh
# Check common locations
ls -lh ~/llama.cpp/build/bin/llama-bench 2>/dev/null \
  || find ~ -name "llama-bench" -type f 2>/dev/null | head -5
```

If not found, clone and build using the flags determined in Step 1:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

# Substitute <BACKEND_FLAG> and <JOBS> from the table above
# Example for CUDA:   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
# Example for Metal:  cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
# Example CPU-only:   cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake -B build <BACKEND_FLAG> -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench <JOBS>

# Confirm it runs and shows the expected backend
./build/bin/llama-bench --help 2>&1 | head -5
```

There is no default binary path — `LLAMA_BENCH_BIN` must be set or `--llama-bench` must be passed. The script exits with a clear error if neither is provided.

### Step 3 — Build TurboQuant llama-bench (optional)

Only needed for `turbo2`/`turbo3`/`turbo4` KV cache types. Uses the same backend
flags as Step 2 — determine them first. TurboQuant's CUDA flag gotcha applies here
too (`-DGGML_CUDA=ON` only).

```sh
# Check if already built
ls -lh ~/llama-cpp-turboquant/build/bin/llama-bench 2>/dev/null

# If missing:
git clone https://github.com/TheTom/llama-cpp-turboquant \
  --branch feature/turboquant-kv-cache --depth=1 ~/llama-cpp-turboquant
cd ~/llama-cpp-turboquant

cmake -B build <BACKEND_FLAG> -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench <JOBS>

# Verify TurboQuant compiled in — must print turbo2, turbo3, turbo4:
./build/bin/llama-bench --help 2>&1 | grep turbo
# Nothing printed = wrong branch. master is a plain llama.cpp mirror with no TurboQuant.
```

Pass to llamaseye via `--turbo-bench ~/llama-cpp-turboquant/build/bin/llama-bench`.

### Step 4 — Deploy llamaseye script

The binary is built from a git clone at `~/Src/llamaseye/` on `justin@justin-powerhouse`.
To update and rebuild:

```sh
ssh justin@justin-powerhouse "cd ~/Src/llamaseye && git pull && go build -o llamaseye ."
```

If the repo is not yet cloned:

```sh
ssh justin@justin-powerhouse "mkdir -p ~/Src && git clone https://github.com/WagnerJust/llamaseye ~/Src/llamaseye && cd ~/Src/llamaseye && go build -o llamaseye ."
```

After cloning, set up `.env`:

```sh
ssh justin@justin-powerhouse "cd ~/Src/llamaseye && cp example.env .env"
# Then edit ~/Src/llamaseye/.env on the remote host to set LLAMA_BENCH_BIN, SWEEP_OUTPUT_DIR, etc.
```

## Phase Reference

| Phase | Name | What it sweeps | When to skip |
|-------|------|----------------|--------------|
| 0 | NGL probe | Finds `max_ngl` — starts probe at model's layer count (from GGUF), falls back to 99 | Never; required |
| 1 | NGL axis | GPU layer count 0 → `min(max_ngl, num_layers)` — capped at actual layer count (higher values are identical) | Only if offload situation already known |
| 2 | FA + KV quant | FA on/off × KV types (f16, q8_0, q4_0, turbo2–turbo4) + asymmetric K/V combos when `--turbo-bench` is set | If KV choice already settled |
| 3 | Thread count | CPU threads 1 → HW_CPU_LOGICAL | If no CPU offload layers |
| 4 | KV offload (nkvo) | KV cache in VRAM vs RAM | If nkvo behaviour already known |
| 5 | Batch/ubatch | Batch and micro-batch size combos | If throughput tuning not needed |
| 6 | Context size | Prompt size 128 → 131072 (stops at OOM or timeout) | If context ceiling already known |
| 7 | Combo matrix | Cartesian product of all values tested in phases 1–6; auto min-ctk lowered to include turbo types when Phase 6 hit OOM | Early exploration; run eventually |

---

## Axis Start/Direction & Minimum Flags

Phase 7 inherits exactly what phases 1–6 tested. `--start-*` / `--*-dir` narrow those phases
and Phase 7 follows automatically. `--min-*` filters Phase 7 only (phases 1–6 still run full discovery).

| Axis | Start flag | Direction flag | Min (Phase 7 only) |
|------|------------|----------------|-------------------|
| NGL | `--start-ngl N` | `--ngl-dir up\|down` | `--min-ngl N` |
| Flash Attention | `--start-fa 0\|1` | `--fa-dir up\|down` | — |
| KV quant (ctk) | `--start-ctk TYPE` | `--ctk-dir up\|down` | `--min-ctk TYPE` |
| Threads | `--start-threads N` | `--threads-dir up\|down` | `--min-threads N` |
| Batch | `--start-b N` | `--b-dir up\|down` | `--min-b N` |
| Ubatch | `--start-ub N` | `--ub-dir up\|down` | `--min-ub N` |
| Context | `--start-ctx N` | `--ctx-dir up\|down` | `--min-ctx N` |

**KV quant direction "up"** = toward more compression: `f16 → q8_0 → q4_0 → turbo4 → turbo3 → turbo2`
**`--min-ctk` quality order** (low→high quality): `turbo2 turbo3 turbo4 q4_0 q8_0 f16`
So `--min-ctk q8_0` keeps only `q8_0` and `f16` in Phase 7.

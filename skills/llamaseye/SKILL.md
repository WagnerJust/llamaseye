---
name: llamaseye
description: >
  Use this skill whenever the user mentions llamaseye, running llama-bench sweeps,
  benchmarking models on the Powerhouse PC, finding the fastest or best config for
  a model, testing GPU layer offload (ngl), context ceiling/frontier testing, KV
  cache benchmarking, flash attention sweeps, TurboQuant KV types, thread count
  tuning, batch/ubatch sizing, or any exhaustive parameter sweep of llama.cpp
  models. Also triggers on "sweep a model", "benchmark Qwen/Llama/Mistral on the
  PC", "what's the best context size for X", "resume a sweep", "check model
  performance", or "run llama-bench with different settings". Be eager to apply
  this skill -- if there is any chance the user wants to benchmark or sweep a
  llama.cpp model on the Powerhouse, use it.
---

# llamaseye Skill

## Overview

**llamaseye** (`llamaseye.sh`) is an exhaustive llama-bench parameter sweep harness that:

- Detects hardware at runtime (CPU cores, RAM, GPU VRAM, backend, thermal sensors)
- Sweeps **7 parameter axes independently** across 8 phases (Phase 0-6 solo, Phase 7 cartesian product)
- Outputs structured JSONL records per run, a markdown summary table (`sweep.md`), a running log
  (`sweep.log`), and state files (`hardware.json`, `state.json`)
- Can be **resumed** at any point, run on a single model or an entire directory, and filtered by a model list
- Optionally uses a **TurboQuant binary** to test `turbo2/turbo3/turbo4` KV cache types

**Key paths on the Powerhouse PC (`justin@justin-powerhouse` via Tailscale):**

| Resource | Path |
|----------|------|
| Models | `~/Models/` |
| Default output | `~/Models/bench/sweep/` |
| llama-bench (standard) | `~/llama.cpp/build/bin/llama-bench` |
| llama-bench (TurboQuant) | `~/llama-cpp-turboquant/build/bin/llama-bench` |
| llamaseye script | `~/llamaseye.sh` (SCP from local Mac if needed) |

**Repo location (local Mac):** `/Users/justin/Side/llamaseye/llamaseye.sh`

---

## Running a Sweep

### Single model -- full sweep (all phases)

```sh
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf"
```

### Single model -- full sweep with TurboQuant binary

```sh
ssh justin@justin-powerhouse \
  "LLAMA_BENCH_BIN_TURBO=~/llama-cpp-turboquant/build/bin/llama-bench \
   bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf"
```

### Whole models directory

```sh
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/"
```

### Whole directory with a model list filter

```sh
# Create the list on the PC first:
ssh justin@justin-powerhouse \
  "printf 'Qwen3-14B-Q4_K_M.gguf\nLlama-3.1-8B-Q6_K.gguf\n' > ~/bench_list.txt"

ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/ --model-list ~/bench_list.txt"
```

### Resume an interrupted sweep

```sh
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf --resume"
```

### Run only specific phases (e.g. phases 5 and 6)

```sh
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf --phases 5,6"
```

### Start mid-axis and sweep downward

```sh
# Start NGL sweep at layer 40, going down
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf \
   --start-ngl 40 --ngl-dir down"
```

### Custom output directory

```sh
ssh justin@justin-powerhouse \
  "bash ~/llamaseye.sh ~/Models/Qwen3-14B-Q4_K_M.gguf \
   --output ~/Models/bench/custom_run/"
```

---

## Choosing Flags for the Situation

| Situation | Flags to use |
|-----------|--------------|
| First time running a model | No extra flags -- full sweep |
| Interrupted run | `--resume` |
| Re-run just one phase | `--phases <N>` |
| Re-run multiple phases | `--phases 5,6,7` |
| Skip Phase 7 (no combo matrix) | `--phases 0,1,2,3,4,5,6` |
| Sweep all models in a dir | Pass the directory path instead of a file |
| Filter which models to sweep | `--model-list ~/list.txt` |
| TurboQuant KV types (turbo2/3/4) | Set `LLAMA_BENCH_BIN_TURBO` env var |
| Start NGL axis at a known-good value | `--start-ngl <N>` |
| NGL sweep going down from max | `--ngl-dir down` |
| Start context sweep at specific size | `--start-ctx <N>` |
| Sweep context downward | `--ctx-dir down` |
| Custom thread starting point | `--start-threads <N>` |
| Custom batch starting point | `--start-batch <N>` |

**When to use the TurboQuant binary:**
- Any time you want to test `turbo2`, `turbo3`, or `turbo4` KV cache quantization types
- Only if `~/llama-cpp-turboquant/build/bin/llama-bench` exists on the PC
- Set via env var, not a flag: `LLAMA_BENCH_BIN_TURBO=~/llama-cpp-turboquant/build/bin/llama-bench`

**When to skip Phase 7:**
- Phase 7 is a full cartesian product -- it can be very long for models with many viable values
- Skip it during initial exploration; run it last once you know the interesting region
- Use `--phases 0,1,2,3,4,5,6` to stop before the combo matrix

---

## Monitoring Progress

### Tail the sweep log in real time

```sh
ssh justin@justin-powerhouse \
  "tail -f ~/Models/bench/sweep/sweep.log"
```

### Watch the last 50 lines (less noise)

```sh
ssh justin@justin-powerhouse \
  "tail -n 50 -f ~/Models/bench/sweep/sweep.log"
```

### Typical log line patterns

```
[llamaseye] Phase 1: NGL axis sweep
[llamaseye] -> ngl=35 | tg=47.3 t/s | pp=312.1 t/s
[llamaseye] CHECK ngl=35 viable (tg >= threshold)
[llamaseye] Phase 2: FA + KV quant axis sweep
[llamaseye] -> fa=1 kv=q8_0 | tg=48.1 t/s
[llamaseye] Phase 7: Combination matrix (N combos)
[llamaseye] CHECK Sweep complete -- results in sweep.md
```

**What to watch for:**
- Lines marked `viable` -- these values will feed Phase 7
- `OOM` or `FAILED` lines -- the config was skipped safely, no intervention needed
- Thermal warnings -- `wait_cool()` pauses automatically; no action needed
- Phase transition headers -- each new phase logs a clear header line

### Check current state

```sh
ssh justin@justin-powerhouse "cat ~/Models/bench/sweep/state.json"
```

---

## Interpreting Results

### View the markdown summary

```sh
ssh justin@justin-powerhouse "cat ~/Models/bench/sweep/sweep.md"
```

`sweep.md` contains per-phase tables with columns such as:

| ngl | fa | kv_type | threads | nkvo | batch | ubatch | ctx | tg (t/s) | pp (t/s) | viable |
|-----|----|---------|---------|------|-------|--------|-----|----------|----------|--------|

**What to look for:**

1. **Best TG t/s** -- highest token generation speed; look in Phase 7 combo rows first
2. **Context frontier** -- the largest `ctx` value that completed without OOM
3. **Viable flag** -- only rows with `viable=1` fed into Phase 7
4. **NGL sweet spot** -- Phase 1 table shows where adding more GPU layers stops helping
5. **KV quant tradeoff** -- Phase 2 shows speed vs. quality tradeoff across `f16`, `q8_0`, `q4_0`,
   and `turbo2`-`turbo4` if the TurboQuant binary was used

**TG t/s vs PP t/s:**
- **TG (token generation)** -- autoregressive decode speed; this is the metric that matters for
  interactive inference and chat
- **PP (prompt processing / prefill)** -- throughput for ingesting the prompt; matters for long
  context and RAG pipelines

### Query the JSONL for the fastest config

```sh
ssh justin@justin-powerhouse \
  "jq -s 'sort_by(-.tg_ts) | .[0]' ~/Models/bench/sweep/sweep.jsonl"
```

### Find the largest viable context size

```sh
ssh justin@justin-powerhouse \
  "jq -s '[.[] | select(.viable==1)] | sort_by(-.ctx) | .[0]' \
   ~/Models/bench/sweep/sweep.jsonl"
```

### Top 5 configs by TG speed

```sh
ssh justin@justin-powerhouse \
  "jq -s 'sort_by(-.tg_ts) | .[:5] | .[] | {ngl,fa,kv_type,threads,ctx,tg_ts}' \
   ~/Models/bench/sweep/sweep.jsonl"
```

---

## Deploying to the PC

If `llamaseye.sh` is not on the PC yet, or needs to be updated from the local repo:

```sh
# From local Mac
scp /Users/justin/Side/llamaseye/llamaseye.sh \
    justin@justin-powerhouse:~/llamaseye.sh

# Make executable
ssh justin@justin-powerhouse "chmod +x ~/llamaseye.sh"
```

### Verify the standard llama-bench binary exists

```sh
ssh justin@justin-powerhouse \
  "ls -lh ~/llama.cpp/build/bin/llama-bench"
```

### Verify the TurboQuant binary (optional)

```sh
ssh justin@justin-powerhouse \
  "ls -lh ~/llama-cpp-turboquant/build/bin/llama-bench"
```

If the TurboQuant binary is missing and you need turbo KV types, build it first:

```sh
ssh justin@justin-powerhouse "
  cd ~/llama-cpp-turboquant &&
  git checkout feature/turboquant-kv-cache &&
  cmake -B build -DGGML_CUDA=ON &&
  cmake --build build --config Release -j8 --target llama-bench
"
```

---

## Phase Reference

| Phase | Name | What it sweeps | When to skip |
|-------|------|----------------|--------------|
| 0 | NGL probe | Finds `max_ngl` -- how many layers fit fully in VRAM | Never; required by Phase 1 |
| 1 | NGL axis | GPU layer count from 0 up to `max_ngl` | Only if VRAM/offload situation is already known |
| 2 | FA + KV quant | Flash attention on/off x KV types (`f16`, `q8_0`, `q4_0`, `turbo2`-`turbo4`) | If KV quant choice is already settled |
| 3 | Thread count | CPU thread count from 1 up to n_cores | If no CPU offload layers are used |
| 4 | KV offload (nkvo) | Whether KV cache is offloaded to GPU | If nkvo behavior is already known |
| 5 | Batch/ubatch | Batch size and micro-batch size combinations | If latency matters more than throughput |
| 6 | Context size | Context length up to 131072 | If context ceiling is already known |
| 7 | Combo matrix | Full cartesian product of all viable values from phases 1-6 | Early exploratory runs; always run eventually |

---

## Axis Start/Direction Flags

| Axis | Start flag | Direction flag | Direction values |
|------|------------|----------------|-----------------|
| NGL | `--start-ngl <N>` | `--ngl-dir <dir>` | `up` / `down` |
| Flash Attention | -- | -- | Always both tested |
| KV quant | `--start-kv <type>` | `--kv-dir <dir>` | `up` / `down` |
| Threads | `--start-threads <N>` | `--threads-dir <dir>` | `up` / `down` |
| KV offload | -- | -- | Both tested (binary axis) |
| Batch | `--start-batch <N>` | `--batch-dir <dir>` | `up` / `down` |
| Context | `--start-ctx <N>` | `--ctx-dir <dir>` | `up` / `down` |

**Tip:** Use `--<axis>-dir down` when you want to confirm a known-good large value works and stop
early once quality degrades. Use `up` (default) when starting from scratch to find the ceiling.

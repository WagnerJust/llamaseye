# llamaseye — Exhaustive llama-bench parameter sweep harness

Systematically sweep every meaningful llama-bench parameter combination for any GGUF model, record every result as JSONL, and surface the optimal configuration for your hardware.

---

## What it does

**llamaseye** runs llama-bench across every meaningful parameter combination for any GGUF model. It sweeps each axis independently — GPU layer offload (ngl), flash attention, KV cache quantisation type, thread count, KV offload ratio, batch size, and context size — then runs a full combination matrix (Phase 7) to confirm which configs work together and find the true performance ceiling.

Every result is recorded as JSONL in a per-model output directory, alongside a human-readable Markdown summary, a raw log, a hardware snapshot, and a resume-state file. Runs that trigger an OOM or timeout are caught, logged, and skipped — the sweep never hangs.

The script is fully portable: it detects CPU core count, available RAM, GPU VRAM, the active compute backend (cuda / metal / cpu), and the correct thermal-sensor commands at runtime. There are no hardcoded machine values. Optionally, pass a TurboQuant build of llama-bench via `--turbo-bench` to unlock turbo2/turbo3/turbo4 KV cache types from the llama-cpp-turboquant fork, which compress the KV cache 3–6× and enable much longer contexts on the same hardware.

---

## Quick start

**Single model:**
```bash
bash llamaseye.sh --model ~/Models/Qwen3-14B-Q4_K_M.gguf --output-dir ./results
```

**All models in a directory:**
```bash
bash llamaseye.sh --models-dir ~/Models --output-dir ./results
```

**From a model list file:**
```bash
bash llamaseye.sh --models-dir ~/Models --model-list my_models.txt --output-dir ./results
```

**With TurboQuant KV types:**
```bash
bash llamaseye.sh --model ~/Models/model.gguf --turbo-bench ~/llama-cpp-turboquant/build/bin/llama-bench
```

**Unattended overnight run:**
```bash
nohup bash llamaseye.sh --models-dir ~/Models --output-dir ./results > /dev/null 2>&1 &
```

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

The binary will be at `build/bin/llama-bench`. Pass its path to llamaseye via `--llama-bench <path>` or set the `LLAMA_BENCH_BIN` environment variable. The default assumed path is `~/llama.cpp/build/bin/llama-bench`.

> The build flags you choose determine which backends and features are available during the sweep. llamaseye works with any valid llama-bench binary — it does not require any specific build flags itself.

### Optional: TurboQuant llama-bench

To enable `turbo2`/`turbo3`/`turbo4` KV cache types, build a second binary from the [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork (branch `feature/turboquant-kv-cache`) and pass it via `--turbo-bench <path>`. The fork is otherwise identical to llama.cpp — same build flags apply.

```sh
git clone https://github.com/TheTom/llama-cpp-turboquant \
  --branch feature/turboquant-kv-cache --depth=1
cd llama-cpp-turboquant
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench -j$(nproc)

# Verify TurboQuant compiled in:
./build/bin/llama-bench --help 2>&1 | grep turbo
# Must print: turbo2, turbo3, turbo4 — if nothing shows, wrong branch was cloned
```

### Other dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| `jq` | JSON record processing | `apt install jq` / `brew install jq` |
| `timeout` | Per-run kill timeout | Built into GNU coreutils (Linux); `brew install coreutils` on macOS |
| `uuidgen` | Unique run IDs | Pre-installed on macOS; `apt install uuid-runtime` on Linux |
| `nvidia-smi` | NVIDIA GPU detection and thermal monitoring | Included with NVIDIA drivers |
| `sensors` | Linux CPU temperature reading | `apt install lm-sensors` |
| `osx-cpu-temp` | macOS CPU temperature reading (optional) | `brew install osx-cpu-temp` |

`nvidia-smi`, `sensors`, and `osx-cpu-temp` are only needed for thermal monitoring. If they are absent, llamaseye disables the thermal guard for that sensor and logs a warning — the sweep still runs.

---

## Key flags

| Flag | Description |
|------|-------------|
| `--model <path>` | Single GGUF model to benchmark |
| `--models-dir <dir>` | Directory to scan for GGUF models |
| `--model-list <file>` | Text file listing model filenames (one per line) |
| `--output-dir <dir>` | Root directory for all results (default: `./results`) |
| `--llama-bench <path>` | Path to standard llama-bench binary |
| `--turbo-bench <path>` | Path to TurboQuant llama-bench binary (enables turbo KV types) |
| `--ngl-step <n>` | Step size for NGL axis sweep (default: 5) |
| `--repetitions <n>` | Repetitions per benchmark run (default: 3) |
| `--timeout <s>` | Per-run timeout in seconds (default: 300) |
| `--resume` | Resume a previous sweep, skipping completed runs |
| `--overwrite` | Ignore existing state and re-run everything |
| `--only-phases <list>` | Comma-separated list of phase numbers to run (e.g. `0,1,7`) |
| `--skip-phases <list>` | Comma-separated list of phase numbers to skip |
| `--dry-run` | Print what would run without executing |
| `--no-confirm` | Skip the pre-run confirmation prompt |
| `--cpu-temp-limit <°C>` | Pause if CPU exceeds this temperature (default: 85) |
| `--gpu-temp-limit <°C>` | Pause if GPU exceeds this temperature (default: 80) |

---

## Sweep phases

| Phase | Name | What varies | Everything else |
|-------|------|-------------|-----------------|
| 0 | **NGL Probe** | ngl from 0 → max in coarse steps | Defaults — establishes max viable ngl |
| 1 | **NGL Axis** | ngl fine-grained around Phase 0 optimum | Defaults |
| 2 | **FA + KV Quant Axis** | flash attention on/off × KV cache type | Best ngl from Phase 1 |
| 3 | **Thread Count** | CPU thread count variants | Best ngl, best FA/KV |
| 4 | **KV Offload** | KV cache offload ratio 0.0 → 1.0 | Best ngl, best FA/KV, best threads |
| 5 | **Batch Size** | ubatch and batch size variants | Best values so far |
| 6 | **Context Ceiling** | Context size scaled up to OOM/timeout | Best values so far |
| 7 | **Full Combination Matrix** | Cartesian product of all best-per-axis values | — |

---

## TurboQuant KV cache types

Passing `--turbo-bench <path>` enables three additional KV cache quantisation types: **turbo2**, **turbo3**, and **turbo4**, sourced from the [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork. These compress the KV cache 3–6× compared to f16, freeing VRAM for more layers or larger contexts without a significant quality penalty.

The TurboQuant binary is verified at startup (a quick `--help` probe). If the path is missing or returns an error, turbo KV types are silently omitted and the sweep continues with the standard KV type set. It is safe to always pass `--turbo-bench` — the script handles an invalid path gracefully.

---

## Output structure

Results are written to `<output-dir>/<model-stem>/`:

```
results/
└── Qwen3-14B-Q4_K_M/
    ├── sweep.jsonl       # One JSON object per completed run
    ├── sweep.md          # Human-readable Markdown summary table
    ├── sweep.log         # Full execution log
    ├── hardware.json     # Hardware snapshot captured at start
    ├── state.json        # Resume state (completed run IDs + best values)
    └── raw/
        └── <run-id>.txt  # Raw llama-bench stdout for each run
```

`sweep.jsonl` is append-only and is the source of truth. `state.json` tracks which phases are complete and the best parameter values discovered so far, enabling `--resume` to pick up exactly where it left off.

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

Each phase sweeps exactly one axis while holding everything else at sane defaults. The only cross-phase dependency is `max_ngl`, which is established by Phase 0 and used as the ceiling for all subsequent phases. Phases 1–6 each produce one "best value" for their axis. Phase 7 takes the Cartesian product of all best-per-axis values and runs the full combination matrix, confirming which configs compose well and revealing the true peak configuration. This one-variable-at-a-time discipline keeps results interpretable and makes it straightforward to re-run individual phases in isolation with `--only-phases`.

---

## Docs

See `docs/spec.md` for the full engineering specification.

# llama-bench Sweep Program — Engineering Spec

A fully portable, model-agnostic benchmark harness that detects available
hardware at runtime, independently characterises every meaningful parameter
axis for a given GGUF model, then exhaustively tests every working combination
to reveal the full capability frontier of whatever machine it is running on.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Hardware Context & Constraints](#hardware-context--constraints)
3. [Hardware Detection](#hardware-detection)
4. [TurboQuant Binary Detection](#turboquant-binary-detection)
5. [Parameter Axes & Default Values](#parameter-axes--default-values)
6. [Sweep Phases Overview](#sweep-phases-overview)
7. [Phase 0 — NGL Probe](#phase-0--ngl-probe)
8. [Phase 1 — NGL Axis Sweep](#phase-1--ngl-axis-sweep)
9. [Phase 2 — Flash Attention + KV Cache Quant Axis Sweep](#phase-2--flash-attention--kv-cache-quant-axis-sweep)
10. [Phase 3 — Thread Count Axis Sweep](#phase-3--thread-count-axis-sweep)
11. [Phase 4 — KV Offload Axis Sweep](#phase-4--kv-offload-axis-sweep)
12. [Phase 5 — Batch & Micro-Batch Axis Sweep](#phase-5--batch--micro-batch-axis-sweep)
13. [Phase 6 — Context Size Axis Sweep](#phase-6--context-size-axis-sweep)
14. [Phase 7 — Full Combination Matrix](#phase-7--full-combination-matrix)
15. [Runnable Config Definition](#runnable-config-definition)
16. [OOM Detection & Handling](#oom-detection--handling)
17. [Thermal Throttle Handling](#thermal-throttle-handling)
18. [Output Format & File Layout](#output-format--file-layout)
19. [Script Architecture](#script-architecture)
20. [Invocation](#invocation)
21. [Implementation Notes & Gotchas](#implementation-notes--gotchas)

---

## Design Philosophy

Each phase sweeps **exactly one axis** while holding every other parameter at a
fixed default. Phases do not carry forward "winners" — each phase is a clean,
independent characterisation of one dimension of the parameter space.

The **only** information propagated from one phase to the next is `max_ngl`
(established in Phase 0), because it is impossible to know the upper bound for
ngl without probing it first. It is infrastructure, not an optimisation choice.

After all single-axis phases complete, **Phase 7** takes the set of working
values discovered for each axis and runs the full cartesian product of all of
them together. This is where you find out which combinations of parameters
actually work simultaneously, and which combinations hit limits that no single-
axis sweep would have revealed.

This approach gives you two distinct bodies of knowledge:
- **Per-axis characterisation** (Phases 1–6): how does each knob independently
  affect throughput and what values are viable?
- **Combination frontier** (Phase 7): which configs work together and what is
  the actual performance at each valid intersection?

---

## Hardware Detection

Before any sweep begins, the script runs a `detect_hardware()` function that
interrogates the host machine and populates a set of hardware variables used
to drive sweep decisions. This makes the script fully portable — it does not
hardcode any machine-specific values.

### What is detected

```
HW_CPU_MODEL        Human-readable CPU name string
HW_CPU_PHYSICAL     Number of physical (non-HT) cores
HW_CPU_LOGICAL      Number of logical threads (physical × HT factor)
HW_RAM_GIB          Total system RAM in GiB (rounded down)
HW_RAM_FREE_GIB     Free + reclaimable RAM at sweep start, in GiB
HW_GPU_COUNT        Number of detected NVIDIA GPUs (via nvidia-smi)
HW_GPU_MODEL        GPU name string (first GPU)
HW_GPU_VRAM_GIB     Total VRAM in GiB (first GPU)
HW_GPU_VRAM_FREE_GIB  Free VRAM at sweep start in GiB (first GPU)
HW_GPU_TEMP_LIMIT   GPU thermal pause threshold (default: 81 °C — overridable)
HW_CPU_TEMP_LIMIT   CPU thermal pause threshold (default: 88 °C — overridable)
HW_BACKEND          Detected compute backend: "cuda", "metal", "cpu"
```

### How each value is read

| Variable | Command |
|----------|---------|
| `HW_CPU_MODEL` | `lscpu \| grep "Model name"` (Linux) / `sysctl -n machdep.cpu.brand_string` (macOS) |
| `HW_CPU_PHYSICAL` | `lscpu \| grep "Core(s) per socket"` × sockets (Linux) / `sysctl -n hw.physicalcpu` (macOS) |
| `HW_CPU_LOGICAL` | `nproc --all` (Linux) / `sysctl -n hw.logicalcpu` (macOS) |
| `HW_RAM_GIB` | `free -g \| awk '/Mem:/{print $2}'` (Linux) / `sysctl -n hw.memsize` ÷ 2³⁰ (macOS) |
| `HW_RAM_FREE_GIB` | `free -g \| awk '/Mem:/{print $4+$6}'` (Linux) |
| `HW_GPU_MODEL` | `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` |
| `HW_GPU_VRAM_GIB` | `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0` ÷ 1024 |
| `HW_GPU_VRAM_FREE_GIB` | `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0` ÷ 1024 |
| `HW_GPU_COUNT` | `nvidia-smi --query-gpu=name --format=csv,noheader \| wc -l` |
| `HW_BACKEND` | `cuda` if nvidia-smi exits 0; `metal` if `system_profiler SPDisplaysDataType` shows GPU on macOS; else `cpu` |

If `nvidia-smi` is not found or exits non-zero, `HW_GPU_COUNT=0`,
`HW_GPU_VRAM_GIB=0`, `HW_BACKEND=cpu`. The sweep continues in CPU-only mode.

### How detected values drive sweep decisions

Hardware detection results feed directly into parameter axis construction:

| Detected value | Effect on sweep |
|----------------|----------------|
| `HW_CPU_PHYSICAL` | Thread sweep values capped at `HW_CPU_LOGICAL`; physical core count always included as an explicit test point |
| `HW_GPU_VRAM_GIB` | Sets the NGL probe starting expectation. If `HW_GPU_VRAM_GIB < 8`, probe starts at a lower initial ngl estimate to save time |
| `HW_GPU_COUNT == 0` | All FA and KV-GPU-offload phases still run, but `ngl` is forced to `0` and Phase 0 probe is skipped |
| `HW_RAM_GIB` | Informs whether large-context + CPU-offload combinations are worth probing in Phase 7 |
| `HW_BACKEND` | Selects appropriate thermal sensor commands (e.g. `sensors` on Linux/CUDA vs `powermetrics` on macOS/Metal) |

### Hardware snapshot in output

`detect_hardware()` writes a `hardware.json` file into the output directory
alongside `sweep.jsonl`. Every JSONL record also embeds a `hardware` key with
a snapshot of these values so results are always self-describing.

```json
{
  "cpu_model": "AMD Ryzen 7 5800X 8-Core Processor",
  "cpu_physical_cores": 8,
  "cpu_logical_threads": 16,
  "ram_gib": 32,
  "ram_free_gib_at_start": 18,
  "gpu_count": 1,
  "gpu_model": "NVIDIA GeForce RTX 3080",
  "gpu_vram_gib": 12,
  "gpu_vram_free_gib_at_start": 11,
  "backend": "cuda"
}
```

### Thermal sensor detection

The thermal monitoring commands vary by OS and installed tools:

| OS | CPU temp source | GPU temp source |
|----|----------------|----------------|
| Linux + lm-sensors | `sensors \| grep Tctl` or `grep Package` | `nvidia-smi --query-gpu=temperature.gpu` |
| Linux (no sensors) | `/sys/class/hwmon/hwmon*/temp*_input` (mV → °C) | same |
| macOS | `sudo powermetrics --samplers smc -n 1` | `nvidia-smi` (if CUDA GPU present) |

`detect_hardware()` probes which commands are available and sets:
```
HW_CPU_TEMP_CMD     shell snippet that outputs a single integer (°C)
HW_GPU_TEMP_CMD     shell snippet that outputs a single integer (°C)
```
`wait_cool()` calls these snippets directly, making it portable across machines.
If neither command is available, thermal guarding is disabled with a warning.

---

## TurboQuant Binary Detection

The sweep optionally supports a second `llama-bench` binary built from the
[llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork
(branch `feature/turboquant-kv-cache`). This fork adds `turbo2`, `turbo3`, and
`turbo4` as additional KV cache types, compressing the KV cache to 2–4 bits
on-the-fly without touching model weights. The primary benefit is dramatically
extended context on VRAM-limited hardware (~4–6× longer context for ~5–8% t/s
penalty).

### Why a separate binary

The turbo KV types are not in upstream llama.cpp — they exist only on the fork
branch. The binary is otherwise API-identical: every standard `llama-bench` flag
still works exactly the same. When the turbo binary is provided, only runs that
use a turbo KV type are dispatched to it; every other run uses the standard
binary as normal.

### Verification at startup

`detect_turbo_binary()` runs once at startup if `SWEEP_TURBO_BENCH_BIN` is set
or `--turbo-bench` is passed. It must pass all three checks:

```
function detect_turbo_binary():
    if SWEEP_TURBO_BENCH_BIN is not set:
        TURBO_AVAILABLE = false
        return

    if not file_exists(SWEEP_TURBO_BENCH_BIN) or not executable:
        warn "Turbo binary not found or not executable: {SWEEP_TURBO_BENCH_BIN}"
        TURBO_AVAILABLE = false
        return

    help_output = run: {SWEEP_TURBO_BENCH_BIN} --help 2>&1

    if help_output does not contain "turbo3":
        warn "Binary at {SWEEP_TURBO_BENCH_BIN} does not support turbo cache types."
        warn "Ensure it was built from: github.com/TheTom/llama-cpp-turboquant"
        warn "Branch: feature/turboquant-kv-cache  (master branch has no TurboQuant)"
        TURBO_AVAILABLE = false
        return

    TURBO_AVAILABLE = true
    log "[TURBO] TurboQuant binary verified: {SWEEP_TURBO_BENCH_BIN}"
    log "[TURBO] Turbo KV types enabled: turbo2, turbo3, turbo4"
```

If `TURBO_AVAILABLE=false`, all turbo combinations are silently excluded from
Phase 2 and Phase 7 and the sweep continues unchanged.

### Turbo KV cache types reference

| Type | Bits | KV compression vs f16 | t/s penalty | Notes |
|------|------|-----------------------|-------------|-------|
| `turbo4` | 4 | ~3.2× | minimal | Quality-sensitive tasks |
| `turbo3` | 3 | ~4.3× | ~5–8% | **Recommended default** |
| `turbo2` | 2 | ~6.4× | noticeable | Maximum context, higher degradation |

The compression ratio directly multiplies available context. On a 12 GB GPU
with a 8.5 GB model (~3.5 GB VRAM headroom at f16), `turbo3` extends context
from ~16K to approximately ~80K.

### Binary selection rule

`run_bench()` inspects the `ctk` parameter and picks the binary automatically:

```
function select_binary(ctk):
    if ctk in ["turbo2", "turbo3", "turbo4"]:
        return SWEEP_TURBO_BENCH_BIN
    else:
        return LLAMA_BENCH_BIN
```

The selected binary path is stored in every JSONL record under the `"binary"`
field so results from the two binaries are always distinguishable.

---

## Hardware Context & Constraints

The values below are **not hardcoded** — they are the expected values for the
primary target machine and serve as documentation only. At runtime, all of these
are sourced from `detect_hardware()`.

| Resource | Reference value | Notes |
|----------|----------------|-------|
| VRAM (usable) | ~11.2 GiB | Hard ceiling for GPU-offloaded layers + KV cache |
| System RAM | 32 GB DDR4-3200 | Available for CPU layers, KV overflow, system |
| CPU | Ryzen 7 5800X — 8C/16T | 8 physical cores; HT rarely helps inference |
| CPU thermal limit | 90 °C (`Tctl`) | Script pauses at `HW_CPU_TEMP_LIMIT` (default 88) |
| GPU thermal limit | 83 °C | Script pauses at `HW_GPU_TEMP_LIMIT` (default 81) |
| Viable interactive speed | ≥ 2.0 t/s TG | Configs below this are recorded but flagged non-viable |

---

## Parameter Axes & Default Values

Every phase holds all axes at their **sweep default** except the one being swept.
Sweep defaults are conservative, well-understood values that produce a reliable
baseline — they are not necessarily the fastest config.

| Axis | Sweep Default | Values Tested in Its Phase |
|------|--------------|---------------------------|
| `ngl` (GPU layers) | `max_ngl` | `0, 4, 8, ... max_ngl` (step 4) |
| `fa` (Flash Attention) | `0` | `0, 1` |
| `ctk`/`ctv` (KV quant) | `f16` / `f16` | `f16, q8_0, q4_0` (matched pairs) |
| `threads` | system default (omit `-t`) | `1, 2, 4, HW_CPU_PHYSICAL, HW_CPU_LOGICAL` + midpoints |
| `nkvo` (KV offload disable) | `0` | `0, 1` |
| `b` (batch size) | `2048` | `512, 1024, 2048` |
| `ub` (ubatch size) | `512` | `128, 256, 512` |
| `n_prompt` / context | `512` | `128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072` |


### Axis Start Points & Sweep Direction

Every axis supports two optional override flags:

- **`--start-<axis> <value>`** — begin the sweep at this value instead of the default list start. Values before the start point in the given direction are skipped entirely.
- **`--<axis>-dir <up|down>`** — control which direction the sweep runs from the start point.

| Axis | Direction "up" means | Direction "down" means | Default |
|------|---------------------|----------------------|---------|
| `ngl` | `0 → max_ngl` | `max_ngl → 0` | `up` |
| `threads` | `1 → HW_CPU_LOGICAL` | `HW_CPU_LOGICAL → 1` | `up` |
| `ctx` | `128 → 131072` | `131072 → 128` | `up` |
| `ctk` | toward more compression: `f16→q8_0→q4_0→turbo4→turbo3→turbo2` | toward less compression | `up` |
| `b` (batch) | `512 → 2048` | `2048 → 512` | `up` |
| `ub` (ubatch) | `128 → 512` | `512 → 128` | `up` |
| `fa` | `0 → 1` (off → on) | `1 → 0` (on → off) | `up` |

These flags are independent per axis and can be mixed freely:

```sh
# Start ngl sweep at 40, going down (tests 40, 36, 32, ... 0)
llamaseye.sh --model model.gguf --start-ngl 40 --ngl-dir down

# Start context sweep at 8k going up (skips 128/512/1k/2k/4k)
llamaseye.sh --model model.gguf --start-ctx 8192 --ctx-dir up

# Start KV quant sweep at q8_0 going up (tests q8_0, q4_0, turbo4, turbo3, turbo2)
llamaseye.sh --model model.gguf --start-ctk q8_0 --ctk-dir up

# Start threads at 8 going down (tests 8, 6, 4, 2, 1)
llamaseye.sh --model model.gguf --start-threads 8 --threads-dir down
```

**Behaviour when start value is not in the list:** A warning is logged and the full list in the given direction is used — the sweep never aborts due to a missing start point.

### Phase 7 Working Set Inheritance

Phase 7 **always uses exactly the values that phases 1–6 actually tested** — not the full possible list of values. This means `--start-*` and `--*-dir` flags naturally narrow Phase 7 as well, because they narrow what phases 1–6 discover.

For example:
- `--start-ctk q8_0 --ctk-dir up` → Phase 2 only tests `q8_0`, `f16`, and turbo types. Phase 7 inherits exactly that set — `f16` and `q4_0` never appear in the matrix.
- `--start-ctx 65536 --ctx-dir up` → Phase 6 only discovers `65536` and `131072`. Phase 7 never combines smaller contexts.

This means you often **don't need `--min-*` flags at all** — just set your start points correctly and Phase 7 follows automatically.

`--min-*` flags exist as an additional explicit filter for cases where you want phases 1–6 to still run full discovery (for the per-axis data) but want Phase 7 to only combine a specific subset:

| Flag | Filters from Phase 7 | Threshold |
|------|----------------------|-----------|
| `--min-ngl N` | ngl values < N | numeric |
| `--min-threads N` | thread counts < N | numeric |
| `--min-ctx N` | context sizes < N | numeric |
| `--min-ctk TYPE` | KV types below TYPE in quality order | ordered |
| `--min-b N` | batch sizes < N | numeric |
| `--min-ub N` | ubatch sizes < N | numeric |

**KV quality order** (low → high): `turbo2 → turbo3 → turbo4 → q4_0 → q8_0 → f16`

`--min-ctk q8_0` keeps only `q8_0` and `f16` in Phase 7, dropping all turbo types and `q4_0`.

Thread sweep values are generated dynamically from `HW_CPU_PHYSICAL` and
`HW_CPU_LOGICAL` at runtime. For example, on an 8C/16T machine the list
becomes: `1, 2, 4, 6, 8, 12, 16`. On a 6C/12T machine: `1, 2, 4, 6, 12`.
Always include `1`, `HW_CPU_PHYSICAL`, and `HW_CPU_LOGICAL` as explicit points.

**Standard workload** used in Phases 1–6 unless the axis being swept is prompt
size: `-p 512 -n 128 -r 3`

**FA + KV quant constraint:** `ctk=q4_0`/`ctv=q4_0` requires `fa=1` in most
llama.cpp builds. The combination `fa=0, ctk=q4_0` must be skipped silently
rather than tested.

---

## Sweep Phases Overview

```
Phase 0:  NGL Probe          → establishes max_ngl (only shared output)
Phase 1:  NGL Axis           → all ngl values, defaults elsewhere
Phase 2:  FA + KV Quant Axis → all fa/ctk/ctv combos, defaults elsewhere
Phase 3:  Thread Axis        → all thread counts, defaults elsewhere
Phase 4:  KV Offload Axis    → nkvo=0 and nkvo=1, defaults elsewhere
Phase 5:  Batch Axis         → all b/ub combos, defaults elsewhere
Phase 6:  Context Axis       → all context sizes up to 128k, defaults elsewhere
Phase 7:  Combination Matrix → cartesian product of all working values
```

Every phase after Phase 0 is **independent** — each one begins from the same
fixed defaults. Running Phase 3 tells you nothing about Phase 2 and vice versa.

---

## Phase 0 — NGL Probe

**Goal:** Find `max_ngl` — the highest `-ngl` value at which this model loads
and runs without OOM on this hardware. This is the only result that is carried
forward to all other phases.

**Why this is necessary:** llama-bench will hang or crash if you request more
GPU layers than VRAM can hold. `max_ngl` bounds all subsequent sweeps.

**Algorithm:**

```
ngl = 99
while ngl >= 0:
    run: llama-bench -m {model} -ngl {ngl} -p 64 -n 0 -r 1 --progress
    if SUCCESS:
        max_ngl = ngl
        log "max_ngl = {ngl}"
        break
    else (OOM or error):
        log "ngl={ngl} → OOM, stepping down"
        ngl = ngl - 4
if ngl < 0:
    ABORT: "Model cannot be loaded at any ngl value on this hardware."
```

Use `-p 64 -n 0 -r 1`: a minimal prompt-only single-rep run. Enough to confirm
the model loads; speed and accuracy do not matter here.

**Output:** `max_ngl` integer. Written to the sweep state file and used as the
`ngl` sweep default in all subsequent phases.

---

## Phase 1 — NGL Axis Sweep

**Goal:** Measure PP and TG t/s at every meaningful GPU layer count from
fully CPU-bound (`ngl=0`) to fully GPU-offloaded (`ngl=max_ngl`).

**What this reveals:** The relationship between layer offload and throughput.
For partial-offload models this shows the exact t/s cliff and helps identify
useful operating points (e.g. "ngl=40 is 90% as fast as ngl=max but uses 2 GiB
less VRAM").

**Fixed parameters (sweep defaults):**

```
fa=0, ctk=f16, ctv=f16, nkvo=0, threads=system-default, b=2048, ub=512
-p 512 -n 128 -r 3
```

**Swept values:**

Build the ngl list as:
```
[0] + [4, 8, 12, ..., max_ngl - (max_ngl % 4)] + [max_ngl]
```
Deduplicate and sort ascending. Always include `0` and `max_ngl` explicitly.

**OOM handling:** If any ngl value OOMs mid-sweep (can happen if KV cache fills
even though the model loaded in the probe), mark it skipped. Do NOT skip higher
values — a mid-sweep OOM does not invalidate higher ngl because the probe
already confirmed max_ngl works. Just skip that specific value and continue.

**Records written:** One JSONL record per ngl value tested.

---

## Phase 2 — Flash Attention + KV Cache Quant Axis Sweep

**Goal:** Characterise every valid combination of Flash Attention on/off and
KV cache quantization type.

**What this reveals:** Whether FA improves or hurts throughput on this model,
and how aggressively the KV cache can be quantized without breaking the run.
These two parameters are coupled (q4_0 requires FA) so they are swept together
as a combined axis.

**Fixed parameters (sweep defaults):**

```
ngl=max_ngl, nkvo=0, threads=system-default, b=2048, ub=512
-p 512 -n 128 -r 3
```

**Swept combinations (valid only):**

| Label | `-fa` | `-ctk` | `-ctv` | Valid? |
|-------|-------|--------|--------|--------|
| `fa0_f16` | `0` | `f16` | `f16` | ✅ Always |
| `fa1_f16` | `1` | `f16` | `f16` | ✅ Always |
| `fa0_q8` | `0` | `q8_0` | `q8_0` | ✅ Usually |
| `fa1_q8` | `1` | `q8_0` | `q8_0` | ✅ Always |
| `fa1_q4` | `1` | `q4_0` | `q4_0` | ✅ Always (requires fa=1) |
| `fa0_q4` | `0` | `q4_0` | `q4_0` | ❌ Skip — invalid combo |
| `fa0_turbo4` | `0` | `turbo4` | `turbo4` | ⚙️ Turbo binary required |
| `fa1_turbo4` | `1` | `turbo4` | `turbo4` | ⚙️ Turbo binary required |
| `fa0_turbo3` | `0` | `turbo3` | `turbo3` | ⚙️ Turbo binary required |
| `fa1_turbo3` | `1` | `turbo3` | `turbo3` | ⚙️ Turbo binary required |
| `fa0_turbo2` | `0` | `turbo2` | `turbo2` | ⚙️ Turbo binary required |
| `fa1_turbo2` | `1` | `turbo2` | `turbo2` | ⚙️ Turbo binary required |

Turbo rows (⚙️) are only included when `TURBO_AVAILABLE=true`. Test both
`fa=0` and `fa=1` for each turbo type — their FA compatibility is not guaranteed
and may vary by model architecture. Treat any crash or OOM as a skipped config
and continue.

Run standard rows first (in the order listed), then turbo rows. If all `fa=1`
configs OOM or crash (common with some MoE architectures), log a warning but
continue with `fa=0` variants.

**OOM handling:** Each combination is independent. An OOM on one does not
skip others.

**Records written:** One JSONL record per combination tested.

---

## Phase 3 — Thread Count Axis Sweep

**Goal:** Find how CPU thread count affects throughput across the full useful
range — including whether hyperthreading (threads > physical cores) helps or
hurts.

**What this reveals:** The optimal thread count for this model on this CPU.
On the Ryzen 5800X (8 physical cores / 16 logical), the answer is usually 8
but can vary. Also reveals the cost of too many threads (contention degrades
throughput).

**Fixed parameters (sweep defaults):**

```
ngl=max_ngl, fa=0, ctk=f16, ctv=f16, nkvo=0, b=2048, ub=512
-p 512 -n 128 -r 3
```

**Swept values:** `1, 2, 4, 6, 8, 12, 16`

Also include one run with **no `-t` flag at all** (label: `system_default`) so
the result can be compared against llama-bench's own default selection.

**Note on ngl:** Even at `ngl=max_ngl` where the GPU does most work, thread
count still matters for CPU-side operations (norm layers, some ops, KV
management). Always run the full thread sweep regardless of ngl.

**OOM handling:** Thread count does not affect memory usage. OOM here is
unexpected and should be logged as an error rather than skipped silently.

**Records written:** One JSONL record per thread count value tested, plus one
for `system_default`.

---

## Phase 4 — KV Offload Axis Sweep

**Goal:** Determine whether disabling GPU KV offload (`-nkvo 1`) — forcing the
KV cache into CPU RAM instead of VRAM — changes throughput or allows the model
to run in configurations that would otherwise OOM.

**What this reveals:** At `ngl=max_ngl`, the GPU holds both model weights and
the KV cache. With `-nkvo 1`, the KV cache moves to RAM, which may free enough
VRAM to raise `max_ngl` further, or may slow down inference due to PCIe
bandwidth. Both outcomes are worth knowing.

**Fixed parameters (sweep defaults):**

```
ngl=max_ngl, fa=0, ctk=f16, ctv=f16, threads=system-default, b=2048, ub=512
-p 512 -n 128 -r 3
```

**Swept values:** `nkvo=0`, `nkvo=1`

Additionally, if `max_ngl < 99`, also test `nkvo=1` at `max_ngl+4`,
`max_ngl+8`, and `max_ngl+12` — these would have OOM'd without `-nkvo 1`
but may succeed when KV cache is in RAM. Stop extending upward on first OOM.

**Records written:** One JSONL record per (nkvo, ngl) combination tested.

---

## Phase 5 — Batch & Micro-Batch Axis Sweep

**Goal:** Find whether non-default batch and micro-batch sizes improve prompt
processing throughput (PP t/s). Batch size primarily affects the prefill phase.

**What this reveals:** Whether larger or smaller physical kernel dispatches
suit this model's architecture and the GPU's CUDA core count better than
the defaults.

**Fixed parameters (sweep defaults):**

```
ngl=max_ngl, fa=0, ctk=f16, ctv=f16, nkvo=0, threads=system-default
-p 512 -n 0 -r 3
```

Use `-n 0` (PP-only) because batch size has minimal effect on TG. Using PP-only
also keeps each run short.

**Swept combinations (all valid pairs where ub <= b):**

| `-b` | `-ub` | Label |
|------|-------|-------|
| `2048` | `512` | `b2048_ub512` (default) |
| `2048` | `256` | `b2048_ub256` |
| `2048` | `128` | `b2048_ub128` |
| `1024` | `512` | `b1024_ub512` |
| `1024` | `256` | `b1024_ub256` |
| `1024` | `128` | `b1024_ub128` |
| `512` | `256` | `b512_ub256` |
| `512` | `128` | `b512_ub128` |

**Records written:** One JSONL record per b/ub pair tested.

---

## Phase 6 — Context Size Axis Sweep

**Goal:** Find the maximum context (prompt length) the model can process before
OOM, and record how PP t/s degrades as context length grows. This sweep uses
`ngl=max_ngl` (which gives the *least* VRAM headroom for KV cache) — giving the
conservative ceiling. Phase 7 will explore lower ngl values that may unlock
larger contexts.

**What this reveals:** The cold context ceiling at default settings. Any
context size that succeeds here is guaranteed to work in Phase 7's combinations.

**Fixed parameters (sweep defaults):**

```
ngl=max_ngl, fa=0, ctk=f16, ctv=f16, nkvo=0, threads=system-default,
b=2048, ub=512
-n 0 -r 2
```

Use `-n 0` (PP-only) and `-r 2` to keep runtime bounded on large contexts.

**Swept context sizes (prompt token counts):**

`128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`

**Stop condition:** On first OOM, record the failing size as the ceiling
boundary and stop the sweep. Do not test larger sizes in this phase.

**Records written:** One JSONL record per context size attempted (including the
OOM record with `status: "oom"`).

---

## Phase 7 — Full Combination Matrix

**Goal:** Test every valid cartesian product of working values discovered in
Phases 1–6 simultaneously. This is the heart of the sweep — it reveals which
parameter combinations work *together* and what the actual performance is at
each valid intersection.

**What this reveals:**
- Combinations that work individually but OOM together (e.g. `ngl=60` works
  and `ctk=f16` with 32k context works, but `ngl=60 + ctk=f16 + 32k` OOMs
  because both compete for VRAM)
- Combinations that unlock capabilities (e.g. `ngl=20 + ctk=q4_0 + nkvo=1`
  reaches 128k context that no single-axis test would have found)
- The actual peak t/s and peak context ceiling for the model

### Building the combination set

From each phase, collect the set of values that produced `status: "ok"`:

```
ngl_values    = all ngl values from Phase 1 where status=ok
fa_ctk_combos = all (fa, ctk, ctv) combos from Phase 2 where status=ok
              # includes turbo combos when TURBO_AVAILABLE=true
thread_values = all thread counts from Phase 3 where status=ok
              + [system_default]
nkvo_values   = all nkvo values from Phase 4 where status=ok
b_ub_combos   = all (b, ub) pairs from Phase 5 where status=ok
ctx_values    = all context sizes from Phase 6 where status=ok
```

When turbo KV types are present in `fa_ctk_combos`, `select_binary()` is called
per-combination inside the matrix loop and the correct binary is used
automatically. Turbo combos participate fully in all ngl × thread × nkvo × b/ub
× ctx combinations — the goal is to find the full frontier including what turbo
types unlock at lower ngl values with very large contexts.

The full combination set is:

```
for ngl in ngl_values:
  for (fa, ctk, ctv) in fa_ctk_combos:
    for threads in thread_values:
      for nkvo in nkvo_values:
        for (b, ub) in b_ub_combos:
          for ctx in ctx_values:
            run: llama-bench -ngl {ngl} -fa {fa} -ctk {ctk} -ctv {ctv}
                             -t {threads} -nkvo {nkvo} -b {b} -ub {ub}
                             -p {ctx} -n 128 -r 3
```

**This is intentionally a large number of runs.** On a model where each axis
has 5–10 working values, the combination count can reach thousands. This is
expected and correct. The sweep is designed to run unattended overnight.

### Pruning rules for Phase 7

To prevent spending hours on known-impossible combinations, apply these pruning
rules as the matrix executes:

**ngl pruning:** If a specific (ngl, fa, ctk, ctv, nkvo) combination OOMs,
skip all larger context values for that combination (adding more context only
makes VRAM pressure worse). Do NOT skip other (fa, ctk, ctv, nkvo) combos at
the same ngl.

**Context pruning:** Track the max successful context per (ngl, ctk, nkvo)
triple. For a given triple, once an OOM is hit at context size C, skip all
C' > C for any fa/thread/b/ub variation of that triple (KV cache size is
determined by ngl, ctk, nkvo, and ctx — the others don't affect it).

**No other pruning.** Do not skip based on t/s performance. Even a slow
config may be useful to know about.

### Phase 7 workload

Use `-p {ctx} -n 128 -r 3` for all combination runs. This tests both PP and TG
throughput in a realistic combined workload and gives clean t/s numbers.

For the context sweep within the matrix (ctx values), use `-p {ctx} -n 0 -r 2`
(PP-only) to keep runtime bounded. Only at the winning configs — highest t/s
TG per ngl — add a full `-n 128` run.

---

## Runnable Config Definition

A run is classified as **ok** if all of:
1. `llama-bench` exits with code `0`
2. stderr contains none of the OOM/error strings (see next section)
3. At least one result row is present in stdout/jsonl output
4. TG avg_ts > 0 when `-n > 0`

A run is **viable for interactive use** if TG avg_ts >= 2.0 t/s. Configs below
this are recorded with `viable: false` — not skipped, just annotated.

---

## OOM Detection & Handling

Capture stderr to a temp file on every invocation. After the process exits,
scan for any of these strings (case-insensitive grep):

```
CUDA error
out of memory
cudaMalloc failed
failed to allocate
GGML_ASSERT
ggml_cuda
cudaMemcpy
Cannot allocate memory
Killed
Segmentation fault
bus error
terminate called
```

If any match is found, classify the run as `status: "oom"` regardless of exit
code, write a skipped record to JSONL, and apply the phase-appropriate pruning
rule from the sections above.

**Timeout:** If the `llama-bench` process has not exited within `SWEEP_TIMEOUT_SEC`
(default: 600 seconds), send SIGTERM, wait 5s, send SIGKILL. Record as
`status: "timeout"`.

**Never hang.** A timed-out or crashed run must never block the sweep. Log the
failure and move to the next configuration.

---

## Thermal Throttle Handling

Implement `wait_cool()`, called before every single run:

```
function wait_cool():
    loop:
        cpu_temp = sensors | parse Tctl
        gpu_temp = nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
        if cpu_temp < 88 AND gpu_temp < 81:
            return
        log "[THERMAL] CPU={cpu_temp}°C  GPU={gpu_temp}°C — waiting {SWEEP_COOL_POLL_SEC}s"
        sleep SWEEP_COOL_POLL_SEC
```

Thresholds are intentionally 2 °C below the hard limits (90 °C / 83 °C) so
cooling begins before hitting the limit, not at it.

Between runs, also sleep `SWEEP_DELAY_SEC` (default: 5) regardless of temperature,
passed as `--delay` to llama-bench as well. This gives a lightweight cooldown
for short runs that don't push thermals.

---

## Output Format & File Layout

### Directory structure

```
<output-dir>/
└── <model-stem>/
    ├── sweep.jsonl          ← one record per run (appended; never truncated)
    ├── sweep.md             ← full results table, written after each phase
    ├── sweep.log            ← timestamped human-readable run log
    ├── state.json           ← sweep state (max_ngl, phase progress, working sets)
    └── raw/
        └── <run-id>.txt     ← raw llama-bench stdout+stderr for every run
```

`<model-stem>` = model filename without path, without `.gguf`, spaces → `_`.

### state.json

Written after each phase completes. Allows a sweep to be resumed if
interrupted.

```json
{
  "model_path": "/path/to/model.gguf",
  "model_stem": "Qwen3-14B-Q4_K_M",
  "max_ngl": 99,
  "phases_complete": [0, 1, 2],
  "working_sets": {
    "ngl": [0, 4, 8, ..., 99],
    "fa_ctk_combos": [
      {"fa": 0, "ctk": "f16", "ctv": "f16"},
      {"fa": 1, "ctk": "f16", "ctv": "f16"},
      {"fa": 1, "ctk": "q8_0", "ctv": "q8_0"}
    ],
    "thread_values": [4, 6, 8, 12],
    "nkvo_values": [0, 1],
    "b_ub_combos": [
      {"b": 2048, "ub": 512},
      {"b": 1024, "ub": 256}
    ],
    "ctx_values": [128, 512, 1024, 2048, 4096, 8192]
  }
}
```

### JSONL record schema

One record per llama-bench invocation. Append only.

```json
{
  "run_id": "a3f9c1d2",
  "timestamp": "2025-01-01T00:00:00Z",
  "model_path": "/path/to/model.gguf",
  "model_stem": "Qwen3-14B-Q4_K_M",
  "phase": 7,
  "phase_label": "combination_matrix",
  "binary": "standard",
  "status": "ok",
  "viable": true,
  "params": {
    "ngl": 60,
    "fa": 1,
    "ctk": "q8_0",
    "ctv": "q8_0",
    "nkvo": 0,
    "threads": 8,
    "threads_is_default": false,
    "b": 2048,
    "ub": 512,
    "n_prompt": 8192,
    "n_gen": 128,
    "repetitions": 3
  },
  "results": [
    {
      "test": "pp",
      "n_prompt": 8192,
      "n_gen": 0,
      "avg_ts": 876.3,
      "stddev_ts": 8.1
    },
    {
      "test": "tg",
      "n_prompt": 0,
      "n_gen": 128,
      "avg_ts": 38.4,
      "stddev_ts": 0.3
    }
  ],
  "raw_output_file": "raw/a3f9c1d2.txt",
  "error_snippet": null
}
```

`status`: `"ok"` | `"oom"` | `"timeout"` | `"error"`  
`viable`: `true` if TG avg_ts >= 2.0, `false` if below, `null` if no TG run.  
`binary`: `"standard"` (used `LLAMA_BENCH_BIN`) or `"turboquant"` (used `SWEEP_TURBO_BENCH_BIN`).  
`threads_is_default`: `true` when no `-t` flag was passed (system default).  
`error_snippet`: first 400 characters of error output when status != `"ok"`.

### Markdown summary table

`sweep.md` is regenerated after each phase from `sweep.jsonl`. It contains
one table per phase with columns:

```
| Phase | Label | ngl | fa | ctk | threads | nkvo | b | ub | n_prompt | PP t/s | TG t/s | Viable | Status |
```

Sorted within each phase by TG t/s descending, OOM/timeout rows at the bottom.

Phase 7 gets its own section: **Combination Matrix Results**, with an additional
summary subsection: **Context Frontier** — a table of max successful context
size per (ngl, ctk, nkvo) triple.

```
### Context Frontier

| ngl | ctk   | nkvo | Max Context | PP t/s at ceiling |
|-----|-------|------|-------------|-------------------|
| 99  | f16   | 0    | 8192        | 654.2             |
| 99  | q8_0  | 0    | 16384       | 423.1             |
| 99  | q4_0  | 0    | 32768       | 311.8             |
| 60  | f16   | 0    | 16384       | 489.3             |
| 60  | q4_0  | 1    | 131072      | 187.4             |
| 0   | q4_0  | 1    | 131072      | 43.2              |
```

---

## Script Architecture

Single Bash script: `llamaseye.sh`

### Required external tools

| Tool | Purpose |
|------|---------|
| `llama-bench` | The benchmark binary |
| `nvidia-smi` | GPU temperature |
| `sensors` | CPU temperature (`lm-sensors` package) |
| `jq` | JSONL record construction |
| `timeout` | Per-run timeout |
| `uuidgen` | Run IDs (fallback: `date +%s%N | md5sum | head -c 8`) |

### Configuration variables

All overridable via environment variable or CLI flag. The script reads
environment variables first, then CLI flags override them, so both styles work.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BENCH_BIN` | `~/llama.cpp/build/bin/llama-bench` | Path to standard llama-bench binary |
| `SWEEP_TURBO_BENCH_BIN` | *(unset)* | Path to TurboQuant llama-bench binary (optional) |
| `SWEEP_MODELS_DIR` | *(required if no model flag)* | Directory to scan for `.gguf` files |
| `SWEEP_OUTPUT_DIR` | `~/Models/bench/sweep` | Root output directory |
| `SWEEP_NGL_STEP` | `4` | Layer step for ngl sweep |
| `SWEEP_REPETITIONS` | `3` | `-r` for main runs |
| `SWEEP_PROBE_REPS` | `1` | `-r` for Phase 0 probe |
| `SWEEP_TIMEOUT_SEC` | `600` | Per-run kill timeout (seconds) |
| `SWEEP_MIN_TG_TS` | `2.0` | Viable t/s threshold |
| `SWEEP_CPU_TEMP_LIMIT` | `88` | °C — pause sweep above this |
| `SWEEP_GPU_TEMP_LIMIT` | `81` | °C — pause sweep above this |
| `SWEEP_COOL_POLL_SEC` | `20` | Seconds between thermal polls |
| `SWEEP_DELAY_SEC` | `5` | Seconds between runs |
| `SWEEP_PRIO` | `2` | `--prio` passed to llama-bench |

### Function outline

```
main()
  parse_args()
  resolve_model_list()       # build list of .gguf paths to sweep (see below)
  detect_hardware()          # populate HW_* variables, write hardware.json
  detect_turbo_binary()      # validate SWEEP_TURBO_BENCH_BIN, set TURBO_AVAILABLE
  print_hardware_summary()   # print detected hardware + binary paths to terminal + log
  for each model in model_list:
    validate_model()
    setup_output_dir()
    load_state()             # resume if state.json exists and --resume set
    phase0_ngl_probe()       # → max_ngl
    phase1_ngl_sweep()       # → working_sets.ngl
    phase2_fa_kv_sweep()     # → working_sets.fa_ctk_combos
    phase3_thread_sweep()    # → working_sets.thread_values
    phase4_nkvo_sweep()      # → working_sets.nkvo_values
    phase5_batch_sweep()     # → working_sets.b_ub_combos
    phase6_ctx_sweep()       # → working_sets.ctx_values
    phase7_combination_matrix()
    write_markdown()
    print_summary()

detect_hardware()
  # Reads CPU model, physical/logical core counts, RAM, GPU VRAM, backend
  # Probes for nvidia-smi, sensors/lm-sensors, powermetrics
  # Sets HW_CPU_TEMP_CMD and HW_GPU_TEMP_CMD for portable thermal polling
  # Writes hardware.json to output dir

resolve_model_list()
  # Returns an ordered list of absolute .gguf paths to benchmark.
  # Priority (highest wins):
  #   1. --model <path>          → single model, ignore everything else
  #   2. --model-list <file>     → read paths/names one per line from file
  #   3. --models-dir <dir>      → glob dir for *.gguf, sort alphabetically
  #   4. SWEEP_MODELS_DIR env    → same as --models-dir
  # Names in --model-list that are not absolute paths are resolved
  # relative to --models-dir (or SWEEP_MODELS_DIR) if set.
  # Prints the resolved list to terminal before starting any sweep.

run_bench(params_assoc_array)
  build_cmd()                # construct llama-bench command from params
  select_binary()            # pick LLAMA_BENCH_BIN or SWEEP_TURBO_BENCH_BIN based on ctk
  wait_cool()                # thermal gate
  sleep SWEEP_DELAY_SEC
  execute with timeout       # stdout → raw/<run-id>.txt, stderr → tmp
  detect_oom()               # scan stderr for error strings
  parse_results()            # parse jsonl output from llama-bench
  write_jsonl_record()       # append to sweep.jsonl (includes hardware snapshot)
  return status              # "ok" | "oom" | "timeout" | "error"

wait_cool()
  loop until HW_CPU_TEMP_CMD < HW_CPU_TEMP_LIMIT
         and HW_GPU_TEMP_CMD < HW_GPU_TEMP_LIMIT
  poll every SWEEP_COOL_POLL_SEC

write_jsonl_record(...)
  jq -n '{...}' >> sweep.jsonl

write_markdown()
  jq + awk to render tables from sweep.jsonl → sweep.md

load_state() / save_state()
  jq read/write of state.json

print_summary()
  print top-5 TG t/s configs
  print context frontier table
  print total runs: ok / oom / timeout / error
```

---

## Invocation

### Model selection

The script accepts models in three ways. Use whichever fits the situation:

```sh
# 1. Single model by path
bash llamaseye.sh --model /path/to/model.gguf

# 2. All .gguf files in a directory (alphabetical order)
bash llamaseye.sh --models-dir ~/Models

# 3. A specific subset — provide a text file, one model name or path per line
bash llamaseye.sh --models-dir ~/Models --model-list ~/my_models.txt
```

The `--model-list` file format: one entry per line. Entries can be:
- An absolute path: `/home/user/Models/Qwen3-14B-Q4_K_M.gguf`
- A filename only: `Qwen3-14B-Q4_K_M.gguf` (resolved against `--models-dir`)
- Blank lines and lines starting with `#` are ignored (comments)

Example `my_models.txt`:
```
# Models to benchmark
Qwen3-14B-Q4_K_M.gguf
Qwen3-14B-Q6_K.gguf
# Qwen3-14B-Q8_0.gguf   ← skipped
Llama-3.1-8B-Q4_K_M.gguf
```

Models are always swept **serially** — never in parallel. Running two benchmarks
simultaneously on the same GPU produces meaningless results.

### Examples

```sh
# Sweep a single model
bash llamaseye.sh --model ~/Models/Qwen3-14B-Q4_K_M.gguf

# Sweep all models in a directory, output to custom dir
bash llamaseye.sh --models-dir ~/Models --output-dir ~/bench_results

# Sweep a curated list of models
bash llamaseye.sh --models-dir ~/Models --model-list ~/bench_list.txt

# Run unattended overnight, follow progress
nohup bash llamaseye.sh --models-dir ~/Models --output-dir ~/bench_results > /dev/null 2>&1 &
tail -f ~/bench_results/sweep.log

# Resume all incomplete sweeps in an output dir
bash llamaseye.sh --models-dir ~/Models --output-dir ~/bench_results --resume

# Run only specific phases (e.g. re-run Phase 6 and 7 for one model)
bash llamaseye.sh --model ~/Models/Qwen3-14B-Q4_K_M.gguf --only-phases 6,7

# Finer ngl step for a model right at the VRAM edge
bash llamaseye.sh --model ~/Models/Qwen3-14B-Q4_K_M.gguf --ngl-step 2

# Dry run — print hardware detection + all planned commands without executing
bash llamaseye.sh --models-dir ~/Models --dry-run
```

### CLI flags reference

```
Model selection (one required):
  --model <path>             Single .gguf file to sweep.
  --models-dir <dir>         Directory to scan for all .gguf files.
  --model-list <file>        Text file of model names/paths to sweep.
                             Use with --models-dir to resolve bare filenames.

Output:
  --output-dir <path>        Root output directory. Default: ~/Models/bench/sweep
                             Each model gets its own subdirectory inside.

Sweep control:
  --ngl-step <n>             NGL sweep step size. Default: 4.
  --timeout <sec>            Per-run kill timeout. Default: 600.
  --repetitions <n>          Repetitions per run (-r). Default: 3.
  --only-phases <n,n,...>    Run only the listed phase numbers.
  --skip-phases <n,n,...>    Skip the listed phase numbers.
  --resume                   Skip phases already marked complete in state.json.
  --overwrite                Delete existing output for a model and start fresh.
  --min-viable-ts <f>        Minimum TG t/s to mark viable. Default: 2.0.

Hardware / environment:
  --llama-bench <path>       Path to standard llama-bench binary.
                             Default: ~/llama.cpp/build/bin/llama-bench
  --turbo-bench <path>       Path to TurboQuant llama-bench binary (optional).
                             Enables turbo2/turbo3/turbo4 KV cache types in
                             Phase 2 and Phase 7. Must be built from:
                             github.com/TheTom/llama-cpp-turboquant
                             branch: feature/turboquant-kv-cache
                             Verified at startup — silently disabled if invalid.
  --cpu-temp-limit <n>       °C above which sweep pauses. Default: 88.
  --gpu-temp-limit <n>       °C above which sweep pauses. Default: 81.
  --no-thermal-guard         Disable wait_cool() entirely (not recommended).

Utility:
  --dry-run                  Detect hardware and print all planned commands
                             without executing any benchmarks.
  --no-confirm               Skip Phase 7 size confirmation prompt.
  Axis start points & directions:
  --start-ngl N         Begin ngl sweep at N (skips values before N in sweep direction).
  --ngl-dir up|down     ngl sweep direction: up=0→max, down=max→0. Default: up.
  --start-threads N     Begin thread sweep at N.
  --threads-dir up|down Thread sweep direction. Default: up.
  --start-ctx N         Begin context sweep at prompt size N.
  --ctx-dir up|down     Context sweep direction: up=128→131072, down=131072→128. Default: up.
  --start-ctk TYPE      Begin KV quant sweep at TYPE.
                        Full ordering (up=more compression):
                        f16 → q8_0 → q4_0 → turbo4 → turbo3 → turbo2
  --ctk-dir up|down     KV type sweep direction. Default: up.
  --start-b N           Begin batch size sweep at N.
  --b-dir up|down       Batch size direction: up=512→2048, down=2048→512. Default: up.
  --start-ub N          Begin ubatch size sweep at N.
  --ub-dir up|down      Ubatch size direction. Default: up.
  --start-fa 0|1        Begin FA sweep at 0 (off) or 1 (on). Default: 0.
  --fa-dir up|down      FA sweep direction: up=0->1, down=1->0. Default: up.

Phase 7 minimum thresholds (filter combination matrix inputs):
  --min-ngl N           Exclude ngl values below N from Phase 7.
  --min-threads N       Exclude thread counts below N from Phase 7.
  --min-ctx N           Exclude context sizes below N from Phase 7.
  --min-ctk TYPE        Exclude KV types below TYPE (quality order) from Phase 7.
                        Quality order lowâhigh: turbo2 turbo3 turbo4 q4_0 q8_0 f16
  --min-b N             Exclude batch sizes below N from Phase 7.
  --min-ub N            Exclude ubatch sizes below N from Phase 7.

  Tip: --start-* flags already narrow Phase 7 naturally. Use --min-* only when
  you want full per-axis discovery in phases 1-6 but a tighter Phase 7.

  -h, --help                 Print usage and exit.
```

---

## Implementation Notes & Gotchas

### Use `-o jsonl` for parsing, not markdown

llama-bench's `-o jsonl` flag emits one JSON object per test result to stdout.
This is far more reliable to parse than scraping the markdown table. Always
pass `-o jsonl` in run_bench. Save raw output to `raw/<run-id>.txt` so it can
be re-parsed if the schema changes.

### stdout vs stderr

llama-bench writes results to **stdout** and diagnostic/progress info to
**stderr**. Capture them independently:
```sh
llama-bench [flags] -o jsonl > stdout.tmp 2> stderr.tmp
```
Parse `stdout.tmp` for results, scan `stderr.tmp` for OOM strings.

### threads=system_default vs explicit threads

When `threads_is_default=true`, do **not** pass `-t` to llama-bench. This is
distinct from `-t 8` even if the system happens to have 8 physical cores.
llama-bench's own default selection may differ from 8 (e.g. it may use only
physical cores and detect this internally). Always include a `system_default`
run in Phase 3.

### FA crashes on MoE models

Some MoE architectures (DeepSeek-V2, Qwen3-MoE) crash with `-fa 1`. In Phase 2,
if all `fa=1` runs crash or OOM, log `"fa1_unusable": true` in `state.json` and
exclude `fa=1` from Phase 7's combination set. Do not abort the sweep.

### q4_0 KV cache requires fa=1

In all known llama.cpp builds, `-ctk q4_0 -ctv q4_0` requires `-fa 1`. The
`fa0_q4` combination is listed as invalid in Phase 2 and must never be
passed to llama-bench. The combination matrix in Phase 7 must enforce this
constraint when building the cartesian product.

### Idempotency and resumption

`sweep.jsonl` is append-only. If `--resume` is set and `state.json` exists,
load `phases_complete` and skip those phases. If `--overwrite` is set, delete
the model's output directory and start clean. Default behavior (no flag) is
to append and re-run all phases — useful if you want to collect more data
without discarding what's already there.

### Estimating Phase 7 size before running

Before Phase 7 begins, print an estimate:
```
Hardware: AMD Ryzen 7 5800X | 32 GiB RAM | RTX 3080 12 GiB | backend=cuda
Binaries: standard=/path/to/llama-bench  turbo=ENABLED (/path/to/turbo/llama-bench)

Phase 7 estimate:
  ngl values:     12
  fa/ctk combos:  4  (+6 turbo combos = 10 total with TurboQuant enabled)
  thread values:  6  (derived from 8 physical / 16 logical cores)
  nkvo values:    2
  b/ub combos:    5
  ctx values:     7
  ─────────────────────────────
  Total runs:     12 × 4 × 6 × 2 × 5 × 7 = 20,160
  Est. time @ 30s/run: ~168 hours

  With context pruning applied, expected actual runs: ~30–40% of total
```

Allow the user to confirm before proceeding, or pass `--no-confirm` to skip.

### Logging format

Every line in `sweep.log` must be timestamped:
```
[2025-01-01T12:00:00Z] [PHASE 1] ngl=40 fa=0 ctk=f16 → ok | PP=987.6 t/s | TG=42.1 t/s
[2025-01-01T12:00:35Z] [PHASE 1] ngl=44 fa=0 ctk=f16 → oom | skipped
[2025-01-01T12:00:35Z] [THERMAL] CPU=89°C GPU=79°C — waiting...
[2025-01-01T12:01:15Z] [THERMAL] Temps normalised. Resuming.
[2025-01-01T12:01:20Z] [PHASE 1] ngl=48 fa=0 ctk=f16 → ok | PP=1102.3 t/s | TG=47.8 t/s
[2025-01-01T12:02:10Z] [PHASE 1 COMPLETE] Working ngl values: [0,4,8,...,40,48,52,...,99]
```

### Context pruning key

The context OOM is determined by VRAM pressure from the KV cache. The KV cache
size depends on `ngl` (how many layers have KV state on GPU), `ctk`/`ctv` (bytes
per KV entry), `nkvo` (whether KV is on GPU or RAM), and `n_prompt` (cache size).
Thread count, FA flag, and batch size do not affect KV memory.

The pruning key for context OOM in Phase 7 is therefore:
```
(ngl, ctk, ctv, nkvo, n_prompt)
```
Not (fa, threads, b, ub). Use this key when memoising known-OOM context ceilings
to avoid redundant runs.
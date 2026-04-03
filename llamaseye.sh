#!/usr/bin/env bash
# =============================================================================
# llamaseye.sh — Exhaustive llama-bench parameter sweep harness
# =============================================================================
# Version : 0.1.0
# Purpose : Systematically benchmark every meaningful llama-bench parameter
#           combination for a given model (or set of models), recording results
#           as JSONL + Markdown and producing a final summary table.
#
# Phases
#   0  ngl_probe        — binary-search max stable NGL (skipped if GPU absent)
#   1  ngl_sweep        — sweep NGL from 0..MAX_NGL in SWEEP_NGL_STEP steps
#   2  fa_kv_sweep      — flash-attn × KV-type combos (+ turbo3 when available)
#   3  thread_sweep     — sweep CPU threads from 1 core up to logical CPU count
#   4  nkvo_sweep       — no-kv-offload combinations with best NGL
#   5  batch_sweep      — batch / ubatch size pairs
#   6  ctx_sweep        — context length 512..max-stable
#   7  combination_matrix — cartesian product of best candidates, pruned
#
# Usage : llamaseye.sh [OPTIONS]
# See   : llamaseye.sh --help
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION VARIABLES — override via environment or CLI flags
# =============================================================================

# Path to the llama-bench binary
LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN:-${HOME}/llama.cpp/build/bin/llama-bench}"

# Optional turbo-bench binary (supports turbo3 KV type)
SWEEP_TURBO_BENCH_BIN="${SWEEP_TURBO_BENCH_BIN:-}"

# Directory to scan for .gguf models when no explicit model is given
SWEEP_MODELS_DIR="${SWEEP_MODELS_DIR:-}"

# Root directory where all sweep output is written
SWEEP_OUTPUT_DIR="${SWEEP_OUTPUT_DIR:-${HOME}/Models/bench/sweep}"

# NGL granularity for phase1 sweep (layers per step)
SWEEP_NGL_STEP="${SWEEP_NGL_STEP:-4}"

# Number of llama-bench repetitions per data point (phase1+)
SWEEP_REPETITIONS="${SWEEP_REPETITIONS:-3}"

# Repetitions used for cheap probe runs (phase0)
SWEEP_PROBE_REPS="${SWEEP_PROBE_REPS:-1}"

# Wall-clock timeout (seconds) per individual llama-bench invocation
SWEEP_TIMEOUT_SEC="${SWEEP_TIMEOUT_SEC:-600}"

# Minimum acceptable token-generation throughput (t/s); below this = skip
SWEEP_MIN_TG_TS="${SWEEP_MIN_TG_TS:-2.0}"

# CPU temperature ceiling (°C) before wait_cool() pauses execution
SWEEP_CPU_TEMP_LIMIT="${SWEEP_CPU_TEMP_LIMIT:-88}"

# GPU temperature ceiling (°C) before wait_cool() pauses execution
SWEEP_GPU_TEMP_LIMIT="${SWEEP_GPU_TEMP_LIMIT:-81}"

# Polling interval (seconds) while waiting for thermals to settle
SWEEP_COOL_POLL_SEC="${SWEEP_COOL_POLL_SEC:-20}"

# Delay (seconds) between consecutive bench runs (allow GPU to breathe)
SWEEP_DELAY_SEC="${SWEEP_DELAY_SEC:-5}"

# ionice / nice priority class (2 = best-effort) to keep system responsive
SWEEP_PRIO="${SWEEP_PRIO:-2}"

# =============================================================================
# RUNTIME STATE — populated during execution, not user-configurable
# =============================================================================

TURBO_AVAILABLE=false         # true when SWEEP_TURBO_BENCH_BIN is valid
MAX_NGL=99                    # resolved max NGL for current model (phase0)

# Hardware inventory
HW_CPU_MODEL=""
HW_CPU_PHYSICAL=8
HW_CPU_LOGICAL=16
HW_RAM_GIB=0
HW_RAM_FREE_GIB=0
HW_GPU_COUNT=0
HW_GPU_MODEL=""
HW_GPU_VRAM_GIB=0
HW_GPU_VRAM_FREE_GIB=0
HW_BACKEND="cpu"             # cpu | cuda | metal | vulkan
HW_CPU_TEMP_CMD=""           # shell snippet that prints a single integer °C
HW_GPU_TEMP_CMD=""           # shell snippet that prints a single integer °C

# Per-model working state
MODEL_LIST=()                 # resolved list of absolute .gguf paths
MODEL_PATH=""                # current model absolute path
MODEL_STEM=""                # basename without extension
OUTPUT_MODEL_DIR=""          # ${SWEEP_OUTPUT_DIR}/${MODEL_STEM}/

# =============================================================================
# CLI FLAGS — set by parse_args()
# =============================================================================

OPT_RESUME=false              # --resume: skip already-completed phases
OPT_OVERWRITE=false           # --overwrite: nuke existing output dir
OPT_DRY_RUN=false             # --dry-run: print commands, do not execute
OPT_NO_CONFIRM=false          # --no-confirm: skip interactive confirmation
OPT_NO_THERMAL=false          # --no-thermal-guard: disable wait_cool()
OPT_ONLY_PHASES=""           # --only-phases: comma-separated phase numbers
OPT_SKIP_PHASES=""           # --skip-phases: comma-separated phase numbers
OPT_MODEL_LIST_FILE=""       # --model-list: path to file with one model/line

# --- Sweep axis start points (default: begin of list for each axis) ---
OPT_START_NGL=""              # --start-ngl N: begin ngl sweep at this value
OPT_START_THREADS=""          # --start-threads N: begin thread sweep at this value
OPT_START_CTX=""              # --start-ctx N: begin context sweep at this prompt size
OPT_START_CTK=""              # --start-ctk TYPE: begin KV quant sweep at this type
OPT_START_B=""                # --start-b N: begin batch sweep at this b value
OPT_START_UB=""               # --start-ub N: begin ubatch sweep at this ub value
OPT_START_FA=""               # --start-fa 0|1: begin FA sweep at this value

# --- Sweep axis directions ---
# "up"   = sweep from start toward the high end of the list
# "down" = sweep from start toward the low end of the list
# KV type ordering (low->high compression): f16 q8_0 q4_0 turbo4 turbo3 turbo2
OPT_DIR_NGL="up"              # --ngl-dir up|down     (up = 0->max_ngl)
OPT_DIR_THREADS="up"          # --threads-dir up|down (up = 1->HW_CPU_LOGICAL)
OPT_DIR_CTX="up"              # --ctx-dir up|down     (up = 128->131072)
OPT_DIR_CTK="up"              # --ctk-dir up|down     (up = toward more compression)
OPT_DIR_B="up"                # --b-dir up|down       (up = 512->2048)
OPT_DIR_UB="up"               # --ub-dir up|down      (up = 128->512)
OPT_DIR_FA="up"               # --fa-dir up|down      (up = 0->1)

# --- Phase 7 minimum thresholds (filter combination matrix inputs) ---
# These trim the per-axis working sets before Phase 7. For numeric axes,
# values strictly below the minimum are excluded. For ctk, types below
# the minimum in quality order are excluded.
# KV quality order (low->high): turbo2 turbo3 turbo4 q4_0 q8_0 f16
OPT_MIN_NGL=""                # --min-ngl N
OPT_MIN_THREADS=""            # --min-threads N
OPT_MIN_CTX=""                # --min-ctx N
OPT_MIN_CTK=""                # --min-ctk TYPE
OPT_MIN_B=""                  # --min-b N
OPT_MIN_UB=""                 # --min-ub N


# =============================================================================
# FUNCTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# log MESSAGE...
#   Write a UTC-timestamped message to stdout and append to sweep.log.
#   OUTPUT_MODEL_DIR must be set before calling log(); until then messages
#   go only to stdout.
# -----------------------------------------------------------------------------
log() {
    local msg="[S$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    if [[ -n "${OUTPUT_MODEL_DIR:-}" && -d "${OUTPUT_MODEL_DIR}" ]]; then
        echo "$msg" | tee -a "${OUTPUT_MODEL_DIR}/sweep.log"
    else
        echo "$msg"
    fi
}

# -----------------------------------------------------------------------------
# warn MESSAGE...
#   Like log() but prefixed with [WARN] so it stands out in the log.
# -----------------------------------------------------------------------------
warn() {
    log "[WARN] $*"
}

# -----------------------------------------------------------------------------
# die MESSAGE...
#   Log a fatal error and exit with status 1.
# -----------------------------------------------------------------------------
die() {
    log "[ERROR] $*"
    exit 1
}

# -----------------------------------------------------------------------------
# usage
#   Print full help text describing all flags, env vars, and phases, then exit.
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
llamaseye.sh v0.1.0 — exhaustive llama-bench parameter sweep harness

USAGE
  llamaseye.sh [OPTIONS]

MODEL SELECTION (at least one required)
  --model PATH          Benchmark a single .gguf file
  --models-dir DIR      Benchmark all .gguf files found directly inside DIR
  --model-list FILE     Benchmark models listed one-per-line in FILE

OUTPUT
  --output-dir DIR      Root directory for results  [${SWEEP_OUTPUT_DIR}]

BINARY PATHS
  --llama-bench PATH    llama-bench binary           [${LLAMA_BENCH_BIN}]
  --turbo-bench PATH    turbo-bench binary (optional, enables turbo3 KV tests)

SWEEP TUNING
  --ngl-step N          NGL step size for phase1     [${SWEEP_NGL_STEP}]
  --repetitions N       Bench reps per data point    [${SWEEP_REPETITIONS}]
  --timeout N           Per-run timeout in seconds   [${SWEEP_TIMEOUT_SEC}]

THERMAL GUARD
  --cpu-temp-limit N    CPU °C ceiling               [${SWEEP_CPU_TEMP_LIMIT}]
  --gpu-temp-limit N    GPU °C ceiling               [${SWEEP_GPU_TEMP_LIMIT}]
  --no-thermal-guard    Disable thermal polling entirely

EXECUTION CONTROL
  --resume              Skip phases already recorded in state.json
  --overwrite           Delete existing output dir before starting
  --only-phases LIST    Run only these phases (comma-separated, e.g. 0,2,5)
  --skip-phases LIST    Skip these phases (comma-separated)

Axis start points & directions:
  --start-ngl N         Begin ngl sweep at this value instead of 0 or max.
  --ngl-dir up|down     Sweep direction from start (default: up = 0->max_ngl).
  --start-threads N     Begin thread count sweep at this value.
  --threads-dir up|down Sweep direction (default: up = 1->HW_CPU_LOGICAL).
  --start-ctx N         Begin context sweep at this prompt size.
  --ctx-dir up|down     Sweep direction (default: up = 128->131072).
  --start-ctk TYPE      Begin KV quant sweep at this type.
                        Ordering (up=more compression): f16 q8_0 q4_0 turbo4 turbo3 turbo2
  --ctk-dir up|down     Sweep direction through KV type list (default: up).
  --start-b N           Begin batch size sweep at this value.
  --b-dir up|down       Sweep direction (default: up = 512->2048).
  --start-ub N          Begin ubatch size sweep at this value.
  --ub-dir up|down      Sweep direction (default: up = 128->512).
  --start-fa 0|1        Begin FA sweep at this value (0=off, 1=on). Default: 0.
  --fa-dir up|down      FA sweep direction: up=0->1, down=1->0. Default: up.

Phase 7 minimum thresholds:
  --min-ngl N           Exclude ngl values below N from Phase 7 matrix.
  --min-threads N       Exclude thread counts below N from Phase 7 matrix.
  --min-ctx N           Exclude context sizes below N from Phase 7 matrix.
  --min-ctk TYPE        Exclude KV types below TYPE (quality order) from Phase 7.
                        Quality order (low->high): turbo2 turbo3 turbo4 q4_0 q8_0 f16
                        e.g. --min-ctk q8_0 keeps only q8_0 and f16.
  --min-b N             Exclude batch sizes below N from Phase 7 matrix.
  --min-ub N            Exclude ubatch sizes below N from Phase 7 matrix.

  Note: Phase 7 always inherits the values actually tested in phases 1-6
  (which are already trimmed by --start-* and --*-dir flags). Use --min-*
  for additional explicit filtering on top of that.

  --dry-run             Print bench commands without executing them
  --no-confirm          Skip the pre-sweep confirmation prompt

  -h, --help            Show this help and exit

ENVIRONMENT VARIABLES
  Any configuration variable (e.g. SWEEP_NGL_STEP=8) can be set in the
  environment and will be picked up as the default, before CLI flags apply.

PHASES
  0  ngl_probe          Binary-search the max NGL that fits in VRAM
  1  ngl_sweep          Sweep NGL 0..MAX_NGL in steps of SWEEP_NGL_STEP
  2  fa_kv_sweep        Flash-attn × KV-type combinations
  3  thread_sweep       CPU thread count from 1..logical-count
  4  nkvo_sweep         No-KV-offload variants with optimal NGL
  5  batch_sweep        Batch / ubatch size pairs
  6  ctx_sweep          Context window sizes 512..max-stable
  7  combination_matrix Cartesian product of top candidates (pruned)

EXAMPLES
  # Benchmark one model with defaults
  llamaseye.sh --model ~/Models/mistral-7b-q4_k_m.gguf

  # Benchmark all models in a directory, resuming a prior run
  llamaseye.sh --models-dir ~/Models --output-dir ~/bench --resume

  # Dry-run only phases 0 and 2
  llamaseye.sh --model ~/Models/foo.gguf --only-phases 0,2 --dry-run
EOF
    exit 0
}

# -----------------------------------------------------------------------------
# parse_args ARGS...
#   Parse all CLI flags into the OPT_* and config variables.
#   Unknown flags cause a fatal error with a usage hint.
# -----------------------------------------------------------------------------
parse_args() {
    local model_explicit=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                [[ $# -lt 2 ]] && die "--model requires an argument"
                model_explicit="$2"; shift 2 ;;
            --models-dir)
                [[ $# -lt 2 ]] && die "--models-dir requires an argument"
                SWEEP_MODELS_DIR="$2"; shift 2 ;;
            --model-list)
                [[ $# -lt 2 ]] && die "--model-list requires an argument"
                OPT_MODEL_LIST_FILE="$2"; shift 2 ;;
            --output-dir)
                [[ $# -lt 2 ]] && die "--output-dir requires an argument"
                SWEEP_OUTPUT_DIR="$2"; shift 2 ;;
            --llama-bench)
                [[ $# -lt 2 ]] && die "--llama-bench requires an argument"
                LLAMA_BENCH_BIN="$2"; shift 2 ;;
            --turbo-bench)
                [[ $# -lt 2 ]] && die "--turbo-bench requires an argument"
                SWEEP_TURBO_BENCH_BIN="$2"; shift 2 ;;
            --ngl-step)
                [[ $# -lt 2 ]] && die "--ngl-step requires an argument"
                SWEEP_NGL_STEP="$2"; shift 2 ;;
            --repetitions)
                [[ $# -lt 2 ]] && die "--repetitions requires an argument"
                SWEEP_REPETITIONS="$2"; shift 2 ;;
            --timeout)
                [[ $# -lt 2 ]] && die "--timeout requires an argument"
                SWEEP_TIMEOUT_SEC="$2"; shift 2 ;;
            --cpu-temp-limit)
                [[ $# -lt 2 ]] && die "--cpu-temp-limit requires an argument"
                SWEEP_CPU_TEMP_LIMIT="$2"; shift 2 ;;
            --gpu-temp-limit)
                [[ $# -lt 2 ]] && die "--gpu-temp-limit requires an argument"
                SWEEP_GPU_TEMP_LIMIT="$2"; shift 2 ;;
            --no-thermal-guard)
                OPT_NO_THERMAL=true; shift ;;
            --resume)
                OPT_RESUME=true; shift ;;
            --overwrite)
                OPT_OVERWRITE=true; shift ;;
            --only-phases)
                [[ $# -lt 2 ]] && die "--only-phases requires an argument"
                OPT_ONLY_PHASES="$2"; shift 2 ;;
            --skip-phases)
                [[ $# -lt 2 ]] && die "--skip-phases requires an argument"
                OPT_SKIP_PHASES="$2"; shift 2 ;;
            --start-ngl)
                [[ $# -lt 2 ]] && die "--start-ngl requires an argument"
                OPT_START_NGL="$2"; shift 2 ;;
            --ngl-dir)
                [[ $# -lt 2 ]] && die "--ngl-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--ngl-dir must be 'up' or 'down'"
                OPT_DIR_NGL="$2"; shift 2 ;;
            --start-threads)
                [[ $# -lt 2 ]] && die "--start-threads requires an argument"
                OPT_START_THREADS="$2"; shift 2 ;;
            --threads-dir)
                [[ $# -lt 2 ]] && die "--threads-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--threads-dir must be 'up' or 'down'"
                OPT_DIR_THREADS="$2"; shift 2 ;;
            --start-ctx)
                [[ $# -lt 2 ]] && die "--start-ctx requires an argument"
                OPT_START_CTX="$2"; shift 2 ;;
            --ctx-dir)
                [[ $# -lt 2 ]] && die "--ctx-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--ctx-dir must be 'up' or 'down'"
                OPT_DIR_CTX="$2"; shift 2 ;;
            --start-ctk)
                [[ $# -lt 2 ]] && die "--start-ctk requires an argument"
                OPT_START_CTK="$2"; shift 2 ;;
            --ctk-dir)
                [[ $# -lt 2 ]] && die "--ctk-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--ctk-dir must be 'up' or 'down'"
                OPT_DIR_CTK="$2"; shift 2 ;;
            --start-b)
                [[ $# -lt 2 ]] && die "--start-b requires an argument"
                OPT_START_B="$2"; shift 2 ;;
            --b-dir)
                [[ $# -lt 2 ]] && die "--b-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--b-dir must be 'up' or 'down'"
                OPT_DIR_B="$2"; shift 2 ;;
            --start-ub)
                [[ $# -lt 2 ]] && die "--start-ub requires an argument"
                OPT_START_UB="$2"; shift 2 ;;
            --ub-dir)
                [[ $# -lt 2 ]] && die "--ub-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--ub-dir must be 'up' or 'down'"
                OPT_DIR_UB="$2"; shift 2 ;;
            --min-ngl)
                [[ $# -lt 2 ]] && die "--min-ngl requires an argument"
                OPT_MIN_NGL="$2"; shift 2 ;;
            --min-threads)
                [[ $# -lt 2 ]] && die "--min-threads requires an argument"
                OPT_MIN_THREADS="$2"; shift 2 ;;
            --min-ctx)
                [[ $# -lt 2 ]] && die "--min-ctx requires an argument"
                OPT_MIN_CTX="$2"; shift 2 ;;
            --min-ctk)
                [[ $# -lt 2 ]] && die "--min-ctk requires an argument"
                OPT_MIN_CTK="$2"; shift 2 ;;
            --min-b)
                [[ $# -lt 2 ]] && die "--min-b requires an argument"
                OPT_MIN_B="$2"; shift 2 ;;
            --min-ub)
                [[ $# -lt 2 ]] && die "--min-ub requires an argument"
                OPT_MIN_UB="$2"; shift 2 ;;

            --start-fa)
                [[ $# -lt 2 ]] && die "--start-fa requires 0 or 1"
                [[ "$2" != "0" && "$2" != "1" ]] && die "--start-fa must be 0 or 1"
                OPT_START_FA="$2"; shift 2 ;;
            --fa-dir)
                [[ $# -lt 2 ]] && die "--fa-dir requires up or down"
                [[ "$2" != "up" && "$2" != "down" ]] && die "--fa-dir must be 'up' or 'down'"
                OPT_DIR_FA="$2"; shift 2 ;;

            --dry-run)
                OPT_DRY_RUN=true; shift ;;
            --no-confirm)
                OPT_NO_CONFIRM=true; shift ;;
            -h|--help)
                usage ;;
            -*)
                die "Unknown flag: $1  (run with --help for usage)" ;;
            *)
                die "Unexpected positional argument: $1  (run with --help for usage)" ;;
        esac
    done

    # Stash explicit --model so resolve_model_list() can prioritise it
    [[ -n "${model_explicit}" ]] && MODEL_LIST=("${model_explicit}")
}

# -----------------------------------------------------------------------------
# detect_hardware
#   Populate all HW_* variables by querying the OS and nvidia-smi.
#
#   Detects:
#     - CPU model string, physical core count, logical thread count
#     - Total and free system RAM (GiB)
#     - GPU count, model, total VRAM, free VRAM via nvidia-smi (if present)
#     - HW_BACKEND: cuda when nvidia-smi succeeds, else cpu
#     - HW_CPU_TEMP_CMD: a shell snippet that echoes the CPU temp as integer °C
#       (tries sensors, /sys/class/thermal, ipmitool in order)
#     - HW_GPU_TEMP_CMD: a shell snippet using nvidia-smi --query-gpu=temperature.gpu
# -----------------------------------------------------------------------------
detect_hardware() {
    local os_type
    os_type="$(uname -s)"   # Linux | Darwin
    local arch
    arch="$(uname -m)"      # x86_64 | arm64 | aarch64

    log "[HW] OS: ${os_type}  Arch: ${arch}"

    # ------------------------------------------------------------------
    # CPU — model string, physical cores, logical threads
    # ------------------------------------------------------------------
    # TODO:
    # if [[ "${os_type}" == "Darwin" ]]; then
    #     HW_CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    #     HW_CPU_PHYSICAL="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 1)"
    #     HW_CPU_LOGICAL="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
    # else  # Linux
    #     HW_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    #     HW_CPU_PHYSICAL="$(lscpu | awk -F: '/^Core\(s\) per socket/{cores=$2} /^Socket\(s\)/{sockets=$2} END{print cores*sockets}' | xargs)"
    #     HW_CPU_LOGICAL="$(nproc --all 2>/dev/null || grep -c '^processor' /proc/cpuinfo)"
    # fi

    # ------------------------------------------------------------------
    # RAM — total and free GiB
    # ------------------------------------------------------------------
    # TODO:
    # if [[ "${os_type}" == "Darwin" ]]; then
    #     local mem_bytes
    #     mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    #     HW_RAM_GIB=$(( mem_bytes / 1073741824 ))
    #     # Free RAM on macOS: vm_stat gives pages; page size is 16384 on Apple Silicon, 4096 on Intel
    #     local page_size
    #     page_size="$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)"
    #     local pages_free
    #     pages_free="$(vm_stat | awk '/^Pages free/{gsub(/\./, "", $3); print $3}')"
    #     HW_RAM_FREE_GIB=$(( pages_free * page_size / 1073741824 ))
    # else  # Linux
    #     HW_RAM_GIB="$(awk '/^MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)"
    #     HW_RAM_FREE_GIB="$(awk '/^MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
    # fi

    # ------------------------------------------------------------------
    # GPU and backend detection
    # ------------------------------------------------------------------
    # Priority order: CUDA (nvidia-smi) → Metal (macOS) → CPU fallback
    #
    # TODO:
    # CUDA path (Linux + Windows + macOS with eGPU):
    # if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    #     HW_BACKEND="cuda"
    #     HW_GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | xargs)"
    #     HW_GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)"
    #     local vram_mib
    #     vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | xargs)"
    #     HW_GPU_VRAM_GIB=$(( vram_mib / 1024 ))
    #     local vram_free_mib
    #     vram_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | xargs)"
    #     HW_GPU_VRAM_FREE_GIB=$(( vram_free_mib / 1024 ))
    #
    # Metal path (macOS — Apple Silicon or Intel Mac with integrated/discrete GPU):
    # On Apple Silicon, GPU VRAM is shared with system RAM (unified memory).
    # llama.cpp uses Metal backend automatically; no separate VRAM budget applies.
    # elif [[ "${os_type}" == "Darwin" ]]; then
    #     HW_BACKEND="metal"
    #     HW_GPU_COUNT=1
    #     HW_GPU_MODEL="$(system_profiler SPDisplaysDataType 2>/dev/null \
    #         | awk '/Chipset Model/{print $3,$4,$5}' | head -1 | xargs || echo 'Apple GPU')"
    #     if [[ "${arch}" == "arm64" ]]; then
    #         # Apple Silicon: unified memory — report total RAM as "VRAM"
    #         HW_GPU_VRAM_GIB="${HW_RAM_GIB}"
    #         HW_GPU_VRAM_FREE_GIB="${HW_RAM_FREE_GIB}"
    #         # Note for NGL probe: all layers fit "in GPU" for Apple Silicon unified memory;
    #         # the real constraint is total RAM. The probe will still find the true ceiling.
    #         log "[HW] Apple Silicon detected — unified memory. VRAM reported = total RAM."
    #     else
    #         # Intel Mac with discrete GPU: attempt to read VRAM from system_profiler
    #         local vram_str
    #         vram_str="$(system_profiler SPDisplaysDataType 2>/dev/null \
    #             | awk '/VRAM/{print $2}' | head -1)"
    #         HW_GPU_VRAM_GIB="${vram_str:-0}"
    #         HW_GPU_VRAM_FREE_GIB=0  # not reliably readable on macOS Intel
    #     fi
    #
    # CPU-only fallback:
    # else
    #     HW_BACKEND="cpu"
    #     HW_GPU_COUNT=0
    #     HW_GPU_MODEL="none"
    #     HW_GPU_VRAM_GIB=0
    #     HW_GPU_VRAM_FREE_GIB=0
    #     log "[HW] No GPU detected — CPU-only mode. NGL probe will be skipped."
    # fi

    # ------------------------------------------------------------------
    # Thermal sensor commands — OS and tool dependent
    # ------------------------------------------------------------------
    # HW_CPU_TEMP_CMD and HW_GPU_TEMP_CMD must be shell snippets that,
    # when eval'd, print a single integer (°C) to stdout.
    #
    # TODO:
    # if [[ "${os_type}" == "Darwin" ]]; then
    #     # macOS: use powermetrics (requires sudo) or osx-cpu-temp if installed
    #     if command -v osx-cpu-temp &>/dev/null; then
    #         HW_CPU_TEMP_CMD="osx-cpu-temp | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f1"
    #     else
    #         # powermetrics needs sudo; skip thermal guard if unavailable
    #         HW_CPU_TEMP_CMD=""
    #         warn "[HW] CPU temp monitoring unavailable on macOS without osx-cpu-temp or sudo powermetrics"
    #     fi
    #     if [[ "${HW_BACKEND}" == "cuda" ]]; then
    #         HW_GPU_TEMP_CMD="nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
    #     else
    #         # Metal GPU temps not readable without third-party tools; disable guard
    #         HW_GPU_TEMP_CMD=""
    #         warn "[HW] GPU temp monitoring unavailable for Metal backend"
    #     fi
    # else  # Linux
    #     # Try lm-sensors first (covers most AMD/Intel CPUs)
    #     if command -v sensors &>/dev/null; then
    #         # AMD: look for Tctl; Intel: look for Package id 0
    #         if sensors 2>/dev/null | grep -qi "tctl"; then
    #             HW_CPU_TEMP_CMD="sensors 2>/dev/null | awk '/Tctl/{gsub(/[^0-9.]/,\"\",\$2); printf \"%d\", \$2}'"
    #         else
    #             HW_CPU_TEMP_CMD="sensors 2>/dev/null | awk '/Package id 0/{gsub(/[^0-9.]/,\"\",\$4); printf \"%d\", \$4}'"
    #         fi
    #     # Fallback: /sys/class/thermal (available on most Linux kernels)
    #     elif [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
    #         HW_CPU_TEMP_CMD="awk '{printf \"%d\", \$1/1000}' /sys/class/thermal/thermal_zone0/temp"
    #     else
    #         HW_CPU_TEMP_CMD=""
    #         warn "[HW] CPU temp monitoring unavailable — install lm-sensors (apt install lm-sensors)"
    #     fi
    #     # Linux GPU temp: nvidia-smi for CUDA, nothing reliable for others
    #     if [[ "${HW_BACKEND}" == "cuda" ]]; then
    #         HW_GPU_TEMP_CMD="nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
    #     else
    #         HW_GPU_TEMP_CMD=""
    #     fi
    # fi
    #
    # If either temp command is empty, wait_cool() will skip that check with a warning.

    log "detect_hardware: not yet implemented — using placeholder defaults"
}

# -----------------------------------------------------------------------------
# detect_turbo_binary
#   Check whether SWEEP_TURBO_BENCH_BIN is set, points to an executable file,
#   and advertises turbo3 support via its --help output.
#   Sets TURBO_AVAILABLE=true on success; logs a warning and leaves it false
#   otherwise.
# -----------------------------------------------------------------------------
detect_turbo_binary() {
    # TODO: implement
    #   [[ -z "${SWEEP_TURBO_BENCH_BIN}" ]] && return 0
    #   [[ ! -x "${SWEEP_TURBO_BENCH_BIN}" ]] && warn "turbo-bench not executable" && return 0
    #   "${SWEEP_TURBO_BENCH_BIN}" --help 2>&1 | grep -qi "turbo3" || {
    #       warn "turbo-bench binary does not appear to support turbo3"
    #       return 0
    #   }
    #   TURBO_AVAILABLE=true
    #   log "turbo-bench available: ${SWEEP_TURBO_BENCH_BIN}"
    log "detect_turbo_binary: not yet implemented"
}

# -----------------------------------------------------------------------------
# print_hardware_summary
#   Print a human-readable table of HW_* values to the terminal (not the log).
#   Called once before the sweep begins so the operator can sanity-check the
#   detected hardware before committing to a long run.
# -----------------------------------------------------------------------------
print_hardware_summary() {
    # TODO: implement
    #   printf with column alignment, something like:
    #
    #   ┌─────────────────────────────────────────────┐
    #   │  Hardware Summary                           │
    #   ├──────────────┬──────────────────────────────┤
    #   │ CPU          │ ${HW_CPU_MODEL}               │
    #   │ Cores        │ ${HW_CPU_PHYSICAL}P / ${HW_CPU_LOGICAL}L  │
    #   │ RAM          │ ${HW_RAM_GIB} GiB (${HW_RAM_FREE_GIB} free) │
    #   │ GPU [0]      │ ${HW_GPU_MODEL}               │
    #   │ VRAM         │ ${HW_GPU_VRAM_GIB} GiB (${HW_GPU_VRAM_FREE_GIB} free) │
    #   │ Backend      │ ${HW_BACKEND}                 │
    #   │ Turbo bench  │ ${TURBO_AVAILABLE}            │
    #   └──────────────┴──────────────────────────────┘
    echo "[print_hardware_summary] TODO: implement hardware summary table"
}

# -----------------------------------------------------------------------------
# resolve_model_list
#   Build MODEL_LIST from the various model-selection mechanisms.
#
#   Priority order (highest wins):
#     1. --model (already placed in MODEL_LIST by parse_args — skip if set)
#     2. --model-list FILE (OPT_MODEL_LIST_FILE)
#     3. --models-dir DIR (SWEEP_MODELS_DIR from CLI)
#     4. SWEEP_MODELS_DIR from environment
#
#   Each resolved path is validated via validate_model().
#   Exits with an error if MODEL_LIST is still empty after resolution.
# -----------------------------------------------------------------------------
resolve_model_list() {
    # TODO: implement
    #   if [[ ${#MODEL_LIST[@]} -gt 0 ]]; then
    #       # --model already populated by parse_args; just validate
    #       return 0
    #   fi
    #   if [[ -n "${OPT_MODEL_LIST_FILE}" ]]; then
    #       while IFS= read -r line; do
    #           [[ -z "${line}" || "${line}" == \#* ]] && continue
    #           MODEL_LIST+=("${line}")
    #       done < "${OPT_MODEL_LIST_FILE}"
    #   elif [[ -n "${SWEEP_MODELS_DIR}" ]]; then
    #       while IFS= read -r -d '' f; do
    #           MODEL_LIST+=("$f")
    #       done < <(find "${SWEEP_MODELS_DIR}" -maxdepth 1 -name "*.gguf" -print0 | sort -z)
    #   fi
    #   [[ ${#MODEL_LIST[@]} -eq 0 ]] && die "No models found. Use --model, --models-dir, or --model-list."
    #   for m in "${MODEL_LIST[@]}"; do validate_model "$m"; done
    log "resolve_model_list: not yet implemented"
}

# -----------------------------------------------------------------------------
# validate_model PATH
#   Verify that PATH:
#     - exists and is a regular file
#     - is readable
#     - has a .gguf extension (case-insensitive)
#   Dies with an informative message on any failure.
# -----------------------------------------------------------------------------
validate_model() {
    local path="$1"
    # TODO: implement
    #   [[ -f "${path}" ]]    || die "Model not found: ${path}"
    #   [[ -r "${path}" ]]    || die "Model not readable: ${path}"
    #   [[ "${path,,}" == *.gguf ]] || die "Model does not appear to be a .gguf file: ${path}"
    log "validate_model: not yet implemented (path=${path})"
}

# -----------------------------------------------------------------------------
# setup_output_dir
#   Create OUTPUT_MODEL_DIR (${SWEEP_OUTPUT_DIR}/${MODEL_STEM}/).
#   If the directory already exists:
#     - with --overwrite: remove it entirely and recreate
#     - with --resume:    leave it in place (load_state() will read progress)
#     - otherwise:        die with an informative message suggesting --resume
#                         or --overwrite
# -----------------------------------------------------------------------------
setup_output_dir() {
    # TODO: implement
    #   OUTPUT_MODEL_DIR="${SWEEP_OUTPUT_DIR}/${MODEL_STEM}"
    #   if [[ -d "${OUTPUT_MODEL_DIR}" ]]; then
    #       if $OPT_OVERWRITE; then
    #           rm -rf "${OUTPUT_MODEL_DIR}"
    #       elif ! $OPT_RESUME; then
    #           die "Output dir exists: ${OUTPUT_MODEL_DIR}. Use --resume or --overwrite."
    #       fi
    #   fi
    #   mkdir -p "${OUTPUT_MODEL_DIR}"
    #   log "Output directory: ${OUTPUT_MODEL_DIR}"
    log "setup_output_dir: not yet implemented"
}

# -----------------------------------------------------------------------------
# load_state
#   If --resume is set and state.json exists in OUTPUT_MODEL_DIR, read the
#   phases_complete array and working_sets map into shell variables so that
#   sweep_model() can skip already-finished phases.
#   No-op when --resume is false or state.json is absent.
# -----------------------------------------------------------------------------
load_state() {
    # TODO: implement
    #   local state_file="${OUTPUT_MODEL_DIR}/state.json"
    #   $OPT_RESUME || return 0
    #   [[ -f "${state_file}" ]] || return 0
    #   # Use jq to extract phases_complete[] into a bash array
    #   # e.g. PHASES_COMPLETE=( $(jq -r ".phases_complete[]" "${state_file}") )
    log "load_state: not yet implemented"
}

# -----------------------------------------------------------------------------
# save_state PHASE_COMPLETED
#   Append PHASE_COMPLETED to the phases_complete list in state.json.
#   Also serialises current working_sets (best NGL, best KV config, etc.)
#   into the JSON so a resumed run has full context.
#   Creates state.json if it does not exist.
# -----------------------------------------------------------------------------
save_state() {
    local phase="${1:-}"
    # TODO: implement
    #   local state_file="${OUTPUT_MODEL_DIR}/state.json"
    #   Use jq to read existing state (or start from {}), append phase to
    #   .phases_complete, merge in current working_set variables, write back.
    log "save_state: not yet implemented (phase=${phase})"
}

# -----------------------------------------------------------------------------
# wait_cool
#   Poll CPU and GPU temperatures until both are below their respective limits.
#   Prints a waiting message each poll interval so the operator knows it is
#   not hung.
#   No-op when OPT_NO_THERMAL=true or when the temperature commands are empty.
# -----------------------------------------------------------------------------
wait_cool() {
    # TODO: implement
    #   $OPT_NO_THERMAL && return 0
    #   local cpu_temp gpu_temp
    #   while true; do
    #       [[ -n "${HW_CPU_TEMP_CMD}" ]] && cpu_temp=$(eval "${HW_CPU_TEMP_CMD}") || cpu_temp=0
    #       [[ -n "${HW_GPU_TEMP_CMD}" ]] && gpu_temp=$(eval "${HW_GPU_TEMP_CMD}") || gpu_temp=0
    #       if [[ "${cpu_temp}" -lt "${SWEEP_CPU_TEMP_LIMIT}" && "${gpu_temp}" -lt "${SWEEP_GPU_TEMP_LIMIT}" ]]; then
    #           break
    #       fi
    #       log "Thermal wait: CPU=${cpu_temp}°C GPU=${gpu_temp}°C — sleeping ${SWEEP_COOL_POLL_SEC}s"
    #       sleep "${SWEEP_COOL_POLL_SEC}"
    #   done
    return 0
}

# -----------------------------------------------------------------------------
# detect_oom LOG_FILE
#   Scan LOG_FILE for common OOM / fatal-error strings:
#     - "CUDA out of memory"
#     - "out of memory"
#     - "failed to allocate"
#     - "ggml_cuda_pool_alloc"
#     - "ggml_backend_alloc" failure patterns
#   Returns 0 (true) if an OOM/error pattern is found, 1 otherwise.
# -----------------------------------------------------------------------------
detect_oom() {
    local log_file="${1:-}"
    # TODO: implement
    #   [[ -f "${log_file}" ]] || return 1
    #   grep -qiE "(out of memory|failed to allocate|ggml_cuda_pool_alloc|CUDA error|cudaMalloc failed)" "${log_file}"
    return 1
}

# -----------------------------------------------------------------------------
# select_binary CTK_TYPE
#   Given a KV cache type string (e.g. "f16", "q8_0", "turbo3"), echo the
#   path to the appropriate binary:
#     - turbo* types  -> SWEEP_TURBO_BENCH_BIN  (only when TURBO_AVAILABLE)
#     - all others    -> LLAMA_BENCH_BIN
#   Dies if turbo* is requested but TURBO_AVAILABLE=false.
# -----------------------------------------------------------------------------
select_binary() {
    local ctk="${1:-f16}"
    # TODO: implement
    #   if [[ "${ctk}" == turbo* ]]; then
    #       $TURBO_AVAILABLE || die "turbo KV type requested but turbo-bench not available"
    #       echo "${SWEEP_TURBO_BENCH_BIN}"
    #   else
    #       echo "${LLAMA_BENCH_BIN}"
    #   fi
    echo "${LLAMA_BENCH_BIN}"
}

# -----------------------------------------------------------------------------
# write_jsonl_record KEY=VALUE...
#   Append a single JSON object to ${OUTPUT_MODEL_DIR}/sweep.jsonl.
#   Accepts key=value pairs as positional arguments and uses jq --arg to build
#   the object safely (no manual JSON escaping required).
#   Always adds a "timestamp" field (UTC ISO-8601) automatically.
#   Example call:
#     write_jsonl_record model="${MODEL_STEM}" phase=1 ngl=32 tg_ts=14.7
# -----------------------------------------------------------------------------
write_jsonl_record() {
    # TODO: implement
    #   local jsonl_file="${OUTPUT_MODEL_DIR}/sweep.jsonl"
    #   local jq_args=() jq_filter="{timestamp: \$ts"
    #   jq_args+=(--arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    #   for kv in "$@"; do
    #       local k="${kv%%=*}" v="${kv#*=}"
    #       jq_args+=(--arg "${k}" "${v}")
    #       jq_filter+=", ${k}: \$${k}"
    #   done
    #   jq_filter+="}"
    #   jq -n "${jq_args[@]}" "${jq_filter}" >> "${jsonl_file}"
    log "write_jsonl_record: not yet implemented (args=$*)"
}

# -----------------------------------------------------------------------------
# run_bench LABEL KEY=VALUE...
#   The central bench execution wrapper.
#   LABEL is a short human-readable string for log messages (e.g. "ngl=32").
#   Remaining KEY=VALUE pairs are llama-bench CLI flags
#     (e.g. ngl=32 fa=1 ctk=q8_0 t=8).
#
#   Execution steps:
#     1. Resolve binary via select_binary (reads ctk= from args if present)
#     2. Call wait_cool()
#     3. Sleep SWEEP_DELAY_SEC
#     4. Construct llama-bench command with --model, all key=value args,
#        -r SWEEP_REPETITIONS, -o json, and ionice/nice wrapping
#     5. If OPT_DRY_RUN: print command and return "dry-run"
#     6. Execute under timeout SWEEP_TIMEOUT_SEC, capture stdout + stderr to
#        per-run log file in OUTPUT_MODEL_DIR/runs/
#     7. Call detect_oom on the run log; if OOM: log warn, return "oom"
#     8. Parse tg (token-gen) t/s from JSON output via jq
#     9. If tg t/s < SWEEP_MIN_TG_TS: log warn, return "too-slow"
#    10. Call write_jsonl_record with all params + tg_ts result
#    11. Return "ok"
# -----------------------------------------------------------------------------
run_bench() {
    local label="${1:-unknown}"
    shift
    # TODO: implement full orchestration described above
    log "run_bench: not yet implemented (label=${label}, args=$*)"
    echo "not-implemented"
}

# apply_axis_opts LIST_VALUES START_VALUE DIRECTION
#
# Given a full ordered list of values (space-separated), a start value, and a
# direction ("up" or "down"), returns the subset of the list beginning at
# START_VALUE and proceeding in the given direction.
#
# If START_VALUE is empty, the full list is returned in the given direction.
# If START_VALUE is not found in the list, the full list is returned with a warning.
#
# "up"   = ascending order starting from START_VALUE
# "down" = descending order starting from START_VALUE
#
# Output: space-separated values printed to stdout, one per line.
#
# Example:
#   apply_axis_opts "0 4 8 12 16 20" "8" "up"   -> 8 12 16 20
#   apply_axis_opts "0 4 8 12 16 20" "8" "down" -> 8 4 0
apply_axis_opts() {
    local -a full_list=($1)
    local start="$2"
    local direction="$3"

    # Reverse list if direction is down
    local -a ordered=()
    if [[ "${direction}" == "down" ]]; then
        local i
        for (( i=${#full_list[@]}-1; i>=0; i-- )); do
            ordered+=("${full_list[$i]}")
        done
    else
        ordered=("${full_list[@]}")
    fi

    # If no start specified, return full ordered list
    if [[ -z "${start}" ]]; then
        printf '%s\n' "${ordered[@]}"
        return 0
    fi

    # Find start index
    local found=false
    local val
    for val in "${ordered[@]}"; do
        if [[ "${found}" == true ]]; then
            echo "${val}"
        elif [[ "${val}" == "${start}" ]]; then
            found=true
            echo "${val}"
        fi
    done

    if [[ "${found}" == false ]]; then
        warn "Start value '${start}' not found in axis list -- using full list"
        printf '%s\n' "${ordered[@]}"
    fi
}

# apply_phase7_mins AXIS VALUES MIN_VALUE
#
# Filters a newline-separated list of working values, removing entries that
# fall below the given minimum threshold. Used to trim working sets before
# building the Phase 7 combination matrix.
#
# For numeric axes (ngl, threads, ctx, b, ub): removes values strictly < MIN_VALUE.
# For the KV type axis (ctk): removes types below MIN_VALUE in quality order.
#   KV quality order (low->high): turbo2 turbo3 turbo4 q4_0 q8_0 f16
#
# If MIN_VALUE is empty, the full list is returned unchanged.
# If MIN_VALUE is not recognised (ctk axis), a warning is logged and the list
# is returned unchanged.
#
# Arguments:
#   $1  axis:   "ngl" | "threads" | "ctx" | "b" | "ub" | "ctk"
#   $2  values: newline-separated working value list
#   $3  min:    minimum value string (may be empty)
apply_phase7_mins() {
    local axis="$1"
    local values="$2"
    local min_val="$3"

    if [[ -z "${min_val}" ]]; then
        echo "${values}"
        return 0
    fi

    if [[ "${axis}" == "ctk" ]]; then
        local -a ctk_order=("turbo2" "turbo3" "turbo4" "q4_0" "q8_0" "f16")
        local min_idx=-1 i
        for i in "${!ctk_order[@]}"; do
            [[ "${ctk_order[$i]}" == "${min_val}" ]] && { min_idx=$i; break; }
        done
        if [[ $min_idx -eq -1 ]]; then
            warn "Unknown --min-ctk value '${min_val}' â no filtering applied"
            echo "${values}"; return 0
        fi
        local val
        while IFS= read -r val; do
            [[ -z "${val}" ]] && continue
            local val_idx=-1
            for i in "${!ctk_order[@]}"; do
                [[ "${ctk_order[$i]}" == "${val}" ]] && { val_idx=$i; break; }
            done
            (( val_idx >= min_idx )) && echo "${val}"
        done <<< "${values}"
    else
        local val
        while IFS= read -r val; do
            [[ -z "${val}" ]] && continue
            (( val >= min_val )) && echo "${val}"
        done <<< "${values}"
    fi
}

# -----------------------------------------------------------------------------
# phase0_ngl_probe
#   Algorithm: binary-search the maximum NGL that fits without OOM.
#     lo=0, hi=99 (or layer count reported by model metadata if available)
#     mid = (lo+hi)/2
#     run_bench with SWEEP_PROBE_REPS at mid
#     OOM -> hi = mid-1
#     OK  -> lo = mid+1, record candidate
#     Continue until lo > hi; MAX_NGL = last OK candidate
#   Uses a single pp/tg token pair (pp=1, tg=1) for speed.
#   Sets MAX_NGL global for all subsequent phases.
#   Skipped entirely when HW_GPU_COUNT=0 (CPU-only inference).
# -----------------------------------------------------------------------------
phase0_ngl_probe() {
    log "[Phase 0] NGL probe — binary-search max stable NGL"
    # TODO: implement binary search described above
}

# -----------------------------------------------------------------------------
# phase1_ngl_sweep
#   Algorithm: iterate NGL from 0 to MAX_NGL in steps of SWEEP_NGL_STEP,
#   always including 0 and MAX_NGL as explicit endpoints.
#   Run run_bench at each step with SWEEP_REPETITIONS.
#   Record tg t/s at each point; the NGL with the highest tg t/s becomes
#   the working BEST_NGL for subsequent phases.
#   Also records pp (prompt-processing) t/s for each point.
# -----------------------------------------------------------------------------
phase1_ngl_sweep() {
    log "[Phase 1] NGL sweep (0.. step ${SWEEP_NGL_STEP})"
    # Apply: ngl_list=$(apply_axis_opts "${full_ngl_list}" "${OPT_START_NGL}" "${OPT_DIR_NGL}")
    # TODO: implement iterative sweep described above
}

# -----------------------------------------------------------------------------
# phase2_fa_kv_sweep
#   Test all flash-attn × KV-type combinations at BEST_NGL.
#   Standard combos (always tested):
#     fa=0  ctk=f16   ctv=f16
#     fa=0  ctk=q8_0  ctv=q8_0
#     fa=0  ctk=q4_0  ctv=f16
#     fa=1  ctk=f16   ctv=f16
#     fa=1  ctk=q8_0  ctv=q8_0
#     fa=1  ctk=q5_1  ctv=q5_1
#     fa=1  ctk=q4_0  ctv=f16
#   Turbo combos (only when TURBO_AVAILABLE=true):
#     fa=1  ctk=turbo3  ctv=turbo3
#     fa=1  ctk=turbo4  ctv=turbo4
#   Best combo (highest tg t/s) stored in BEST_FA / BEST_CTK / BEST_CTV.
# -----------------------------------------------------------------------------
phase2_fa_kv_sweep() {
    log "[Phase 2] Flash-attn × KV-type sweep"
    # Apply: ctk_list=$(apply_axis_opts "${full_ctk_list}" "${OPT_START_CTK}" "${OPT_DIR_CTK}")
    # Apply FA direction:
    # fa_list=$(apply_axis_opts "0 1" "${OPT_START_FA}" "${OPT_DIR_FA}")
    # Then iterate fa_list x ctk_list, skipping invalid combos (fa=0 + q4_0, fa=0 + turbo*)
    # TODO: implement combo iteration described above
    #
    # Standard combos array (fa ctk ctv):
    #   ("0 f16 f16" "0 q8_0 q8_0" "0 q4_0 f16"
    #    "1 f16 f16" "1 q8_0 q8_0" "1 q5_1 q5_1" "1 q4_0 f16")
    # Turbo combos (appended when TURBO_AVAILABLE):
    #   ("1 turbo3 turbo3" "1 turbo4 turbo4")
}

# -----------------------------------------------------------------------------
# phase3_thread_sweep
#   Sweep CPU thread count at NGL=0 (pure-CPU path) to find the optimal
#   thread count for CPU-side work (used in mixed GPU+CPU offload).
#   Thread values tested are derived from hardware:
#     1, HW_CPU_PHYSICAL/2, HW_CPU_PHYSICAL, HW_CPU_LOGICAL
#   Plus any values in between that are round numbers (e.g. 4, 6, 8, 12, 16).
#   Best thread count stored in BEST_THREADS.
# -----------------------------------------------------------------------------
phase3_thread_sweep() {
    log "[Phase 3] CPU thread sweep"
    # Apply: thread_list=$(apply_axis_opts "${full_thread_list}" "${OPT_START_THREADS}" "${OPT_DIR_THREADS}")
    # TODO: implement
    # Thread candidates derived from HW_CPU_PHYSICAL and HW_CPU_LOGICAL:
    #   e.g. for 8P/16L: 1 2 4 6 8 12 16
    # Run each at NGL=0 with BEST_FA/BEST_CTK/BEST_CTV
}

# -----------------------------------------------------------------------------
# phase4_nkvo_sweep
#   Test the --no-kv-offload flag against the best configuration found so far
#   (BEST_NGL, BEST_FA, BEST_CTK, BEST_CTV, BEST_THREADS).
#   Combos:
#     nkvo=0  (default, KV offloaded to GPU)
#     nkvo=1  (KV stays in CPU RAM — may free VRAM for larger context later)
#   Records which is faster; stores result in BEST_NKVO.
# -----------------------------------------------------------------------------
phase4_nkvo_sweep() {
    log "[Phase 4] No-KV-offload sweep"
    # TODO: implement nkvo=0 vs nkvo=1 comparison
}

# -----------------------------------------------------------------------------
# phase5_batch_sweep
#   Test batch (b) and micro-batch (ub) size pairs.
#   Pairs to test:
#     b=512   ub=512
#     b=1024  ub=512
#     b=1024  ub=1024
#     b=2048  ub=512
#     b=2048  ub=1024
#     b=2048  ub=2048
#     b=4096  ub=512
#     b=4096  ub=1024
#     b=4096  ub=2048
#   Run at full best config. Best pair (highest pp t/s) stored in
#   BEST_BATCH / BEST_UBATCH.
# -----------------------------------------------------------------------------
phase5_batch_sweep() {
    log "[Phase 5] Batch / ubatch sweep"
    # Apply: b_list=$(apply_axis_opts "${full_b_list}" "${OPT_START_B}" "${OPT_DIR_B}")
    # Apply: ub_list=$(apply_axis_opts "${full_ub_list}" "${OPT_START_UB}" "${OPT_DIR_UB}")
    # TODO: implement batch pair iteration described above
}

# -----------------------------------------------------------------------------
# phase6_ctx_sweep
#   Sweep context window sizes at the best configuration found so far.
#   Sizes to test: 512 1024 2048 4096 8192 16384 32768 65536
#   Stop (and record last-OK context) when:
#     - detect_oom returns true, OR
#     - tg t/s drops below SWEEP_MIN_TG_TS, OR
#     - run_bench returns "too-slow"
#   Best (largest stable) context stored in BEST_CTX.
# -----------------------------------------------------------------------------
phase6_ctx_sweep() {
    log "[Phase 6] Context window sweep"
    # Apply: ctx_list=$(apply_axis_opts "${full_ctx_list}" "${OPT_START_CTX}" "${OPT_DIR_CTX}")
    # TODO: implement context sweep with OOM/speed stop condition
    # Context sizes: 512 1024 2048 4096 8192 16384 32768 65536
}

# -----------------------------------------------------------------------------
# phase7_combination_matrix
#   Build a pruned cartesian product of the top-N candidates from each prior
#   phase and run every combination.
#
#   Algorithm:
#     1. For each tunable axis (ngl, fa+ctk+ctv, threads, nkvo, b+ub, ctx),
#        select the top-2 or top-3 results by tg t/s from sweep.jsonl.
#     2. Compute the full cartesian product of these candidate sets.
#     3. Prune combinations that are known-bad:
#          - ctk requires fa=1 (e.g. turbo* types)
#          - ctx > BEST_CTX (already proven to OOM)
#          - b < ub (invalid)
#     4. Run each surviving combination with SWEEP_REPETITIONS.
#
#   Pruning key: skip any combo where a dependent constraint from earlier
#   phases is violated to avoid redundant OOM runs.
# -----------------------------------------------------------------------------
phase7_combination_matrix() {
    log "[Phase 7] Combination matrix sweep"

    # Phase 7 uses exactly the values that phases 1-6 actually tested.
    # --start-* and --*-dir flags already trimmed those phases, so the
    # working sets here naturally reflect the user's sweep choices.
    #
    # Additionally, --min-* flags can further filter these sets:
    #
    # ngl_p7=$(apply_phase7_mins "ngl"     "${working_ngl}"     "${OPT_MIN_NGL}")
    # thr_p7=$(apply_phase7_mins "threads" "${working_threads}" "${OPT_MIN_THREADS}")
    # ctx_p7=$(apply_phase7_mins "ctx"     "${working_ctx}"     "${OPT_MIN_CTX}")
    # ctk_p7=$(apply_phase7_mins "ctk"     "${working_ctk}"     "${OPT_MIN_CTK}")
    # b_p7=$(apply_phase7_mins   "b"       "${working_b}"       "${OPT_MIN_B}")
    # ub_p7=$(apply_phase7_mins  "ub"      "${working_ub}"       "${OPT_MIN_UB}")
    #
    # Log what was filtered:
    # log "[PHASE 7] Working sets after minimums: ngl=${ngl_count} ctx=${ctx_count} ctk=${ctk_count}"
    # log "[PHASE 7] Estimated combinations: ${total_estimate}"

    # TODO: implement cartesian product + pruning described above
}

# -----------------------------------------------------------------------------
# write_markdown
#   Generate ${OUTPUT_MODEL_DIR}/results.md from sweep.jsonl.
#   Sections:
#     ## Hardware         — hardware summary block
#     ## Phase Results    — one table per phase, sorted by tg t/s descending
#     ## Best Configuration — the single winning parameter set
#   Uses jq to pivot JSONL into Markdown table rows.
# -----------------------------------------------------------------------------
write_markdown() {
    log "write_markdown: not yet implemented"
    # TODO: implement jq-based JSONL -> Markdown table generation
}

# -----------------------------------------------------------------------------
# print_summary
#   Print a concise ASCII table to the terminal summarising the sweep results:
#     - Best NGL, best KV config, best threads, best batch, best context
#     - Peak tg t/s and pp t/s achieved
#     - Total wall-clock time for the sweep
#     - Path to results.md and sweep.jsonl
# -----------------------------------------------------------------------------
print_summary() {
    log "print_summary: not yet implemented"
    # TODO: implement terminal summary table
}

# -----------------------------------------------------------------------------
# sweep_model PATH
#   Run all phases for a single model file.
#   Responsibilities:
#     1. Set MODEL_PATH, MODEL_STEM, OUTPUT_MODEL_DIR
#     2. Call setup_output_dir and load_state
#     3. For each phase 0..7:
#          a. Check OPT_ONLY_PHASES / OPT_SKIP_PHASES — skip if excluded
#          b. Check phases_complete (from load_state) — skip if already done
#          c. Call the phase function
#          d. Call save_state with the completed phase number
#     4. Call write_markdown and print_summary
# -----------------------------------------------------------------------------
sweep_model() {
    local path="${1}"
    MODEL_PATH="${path}"
    MODEL_STEM="$(basename "${path}" .gguf)"
    OUTPUT_MODEL_DIR="${SWEEP_OUTPUT_DIR}/${MODEL_STEM}"

    log "===== Starting sweep: ${MODEL_STEM} ====="
    setup_output_dir
    load_state

    # Phase execution helper — respects --only-phases and --skip-phases
    local phase
    for phase in 0 1 2 3 4 5 6 7; do
        # TODO: implement phase skip/only logic
        #   if [[ -n "${OPT_ONLY_PHASES}" ]] && ! echo "${OPT_ONLY_PHASES}" | grep -qw "${phase}"; then
        #       log "Phase ${phase}: skipped (not in --only-phases)"
        #       continue
        #   fi
        #   if [[ -n "${OPT_SKIP_PHASES}" ]] && echo "${OPT_SKIP_PHASES}" | grep -qw "${phase}"; then
        #       log "Phase ${phase}: skipped (in --skip-phases)"
        #       continue
        #   fi
        #   if <phase already in phases_complete>; then
        #       log "Phase ${phase}: already complete — skipping (--resume)"
        #       continue
        #   fi

        case "${phase}" in
            0) phase0_ngl_probe ;;
            1) phase1_ngl_sweep ;;
            2) phase2_fa_kv_sweep ;;
            3) phase3_thread_sweep ;;
            4) phase4_nkvo_sweep ;;
            5) phase5_batch_sweep ;;
            6) phase6_ctx_sweep ;;
            7) phase7_combination_matrix ;;
        esac

        save_state "${phase}"
    done

    write_markdown
    print_summary
    log "===== Sweep complete: ${MODEL_STEM} ====="
}

# -----------------------------------------------------------------------------
# main ARGS...
#   Entry point. Orchestrates the full sweep pipeline.
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    # Resolve the model list before printing anything substantial
    resolve_model_list

    # Hardware detection must run before hardware summary or binary detection
    detect_hardware
    detect_turbo_binary
    print_hardware_summary

    # Validate binary exists
    [[ -x "${LLAMA_BENCH_BIN}" ]] || die "llama-bench not found or not executable: ${LLAMA_BENCH_BIN}"

    # Pre-sweep confirmation (skipped with --no-confirm or --dry-run)
    if ! $OPT_NO_CONFIRM && ! $OPT_DRY_RUN; then
        echo
        printf "Ready to sweep %d model(s). Output -> %s\n" "${#MODEL_LIST[@]}" "${SWEEP_OUTPUT_DIR}"
        # TODO: read -r -p "Continue? [y/N] " reply; [[ "${reply}" =~ ^[Yy]$ ]] || die "Aborted by user"
    fi

    local model
    for model in "${MODEL_LIST[@]}"; do
        sweep_model "${model}"
    done

    log "All sweeps complete."
}

# =============================================================================
# Entry point
# =============================================================================
main "$@"

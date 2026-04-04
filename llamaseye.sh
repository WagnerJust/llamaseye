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

# Per-sweep working sets (accumulated as phases complete)
BEST_NGL=99
BEST_FA=0
BEST_CTK="f16"
BEST_CTV="f16"
BEST_THREADS=""          # empty = system default
BEST_NKVO=0
BEST_B=2048
BEST_UB=512
BEST_CTX=512

WS_NGL=""                # space-separated ok NGL values from phase 1
WS_FA_CTK=""             # newline-separated "fa ctk ctv" combos from phase 2
WS_THREADS=""            # space-separated ok thread counts from phase 3
WS_NKVO=""               # space-separated ok nkvo values from phase 4
WS_B_UB=""               # newline-separated "b ub" pairs from phase 5
WS_CTX=""                # space-separated ok context sizes from phase 6

PHASES_COMPLETE=""        # space-separated completed phase numbers

# =============================================================================
# CLI FLAGS — set by parse_args()
# =============================================================================

OPT_RESUME="${SWEEP_RESUME:-false}"              # --resume / SWEEP_RESUME
OPT_OVERWRITE="${SWEEP_OVERWRITE:-false}"        # --overwrite / SWEEP_OVERWRITE
OPT_DRY_RUN="${SWEEP_DRY_RUN:-false}"           # --dry-run / SWEEP_DRY_RUN
OPT_NO_CONFIRM="${SWEEP_NO_CONFIRM:-false}"      # --no-confirm / SWEEP_NO_CONFIRM
OPT_NO_THERMAL="${SWEEP_NO_THERMAL:-false}"      # --no-thermal-guard / SWEEP_NO_THERMAL
OPT_ONLY_PHASES="${SWEEP_ONLY_PHASES:-}"         # --only-phases / SWEEP_ONLY_PHASES
OPT_SKIP_PHASES="${SWEEP_SKIP_PHASES:-}"         # --skip-phases / SWEEP_SKIP_PHASES
OPT_MODEL_LIST_FILE="${SWEEP_MODEL_LIST:-}"      # --model-list / SWEEP_MODEL_LIST

# --- Sweep axis start points (default: begin of list for each axis) ---
OPT_START_NGL="${SWEEP_START_NGL:-}"             # --start-ngl / SWEEP_START_NGL
OPT_START_THREADS="${SWEEP_START_THREADS:-}"     # --start-threads / SWEEP_START_THREADS
OPT_START_CTX="${SWEEP_START_CTX:-}"             # --start-ctx / SWEEP_START_CTX
OPT_START_CTK="${SWEEP_START_CTK:-}"             # --start-ctk / SWEEP_START_CTK
OPT_START_B="${SWEEP_START_B:-}"                 # --start-b / SWEEP_START_B
OPT_START_UB="${SWEEP_START_UB:-}"               # --start-ub / SWEEP_START_UB
OPT_START_FA="${SWEEP_START_FA:-}"               # --start-fa / SWEEP_START_FA

# --- Sweep axis directions ---
# "up"   = sweep from start toward the high end of the list
# "down" = sweep from start toward the low end of the list
# KV type ordering (low->high compression): f16 q8_0 q4_0 turbo4 turbo3 turbo2
OPT_DIR_NGL="${SWEEP_NGL_DIR:-up}"               # --ngl-dir / SWEEP_NGL_DIR       (up = 0->max_ngl)
OPT_DIR_THREADS="${SWEEP_THREADS_DIR:-up}"       # --threads-dir / SWEEP_THREADS_DIR (up = 1->HW_CPU_LOGICAL)
OPT_DIR_CTX="${SWEEP_CTX_DIR:-up}"               # --ctx-dir / SWEEP_CTX_DIR        (up = 128->131072)
OPT_DIR_CTK="${SWEEP_CTK_DIR:-up}"               # --ctk-dir / SWEEP_CTK_DIR        (up = toward more compression)
OPT_DIR_B="${SWEEP_B_DIR:-up}"                   # --b-dir / SWEEP_B_DIR            (up = 512->2048)
OPT_DIR_UB="${SWEEP_UB_DIR:-up}"                 # --ub-dir / SWEEP_UB_DIR          (up = 128->512)
OPT_DIR_FA="${SWEEP_FA_DIR:-up}"                 # --fa-dir / SWEEP_FA_DIR          (up = 0->1)

# --- Phase 7 minimum thresholds (filter combination matrix inputs) ---
# These trim the per-axis working sets before Phase 7. For numeric axes,
# values strictly below the minimum are excluded. For ctk, types below
# the minimum in quality order are excluded.
# KV quality order (low->high): turbo2 turbo3 turbo4 q4_0 q8_0 f16
OPT_MIN_NGL="${SWEEP_MIN_NGL:-}"                 # --min-ngl / SWEEP_MIN_NGL
OPT_MIN_THREADS="${SWEEP_MIN_THREADS:-}"         # --min-threads / SWEEP_MIN_THREADS
OPT_MIN_CTX="${SWEEP_MIN_CTX:-}"                 # --min-ctx / SWEEP_MIN_CTX
OPT_MIN_CTK="${SWEEP_MIN_CTK:-}"                 # --min-ctk / SWEEP_MIN_CTK
OPT_MIN_B="${SWEEP_MIN_B:-}"                     # --min-b / SWEEP_MIN_B
OPT_MIN_UB="${SWEEP_MIN_UB:-}"                   # --min-ub / SWEEP_MIN_UB


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
        echo "$msg" | tee -a "${OUTPUT_MODEL_DIR}/sweep.log" >&2
    else
        echo "$msg" >&2
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

Phase 7 minimum thresholds (auto-derived by default, override to disable):
  --min-ngl N           Exclude ngl values below N from Phase 7 matrix.
                        Default: MAX_NGL - 1 step (top 2 ngl values).
                        Use --min-ngl 0 to include all ngl values.
  --min-threads N       Exclude thread counts below N from Phase 7 matrix.
                        Default: HW_CPU_PHYSICAL (physical core count).
                        Use --min-threads 1 to include all thread counts.
  --min-ctx N           Exclude context sizes below N from Phase 7 matrix.
                        Default: inherited from --start-ctx if set, else 8192.
                        Use --min-ctx 0 to include all context sizes.
                        If no ctx values pass the filter, Phase 7 is skipped with a warning.
  --min-ctk TYPE        Exclude KV types below TYPE (quality order) from Phase 7.
                        Quality order (low->high): turbo2 turbo3 turbo4 q4_0 q8_0 f16
                        Default: q8_0. Use --min-ctk q4_0 (or turbo2) to include all types.
                        e.g. --min-ctk turbo3 keeps turbo3, turbo4, q4_0, q8_0, f16.
  --min-b N             Exclude batch sizes below N from Phase 7 matrix.
                        Default: BEST_B / 2 (top half of batch sizes found).
                        Use --min-b 512 to include all batch sizes.
  --min-ub N            Exclude ubatch sizes below N from Phase 7 matrix.

  Note: Phase 7 always inherits the values actually tested in phases 1-6
  (which are already trimmed by --start-* and --*-dir flags). --min-* applies
  additional filtering on top of that. Auto-derived minimums are logged before
  the matrix runs so you can see exactly what was used.

  --dry-run             Print bench commands without executing them
  --no-confirm          Skip the pre-sweep confirmation prompt

  -h, --help            Show this help and exit

ENVIRONMENT VARIABLES
  Any configuration variable (e.g. SWEEP_NGL_STEP=8) can be set in the
  environment and will be picked up as the default, before CLI flags apply.

PHASES
  0  ngl_probe          Binary-search the max NGL that fits in VRAM
  1  ngl_sweep          Sweep NGL near MAX_NGL by default (use --start-ngl 0 for full sweep)
  2  fa_kv_sweep        Flash-attn × KV-type combinations
  3  thread_sweep       CPU thread count from physical-cores..logical-count
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
# gen_run_id
#   Generate a short unique run identifier (8 hex chars).
# -----------------------------------------------------------------------------
gen_run_id() {
    if command -v uuidgen &>/dev/null; then
        uuidgen | tr -d '-' | head -c 8 | tr '[:upper:]' '[:lower:]'
    else
        date +%s%N 2>/dev/null | md5sum | head -c 8 || date +%s | md5sum | head -c 8
    fi
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
    if [[ "${os_type}" == "Darwin" ]]; then
        HW_CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
        HW_CPU_PHYSICAL="$(sysctl -n hw.physicalcpu 2>/dev/null || echo 1)"
        HW_CPU_LOGICAL="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 1)"
    else  # Linux
        HW_CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
        HW_CPU_PHYSICAL="$(lscpu | awk -F: '/^Core\(s\) per socket/{cores=$2} /^Socket\(s\)/{sockets=$2} END{print cores*sockets}' | xargs)"
        HW_CPU_LOGICAL="$(nproc --all 2>/dev/null || grep -c '^processor' /proc/cpuinfo)"
    fi

    # ------------------------------------------------------------------
    # RAM — total and free GiB
    # ------------------------------------------------------------------
    if [[ "${os_type}" == "Darwin" ]]; then
        local mem_bytes
        mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
        HW_RAM_GIB=$(( mem_bytes / 1073741824 ))
        # Free RAM on macOS: vm_stat gives pages; page size is 16384 on Apple Silicon, 4096 on Intel
        local page_size
        page_size="$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)"
        local pages_free
        pages_free="$(vm_stat | awk '/^Pages free/{gsub(/\./, "", $3); print $3}')"
        HW_RAM_FREE_GIB=$(( pages_free * page_size / 1073741824 ))
    else  # Linux
        HW_RAM_GIB="$(awk '/^MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)"
        HW_RAM_FREE_GIB="$(awk '/^MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
    fi

    # ------------------------------------------------------------------
    # GPU and backend detection
    # ------------------------------------------------------------------
    # Priority order: CUDA (nvidia-smi) → Metal (macOS) → CPU fallback
    #
    # CUDA path (Linux + Windows + macOS with eGPU):
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HW_BACKEND="cuda"
        HW_GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | xargs)"
        HW_GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)"
        local vram_mib
        vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | xargs)"
        HW_GPU_VRAM_GIB=$(( vram_mib / 1024 ))
        local vram_free_mib
        vram_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | xargs)"
        HW_GPU_VRAM_FREE_GIB=$(( vram_free_mib / 1024 ))

    # Metal path (macOS — Apple Silicon or Intel Mac with integrated/discrete GPU):
    # On Apple Silicon, GPU VRAM is shared with system RAM (unified memory).
    # llama.cpp uses Metal backend automatically; no separate VRAM budget applies.
    elif [[ "${os_type}" == "Darwin" ]]; then
        HW_BACKEND="metal"
        HW_GPU_COUNT=1
        HW_GPU_MODEL="$(system_profiler SPDisplaysDataType 2>/dev/null \
            | awk '/Chipset Model/{print $3,$4,$5}' | head -1 | xargs || echo 'Apple GPU')"
        if [[ "${arch}" == "arm64" ]]; then
            # Apple Silicon: unified memory — report total RAM as "VRAM"
            HW_GPU_VRAM_GIB="${HW_RAM_GIB}"
            HW_GPU_VRAM_FREE_GIB="${HW_RAM_FREE_GIB}"
            # Note for NGL probe: all layers fit "in GPU" for Apple Silicon unified memory;
            # the real constraint is total RAM. The probe will still find the true ceiling.
            log "[HW] Apple Silicon detected — unified memory. VRAM reported = total RAM."
        else
            # Intel Mac with discrete GPU: attempt to read VRAM from system_profiler
            local vram_str
            vram_str="$(system_profiler SPDisplaysDataType 2>/dev/null \
                | awk '/VRAM/{print $2}' | head -1)"
            HW_GPU_VRAM_GIB="${vram_str:-0}"
            HW_GPU_VRAM_FREE_GIB=0  # not reliably readable on macOS Intel
        fi

    # CPU-only fallback:
    else
        HW_BACKEND="cpu"
        HW_GPU_COUNT=0
        HW_GPU_MODEL="none"
        HW_GPU_VRAM_GIB=0
        HW_GPU_VRAM_FREE_GIB=0
        log "[HW] No GPU detected — CPU-only mode. NGL probe will be skipped."
    fi

    # ------------------------------------------------------------------
    # Thermal sensor commands — OS and tool dependent
    # ------------------------------------------------------------------
    # HW_CPU_TEMP_CMD and HW_GPU_TEMP_CMD must be shell snippets that,
    # when eval'd, print a single integer (°C) to stdout.
    #
    if [[ "${os_type}" == "Darwin" ]]; then
        # macOS: use powermetrics (requires sudo) or osx-cpu-temp if installed
        if command -v osx-cpu-temp &>/dev/null; then
            HW_CPU_TEMP_CMD="osx-cpu-temp | grep -oE '[0-9]+\.[0-9]+' | head -1 | cut -d. -f1"
        else
            # powermetrics needs sudo; skip thermal guard if unavailable
            HW_CPU_TEMP_CMD=""
            warn "[HW] CPU temp monitoring unavailable on macOS without osx-cpu-temp or sudo powermetrics"
        fi
        if [[ "${HW_BACKEND}" == "cuda" ]]; then
            HW_GPU_TEMP_CMD="nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
        else
            # Metal GPU temps not readable without third-party tools; disable guard
            HW_GPU_TEMP_CMD=""
            warn "[HW] GPU temp monitoring unavailable for Metal backend"
        fi
    else  # Linux
        # Try lm-sensors first (covers most AMD/Intel CPUs)
        if command -v sensors &>/dev/null; then
            # AMD: look for Tctl; Intel: look for Package id 0
            if sensors 2>/dev/null | grep -qi "tctl"; then
                HW_CPU_TEMP_CMD="sensors 2>/dev/null | awk '/Tctl/{gsub(/[^0-9.]/,\"\",\$2); printf \"%d\", \$2}'"
            else
                HW_CPU_TEMP_CMD="sensors 2>/dev/null | awk '/Package id 0/{gsub(/[^0-9.]/,\"\",\$4); printf \"%d\", \$4}'"
            fi
        # Fallback: /sys/class/thermal (available on most Linux kernels)
        elif [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
            HW_CPU_TEMP_CMD="awk '{printf \"%d\", \$1/1000}' /sys/class/thermal/thermal_zone0/temp"
        else
            HW_CPU_TEMP_CMD=""
            warn "[HW] CPU temp monitoring unavailable — install lm-sensors (apt install lm-sensors)"
        fi
        # Linux GPU temp: nvidia-smi for CUDA, nothing reliable for others
        if [[ "${HW_BACKEND}" == "cuda" ]]; then
            HW_GPU_TEMP_CMD="nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
        else
            HW_GPU_TEMP_CMD=""
        fi
    fi
    #
    # If either temp command is empty, wait_cool() will skip that check with a warning.

    if [[ -n "${OUTPUT_MODEL_DIR:-}" && -d "${OUTPUT_MODEL_DIR}" ]]; then
        jq -n \
            --arg cpu_model "${HW_CPU_MODEL}" \
            --argjson cpu_physical "${HW_CPU_PHYSICAL}" \
            --argjson cpu_logical "${HW_CPU_LOGICAL}" \
            --argjson ram_gib "${HW_RAM_GIB}" \
            --argjson ram_free "${HW_RAM_FREE_GIB}" \
            --argjson gpu_count "${HW_GPU_COUNT}" \
            --arg gpu_model "${HW_GPU_MODEL}" \
            --argjson gpu_vram "${HW_GPU_VRAM_GIB}" \
            --argjson gpu_vram_free "${HW_GPU_VRAM_FREE_GIB}" \
            --arg backend "${HW_BACKEND}" \
            '{cpu_model:$cpu_model,cpu_physical_cores:$cpu_physical,cpu_logical_threads:$cpu_logical,
              ram_gib:$ram_gib,ram_free_gib_at_start:$ram_free,gpu_count:$gpu_count,
              gpu_model:$gpu_model,gpu_vram_gib:$gpu_vram,gpu_vram_free_gib_at_start:$gpu_vram_free,
              backend:$backend}' > "${OUTPUT_MODEL_DIR}/hardware.json"
    fi
    log "[HW] CPU: ${HW_CPU_MODEL} (${HW_CPU_PHYSICAL}P/${HW_CPU_LOGICAL}L)  RAM: ${HW_RAM_GIB} GiB"
    log "[HW] GPU: ${HW_GPU_MODEL}  VRAM: ${HW_GPU_VRAM_GIB} GiB (${HW_GPU_VRAM_FREE_GIB} free)  Backend: ${HW_BACKEND}"
}

# -----------------------------------------------------------------------------
# detect_turbo_binary
#   Check whether SWEEP_TURBO_BENCH_BIN is set, points to an executable file,
#   and advertises turbo3 support via its --help output.
#   Sets TURBO_AVAILABLE=true on success; logs a warning and leaves it false
#   otherwise.
# -----------------------------------------------------------------------------
detect_turbo_binary() {
    [[ -z "${SWEEP_TURBO_BENCH_BIN}" ]] && return 0
    [[ ! -x "${SWEEP_TURBO_BENCH_BIN}" ]] && warn "turbo-bench not executable" && return 0
    "${SWEEP_TURBO_BENCH_BIN}" --help 2>&1 | grep -qi "turbo3" || {
        warn "turbo-bench binary does not appear to support turbo3"
        return 0
    }
    TURBO_AVAILABLE=true
    log "turbo-bench available: ${SWEEP_TURBO_BENCH_BIN}"
}

# -----------------------------------------------------------------------------
# print_hardware_summary
#   Print a human-readable table of HW_* values to the terminal (not the log).
#   Called once before the sweep begins so the operator can sanity-check the
#   detected hardware before committing to a long run.
# -----------------------------------------------------------------------------
print_hardware_summary() {
    printf '\n'
    printf '┌─────────────────────────────────────────────────────┐\n'
    printf '│  Hardware Summary                                   │\n'
    printf '├────────────────────┬────────────────────────────────┤\n'
    printf '│ %-18s │ %-30s │\n' "CPU" "${HW_CPU_MODEL:0:30}"
    printf '│ %-18s │ %-30s │\n' "Cores" "${HW_CPU_PHYSICAL}P / ${HW_CPU_LOGICAL}L"
    printf '│ %-18s │ %-30s │\n' "RAM" "${HW_RAM_GIB} GiB (${HW_RAM_FREE_GIB} GiB free)"
    printf '│ %-18s │ %-30s │\n' "GPU" "${HW_GPU_MODEL:0:30}"
    printf '│ %-18s │ %-30s │\n' "VRAM" "${HW_GPU_VRAM_GIB} GiB (${HW_GPU_VRAM_FREE_GIB} GiB free)"
    printf '│ %-18s │ %-30s │\n' "Backend" "${HW_BACKEND}"
    printf '│ %-18s │ %-30s │\n' "llama-bench" "$(basename "${LLAMA_BENCH_BIN}")"
    printf '│ %-18s │ %-30s │\n' "TurboQuant" "${TURBO_AVAILABLE}"
    printf '└────────────────────┴────────────────────────────────┘\n'
    printf '\n'
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
    # If --model already populated MODEL_LIST by parse_args, just validate
    if [[ ${#MODEL_LIST[@]} -gt 0 ]]; then
        local m
        for m in "${MODEL_LIST[@]}"; do validate_model "$m"; done
        return 0
    fi
    if [[ -n "${OPT_MODEL_LIST_FILE}" ]]; then
        [[ -f "${OPT_MODEL_LIST_FILE}" ]] || die "Model list file not found: ${OPT_MODEL_LIST_FILE}"
        while IFS= read -r line; do
            [[ -z "${line}" || "${line}" == \#* ]] && continue
            if [[ "${line}" != /* ]] && [[ -n "${SWEEP_MODELS_DIR:-}" ]]; then
                line="${SWEEP_MODELS_DIR%/}/${line}"
            fi
            MODEL_LIST+=("${line}")
        done < "${OPT_MODEL_LIST_FILE}"
    elif [[ -n "${SWEEP_MODELS_DIR:-}" ]]; then
        while IFS= read -r -d '' f; do
            MODEL_LIST+=("$f")
        done < <(find "${SWEEP_MODELS_DIR}" -maxdepth 1 -name "*.gguf" -print0 | sort -z)
    fi
    [[ ${#MODEL_LIST[@]} -eq 0 ]] && die "No models found. Use --model, --models-dir, or --model-list."
    local m
    for m in "${MODEL_LIST[@]}"; do validate_model "$m"; done
    log "Models to sweep (${#MODEL_LIST[@]}):"
    for m in "${MODEL_LIST[@]}"; do log "  ${m}"; done
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
    [[ -f "${path}" ]]    || die "Model not found: ${path}"
    [[ -r "${path}" ]]    || die "Model not readable: ${path}"
    [[ "${path,,}" == *.gguf ]] || die "Model does not appear to be a .gguf file: ${path}"
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
    OUTPUT_MODEL_DIR="${SWEEP_OUTPUT_DIR}/${MODEL_STEM}"
    if [[ -d "${OUTPUT_MODEL_DIR}" ]]; then
        if $OPT_OVERWRITE; then
            rm -rf "${OUTPUT_MODEL_DIR}"
            log "Overwrite: removed existing output dir"
        elif ! $OPT_RESUME; then
            die "Output dir already exists: ${OUTPUT_MODEL_DIR}
  Use --resume to continue an existing sweep, or --overwrite to start fresh."
        fi
    fi
    mkdir -p "${OUTPUT_MODEL_DIR}/raw"
    log "Output directory: ${OUTPUT_MODEL_DIR}"
}

# -----------------------------------------------------------------------------
# load_state
#   If --resume is set and state.json exists in OUTPUT_MODEL_DIR, read the
#   phases_complete array and working_sets map into shell variables so that
#   sweep_model() can skip already-finished phases.
#   No-op when --resume is false or state.json is absent.
# -----------------------------------------------------------------------------
load_state() {
    local state_file="${OUTPUT_MODEL_DIR}/state.json"
    $OPT_RESUME || return 0
    [[ -f "${state_file}" ]] || return 0

    log "[STATE] Resuming from ${state_file}"

    # Read phases_complete
    PHASES_COMPLETE="$(jq -r '.phases_complete // [] | join(" ")' "${state_file}" 2>/dev/null || true)"
    MAX_NGL="$(jq -r '.max_ngl // 99' "${state_file}" 2>/dev/null || echo 99)"
    BEST_NGL="$(jq -r '.best.ngl // 99' "${state_file}" 2>/dev/null || echo 99)"
    BEST_FA="$(jq -r '.best.fa // 0' "${state_file}" 2>/dev/null || echo 0)"
    BEST_CTK="$(jq -r '.best.ctk // "f16"' "${state_file}" 2>/dev/null || echo f16)"
    BEST_CTV="$(jq -r '.best.ctv // "f16"' "${state_file}" 2>/dev/null || echo f16)"
    BEST_THREADS="$(jq -r '.best.threads // ""' "${state_file}" 2>/dev/null || true)"
    BEST_NKVO="$(jq -r '.best.nkvo // 0' "${state_file}" 2>/dev/null || echo 0)"
    BEST_B="$(jq -r '.best.b // 2048' "${state_file}" 2>/dev/null || echo 2048)"
    BEST_UB="$(jq -r '.best.ub // 512' "${state_file}" 2>/dev/null || echo 512)"
    BEST_CTX="$(jq -r '.best.ctx // 512' "${state_file}" 2>/dev/null || echo 512)"

    WS_NGL="$(jq -r '.working_sets.ngl // [] | join(" ")' "${state_file}" 2>/dev/null || true)"
    WS_FA_CTK="$(jq -r '.working_sets.fa_ctk_combos // [] | .[] | "\(.fa) \(.ctk) \(.ctv)"' "${state_file}" 2>/dev/null || true)"
    WS_THREADS="$(jq -r '.working_sets.thread_values // [] | join(" ")' "${state_file}" 2>/dev/null || true)"
    WS_NKVO="$(jq -r '.working_sets.nkvo_values // [] | join(" ")' "${state_file}" 2>/dev/null || true)"
    WS_B_UB="$(jq -r '.working_sets.b_ub_combos // [] | .[] | "\(.b) \(.ub)"' "${state_file}" 2>/dev/null || true)"
    WS_CTX="$(jq -r '.working_sets.ctx_values // [] | join(" ")' "${state_file}" 2>/dev/null || true)"

    log "[STATE] Phases complete: ${PHASES_COMPLETE:-none}"
}

# -----------------------------------------------------------------------------
# save_state PHASE_COMPLETED
#   Append PHASE_COMPLETED to the phases_complete list in state.json.
#   Also serialises current working_sets (best NGL, best KV config, etc.)
#   into the JSON so a resumed run has full context.
#   Creates state.json if it does not exist.
# -----------------------------------------------------------------------------
# _nums_to_json SPACE_OR_NEWLINE_SEPARATED_NUMBERS
#   Converts a list of numbers (space or newline separated) into a JSON array
#   string. Always outputs a valid JSON array — emits "[]" on empty input.
#   No jq required; pure bash arithmetic.
_nums_to_json() {
    local result="[" sep="" val
    while IFS= read -r val; do
        [[ -z "${val}" ]] && continue
        result+="${sep}${val}"
        sep=","
    done <<< "$(echo "${1:-}" | tr ' ' '\n')"
    result+="]"
    echo "${result}"
}

# _strs_to_json SPACE_OR_NEWLINE_SEPARATED_STRINGS
#   Like _nums_to_json but wraps each value in JSON double-quotes.
_strs_to_json() {
    local result="[" sep="" val
    while IFS= read -r val; do
        [[ -z "${val}" ]] && continue
        # escape backslashes and double-quotes
        val="${val//\\/\\\\}"
        val="${val//\"/\\\"}"
        result+="${sep}\"${val}\""
        sep=","
    done <<< "$(echo "${1:-}" | tr ' ' '\n')"
    result+="]"
    echo "${result}"
}

save_state() {
    local phase="${1:-}"
    local state_file="${OUTPUT_MODEL_DIR}/state.json"

    # Add phase to PHASES_COMPLETE
    if [[ -n "${phase}" ]]; then
        if [[ -z "${PHASES_COMPLETE}" ]]; then
            PHASES_COMPLETE="${phase}"
        elif ! echo " ${PHASES_COMPLETE} " | grep -q " ${phase} "; then
            PHASES_COMPLETE="${PHASES_COMPLETE} ${phase}"
        fi
    fi

    # Build JSON arrays using pure-bash helpers (no jq pipelines — avoids empty-output edge cases)
    local phases_json ngl_json nkvo_json ctx_json thread_json fa_ctk_json b_ub_json
    phases_json="$(_nums_to_json "${PHASES_COMPLETE}")"
    ngl_json="$(_nums_to_json "${WS_NGL}")"
    nkvo_json="$(_nums_to_json "${WS_NKVO}")"
    ctx_json="$(_nums_to_json "${WS_CTX}")"
    # Threads can contain "system_default" so keep as strings
    thread_json="$(_strs_to_json "${WS_THREADS}")"

    # fa_ctk combos: newline-separated "fa ctk ctv" triples
    local fa_ctk_json="["
    local fa_ctk_sep=""
    local _line
    while IFS= read -r _line; do
        [[ -z "${_line}" ]] && continue
        local _fa _ctk _ctv
        read -r _fa _ctk _ctv <<< "${_line}"
        local _obj
        _obj="$(jq -cn --argjson fa "${_fa}" --arg ctk "${_ctk}" --arg ctv "${_ctv}" \
            '{fa:$fa,ctk:$ctk,ctv:$ctv}')"
        fa_ctk_json+="${fa_ctk_sep}${_obj}"
        fa_ctk_sep=","
    done <<< "${WS_FA_CTK}"
    fa_ctk_json+="]"

    # b_ub combos: newline-separated "b ub" pairs
    local b_ub_json="["
    local b_ub_sep=""
    while IFS= read -r _line; do
        [[ -z "${_line}" ]] && continue
        local _b _ub
        read -r _b _ub <<< "${_line}"
        local _obj
        _obj="$(jq -cn --argjson b "${_b}" --argjson ub "${_ub}" '{b:$b,ub:$ub}')"
        b_ub_json+="${b_ub_sep}${_obj}"
        b_ub_sep=","
    done <<< "${WS_B_UB}"
    b_ub_json+="]"

    jq -n \
        --arg model_path "${MODEL_PATH}" \
        --arg model_stem "${MODEL_STEM}" \
        --argjson max_ngl "${MAX_NGL}" \
        --argjson phases "${phases_json}" \
        --argjson ngl_ws "${ngl_json}" \
        --argjson fa_ctk_ws "${fa_ctk_json}" \
        --argjson thread_ws "${thread_json}" \
        --argjson nkvo_ws "${nkvo_json}" \
        --argjson b_ub_ws "${b_ub_json}" \
        --argjson ctx_ws "${ctx_json}" \
        --argjson best_ngl "${BEST_NGL}" \
        --argjson best_fa "${BEST_FA}" \
        --arg best_ctk "${BEST_CTK}" \
        --arg best_ctv "${BEST_CTV}" \
        --arg best_threads "${BEST_THREADS}" \
        --argjson best_nkvo "${BEST_NKVO}" \
        --argjson best_b "${BEST_B}" \
        --argjson best_ub "${BEST_UB}" \
        --argjson best_ctx "${BEST_CTX}" \
        '{
          model_path: $model_path,
          model_stem: $model_stem,
          max_ngl: $max_ngl,
          phases_complete: $phases,
          best: {
            ngl: $best_ngl, fa: $best_fa, ctk: $best_ctk, ctv: $best_ctv,
            threads: (if $best_threads == "" then null else ($best_threads|tonumber) end),
            nkvo: $best_nkvo, b: $best_b, ub: $best_ub, ctx: $best_ctx
          },
          working_sets: {
            ngl: $ngl_ws,
            fa_ctk_combos: $fa_ctk_ws,
            thread_values: $thread_ws,
            nkvo_values: $nkvo_ws,
            b_ub_combos: $b_ub_ws,
            ctx_values: $ctx_ws
          }
        }' > "${state_file}"
}

# -----------------------------------------------------------------------------
# wait_cool
#   Poll CPU and GPU temperatures until both are below their respective limits.
#   Prints a waiting message each poll interval so the operator knows it is
#   not hung.
#   No-op when OPT_NO_THERMAL=true or when the temperature commands are empty.
# -----------------------------------------------------------------------------
wait_cool() {
    $OPT_NO_THERMAL && return 0
    local cpu_temp gpu_temp
    while true; do
        if [[ -n "${HW_CPU_TEMP_CMD}" ]]; then
            cpu_temp="$(eval "${HW_CPU_TEMP_CMD}" 2>/dev/null || echo 0)"
            cpu_temp="${cpu_temp:-0}"
        else
            cpu_temp=0
        fi
        if [[ -n "${HW_GPU_TEMP_CMD}" ]]; then
            gpu_temp="$(eval "${HW_GPU_TEMP_CMD}" 2>/dev/null || echo 0)"
            gpu_temp="${gpu_temp:-0}"
        else
            gpu_temp=0
        fi
        # Ensure numeric
        [[ "${cpu_temp}" =~ ^[0-9]+$ ]] || cpu_temp=0
        [[ "${gpu_temp}" =~ ^[0-9]+$ ]] || gpu_temp=0
        if [[ "${cpu_temp}" -lt "${SWEEP_CPU_TEMP_LIMIT}" && "${gpu_temp}" -lt "${SWEEP_GPU_TEMP_LIMIT}" ]]; then
            break
        fi
        log "Thermal wait: CPU=${cpu_temp}°C GPU=${gpu_temp}°C — sleeping ${SWEEP_COOL_POLL_SEC}s"
        sleep "${SWEEP_COOL_POLL_SEC}"
    done
}

# -----------------------------------------------------------------------------
# detect_oom LOG_FILE
#   Scan LOG_FILE for common OOM / fatal-error strings.
#   Returns 0 (true) if an OOM/error pattern is found, 1 otherwise.
# -----------------------------------------------------------------------------
detect_oom() {
    local log_file="${1:-}"
    [[ -f "${log_file}" ]] || return 1
    grep -qiE "(out of memory|failed to allocate|ggml_cuda_pool_alloc|CUDA error|cudaMalloc failed|ggml_backend_alloc|Cannot allocate memory|Killed|Segmentation fault|bus error|terminate called|GGML_ASSERT|failed to load model)" "${log_file}"
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
    if [[ "${ctk}" == turbo* ]]; then
        $TURBO_AVAILABLE || die "turbo KV type requested but turbo-bench not available"
        echo "${SWEEP_TURBO_BENCH_BIN}"
    else
        echo "${LLAMA_BENCH_BIN}"
    fi
}

# -----------------------------------------------------------------------------
# write_jsonl_record KEY=VALUE...
#   Append a single JSON object to ${OUTPUT_MODEL_DIR}/sweep.jsonl.
# -----------------------------------------------------------------------------
write_jsonl_record() {
    # Usage: write_jsonl_record key=value ...
    # Keys: run_id ts status viable phase phase_label binary
    #       ngl fa ctk ctv nkvo threads threads_is_default b ub n_prompt n_gen reps
    #       pp_ts pp_stddev tg_ts tg_stddev raw_output_file error_snippet
    local run_id="" ts="" status="ok" viable="" phase=0 phase_label="" binary="standard"
    local ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}"
    local nkvo="${BEST_NKVO}" threads="" threads_is_default="true"
    local b="${BEST_B}" ub="${BEST_UB}" n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}"
    local pp_ts="0" pp_stddev="0" tg_ts="0" tg_stddev="0"
    local raw_output_file="" error_snippet=""

    for kv in "$@"; do
        local k="${kv%%=*}" v="${kv#*=}"
        case "${k}" in
            run_id)             run_id="${v}" ;;
            ts)                 ts="${v}" ;;
            status)             status="${v}" ;;
            viable)             viable="${v}" ;;
            phase)              phase="${v}" ;;
            phase_label)        phase_label="${v}" ;;
            binary)             binary="${v}" ;;
            ngl)                ngl="${v}" ;;
            fa)                 fa="${v}" ;;
            ctk)                ctk="${v}" ;;
            ctv)                ctv="${v}" ;;
            nkvo)               nkvo="${v}" ;;
            threads)            threads="${v}" ;;
            threads_is_default) threads_is_default="${v}" ;;
            b)                  b="${v}" ;;
            ub)                 ub="${v}" ;;
            n_prompt)           n_prompt="${v}" ;;
            n_gen)              n_gen="${v}" ;;
            reps)               reps="${v}" ;;
            pp_ts)              pp_ts="${v}" ;;
            pp_stddev)          pp_stddev="${v}" ;;
            tg_ts)              tg_ts="${v}" ;;
            tg_stddev)          tg_stddev="${v}" ;;
            raw_output_file)    raw_output_file="${v}" ;;
            error_snippet)      error_snippet="${v}" ;;
        esac
    done

    [[ -z "${ts}" ]] && ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ -n "${threads}" ]]; then threads_is_default="false"; else threads_is_default="true"; fi

    # viable JSON value
    local viable_json
    case "${viable}" in
        true|false) viable_json="${viable}" ;;
        *)          viable_json="null" ;;
    esac

    # threads JSON value
    local threads_json
    if [[ -z "${threads}" ]]; then threads_json="null"; else threads_json="${threads}"; fi

    # results JSON array
    local results_json="[]"
    if [[ "${status}" == "ok" ]]; then
        local arr=""
        if [[ "${n_prompt}" -gt 0 && "${pp_ts}" != "0" ]]; then
            arr+='{"test":"pp","n_prompt":'"${n_prompt}"',"n_gen":0,"avg_ts":'"${pp_ts}"',"stddev_ts":'"${pp_stddev}"'}'
        fi
        if [[ "${n_gen}" -gt 0 && "${tg_ts}" != "0" ]]; then
            [[ -n "${arr}" ]] && arr+=","
            arr+='{"test":"tg","n_prompt":0,"n_gen":'"${n_gen}"',"avg_ts":'"${tg_ts}"',"stddev_ts":'"${tg_stddev}"'}'
        fi
        [[ -n "${arr}" ]] && results_json="[${arr}]"
    fi

    jq -cn \
        --arg run_id "${run_id}" \
        --arg ts "${ts}" \
        --arg model_path "${MODEL_PATH}" \
        --arg model_stem "${MODEL_STEM}" \
        --argjson phase "${phase}" \
        --arg phase_label "${phase_label}" \
        --arg binary "${binary}" \
        --arg status "${status}" \
        --argjson viable "${viable_json}" \
        --argjson ngl "${ngl}" \
        --argjson fa "${fa}" \
        --arg ctk "${ctk}" \
        --arg ctv "${ctv}" \
        --argjson nkvo "${nkvo}" \
        --argjson threads "${threads_json}" \
        --argjson threads_is_default "${threads_is_default}" \
        --argjson b "${b}" \
        --argjson ub "${ub}" \
        --argjson n_prompt "${n_prompt}" \
        --argjson n_gen "${n_gen}" \
        --argjson reps "${reps}" \
        --argjson results "${results_json}" \
        --arg raw_output_file "${raw_output_file}" \
        --arg error_snippet "${error_snippet}" \
        '{run_id:$run_id,timestamp:$ts,model_path:$model_path,model_stem:$model_stem,
          phase:$phase,phase_label:$phase_label,binary:$binary,status:$status,viable:$viable,
          params:{ngl:$ngl,fa:$fa,ctk:$ctk,ctv:$ctv,nkvo:$nkvo,threads:$threads,
                  threads_is_default:$threads_is_default,b:$b,ub:$ub,
                  n_prompt:$n_prompt,n_gen:$n_gen,repetitions:$reps},
          results:$results,
          raw_output_file:(if $raw_output_file=="" then null else $raw_output_file end),
          error_snippet:(if $error_snippet=="" then null else $error_snippet end)
        }' >> "${OUTPUT_MODEL_DIR}/sweep.jsonl"
}

# -----------------------------------------------------------------------------
# run_bench LABEL KEY=VALUE...
#   The central bench execution wrapper.
# -----------------------------------------------------------------------------
run_bench() {
    local label="${1:-unknown}"
    shift

    # Defaults from current best config
    local ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}"
    local nkvo="${BEST_NKVO}" threads="" b="${BEST_B}" ub="${BEST_UB}"
    local n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}"
    local phase=0 phase_label="unknown"

    for kv in "$@"; do
        local k="${kv%%=*}" v="${kv#*=}"
        case "${k}" in
            ngl)          ngl="${v}" ;;
            fa)           fa="${v}" ;;
            ctk)          ctk="${v}" ;;
            ctv)          ctv="${v}" ;;
            nkvo)         nkvo="${v}" ;;
            threads)      threads="${v}" ;;
            b)            b="${v}" ;;
            ub)           ub="${v}" ;;
            n_prompt)     n_prompt="${v}" ;;
            n_gen)        n_gen="${v}" ;;
            reps)         reps="${v}" ;;
            phase)        phase="${v}" ;;
            phase_label)  phase_label="${v}" ;;
        esac
    done

    local binary
    binary="$(select_binary "${ctk}")"
    local binary_label="standard"
    [[ "${binary}" == "${SWEEP_TURBO_BENCH_BIN}" ]] && binary_label="turboquant"

    local run_id
    run_id="$(gen_run_id)"

    local raw_dir="${OUTPUT_MODEL_DIR}/raw"
    local raw_file="${raw_dir}/${run_id}.txt"
    local stderr_file="${raw_dir}/${run_id}.err"

    # Build command array
    local cmd=( "${binary}" -m "${MODEL_PATH}" )
    cmd+=( -ngl "${ngl}" )
    cmd+=( -fa "${fa}" )
    cmd+=( -ctk "${ctk}" -ctv "${ctv}" )
    cmd+=( -nkvo "${nkvo}" )
    [[ -n "${threads}" ]] && cmd+=( -t "${threads}" )
    cmd+=( -b "${b}" -ub "${ub}" )
    cmd+=( -p "${n_prompt}" -n "${n_gen}" )
    cmd+=( -r "${reps}" )
    cmd+=( -o jsonl )
    cmd+=( --prio "${SWEEP_PRIO}" )

    local threads_display="${threads:-sys}"
    log "[run] ${label} | ngl=${ngl} fa=${fa} ctk=${ctk} nkvo=${nkvo} t=${threads_display} b=${b} ub=${ub} ctx=${n_prompt} gen=${n_gen}"

    if $OPT_DRY_RUN; then
        log "[dry-run] ${cmd[*]}"
        echo "dry-run"
        return 0
    fi

    wait_cool
    sleep "${SWEEP_DELAY_SEC}"

    local exit_code=0
    timeout "${SWEEP_TIMEOUT_SEC}" "${cmd[@]}" > "${raw_file}" 2>"${stderr_file}" || exit_code=$?

    # Timeout
    if [[ ${exit_code} -eq 124 ]]; then
        log "[TIMEOUT] ${label} — killed after ${SWEEP_TIMEOUT_SEC}s"
        write_jsonl_record \
            run_id="${run_id}" phase="${phase}" phase_label="${phase_label}" \
            binary="${binary_label}" status="timeout" viable="false" \
            ngl="${ngl}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" nkvo="${nkvo}" \
            threads="${threads}" b="${b}" ub="${ub}" \
            n_prompt="${n_prompt}" n_gen="${n_gen}" reps="${reps}" \
            error_snippet="$(head -c 400 "${stderr_file}" 2>/dev/null || true)"
        echo "timeout"
        return 0
    fi

    # OOM detection — check both stdout (some errors go there) and stderr
    local combined="${raw_dir}/${run_id}.combined"
    cat "${raw_file}" "${stderr_file}" > "${combined}" 2>/dev/null || true

    if detect_oom "${combined}"; then
        local err_snip
        err_snip="$(grep -iEm1 '(out of memory|CUDA error|failed to allocate|Killed)' "${combined}" 2>/dev/null | head -c 400 || true)"
        log "[OOM] ${label}"
        write_jsonl_record \
            run_id="${run_id}" phase="${phase}" phase_label="${phase_label}" \
            binary="${binary_label}" status="oom" viable="false" \
            ngl="${ngl}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" nkvo="${nkvo}" \
            threads="${threads}" b="${b}" ub="${ub}" \
            n_prompt="${n_prompt}" n_gen="${n_gen}" reps="${reps}" \
            error_snippet="${err_snip}"
        rm -f "${combined}"
        echo "oom"
        return 0
    fi
    rm -f "${combined}"

    # Parse JSON output from llama-bench -o jsonl
    local pp_ts pp_stddev tg_ts tg_stddev
    pp_ts="$(jq -r 'select(.n_gen==0 and .n_prompt>0) | .avg_ts' "${raw_file}" 2>/dev/null | tail -1)"
    pp_stddev="$(jq -r 'select(.n_gen==0 and .n_prompt>0) | .stddev_ts' "${raw_file}" 2>/dev/null | tail -1)"
    tg_ts="$(jq -r 'select(.n_gen>0) | .avg_ts' "${raw_file}" 2>/dev/null | tail -1)"
    tg_stddev="$(jq -r 'select(.n_gen>0) | .stddev_ts' "${raw_file}" 2>/dev/null | tail -1)"

    # Default nulls to 0
    pp_ts="${pp_ts:-0}"; pp_ts="${pp_ts/null/0}"
    pp_stddev="${pp_stddev:-0}"; pp_stddev="${pp_stddev/null/0}"
    tg_ts="${tg_ts:-0}"; tg_ts="${tg_ts/null/0}"
    tg_stddev="${tg_stddev:-0}"; tg_stddev="${tg_stddev/null/0}"

    # Check we got some output
    if [[ "${pp_ts}" == "0" && "${tg_ts}" == "0" && ${exit_code} -ne 0 ]]; then
        local err_snip
        err_snip="$(head -c 400 "${stderr_file}" 2>/dev/null || true)"
        log "[ERROR] ${label} — exit ${exit_code}, no results parsed"
        write_jsonl_record \
            run_id="${run_id}" phase="${phase}" phase_label="${phase_label}" \
            binary="${binary_label}" status="error" viable="false" \
            ngl="${ngl}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" nkvo="${nkvo}" \
            threads="${threads}" b="${b}" ub="${ub}" \
            n_prompt="${n_prompt}" n_gen="${n_gen}" reps="${reps}" \
            error_snippet="${err_snip}"
        echo "error"
        return 0
    fi

    # Viability check
    local viable="null"
    if [[ "${n_gen}" -gt 0 ]]; then
        if awk "BEGIN{exit !(${tg_ts}+0 >= ${SWEEP_MIN_TG_TS}+0)}"; then
            viable="true"
        else
            viable="false"
        fi
    fi

    local tg_display="${tg_ts}"
    [[ "${n_gen}" -eq 0 ]] && tg_display="n/a"
    log "[ok] ${label} | PP=${pp_ts} t/s  TG=${tg_display} t/s"

    write_jsonl_record \
        run_id="${run_id}" phase="${phase}" phase_label="${phase_label}" \
        binary="${binary_label}" status="ok" viable="${viable}" \
        ngl="${ngl}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" nkvo="${nkvo}" \
        threads="${threads}" b="${b}" ub="${ub}" \
        n_prompt="${n_prompt}" n_gen="${n_gen}" reps="${reps}" \
        pp_ts="${pp_ts}" pp_stddev="${pp_stddev}" \
        tg_ts="${tg_ts}" tg_stddev="${tg_stddev}" \
        raw_output_file="raw/${run_id}.txt"

    echo "ok"
    return 0
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

# _best_fa_ctv_for_ctk CTK
#
# Given a KV cache type string, scans WS_FA_CTK and returns the best matching
# "fa ctv" pair for that ctk — preferring fa=1 over fa=0 when both exist.
# Prints "fa ctv" to stdout; returns 1 if the ctk is not in WS_FA_CTK.
_best_fa_ctv_for_ctk() {
    local target_ctk="$1"
    local best_fa="" best_ctv=""
    local line fa ctk ctv
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        read -r fa ctk ctv <<< "${line}"
        if [[ "${ctk}" == "${target_ctk}" ]]; then
            if [[ -z "${best_fa}" || "${fa}" == "1" ]]; then
                best_fa="${fa}"; best_ctv="${ctv}"
            fi
        fi
    done <<< "${WS_FA_CTK}"
    [[ -n "${best_fa}" ]] && echo "${best_fa} ${best_ctv}" && return 0
    return 1
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
            warn "Unknown --min-ctk value '${min_val}' — no filtering applied"
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
#   Linear step-down from 99, finding the highest NGL that loads without OOM.
#   Sets MAX_NGL and BEST_NGL globals.
#   Skipped entirely when HW_GPU_COUNT=0 (CPU-only inference).
# -----------------------------------------------------------------------------
phase0_ngl_probe() {
    log "[Phase 0] NGL probe — finding max stable NGL"
    if [[ "${HW_GPU_COUNT}" -eq 0 ]]; then
        log "[Phase 0] No GPU detected — skipping probe, setting MAX_NGL=0"
        MAX_NGL=0
        BEST_NGL=0
        return 0
    fi

    local ngl=99
    while [[ ${ngl} -ge 0 ]]; do
        log "[Phase 0] Probing ngl=${ngl}"
        local status
        status="$(run_bench "phase0/ngl=${ngl}" \
            ngl="${ngl}" fa=0 ctk=f16 ctv=f16 nkvo=0 \
            n_prompt=64 n_gen=0 reps="${SWEEP_PROBE_REPS}" \
            phase=0 phase_label="ngl_probe")"
        if [[ "${status}" == "ok" || "${status}" == "dry-run" ]]; then
            MAX_NGL="${ngl}"
            BEST_NGL="${ngl}"
            log "[Phase 0] max_ngl=${ngl}"
            return 0
        fi
        ngl=$(( ngl - 4 ))
    done

    die "Model cannot be loaded at any ngl value on this hardware."
}

# -----------------------------------------------------------------------------
# phase1_ngl_sweep
#   Sweep NGL from a smart default start (near MAX_NGL) to MAX_NGL.
#   Best TG t/s config becomes BEST_NGL for subsequent phases.
#
#   Default start: MAX_NGL - 2*step (tests ~3 high values + exact max).
#   Very-low NGL values are rarely interesting at large contexts; set
#   --start-ngl 0 (or SWEEP_START_NGL=0) for a full 0→MAX_NGL sweep.
# -----------------------------------------------------------------------------
phase1_ngl_sweep() {
    log "[Phase 1] NGL axis sweep (step=${SWEEP_NGL_STEP})"

    # Build full NGL list: 0, step, 2*step, ..., MAX_NGL (deduplicated, sorted)
    local -a full_list=()
    full_list+=(0)
    local n
    for (( n=SWEEP_NGL_STEP; n<MAX_NGL; n+=SWEEP_NGL_STEP )); do
        full_list+=("${n}")
    done
    full_list+=("${MAX_NGL}")
    # Deduplicate
    local -A seen=()
    local -a deduped=()
    local v
    for v in "${full_list[@]}"; do
        [[ -z "${seen[$v]+x}" ]] && { deduped+=("${v}"); seen[$v]=1; }
    done

    # Smart default start: begin 2 steps below MAX_NGL so Phase 1 characterises
    # the high-performance region rather than burning time on low-ngl configs.
    # Snapped to a step boundary so it always hits a list member.
    # Override with --start-ngl 0 for a full sweep.
    local effective_start_ngl="${OPT_START_NGL}"
    if [[ -z "${effective_start_ngl}" ]]; then
        local steps_from_max=2
        local smart_start=$(( (MAX_NGL / SWEEP_NGL_STEP - steps_from_max) * SWEEP_NGL_STEP ))
        [[ ${smart_start} -lt 0 ]] && smart_start=0
        effective_start_ngl="${smart_start}"
        log "[Phase 1] Auto start-ngl=${effective_start_ngl} (max_ngl=${MAX_NGL} − ${steps_from_max}×step); use --start-ngl 0 for full sweep"
    fi

    # Apply start/direction
    local ngl_list
    ngl_list="$(apply_axis_opts "${deduped[*]}" "${effective_start_ngl}" "${OPT_DIR_NGL}")"

    local best_tg=-1
    WS_NGL=""

    while IFS= read -r ngl; do
        [[ -z "${ngl}" ]] && continue
        local status
        status="$(run_bench "phase1/ngl=${ngl}" \
            ngl="${ngl}" fa=0 ctk=f16 ctv=f16 nkvo=0 \
            n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
            phase=1 phase_label="ngl_sweep")"
        if [[ "${status}" == "ok" ]]; then
            WS_NGL="${WS_NGL:+${WS_NGL} }${ngl}"
            # Track best TG
            local tg
            tg="$(jq -s --argjson ph 1 '[.[] | select(.phase==$ph and .params.ngl=='"${ngl}"' and .status=="ok")] | sort_by(-.results[]?.avg_ts) | .[0].results[]? | select(.test=="tg") | .avg_ts' "${OUTPUT_MODEL_DIR}/sweep.jsonl" 2>/dev/null | tail -1 || echo 0)"
            tg="${tg:-0}"; tg="${tg/null/0}"
            if awk "BEGIN{exit !(${tg}+0 > ${best_tg}+0)}"; then
                best_tg="${tg}"
                BEST_NGL="${ngl}"
            fi
        fi
    done <<< "${ngl_list}"

    # Ensure MAX_NGL is in working set if it produced ok
    [[ -z "${WS_NGL}" ]] && WS_NGL="${MAX_NGL}"
    log "[Phase 1] Best NGL: ${BEST_NGL} (TG=${best_tg} t/s)  Working set: ${WS_NGL}"
}

# -----------------------------------------------------------------------------
# phase2_fa_kv_sweep
#   Test flash-attention × KV-quant combinations at BEST_NGL.
#   Best TG t/s combo becomes BEST_FA / BEST_CTK / BEST_CTV.
# -----------------------------------------------------------------------------
phase2_fa_kv_sweep() {
    log "[Phase 2] Flash attention × KV quant sweep"

    # Standard combos: "fa ctk ctv"
    local -a combos=(
        "0 f16 f16"
        "1 f16 f16"
        "0 q8_0 q8_0"
        "1 q8_0 q8_0"
        "1 q4_0 q4_0"
    )
    # Turbo combos
    # turbo3/turbo4: FA is auto-enabled internally by turbo-llama-bench even when
    # -fa 0 is passed, so testing fa=0 produces ambiguous output. Only run fa=1.
    # turbo2: genuinely works without FA, so test both fa=0 and fa=1.
    if $TURBO_AVAILABLE; then
        combos+=(
            "1 turbo4 turbo4"
            "1 turbo3 turbo3"
            "0 turbo2 turbo2"
            "1 turbo2 turbo2"
        )
    fi

    # Apply FA direction filter
    local fa_start="${OPT_START_FA}"
    local fa_dir="${OPT_DIR_FA}"
    local fa_order
    fa_order="$(apply_axis_opts "0 1" "${fa_start}" "${fa_dir}")"

    # Apply CTK direction filter
    local ctk_full_order="f16 q8_0 q4_0 turbo4 turbo3 turbo2"
    local ctk_filtered
    ctk_filtered="$(apply_axis_opts "${ctk_full_order}" "${OPT_START_CTK}" "${OPT_DIR_CTK}")"

    local best_tg=-1
    WS_FA_CTK=""

    local combo
    for combo in "${combos[@]}"; do
        local fa ctk ctv
        read -r fa ctk ctv <<< "${combo}"

        # Filter by FA direction
        echo "${fa_order}" | grep -qw "${fa}" || continue
        # Filter by CTK direction
        echo "${ctk_filtered}" | grep -qw "${ctk}" || continue
        # Skip fa=0 + q4_0 (invalid combo)
        [[ "${fa}" == "0" && "${ctk}" == "q4_0" ]] && continue

        local status
        status="$(run_bench "phase2/fa=${fa}_ctk=${ctk}" \
            ngl="${BEST_NGL}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" nkvo=0 \
            n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
            phase=2 phase_label="fa_kv_sweep")"

        if [[ "${status}" == "ok" ]]; then
            WS_FA_CTK="${WS_FA_CTK:+${WS_FA_CTK}
}${fa} ${ctk} ${ctv}"
            local tg
            tg="$(jq -rs --argjson ph 2 --argjson fa "${fa}" --arg ctk "${ctk}" \
                '[.[] | select(.phase==$ph and .params.fa==$fa and .params.ctk==$ctk and .status=="ok")] | .[0].results[]? | select(.test=="tg") | .avg_ts' \
                "${OUTPUT_MODEL_DIR}/sweep.jsonl" 2>/dev/null | tail -1 || echo 0)"
            tg="${tg:-0}"; tg="${tg/null/0}"
            if awk "BEGIN{exit !(${tg}+0 > ${best_tg}+0)}"; then
                best_tg="${tg}"
                BEST_FA="${fa}"
                BEST_CTK="${ctk}"
                BEST_CTV="${ctv}"
            fi
        fi
    done

    [[ -z "${WS_FA_CTK}" ]] && WS_FA_CTK="0 f16 f16"
    log "[Phase 2] Best: fa=${BEST_FA} ctk=${BEST_CTK} (TG=${best_tg} t/s)"
}

# -----------------------------------------------------------------------------
# phase3_thread_sweep
#   Sweep CPU thread counts at BEST_NGL/FA/CTK to find optimal thread count.
#   Best TG t/s config becomes BEST_THREADS.
# -----------------------------------------------------------------------------
phase3_thread_sweep() {
    log "[Phase 3] CPU thread count sweep"

    # Build thread list from hardware
    local -a thread_candidates=(1)
    local half=$(( HW_CPU_PHYSICAL / 2 ))
    [[ ${half} -gt 1 ]] && thread_candidates+=("${half}")
    thread_candidates+=("${HW_CPU_PHYSICAL}")
    local three_qtr=$(( (HW_CPU_PHYSICAL + HW_CPU_LOGICAL) / 2 ))
    [[ ${three_qtr} -gt ${HW_CPU_PHYSICAL} && ${three_qtr} -lt ${HW_CPU_LOGICAL} ]] && thread_candidates+=("${three_qtr}")
    [[ ${HW_CPU_LOGICAL} -gt ${HW_CPU_PHYSICAL} ]] && thread_candidates+=("${HW_CPU_LOGICAL}")
    # Deduplicate and sort
    local -a full_thread_list=()
    local t
    for t in $(printf '%s\n' "${thread_candidates[@]}" | sort -n | uniq); do
        full_thread_list+=("${t}")
    done

    local thread_list
    thread_list="$(apply_axis_opts "${full_thread_list[*]}" "${OPT_START_THREADS}" "${OPT_DIR_THREADS}")"

    local best_tg=-1
    WS_THREADS=""

    # System default run (no -t flag)
    local status
    status="$(run_bench "phase3/threads=system_default" \
        ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" nkvo=0 \
        n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
        phase=3 phase_label="thread_sweep")"
    if [[ "${status}" == "ok" ]]; then
        WS_THREADS="system_default"
    fi

    while IFS= read -r t; do
        [[ -z "${t}" ]] && continue
        status="$(run_bench "phase3/threads=${t}" \
            ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" nkvo=0 \
            threads="${t}" n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
            phase=3 phase_label="thread_sweep")"
        if [[ "${status}" == "ok" ]]; then
            WS_THREADS="${WS_THREADS:+${WS_THREADS} }${t}"
            local tg
            tg="$(jq -rs --argjson ph 3 --argjson t "${t}" \
                '[.[] | select(.phase==$ph and .params.threads==$t and .status=="ok")] | .[0].results[]? | select(.test=="tg") | .avg_ts' \
                "${OUTPUT_MODEL_DIR}/sweep.jsonl" 2>/dev/null | tail -1 || echo 0)"
            tg="${tg:-0}"; tg="${tg/null/0}"
            if awk "BEGIN{exit !(${tg}+0 > ${best_tg}+0)}"; then
                best_tg="${tg}"
                BEST_THREADS="${t}"
            fi
        fi
    done <<< "${thread_list}"

    [[ -z "${WS_THREADS}" ]] && WS_THREADS="system_default"
    log "[Phase 3] Best threads: ${BEST_THREADS:-system_default} (TG=${best_tg} t/s)"
}

# -----------------------------------------------------------------------------
# phase4_nkvo_sweep
#   Test nkvo=0 vs nkvo=1 at best config; also probe higher NGL with nkvo=1.
#   Best config becomes BEST_NKVO.
# -----------------------------------------------------------------------------
phase4_nkvo_sweep() {
    log "[Phase 4] KV offload sweep (nkvo)"
    local best_tg=-1
    WS_NKVO=""

    local nkvo
    for nkvo in 0 1; do
        local status
        status="$(run_bench "phase4/nkvo=${nkvo}" \
            ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" \
            threads="${BEST_THREADS}" nkvo="${nkvo}" \
            n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
            phase=4 phase_label="nkvo_sweep")"
        if [[ "${status}" == "ok" ]]; then
            WS_NKVO="${WS_NKVO:+${WS_NKVO} }${nkvo}"
            local tg
            tg="$(jq -rs --argjson ph 4 --argjson nkvo "${nkvo}" \
                '[.[] | select(.phase==$ph and .params.nkvo==$nkvo and .status=="ok")] | .[0].results[]? | select(.test=="tg") | .avg_ts' \
                "${OUTPUT_MODEL_DIR}/sweep.jsonl" 2>/dev/null | tail -1 || echo 0)"
            tg="${tg:-0}"; tg="${tg/null/0}"
            if awk "BEGIN{exit !(${tg}+0 > ${best_tg}+0)}"; then
                best_tg="${tg}"
                BEST_NKVO="${nkvo}"
            fi
        fi
    done

    # Also test nkvo=1 at higher NGL values (may unlock more layers)
    if [[ ${MAX_NGL} -lt 99 ]]; then
        local extra_ngl
        for extra_ngl in $(( MAX_NGL + 4 )) $(( MAX_NGL + 8 )) $(( MAX_NGL + 12 )); do
            [[ ${extra_ngl} -gt 99 ]] && break
            local s
            s="$(run_bench "phase4/nkvo=1_ngl=${extra_ngl}" \
                ngl="${extra_ngl}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" \
                threads="${BEST_THREADS}" nkvo=1 \
                n_prompt=512 n_gen=128 reps="${SWEEP_REPETITIONS}" \
                phase=4 phase_label="nkvo_sweep")"
            [[ "${s}" == "oom" || "${s}" == "error" || "${s}" == "timeout" ]] && break
        done
    fi

    [[ -z "${WS_NKVO}" ]] && WS_NKVO="0"
    log "[Phase 4] Best nkvo: ${BEST_NKVO} (TG=${best_tg} t/s)"
}

# -----------------------------------------------------------------------------
# phase5_batch_sweep
#   Test batch (b) and micro-batch (ub) size pairs at best config.
#   Best PP t/s pair becomes BEST_B / BEST_UB.
# -----------------------------------------------------------------------------
phase5_batch_sweep() {
    log "[Phase 5] Batch / ubatch sweep"

    local -a b_ub_pairs=(
        "2048 512"
        "2048 256"
        "2048 128"
        "1024 512"
        "1024 256"
        "1024 128"
        "512 256"
        "512 128"
    )

    local best_pp=-1
    WS_B_UB=""

    local pair
    for pair in "${b_ub_pairs[@]}"; do
        local b ub
        read -r b ub <<< "${pair}"

        # Apply start/direction filters
        local b_filtered ub_filtered
        b_filtered="$(apply_axis_opts "512 1024 2048" "${OPT_START_B}" "${OPT_DIR_B}")"
        ub_filtered="$(apply_axis_opts "128 256 512" "${OPT_START_UB}" "${OPT_DIR_UB}")"
        echo "${b_filtered}" | grep -qw "${b}" || continue
        echo "${ub_filtered}" | grep -qw "${ub}" || continue
        [[ ${ub} -gt ${b} ]] && continue

        local status
        status="$(run_bench "phase5/b=${b}_ub=${ub}" \
            ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" \
            threads="${BEST_THREADS}" nkvo="${BEST_NKVO}" \
            b="${b}" ub="${ub}" n_prompt=512 n_gen=0 reps="${SWEEP_REPETITIONS}" \
            phase=5 phase_label="batch_sweep")"
        if [[ "${status}" == "ok" ]]; then
            WS_B_UB="${WS_B_UB:+${WS_B_UB}
}${b} ${ub}"
            local pp
            pp="$(jq -rs --argjson ph 5 --argjson b "${b}" --argjson ub "${ub}" \
                '[.[] | select(.phase==$ph and .params.b==$b and .params.ub==$ub and .status=="ok")] | .[0].results[]? | select(.test=="pp") | .avg_ts' \
                "${OUTPUT_MODEL_DIR}/sweep.jsonl" 2>/dev/null | tail -1 || echo 0)"
            pp="${pp:-0}"; pp="${pp/null/0}"
            if awk "BEGIN{exit !(${pp}+0 > ${best_pp}+0)}"; then
                best_pp="${pp}"
                BEST_B="${b}"
                BEST_UB="${ub}"
            fi
        fi
    done

    [[ -z "${WS_B_UB}" ]] && WS_B_UB="2048 512"
    log "[Phase 5] Best batch: b=${BEST_B} ub=${BEST_UB} (PP=${best_pp} t/s)"
}

# -----------------------------------------------------------------------------
# phase6_ctx_sweep
#   Sweep context window sizes, starting with the best config from phases 1–5.
#   When a context OOMs or errors, tries progressively more memory-friendly
#   fallback configs before giving up on that size:
#     1. Flip nkvo (KV cache in RAM instead of VRAM)
#     2. More-compressed ctk types from WS_FA_CTK, with both nkvo values
#   Timeout = break immediately (more compression won't fix throughput).
#   Largest stable context (with any config) becomes BEST_CTX.
# -----------------------------------------------------------------------------
phase6_ctx_sweep() {
    log "[Phase 6] Context size sweep"

    local full_ctx_list="128 512 1024 2048 4096 8192 16384 32768 65536 131072"
    local ctx_list
    ctx_list="$(apply_axis_opts "${full_ctx_list}" "${OPT_START_CTX}" "${OPT_DIR_CTX}")"

    # ctk quality order low→high (= compression order high→low)
    # Fallbacks use more-compressed types than BEST_CTK, so we iterate this
    # list and skip anything at or above BEST_CTK's index.
    local -a ctk_quality_order=("turbo2" "turbo3" "turbo4" "q4_0" "q8_0" "f16")
    local best_ctk_idx=5  # default to f16 (least compressed)
    local i
    for i in "${!ctk_quality_order[@]}"; do
        [[ "${ctk_quality_order[$i]}" == "${BEST_CTK}" ]] && { best_ctk_idx=$i; break; }
    done

    WS_CTX=""
    BEST_CTX=128

    while IFS= read -r ctx; do
        [[ -z "${ctx}" ]] && continue

        # --- Primary config ---
        local status
        status="$(run_bench "phase6/ctx=${ctx}" \
            ngl="${BEST_NGL}" fa="${BEST_FA}" ctk="${BEST_CTK}" ctv="${BEST_CTV}" \
            threads="${BEST_THREADS}" nkvo="${BEST_NKVO}" \
            b="${BEST_B}" ub="${BEST_UB}" \
            n_prompt="${ctx}" n_gen=0 reps=2 \
            phase=6 phase_label="ctx_sweep")"

        if [[ "${status}" == "ok" ]]; then
            WS_CTX="${WS_CTX:+${WS_CTX} }${ctx}"
            BEST_CTX="${ctx}"
            continue
        fi

        # Timeout = throughput ceiling, not memory — no fallback will help.
        if [[ "${status}" == "timeout" ]]; then
            log "[Phase 6] Stopping at ctx=${ctx} (timeout)"
            break
        fi

        # OOM or error — try fallbacks before giving up on this ctx size.
        log "[Phase 6] ctx=${ctx} failed (${status}) with best config — trying fallbacks"

        local fell_back=false

        # Build fallback sequence:
        #   Part 1: same ctk, flip nkvo
        #   Part 2: more-compressed ctk types from WS_FA_CTK × both nkvo values
        # Each entry: "fa ctk ctv nkvo"
        local -a fallbacks=()

        # Part 1 — nkvo flip (same quality, just moves KV cache to RAM)
        local alt_nkvo
        [[ "${BEST_NKVO}" == "0" ]] && alt_nkvo="1" || alt_nkvo="0"
        # Only add if alt_nkvo is in WS_NKVO
        echo "${WS_NKVO}" | tr ' ' '\n' | grep -qw "${alt_nkvo}" && \
            fallbacks+=("${BEST_FA} ${BEST_CTK} ${BEST_CTV} ${alt_nkvo}")

        # Part 2 — more-compressed ctk types (lower index = more compressed)
        for (( i=0; i<best_ctk_idx; i++ )); do
            local fb_ctk="${ctk_quality_order[$i]}"
            # Skip turbo types if binary not available
            [[ "${fb_ctk}" == turbo* && "${TURBO_AVAILABLE}" != "true" ]] && continue
            # Only try if Phase 2 found this ctk viable
            local fa_ctv
            fa_ctv="$(_best_fa_ctv_for_ctk "${fb_ctk}")" || continue
            local fb_fa fb_ctv
            read -r fb_fa fb_ctv <<< "${fa_ctv}"
            # Try nkvo=0 first, then nkvo=1
            local nkvo_fb
            for nkvo_fb in 0 1; do
                echo "${WS_NKVO}" | tr ' ' '\n' | grep -qw "${nkvo_fb}" || continue
                fallbacks+=("${fb_fa} ${fb_ctk} ${fb_ctv} ${nkvo_fb}")
            done
        done

        # Run through fallbacks
        local fb_entry fb_fa fb_ctk fb_ctv fb_nkvo
        for fb_entry in "${fallbacks[@]}"; do
            read -r fb_fa fb_ctk fb_ctv fb_nkvo <<< "${fb_entry}"
            local fb_status
            fb_status="$(run_bench "phase6/ctx=${ctx}/nkvo=${fb_nkvo}_ctk=${fb_ctk}" \
                ngl="${BEST_NGL}" fa="${fb_fa}" ctk="${fb_ctk}" ctv="${fb_ctv}" \
                threads="${BEST_THREADS}" nkvo="${fb_nkvo}" \
                b="${BEST_B}" ub="${BEST_UB}" \
                n_prompt="${ctx}" n_gen=0 reps=2 \
                phase=6 phase_label="ctx_sweep")"
            if [[ "${fb_status}" == "ok" ]]; then
                log "[Phase 6] ctx=${ctx} succeeded with fallback nkvo=${fb_nkvo} ctk=${fb_ctk}"
                WS_CTX="${WS_CTX:+${WS_CTX} }${ctx}"
                BEST_CTX="${ctx}"
                fell_back=true
                break
            fi
            [[ "${fb_status}" == "timeout" ]] && break  # timeout mid-fallback: stop trying
        done

        if [[ "${fell_back}" != "true" ]]; then
            log "[Phase 6] Stopping at ctx=${ctx} — primary and all fallbacks failed"
            break
        fi

    done <<< "${ctx_list}"

    if [[ -z "${WS_CTX}" ]]; then
        if [[ -n "${OPT_START_CTX}" ]]; then
            warn "[Phase 6] No context ≥ ${OPT_START_CTX} succeeded. WS_CTX will be empty — Phase 7 will produce 0 combinations for ctx."
            warn "[Phase 6] To include smaller contexts, re-run with --start-ctx lowered or omitted."
        else
            WS_CTX="512"
        fi
    fi
    log "[Phase 6] Best (max) context: ${BEST_CTX}  Working set: ${WS_CTX:-<empty>}"
}

# -----------------------------------------------------------------------------
# phase7_combination_matrix
#   Build a pruned cartesian product of top candidates from each prior phase
#   and run every surviving combination.
# -----------------------------------------------------------------------------
phase7_combination_matrix() {
    log "[Phase 7] Full combination matrix"

    # --- Auto-derive Phase 7 minimums when not explicitly set ---
    # Each auto-min is skipped when the user provides the corresponding
    # --min-* flag or SWEEP_MIN_* env var. Override any auto-min with
    # --min-ngl 0, --min-threads 1, etc. to disable it.

    # NGL: keep only the top 2 values (MAX_NGL and one step below).
    # Rationale: low-ngl configs rarely produce useful context sizes and
    # inflate the matrix combinatorially.
    local eff_min_ngl="${OPT_MIN_NGL}"
    if [[ -z "${eff_min_ngl}" ]]; then
        eff_min_ngl=$(( MAX_NGL - SWEEP_NGL_STEP ))
        [[ ${eff_min_ngl} -lt 0 ]] && eff_min_ngl=0
        log "[Phase 7] Auto min-ngl=${eff_min_ngl} (max_ngl=${MAX_NGL} − 1 step); override with --min-ngl"
    fi

    # Threads: skip counts below physical-core count.
    # Sub-physical thread counts are almost never optimal for large models.
    local eff_min_threads="${OPT_MIN_THREADS}"
    if [[ -z "${eff_min_threads}" && "${HW_CPU_PHYSICAL}" -gt 1 ]]; then
        eff_min_threads="${HW_CPU_PHYSICAL}"
        log "[Phase 7] Auto min-threads=${eff_min_threads} (physical cores); override with --min-threads"
    fi

    # Context: inherit --start-ctx as the Phase 7 minimum when --min-ctx is not set.
    # If the user said "start at 32k", they don't care about smaller contexts in Phase 7.
    # If neither is set, apply a sensible baseline so trivially small contexts are excluded.
    local eff_min_ctx="${OPT_MIN_CTX}"
    if [[ -z "${eff_min_ctx}" && -n "${OPT_START_CTX}" ]]; then
        eff_min_ctx="${OPT_START_CTX}"
        log "[Phase 7] Auto min-ctx=${eff_min_ctx} (inherited from --start-ctx); override with --min-ctx"
    elif [[ -z "${eff_min_ctx}" ]]; then
        eff_min_ctx="8192"
        log "[Phase 7] Auto min-ctx=${eff_min_ctx} (default minimum; override with --min-ctx N or SWEEP_MIN_CTX=N)"
    fi

    # Batch: keep only batch sizes ≥ BEST_B/2 (top half of what Phase 5 found).
    # This drops small batch sizes that consistently underperform.
    local eff_min_b="${OPT_MIN_B}"
    if [[ -z "${eff_min_b}" && -n "${BEST_B}" && "${BEST_B}" -gt 0 ]]; then
        eff_min_b=$(( BEST_B / 2 ))
        [[ ${eff_min_b} -lt 512 ]] && eff_min_b=512
        log "[Phase 7] Auto min-b=${eff_min_b} (best_b=${BEST_B} / 2); override with --min-b"
    fi

    # Apply minimums to working sets
    local ngl_p7 thread_p7 ctx_p7 nkvo_p7
    ngl_p7="$(apply_phase7_mins "ngl"     "$(echo "${WS_NGL}" | tr ' ' '\n')"     "${eff_min_ngl}")"
    thread_p7="$(apply_phase7_mins "threads" "$(echo "${WS_THREADS}" | tr ' ' '\n')" "${eff_min_threads}")"
    ctx_p7="$(apply_phase7_mins "ctx"     "$(echo "${WS_CTX}" | tr ' ' '\n')"     "${eff_min_ctx}")"
    nkvo_p7="$(echo "${WS_NKVO}" | tr ' ' '\n' | grep -v '^$')"

    # KV type: default to q8_0 (int8) minimum — exclude heavily-compressed types
    # unless the user explicitly asks for them or lowers the bar.
    local eff_min_ctk="${OPT_MIN_CTK}"
    if [[ -z "${eff_min_ctk}" ]]; then
        eff_min_ctk="q8_0"
        log "[Phase 7] Auto min-ctk=${eff_min_ctk} (default minimum quality; override with --min-ctk TYPE or SWEEP_MIN_CTK=TYPE)"
    fi

    local ctk_values
    ctk_values="$(echo "${WS_FA_CTK}" | grep -v '^$' | awk '{print $2}' | sort -u)"
    ctk_values="$(apply_phase7_mins "ctk" "${ctk_values}" "${eff_min_ctk}")"

    # Batch/ubatch: filter WS_B_UB pairs by eff_min_b (b value must be >= threshold)
    local b_ub_p7
    if [[ -n "${eff_min_b}" ]]; then
        b_ub_p7="$(echo "${WS_B_UB}" | grep -v '^$' | awk -v min="${eff_min_b}" '$1+0 >= min+0')"
        [[ -z "${b_ub_p7}" ]] && b_ub_p7="${WS_B_UB}"  # never drop all combos
    else
        b_ub_p7="${WS_B_UB}"
    fi

    # Bail early with a clear message if ctx filtering left nothing to test.
    if [[ -z "$(echo "${ctx_p7}" | grep -v '^$')" ]]; then
        warn "[Phase 7] No ctx values passed the minimum filter (min-ctx=${eff_min_ctx:-unset}). Phase 7 skipped."
        warn "[Phase 7] To run Phase 7, lower or remove --min-ctx / --start-ctx, or re-run Phase 6 with a lower --start-ctx."
        return 0
    fi

    # Count estimate
    local ngl_count thread_count ctx_count fa_ctk_count nkvo_count b_ub_count
    ngl_count="$(echo "${ngl_p7}" | grep -c '[0-9]' || echo 1)"
    thread_count="$(echo "${thread_p7}" | grep -c '[0-9a-z]' || echo 1)"
    ctx_count="$(echo "${ctx_p7}" | grep -c '[0-9]' || echo 1)"
    fa_ctk_count="$(echo "${WS_FA_CTK}" | grep -c '[0-9]' || echo 1)"
    nkvo_count="$(echo "${nkvo_p7}" | grep -c '[0-9]' || echo 1)"
    b_ub_count="$(echo "${b_ub_p7}" | grep -c '[0-9]' || echo 1)"
    local total=$(( ngl_count * fa_ctk_count * thread_count * nkvo_count * b_ub_count * ctx_count ))
    log "[Phase 7] Estimated combinations: ${total} (ngl×${ngl_count} fa_ctk×${fa_ctk_count} threads×${thread_count} nkvo×${nkvo_count} b_ub×${b_ub_count} ctx×${ctx_count})"

    # Context OOM tracking: max ok ctx per "ngl ctk nkvo" triple
    declare -A ctx_ceil=()

    local run_count=0
    local ngl fa ctk ctv threads b ub ctx nkvo

    while IFS= read -r ngl; do
        [[ -z "${ngl}" ]] && continue
        while IFS= read -r fa_ctk_line; do
            [[ -z "${fa_ctk_line}" ]] && continue
            read -r fa ctk ctv <<< "${fa_ctk_line}"
            # Filter by ctk minimums
            echo "${ctk_values}" | grep -qw "${ctk}" || continue

            while IFS= read -r nkvo; do
                [[ -z "${nkvo}" ]] && continue
                while IFS= read -r threads; do
                    [[ -z "${threads}" ]] && continue
                    while IFS= read -r b_ub_line; do
                        [[ -z "${b_ub_line}" ]] && continue
                        read -r b ub <<< "${b_ub_line}"
                        [[ ${ub} -gt ${b} ]] && continue

                        while IFS= read -r ctx; do
                            [[ -z "${ctx}" ]] && continue

                            # Context ceiling pruning
                            local ceil_key="${ngl}_${ctk}_${nkvo}"
                            if [[ -n "${ctx_ceil[${ceil_key}]+x}" ]]; then
                                local max_ok="${ctx_ceil[${ceil_key}]}"
                                if [[ ${ctx} -gt ${max_ok} ]]; then
                                    log "[Phase 7] Skip ctx=${ctx} for ngl=${ngl}/ctk=${ctk}/nkvo=${nkvo} (OOM ceiling: ${max_ok})"
                                    continue
                                fi
                            fi

                            local t_arg=""
                            [[ "${threads}" != "system_default" ]] && t_arg="threads=${threads}"

                            local label="p7/ngl=${ngl}_fa=${fa}_ctk=${ctk}_nkvo=${nkvo}_b=${b}_ub=${ub}_ctx=${ctx}"
                            local status
                            status="$(run_bench "${label}" \
                                ngl="${ngl}" fa="${fa}" ctk="${ctk}" ctv="${ctv}" \
                                nkvo="${nkvo}" ${t_arg:+${t_arg}} \
                                b="${b}" ub="${ub}" \
                                n_prompt="${ctx}" n_gen=128 reps="${SWEEP_REPETITIONS}" \
                                phase=7 phase_label="combination_matrix")"

                            (( run_count++ )) || true
                            [[ $(( run_count % 10 )) -eq 0 ]] && log "[Phase 7] ${run_count}/${total} combinations run"

                            if [[ "${status}" == "oom" || "${status}" == "timeout" ]]; then
                                # Record context ceiling for this triple
                                if [[ -z "${ctx_ceil[${ceil_key}]+x}" ]] || \
                                   [[ ${ctx} -lt ${ctx_ceil[${ceil_key}]} ]]; then
                                    ctx_ceil[${ceil_key}]=$(( ctx / 2 ))
                                    [[ ${ctx_ceil[${ceil_key}]} -lt 128 ]] && ctx_ceil[${ceil_key}]=0
                                fi
                            fi

                        done <<< "${ctx_p7}"
                    done <<< "${b_ub_p7}"
                done <<< "${thread_p7}"
            done <<< "${nkvo_p7}"
        done <<< "${WS_FA_CTK}"
    done <<< "${ngl_p7}"

    log "[Phase 7] Complete — ${run_count} combinations run"
}

# -----------------------------------------------------------------------------
# write_markdown
#   Generate ${OUTPUT_MODEL_DIR}/sweep.md from sweep.jsonl.
# -----------------------------------------------------------------------------
write_markdown() {
    local md_file="${OUTPUT_MODEL_DIR}/sweep.md"
    local jsonl="${OUTPUT_MODEL_DIR}/sweep.jsonl"
    [[ -f "${jsonl}" ]] || return 0

    {
        echo "# Sweep Results: ${MODEL_STEM}"
        echo
        echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo

        local phase phase_label
        for phase in 0 1 2 3 4 5 6 7; do
            phase_label="$(jq -r --argjson ph "${phase}" '[.[] | select(.phase==$ph)] | .[0].phase_label // empty' "${jsonl}" 2>/dev/null | head -1)"
            [[ -z "${phase_label}" ]] && continue

            echo "## Phase ${phase} — ${phase_label}"
            echo
            echo "| ngl | fa | ctk | threads | nkvo | b | ub | n_prompt | PP t/s | TG t/s | viable | status |"
            echo "|-----|-----|-----|---------|------|---|----|---------:|-------:|-------:|--------|--------|"

            jq -r --argjson ph "${phase}" '
                [.[] | select(.phase==$ph)] |
                sort_by(
                    if .status == "ok" then
                        (-.results[]? | select(.test=="tg") | .avg_ts) // 0
                    else 1000000 end
                ) |
                .[] |
                [
                    .params.ngl,
                    .params.fa,
                    .params.ctk,
                    (if .params.threads_is_default then "sys" else (.params.threads|tostring) end),
                    .params.nkvo,
                    .params.b,
                    .params.ub,
                    .params.n_prompt,
                    ((.results[]? | select(.test=="pp") | .avg_ts | tostring) // "-"),
                    ((.results[]? | select(.test=="tg") | .avg_ts | tostring) // "-"),
                    (.viable // "-" | tostring),
                    .status
                ] | "| " + join(" | ") + " |"
            ' "${jsonl}" 2>/dev/null
            echo
        done

        # Context frontier (from Phase 7)
        if jq -e '[.[] | select(.phase==7 and .status=="ok")] | length > 0' "${jsonl}" &>/dev/null; then
            echo "## Context Frontier"
            echo
            echo "| ngl | ctk | nkvo | Max Context | PP t/s |"
            echo "|-----|-----|------|------------:|-------:|"
            jq -r '
                [.[] | select(.phase==7 and .status=="ok" and .params.n_gen==0)] |
                group_by([.params.ngl, .params.ctk, .params.nkvo]) |
                .[] |
                sort_by(-.params.n_prompt) | .[0] |
                [
                    .params.ngl,
                    .params.ctk,
                    .params.nkvo,
                    .params.n_prompt,
                    ((.results[]? | select(.test=="pp") | .avg_ts | tostring) // "-")
                ] | "| " + join(" | ") + " |"
            ' "${jsonl}" 2>/dev/null | sort -t'|' -k2 -rn
            echo
        fi

    } > "${md_file}"

    log "Markdown written: ${md_file}"
}

# -----------------------------------------------------------------------------
# print_summary
#   Print a concise ASCII summary of sweep results to the terminal.
# -----------------------------------------------------------------------------
print_summary() {
    local jsonl="${OUTPUT_MODEL_DIR}/sweep.jsonl"
    [[ -f "${jsonl}" ]] || return 0

    echo
    printf '═%.0s' {1..60}; echo
    printf ' Sweep complete: %s\n' "${MODEL_STEM}"
    printf '═%.0s' {1..60}; echo
    printf ' Best config:  ngl=%-4s fa=%-2s ctk=%-8s nkvo=%s\n' "${BEST_NGL}" "${BEST_FA}" "${BEST_CTK}" "${BEST_NKVO}"
    printf ' Best context: %s tokens\n' "${BEST_CTX}"
    printf ' Best threads: %s\n' "${BEST_THREADS:-system default}"
    printf ' Output dir:   %s\n' "${OUTPUT_MODEL_DIR}"

    # Top 5 TG configs
    echo
    echo " Top 5 TG configs:"
    jq -rs '[.[] | select(.status=="ok")] |
        sort_by(-(.results[]? | select(.test=="tg") | .avg_ts // 0)) |
        .[:5] |
        .[] |
        "   ngl=\(.params.ngl) fa=\(.params.fa) ctk=\(.params.ctk) nkvo=\(.params.nkvo) → TG=\((.results[]? | select(.test=="tg") | .avg_ts) // "n/a") t/s"
    ' "${jsonl}" 2>/dev/null || true

    echo
    local ok oom timeout error
    ok="$(jq -s '[.[] | select(.status=="ok")] | length' "${jsonl}" 2>/dev/null || echo 0)"
    oom="$(jq -s '[.[] | select(.status=="oom")] | length' "${jsonl}" 2>/dev/null || echo 0)"
    timeout="$(jq -s '[.[] | select(.status=="timeout")] | length' "${jsonl}" 2>/dev/null || echo 0)"
    error="$(jq -s '[.[] | select(.status=="error")] | length' "${jsonl}" 2>/dev/null || echo 0)"
    printf ' Runs: %s ok  %s oom  %s timeout  %s error\n' "${ok}" "${oom}" "${timeout}" "${error}"
    printf '═%.0s' {1..60}; echo
    echo
}

# -----------------------------------------------------------------------------
# sweep_model PATH
#   Run all phases for a single model file.
# -----------------------------------------------------------------------------
sweep_model() {
    local path="${1}"
    MODEL_PATH="${path}"
    MODEL_STEM="$(basename "${path}" .gguf)"
    OUTPUT_MODEL_DIR="${SWEEP_OUTPUT_DIR}/${MODEL_STEM}"

    log "===== Starting sweep: ${MODEL_STEM} ====="
    setup_output_dir
    load_state

    local phase
    for phase in 0 1 2 3 4 5 6 7; do
        # --only-phases filter
        if [[ -n "${OPT_ONLY_PHASES}" ]]; then
            if ! echo ",${OPT_ONLY_PHASES}," | grep -q ",${phase},"; then
                log "[Phase ${phase}] Skipped (not in --only-phases)"
                continue
            fi
        fi
        # --skip-phases filter
        if [[ -n "${OPT_SKIP_PHASES}" ]]; then
            if echo ",${OPT_SKIP_PHASES}," | grep -q ",${phase},"; then
                log "[Phase ${phase}] Skipped (in --skip-phases)"
                continue
            fi
        fi
        # Resume: skip completed phases
        if echo " ${PHASES_COMPLETE} " | grep -q " ${phase} "; then
            log "[Phase ${phase}] Already complete — skipping (--resume)"
            continue
        fi

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
        read -r -p "Continue? [y/N] " reply
        [[ "${reply}" =~ ^[Yy]$ ]] || die "Aborted by user"
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
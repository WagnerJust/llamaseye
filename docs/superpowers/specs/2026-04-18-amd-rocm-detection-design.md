# AMD ROCm GPU Detection — Design Spec

**Date:** 2026-04-18  
**Status:** Approved

## Problem

On Linux with an AMD GPU, llamaseye's hardware detection falls back to `Backend = "cpu"` because it only probes `nvidia-smi`. The actual llama-bench runs still use the GPU (if the binary was compiled with ROCm/HIP), but llamaseye reports no VRAM and provides no GPU temperature monitoring.

## Goal

Detect AMD GPUs on Linux via `rocm-smi`, populate VRAM info, enable thermal monitoring, and expose `backend = "rocm"` in `hardware.json`.

## Approach

Mirror the existing NVIDIA detection pattern exactly. All detection helpers live in `shared.go`; platform entry points in `detect_linux.go`.

## Changes

### `hardware/detect.go`

Add one constant:

```go
BackendROCm Backend = "rocm"
```

### `hardware/shared.go`

Add `detectROCmSMI(h *HardwareInfo) bool`:

1. Check `rocm-smi` is in PATH via `exec.LookPath`
2. Run `rocm-smi` (no args) as availability check — return false on error
3. Set `h.Backend = BackendROCm`
4. Query GPU count: `rocm-smi --showgpucount`, parse integer from output
5. Query model name: `rocm-smi --showproductname`, parse `Card series:` line
6. Query VRAM: `rocm-smi --showmeminfo vram`, parse `Total` and `Free` lines (values in bytes, convert to GiB)

### `hardware/detect_linux.go`

Update the GPU detection block:

```go
if !detectNvidiaSMI(h) && !detectROCmSMI(h) {
    h.Backend = BackendCPU
    h.GPUCount = 0
    h.GPUModel = "none"
}
if h.Backend == BackendROCm {
    h.GPUTempCmd = "rocm-smi --showtemp 2>/dev/null | awk '/Junction/{gsub(/[^0-9.]/,\"\",$NF); printf \"%d\", $NF}'"
}
```

### `hardware/detect_test.go`

Extend the smoke test to accept `"rocm"` as a valid Linux backend alongside `"cuda"` and `"cpu"`.

## Data Flow

```
Detect() [linux]
  → detectNvidiaSMI()   # NVIDIA path (unchanged)
  → detectROCmSMI()     # new AMD path
    → rocm-smi --showgpucount    → h.GPUCount
    → rocm-smi --showproductname → h.GPUModel
    → rocm-smi --showmeminfo vram → h.GPUVRAMGiB, h.GPUVRAMFreeGiB
    → sets h.GPUTempCmd for thermal monitoring
  → BackendCPU fallback
```

## Error Handling

- If `rocm-smi` is missing from PATH: return false, fall through to CPU
- If `rocm-smi` exits non-zero: return false, fall through to CPU
- If individual field queries fail: leave field at zero-value (same as NVIDIA path)

## Testing

The existing smoke test (`TestDetect_Smoke`) runs on the current machine. Extend it to allow `"rocm"` as a valid Linux backend. No mock-based unit tests are added — the NVIDIA path has none either, and ROCm hardware won't be present in CI.

## Out of Scope

- `rocminfo` fallback (not needed; `rocm-smi` is standard on all ROCm installs)
- Multi-GPU AMD support (same single-GPU-0 constraint as the NVIDIA path)
- macOS AMD detection (Metal covers all macOS GPU backends)

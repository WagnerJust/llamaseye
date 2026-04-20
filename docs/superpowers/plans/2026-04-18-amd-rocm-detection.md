# AMD ROCm GPU Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect AMD GPUs on Linux via `rocm-smi`, populate VRAM/model/temp fields, and expose `backend = "rocm"` in `hardware.json`.

**Architecture:** Mirror the existing NVIDIA detection pattern — add `BackendROCm` constant, add `detectROCmSMI` helper in `shared.go`, wire it into `detect_linux.go` between the NVIDIA probe and the CPU fallback. No new files needed.

**Tech Stack:** Go stdlib (`os/exec`, `strings`, `strconv`), `rocm-smi` CLI tool (standard on all ROCm installs).

---

## File Map

| File | Change |
|------|--------|
| `hardware/detect.go` | Add `BackendROCm` constant |
| `hardware/shared.go` | Add `detectROCmSMI(h *HardwareInfo) bool` |
| `hardware/detect_linux.go` | Wire ROCm probe + set GPU temp command |
| `hardware/detect_test.go` | Accept `"rocm"` as valid Linux backend |
| `CHANGELOG.md` | Add v1.8.0 entry |

---

### Task 1: Add `BackendROCm` constant and extend smoke test

**Files:**
- Modify: `hardware/detect.go:7-11`
- Modify: `hardware/detect_test.go:25-28`

- [ ] **Step 1: Add the constant**

In `hardware/detect.go`, update the `const` block:

```go
const (
	BackendCUDA   Backend = "cuda"
	BackendMetal  Backend = "metal"
	BackendCPU    Backend = "cpu"
	BackendROCm   Backend = "rocm"
)
```

- [ ] **Step 2: Extend the smoke test to accept `"rocm"` on Linux**

In `hardware/detect_test.go`, replace the backend check block (currently lines 25-28):

```go
	// Backend must be a known value on every platform.
	validBackends := map[Backend]bool{
		BackendMetal: true,
		BackendCUDA:  true,
		BackendROCm:  true,
		BackendCPU:   true,
	}
	if !validBackends[hw.Backend] {
		t.Errorf("Backend = %q, want one of metal/cuda/rocm/cpu", hw.Backend)
	}
```

- [ ] **Step 3: Run tests — expect pass (existing backends still accepted)**

```bash
go test ./hardware/...
```

Expected: `PASS` — no new function exists yet but the constant and test compile cleanly.

- [ ] **Step 4: Commit**

```bash
git add hardware/detect.go hardware/detect_test.go
git commit -m "feat: add BackendROCm constant and extend smoke test"
```

---

### Task 2: Implement `detectROCmSMI` in `shared.go`

**Files:**
- Modify: `hardware/shared.go` (append after `isCommandAvailable`)

`rocm-smi` output reference (used for parsing below):

```
# rocm-smi --showgpucount
GPU count: 1

# rocm-smi --showproductname
GPU[0]		: Card series:		Radeon RX 7900 XTX

# rocm-smi --showmeminfo vram
GPU[0]		: VRAM Total Memory (B): 25753026560
GPU[0]		: VRAM Total Used Memory (B): 1234567890
```

- [ ] **Step 1: Add `detectROCmSMI` to `shared.go`**

Append this function after `isCommandAvailable`:

```go
// detectROCmSMI attempts to query rocm-smi and fill AMD GPU fields.
// Returns true if an AMD GPU was found.
func detectROCmSMI(h *HardwareInfo) bool {
	if _, err := exec.LookPath("rocm-smi"); err != nil {
		return false
	}
	if err := exec.Command("rocm-smi").Run(); err != nil {
		return false
	}

	h.Backend = BackendROCm

	if out, err := exec.Command("rocm-smi", "--showgpucount").Output(); err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(strings.TrimSpace(line), "GPU count:") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					n, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
					h.GPUCount = n
				}
				break
			}
		}
	}

	if out, err := exec.Command("rocm-smi", "--showproductname").Output(); err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.Contains(line, "Card series:") {
				parts := strings.SplitN(line, "Card series:", 2)
				if len(parts) == 2 {
					h.GPUModel = strings.TrimSpace(parts[1])
				}
				break
			}
		}
	}

	if out, err := exec.Command("rocm-smi", "--showmeminfo", "vram").Output(); err == nil {
		var totalBytes, usedBytes int64
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if strings.Contains(line, "VRAM Total Memory (B):") {
				parts := strings.SplitN(line, "VRAM Total Memory (B):", 2)
				if len(parts) == 2 {
					totalBytes, _ = strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
				}
			}
			if strings.Contains(line, "VRAM Total Used Memory (B):") {
				parts := strings.SplitN(line, "VRAM Total Used Memory (B):", 2)
				if len(parts) == 2 {
					usedBytes, _ = strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
				}
			}
		}
		h.GPUVRAMGiB = int(totalBytes / (1 << 30))
		if totalBytes > 0 {
			h.GPUVRAMFreeGiB = int((totalBytes - usedBytes) / (1 << 30))
		}
	}

	return true
}
```

- [ ] **Step 2: Verify it compiles**

```bash
go build ./hardware/...
```

Expected: no errors.

- [ ] **Step 3: Run tests**

```bash
go test ./hardware/...
```

Expected: `PASS` — `rocm-smi` is absent on the dev machine so `detectROCmSMI` returns false immediately; all existing paths still work.

- [ ] **Step 4: Commit**

```bash
git add hardware/shared.go
git commit -m "feat: add detectROCmSMI for AMD GPU detection on Linux"
```

---

### Task 3: Wire ROCm into `detect_linux.go`

**Files:**
- Modify: `hardware/detect_linux.go:26-38`

- [ ] **Step 1: Update the GPU detection block**

Replace the current GPU block in `Detect()` (lines 26-38):

```go
	// GPU — NVIDIA first, then AMD, then CPU fallback.
	if !detectNvidiaSMI(h) && !detectROCmSMI(h) {
		h.Backend = BackendCPU
		h.GPUCount = 0
		h.GPUModel = "none"
	}

	// Thermal
	h.CPUTempCmd = linuxCPUTempCmd()
	if h.Backend == BackendCUDA {
		h.GPUTempCmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
	}
	if h.Backend == BackendROCm {
		h.GPUTempCmd = "rocm-smi --showtemp 2>/dev/null | awk '/[Jj]unction.*\\(C\\):/{gsub(/[^0-9.]/,\"\",$NF); printf \"%d\", $NF}'"
	}
```

- [ ] **Step 2: Build and test**

```bash
go build ./... && go test ./...
```

Expected: `PASS` on all packages.

- [ ] **Step 3: Commit**

```bash
git add hardware/detect_linux.go
git commit -m "feat: wire AMD ROCm detection into Linux hardware probe"
```

---

### Task 4: CHANGELOG and docs

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/spec.md` (backend enum)
- Modify: `.agents/skills/llamaseye/SKILL.md` (hardware detection section)

- [ ] **Step 1: Add CHANGELOG entry**

Add a new section at the top of `CHANGELOG.md` (after the header, before `[1.7.6]`):

```markdown
## [1.8.0] — 2026-04-18

### Added
- AMD GPU detection on Linux via `rocm-smi`: populates GPU model, VRAM total/free, and GPU temperature monitoring. `hardware.json` now records `backend = "rocm"` for AMD systems. Requires ROCm userspace tools (`rocm-smi`) to be installed; falls back to `cpu` if absent.
```

- [ ] **Step 2: Update `docs/spec.md` backend enum**

Find the `backend` field description in `docs/spec.md` and add `"rocm"` to the allowed values. The line will look something like:

> `backend` — string, one of `"cuda"`, `"metal"`, `"cpu"`

Change it to:

> `backend` — string, one of `"cuda"`, `"metal"`, `"rocm"`, `"cpu"`

- [ ] **Step 3: Update skill doc**

In `.agents/skills/llamaseye/SKILL.md`, find any reference to GPU backends or hardware detection and add `rocm` to the list of supported backends.

- [ ] **Step 4: Run full test suite one final time**

```bash
go test ./...
```

Expected: `PASS`.

- [ ] **Step 5: Commit docs**

```bash
git add CHANGELOG.md docs/spec.md .agents/skills/llamaseye/SKILL.md
git commit -m "docs: update spec, skill, and changelog for AMD ROCm detection (v1.8.0)"
```

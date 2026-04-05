//go:build darwin

package hardware

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Detect populates HardwareInfo for macOS (both Apple Silicon and Intel).
func Detect() (*HardwareInfo, error) {
	h := &HardwareInfo{}
	h.CPUModel = sysctlString("machdep.cpu.brand_string")
	if h.CPUModel == "" {
		h.CPUModel = "Apple Silicon" // arm64 doesn't expose brand_string
	}
	h.CPUPhysical = sysctlInt("hw.physicalcpu", 1)
	h.CPULogical = sysctlInt("hw.logicalcpu", 1)

	// RAM
	memBytes := sysctlInt64("hw.memsize", 0)
	h.RAMGiB = int(memBytes / (1 << 30))
	h.RAMFreeGiB = freeRAMGiB()

	// GPU — CUDA check first, then Metal
	if detectNvidiaSMI(h) {
		// CUDA path (rare on macOS but possible with eGPU)
	} else {
		h.Backend = BackendMetal
		h.GPUCount = 1
		h.GPUModel = metalGPUModel()
		if runtime.GOARCH == "arm64" {
			// Apple Silicon: unified memory
			h.GPUVRAMGiB = h.RAMGiB
			h.GPUVRAMFreeGiB = h.RAMFreeGiB
		} else {
			h.GPUVRAMGiB = 0
			h.GPUVRAMFreeGiB = 0
		}
	}

	// Thermal
	if isCommandAvailable("osx-cpu-temp") {
		h.CPUTempCmd = "osx-cpu-temp"
	}
	if h.Backend == BackendCUDA {
		h.GPUTempCmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i 0"
	}

	return h, nil
}

func metalGPUModel() string {
	out, err := exec.Command("system_profiler", "SPDisplaysDataType").Output()
	if err != nil {
		return "Apple GPU"
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Chipset Model:") {
			return strings.TrimSpace(strings.TrimPrefix(line, "Chipset Model:"))
		}
	}
	return "Apple GPU"
}

func sysctlString(key string) string {
	out, err := exec.Command("sysctl", "-n", key).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func sysctlInt(key string, def int) int {
	v := sysctlString(key)
	if n, err := strconv.Atoi(v); err == nil {
		return n
	}
	return def
}

func sysctlInt64(key string, def int64) int64 {
	v := sysctlString(key)
	if n, err := strconv.ParseInt(v, 10, 64); err == nil {
		return n
	}
	return def
}

func freeRAMGiB() int {
	pageSize := sysctlInt("hw.pagesize", 4096)
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(line, "Pages free:") {
			parts := strings.Fields(line)
			if len(parts) >= 3 {
				pages, err := strconv.ParseInt(strings.TrimRight(parts[2], "."), 10, 64)
				if err == nil {
					return int(pages * int64(pageSize) / (1 << 30))
				}
			}
		}
	}
	return 0
}

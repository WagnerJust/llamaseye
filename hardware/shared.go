package hardware

import (
	"os/exec"
	"strconv"
	"strings"
)

// detectNvidiaSMI attempts to query nvidia-smi and fill CUDA fields.
// Returns true if CUDA GPU was found.
func detectNvidiaSMI(h *HardwareInfo) bool {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return false
	}
	// Quick availability check
	if err := exec.Command("nvidia-smi").Run(); err != nil {
		return false
	}

	h.Backend = BackendCUDA

	if count, err := nvidiaSMIQuery("count"); err == nil {
		n, _ := strconv.Atoi(strings.TrimSpace(count))
		h.GPUCount = n
	}
	if model, err := nvidiaSMIQuery("name"); err == nil {
		h.GPUModel = strings.TrimSpace(model)
	}
	if vram, err := nvidiaSMIQuery("memory.total"); err == nil {
		mib, _ := strconv.Atoi(strings.TrimSpace(vram))
		h.GPUVRAMGiB = mib / 1024
	}
	if free, err := nvidiaSMIQuery("memory.free"); err == nil {
		mib, _ := strconv.Atoi(strings.TrimSpace(free))
		h.GPUVRAMFreeGiB = mib / 1024
	}
	return true
}

func nvidiaSMIQuery(field string) (string, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu="+field,
		"--format=csv,noheader,nounits",
		"-i", "0").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

// isCommandAvailable returns true if the command is in PATH.
func isCommandAvailable(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

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
	h.GPUCount = 1 // safe default; overridden below if --showgpucount parses successfully

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
			if strings.HasPrefix(strings.TrimSpace(line), "GPU[") && strings.Contains(line, "Card series:") {
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

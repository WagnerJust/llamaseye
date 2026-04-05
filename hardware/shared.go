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

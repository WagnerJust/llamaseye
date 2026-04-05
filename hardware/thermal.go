package hardware

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ThermalMonitor polls CPU and GPU temperatures.
type ThermalMonitor struct {
	HW          *HardwareInfo
	CPULimit    int
	GPULimit    int
	PollSeconds int
	Disabled    bool
	Log         func(format string, args ...any)
}

// WaitCool blocks until both CPU and GPU temperatures are below their limits.
// No-op when Disabled is true or temp commands are unavailable.
func (tm *ThermalMonitor) WaitCool(ctx context.Context) {
	if tm.Disabled {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		cpuTemp := tm.readTemp(tm.HW.CPUTempCmd)
		gpuTemp := tm.readTemp(tm.HW.GPUTempCmd)

		if cpuTemp < tm.CPULimit && gpuTemp < tm.GPULimit {
			return
		}

		if tm.Log != nil {
			tm.Log("Thermal wait: CPU=%d°C GPU=%d°C — sleeping %ds",
				cpuTemp, gpuTemp, tm.PollSeconds)
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Duration(tm.PollSeconds) * time.Second):
		}
	}
}

// readTemp runs a temp command and returns the integer °C value.
// Returns 0 if the command is empty or fails.
func (tm *ThermalMonitor) readTemp(cmdStr string) int {
	if cmdStr == "" {
		return 0
	}
	// For simple single-word commands like "osx-cpu-temp"
	parts := strings.Fields(cmdStr)
	if len(parts) == 0 {
		return 0
	}
	out, err := exec.Command(parts[0], parts[1:]...).Output()
	if err != nil {
		return 0
	}
	// Extract first integer from output
	s := strings.TrimSpace(string(out))
	// Some outputs have decimals; take integer part
	if idx := strings.IndexByte(s, '.'); idx >= 0 {
		s = s[:idx]
	}
	// Take first whitespace-delimited token
	s = strings.Fields(s)[0]
	n, _ := strconv.Atoi(s)
	return n
}

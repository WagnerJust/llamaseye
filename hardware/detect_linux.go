//go:build linux

package hardware

import (
	"bufio"
	"bytes"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// Detect populates HardwareInfo for Linux.
func Detect() (*HardwareInfo, error) {
	h := &HardwareInfo{}

	// CPU
	h.CPUModel = linuxCPUModel()
	h.CPUPhysical, h.CPULogical = linuxCPUCounts()

	// RAM
	h.RAMGiB, h.RAMFreeGiB = linuxRAM()

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

	return h, nil
}

func linuxCPUModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "unknown"
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "model name") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				return strings.TrimSpace(parts[1])
			}
		}
	}
	return "unknown"
}

func linuxCPUCounts() (physical, logical int) {
	logical = 1
	physical = 1

	// logical count from /proc/cpuinfo
	data, err := os.ReadFile("/proc/cpuinfo")
	if err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(data))
		procs := 0
		for scanner.Scan() {
			if strings.HasPrefix(scanner.Text(), "processor") {
				procs++
			}
		}
		if procs > 0 {
			logical = procs
		}
	}

	// physical cores via lscpu
	out, err := exec.Command("lscpu").Output()
	if err == nil {
		coresPerSocket := 0
		sockets := 0
		scanner := bufio.NewScanner(bytes.NewReader(out))
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "Core(s) per socket:") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					n, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
					coresPerSocket = n
				}
			}
			if strings.HasPrefix(line, "Socket(s):") {
				parts := strings.SplitN(line, ":", 2)
				if len(parts) == 2 {
					n, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
					sockets = n
				}
			}
		}
		if coresPerSocket > 0 && sockets > 0 {
			physical = coresPerSocket * sockets
		}
	}
	return
}

func linuxRAM() (total, free int) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.Atoi(fields[1])
				total = kb / (1 << 20)
			}
		}
		if strings.HasPrefix(line, "MemAvailable:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.Atoi(fields[1])
				free = kb / (1 << 20)
			}
		}
	}
	return
}

func linuxCPUTempCmd() string {
	if !isCommandAvailable("sensors") {
		if _, err := os.Stat("/sys/class/thermal/thermal_zone0/temp"); err == nil {
			// sysfs returns millidegrees — awk converts: 45000 → 45
			return "awk '{printf \"%d\", $1/1000}' /sys/class/thermal/thermal_zone0/temp"
		}
		return ""
	}
	out, err := exec.Command("sensors").Output()
	if err != nil {
		return ""
	}
	lower := strings.ToLower(string(out))
	if strings.Contains(lower, "tctl") {
		return "sensors 2>/dev/null | awk '/Tctl/{gsub(/[^0-9.]/,\"\",$2); printf \"%d\", $2}'"
	}
	return "sensors 2>/dev/null | awk '/Package id 0/{gsub(/[^0-9.]/,\"\",$4); printf \"%d\", $4}'"
}

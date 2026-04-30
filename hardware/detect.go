// Package hardware detects CPU, RAM, GPU info and provides thermal monitoring.
package hardware

// Backend represents the GPU compute backend.
type Backend string

const (
	BackendCUDA   Backend = "cuda"
	BackendMetal  Backend = "metal"
	BackendCPU    Backend = "cpu"
	BackendROCm   Backend = "rocm"
)

// HardwareInfo holds the detected hardware inventory.
type HardwareInfo struct {
	CPUModel       string
	CPUPhysical    int
	CPULogical     int
	RAMGiB         int
	RAMFreeGiB     int
	GPUCount       int
	GPUModel       string
	GPUVRAMGiB     int
	GPUVRAMFreeGiB int
	Backend        Backend

	// Temperature commands (empty = not available)
	CPUTempCmd string
	GPUTempCmd string
}

// HardwareJSON is the schema written to hardware.json.
type HardwareJSON struct {
	CPUModel           string  `json:"cpu_model"`
	CPUPhysicalCores   int     `json:"cpu_physical_cores"`
	CPULogicalThreads  int     `json:"cpu_logical_threads"`
	RAMGiB             int     `json:"ram_gib"`
	RAMFreeGiBAtStart  int     `json:"ram_free_gib_at_start"`
	GPUCount           int     `json:"gpu_count"`
	GPUModel           string  `json:"gpu_model"`
	GPUVRAMGiB         int     `json:"gpu_vram_gib"`
	GPUVRAMFreeGiBAtStart int  `json:"gpu_vram_free_gib_at_start"`
	Backend            string  `json:"backend"`
}

// ToJSON converts HardwareInfo to the hardware.json schema.
func (h *HardwareInfo) ToJSON() HardwareJSON {
	return HardwareJSON{
		CPUModel:              h.CPUModel,
		CPUPhysicalCores:      h.CPUPhysical,
		CPULogicalThreads:     h.CPULogical,
		RAMGiB:                h.RAMGiB,
		RAMFreeGiBAtStart:     h.RAMFreeGiB,
		GPUCount:              h.GPUCount,
		GPUModel:              h.GPUModel,
		GPUVRAMGiB:            h.GPUVRAMGiB,
		GPUVRAMFreeGiBAtStart: h.GPUVRAMFreeGiB,
		Backend:               string(h.Backend),
	}
}

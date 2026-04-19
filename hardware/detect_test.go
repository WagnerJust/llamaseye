package hardware

import (
	"testing"
)

func TestDetect_Smoke(t *testing.T) {
	hw, err := Detect()
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if hw.CPULogical <= 0 {
		t.Errorf("CPULogical = %d, want > 0", hw.CPULogical)
	}
	if hw.CPUPhysical <= 0 {
		t.Errorf("CPUPhysical = %d, want > 0", hw.CPUPhysical)
	}
	if hw.RAMGiB <= 0 {
		t.Errorf("RAMGiB = %d, want > 0", hw.RAMGiB)
	}
	if hw.CPUModel == "" {
		t.Error("CPUModel is empty")
	}
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
}

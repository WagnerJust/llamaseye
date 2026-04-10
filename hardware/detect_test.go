package hardware

import (
	"runtime"
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
	// Backend should be set on macOS (metal) or linux (cuda/cpu)
	if runtime.GOOS == "darwin" && hw.Backend != "metal" {
		t.Errorf("Backend = %q on darwin, want metal", hw.Backend)
	}
}

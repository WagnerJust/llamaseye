package bench

import (
	"fmt"
	"strings"
)

// BinarySelector picks the right llama-bench binary for a given KV cache type.
type BinarySelector struct {
	StandardBin    string
	TurboBin       string
	TurboAvailable bool
	RotorBin       string
	RotorAvailable bool
}

// Select returns the binary path for the given ctk type.
// Returns an error if a specialised type is requested but its binary is unavailable.
func (s *BinarySelector) Select(ctk string) (path string, label string, err error) {
	if strings.HasPrefix(ctk, "turbo") {
		if !s.TurboAvailable {
			return "", "", fmt.Errorf("turbo KV type %q requested but turbo-bench not available", ctk)
		}
		return s.TurboBin, "turboquant", nil
	}
	if strings.HasPrefix(ctk, "planar") || strings.HasPrefix(ctk, "iso") {
		if !s.RotorAvailable {
			return "", "", fmt.Errorf("rotor KV type %q requested but rotor-bench not available", ctk)
		}
		return s.RotorBin, "rotorquant", nil
	}
	return s.StandardBin, "standard", nil
}

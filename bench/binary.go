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

// Select returns the binary path for the given ctk and ctv types.
// If either type requires a specialised binary, that binary is used.
// Returns an error if a specialised type is requested but its binary is unavailable.
func (s *BinarySelector) Select(ctk, ctv string) (path string, label string, err error) {
	isTurbo := func(t string) bool { return strings.HasPrefix(t, "turbo") }
	isRotor := func(t string) bool { return strings.HasPrefix(t, "planar") || strings.HasPrefix(t, "iso") }

	if isTurbo(ctk) || isTurbo(ctv) {
		if !s.TurboAvailable {
			return "", "", fmt.Errorf("turbo KV type requested (ctk=%s ctv=%s) but turbo-bench not available", ctk, ctv)
		}
		return s.TurboBin, "turboquant", nil
	}
	if isRotor(ctk) || isRotor(ctv) {
		if !s.RotorAvailable {
			return "", "", fmt.Errorf("rotor KV type requested (ctk=%s ctv=%s) but rotor-bench not available", ctk, ctv)
		}
		return s.RotorBin, "rotorquant", nil
	}
	return s.StandardBin, "standard", nil
}

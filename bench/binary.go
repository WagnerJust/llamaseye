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
}

// Select returns the binary path for the given ctk type.
// Returns an error if a turbo type is requested but TurboQuant is unavailable.
func (s *BinarySelector) Select(ctk string) (path string, label string, err error) {
	if strings.HasPrefix(ctk, "turbo") {
		if !s.TurboAvailable {
			return "", "", fmt.Errorf("turbo KV type %q requested but turbo-bench not available", ctk)
		}
		return s.TurboBin, "turboquant", nil
	}
	return s.StandardBin, "standard", nil
}

package output

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/WagnerJust/llamaseye/hardware"
)

// WriteHardwareJSON writes hardware.json to outputDir.
func WriteHardwareJSON(outputDir string, hw *hardware.HardwareInfo) error {
	data, err := json.MarshalIndent(hw.ToJSON(), "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outputDir, "hardware.json"), data, 0644)
}

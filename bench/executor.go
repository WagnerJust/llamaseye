package bench

import (
	"context"
	"io"
	"os/exec"
)

// OSExecutor implements CommandExecutor using os/exec.
type OSExecutor struct{}

func (OSExecutor) Run(ctx context.Context, binary string, args []string, stdout, stderr io.Writer) (int, error) {
	cmd := exec.CommandContext(ctx, binary, args...)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode(), nil
		}
		// Context deadline exceeded or other non-exit error
		return 1, err
	}
	return 0, nil
}

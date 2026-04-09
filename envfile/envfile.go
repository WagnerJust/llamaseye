// Package envfile loads KEY=VALUE pairs from a .env file into the process environment.
// Existing env vars are NOT overwritten — process environment always wins over the file.
package envfile

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Load reads path and sets any KEY=VALUE pairs as environment variables,
// skipping keys that are already set in the process environment.
// Blank lines and lines beginning with # are ignored.
// Returns an error only if the file exists but cannot be read or parsed.
func Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Strip inline comments: KEY=VALUE # comment
		if idx := strings.Index(line, " #"); idx != -1 {
			line = strings.TrimSpace(line[:idx])
		}
		idx := strings.IndexByte(line, '=')
		if idx < 1 {
			return fmt.Errorf("%s:%d: expected KEY=VALUE, got %q", path, lineNum, line)
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		// Strip optional surrounding quotes.
		// Single-quoted values are kept literal (no variable expansion).
		// Unquoted and double-quoted values have $VAR / ${VAR} references expanded.
		singleQuoted := len(val) >= 2 && val[0] == '\'' && val[len(val)-1] == '\''
		if len(val) >= 2 && ((val[0] == '"' && val[len(val)-1] == '"') ||
			(val[0] == '\'' && val[len(val)-1] == '\'')) {
			val = val[1 : len(val)-1]
		}
		if !singleQuoted {
			val = os.ExpandEnv(val)
		}
		// Process env wins: only set if not already present
		if _, exists := os.LookupEnv(key); !exists {
			if err := os.Setenv(key, val); err != nil {
				return fmt.Errorf("%s:%d: setenv %s: %w", path, lineNum, key, err)
			}
		}
	}
	return scanner.Err()
}

// LoadIfExists loads path if it exists, silently returns nil if it does not.
func LoadIfExists(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil
	}
	return Load(path)
}

// Package skills holds the canonical user-facing operational skill for
// llamaseye, embedded into the binary at build time. The companion
// internal package internal/skill consumes Body to install copies into
// agent-tool skill directories (see "llamaseye install-skill").
package skills

import _ "embed"

//go:embed llamaseye.md
var body []byte

// Body returns the canonical skill markdown (with YAML frontmatter).
// Callers should not mutate the returned slice.
func Body() []byte { return body }

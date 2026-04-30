// Package skill installs the embedded llamaseye operational skill into
// the directories that coding-agent tools read.
//
// The canonical skill body (markdown with YAML frontmatter) lives in
// the top-level skills/ package. This package exposes a small target
// registry and an install function consumed by `llamaseye install-skill`.
//
// Both supported targets share an identical body — only the install
// path differs:
//
//	claude   $HOME/.claude/skills/llamaseye/SKILL.md   (or .claude/skills/... in the cwd, with --local)
//	agents   $HOME/.agents/skills/llamaseye/SKILL.md   (or .agents/skills/... in the cwd, with --local)
package skill

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/WagnerJust/llamaseye/skills"
)

// Target identifies one place llamaseye can install its skill.
type Target struct {
	// ID is the stable CLI identifier ("claude", "agents").
	ID string
	// Name is human-readable.
	Name string
	// path is the install location relative to the chosen scope root
	// (the user's home dir for global, or the cwd for local).
	path string
}

// Path returns the install location for this target relative to the
// given scope root.
func (t Target) Path() string { return t.path }

var registry = []Target{
	{ID: "claude", Name: "Claude Code", path: ".claude/skills/llamaseye/SKILL.md"},
	{ID: "agents", Name: "cross-agent (.agents)", path: ".agents/skills/llamaseye/SKILL.md"},
}

// Targets returns all registered targets in display order.
func Targets() []Target {
	out := make([]Target, len(registry))
	copy(out, registry)
	return out
}

// IDs returns every target ID in display order.
func IDs() []string {
	out := make([]string, len(registry))
	for i, t := range registry {
		out[i] = t.ID
	}
	return out
}

// TargetByID looks up a target by its CLI ID.
func TargetByID(id string) (Target, bool) {
	for _, t := range registry {
		if t.ID == id {
			return t, true
		}
	}
	return Target{}, false
}

// Options configure an Install call.
type Options struct {
	// TargetIDs selects which targets to install. Empty means all.
	TargetIDs []string
	// Global writes under $HOME (e.g. ~/.claude/skills/...). When
	// false, paths are resolved against Cwd.
	Global bool
	// Cwd is the local-scope root. Ignored when Global is true.
	// If empty, the current working directory is used.
	Cwd string
	// DryRun reports what would be written without touching the disk.
	DryRun bool
	// Force overwrites an existing file at the install path. Without
	// it, Install returns ErrExists for that target and does not write.
	Force bool
}

// Result describes one resolved install action.
type Result struct {
	Target Target
	// AbsPath is the absolute file path that was (or would be) written.
	AbsPath string
	// Existed is true when a file was already present at AbsPath.
	Existed bool
	// Wrote is true when the file was actually written. Always false
	// in DryRun mode, and false when Existed && !Force.
	Wrote bool
	// Bytes is len(skills.Body()).
	Bytes int
}

// ErrExists is returned (wrapped) for each target whose install file
// already exists when Force is false.
var ErrExists = errors.New("install path already exists; pass --force to overwrite")

// Install resolves the requested targets and writes the embedded skill
// body to each resolved path. With Options.DryRun, no files are written.
func Install(opts Options) ([]Result, error) {
	targets, err := resolveTargets(opts.TargetIDs)
	if err != nil {
		return nil, err
	}

	root, err := scopeRoot(opts)
	if err != nil {
		return nil, err
	}

	body := skills.Body()
	results := make([]Result, 0, len(targets))
	var errs []error

	for _, t := range targets {
		abs := filepath.Join(root, t.path)
		res := Result{Target: t, AbsPath: abs, Bytes: len(body)}

		if _, statErr := os.Stat(abs); statErr == nil {
			res.Existed = true
			if !opts.Force && !opts.DryRun {
				results = append(results, res)
				errs = append(errs, fmt.Errorf("%s (%s): %w", t.ID, abs, ErrExists))
				continue
			}
		}

		if !opts.DryRun {
			if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
				results = append(results, res)
				errs = append(errs, fmt.Errorf("%s: mkdir %s: %w", t.ID, filepath.Dir(abs), err))
				continue
			}
			if err := os.WriteFile(abs, body, 0o644); err != nil {
				results = append(results, res)
				errs = append(errs, fmt.Errorf("%s: write %s: %w", t.ID, abs, err))
				continue
			}
			res.Wrote = true
		}

		results = append(results, res)
	}

	return results, errors.Join(errs...)
}

func resolveTargets(ids []string) ([]Target, error) {
	if len(ids) == 0 {
		return Targets(), nil
	}
	out := make([]Target, 0, len(ids))
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true
		t, ok := TargetByID(id)
		if !ok {
			return nil, fmt.Errorf("unknown target %q (available: %s)", id, strings.Join(IDs(), ", "))
		}
		out = append(out, t)
	}
	return out, nil
}

func scopeRoot(opts Options) (string, error) {
	if opts.Global {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve user home dir: %w", err)
		}
		return home, nil
	}
	if opts.Cwd != "" {
		return opts.Cwd, nil
	}
	return os.Getwd()
}

// PrintTargetList writes the target registry to w as a human-readable table.
func PrintTargetList(w io.Writer) {
	fmt.Fprintln(w, "Available install-skill targets:")
	fmt.Fprintln(w)
	for _, t := range registry {
		fmt.Fprintf(w, "  %-8s  %-22s  %s\n", t.ID, t.Name, t.path)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Default scope is global (paths resolved against $HOME).")
	fmt.Fprintln(w, "Pass --local to install into the current directory instead.")
}

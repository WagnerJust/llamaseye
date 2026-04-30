package cmd

import (
	"errors"
	"fmt"
	"io"

	"github.com/WagnerJust/llamaseye/internal/skill"
	flag "github.com/spf13/pflag"
)

// IsInstallSkillSubcommand reports whether args[0] selects the
// install-skill subcommand. Callers strip args[0] before passing the
// remainder to RunInstallSkill.
func IsInstallSkillSubcommand(args []string) bool {
	return len(args) > 0 && args[0] == "install-skill"
}

// RunInstallSkill executes the install-skill subcommand. Args is the
// slice *after* the "install-skill" token has been stripped.
//
// Default behavior: dry-run, both targets, global scope. The user must
// pass --apply to actually write files.
func RunInstallSkill(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("llamaseye install-skill", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var (
		target string
		local  bool
		apply  bool
		force  bool
		list   bool
	)
	fs.StringVar(&target, "target", "", "Install only this target (claude|agents). Default: install both.")
	fs.BoolVar(&local, "local", false, "Install into the current directory instead of $HOME.")
	fs.BoolVar(&apply, "apply", false, "Actually write files (default: dry-run that only prints the plan).")
	fs.BoolVar(&force, "force", false, "Overwrite an existing skill file at the install path.")
	fs.BoolVar(&list, "list", false, "Print available targets and exit.")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "Usage: llamaseye install-skill [--target claude|agents] [--local] [--apply] [--force] [--list]")
		fmt.Fprintln(stderr)
		fmt.Fprintln(stderr, "Installs the llamaseye operational skill into the directories that")
		fmt.Fprintln(stderr, "coding-agent tools read. Defaults to both Claude (.claude/skills/) and the")
		fmt.Fprintln(stderr, "cross-agent (.agents/skills/) conventions, at user scope ($HOME).")
		fmt.Fprintln(stderr)
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	if list {
		skill.PrintTargetList(stdout)
		return nil
	}

	var ids []string
	switch target {
	case "":
		// default: both targets
	case "claude", "agents":
		ids = []string{target}
	case "both", "all":
		// explicit "both" — same as default
	default:
		return fmt.Errorf("--target: unknown value %q (want: claude, agents, or omit for both)", target)
	}

	results, runErr := skill.Install(skill.Options{
		TargetIDs: ids,
		Global:    !local,
		DryRun:    !apply,
		Force:     force,
	})

	scope := "global ($HOME)"
	if local {
		scope = "local (cwd)"
	}
	mode := "dry-run"
	if apply {
		mode = "apply"
	}
	fmt.Fprintf(stdout, "llamaseye install-skill — scope: %s, mode: %s\n\n", scope, mode)

	for _, r := range results {
		status := planStatus(r, apply)
		fmt.Fprintf(stdout, "  [%s] %-8s  %s  (%d bytes)\n", status, r.Target.ID, r.AbsPath, r.Bytes)
	}

	if !apply {
		fmt.Fprintln(stdout)
		fmt.Fprintln(stdout, "Re-run with --apply to write the file(s).")
	}

	if runErr != nil {
		// Hide the noisy --force hint if every error was ErrExists and
		// the user is in dry-run mode (the plan output already shows it).
		if !apply && allExistErrors(runErr) {
			return nil
		}
		return runErr
	}

	if apply {
		fmt.Fprintln(stdout)
		fmt.Fprintln(stdout, "Done.")
	}
	return nil
}

func planStatus(r skill.Result, apply bool) string {
	switch {
	case r.Wrote:
		return "wrote"
	case r.Existed && !apply:
		return "exists"
	case r.Existed:
		return "skip"
	case !apply:
		return "plan"
	default:
		return "?"
	}
}

func allExistErrors(err error) bool {
	if err == nil {
		return false
	}
	var multi interface{ Unwrap() []error }
	if errors.As(err, &multi) {
		for _, e := range multi.Unwrap() {
			if !errors.Is(e, skill.ErrExists) {
				return false
			}
		}
		return true
	}
	return errors.Is(err, skill.ErrExists)
}

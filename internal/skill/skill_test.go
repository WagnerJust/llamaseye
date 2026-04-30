package skill_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/WagnerJust/llamaseye/internal/skill"
	"github.com/WagnerJust/llamaseye/skills"
)

func TestEmbeddedBodyHasFrontmatter(t *testing.T) {
	body := skills.Body()
	if len(body) == 0 {
		t.Fatal("skills.Body() returned empty bytes")
	}
	if !strings.HasPrefix(string(body), "---\n") {
		t.Errorf("expected YAML frontmatter at top of skill body, got: %q", string(body[:min(40, len(body))]))
	}
	if !strings.Contains(string(body), "name: llamaseye") {
		t.Error("frontmatter missing 'name: llamaseye'")
	}
}

func TestTargetsAndIDs(t *testing.T) {
	ids := skill.IDs()
	if len(ids) != 2 {
		t.Fatalf("want 2 targets, got %d (%v)", len(ids), ids)
	}
	want := map[string]string{
		"claude": ".claude/skills/llamaseye/SKILL.md",
		"agents": ".agents/skills/llamaseye/SKILL.md",
	}
	for id, expected := range want {
		got, ok := skill.TargetByID(id)
		if !ok {
			t.Errorf("TargetByID(%q) not found", id)
			continue
		}
		if got.Path() != expected {
			t.Errorf("target %q path = %q, want %q", id, got.Path(), expected)
		}
	}
}

func TestInstallDryRun(t *testing.T) {
	dir := t.TempDir()
	results, err := skill.Install(skill.Options{
		Cwd:    dir,
		DryRun: true,
	})
	if err != nil {
		t.Fatalf("dry-run install: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	for _, r := range results {
		if r.Wrote {
			t.Errorf("dry-run target %q reported Wrote=true", r.Target.ID)
		}
		if _, statErr := os.Stat(r.AbsPath); statErr == nil {
			t.Errorf("dry-run target %q file appeared at %s", r.Target.ID, r.AbsPath)
		}
	}
}

func TestInstallApply(t *testing.T) {
	dir := t.TempDir()
	results, err := skill.Install(skill.Options{
		Cwd:    dir,
		DryRun: false,
	})
	if err != nil {
		t.Fatalf("apply install: %v", err)
	}
	body := skills.Body()
	for _, r := range results {
		if !r.Wrote {
			t.Errorf("target %q Wrote=false after apply", r.Target.ID)
		}
		got, readErr := os.ReadFile(r.AbsPath)
		if readErr != nil {
			t.Fatalf("read %s: %v", r.AbsPath, readErr)
		}
		if string(got) != string(body) {
			t.Errorf("target %q wrote %d bytes; want %d (mismatch)", r.Target.ID, len(got), len(body))
		}
	}
}

func TestInstallTargetSelection(t *testing.T) {
	dir := t.TempDir()
	results, err := skill.Install(skill.Options{
		TargetIDs: []string{"claude"},
		Cwd:       dir,
		DryRun:    false,
	})
	if err != nil {
		t.Fatalf("install claude only: %v", err)
	}
	if len(results) != 1 || results[0].Target.ID != "claude" {
		t.Fatalf("want one claude result, got %+v", results)
	}
	if _, err := os.Stat(filepath.Join(dir, ".agents/skills/llamaseye/SKILL.md")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("agents target should not exist when only claude selected; got err=%v", err)
	}
}

func TestInstallExistingFileWithoutForce(t *testing.T) {
	dir := t.TempDir()
	// Pre-create the claude target file to trigger ErrExists.
	claudePath := filepath.Join(dir, ".claude/skills/llamaseye/SKILL.md")
	if err := os.MkdirAll(filepath.Dir(claudePath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(claudePath, []byte("preexisting"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := skill.Install(skill.Options{
		TargetIDs: []string{"claude"},
		Cwd:       dir,
		DryRun:    false,
		Force:     false,
	})
	if !errors.Is(err, skill.ErrExists) {
		t.Fatalf("want ErrExists, got %v", err)
	}
	got, _ := os.ReadFile(claudePath)
	if string(got) != "preexisting" {
		t.Errorf("file should not have been overwritten without --force; got %q", string(got))
	}
}

func TestInstallExistingFileWithForce(t *testing.T) {
	dir := t.TempDir()
	claudePath := filepath.Join(dir, ".claude/skills/llamaseye/SKILL.md")
	if err := os.MkdirAll(filepath.Dir(claudePath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(claudePath, []byte("preexisting"), 0o644); err != nil {
		t.Fatal(err)
	}

	results, err := skill.Install(skill.Options{
		TargetIDs: []string{"claude"},
		Cwd:       dir,
		DryRun:    false,
		Force:     true,
	})
	if err != nil {
		t.Fatalf("install with --force: %v", err)
	}
	if len(results) != 1 || !results[0].Wrote {
		t.Fatalf("want one wrote=true result, got %+v", results)
	}
	got, _ := os.ReadFile(claudePath)
	if string(got) == "preexisting" {
		t.Error("file was not overwritten despite --force")
	}
}

func TestInstallUnknownTarget(t *testing.T) {
	dir := t.TempDir()
	_, err := skill.Install(skill.Options{
		TargetIDs: []string{"bogus"},
		Cwd:       dir,
		DryRun:    true,
	})
	if err == nil || !strings.Contains(err.Error(), "unknown target") {
		t.Fatalf("want unknown-target error, got %v", err)
	}
}

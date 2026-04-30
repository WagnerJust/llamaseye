# AI Context Pointer

All project standards, tech stack details, architecture, and rules are centralized in:

**@AGENTS.md**

Always read AGENTS.md before starting a task.

## Claude Code-Specific Notes

- The canonical operational skill lives at `skills/llamaseye.md` and is embedded into the binary.
- To make it discoverable to Claude Code, run `llamaseye install-skill --apply` once — that writes it to `~/.claude/skills/llamaseye/SKILL.md`.
- `.claude/skills/` and `.agents/skills/` directories in this repo are gitignored. Anything placed there by `--local` installs is for local agent use and stays out of source control.

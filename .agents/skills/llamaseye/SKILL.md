---
name: llamaseye
description: Operating the llamaseye sweep tool, running benchmarks, and interpreting results.
---

# llamaseye Skill

This skill covers running and interpreting llamaseye benchmark sweeps. For full documentation, see:

- **AGENTS.md** (root) -- project architecture, phases, flag reference
- **.claude/skills/llamaseye/SKILL.md** -- detailed operational guide with SSH paths and examples
- **docs/spec.md** -- engineering specification
- **README.md** -- user-facing documentation
- **example.env** -- all environment variables with defaults

## Quick Reference

```bash
# Build
go build -o llamaseye .

# Run a sweep
./llamaseye --model ~/Models/model.gguf --output-dir ./results

# Resume interrupted sweep
./llamaseye --model ~/Models/model.gguf --resume

# Specific phases only
./llamaseye --model ~/Models/model.gguf --only-phases 6,7

# Dry run
./llamaseye --model ~/Models/model.gguf --dry-run
```

## Key Outputs

- `sweep.jsonl` -- structured run data (source of truth)
- `sweep.md` -- human-readable summary with best configs
- `state.json` -- resume state

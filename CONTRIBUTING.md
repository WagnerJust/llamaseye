# Contributing to llamaseye

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development setup

```bash
git clone https://github.com/WagnerJust/llamaseye
cd llamaseye
go build -o llamaseye .
go test ./...
```

Requires Go 1.22+.

## Making changes

1. **Open an issue first** — describe the bug or feature you want to work on.
2. **Create a branch** from `main`: `git checkout -b feat/my-feature` or `fix/my-bug`.
3. **Write code** — follow existing patterns and conventions in the codebase.
4. **Run tests**: `go test ./...`
5. **Run vet**: `go vet ./...`
6. **Open a PR** against `main`.

## PR requirements

Every PR that changes behavior must update:

- [ ] `CHANGELOG.md` — add an entry with a semver bump ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format)
- [ ] `README.md` — if user-facing behavior changed
- [ ] `docs/spec.md` — if phase behavior, JSONL schema, or output format changed
- [ ] `skills/llamaseye.md` — if flags, phases, or usage patterns changed (this file is embedded into the binary and installed by `llamaseye install-skill`)

Version bumps follow [Semantic Versioning](https://semver.org/):
- **Patch** (`x.y.Z`) — bug fixes, cleanup, docs
- **Minor** (`x.Y.0`) — new features, new flags
- **Major** (`X.0.0`) — breaking changes to CLI, output format, or JSONL schema

## Code style

- Follow standard Go conventions (`gofmt`, `go vet`).
- Keep functions focused — one function, one job.
- Use typed representations over `interface{}` / `any` where possible.
- Don't add features beyond what the PR addresses.

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- llamaseye version (`./llamaseye --version`)
- Hardware info (OS, GPU, backend)

## Feature requests

Open an issue describing the use case and proposed solution. For large changes, discuss the approach before writing code.

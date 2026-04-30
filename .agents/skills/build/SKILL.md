---
name: build
description: Build, test, and lint the llamaseye project.
---

# Build Skill

## Build

```bash
go build -o llamaseye .
```

## Test

```bash
go test ./...
```

## Lint

```bash
golangci-lint run
```

## Full Verification

```bash
go build -o llamaseye . && go test ./... && golangci-lint run
```

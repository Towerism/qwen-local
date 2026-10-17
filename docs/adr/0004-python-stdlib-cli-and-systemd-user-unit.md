# ADR 0004: One Python standard-library CLI (`qlm`) and a systemd user service

- **Status:** Accepted (2026-10-16)
- **Date:** 2026-10-16

## Decision

- The scripts are one command, `qlm`, written in Python 3 using only the standard library, with subcommands for each job.
- Tests use pytest, which is already installed (9.1.1, on Python 3.14.8).
- The server runs as a systemd **user** service, `qlm-server.service`. Its settings come from an environment file that `qlm` writes.

## Context

The jobs involve shell work (build, download, systemctl) and HTTP/JSON work (smoke tests, benchmarks, tool-call checks).

- `jq` is not installed, so pure Bash would need a new dependency or fragile parsing.
- Bash scripts are hard to unit-test, and `bats` and `shellcheck` aren't installed.
- Python's standard library covers HTTP (`urllib`), JSON, SHA-256 hashing and subprocess calls.
- A single entry point keeps the shared parts in one place: paths, profiles and the video-memory (VRAM) check.

A user service, not a system one, because:

- Installing and restarting it needs no `sudo`.
- The model files and build stay under the home directory.
- It runs only while the user is logged in, which matches interactive use.

llama-server reads every flag from a `LLAMA_ARG_*` environment variable, so a profile can be a plain env file loaded with `EnvironmentFile=`. Nothing needs a generated command line.

## Options considered

| Option | Testability | Dependencies | Verdict |
|---|---|---|---|
| **Python stdlib + pytest (chosen)** | Unit tests with fakes for subprocess and HTTP | None new | Chosen |
| Bash scripts | Needs bats; JSON needs jq | jq, bats, shellcheck | Rejected |
| Docker Compose | Container hides the build; GPU passthrough adds a layer | nvidia-container-toolkit | Rejected; ADR 0001 needs a pinned native build |
| System service (root) | Same | sudo for every change | Rejected |

## Consequences

- `qlm` runs with `python3 -I`, so a stray module in the working directory can't shadow the standard library.
- The user service needs `loginctl enable-linger` only if the server should outlive a logout. That is out of scope; the default doesn't enable it.

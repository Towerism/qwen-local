# qwen-local

`qlm` sets up and runs Qwen3.8-27B (Q4_K_M) on one RTX 4090 with llama.cpp's `llama-server`, as the local model behind the [Con](https://github.com/nowledge-co/con-terminal) AI terminal.

**Status:** `qlm doctor`, `qlm install` and `qlm fetch` work. Starting the server and connecting Con come next ([#4](https://github.com/Towerism/qwen-local/issues/4)), then benchmarks and updates ([#5](https://github.com/Towerism/qwen-local/issues/5)).

Read these first:

- [docs/architecture.md](docs/architecture.md): what `qlm` does, its commands, file locations, the default settings and how it's tested.
- [docs/adr/](docs/adr/): the decisions behind it and the reasons for each.
  - [0001](docs/adr/0001-runtime-llama-cpp-server.md): llama-server instead of Ollama, for multi-token prediction and prompt-cache reuse.
  - [0002](docs/adr/0002-model-file-and-vram-budget.md): the model file and the video-memory (VRAM) budget.
  - [0003](docs/adr/0003-con-integration.md): how Con connects.
  - [0004](docs/adr/0004-python-stdlib-cli-and-systemd-user-unit.md): a Python standard-library CLI with a systemd user service.

The design targets one machine: i9-13900KF, 62 GiB RAM, RTX 4090 24 GB, Arch Linux. On other hardware, redo the VRAM arithmetic in ADR 0002 before using the settings.

## First run

The first time, run `bin/qlm` from the clone; `qlm install` then links it into `~/.local/bin`.

```bash
bin/qlm doctor      # check the machine; it changes nothing
bin/qlm install     # build llama.cpp with CUDA, install the service unit, profile and API key
qlm fetch           # download and verify the model files (about 17.8 GB)
```

`qlm doctor` prints the line to add to `~/.zshrc` if `~/.local/bin` isn't on your `PATH`.

Still to come: `qlm start`, `qlm smoke` and `qlm con --apply` (#4).

## Tests

```bash
python3 -I -m pytest            # unit and integration tests, no network or GPU, under a second
python3 -I -m pytest -m machine # real tools on this machine; hashes the real model files
```

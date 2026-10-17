# qwen-local

`qlm` sets up and runs Qwen3.8-27B (Q4_K_M) on one RTX 4090 with llama.cpp's `llama-server`, as the local model behind the [Con](https://github.com/nowledge-co/con-terminal) AI terminal.

**Status:** design only. The commands below are planned, and none exist yet.

Read these first:

- [docs/architecture.md](docs/architecture.md): what `qlm` does, its commands, file locations, the default settings and how it's tested.
- [docs/adr/](docs/adr/): the decisions behind it and the reasons for each.
  - [0001](docs/adr/0001-runtime-llama-cpp-server.md): llama-server instead of Ollama, for multi-token prediction and prompt-cache reuse.
  - [0002](docs/adr/0002-model-file-and-vram-budget.md): the model file and the video-memory (VRAM) budget.
  - [0003](docs/adr/0003-con-integration.md): how Con connects.
  - [0004](docs/adr/0004-python-stdlib-cli-and-systemd-user-unit.md): a Python standard-library CLI with a systemd user service.

The design targets one machine: i9-13900KF, 62 GiB RAM, RTX 4090 24 GB, Arch Linux. On other hardware, redo the VRAM arithmetic in ADR 0002 before using the settings.

The planned first run:

```bash
qlm doctor          # check the machine
qlm install         # build llama.cpp with CUDA, install the service
qlm fetch           # download and verify the model
qlm start           # start the server once VRAM is free
qlm smoke           # prove tool calling works
qlm con --apply     # point Con at the server (close Con first)
```

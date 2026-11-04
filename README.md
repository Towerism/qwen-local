# qwen-local

`qlm` installs llama.cpp and runs a pinned local model as a systemd user service.

**Status:** every command in the design works: set up, run and check the server, benchmark, update llama.cpp and uninstall.

Read these first:

- [docs/architecture.md](docs/architecture.md): what `qlm` does, its commands, file locations, the default settings and how it's tested.
- [docs/adr/](docs/adr/): the decisions behind it and the reasons for each.
  - [0001](docs/adr/0001-runtime-llama-cpp-server.md): llama-server instead of Ollama, for multi-token prediction (MTP) and prompt-cache reuse.
  - [0002](docs/adr/0002-model-file-and-vram-budget.md): the model file and the video-memory (VRAM) budget.
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

Then start the server and check it:

```bash
qlm start           # check free VRAM, start the server and wait until it is ready
qlm smoke           # eight API checks: tool calls, thinking, MTP and the API key
```

Clients reach the server's OpenAI-compatible API at `http://127.0.0.1:9931/v1` with the key in `~/.config/qlm/api-key`.

## Everyday use

| Command | What it does |
|---|---|
| `qlm start`, `qlm stop` | Start the server (it holds about 19 GiB of VRAM while it runs) or stop it |
| `qlm status` | State, profile, uptime, VRAM and the largest request so far |
| `qlm logs [-n N] [-f]` | The server's journal |
| `qlm profile use <agent\|long\|vision>` | Switch profile; restarts the server if it is running. `long` has 96k context without MTP; `vision` needs `qlm fetch --vision` first |

## Measuring, updating and removing

```bash
qlm bench --quick                                   # a few seconds: one prompt size, one generation, no restarts
qlm bench                                           # about 10 minutes: restarts the server twice to compare MTP on and off
qlm update --tag b11700 --commit <40-character sha> # build a newer llama.cpp; goes back by itself if smoke fails
qlm uninstall                                       # lists what it removes and asks; --models also removes the 17.8 GB of models
```

`qlm bench` needs the server running, and other clients should stay idle until it finishes, because the bench replaces the server's cached prompt. Each run is saved under `~/.local/state/qlm/bench/` and compared with the previous run of the same kind and profile.

`qlm update` trusts the commit you give it, so copy it from the tag's page on [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tags). The pinned tag in [ADR 0001](docs/adr/0001-runtime-llama-cpp-server.md) and `qlm/pins.py` doesn't move by itself; update both by hand if the new tag should become the pinned one.

`qlm uninstall` keeps `~/.local/state/qlm/`, where the bench results and logs live.

## Tests

```bash
python3 -I -m pytest            # unit and integration tests, no network or GPU, a few seconds
python3 -I -m pytest -m machine # real tools on this machine; starts and stops the real server
```

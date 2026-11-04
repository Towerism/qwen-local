# qwen-local

`qlm` installs llama.cpp and runs one model from its catalog with `llama-server`, as a systemd user service. Clients reach the server's OpenAI-compatible API at `http://127.0.0.1:9931/v1` with the key in `~/.config/qlm/api-key`.

**Status:** every command in the design works: select a model, set up, run and check the server, benchmark, update llama.cpp and uninstall.

Read these first:

- [docs/architecture.md](docs/architecture.md): what `qlm` does, its commands, file locations, the default settings and how it's tested.
- [docs/adr/](docs/adr/): the decisions behind it and the reasons for each. ADR numbers are never reused.
  - [0001](docs/adr/0001-runtime-llama-cpp-server.md): llama-server instead of Ollama, for multi-token prediction (MTP) and prompt-cache reuse.
  - [0002](docs/adr/0002-model-file-and-vram-budget.md): the Qwen3.8-27B model files and the video-memory (VRAM) budget.
  - 0003: withdrawn.
  - [0004](docs/adr/0004-python-stdlib-cli-and-systemd-user-unit.md): a Python standard-library CLI with a systemd user service.
  - [0005](docs/adr/0005-qwen3.6-35b-a3b-model-files-and-vram-budget.md): the Qwen3.6-35B-A3B model files, why it is the default, and its VRAM budget.

The design targets one machine: i9-13900KF, 62 GiB RAM, RTX 4090 24 GB, Arch Linux. On other hardware, redo the VRAM arithmetic in ADRs 0002 and 0005 before using the settings.

## Models

| Model id | Files (each pinned by repo, revision, size and SHA-256) | Download | Profiles |
|---|---|---|---|
| `qwen3.6-35b-a3b` (default) | `Qwen3.6-35B-A3B.Q4_K_M.gguf` and its MTP draft | 22.2 GB | `agent`, `long` |
| `qwen3.8-27b` | `Qwen3.8-27B-UD-Q4_K_M.gguf`, its MTP draft, and with `--vision` the vision projector | 17.8 GB, plus 0.9 GB for vision | `agent`, `long`, `vision` |

The 35B model generates about 3 times faster than the 27B one on this machine, which is why fresh installs use it. Its VRAM margin is small; see [VRAM](#vram).

## First run

The first time, run `bin/qlm` from the clone; `qlm install` then links it into `~/.local/bin`.

```bash
bin/qlm doctor      # check the machine; it changes nothing
bin/qlm install     # build llama.cpp with CUDA, install the service unit, profile and API key
qlm fetch           # download and verify the selected model's files
```

`qlm doctor` prints the line to add to `~/.zshrc` if `~/.local/bin` isn't on your `PATH`.

A fresh install selects `qwen3.6-35b-a3b`. To start with the 27B model instead, run `qlm use qwen3.8-27b` before `qlm fetch`.

If you already have one of the model files, link it instead of downloading it:

```bash
qlm fetch --from ~/models/Qwen3.6-35B-A3B.Q4_K_M.gguf
```

`--from` accepts only a file whose size and SHA-256 match a pinned file of the selected model, so run `qlm use` first when the file belongs to the other model. qlm makes a symlink to the file and never copies, moves or deletes it. Don't modify the linked file: qlm checks it when linking and on each `qlm fetch`, but not when the server starts. Files not given with `--from` are downloaded as usual.

Then start the server and check it:

```bash
qlm start           # check free VRAM, start the server and wait until it is ready
qlm smoke           # API checks: tool calls, thinking, MTP and the API key
```

## Switching models

```bash
qlm use qwen3.8-27b    # select the model; restarts the server if it is running
```

`qlm use` saves the choice in `~/.config/qlm/model` and switches to the new model's profile of the same name, or to `agent` if it has none. If the new model's files aren't there yet, it saves the choice only, leaves a running server alone, and says to run `qlm fetch` and then `qlm start`. `qlm status` shows the model the server runs, plus a `selected:` line when the selection differs.

An install from before model selection keeps running Qwen3.8-27B until you run `qlm use`.

## Everyday use

| Command | What it does |
|---|---|
| `qlm start [profile]`, `qlm stop` | Start the server, first switching to the named profile of the selected model; or stop it |
| `qlm status` | State, model, profile, uptime, VRAM and the largest request so far |
| `qlm logs [-n N] [-f]` | The server's journal |
| `qlm profile use <profile>` | Switch to a profile of the selected model; restarts the server if it is running |

Each model has its own profiles, in `profiles/<model id>/`:

| Model | Profile | Context | MTP | Server VRAM |
|---|---|---|---|---|
| `qwen3.6-35b-a3b` | `agent` | 48k | On | 21.8 GiB |
| `qwen3.6-35b-a3b` | `long` | 256k, with the experts of 4 layers in system RAM, so it generates at about half of agent's speed | Off | 21.7 GiB |
| `qwen3.8-27b` | `agent` | 64k | On | 20.3 GiB |
| `qwen3.8-27b` | `long` | 96k | Off | 20.0 GiB |
| `qwen3.8-27b` | `vision` | 32k, reads images; needs `qlm fetch --vision` first | Off | 18.7 GiB |

The 35B model has no vision profile, because its pinned repo has no projector file. Asking for `vision` with 35B selected exits with `the vision profile is not available for qwen3.6-35b-a3b`.

### VRAM

`qlm start` needs the profile's VRAM plus 0.5 GiB free. For the 35B profiles that is 22.3 GiB of the card's 24 GiB, so it refuses when the desktop and other apps hold more than about 1.7 GiB. The KDE screen locker alone holds about 278 MiB. When `qlm start` refuses, it lists the processes holding VRAM: close what you can, usually browser windows, and run it again, or switch to the 27B model with `qlm use qwen3.8-27b`.

## Measuring, updating and removing

```bash
qlm bench --quick                                   # a few seconds: one prompt size, one generation, no restarts
qlm bench                                           # about 10 minutes: restarts the server twice to compare MTP on and off
qlm update --tag b11700 --commit <40-character sha> # build a newer llama.cpp; goes back by itself if smoke fails
qlm uninstall                                       # lists what it removes and asks; --models also removes the model files
```

`qlm bench` needs the server running, and other clients should stay idle until it finishes, because the bench replaces the server's cached prompt. Each run is saved under `~/.local/state/qlm/bench/` and compared with the previous run of the same kind, model and profile.

`qlm update` trusts the commit you give it, so copy it from the tag's page on [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tags). The pinned tag in [ADR 0001](docs/adr/0001-runtime-llama-cpp-server.md) and `qlm/pins.py` doesn't move by itself; update both by hand if the new tag should become the pinned one.

`qlm uninstall` keeps `~/.local/state/qlm/`, where the bench results and logs live. With `--models`, a model linked with `--from` loses its link, and your copy stays.

## Tests

```bash
python3 -I -m pytest            # unit and integration tests, no network or GPU, a few seconds
python3 -I -m pytest -m machine # real tools on this machine; starts and stops the real server
```

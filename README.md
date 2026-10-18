# qwen-local

`qlm` sets up and runs Qwen3.8-27B (Q4_K_M) on one RTX 4090 with llama.cpp's `llama-server`, as the local model behind the [Con](https://github.com/nowledge-co/con-terminal) AI terminal.

**Status:** setting up, running the server, checking it and connecting Con all work. Benchmarks, updates and uninstall come next ([#5](https://github.com/Towerism/qwen-local/issues/5)).

Read these first:

- [docs/architecture.md](docs/architecture.md): what `qlm` does, its commands, file locations, the default settings and how it's tested.
- [docs/adr/](docs/adr/): the decisions behind it and the reasons for each.
  - [0001](docs/adr/0001-runtime-llama-cpp-server.md): llama-server instead of Ollama, for multi-token prediction (MTP) and prompt-cache reuse.
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

Then start the server, check it, and point Con at it. Quit Con before `qlm con --apply`, because Con must not be running while its config changes:

```bash
qlm start           # check free VRAM, start the server and wait until it is ready
qlm smoke           # eight API checks: tool calls, thinking, MTP and the API key
qlm con --print     # show what --apply would change in ~/.config/con/con.conf
qlm con --apply     # write the settings, after a backup; undo with qlm con --revert
```

Start Con again, open a new tab and ask its agent something. That last check is by hand.

Tabs that Con restores from your last session keep the provider they had before, because Con saves each tab's provider in `~/.local/share/con/session.json` and that overrides `con.conf`. Only new tabs pick up the local model. If a restored tab says `No API key found for anthropic`, switch that tab's provider to OpenAI-compatible in Con's provider picker, or close it and open a new one.

## Everyday use

| Command | What it does |
|---|---|
| `qlm start`, `qlm stop` | Start the server (it holds about 19 GiB of VRAM while it runs) or stop it |
| `qlm status` | State, profile, uptime, VRAM and the largest request so far |
| `qlm logs [-n N] [-f]` | The server's journal |
| `qlm profile use <agent\|long\|vision>` | Switch profile; restarts the server if it is running. `long` has 96k context without MTP; `vision` needs `qlm fetch --vision` first |
| `qlm suggest --enable` / `--disable` | Turn Con's shell suggestions on, from a small CPU-only Ollama model (about 1 GB download), or off |

## Tests

```bash
python3 -I -m pytest            # unit and integration tests, no network or GPU, a few seconds
python3 -I -m pytest -m machine # real tools on this machine; starts and stops the real server
```

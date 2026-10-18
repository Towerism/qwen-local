# ADR 0001: Serve the model with llama.cpp's llama-server, not Ollama

- **Status:** Accepted (2026-10-16)
- **Date:** 2026-10-16

## Decision

Run Qwen3.8-27B with `llama-server` from llama.cpp, built from a pinned release tag with CUDA (NVIDIA's GPU toolkit) for the RTX 4090 (compute capability 8.9). Ollama stays installed but must not hold a model in video memory (VRAM) while `llama-server` runs.

## Context

Con drives the model in a loop: long prompt, tool call, tool result, repeat. Two runtime properties decide how usable that loop is on one 24 GB card.

1. **Generation speed.** Qwen3.8-27B ships a multi-token prediction (MTP) head. With it, the runtime drafts several tokens per step and the main model verifies them. A published RTX 4090 test of the closely related Qwen3.6-27B at Q4_K_M measured 46.8 tok/s without MTP and about 133 tok/s with it on code-editing turns, using llama.cpp ([Pedro Alonso, 2026](https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090/)). Ollama lists `mtp` tags, but no source confirms Ollama runs MTP decoding on CUDA.
2. **Prompt cache reuse.** Qwen3.8 is a hybrid model. 48 of its 64 layers are linear-attention (Gated DeltaNet) layers whose state can't be rolled back to an earlier token. A runtime can only reuse a cached prompt if it saved a checkpoint of that state at or before the point where the new prompt differs. llama-server does this with `--ctx-checkpoints` (default 32 per slot). Reports show Ollama re-processing the whole prompt on hybrid Qwen models ([llama.cpp #20225](https://github.com/ggml-org/llama.cpp/issues/20225), [local-dictation-cleanup #3](https://github.com/xuancongwen/local-dictation-cleanup/issues/3)). At an estimated 1,500–2,500 tok/s prompt processing, re-reading a 50k-token context costs 20–35 s on every turn.

llama-server also gives explicit control over every setting this design depends on: GPU layers, KV-cache type, slot count, whether the vision projector loads, and sampling defaults. It serves both an OpenAI-compatible API (`/v1/chat/completions`) and an Anthropic-compatible one (`/v1/messages`).

## Options considered

| Option | MTP on CUDA | Hybrid prompt cache | Control | Setup cost |
|---|---|---|---|---|
| **llama-server (chosen)** | Yes (`--spec-type draft-mtp`) | Checkpoints, documented | Full | Build from source, ~10 min |
| Ollama 0.40.x | Unconfirmed | Reported to re-process | Env vars only; projector always loads | Already installed |
| vLLM | Yes | Yes | Full | Needs FP8/AWQ weights and more VRAM; Q4_K_M GGUF is not its native format |
| LM Studio | Uses llama.cpp, lags upstream | Depends on bundled build | GUI-first | Installed (`~/.lmstudio`) |

## Consequences

- The pinned source is tag `b11669`, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Tags can be moved, so `qlm install` checks the commit after checkout. `qlm update` takes the new commit as an argument and checks it the same way.
- `qlm update --tag <bN> --commit <sha>` builds another tag beside the pinned one and points `current` at it if smoke passes. The pin in this ADR and `qlm/pins.py` moves only with an edit to both, and a test keeps them equal (added 2026-10-17, issue #5).
- We own a build step. `qlm install` builds a pinned tag and `qlm update` rebuilds into a new directory, so a bad build never replaces a working one (see the architecture doc).
- The Ollama service can keep running for other uses, but `qlm start` refuses to start while Ollama has a model loaded, because both would compete for the same VRAM.
- The host has CUDA 13.4 and GCC (the GNU compiler) 16. If `nvcc` rejects GCC 16, the build falls back to `g++-15`, which is installed.
- If a future Ollama release confirms CUDA MTP and checkpointed caching, revisit this ADR.

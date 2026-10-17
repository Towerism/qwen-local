# ADR 0002: Use Unsloth's UD-Q4_K_M file with the multi-token prediction draft file, 64k context, 8-bit KV cache

- **Status:** Proposed
- **Date:** 2026-10-16

## Decision

| Setting | Value |
|---|---|
| Model file | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_M.gguf` (16,464,440,224 bytes) |
| Multi-token prediction (MTP) draft file | `unsloth/Qwen3.8-27B-GGUF`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 bytes) |
| Vision projector | Not loaded (`--no-mmproj`) in the default profile |
| Context | 65,536 tokens |
| Key/value (KV) cache | `q8_0` for K and V, flash attention on |
| Slots | 1 (`--parallel 1`) |
| GPU layers | All (`-ngl all`), with `--fit off` so a misfit fails at startup instead of silently moving layers to the CPU |

## Context

The card has 24,564 MiB (23.99 GiB) of video memory (VRAM). The KDE Plasma desktop, browsers and other apps held 2,565 MiB when measured on 2026-10-16, and the 13900KF has no integrated GPU to move them to. That leaves about 21.5 GiB.

Several published files are labelled Q4_K_M, and their sizes differ by 2.3 GiB:

| File | Size | Fits 64k with MTP? |
|---|---|---|
| `unsloth/...UD-Q4_K_M.gguf` | 15.33 GiB | Yes, ~1.2 GiB spare |
| Ollama `qwen3.8:27b-q4_K_M` | 15.8 GiB + 0.87 GiB projector | MTP not confirmed in Ollama |
| `ggml-org/...Q4_K_M.gguf` | 17.67 GiB | No (server needs ~22.7 GiB, 25.2 with the desktop) |

Unsloth's "UD" (Unsloth Dynamic) quant keeps Q4_K_M as the base type but chooses per-tensor precision using an importance matrix, which is how it ends up smaller. Its quality is reported to be on par with or better than a uniform Q4_K_M. That claim comes from the publisher, and `qlm bench` doesn't measure quality.

## VRAM arithmetic (default profile)

The KV cache is small because only 16 of the 64 layers use full attention (`full_attention_interval: 4` in [config.json](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)).

| Item | Calculation | GiB |
|---|---|---|
| Main weights | 16,464,440,224 B | 15.33 |
| MTP weights | 1,369,590,656 B | 1.28 |
| KV cache, main | 16 layers × 4 KV heads × 256 dim × 2 (K, V) = 32,768 values/token × 1.0625 B (q8_0) × 65,536 tokens | 2.13 |
| KV cache, MTP layer | 1 layer, same shape | 0.13 |
| Linear-attention state | 48 layers × 48 heads × 128 × 128 × 4 B | 0.14 |
| Compute buffers + CUDA context | estimate at `-ub 512` | 1.3 |
| **Server total** | | **20.3** |
| Desktop (measured) | | 2.5 |
| **Card total** | | **22.8 of 23.99** |

Context checkpoints and the prompt cache (`--cache-ram`) live in system RAM, not VRAM. Each checkpoint holds one copy of the linear-attention state (~0.14 GiB), so 32 checkpoints use about 4.5 GiB of the 62 GiB RAM.

## Profiles

| Profile | Context | MTP | Projector | Server VRAM | Use |
|---|---|---|---|---|---|
| `agent` (default) | 64k | On | Off | ~20.3 GiB | Con, everyday |
| `long` | 96k | Off | Off | ~20.0 GiB | Large refactors where context matters more than speed |
| `vision` | 32k | Off | On (BF16, 0.87 GiB) | ~18.7 GiB | Screenshots or images |

## Consequences

- The estimate leaves about 1.2 GiB spare. A browser with many tabs can use that up. `qlm start` checks free VRAM before launching and names the processes holding it.
- `qlm bench` measures the real peak and writes it to the run log, replacing the 1.3 GiB compute estimate with a measured value.
- The model's name, Qwen3.8-27B Q4_K_M, matches what the user picked. The file differs from Ollama's `qwen3.8:27b-q4_K_M` tag; this ADR records why.

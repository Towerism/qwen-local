# ADR 0003: Connect Con through its `openaicompatible` provider, and keep its background requests off the main slot

- **Status:** Proposed
- **Date:** 2026-10-16

## Decision

1. Con talks to llama-server through the `openaicompatible` provider at `http://127.0.0.1:9931/v1`.
2. Sampling settings live on the server. Con sends a temperature only when `con.agent.temperature` is set, so the script leaves that key unset.
3. Con's suggestion model (shell completions and tab labels) never uses llama-server. By default it is turned off. An opt-in setting sends it to a small model in Ollama that runs on the CPU only.
4. `qlm con` edits `~/.config/con/con.conf` only while Con is not running, after a backup, and writes only keys confirmed in Con's source.

The keys it writes:

```ini
con.agent.provider = "openaicompatible"
con.agent.provider_is_explicit = true
con.agent.providers.openaicompatible.base_url = "http://127.0.0.1:9931/v1"
con.agent.providers.openaicompatible.model = "qwen3.8-27b"
con.agent.suggestion_model.enabled = false
```

## Context

These facts come from Con's source, [nowledge-co/con-terminal](https://github.com/nowledge-co/con-terminal) at commit `de2d7c7e`. Paths are relative to that repo.

- **The provider key has no dash.** `openai-compatible` is accepted as a value of `con.agent.provider`, but in a key path only `openaicompatible` is valid. An unknown `con.` key makes the whole config fail to load (`crates/con-core/src/config/ghostty.rs:147-200,268-285`). The key `con.agent.providers.openai-compatible.base_url` would break Con.
- **What the OpenAI-compatible client sends.** It calls `{base_url}/chat/completions`, so `base_url` must end in `/v1`. It streams over server-sent events (SSE) and sends tools in the native `{"type":"function"}` format. It sends `temperature` and `max_tokens` only when set. For a loopback `base_url` it needs no API key and sends no `Authorization` header (`crates/con-agent/src/provider.rs:2464-2494,2477-2487`).
- **Why not the `ollama` provider.** It calls Ollama's native `/api/chat`, never sends `num_ctx`, and gets tool calls back with no `id`, which Con logs as malformed (`provider.rs:2984-2991`). It doesn't fit a llama-server backend anyway.
- **Con has no context management.** It doesn't count tokens or compact. One request can run `max_turns + 2` model calls (32 by default), each resending the system prompt, 34 tool definitions and every tool result so far (`provider.rs:2203,2249-2251`). The fixed part is an estimated 11–13k tokens, plus `AGENTS.md`, a git diff of up to 8,000 characters and pane output. When a request passes 64k tokens, llama-server returns an error and Con shows it. That is the behaviour we want: context shift stays off, so the server never silently drops the start of the prompt.
- **Thinking.** Con shows `reasoning_content` as thinking and doesn't store it in history (`provider.rs:2958-2980`, `crates/con-agent/src/conversation.rs:198-204`). llama-server's `--reasoning-format auto` (`LLAMA_ARG_THINK=auto` in the profile) produces that field.
- **No timeouts.** Con sets no HTTP, first-token or idle timeout (`provider.rs:50`), so a slow prompt-processing step won't abort a turn.
- **Background requests.** The suggestion model is on by default and uses the main provider unless told otherwise (`provider.rs:2045-2091`):
  - Shell completions fire 180 ms after you stop typing.
  - Tab-label summaries are polled every 3 s, at most one per tab per 5 s, with up to 2,048 tokens each.

## Why the suggestion model must stay off llama-server

llama-server runs one slot (ADR 0002). A suggestion request has a different prompt, so it replaces the agent conversation's cached state in that slot. The next agent turn then re-reads its whole context, which at 50k tokens is 20–35 s. The server's RAM prompt cache (`--cache-ram`) can restore some of that, but tab summaries every few seconds would also queue in front of agent turns. A second slot would need its own share of the 64k context.

| Option | Cost |
|---|---|
| **Disable (default)** | No inline shell completions; tab labels stay as the command name |
| Ollama, CPU-only small model (opt-in, `qlm suggest --enable`) | About 1 GiB RAM; completions an estimated 0.3–1 s on the 13900KF; no VRAM |
| Point at llama-server | Evicts the agent's cache; rejected |

The opt-in setting creates an Ollama model `con-suggest` from `qwen2.5-coder:1.5b` with `PARAMETER num_gpu 0`, and sets `con.agent.suggestion_model.provider = "ollama"` and `con.agent.suggestion_model.model = "con-suggest"`. A small coder model without a thinking mode is the right fit, because Con limits these answers to 48–100 tokens.

## Consequences

- The dashed key `con.agent.providers.openai-compatible.*` is easy to write by mistake, because the dashed spelling works as a provider value. It stops Con from loading its config, so `qlm con` writes only `openaicompatible`.
- Con rewrites `con.conf` when you change settings in its UI, so the script refuses to edit while Con runs and keeps a timestamped backup in `~/.local/state/qlm/backups/`.
- `con.agent.auto_context` has no effect in this Con build (`provider.rs:1862` defines it; nothing reads it), so the design doesn't depend on it.
- Con can also reach llama-server through its `anthropic` provider at `/v1/messages`. That path isn't used, but `qlm smoke` tests it so it stays available as a fallback.

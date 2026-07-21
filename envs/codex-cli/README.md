# Codex CLI → LiteLLM (GPT-5.5)

[Codex CLI](https://github.com/openai/codex) is OpenAI's terminal coding agent.
This env runs it against the OSU LiteLLM proxy's **Responses API** endpoint via a
custom provider, using **GPT-5.5** (`gpt-5.5-2026-04-24`).

## Prerequisites

- Nix with flakes enabled.
- Repo-root `.env` (`../../.env`) with `LITELLM_URL` and `LITELLM_KEY` (already present, git-ignored).

## Quickstart

```bash
cd envs/codex-cli
nix develop          # installs codex, loads .env, seeds project-local CODEX_HOME
codex                # TUI, model gpt-5.5-2026-04-24 via LiteLLM
```

One-shot / headless:

```bash
nix develop -c codex exec "Summarize this repo and how to run its checks."
```

## How it's wired

- `flake.nix` provides `codex` (from nixpkgs), sets `CODEX_HOME=envs/codex-cli/.codex`
  (so your global `~/.codex` is untouched), and seeds it from the tracked
  [`config.toml`](./config.toml).
- `config.toml` selects the model and a custom provider:

  ```toml
  model = "gpt-5.5-2026-04-24"
  model_provider = "litellm"

  [model_providers.litellm]
  name = "OSU LiteLLM Proxy"
  base_url = "https://litellm.cloud.osu.edu/v1"
  env_key = "LITELLM_KEY"     # Codex reads the key from $LITELLM_KEY
  wire_api = "responses"
  ```

  `env_key` means the key is read from the environment (`LITELLM_KEY`), never stored
  in the file.

## Why `wire_api = "responses"` (and not `chat`)

Recent Codex removed the Chat Completions protocol — `wire_api = "chat"` is now a hard
error, and `responses` is the only accepted value. The OSU LiteLLM proxy exposes
`POST /v1/responses` (verified), so Codex works against it. If you point this at a proxy
that only offers Chat Completions, Codex will refuse to start.

## Switching models

Edit `model` in [`config.toml`](./config.toml), or override per run:

```bash
nix develop -c codex -m gpt-5.5-pro-2026-04-24 -c model_provider=litellm
```

List the catalog with `curl -s $LITELLM_URL/v1/models -H "Authorization: Bearer $LITELLM_KEY"`.

## Using a different proxy, or going direct to the vendor

`base_url` **defaults to OSU's LiteLLM proxy**. To use a different OpenAI-compatible
proxy, change `base_url` in [`config.toml`](./config.toml) (and `env_key` if that proxy's
key lives in another variable). To go **directly to OpenAI**, set
`base_url = "https://api.openai.com/v1"`, `env_key = "OPENAI_API_KEY"`, and a real OpenAI
model id.

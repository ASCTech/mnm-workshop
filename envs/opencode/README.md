# OpenCode → LiteLLM (Opus 4.8)

[OpenCode](https://opencode.ai) ([anomalyco/opencode](https://github.com/anomalyco/opencode))
is an open-source terminal coding agent. This env runs it against the OSU LiteLLM
proxy's **OpenAI-compatible** endpoint via a custom provider, defaulting to
**Opus 4.8** (`claude-opus-4-8`).

## Prerequisites

- Nix with flakes enabled.
- Repo-root `.env` (`../../.env`) with `LITELLM_URL` and `LITELLM_KEY` (already present, git-ignored).

## Quickstart

```bash
cd envs/opencode
nix develop          # installs opencode, loads .env, exports OPENCODE_CONFIG
opencode             # TUI; provider "OSU LiteLLM Proxy" / model Opus 4.8
```

One-shot / headless:

```bash
nix develop -c opencode run "Summarize this repo and how to run its checks."
```

## How it's wired

- `flake.nix` provides `opencode` (from nixpkgs) and sets `OPENCODE_CONFIG` to the
  project-local [`opencode.json`](./opencode.json).
- `opencode.json` defines a custom provider `litellm` using the
  `@ai-sdk/openai-compatible` adapter:

  ```json
  {
    "provider": {
      "litellm": {
        "npm": "@ai-sdk/openai-compatible",
        "options": {
          "baseURL": "https://litellm.cloud.osu.edu/v1",
          "apiKey": "{env:LITELLM_KEY}"
        },
        "models": { "claude-opus-4-8": { "name": "Claude Opus 4.8 (via LiteLLM)" } }
      }
    },
    "model": "litellm/claude-opus-4-8"
  }
  ```

  `{env:LITELLM_KEY}` is substituted from the environment at load time, so the key
  is never written to the config file.

## Switching models

Add more entries under `provider.litellm.models` (any id from the LiteLLM catalog),
then either set the top-level `model` key or override per run:

```bash
nix develop -c opencode run "hi" --model litellm/claude-sonnet-5
```

List the catalog with:
`curl -s $LITELLM_URL/v1/models -H "Authorization: Bearer $LITELLM_KEY"`.

## Using a different proxy, or going direct to the vendor

`baseURL` in [`opencode.json`](./opencode.json) **defaults to OSU's LiteLLM proxy**. To
use a different OpenAI-compatible proxy, change `baseURL`/`apiKey`. To go **directly to a
vendor**, either point `baseURL` at the vendor's own OpenAI-compatible endpoint, or drop
the custom `litellm` provider entirely and use one of OpenCode's built-in providers (e.g.
`anthropic`, `openai`) with that vendor's key.

## Notes

- The `litellm` provider uses `@ai-sdk/openai-compatible` (the `/v1/chat/completions`
  shape LiteLLM exposes by default). If you route a model through LiteLLM's
  `/v1/responses` passthrough instead, switch that provider's `npm` to `@ai-sdk/openai`.
- OpenCode fetches the AI SDK npm packages on first run, so the first launch needs network.

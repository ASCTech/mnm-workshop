# Pi → LiteLLM (Opus 4.8)

[Pi](https://pi.dev) ([earendil-works/pi](https://github.com/earendil-works/pi),
npm `@earendil-works/pi-coding-agent`) is a minimal, hackable terminal coding agent.
This env runs it against the OSU LiteLLM proxy's **Anthropic Messages** endpoint via a
custom provider, defaulting to **Opus 4.8** (`claude-opus-4-8`).

## Prerequisites

- Nix with flakes enabled.
- Repo-root `.env` (`../../.env`) with `LITELLM_URL` and `LITELLM_KEY` (already present, git-ignored).
- Network access on first `nix develop` (Pi is installed from npm).

## Quickstart

```bash
cd envs/pi
nix develop          # installs Node + Pi CLI locally, seeds project-local config
pi                   # TUI, provider "litellm" / model Opus 4.8
```

One-shot / headless:

```bash
nix develop -c pi "Summarize this repo and how to run its checks."
```

## How it's wired

Pi is **not** in nixpkgs, so the flake provides `nodejs_22` and installs the CLI into a
project-local prefix (`envs/pi/.npm`, git-ignored) on first entry. Its config is kept
project-local via `PI_CODING_AGENT_DIR` so your global `~/.pi/agent` is never touched.

The shell hook:

- sets `PI_CODING_AGENT_DIR=envs/pi/.pi-agent` and seeds it from the tracked templates
  [`models.json`](./models.json) and [`settings.json`](./settings.json);
- `models.json` defines a `litellm` provider (`api: "anthropic-messages"`,
  `baseUrl: https://litellm.cloud.osu.edu`, `apiKey: "$LITELLM_KEY"` — Pi interpolates
  the env var at read, so no secret is stored);
- `settings.json` sets `defaultProvider: "litellm"` and `defaultModel: "claude-opus-4-8"`.

## Switching models

Add entries to the `models` array in [`models.json`](./models.json) (any id from the
LiteLLM catalog), then select per run:

```bash
nix develop -c pi --provider litellm --model claude-sonnet-5 "hi"
```

`pi --list-models` shows what's configured; `/model` (or `Ctrl+L`) switches inside the TUI.
List the catalog with `curl -s $LITELLM_URL/v1/models -H "Authorization: Bearer $LITELLM_KEY"`.

## Notes

- Editing `models.json` / `settings.json` here updates the live config on the next
  `nix develop` (they're copied into `.pi-agent/` each entry).
- **Why `anthropic-messages` and not `openai-completions`:** Pi's OpenAI-completions
  path sends a `store` parameter that the proxy's Bedrock/Anthropic backend rejects
  (`UnsupportedParamsError: bedrock does not support parameters: ['store']`). Since the
  default model is a Claude model, using the Anthropic Messages API (`/v1/messages`, the
  same endpoint Claude Code uses) sidesteps that entirely. To serve a non-Anthropic model
  through Pi, add a second provider with `api: "openai-completions"` and
  `baseUrl: .../v1` — but be aware of the `store`-param limitation on this proxy.

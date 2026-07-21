# Claude Code → LiteLLM (Opus 4.8)

[Claude Code](https://docs.claude.com/en/docs/claude-code) is Anthropic's official
terminal coding agent. This env runs it against the OSU LiteLLM proxy's
**Anthropic Messages** endpoint, defaulting to **Opus 4.8** (`claude-opus-4-8`).

## Prerequisites

- Nix with flakes enabled (`experimental-features = nix-command flakes`).
- A `.env` at the repo root (`../../.env`) containing:
  ```
  LITELLM_URL=https://litellm.cloud.osu.edu/
  LITELLM_KEY=sk-...
  ```
  (Already present in this repo; it is git-ignored.)

## Quickstart

```bash
cd envs/claude-code
nix develop          # installs claude-code, loads .env, exports ANTHROPIC_* vars
claude               # starts the TUI, talking to LiteLLM as Opus 4.8
```

One-shot / headless:

```bash
nix develop -c claude -p "Summarize this repo and how to run its checks."
```

## What the flake sets

The `devShell` puts `claude-code` on `PATH` and exports:

| Variable | Value | Purpose |
|---|---|---|
| `ANTHROPIC_BASE_URL` | `https://litellm.cloud.osu.edu` | Claude Code POSTs to `$BASE/v1/messages` |
| `ANTHROPIC_AUTH_TOKEN` | `$LITELLM_KEY` | sent as `Authorization: Bearer …` |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | main model |
| `ANTHROPIC_SMALL_FAST_MODEL` | `claude-haiku-4-5-20251001` | background/haiku tasks |

`ANTHROPIC_API_KEY` is unset in the shell so an ambient key can't shadow the token.

## Switching models

Override for a single session:

```bash
ANTHROPIC_MODEL=claude-opus-4-7 nix develop -c claude
```

or from inside the TUI with `/model claude-sonnet-5`. Any model id returned by
`curl -s $LITELLM_URL/v1/models -H "Authorization: Bearer $LITELLM_KEY"` works.

## Using a different proxy, or going direct to the vendor

The endpoint **defaults to OSU's LiteLLM proxy** (`ANTHROPIC_BASE_URL` ← `LITELLM_URL`
from the repo-root `.env`). To use a *different* Anthropic-compatible proxy, just point
`LITELLM_URL`/`LITELLM_KEY` at it. To skip the proxy and talk **directly to Anthropic**,
edit the flake's `shellHook`: leave `ANTHROPIC_BASE_URL` unset (Claude Code then defaults
to `https://api.anthropic.com`) and export `ANTHROPIC_API_KEY=<your Anthropic key>` in
place of `ANTHROPIC_AUTH_TOKEN`.

## Troubleshooting

- **401 / auth errors** — check `LITELLM_KEY` is in the repo-root `.env`; the shell
  prints a warning if it's missing.
- **Model not found** — confirm the id is in the LiteLLM model list (see above).
- Claude Code caches an onboarding login; if it tries to log in to console.anthropic.com,
  make sure you launched via `nix develop` so `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`
  are set.

# Coding-agent environments

Each subdirectory is a self-contained Nix flake that installs one terminal coding
agent and wires it to the **OSU LiteLLM proxy** (`https://litellm.cloud.osu.edu`).
`nix develop` in any of them drops you into a shell with the agent on `PATH`,
configured and ready.

| Agent | Dir | Model | LiteLLM API | In nixpkgs? |
|---|---|---|---|---|
| [Claude Code](https://docs.claude.com/en/docs/claude-code) | [`claude-code/`](./claude-code) | `claude-opus-4-8` | Anthropic Messages (`/v1/messages`) | yes |
| [Codex CLI](https://github.com/openai/codex) | [`codex-cli/`](./codex-cli) | `gpt-5.5-2026-04-24` | Responses (`/v1/responses`) | yes |
| [Pi](https://pi.dev) | [`pi/`](./pi) | `claude-opus-4-8` | Anthropic Messages (`/v1/messages`) | no (npm) |
| [OpenCode](https://opencode.ai) | [`opencode/`](./opencode) | `claude-opus-4-8` | Chat Completions (`/v1/chat/completions`) | yes |

Opus 4.8 is the default everywhere except Codex CLI, which uses GPT-5.5.

## Credentials

All four read `LITELLM_URL` and `LITELLM_KEY` from the repo-root `.env`
(git-ignored). Each flake's `shellHook` sources it and maps it to whatever the agent
expects — no key is ever written into a flake or a tracked config file.

```
# ../.env
LITELLM_URL=https://litellm.cloud.osu.edu/
LITELLM_KEY=sk-...
```

## Using a different proxy, or going direct to the vendor

Everything here **defaults to OSU's LiteLLM proxy** — that's the workshop setup, but
nothing is locked to it. Two ways to change the target:

- **A different proxy** (any LiteLLM / OpenAI- / Anthropic-compatible gateway): point
  `LITELLM_URL` / `LITELLM_KEY` in the repo-root `.env` at it. The Claude Code and OpenCode
  flakes read `LITELLM_URL` directly; Codex (`config.toml`) and Pi (`models.json`) hard-code
  the host, so edit `base_url` / `baseUrl` there too.
- **Direct to the vendor** (no proxy): edit that agent's config to the vendor's own endpoint
  and key — `https://api.anthropic.com` + an Anthropic key for the Claude-model agents
  (Claude Code, Pi), `https://api.openai.com/v1` + an OpenAI key for Codex.

See each dir's README ("Using a different proxy…") for the exact field to change.

## Usage

```bash
cd envs/<agent>
nix develop            # enter the configured shell
```

Then start the agent (`claude`, `codex`, `pi`, or `opencode`) — or run headless, e.g.
`nix develop -c claude -p "..."`. See each dir's README for specifics.

## How config is kept isolated

To avoid clobbering any global agent config you already have, each env keeps its
state project-local:

- **Claude Code** — pure env vars (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`).
- **Codex CLI** — `CODEX_HOME` → `codex-cli/.codex`, seeded from `codex-cli/config.toml`.
- **Pi** — `PI_CODING_AGENT_DIR` → `pi/.pi-agent`, seeded from `pi/models.json` + `pi/settings.json`.
- **OpenCode** — `OPENCODE_CONFIG` → `opencode/opencode.json`.

The tracked config templates (`config.toml`, `models.json`, `settings.json`,
`opencode.json`) are the source of truth; runtime dirs (`.codex/`, `.pi-agent/`,
`.npm/`) are git-ignored.

## Notes

- First launch of **Pi** and **OpenCode** needs network (npm / AI-SDK package fetch).
- Discover available models: `curl -s $LITELLM_URL/v1/models -H "Authorization: Bearer $LITELLM_KEY"`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`mnms` is a **workshop reference implementation** for a conference workshop on agentic
tooling for academic researchers. It builds *openly-licensed, redistributable* stand-ins
for a set of private research-consulting projects (OSU Arts & Sciences / ASC-ETS). Read
`WORKSHOP_ARCHETYPES.md` first — it is the design document that motivates every branch,
mapping each workspace member to the real ("archetype") project it demonstrates and to the
"what did agents actually unlock" argument the workshop makes.

The recurring shape of every branch is **corpus in → structured data out**, with
multi-model comparison and cost accounting, driven through the OSU **LiteLLM proxy**.

## Layout: a uv workspace of independent branches

This is a `uv` workspace (root `pyproject.toml`, `[tool.uv.workspace]`). Each member is a
self-contained "branch" demonstrating one archetype. Current state matters:

| Member | Archetype (private original) | Status |
|---|---|---|
| `data_acquisition/` | — (supplies data for all others) | **working** — polite fetchers for openly-licensed corpora |
| `transcription/` | ASR / `opi-transcript` | **working** — Granite-Speech on Common Voice, WER scoring |
| `corpus_coding/` | `article_coding` — multi-model codebook coding + agreement | **stub** (`main.py` prints TODO) |
| `structure_analysis/` | `syllabi` — LLM extraction → BERTopic clustering | **stub** |
| `bt_scoring/` | `podcasts` — pairwise LLM-judge → Bradley-Terry scale | **stub** |

The three stubs have detailed READMEs describing what to build and which
`data_acquisition` fetcher feeds them. `data_acquisition` is the shared data layer: each
stub branch expects you to run its fetcher first, producing a git-ignored
`data_acquisition/data/<source>/` tree plus a JSON-Lines manifest.

## Commands

Everything runs through `uv` from the repo root. `uv sync` installs the whole workspace.

Run a branch's script with `--package <member>`:

```bash
uv run --package transcription python transcription/main.py
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 20
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 500
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 500 --keyword "topic modeling"
uv run --package data_acquisition python data_acquisition/fetch_presidential_audio.py --limit 20
```

There is no test suite or linter configured yet. Python is pinned to 3.13.

## Credentials and .env

A single git-ignored `.env` at the repo root holds every secret; each branch loads it via
`python-dotenv` (`load_dotenv(REPO_ROOT / ".env")`). Expected keys:

- `HF_TOKEN` — Hugging Face token (transcription dataset streaming)
- `LITELLM_URL`, `LITELLM_KEY` — the OSU LiteLLM proxy (`https://litellm.cloud.osu.edu`), the
  single entry point for all LLM calls in the coding/clustering/judging branches
- `CONTACT_EMAIL` — contact address for the data-acquisition User-Agent

Never write a key into a tracked config file or a flake — everything reads from `.env`.

## Conventions to follow

- **Data acquisition is deliberately polite.** All fetchers go through
  `data_acquisition/common.py::PoliteSession`: identifying User-Agent, per-host rate limit
  (`--rate`), exponential backoff honoring `Retry-After`, atomic streaming downloads that
  **skip files already present** so re-runs resume. New fetchers should use it, expose a
  `--limit` (start small), and write a JSON-Lines manifest recording ids, licenses, and
  paths. Only fetch openly-licensed / public-domain data — the point is redistributability.
- **Resumability and per-item resilience** are the house style: cache/skip completed work,
  record per-document failures instead of aborting the batch.
- The LLM branches should produce **Pydantic-schema JSON** and account for cost, matching
  the archetypes in `WORKSHOP_ARCHETYPES.md`.

## envs/ — coding-agent environments (not application code)

`envs/` contains one Nix flake per terminal coding agent (Claude Code, Codex CLI, Pi,
OpenCode), each wiring the agent to the LiteLLM proxy via `LITELLM_URL`/`LITELLM_KEY` from
the root `.env`. `nix develop` in any subdir drops into a configured shell. These are for
*running agents against the proxy*, unrelated to the Python workspace. See `envs/README.md`.

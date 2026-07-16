# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Who's asking, and the first thing to do.** This repo is used both by its developer and
> by people following up after the workshop who are exploring it for the first time. Either
> way: **if anyone asks a question about this project — what it is, what a branch does, how
> to run it, how to read its output — immediately load the `mnms-guide` skill** (under
> `.claude/skills/`, invocable as `/mnms-guide`) before answering. Do not rely on it being
> triggered automatically; load it proactively at the first such question, then answer from
> it. It is the maintained tour of the repo and routes to the authoritative docs.

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
self-contained "branch" demonstrating one archetype. All five are now implemented and
produce results end-to-end:

| Member | Archetype (private original) | Status |
|---|---|---|
| `data_acquisition/` | — (supplies data for all others) | **working** — polite, resumable fetchers for openly-licensed corpora |
| `transcription/` | ASR / `opi-transcript` | **working** — Granite-Speech on Common Voice, WER scoring |
| `corpus_coding/` | `article_coding` — multi-model codebook coding + agreement | **working** — codes N articles × several models × repeated runs → inter-model agreement (Cohen κ / Krippendorff α / Fleiss κ) + intra-model consistency + cost; CSVs, HTML report, Plotly `viz/` |
| `structure_analysis/` | `syllabi` — LLM extraction → BERTopic clustering | **working** — extract → local MiniLM embeddings → BERTopic (UMAP+HDBSCAN) → validate vs. ground-truth label (ARI/NMI/homogeneity); CSVs + Plotly `viz/` |
| `bt_scoring/` | `podcasts` — pairwise LLM-judge → Bradley-Terry scale | **working** — Granite ASR → pairwise multi-judge → Bradley-Terry scale w/ SEs + judge rank-correlation; CSVs + Plotly `viz/` |

Each branch's README describes its archetype and which `data_acquisition` fetcher feeds it.
`data_acquisition` is the shared data layer: **run its fetcher first**, producing a
git-ignored `data_acquisition/data/<source>/` tree plus a JSON-Lines manifest, before
running the branch that consumes it.

For a guided tour of the repo — what each branch is, how to run it, how to read its
outputs, and how skills like this are built — there is a `mnms-guide` skill under
`.claude/skills/` (and `.agent/skills/` for other agents), invocable as `/mnms-guide`.

## Commands

Everything runs through `uv` from the repo root. Use `uv sync --all-packages` to install
the whole workspace into one shared `.venv` — a bare `uv sync` only syncs the root member
and leaves the branches' dependencies out.

Acquire data first, then run the branch that consumes it. Run a branch's script with
`--package <member>`:

```bash
# data acquisition (start small with --limit; --seed makes sampling reproducible)
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 30 --seed 7
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 800 --seed 7
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 500 --keyword "topic modeling"
uv run --package data_acquisition python data_acquisition/fetch_presidential_audio.py --limit 60 --seed 7

# branches (each reads the corresponding data/<source>/ manifest)
uv run --package corpus_coding python corpus_coding/main.py --n-articles 10 --model-group default
uv run --package structure_analysis python structure_analysis/main.py --source arxiv --limit 600
uv run --package bt_scoring python bt_scoring/main.py --num-items 0 --min-matchups 8
uv run --package transcription python transcription/main.py
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
- **Reliable, clean re-runs are a first-class requirement** — attendees run these branches
  themselves, repeatedly, sometimes without a GPU. Prefer determinism and portability over
  speed (e.g. `structure_analysis` pins MiniLM to CPU and numba to a single fork-safe
  thread; fetchers take `--seed`), and ensure each branch **exits cleanly with no lingering
  processes**. When touching cost/ledger, clustering, or exit paths, verify a clean exit —
  a past "exit hang" was really a self-deadlock in `CostLedger.summary_lines()`, and
  `bt_scoring` ends with a deliberate, documented `os._exit(0)` to dodge an ASR-stack
  shutdown hang.

## envs/ — coding-agent environments (not application code)

`envs/` contains one Nix flake per terminal coding agent (Claude Code, Codex CLI, Pi,
OpenCode), each wiring the agent to the LiteLLM proxy via `LITELLM_URL`/`LITELLM_KEY` from
the root `.env`. `nix develop` in any subdir drops into a configured shell. These are for
*running agents against the proxy*, unrelated to the Python workspace. See `envs/README.md`.

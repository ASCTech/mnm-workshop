# mnms

Reference implementation for a conference workshop on **agentic tooling for
academic researchers** (OSU Arts & Sciences / ASC-ETS). It builds *openly-licensed,
redistributable* stand-ins for a set of private research-consulting projects, so the
patterns can be shown and re-run by anyone.

Every branch has the same shape: **corpus in → structured data out**, with
multi-model comparison and cost accounting, driven through the OSU **LiteLLM proxy**.

> **New here?** Read [`WORKSHOP_ARCHETYPES.md`](./WORKSHOP_ARCHETYPES.md) for the
> design argument — which real project each branch stands in for and the "what did
> agents actually unlock" case the workshop makes. If you're working through this
> with a coding agent, point it at the **`mnms-guide`** skill (`/mnms-guide` in
> Claude Code) for a guided tour, or just ask it to explain or run a branch —
> the layout is a plain `uv` workspace and the code is commented.

## Layout — a `uv` workspace of independent branches

Each member is a self-contained "branch" demonstrating one archetype.

| Member | Archetype (private original) | What it does |
|---|---|---|
| `data_acquisition/` | — (feeds all others) | Polite fetchers for openly-licensed corpora → a git-ignored `data/<source>/` tree + JSON-Lines manifest |
| `transcription/` | ASR / `opi-transcript` | IBM Granite-Speech on Common Voice, WER scoring |
| `corpus_coding/` | `article_coding` | Multi-model codebook coding → inter-model agreement + intra-model consistency + cost |
| `structure_analysis/` | `syllabi` | LLM field extraction → embeddings → BERTopic clustering → validation vs. ground-truth labels |
| `bt_scoring/` | `podcasts` | Pairwise LLM-as-judge → Bradley-Terry latent scale with standard errors |

## Quickstart

```bash
uv sync --all-packages          # install the whole workspace into one .venv
```

Create a git-ignored `.env` at the repo root with:

```
HF_TOKEN=...                              # Hugging Face (dataset streaming / ASR)
LITELLM_URL=https://litellm.cloud.osu.edu/
LITELLM_KEY=sk-...                        # OSU LiteLLM proxy — the single LLM entry point
CONTACT_EMAIL=you@example.edu             # identifies the data-acquisition User-Agent
```

Fetch data first (start small with `--limit`), then run a branch:

```bash
# 1. acquire a corpus (writes data_acquisition/data/<source>/ + manifest.jsonl)
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 30

# 2. run the branch that consumes it
uv run --package corpus_coding python corpus_coding/main.py --n-articles 10
```

See each branch's `README.md` for its specific fetcher and options, and
`.claude/skills/mnms-guide/references/running-branches.md` for the full run guide.

## Conventions

- **Only openly-licensed / public-domain data** — the point is redistributability.
- Fetchers are **polite and resumable** (identifying User-Agent, rate limits,
  backoff, skip-already-downloaded); LLM branches emit **Pydantic-schema JSON** and
  **account for cost**; per-item failures are recorded, never fatal.
- Branches must **run reliably and exit cleanly** on re-runs — attendees run these
  themselves, repeatedly. See the reliability notes in the `mnms-guide` skill.

Python is pinned to 3.13. `CLAUDE.md` holds the working guidance for coding agents.

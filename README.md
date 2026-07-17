# mnms

A set of runnable **case studies** for a conference workshop on *agentic tooling for
academic researchers*, put together by the research-consulting team in OSU Arts &
Sciences (ASC-ETS). Each branch is an openly-licensed, redistributable rebuild of a
real project we did with a researcher — close enough to show how the work actually
went, but built entirely on public data so anyone can clone it and follow along.

We don't think the most useful thing we can share is any one tool. We've worked with
a lot of researchers across a lot of fields, and the tools varied enormously — every
project's goals and constraints were different. What carried over was the *process*.
These branches are our attempt to show that process rather than just assert it:
motivating examples you can run, take apart, and adapt to your own corpus.

One thing you'll notice across all of them is a recurring shape — **corpus in →
structured data out**, usually with a few models compared and their cost tracked,
routed through the OSU **LiteLLM proxy**. That shape isn't a rule we're prescribing;
it's just what kept showing up, and it turned out to be a useful thing to build
around.

> **New here?** If you're working through this with a coding agent, point it at the
> **`mnms-guide`** skill (`/mnms-guide` in Claude Code) — a guided tour of what each
> branch is, how to run it, and how to read what comes out. Or just ask the agent to
> explain or run a branch: the layout is a plain `uv` workspace and the code is
> commented.

## Layout — a `uv` workspace of independent branches

Each member is a self-contained branch built around one case study.

| Member | Modeled on a project we did | What it does |
|---|---|---|
| `data_acquisition/` | — (feeds all the others) | Polite fetchers for openly-licensed corpora → a git-ignored `data/<source>/` tree + JSON-Lines manifest |
| `transcription/` | self-hosted ASR work | IBM Granite-Speech on Common Voice, WER scoring |
| `corpus_coding/` | multi-model article coding | Multi-model codebook coding → inter-model agreement + intra-model consistency + cost |
| `structure_analysis/` | a syllabus-clustering study | LLM field extraction → embeddings → BERTopic clustering → validation vs. ground-truth labels |
| `bt_scoring/` | a pairwise-judging study | Pairwise LLM-as-judge → Bradley-Terry latent scale with standard errors |

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

## A few things we've settled into

These aren't commandments — they're habits that made the projects behind these
branches easier to trust and hand off, so we've kept them here too:

- **Only openly-licensed / public-domain data.** The whole point of these stand-ins
  is that we can redistribute them; the private originals we couldn't.
- **Polite, resumable fetchers** (identifying User-Agent, rate limits, backoff,
  skip-already-downloaded) — because attendees re-run these on real endpoints, and a
  fetcher that resumes instead of re-hammering is just good manners.
- **Structured, cost-accounted LLM output** (Pydantic-schema JSON, a cost table).
  Recording what each model cost is what lets you defend a model choice later.
- **Per-item failures are recorded, not fatal** — one bad document shouldn't sink a
  batch.
- **Runs reliably and exits cleanly.** Attendees run these themselves, repeatedly,
  sometimes without a GPU, so we lean toward determinism and portability. There are a
  couple of hard-won reliability notes in the `mnms-guide` skill.

Python is pinned to 3.13. `CLAUDE.md` holds working guidance for coding agents.

# mnms

## Mind and Machine Alignment Summit Workshop - AI in Research Case Studies


<a href="https://github.com/ASCTech/mnm-workshop"><img src="repo-qr.svg" alt="QR code linking to this repository" align="right" width="160" height="160"></a>


This is the companion repository for part of The Ohio State University's Emerging Technology
Studio's workshop for the Mind and Machine Alignment Summit. Reproduced here are several 
archetypal case studies of real work we've done with researchers, faculty, and staff. 
Each of tbe case studies uses data that is permissively licensed or public domain
so anyone is free to share, remix, and adapt as they'd like. Likewise, the repository
is Apache 2.0 licensed so you can do so similarly with the code.

Over the past three or four years the field of AI has moved so fast the primary constant
has been change, workflows and techniques changing in periods of months or weeks. So
rather than a snapshot of how we do things at any particular time this repository aims to
communicate more the processes on top of existing capabilities so hopefully future changes
layer on top. These examples and the stories behind them aren't meant to be prescriptive,
rather some of what has worked for us over the course of dozens of projects of varying goals,
scopes, and techniques.

Many, but not all, of the engagements we've worked in have had a similar form:
structured, semi-structured, or unstructured data/corpus in, structured analysis out.
Many had substantial overlap in requirements and approaches, and that overlap is what's 
meant to be represented here. The examples will cover both specific, local strategies
in addition to the process and strategy that went into them.

---

## Agent Quickstart

If you've cloned this repository and use an agent, eg Claude Code, Codex, OpenCode, you can 
likely make use of a 'skill' that's in this repository that should help your agent help you
explore the project. Agents differ and things change, but ideally the agent loads it automatically,
otherwise you can (likely) tell it to load the `mnms-guide` skill or prefix your prompt with `/mnms-guide`.

## Technical Quickstart

For running the code, you'll want to install the tool `uv` from https://astral.sh/ .
If you're not familiar, it's a tool that makes it more convenient to manage Python projects.

```bash
uv sync --all-packages          # install the whole workspace into one .venv
```

Create a `.env` at the repo root with:

```
HF_TOKEN=...                              # Hugging Face (dataset streaming / ASR)
LITELLM_URL=https://litellm.cloud.osu.edu/
LITELLM_KEY=sk-...                        # OSU LiteLLM proxy — the single LLM entry point
CONTACT_EMAIL=you@example.edu             # identifies the data-acquisition User-Agent
```

Fetch data first (start small with `--limit`), then run a branch:

```bash
# 0. acquire a corpus (writes data_acquisition/data/<source>/ + manifest.jsonl)
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 29

# 1. run the branch that consumes it
uv run --package corpus_coding python corpus_coding/main.py --n-articles 9
```

See each branch's `README.md` for its specific fetcher and options, and
`.claude/skills/mnms-guide/references/running-branches.md` for the full run guide.


## Layout — a `uv` workspace of independent branches

Each member is a self-contained branch built around one case study.

| Member | Modeled on a project we did                  | What it does |
|---|----------------------------------------------|---|
| `data_acquisition/` | generic tooling                              | Polite fetchers for openly-licensed corpora → a git-ignored `data/<source>/` tree + JSON-Lines manifest |
| `transcription/` | self-hosted ASR work                         | IBM Granite-Speech on Common Voice, WER scoring |
| `corpus_coding/` | multi-model article coding for meta analysis | Multi-model codebook coding → inter-model agreement + intra-model consistency + cost |
| `structure_analysis/` | a syllabus clustering study                  | LLM field extraction → embeddings → BERTopic clustering → validation vs. ground-truth labels |
| `bt_scoring/` | a ranking of political viewpoint in podcasts | Pairwise LLM-as-judge → Bradley-Terry latent scale with standard errors |


## Common patterns

A list of things, in no particular order, we've noticed or used that appears in the projects

- Use the right/best tool for the job - Not everything needs full LLMs or even AI methods at all. Expensive frontier models aren't necessarily always better at all tasks. Starting small, testing, and comparing results across models goes a long way.
- Cost accounting from the beginning - Total cost is tracked by API keys, but this is for planning. Knowing a certain subset or model is cheap allows more freedom to experiment and iterate. Being able to give a max cost with confidence can be very helpful when organizing funds and approvals.
- Common infrastructure - Being lazy is smart. A centralized proxy for serving LLMs makes it trivial to swap/update models while keeping access control easy. Having a common pattern or template helps avoid mistakes while saving time.
- Legibility - Making it clear and easy to tell what success or failure looks like saves a lot of time in the long run. Using AI tools can increase the distance and abstraction level, so being able to make confident observations is critical.


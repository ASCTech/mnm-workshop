# corpus_coding

**The case study:** multi-model corpus coding. This branch is modeled on a project
where we coded 1,099 PDFs through a 10-question codebook across ~5 model-runs (5,120
results), then scored **inter-model agreement** and **intra-model consistency**, with
a cost estimate produced before anything was spent. Ultimately this fed into a meta analysis paper.

Many Research/Grad/Teaching Assistants have been asked to code many a data point,
a process that is expensive in time, their pay, and the person's sanity. Recently,
LLMs have acquired enormous context lengths, namely ~1M tokens, and have reduced
(still non zero) confabulation rates. Applied to scoring, when careful, can result
in being able to do in an afternoon and $50 what previously took months. The results
are even, in some senses, more robust; the LLMs will answer similarly whether it's the 
first or 1,000th paper and will 'forget', in that repeated runs are independent. 
A curious result from the original case: when comparing LLM coded results against 
human scored 'ground truth', the cheapest model most closely matched the human results.
Was the cheapest model actually better, or were the more expensive models better than humans?

**Data:** the openly-licensed stand-in for paywalled article PDFs —
`../data_acquisition/fetch_pmc_oa.py`, which pulls the **PMC Open Access subset**
restricted to **CC0 / CC BY** full-text articles (`.txt` + `.xml` JATS). Run it
first:

```bash
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 30 --seed 7
# -> data_acquisition/data/pmc_oa/{manifest.jsonl, articles/PMC*/...}
```

**What this branch does:**
1. Define the codebook/rubric - just as you would with a human graded, define what it is you're looking for.
2. Run the batches - script up your codebook against your dataset and set it off, logging throughout.
3. Analyze result quality - Ideally, you'd have a human coded sample set since that's often the point of comparison. Otherwise, or in addition, intra and inter model agreement can give some information. 

## Quickstart

Needs `LITELLM_URL` and `LITELLM_KEY` in the repo-root `.env` (the OSU LiteLLM proxy; see
the root README). The fetcher also uses `CONTACT_EMAIL` for its polite User-Agent. Install
the workspace once with `uv sync --all-packages`.

```bash
# 1. Fetch the corpus (run once; resumable, skips files already present):
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 30 --seed 7

# 2. Code the corpus (10 articles x 6 models x 2 runs by default):
uv run --package corpus_coding python corpus_coding/main.py --n-articles 10 --model-group default
# -> corpus_coding/output/{results.jsonl, agreement.csv, consistency.csv,
#                          cost_by_model.csv, report.html, viz/index.html}
```

Re-runs are cheap and resumable: completed (article, model, run) records in
`results.jsonl` are skipped, so raising `--n-articles` only codes the new articles. Handy
knobs: `--model-group {default,claude-tiers,gpt-tiers,gemini-tiers,all}` (same-family
capability-tier comparisons), `--n-runs` (repeats per pair; default 2), `--n-articles`.
Open `corpus_coding/output/viz/index.html` in a browser for the interactive dashboard.

## Reading the output

`output/report.html` is the formatted view of three CSVs:

- **`agreement.csv`** — one row per codebook question, *across* models. `pct_agreement` is
  raw agreement; **Cohen's κ** (chance-corrected, pairwise), **Fleiss' κ** (chance-corrected,
  3+ raters), and **Krippendorff's α** (handles any number of raters + missing data) all
  correct for how often raters would agree *by chance*. Near **1.0** = models code the
  question identically; near **0** = they diverge.
- **`consistency.csv`** — *intra*-model: does one model give the same answer across its own
  repeated runs? (Inter-model asks whether *different* models agree; intra-model asks whether
  *one* model repeats itself.)
- **`cost_by_model.csv`** — calls, tokens, and USD per model, the number you estimate before
  spending and defend afterward.

### Notes / things that look wrong but aren't

- **High agreement, κ = 0 (the "kappa paradox").** `open_data_statement` shows **96.7%
  agreement but κ = 0.0**. Not a bug: when nearly every article gets the same label there's
  almost no variance for chance-correction to credit, so κ collapses even though raw
  agreement is high. Always read `pct_agreement` and κ *together*; never quote κ alone on a
  near-constant field.
- **Blank κ (shown as `-`).** `funding_disclosed` and `study_registered` hit **100%
  agreement** — zero variance means κ is mathematically undefined, so the cells are empty.
- **Low κ on a subjective question is a finding, not a failure.** `primary_field` (~28%
  agreement, κ ≈ 0.25) says the *question itself* is subjective — useful information about
  your instrument, not a broken model.
- **With `--n-runs 2`, per-question consistency is binary** (a pair of runs either matches or
  it doesn't → 50% or 100%). Use `--n-runs 3`+ for a more granular consistency measure.
- **The cost spread is the punchline.** In the shipped run the *identical* 20 calls cost
  **~$0.51 on `claude-opus-4-8` vs ~$0.018 on `gemini-3.1-flash-lite`** — a ~28× spread.
  The workshop argument: pick the cheapest model whose agreement/consistency is good enough,
  and this table is how you defend that choice.
- **Model IDs (`claude-opus-4-8`, `gemini-3.1-flash-lite`, …) are OSU LiteLLM proxy aliases**;
  the models available in your deployment may differ.

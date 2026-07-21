# bt_scoring

**The case study:** LLM-as-judge → statistics. This branch is modeled on a project
that scored items pairwise for political orientation (each item ≥50 matchups) → a
**Bradley-Terry** left–right scale with standard errors, via an async worker pool with
checkpointing and retry.

This is the one that most changed how we think about what agents make possible: the study
is *effectively impossible by hand* — 10k+ pairwise comparisons, and no rater can hold dozens
of long texts in working memory at once, or stay unbiased across all of them. It only works
because the judge is **cheap and consistent** — reliability is the unlock, not raw capability.

**Data:** the openly-licensed stand-in — `../data_acquisition/fetch_sotu.py`, which
scrapes **State of the Union addresses as plain text** from The American Presidency
Project (presidency.ucsb.edu) for 1950–2020. SOTU addresses are US-government works,
so they're public domain and freely redistributable. Each is labelled by **president**
and **party** — the party label is the ground truth the judge never sees, used only to check
how good or bad the rankings are. Being text, they feed the pairwise judge directly, with no
lossy ASR step (this branch used to run on presidential *audio* + Granite ASR, but on
commodity hardware the transcripts were too poor to judge directly — hence the pivot to text).

```bash
uv run --package data_acquisition python data_acquisition/fetch_sotu.py            # all 1950–2020
uv run --package data_acquisition python data_acquisition/fetch_sotu.py --limit 6  # quick sample
# -> data_acquisition/data/sotu/{manifest.jsonl, texts/<president>_<year>.txt}
```

**What this branch does:**

1. Judge addresses **pairwise** on a latent dimension — default **economic orientation:
   left ↔ right** — with the IDENTICAL pair set judged by every model in `--judges` via the
   LiteLLM proxy, `--min-matchups` per address, checkpointed to `comparisons.jsonl` with retry.
2. Fit a **Bradley-Terry** model per judge → a scale with standard errors (from the Hessian,
   or a ridge + bootstrap fallback under quasi-complete separation), cross-checked against a
   `choix` ILSR point estimate.
3. **Validate** the recovered scale against the **party** label the judge never saw
   (point-biserial + Mann-Whitney: do Republican addresses land to the "right"?), plus a
   secondary year-trend check.
4. **Compare judges:** a Spearman rank-correlation matrix of their BT scales, plus Plotly
   figures (forest, scatter, heatmap, summary) and an HTML index.


# The Latent Axis 

The default prompt asks the model to compare on 'left' vs 'right', explicitly without
a rubric. This is very underdefined, and yet we see coherent results. Models are very
much a product of their weights and training and bring their own 'world view' based on
them. This can in part be mitigated by objective metrics, but never certainly reduced to
zero, and care must be taken to distinguish measuring reality and measuring the model.
In this demo, the models have varying levels of agreement, different orderings, and 
from that likely filled the definitional gap with their own views. 


## Quickstart

Needs `LITELLM_URL` and `LITELLM_KEY` in the repo-root `.env` (all judging goes through the
OSU LiteLLM proxy); the fetcher uses `CONTACT_EMAIL`. Install once with `uv sync
--all-packages`, and fetch the SOTU texts (above) first.

```bash
# cheap smoke (one judge, few items) -- do this first:
uv run --package bt_scoring python bt_scoring/main.py --num-items 8 --min-matchups 3 --judges gemini-3.1-flash-lite

# full run: all 5 default judges, >=6 matchups per address
uv run --package bt_scoring python bt_scoring/main.py --min-matchups 6
# -> bt_scoring/output/{scores_all_judges.csv, scores_by_judge/<judge>.csv,
#                       judge_rank_correlation.csv, viz/index.html}

# refit + re-visualize from cached comparisons WITHOUT re-judging (free):
uv run --package bt_scoring python bt_scoring/main.py --skip-judge
# knobs: --char-budget (0 = full speech text), --judges, --dimension, --num-items
```

**Cost:** the full default run (5 judges × 63 addresses, ≥6 matchups) cost **~$30** in our
run — most of it `claude-opus-4-8` (~$23); `gemini-3.1-flash-lite` alone was ~$0.35. Run the
smoke test first, then use `--skip-judge` to iterate on the stats and plots for free.

## Reading the output

- **`scores_by_judge/<judge>.csv`** — one judge's Bradley-Terry scale. `strength` = latent
  position (higher = further toward the second pole, here economically "right"); `se` = its
  standard error; `reference_item` is the address pinned to strength 0 (a BT scale is only
  defined up to a shift, so one item anchors the zero); `choix_strength` is an independent
  ILSR cross-check.
- **`judge_rank_correlation.csv`** — judge × judge Spearman correlation of their scales: how
  much independent judges reproduce each other's ordering. In our run four judges (both
  Geminis, Opus, Haiku) cluster at **ρ ≈ 0.80–0.92**, while **`gpt-5.4-mini` sits apart at
  ρ ≈ 0.58–0.60** with everyone.
- **Validation vs. `party`** (the headline, printed to the console + summary) is a
  point-biserial correlation of BT strength against party coded right = 1. Every judge came
  out significant (r ≈ 0.44–0.78, p < 0.001) — the emergent scale really does separate
  Republican from Democratic addresses, without the judge ever being told the party.

Stats in one breath: **Bradley-Terry** turns many noisy "A beats B" pairwise judgments into a
single interval scale of latent strengths (it's logistic regression on the win/loss pairs);
**Spearman's ρ** asks whether two rankings agree in *order* (1 = identical); a **point-biserial**
correlation relates a continuous score to a binary label.

### Notes / things that look wrong but aren't

- **`regularized = True` on every row** is expected at this size: with few items some address
  wins or loses nearly all its matchups (quasi-complete separation), so the fit falls back to
  ridge + bootstrap SEs rather than the Hessian. Read those `strength`s as shrunk toward 0 and
  lean on `se`.
- **The `gpt-5.4-mini` outlier is a feature, not a bug** — a scale only means something if
  independent judges reproduce it, and the disagreement is exactly the "what does *this* model
  think 'right' means?" discussion the branch is built to provoke.
- **`output/` is git-ignored** — a fresh clone has no pre-generated results; run the branch (or
  `--skip-judge` on cached comparisons) to produce them.

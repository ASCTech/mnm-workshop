# bt_scoring

**The case study:** LLM-as-judge → statistics. This branch is modeled on a project
that scored items pairwise for political orientation (each podcast ≥50 matchups) → a
**Bradley-Terry** left–right scale with standard errors, via an async worker pool with
checkpointing and retry. This is the one that most changed how we think about what
agents make possible: the study is *effectively impossible by hand* (10k+ pairwise
comparisons; no one can hold hundreds of shows in mind at once). It only works because
the judge is **cheap and consistent** — reliability is the unlock, not raw capability.

**Data:** the openly-licensed stand-in — `../data_acquisition/fetch_sotu.py`, which
scrapes **State of the Union addresses as plain text** from The American Presidency
Project (presidency.ucsb.edu) for 1950–2020. SOTU addresses are US-government works,
so they're public domain and freely redistributable. Each is labelled by **president**
and **party** — the party label is the ground truth the judge never sees. Being text,
they feed the pairwise judge directly, with no lossy ASR step (this branch used to run
on presidential *audio* + Granite ASR, but on commodity hardware the transcripts were
too noisy to carry the judgment — hence the pivot to text).

```bash
uv run --package data_acquisition python data_acquisition/fetch_sotu.py            # all 1950–2020
uv run --package data_acquisition python data_acquisition/fetch_sotu.py --limit 6  # quick sample
# -> data_acquisition/data/sotu/{manifest.jsonl, texts/<president>_<year>.txt}
```

**What this branch does:**

1. Judge addresses **pairwise** on a latent dimension — default **economic
   orientation: left ↔ right** — with the IDENTICAL pair set judged by every model
   in `--judges` via the LiteLLM proxy (`../envs/`), `--min-matchups` each,
   checkpointed to `comparisons.jsonl` with retry.
2. Fit a **Bradley-Terry** model per judge → a scale with standard errors (Hessian,
   or a ridge + bootstrap fallback under quasi-complete separation), cross-checked
   against a `choix` ILSR point estimate.
3. **Validate** the recovered scale against the **party** label the judge never saw
   (point-biserial + Mann-Whitney: do Republican addresses land to the "right"?),
   plus a secondary year trend.
4. Compare judges: a Spearman rank-correlation matrix of their BT scales, and Plotly
   figures (forest, scatter, heatmap, summary) + an HTML index.

## The point: an ambiguous axis, on purpose

The judge is given **no rubric** for "economic left/right" — just the two poles. How
each model resolves that contested, training-data-laden axis, and how much the judges
then *disagree*, is what the branch surfaces. The party label turns "did it measure
anything real?" into a checkable question without ever defining the axis for the model.

```bash
# cheap smoke (one judge, few items); then the full run
uv run --package bt_scoring python bt_scoring/main.py --num-items 8 --min-matchups 3 --judges gemini-3.1-flash-lite
uv run --package bt_scoring python bt_scoring/main.py --min-matchups 6
# knobs: --char-budget (0 = full speech text), --judges, --dimension, --num-items
```

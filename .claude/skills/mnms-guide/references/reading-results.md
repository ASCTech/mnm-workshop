# Reading the results

This is the most useful thing the skill does: help a researcher *interpret* what a
branch produced, and know when to trust it. Each branch writes CSV/JSON to its
`output/` plus an interactive `viz/index.html` (open it in a browser). The numbers
below are from example runs to show you what "good" and "suspicious" look like —
they regenerate when you re-run, so read them as calibration, not constants.

---

## corpus_coding — agreement, consistency, cost

**`agreement.csv`** — one row per codebook question, across models. Columns:
`pct_agreement`, `cohen_kappa`, `krippendorff_alpha`, `fleiss_kappa`.

- κ near **1.0** = models code the question the same way (e.g. `study_registered`,
  `has_animal_subjects`). Easy, objective questions.
- κ near **0.2** on an open-ended question (e.g. `primary_field`) = models genuinely
  diverge. Not necessarily a bug — it says the *question* is subjective, which is
  itself a finding about your instrument.
- **The one that trips people up:** a question can show **98.9% agreement but κ = 0.0**
  (`open_data_statement` did). That's the **kappa paradox** — when almost every
  article gets the same label, there's no variance for chance-corrected agreement to
  credit, so κ collapses even though raw agreement is high. Read `pct_agreement` and
  κ *together*; never quote κ alone on a near-constant field.

**`consistency.csv`** — intra-model: does one model repeat itself across its own
runs? Low consistency with high inter-model agreement is a red flag that a model is
unstable even when the crowd converges.

**`cost_by_model.csv`** — the cost story in one table. In one run the *identical* 60
calls cost **~$1.50 on `claude-opus-4-8` vs ~$0.056 on `gemini-3.1-flash-lite`** — a
~27× spread. The workshop point: pick the cheapest model whose agreement/consistency
is good enough for the question, and this table is how you defend that choice.

---

## structure_analysis — cluster quality vs. ground truth

**`validation_metrics.json`** — how well the recovered topics line up with the
held-out label (arXiv category / NSF program):

- `normalized_mutual_info` (NMI) and `v_measure` around **0.50** = clusters capture
  about half the category structure — solidly meaningful for unsupervised topics on
  short text. `homogeneity` (are clusters pure?) vs `completeness` (is each category
  in one cluster?) tells you *which way* it's imperfect.
- `adjusted_rand_index` (~0.16) runs lower than NMI here and that's expected — ARI
  is stricter and punishes the large outlier group.
- `outlier_pct` (~40%) = HDBSCAN's noise label (`topic == -1`). Reported honestly,
  not hidden; a high outlier rate on short abstracts is normal, not a failure.

**`topic_dominant_label.csv`** — the payoff table: each topic's dominant true label
and what fraction of the topic it covers. Clean topics look like `astro-ph.GA` at
100%, `cond-mat.stat-mech` at 97%, `econ.EM` at 91% — i.e. the clustering rediscovered
real fields from extracted text alone. That is the "capability" claim made concrete.

`topics.csv` lists top words per topic; `viz/index.html` has the intertopic map and
hierarchy.

---

## bt_scoring — a latent scale and how much to trust it

The corpus is **State of the Union addresses (1950–2020)**, judged pairwise on
**economic orientation: left ↔ right** — deliberately with *no rubric*, so how each
model reads that contested axis (and how much the judges disagree) is part of the
result. Each address is labelled by `president` and `party`, which the judge never sees.

**`output/scores_by_judge/<judge>.csv`** — the Bradley-Terry scale from one judge:
`strength` (latent position), `se` (standard error), `n_matchups`, `rank`, plus
`president` / `party` / `year`.

- Higher `strength` = further toward the second pole (here, economically "right").
  In our run the top of every judge's scale was Reagan / the Bushes / Trump and the
  bottom was Democrats — a face-valid economic ordering.
- **`regularized = True`** means ridge/bootstrap kicked in because the comparison
  graph was near-separable (some item won/lost almost everything). Treat those
  `strength`s as shrunk toward 0 and lean on `se`.

**`judge_rank_correlation.csv`** — judge × judge Spearman correlation of their scales.
In our run four judges (the two Geminis, Opus, Haiku) clustered at **~0.80–0.92**,
while **GPT-mini sat apart at ~0.58–0.60 with everyone** — the reliability measure
made visible. A scale only means something if independent judges reproduce it; the
outlier is a feature to discuss ("what does *this* model think 'right' means?"), not
a bug to hide.

**Validation against `party`** is the headline (`strength_vs_party` in the console /
summary table): a point-biserial correlation of BT strength with party coded right=1.
Every judge came out **significant (r ≈ 0.44–0.78, all p < 0.001)** — the emergent
scale really does separate Republican from Democratic addresses, without the judge
ever being told the party. `year` is a secondary, non-causal trend check.

---

## How we've learned to read these

A few habits that have saved us from over-claiming:

- Look at **agreement/reliability before the headline numbers** — an unreliable
  instrument makes the rest meaningless.
- Report **outliers, nulls, and regularization** rather than hiding them; in this
  repo they're part of the honest telling, and they usually make the result *more*
  credible, not less.
- Tie any model choice back to the **cost table**. "Good enough for this question,
  and here's what it cost" is the argument we keep coming back to.

# Running the branches

`mnms` is a plain `uv` workspace, so you can also just ask the agent to run a
branch and it will work out the command. This file is the reference for doing it
deliberately.

## Once, up front

```bash
uv sync --all-packages     # installs every workspace member into one shared .venv
```

Create a git-ignored `.env` at the repo root (see the root `README.md` for the
keys: `HF_TOKEN`, `LITELLM_URL`, `LITELLM_KEY`, `CONTACT_EMAIL`). Every branch loads
it via `python-dotenv`. The LiteLLM proxy is the single entry point for all LLM
calls; discover available models with:

```bash
curl -s "$LITELLM_URL/v1/models" -H "Authorization: Bearer $LITELLM_KEY"
```

## The invariant: acquire data first

Each LLM branch reads a corpus that a `data_acquisition` fetcher produces. Run the
fetcher **before** the branch. Fetchers are polite and resumable — start with a
small `--limit`, and a re-run skips files already downloaded. `--seed` controls the
(stratified / balanced) sampling so runs are reproducible.

```bash
# corpus_coding  <- PMC Open Access (CC0 / CC BY) full-text articles
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 30 --seed 7

# structure_analysis  <- arXiv metadata (balanced across ~10 categories) or NSF awards
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 800 --seed 7

# bt_scoring + transcription  <- public-domain presidential audio (speaker-stratified)
uv run --package data_acquisition python data_acquisition/fetch_presidential_audio.py --limit 60 --seed 7
```

Each fetcher writes `data_acquisition/data/<source>/manifest.jsonl` plus the files.

## Running each branch

```bash
# corpus_coding: code N articles with a roster of models, repeated runs
uv run --package corpus_coding python corpus_coding/main.py \
    --n-articles 10 --model-group default --n-runs 2
#   --model-group: default | claude-tiers | gpt-tiers | gemini-tiers | all
#   outputs -> corpus_coding/output/ (results.jsonl, agreement.csv, consistency.csv,
#              cost_by_model.csv, report.html, viz/index.html)

# structure_analysis: extract -> embed (local MiniLM) -> BERTopic -> validate
uv run --package structure_analysis python structure_analysis/main.py \
    --source arxiv --limit 600
#   outputs -> structure_analysis/output/arxiv/ (extractions.jsonl, topics.csv,
#              validation_metrics.json, topic_*_label.csv, viz/index.html)

# bt_scoring: transcribe -> pairwise judge (multi-judge) -> Bradley-Terry
uv run --package bt_scoring python bt_scoring/main.py \
    --num-items 0 --min-matchups 8
#   --skip-transcribe / --skip-judge reuse cached transcripts / comparisons
#   outputs -> bt_scoring/output/ + bt_scoring/comparisons.jsonl + transcripts/

# transcription: Granite-Speech ASR on Common Voice, WER
uv run --package transcription python transcription/main.py
```

Every branch is **resumable**: `results.jsonl` / `comparisons.jsonl` / cached
transcripts and embeddings mean a re-run only does the missing work. Re-running with
a larger `--n-articles` / `--limit` extends rather than redoes.

## Reliability house rules (why re-runs stay clean)

Attendees run these repeatedly, sometimes without a GPU. The house style:

- **Prefer determinism and portability over speed.** `structure_analysis` pins the
  MiniLM embedder to CPU and numba/OpenMP to a single fork-safe thread, and seeds
  UMAP — so clustering is reproducible and needs no GPU. Fetchers take `--seed`.
- **Exit cleanly — no hangs, no zombie processes.** A pipeline that needs a manual
  `kill` undercuts the whole "run it yourself" premise.
- **Recognize the two cautionary tales in this repo:**
  - `structure_analysis` once appeared to "hang at exit." The real cause was a
    self-deadlock in `CostLedger.summary_lines()` re-acquiring a non-reentrant
    `Lock`. It *looked* like a threading/numba problem but wasn't — it was fixed at
    the source, and the code now shuts down normally. When you touch cost/ledger,
    clustering, or exit paths, verify a clean exit with nothing left running.
  - `bt_scoring/main.py` deliberately ends with `os._exit(0)` to dodge a
    HF/torch background-thread shutdown hang from the ASR stack. That is a *known,
    documented tradeoff*, not an oversight: `os._exit` skips interpreter cleanup, so
    it's the last line and only there. Don't copy the pattern casually — reach for a
    real fix first, as `structure_analysis` did.

If a run hangs, check for stray child processes before assuming the logic is wrong —
the failure is often at teardown, not in the analysis.

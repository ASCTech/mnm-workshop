# structure_analysis

**The case study:** extraction → clustering. This branch is modeled on a project that
took 1,716 syllabi across 79 courses → **5-field LLM extraction** → embeddings →
**BERTopic** (UMAP + HDBSCAN) → a topic hierarchy that fed decision-making.

Answers the question of "what can I get from this pile of data, and how". Some aspects
never strictly required recent AI techniques, some made existing approaches cheap,
some are new kinds of analysis that are entirely based in modern language models. 
This can help quickly get oriented in a complex space, make decisions, and enable more 
reproducible  results across subjective matters. However, even though these techniques can 
give quantitative results that doesn't make them objective; the subjectivity is applied
by methodology selection and in the weights of the model.

**Data:** two openly-licensed stand-ins, both from `../data_acquisition/`:

- `fetch_arxiv_metadata.py` — **arXiv** titles/abstracts/categories (full dump is
  **CC0** on Kaggle). The arXiv **category** is a ready-made ground-truth label to
  validate clusters against.
- `fetch_nsf_awards.py` — **NSF award abstracts** (US-gov **public domain**), with
  program/directorate as the analogous label.

```bash
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 800 --seed 7
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 500 --seed 7 --keyword "topic modeling"
```

**What this branch does:**

Big picture, three phases: 
1. Extract - given unstructured data, we use LLMs to extract information that's present in a document but not structured. For example, papers all list author, title, usually have abstracts, but it's hard for a human to write a single rule to accurately identify all of them.
2. Embed - Given that extracted data, embed the words in a vector space. The end result is a vector encoding the semantics of the data according to the model.
3. Compute - Now that the data is in a computable form we can operate on it. These can be simple things, like distance within the vector space, or more complex like dimensional reduction with emergent labels. Here that's BERTopic (UMAP reduces the embeddings, HDBSCAN clusters them) producing topics with no labels given up front.
4. Validate - the clusters emerged unsupervised, so we check them against a held-out ground-truth label the pipeline never used (the arXiv category / NSF program) with ARI, NMI, and homogeneity. This is what turns "some clusters came out" into "the clusters recovered real structure."

## Quickstart

Needs `LITELLM_URL` and `LITELLM_KEY` in the repo-root `.env` (extraction runs through the
OSU LiteLLM proxy); the fetcher uses `CONTACT_EMAIL`. Embeddings and clustering are **local**
(MiniLM pinned to CPU, numba to a single fork-safe thread) so the branch is deterministic and
GPU-free. Install once with `uv sync --all-packages`.

```bash
# 1. Fetch metadata (run once; resumable):
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 800 --seed 7

# 2. Extract -> embed -> cluster -> validate -> visualize:
uv run --package structure_analysis python structure_analysis/main.py --source arxiv --limit 600
# -> structure_analysis/output/arxiv/{extractions.jsonl, doc_topics.csv, topics.csv,
#      topic_tree.txt, topic_dominant_label.csv, validation_metrics.json, viz/index.html}
```

Extraction is cached to `extractions.jsonl` and embeddings to `embeddings_minilm.npy`, so
re-runs skip the LLM and re-cluster instantly. Use `--source nsf` for the NSF corpus.
Open `structure_analysis/output/arxiv/viz/index.html` for the metric tiles and interactive
figures (the `documents.html` button toggles coloring between recovered topic and true label).

## Reading the output

- **`validation_metrics.json`** — how well the recovered topics line up with the held-out
  label. **NMI / V-measure** (mutual information, 0–1) and **ARI** (adjusted Rand index,
  agreement on which docs share a cluster, corrected for chance) measure the same alignment
  from two angles; **homogeneity** (are clusters pure?) vs **completeness** (is each category
  kept in one cluster?) tells you *which way* it's imperfect. `outlier_pct` is HDBSCAN's noise
  label (`topic == -1`) — reported honestly, not hidden.
- **`topic_dominant_label.csv`** — the payoff table: each topic's dominant true label and the
  fraction it covers. In the shipped run topics land at `astro-ph.GA` **100%**, `eess.AS`
  **90%**, `math.PR` **89%** — the clustering rediscovered real fields from extracted text
  alone (NMI ≈ 0.66, ARI ≈ 0.50, ~19% outliers).
- **`doc_topics.csv`** — one row per document; the `Document` column is the pipe-joined
  embedding text (`domain | methods | techniques | contribution`), i.e. exactly what went
  into the embedding, so you can see *why* a doc clustered where it did.

### Notes / things that look wrong but aren't

- **ARI runs lower than NMI** — expected; ARI is stricter and penalizes the large outlier
  group. Both being "only" ~0.5–0.66 is *solidly meaningful* for unsupervised topics on short
  abstracts, not a poor result.
- **A high `outlier_pct` is normal on short text** — HDBSCAN declining to force every abstract
  into a cluster is a feature; those docs are the `-1` topic.
- **Metrics shift run to run** — they depend on corpus sample and size; treat the numbers above
  as calibration, not constants.
- **Model IDs are OSU LiteLLM proxy aliases**; availability may differ in your deployment.



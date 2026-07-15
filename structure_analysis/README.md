# structure_analysis

**Archetype:** *Extraction → clustering* (after `syllabi` in
`../WORKSHOP_ARCHETYPES.md`). The original: 1,716 syllabi / 79 courses →
**5-field LLM extraction** → embeddings → **BERTopic** (UMAP + HDBSCAN) → a topic
hierarchy that fed real college course-planning decisions. This is the
"extract then analyze" two-stage shape. The unlock here is *capability, not
cost*: corpus-wide topic clustering was out of reach for a domain expert alone —
a computational-methods skill, not just effort.

**Data:** two openly-licensed stand-ins, both from `../data_acquisition/`:

- `fetch_arxiv_metadata.py` — **arXiv** titles/abstracts/categories (full dump is
  **CC0** on Kaggle). The arXiv **category** is a ready-made ground-truth label to
  validate clusters against.
- `fetch_nsf_awards.py` — **NSF award abstracts** (US-gov **public domain**), with
  program/directorate as the analogous label.

```bash
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 500
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 500 --keyword "topic modeling"
```

**What this branch does (to build):**

1. **Extract** a few structured fields from each abstract via the LiteLLM proxy
   (see `../envs/`) — e.g. domain, methods, techniques, contribution.
2. **Embed** the extracted text and run **BERTopic** (UMAP + HDBSCAN) into a topic
   hierarchy.
3. **Validate** the recovered clusters against the held-out label (arXiv category
   / NSF program) — a check the original syllabi corpus couldn't offer.

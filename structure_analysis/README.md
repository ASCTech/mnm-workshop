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
uv run --package data_acquisition python data_acquisition/fetch_arxiv_metadata.py --limit 500
uv run --package data_acquisition python data_acquisition/fetch_nsf_awards.py --limit 500 --keyword "topic modeling"
```

**What this branch does:**

Big picture, three phases: 
1. Extract - given unstructured data, we use LLMs to extract information that's present in a document but not structured. For example, papers all list author, title, usually have abstracts, but it's hard for a human to write a single rule to accurately identify all of them.
2. Embed - Given that extracted data, embed the words in a vector space. The end result is a vector encoding the semantics of the data according to the model.
3. Compute - Now that the data is in a computable form we can operate on it. These can be simple things, like distance within the vector space, or more complex like dimensional reduction with emergent labels.



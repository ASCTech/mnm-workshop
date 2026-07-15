# corpus_coding

**Archetype:** *Multi-model corpus coding* (after `article_coding` in
`../WORKSHOP_ARCHETYPES.md`). The flagship case: 1,099 PDFs run through a
10-question codebook across ~5 model-runs (5,120 results), then scored for
**inter-model agreement** and **intra-model consistency**, with a pre-spend cost
estimator. The argument it makes: the grunt-work coding was merely expensive, but
*instrument iteration and model-agreement reliability* were the real unlock — 15×
redundant coding for reliability is infeasible by hand.

**Data:** the openly-licensed stand-in for paywalled article PDFs —
`../data_acquisition/fetch_pmc_oa.py`, which pulls the **PMC Open Access subset**
restricted to **CC0 / CC BY** full-text articles (`.txt` + `.xml` JATS). Run it
first:

```bash
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 50
# -> data_acquisition/data/pmc_oa/{manifest.jsonl, articles/PMC*/...}
```

**What this branch does (to build):**

1. Define a small codebook (N interpretive questions with a fixed answer schema)
   over the article text — e.g. study type, has-RCT, sample size reported,
   funding disclosed, human/animal/in-vitro.
2. Code every article with several models through the **LiteLLM proxy** (see
   `../envs/`), M runs each, into Pydantic-schema JSON.
3. Score **inter-model agreement** (do models code the same article alike?) and
   **intra-model consistency** (does one model repeat itself across runs?), plus
   per-run cost accounting; emit an HTML/CSV report.

CC BY / CC0 licensing means the resulting demo corpus is redistributable, unlike
the original.

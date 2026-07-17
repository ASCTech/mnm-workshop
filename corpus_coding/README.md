# corpus_coding

**The case study:** multi-model corpus coding. This branch is modeled on a project
where we coded 1,099 PDFs through a 10-question codebook across ~5 model-runs (5,120
results), then scored **inter-model agreement** and **intra-model consistency**, with
a cost estimate produced before anything was spent. What stuck with us afterward: the
grunt-work coding was only ever *expensive* — doable, just slow and costly. The part
that had actually been out of reach was the reliability work, running the codebook
15× over for redundancy, which nobody does by hand.

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

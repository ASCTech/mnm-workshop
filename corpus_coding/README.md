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
uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 50
# -> data_acquisition/data/pmc_oa/{manifest.jsonl, articles/PMC*/...}
```

**What this branch does:**
1. Define the codebook/rubric - just as you would with a human graded, define what it is you're looking for.
2. Run the batches - script up your codebook against your dataset and set it off, logging throughout.
3. Analyze result quality - Ideally, you'd have a human coded sample set since that's often the point of comparison. Otherwise, or in addition, intra and inter model agreement can give some information. 

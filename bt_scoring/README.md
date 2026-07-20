# bt_scoring

**The case study:** LLM-as-judge → statistics. This branch is modeled on a project
that scored items pairwise for political orientation (each podcast ≥50 matchups) → a
**Bradley-Terry** left–right scale with standard errors, via an async worker pool with
checkpointing and retry. 

The original example was striking in that practically impossible became tractable. 
While theoretically possible, 10k+ comparisons would take a lot of time and individuals
would likely be biased while being unable to keep all the information in working memory. 
LLM-as-judge makes this orders of magnitude cheaper and faster while being more consistent.

**Data:** the openly-licensed stand-in — `../data_acquisition/fetch_sotu.py`, which
scrapes **State of the Union addresses as plain text** from The American Presidency
Project (presidency.ucsb.edu) for 1950–2020. SOTU addresses are US-government works,
so they're public domain and freely redistributable. Each is labelled by **president**
and **party**; this is information to help us judget how good or bad the rankings are. 
Being text, they feed the pairwise judge directly, with no lossy ASR step (this branch 
used to run on presidential *audio* + Granite ASR, but on commodity hardware the transcripts 
were too poor of quality to be done directly.

```bash
uv run --package data_acquisition python data_acquisition/fetch_sotu.py            # all 1950–2020
uv run --package data_acquisition python data_acquisition/fetch_sotu.py --limit 6  # quick sample
# -> data_acquisition/data/sotu/{manifest.jsonl, texts/<president>_<year>.txt}
```

**What this branch does:**

1. Script arranges all the pairwise comparisons across models and pairs.
2. Fit a Bradley-Terry model to each judge
3. Visualize and validate. Quantitatively, Spearman's rank correlation coefficient tells how much models agree with each other (or against a human). Qualitatively, we kept the president and party here to give an accessible feel.


# The Latent Axis 

The default prompt asks the model to compare on 'left' vs 'right', explicitly without
a rubric. This is very underdefined, and yet we see coherent results. Models are very
much a product of their weights and training and bring their own 'world view' based on
them. This can in part be mitigated by objective metrics, but never certainly reduced to
zero, and care must be taken to distinguish measuring reality and measuring the model.
In this demo, the models have varying levels of agreement, different orderings, and 
from that likely filled the definitional gap with their own views. 


```bash
# cheap smoke (one judge, few items); then the full run
uv run --package bt_scoring python bt_scoring/main.py --num-items 8 --min-matchups 3 --judges gemini-3.1-flash-lite
uv run --package bt_scoring python bt_scoring/main.py --min-matchups 6
# knobs: --char-budget (0 = full speech text), --judges, --dimension, --num-items
```

---
name: mnms-guide
description: >-
  Orientation and tutor for the mnms workshop repository. Use when someone asks
  what this repo or a branch is, what corpus_coding / structure_analysis /
  bt_scoring / transcription / data_acquisition do and why, how to run a branch,
  how to read or interpret its result files (agreement / Cohen's kappa, cost
  tables, topic clusters / NMI, Bradley-Terry scales, judge agreement), or how to
  package their own research method as a Claude Code skill. Explains the
  "corpus in -> structured data out" workshop thesis.
---

# mnms workshop guide

`mnms` is a set of runnable **case studies** for a workshop on agentic tooling for
researchers: openly-licensed, redistributable rebuilds of real OSU
research-consulting projects. They came out of working with a lot of researchers
across a lot of fields — the tools differed every time, but a shape kept recurring:
**corpus in → structured data out**, usually multi-model, cost-accounted, routed
through the OSU LiteLLM proxy.

Your job when this skill is active is to **orient and explain**, not to rebuild
anything. Help people find the branch closest to what they're doing, get it running,
and read what comes out. The branch `README.md`s and the reference files below carry
the details — point *into* those rather than restating them.

## The branches

| Member | Modeled on | One-liner |
|---|---|---|
| `data_acquisition/` | — | Polite, resumable fetchers → `data/<source>/` + a JSONL manifest. Feeds everything else. |
| `corpus_coding/` | a multi-model coding study | Codes a corpus with several models × repeated runs → inter-model **agreement** + intra-model **consistency** + cost. |
| `structure_analysis/` | a syllabus-clustering study | LLM field extraction → local embeddings → **BERTopic** clusters → validated against a held-out label. |
| `bt_scoring/` | a pairwise-judging study | Pairwise **LLM-as-judge** over SOTU address texts (economic left↔right) → a **Bradley-Terry** scale with SEs, validated against party. |
| `transcription/` | self-hosted ASR work | Granite-Speech ASR on Common Voice, WER scoring. |

## Where to go next

Load only the reference file the question needs:

- **Why does this repo exist / what did agents unlock?** → `references/the-argument.md`
- **How do I run a branch?** → `references/running-branches.md`
- **I'm looking at an output file and don't know what it means** → `references/reading-results.md`
- **How do I turn my own method into a skill like this one?** → `references/authoring-skills.md`

## Two things worth surfacing as you help

1. **The recurring shape is `corpus in → structured data out`.** It's not a rule to
   impose — but when someone describes a research task, it's often useful to map it
   onto that shape and onto whichever branch is closest. That's usually the fastest
   way to a running starting point.
2. **Reliability turned out to matter a lot.** Attendees re-run these repeatedly,
   sometimes on machines without a GPU, so we've leaned toward determinism and a
   clean exit over speed. If something hangs or leaves stray processes, it's worth
   flagging — see `running-branches.md` for the two times this bit us.

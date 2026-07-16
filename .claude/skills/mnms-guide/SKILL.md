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

`mnms` is a **workshop reference implementation**: openly-licensed, redistributable
stand-ins for real OSU research-consulting projects. Every branch has the same
shape — **corpus in → structured data out**, multi-model, cost-accounted, routed
through the OSU LiteLLM proxy.

Your job when this skill is active is to **orient and explain**, not to rebuild
anything. Route people to the right branch, help them run it, and help them read
what comes out. The full design argument lives in `WORKSHOP_ARCHETYPES.md` and each
branch's own `README.md` — point *into* those rather than restating them.

## The branches

| Member | Stands in for | One-liner |
|---|---|---|
| `data_acquisition/` | — | Polite, resumable fetchers → `data/<source>/` + a JSONL manifest. Feeds everything else. |
| `corpus_coding/` | `article_coding` | Codes a corpus with several models × repeated runs → inter-model **agreement** + intra-model **consistency** + cost. |
| `structure_analysis/` | `syllabi` | LLM field extraction → local embeddings → **BERTopic** clusters → validated against a held-out label. |
| `bt_scoring/` | `podcasts` | Pairwise **LLM-as-judge** over transcripts → a **Bradley-Terry** scale with standard errors. |
| `transcription/` | `opi-transcript` | Granite-Speech ASR on Common Voice, WER scoring (the self-hosted ASR counterpart). |

## Where to go next

Load only the reference file the question needs:

- **Why does this repo exist / what did agents unlock?** → `references/the-argument.md`
- **How do I run a branch?** → `references/running-branches.md`
- **I'm looking at an output file and don't know what it means** → `references/reading-results.md`
- **How do I turn my own method into a skill like this one?** → `references/authoring-skills.md`

## Two things to model for people

1. **Everything is `corpus in → structured data out`.** When someone describes a
   research task, map it onto that shape and onto whichever branch is closest.
2. **Reliability is a feature here.** Attendees re-run these repeatedly, sometimes
   on machines without a GPU. Prefer determinism and a clean exit over speed, and
   flag anything that hangs or leaves stray processes. See `running-branches.md`.

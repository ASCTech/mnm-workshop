# The argument: what agents actually unlocked

This is the distilled version. The full case — the project catalog, the tiered
"house style", the qualtrics-chatbot deep-dive — is in
[`WORKSHOP_ARCHETYPES.md`](../../../../WORKSHOP_ARCHETYPES.md). Read that for the
whole story; use this file to explain the point quickly.

## The setup

Every branch imitates a real OSU Arts & Sciences research-consulting project. Those
projects share one signature: a domain expert (not a software team) needed to turn
a *method* into working software that processes their corpus — PDFs, audio,
syllabi, survey transcripts — into structured data they can analyze. Python + `uv`,
LLM calls through one LiteLLM proxy, multi-model comparison, cost accounting, and a
report. **Corpus in → structured data out.**

## The two baselines to keep separate

When someone says "couldn't they just do this by hand?", separate two things:

1. **Doing the task by hand** — coding 1,000 PDFs, transcribing hundreds of clips.
2. **Building the tool yourself, without a dev team.**

Agents mostly collapsed the *second*. That's the story worth telling.

## Three tiers of change (weakest → strongest rhetoric)

Map each branch onto the tier it best illustrates:

1. **Merely expensive** — grunt work that was always possible, just costly. Months
   of RA time → a days-long round trip for a few dollars.
   → **`corpus_coding`** is the cost story: the same codebook run across many
   models and repeated runs, for dollars. (See the cost spread in
   `reading-results.md`.)

2. **Previously out of reach for the domain expert alone** — not effort, but a
   computational-methods skill they didn't have.
   → **`structure_analysis`** is the capability story: corpus-wide topic clustering
   (extract → embed → BERTopic) is not something a humanities/social-science
   researcher assembles solo.

3. **The bottleneck was the code, not the compute** — and some designs are simply
   infeasible by hand at all.
   → **`bt_scoring`** is the strongest: thousands of pairwise judgments that no
   human can hold in mind. It only works because the judge is *cheap and
   consistent*. **Reliability is the unlock, not raw capability.**

## The meta-lesson (why a skill lives in this repo)

The archetype doc's `comparatron` and qualtrics entries make the same point twice:
the durable value is **packaging a hard-won capability into a stable chassis with a
small, safe surface**, so the expert invokes it instead of rebuilding it. A Claude
Code skill *is* that packaging. The `mnms-guide` skill you're reading is a worked
example — see `authoring-skills.md`, which closes the loop by teaching people to
package their *own* methods the same way.

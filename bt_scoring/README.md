# bt_scoring

**Archetype:** *LLM-as-judge → statistics* (after `podcasts` in
`../WORKSHOP_ARCHETYPES.md`). The original: pairwise political-orientation scoring
(each podcast ≥50 matchups) → a **Bradley-Terry** left–right scale with standard
errors, via an async worker pool with checkpointing and retry. Its argument is the
strongest of the three: the study is *effectively impossible by hand* (10k+
pairwise comparisons; humans can't hold hundreds of shows in mind) — it only works
because the judge is **cheap + consistent**. Reliability is the unlock.

**Data:** the openly-licensed stand-in — `../data_acquisition/fetch_presidential_audio.py`,
which pulls **public-domain presidential audio** from the Internet Archive
`presidential_recordings` collection. It was chosen for *paired audio* (so it also
feeds the transcription/ASR branch) and for spanning decades (a time axis).

```bash
uv run --package data_acquisition python data_acquisition/fetch_presidential_audio.py --limit 20
# -> data_acquisition/data/presidential_audio/{manifest.jsonl, <id>/<id>_64kb.mp3}
```

**What this branch does (to build):**

1. Judge items **pairwise** on a latent dimension (e.g. economically
   conservative ↔ liberal, or formal ↔ populist) via the LiteLLM proxy
   (`../envs/`), ≥N matchups each, with checkpointing + retry.
2. Fit a **Bradley-Terry** model → a scale with standard errors.
3. **Validate** the recovered scale against ground truth the original lacked:
   president's party, or ordering in time (the "data over time" axis).

## Planned experiment: audio-direct vs. a transcription stage

These items are *audio*, so there are two ways to get the judge its input, and
comparing them is the interesting part:

- **(A) Transcribe first** — run the audio through `../transcription` (Granite
  ASR), then judge the **text** pairwise. Cheaper judge calls, but ASR error is
  injected upstream of the scale.
- **(B) Audio-direct** — feed the audio to a multimodal model and judge without a
  transcript. No transcription loss, but relies on the judge hearing the content.

Fit a Bradley-Terry scale from each and compare: do the rankings agree (rank
correlation), do the standard errors differ, and does ASR noise from (A) or
prosody/audio cues in (B) move any item? Expected to be informative either way —
it isolates how much the transcription stage costs or adds for this kind of
measurement.

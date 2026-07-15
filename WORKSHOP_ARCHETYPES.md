# Research-Consulting Archetypes — Workshop Reference

Recon synthesis for the AI-practitioner conference workshop (agentic tooling for
academic researchers). Catalogs real projects on `~/Desktop`, the patterns they
exemplify, and the "what did agents actually unlock" argument. Working notes, not
a script.

## The house style

Everything here is one ecosystem: **OSU Arts & Sciences (ASC-ETS / DLATO)**, routed
through a shared **LiteLLM proxy** (`litellmproxy.osu-ai.org`). The recurring
signature of a research-consulting project:

- Python + `uv`, `pyproject.toml`, usually a `CLAUDE.md` + `docs/`
- LLM calls via the LiteLLM proxy (one key, many models; live pricing from `model_hub`).
  Mature ones hit *native* provider SDK paths through the proxy to dodge a translation
  quirk that degrades reasoning-model output.
- **Corpus in → structured data out**: PDFs / audio / images / survey data →
  Pydantic-schema JSON, transcripts, or topic clusters
- Multi-model comparison, agreement/accuracy scoring, cost accounting, HTML/CSV report
- Packaged as a **CLI for a non-technical academic client**; resilient to bad inputs
  (per-doc failures recorded, never fatal)

Core test: *"was this built to help a researcher process/analyze their data or automate
a research workflow?"* — not "does it match every trait."

## Catalog (tiered)

### Tier 1 — flagship case studies
| Project | Archetype | One-liner |
|---|---|---|
| `article_coding` | Multi-model corpus coding | 1,099 PDFs × 10-question codebook × ~5 model-runs (5,120 results); inter-model agreement + intra-model consistency + a pre-spend cost estimator. Most polished (7 docs, tests). |
| `townshand` | Multi-model vision-OCR | Frontier vision models transcribe historical newspapers/sermons/church records vs. human ground truth; CER/WER scoring. Tiny, exemplary `CLAUDE.md`. Purest archetype match. |
| `syllabi` | Extraction → clustering | 1,716 syllabi / 79 courses → 5-field LLM extraction → embeddings → BERTopic (UMAP+HDBSCAN) → topic hierarchy. The "extract then analyze" two-stage shape. |
| `podcasts` | LLM-as-judge → statistics | Pairwise political-orientation scoring (each podcast ≥50 matchups) → Bradley-Terry left-right scale w/ SEs (R). Async worker pool, checkpointing, retry. |
| `comparatron` | Meta-archetype (RFC) | Design doc for a reusable "N models × M runs, compare cost + quality" library; cites `article_coding`/`townshand`/`syllabi` as the three it unifies. The workshop thesis, written down. |

### Tier 2 — complementary
- **Transcription/ASR:** `opi-transcript` (AWS Transcribe batch CLI, layered config) · `mnms/transcription` (this repo — IBM Granite-Speech on CommonVoice, WER scoring; the self-hosted counterpart)
- **LLM-as-judge / measurement:** `chess-engine-analysis` (model×language×framing "design taste" grid on the `article_coding` harness) · `tokenizer_math` (minimal clean LiteLLM-usage exemplar, hash-cached token counts)
- **Participant-facing instruments:** `qualtrics-chats` / `qualtrics-client` / `qualtrix-test` (spike→prototype→prod chat widget in surveys; ephemeral $1/5-min per-respondent keys) — *see deep-dive below*
- **Document remediation service:** `node_js_pdf` (Node + Adobe PDF Services + Bedrock vision → WCAG alt-text; Postgres job state, productionized). `PDF_Accessibility` is vendored ASU/AWS, not ours.
- **Build-vs-adopt eval:** `kraken` (vendored OCR engine + custom suitability report for historical Hebrew/Arabic manuscripts)
- **Durable corpus web apps ("data out"):** `isogenie-rails` (research-dataset catalog, Neo4j graph, 1,332 commits) · `ociana` (ancient N. Arabian inscriptions, Rails/GIS) · `iris-inventory-cc` (college asset mgmt)
- **Cost/usage governance CLIs:** `budget-monitor` (the archetype turned inward — LiteLLM spend → reports + CI badges) · `deadline-reporting` (+ GUI `deadline-client`; render-farm cost)
- **Institutional ETL:** `idw-meltano` + `idw-airflow` + custom `tap/target-{teamdynamix,workday}` (HR ↔ ITSM warehouse)

### Shared infrastructure (plumbing behind every tool)
`litellm-deploy-compose` (likely prod: EC2+ALB, LiteLLM+Postgres+MCP+Grafana) ·
`litellm-compose`, `mcp-proxy-compose` (dev/component) · `llm-templates` (Bash/Python/R
client bootstrap kit — the day-one onboarding layer).

### Instructive non-matches (a slide of their own)
- **False positives by fingerprint** (Python/uv/CLAUDE.md but *not* LLM corpus tools):
  `par`/`tmp` (edtech CLI), `multicat` (Rails placement test), `firefly` (Canvas-RAG
  courseware), `gap` (agent platform), `dash` (PM dashboard). Lesson: tooling profile ≠ purpose.
- **"Before LLMs" contrast:** `LabanotationAR` (2016–17 Unity/HoloLens tool built for an OSU
  Dance professor). Same practice — bespoke tool for a named researcher — predating the archetype.
- **Day-zero stage:** `wni-research` (HuggingFace Europeana corpus-acquisition spike only).
- **Clean skips:** games/engines, vendored OSS (`litellm`, `amplifier`, `blender-chemicals`…),
  SDK/binary downloads (`emsdk`, `llama.cpp`, `neo4j-*`), infra/scratch.

## What agents actually unlocked (the argument)

Two "manual" baselines must be kept separate: **doing the task by hand** vs.
**building the tool yourself without a dev team**. Agents collapsed the *second*;
that's the story.

| Project | Manual task time (est.) | Feasible for a solo domain expert? |
|---|---|---|
| `article_coding` | ~400–600 person-hrs (months, RA salaries; double-coded) | Feasible but slow/costly. **Instrument iteration + model-agreement reliability were not.** |
| `syllabi` | ~570 hrs extraction | Extraction yes. **The clustering that drove college course-planning: no** (computational-methods skill, not effort). |
| `podcasts` | Effectively impossible (10k+ pairwise, humans can't hold hundreds of shows) | **No** — the design only works because the judge is cheap + consistent. |

*(Hour figures are order-of-magnitude, from typical coding/extraction rates + standard
double-coding norms.)*

Three tiers of change, in order of rhetorical strength:
1. **Some results were merely expensive** — grunt-work coding/extraction. Months of RA
   time → days-long round trip for a few dollars. *(Cost story.)*
2. **Some were previously out of reach for the domain expert alone** — corpus-wide topic
   clustering, a 10k-comparison judged ranking, 15× redundant coding for reliability.
   *(Capability story — the stronger one.)*
3. **The bottleneck agents removed was the code, not the compute.** These researchers had
   API access; they lacked the ability to turn a method into working software without
   becoming programmers or funding/coordinating a team. That coordination cost is what
   quietly kills the ambitious version of a project.

Proof points: the podcast + article-coding studies are now in draft/preprint; the syllabi
clustering fed real college-level course-offering decisions.

## Deep-dive: the Qualtrics survey chatbot (most-requested by volume, ~10 studies)

**Shape:** a **stable chassis + swappable modules**, not a box of primitives.

- *Chassis (hard, fragile, hidden):* `${e://Field/key}` credential injection; streaming SSE
  fetch loop; **transcript capture** — JSON-stringified, split into 1000-char chunks, written
  back via `setJSEmbeddedData('conversation_N',…)` (this is what makes it a *research
  instrument*, not a toy); error handling; citation rendering.
- *Knobs (small, safe surface, ~15–30 min):* `model` (one line), `prompt` (persona/task),
  `conversationHistory` seed (AI-turn-first), `search.js` vs `default.js` (web queries +
  mechanical citations), `style` block (cosmetics).

**(1) Is the adapting process a useful capability?** Yes — leveraged, not just convenient.
The risky integration knowledge (Qualtrics gotchas, ephemeral-key security, SSE) is amortized
into the chassis; the exposed surface is small and safe. Marginal cost of study N+1 ≈ 0
(→ product-market fit). Agents lower the barrier even on the knobs (researcher states intent,
agent edits the right brick) and can *extend the chassis* when a study needs a new mechanic —
the difference between a frozen template and a living toolkit. **Caveat:** template models
carry silent-breakage risk (a provider changes streaming format → all 10 break) and
guardrail-deletion risk (a pasted prompt drops a safety rule). So the role shifts from
*build* to *steward* — where the consultant/agent still earns its place.

**(2) Does chat-in-survey open new research at scale?** (Excluding studies *of* LLM
interaction — trivially new.) Yes. Affordances a static survey can't provide:
- **Adaptive interviewing at n = thousands** — collapses the old depth-vs-scale tradeoff
  (deep human interviews, n=dozens) vs. (scalable closed surveys). Qualitative depth at
  quantitative scale, low interviewer variance.
- **Standardized interactive *treatments*** — persuasion, deliberation, misinformation
  correction, emotional support ("empathic" persona = an intervention). Pre-LLM needed human
  confederates: unscalable, un-standardizable. A system prompt *can* be held constant → clean
  causal inference; swap a brick = an experimental condition.
- **Controlled-but-realistic information environments** (`search.js`) — real cited answers,
  logged, inside the instrument.
- **Rich open-ended elicitation + automatic coding loop**; standardized role-play/scenario
  instruments.

**The honest boundary:** almost none of this is *conceptually* impossible pre-LLM (human
interviewers, confederates, RA coders existed). What's new is **scale × consistency × cost
simultaneously** — and for most budgets, "cost-prohibited" = "unavailable." The other genuinely
new element is **standardizing an interactive stimulus**, which is what makes the causal
versions valid. And *why now:* rule-based bots couldn't sustain coherent conversation, so the
instrument wasn't trustworthy — **reliability is the unlock, not raw capability.**

**Frontier caveats (where domain + technical expertise co-produce value):** non-determinism
(every respondent gets a slightly different conversation — realism vs. reproducibility);
instrument drift across waves (pinning `model="gpt-5-2025-08-07"` is a *validity* decision
hiding in a config line); IRB/consent/privacy (the ephemeral capped key is the safety primitive).

## Candidate workshop set

Anchor on seven, in narrative order: **`article_coding`** (flagship: coding + agreement +
cost) → **`townshand`** (smallest complete example, vision-OCR) → **`syllabi`** (extraction +
clustering) → **`podcasts`** (LLM as instrument feeding real statistics) → **`qualtrics-chats`**
(participant-facing + credential safety) → **`comparatron`** (the "extract the reusable library"
meta-lesson that ties it together) → **`budget-monitor`** + LiteLLM proxy (governance backstory).
Bookend with **`LabanotationAR`** ("before LLMs") and **`par`/`multicat`** ("looks like it, isn't").

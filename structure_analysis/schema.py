"""Pydantic schema for the per-document LLM extraction (stage 1).

Five short structured fields, mirroring the `syllabi` archetype's 5-field
extraction but tuned for a research-abstract corpus (arXiv / NSF) instead of
a syllabus: domain, methods, techniques, contribution, application. These
feed stage 2's embedding text and stage 3's topic labels.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AbstractFields(BaseModel):
    domain: str = Field(
        ...,
        description="The short research field/subfield this work belongs to (e.g. 'computer vision', 'algebraic topology', 'freshwater ecology').",
    )
    methods: list[str] = Field(
        default_factory=list,
        description="Short names of the methodological approaches used (e.g. 'randomized controlled trial', 'diffusion model', 'field survey'). 1-4 items.",
    )
    techniques: list[str] = Field(
        default_factory=list,
        description="Specific named techniques, models, algorithms, or instruments (e.g. 'transformer', 'HDBSCAN', 'RNA-seq', 'finite element analysis'). 1-4 items.",
    )
    contribution: str = Field(
        ...,
        description="One short phrase for the main contribution or finding (e.g. 'a new benchmark dataset', 'a tighter regret bound', 'evidence of X causing Y').",
    )
    application: str = Field(
        ...,
        description="One short phrase for the target application or downstream use (e.g. 'autonomous driving perception', 'climate policy', 'drug discovery').",
    )

    @classmethod
    def example_json(cls) -> str:
        return (
            "{\n"
            '  "domain": "computer vision",\n'
            '  "methods": ["self-supervised learning", "benchmark evaluation"],\n'
            '  "techniques": ["vision transformer", "contrastive loss"],\n'
            '  "contribution": "a video foundation model that improves reconstruction quality",\n'
            '  "application": "video generation"\n'
            "}"
        )


SYSTEM_PROMPT = f"""You extract structured metadata from a research abstract.

Return STRICT JSON with exactly these keys: domain, methods, techniques, contribution, application.
- domain: a short (2-4 word) research field/subfield label.
- methods: a list of 1-4 short methodological-approach names.
- techniques: a list of 1-4 short specific technique/model/instrument names.
- contribution: one short phrase (<= 15 words) for the main contribution or finding.
- application: one short phrase (<= 10 words) for the target application or downstream use.

Keep every field terse — these are labels for clustering, not sentences. Use the
abstract's own vocabulary where possible. Do not include any keys other than the
five listed. Example:

{AbstractFields.example_json()}
"""

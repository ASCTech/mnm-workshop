"""Pydantic models for the codebook, LLM responses, and cost tracking.

The codebook is deliberately small (9 questions) and generic across biomedical
article types so that any of the cheap comparison models can answer it
reliably from a truncated excerpt. Every question has a fixed answer type
(bool / enum / short string) so multiple models and multiple runs of the same
model can be compared field-by-field.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class StudyType(str, Enum):
    RCT = "rct"
    OBSERVATIONAL = "observational"
    REVIEW = "review"
    CASE_REPORT = "case_report"
    METHODS = "methods"
    OTHER = "other"


class CodingResponse(BaseModel):
    """Fixed-schema answers to the corpus_coding codebook for one article."""

    study_type: StudyType = Field(
        description=(
            "rct = randomized controlled trial; observational = cohort / "
            "case-control / cross-sectional; review = review, systematic "
            "review, or meta-analysis; case_report = case report or case "
            "series; methods = a new method, protocol, or instrument paper; "
            "other = anything else (editorial, commentary, ...)"
        )
    )
    has_human_subjects: bool = Field(
        description="Study involved human participants (not purely animal/in-vitro/computational)"
    )
    has_animal_subjects: bool = Field(description="Study involved live animal subjects")
    in_vitro: bool = Field(
        description="Study is (also) in-vitro / cell-culture / bench / computational work"
    )
    sample_size_reported: bool = Field(
        description="A specific numeric sample size (n=...) is stated for the study population"
    )
    funding_disclosed: bool = Field(
        description="A funding source or grant is explicitly named, including an explicit 'no funding' statement"
    )
    study_registered: bool = Field(
        description="A clinical trial registry number or preregistration is mentioned (e.g. ClinicalTrials.gov, PROSPERO)"
    )
    open_data_statement: bool = Field(
        description="The article states data/code availability, even if only 'available upon request'"
    )
    primary_field: str = Field(
        max_length=60,
        description="Short label (<=4 words) for the primary scientific field/subfield, e.g. 'oncology', 'neuroscience'",
    )


# Ordered list of question names, reused everywhere scoring needs to iterate
# the codebook generically.
QUESTIONS: list[str] = list(CodingResponse.model_fields.keys())


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CodedRecord(BaseModel):
    """One (article, model, run) coding attempt, as persisted to results.jsonl."""

    pmcid: str
    model: str
    run: int
    response: CodingResponse | None = None
    usage: Usage = Usage()
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error: str | None = None
    timestamp: str

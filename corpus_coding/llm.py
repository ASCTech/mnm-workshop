"""LiteLLM-proxy client: prompt construction, structured parsing, pricing.

Structured output strategy: ask for JSON via a strict system prompt plus
``response_format={"type": "json_object"}``, then validate with
:class:`schema.CodingResponse`. Some proxied models are flaky under
``json_object`` mode (a couple of them are known to occasionally return an
empty ``{}``), so on a parse/validation failure we retry once *without*
``response_format``, relying on the prompt alone. Any failure on the second
attempt is recorded as a per-item error rather than raised.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import OpenAI
from pydantic import ValidationError

from schema import CodingResponse, QUESTIONS

DEFAULT_MODELS = [
    "gemini-2.5-flash-lite",
    "claude-haiku-4-5-20251001",
    "llama3-3-70b-instruct",
]

# USD per 1M tokens. Pulled from the OSU LiteLLM model_hub where known;
# entries marked (est.) are reasonable public-pricing estimates, not billed
# figures from the proxy itself.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "llama3-3-70b-instruct": (0.72, 0.72),
    "gemini-2.5-flash-lite": (0.10, 0.40),  # (est.)
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "nova-lite-v1": (0.06, 0.24),
}
DEFAULT_PRICE = (1.00, 3.00)  # (est.) fallback for a model not in the table above

SYSTEM_PROMPT = f"""You are a careful research methodologist coding academic articles \
for a structured meta-analysis. You will be given an excerpt of one article and must \
answer a fixed set of coding questions about it.

Respond with ONLY a single JSON object (no markdown fences, no commentary) with exactly \
these keys:

{{
  "study_type": one of "rct", "observational", "review", "case_report", "methods", "other",
  "has_human_subjects": true/false,
  "has_animal_subjects": true/false,
  "in_vitro": true/false,
  "sample_size_reported": true/false,
  "funding_disclosed": true/false,
  "study_registered": true/false,
  "open_data_statement": true/false,
  "primary_field": "short label, 4 words or fewer"
}}

Field definitions:
- study_type: rct = randomized controlled trial; observational = cohort/case-control/
  cross-sectional; review = review/systematic review/meta-analysis; case_report = case
  report or case series; methods = a new method/protocol/instrument paper; other =
  anything else (editorial, commentary, perspective, ...).
- has_human_subjects: true if human participants were studied directly.
- has_animal_subjects: true if live animal subjects were studied.
- in_vitro: true if the study is (also) in-vitro/cell-culture/bench/computational work.
- sample_size_reported: true only if a specific numeric n is stated for the study
  population.
- funding_disclosed: true if a funding source/grant is named, including an explicit
  "no funding" statement.
- study_registered: true if a trial registry number or preregistration is mentioned
  (e.g. ClinicalTrials.gov, PROSPERO).
- open_data_statement: true if the article states data/code availability, even
  "available upon request".
- primary_field: your best short label for the primary scientific field/subfield.

Base every answer only on the excerpt provided. If something is not stated, answer
conservatively (false, or "other" for study_type)."""


def build_messages(article_text: str) -> list[dict]:
    user = (
        "Article excerpt (may be truncated):\n\n"
        f"{article_text}\n\n"
        "Respond with the JSON object described in the system prompt."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating ```-fences."""
    content = text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        end = len(lines) - 1
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        content = "\n".join(lines[1:end])
    # Some models add stray prose around the object; take the outermost braces.
    start, stop = content.find("{"), content.rfind("}")
    if start != -1 and stop != -1 and stop > start:
        content = content[start : stop + 1]
    return json.loads(content)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICE_TABLE.get(model, DEFAULT_PRICE)
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


@dataclass
class CallResult:
    response: CodingResponse | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None


def _one_attempt(client: OpenAI, model: str, article_text: str, temperature: float, use_json_mode: bool):
    kwargs = dict(
        model=model,
        messages=build_messages(article_text),
        temperature=temperature,
    )
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    parsed = CodingResponse(**_extract_json(text))
    return parsed, prompt_tokens, completion_tokens


def call_model(
    client: OpenAI,
    model: str,
    article_text: str,
    temperature: float,
) -> CallResult:
    """Code one article with one model. Retries once (without json_mode) on failure."""
    start = time.monotonic()
    last_error: Exception | None = None
    prompt_tokens = completion_tokens = 0

    for attempt, use_json_mode in enumerate((True, False)):
        try:
            parsed, prompt_tokens, completion_tokens = _one_attempt(
                client, model, article_text, temperature, use_json_mode
            )
            latency = time.monotonic() - start
            return CallResult(parsed, prompt_tokens, completion_tokens, latency, None)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            last_error = e
            continue
        except Exception as e:  # network/API errors: no point retrying differently
            last_error = e
            break

    latency = time.monotonic() - start
    return CallResult(None, prompt_tokens, completion_tokens, latency, f"{type(last_error).__name__}: {last_error}")

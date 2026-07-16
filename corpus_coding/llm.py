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

# Full workshop roster (round 2). deepseek-r1 was dropped from the workshop
# entirely and must never appear here.
ALL_MODELS = [
    # OpenAI
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
    # Gemini
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    # Open / other
    "minimax-m2.5",
    "kimi-k2.5",
    "llama3-3-70b-instruct",
    # Anthropic
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-8",
]

# Diverse default: both families (OpenAI/Gemini/open/Anthropic) and both
# capability tiers (small vs. large) represented. kimi-k2.5 is left out of
# the default -- it works but is slow (~26s/call); pass it explicitly via
# --models or the "all" group if you want it.
DEFAULT_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-opus-4-8",
    "gpt-5.4-mini-2026-03-17",
    "gemini-3.1-flash-lite",
    "llama3-3-70b-instruct",
    "minimax-m2.5",
]

# Convenience groups for same-family, different-capability-tier comparisons
# ("does capability tier change the coding?"), selectable via --model-group.
TIER_GROUPS: dict[str, list[str]] = {
    "default": DEFAULT_MODELS,
    "claude-tiers": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-8"],
    "gpt-tiers": ["gpt-5.4-nano-2026-03-17", "gpt-5.4-mini-2026-03-17", "gpt-5.4-2026-03-05"],
    "gemini-tiers": ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.1-pro-preview"],
    "all": ALL_MODELS,
}

# USD per 1M tokens. Pulled from the OSU LiteLLM model_hub where known;
# entries marked (est.) are reasonable public-pricing estimates, not billed
# figures from the proxy itself.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),  # (est.) not yet in the pricing registry snapshot used here
    "claude-opus-4-8": (5.00, 25.00),
    "llama3-3-70b-instruct": (0.72, 0.72),
    "minimax-m2.5": (0.30, 1.20),
    "kimi-k2.5": (0.60, 3.03),
    "gpt-5.4-2026-03-05": (2.50, 15.00),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "gpt-5.4-nano-2026-03-17": (0.20, 1.25),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),  # (est.) mapped from gemini-3.1-flash-lite-preview
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),  # (est.)
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "nova-lite-v1": (0.06, 0.24),
}
DEFAULT_PRICE = (1.00, 3.00)  # (est.) fallback for a model not in the table above

# Models whose proxy/provider rejects a custom temperature (gpt-5.x only
# accepts the implicit default of 1; Bedrock rejects the parameter entirely
# for some newer Claude models, calling it "deprecated for this model").
# For these we omit the "temperature" kwarg rather than send the caller's
# value. claude-sonnet-5 confirmed empirically (BedrockException: "temperature
# is deprecated for this model"); call_model() also message-sniffs for any
# other model not listed here, so this set is a fast-path, not the only net.
NO_CUSTOM_TEMPERATURE: set[str] = {
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
    "claude-opus-4-8",
    "claude-sonnet-5",
}

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


def _one_attempt(
    client: OpenAI,
    model: str,
    article_text: str,
    temperature: float,
    use_json_mode: bool,
    force_no_temperature: bool = False,
):
    kwargs = dict(
        model=model,
        messages=build_messages(article_text),
    )
    if not force_no_temperature and model not in NO_CUSTOM_TEMPERATURE:
        kwargs["temperature"] = temperature
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    completion = client.chat.completions.create(**kwargs)
    text = completion.choices[0].message.content or ""
    usage = completion.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    parsed = CodingResponse(**_extract_json(text))
    return parsed, prompt_tokens, completion_tokens


def _attempt_with_temperature_fallback(
    client: OpenAI, model: str, article_text: str, temperature: float, use_json_mode: bool, temp_disabled: bool,
):
    """Run one (json_mode) attempt; if the provider rejects a custom
    temperature (message mentions "temperature") on a model we didn't already
    know about, retry the *same* json_mode setting once with temperature
    omitted. Returns (parsed, prompt_tokens, completion_tokens, temp_disabled).
    Raises the parse/validation error (if any) or the final API error.
    """
    try:
        parsed, pt, ct = _one_attempt(
            client, model, article_text, temperature, use_json_mode, force_no_temperature=temp_disabled
        )
        return parsed, pt, ct, temp_disabled
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise
    except Exception as e:
        if not temp_disabled and "temperature" in str(e).lower():
            parsed, pt, ct = _one_attempt(
                client, model, article_text, temperature, use_json_mode, force_no_temperature=True
            )
            return parsed, pt, ct, True
        raise


def call_model(
    client: OpenAI,
    model: str,
    article_text: str,
    temperature: float,
) -> CallResult:
    """Code one article with one model.

    Two independent, chainable fallbacks (both can fire on the same call):
    - temperature rejection (some models only accept their fixed default;
      known ones are in NO_CUSTOM_TEMPERATURE, others are caught by message
      sniffing) -> retry the same json_mode setting with temperature omitted.
    - json_object mode returning an empty/invalid body (a few Claude models
      do this some or all of the time) -> retry once without response_format,
      relying on the prompt alone.
    """
    start = time.monotonic()
    last_error: Exception | None = None
    prompt_tokens = completion_tokens = 0
    temp_disabled = model in NO_CUSTOM_TEMPERATURE

    for use_json_mode in (True, False):
        try:
            parsed, prompt_tokens, completion_tokens, temp_disabled = _attempt_with_temperature_fallback(
                client, model, article_text, temperature, use_json_mode, temp_disabled
            )
            latency = time.monotonic() - start
            return CallResult(parsed, prompt_tokens, completion_tokens, latency, None)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            last_error = e
            continue  # try again without json_mode
        except Exception as e:  # network/API errors unrelated to temperature: no point retrying differently
            last_error = e
            break

    latency = time.monotonic() - start
    return CallResult(None, prompt_tokens, completion_tokens, latency, f"{type(last_error).__name__}: {last_error}")

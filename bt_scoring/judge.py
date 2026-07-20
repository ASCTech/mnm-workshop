"""Pairwise LLM-as-judge over document (speech-text) pairs, via the OSU LiteLLM proxy.

Mirrors the shape of the ``podcasts`` reference (async worker pool, JSON
checkpoint per comparison, retry-on-failure) but compacted: one
``comparisons.jsonl`` file instead of one JSON file per matchup, and a single
asyncio semaphore instead of an explicit queue/worker set.

Round 2 extends this to a *multi-judge* comparison: the identical pair set is
judged by several models so their Bradley-Terry scales can be compared at low
n. To keep that comparison fair, the A/B position for a given (item_i, item_j)
pair is derived deterministically from the pair + a seed (``_position_for_pair``)
rather than a per-call RNG draw, so every judge sees the same left/right
assignment for the same pair. ``comparisons.jsonl`` records are keyed for
resumability by (pair, judge model, order) so re-running with a different or
expanded judge roster only judges the missing (pair, judge) combinations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

# Rough, labeled-as-estimate USD price per 1M tokens (input, output). Not
# pulled live from the proxy's model_hub -- for order-of-magnitude cost
# accounting only. Entries not covering a requested judge simply yield a
# `None` cost (reported as "n/a") rather than a fabricated number.
PRICE_PER_1M_USD: dict[str, tuple[float, float]] = {
    # Default judge roster (DEFAULT_JUDGES in main.py)
    "gemini-3.1-flash-lite": (0.10, 0.40),         # flash-lite tier, ~stable across gens
    "gemini-3.1-pro-preview": (1.25, 5.00),        # pro tier, order-of-magnitude estimate
    "gpt-5.4-mini-2026-03-17": (0.25, 1.00),       # mini tier, order-of-magnitude estimate
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),              # confirmed current Anthropic list price
    # Other models you might pass via --judges (kept so cost is still accounted)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "llama3-3-70b-instruct": (0.20, 0.20),
}

# Judges known (empirically, per the workshop coordinator) to sometimes return
# a bare "{}" under response_format={"type": "json_object"} for this task's
# prompt shape. For these, a validation failure on the first (structured)
# attempt triggers a retry WITHOUT response_format, parsing JSON out of the
# raw text reply instead.
_EMPTY_JSON_RETRY_PREFIXES = ("claude-",)


def _needs_empty_json_fallback(model: str) -> bool:
    return model.startswith(_EMPTY_JSON_RETRY_PREFIXES)


# Models observed (empirically, against this LiteLLM proxy) to 400 on
# temperature=0. "gpt-5*" models require temperature=1 (or omitted);
# claude-opus-4-8 rejects the temperature param outright on this proxy's
# Bedrock backend. Learned quirks (any other model that 400s on temperature)
# are added to this set at runtime by ``_call_judge`` so the cost of
# discovering them is paid once per model, not once per call.
_NO_TEMPERATURE: set[str] = {"claude-opus-4-8"}


def _omit_temperature(model: str) -> bool:
    return model.startswith("gpt-5") or model in _NO_TEMPERATURE


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of a single JSON object from a plain-text reply
    (strips markdown code fences, takes the outermost {...} span)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found in reply", text, 0)
    return json.loads(text[start : end + 1])


class Verdict(BaseModel):
    winner: Literal["A", "B"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def sample_matchups(
    identifiers: list[str], min_matchups: int, seed: int,
) -> list[tuple[str, str]]:
    """Smallest random subset of all pairwise combinations such that every
    identifier appears in at least ``min_matchups`` comparisons.

    Adapted from the podcasts reference's ``smallest_random_sample``.
    """
    rng = random.Random(seed)
    all_pairs = list(combinations(sorted(identifiers), 2))
    rng.shuffle(all_pairs)

    counts: dict[str, int] = defaultdict(int)
    remaining = set(identifiers)
    selected: list[tuple[str, str]] = []

    for a, b in all_pairs:
        if a not in remaining and b not in remaining:
            continue
        selected.append((a, b))
        counts[a] += 1
        counts[b] += 1
        if counts[a] >= min_matchups:
            remaining.discard(a)
        if counts[b] >= min_matchups:
            remaining.discard(b)
        if not remaining:
            break

    return selected


def _clip(text: str, max_chars: int) -> str:
    """Truncate to ``max_chars`` characters; ``max_chars <= 0`` means no limit."""
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def build_prompt(dimension: str, text_a: str, text_b: str, max_chars: int = 0) -> str:
    """Prompt for one pairwise comparison on ``dimension``, written as
    ``left-pole <-> right-pole``. ``max_chars`` optionally truncates each
    document (0 = send the full text). The prompt deliberately does NOT hand
    the model a rubric for the dimension -- how it interprets an ambiguous
    axis (e.g. "economic left/right") is part of what we're measuring across
    judges."""
    text_a = _clip(text_a, max_chars)
    text_b = _clip(text_b, max_chars)
    return f"""You are judging two speech excerpts on a single dimension, written as "left-pole <-> right-pole":

    {dimension}

Excerpt A:
\"\"\"{text_a}\"\"\"

Excerpt B:
\"\"\"{text_b}\"\"\"

Which excerpt sits MORE toward the SECOND pole of the dimension (the side after
the "<->")? Judge the substance of the text against your own understanding of
that pole -- there is no single official rubric.

Respond with a JSON object with exactly these keys:
- "winner": "A" or "B"
- "confidence": a float from 0.0 (coin flip) to 1.0 (certain)
- "reason": one short sentence"""


def _usage_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prices = PRICE_PER_1M_USD.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return prompt_tokens / 1e6 * in_price + completion_tokens / 1e6 * out_price


async def _call_judge(
    client: AsyncOpenAI, model: str, prompt: str, timeout_s: float,
) -> tuple[Verdict | None, dict, str | None]:
    """One judge call with retries on parse/validation/param failures.
    Returns (verdict_or_None, usage_dict, error_or_None).

    Handles three distinct, empirically-observed proxy/model quirks:
      - ``gpt-5.4-mini-2026-03-17`` and ``claude-opus-4-8`` 400 on
        temperature=0 through this LiteLLM proxy ("temperature is deprecated
        for this model" / "only temperature=1 is supported"). Detected by
        message content and cached per-model in ``_NO_TEMPERATURE`` so later
        calls for the same model skip the wasted round-trip.
      - Claude models (``_needs_empty_json_fallback``) have been observed to
        return a bare ``{}`` under response_format={"type": "json_object"}
        for this prompt shape -- valid JSON, but fails Verdict validation
        (missing keys). On that failure we retry WITHOUT response_format,
        asking for a plain-text JSON object and parsing it out of the reply.
      - Generic parse/validation failure otherwise: retry the same request
        (still with response_format=json_object) with a short correction
        note appended.
    """
    last_err = None
    current_prompt = prompt
    use_response_format = True
    for attempt in range(3):
        try:
            kwargs = dict(model=model, messages=[{"role": "user", "content": current_prompt}])
            if not _omit_temperature(model):
                kwargs["temperature"] = 0
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), timeout=timeout_s,
            )
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            }
            raw = resp.choices[0].message.content or ""
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = _extract_json_object(raw)
            verdict = Verdict.model_validate(parsed)
            return verdict, usage, None
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if use_response_format and _needs_empty_json_fallback(model):
                use_response_format = False
                current_prompt = (
                    prompt
                    + "\n\nRespond with ONLY a raw JSON object matching the schema above"
                    " -- no markdown code fence, no commentary before or after it."
                )
            else:
                current_prompt = (
                    current_prompt
                    + "\n\nYour previous reply was not valid JSON matching the schema."
                    " Reply with ONLY the JSON object."
                )
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if "temperature" in str(e).lower() and not _omit_temperature(model):
                _NO_TEMPERATURE.add(model)  # learned quirk; skip sending temperature from here on
            else:
                await asyncio.sleep(1.5 * (attempt + 1))
    return None, {"prompt_tokens": 0, "completion_tokens": 0}, last_err


def _position_for_pair(item_i: str, item_j: str, seed: int) -> tuple[str, str, str]:
    """Deterministic A/B position assignment for a pair, shared across every
    judge model so all judges see an identical left/right text order for the
    same pair (position bias is still controlled -- randomized across pairs
    -- but not re-randomized per judge, which would confound judge-vs-judge
    comparison with position-bias noise)."""
    digest = hashlib.sha256(f"{seed}:{_pair_key(item_i, item_j)}".encode()).hexdigest()
    swap = int(digest[:8], 16) % 2 == 1
    left, right = (item_j, item_i) if swap else (item_i, item_j)
    order = "swapped" if swap else "unswapped"
    return left, right, order


async def judge_one(
    client: AsyncOpenAI,
    item_i: str,
    item_j: str,
    texts: dict[str, str],
    dimension: str,
    model: str,
    left: str,
    right: str,
    order: str,
    timeout_s: float,
    max_chars: int = 0,
) -> dict:
    """Judge the (item_i, item_j) pair once with model ``model``, using the
    precomputed (left, right) position assignment shared across all judges
    for this pair."""
    prompt = build_prompt(dimension, texts[left], texts[right], max_chars)

    verdict, usage, error = await _call_judge(client, model, prompt, timeout_s)
    record = {
        "item_i": item_i,
        "item_j": item_j,
        "position_a": left,
        "position_b": right,
        "order": order,
        "model": model,
        "usage": usage,
        "cost_usd_est": _usage_cost(model, usage["prompt_tokens"], usage["completion_tokens"]),
    }
    if verdict is None:
        record["status"] = "error"
        record["error"] = error
        return record

    winner_id = left if verdict.winner == "A" else right
    record.update(
        status="ok",
        winner=winner_id,
        confidence=verdict.confidence,
        reason=verdict.reason,
    )
    return record


def _pair_key(a: str, b: str) -> str:
    return "::".join(sorted((a, b)))


def _infer_order(rec: dict) -> str:
    """Round-1 records predate the explicit "order" field; recover it from
    position_a vs item_i for backward-compatible resumability keys."""
    if "order" in rec:
        return rec["order"]
    return "swapped" if rec.get("position_a") != rec.get("item_i") else "unswapped"


def load_completed(comparisons_path: Path) -> set[tuple[str, str, str]]:
    """Returns the set of (pair_key, judge_model, order) triples already
    recorded with status "ok" -- the resumability key for the multi-judge
    checkpoint. ``order`` is a deterministic function of the pair (see
    ``_position_for_pair``), so in practice this reduces to (pair, model),
    but is kept explicit per the (pair, judge, order) keying contract."""
    completed: set[tuple[str, str, str]] = set()
    if not comparisons_path.exists():
        return completed
    with comparisons_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                completed.add(
                    (_pair_key(rec["item_i"], rec["item_j"]), rec.get("model"), _infer_order(rec))
                )
    return completed


async def run_judging(
    client: AsyncOpenAI,
    matchups: list[tuple[str, str]],
    judges: list[str],
    texts: dict[str, str],
    dimension: str,
    comparisons_path: Path,
    seed: int,
    concurrency: int = 6,
    timeout_s: float = 60.0,
    max_chars: int = 0,
) -> None:
    """Judge every (pair, judge) combination not already recorded as 'ok' in
    comparisons_path, appending each result as soon as it completes
    (resumable checkpoint, per-judge). The SAME pair set (``matchups``) is
    judged by every model in ``judges``, and the A/B position for a given
    pair is identical across judges (``_position_for_pair``), so per-judge
    Bradley-Terry scales are directly comparable."""
    completed = load_completed(comparisons_path)

    tasks: list[tuple[str, str, str, str, str, str]] = []  # (i, j, left, right, order, model)
    for item_i, item_j in matchups:
        left, right, order = _position_for_pair(item_i, item_j, seed)
        pk = _pair_key(item_i, item_j)
        for model in judges:
            if (pk, model, order) in completed:
                continue
            tasks.append((item_i, item_j, left, right, order, model))

    print(
        f"Comparisons done: {len(completed)}; to run: {len(tasks)} "
        f"({len(matchups)} pairs x {len(judges)} judges)",
        flush=True,
    )
    if not tasks:
        return

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    comparisons_path.parent.mkdir(parents=True, exist_ok=True)

    async def worker(task: tuple[str, str, str, str, str, str], idx: int) -> None:
        item_i, item_j, left, right, order, model = task
        async with semaphore:
            record = await judge_one(
                client, item_i, item_j, texts, dimension, model, left, right, order, timeout_s, max_chars,
            )
        async with lock:
            with comparisons_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        tag = record.get("winner", record.get("error"))
        print(f"[{idx + 1}/{len(tasks)}] {model}: {item_i} vs {item_j} -> {tag}", flush=True)

    await asyncio.gather(*(worker(t, i) for i, t in enumerate(tasks)))

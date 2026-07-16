"""Pairwise LLM-as-judge over transcript pairs, via the OSU LiteLLM proxy.

Mirrors the shape of the ``podcasts`` reference (async worker pool, JSON
checkpoint per comparison, retry-on-failure) but compacted: one
``comparisons.jsonl`` file instead of one JSON file per matchup, and a single
asyncio semaphore instead of an explicit queue/worker set.
"""

from __future__ import annotations

import asyncio
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
# accounting only.
PRICE_PER_1M_USD: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "llama3-3-70b-instruct": (0.20, 0.20),
}


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


def build_prompt(dimension: str, text_a: str, text_b: str) -> str:
    return f"""You are judging two short transcript excerpts on a single dimension:

    {dimension}

Excerpt A:
\"\"\"{text_a}\"\"\"

Excerpt B:
\"\"\"{text_b}\"\"\"

Which excerpt is MORE toward the second pole of the dimension above ("B" side
of the "/" as written)? If the dimension reads "measured/formal <-> emotionally
intense/populist", answer "A" or "B" for whichever excerpt is more emotionally
intense/populist; the other pole is the default.

Respond with a JSON object with exactly these keys:
- "winner": "A" or "B"
- "confidence": a float from 0.0 (coin flip) to 1.0 (certain)
- "reason": one short sentence

Base your judgment only on the text given; ignore transcription artifacts."""


def _usage_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prices = PRICE_PER_1M_USD.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return prompt_tokens / 1e6 * in_price + completion_tokens / 1e6 * out_price


async def _call_judge(
    client: AsyncOpenAI, model: str, prompt: str, timeout_s: float,
) -> tuple[Verdict | None, dict, str | None]:
    """One judge call with one retry on parse/validation failure. Returns
    (verdict_or_None, usage_dict, error_or_None)."""
    last_err = None
    for attempt in range(2):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout_s,
            )
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            }
            raw = resp.choices[0].message.content
            verdict = Verdict.model_validate(json.loads(raw))
            return verdict, usage, None
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = f"{type(e).__name__}: {e}"
            prompt = prompt + "\n\nYour previous reply was not valid JSON matching the schema. Reply with ONLY the JSON object."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(1.5 * (attempt + 1))
    return None, {"prompt_tokens": 0, "completion_tokens": 0}, last_err


async def judge_one(
    client: AsyncOpenAI,
    item_i: str,
    item_j: str,
    transcripts: dict[str, str],
    dimension: str,
    model: str,
    rng: random.Random,
    timeout_s: float,
) -> dict:
    """Judge the (item_i, item_j) pair once, with a randomized A/B position
    assignment to control for position bias."""
    swap = rng.random() < 0.5
    left, right = (item_j, item_i) if swap else (item_i, item_j)
    prompt = build_prompt(dimension, transcripts[left], transcripts[right])

    verdict, usage, error = await _call_judge(client, model, prompt, timeout_s)
    record = {
        "item_i": item_i,
        "item_j": item_j,
        "position_a": left,
        "position_b": right,
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


def load_completed_pairs(comparisons_path: Path) -> set[str]:
    completed: set[str] = set()
    if not comparisons_path.exists():
        return completed
    with comparisons_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                completed.add(_pair_key(rec["item_i"], rec["item_j"]))
    return completed


async def run_judging(
    client: AsyncOpenAI,
    matchups: list[tuple[str, str]],
    transcripts: dict[str, str],
    dimension: str,
    model: str,
    comparisons_path: Path,
    seed: int,
    concurrency: int = 6,
    timeout_s: float = 60.0,
) -> None:
    """Judge every matchup not already recorded as 'ok' in comparisons_path,
    appending each result as soon as it completes (resumable checkpoint)."""
    completed = load_completed_pairs(comparisons_path)
    todo = [(a, b) for a, b in matchups if _pair_key(a, b) not in completed]
    print(f"Comparisons done: {len(completed)}; to run: {len(todo)}", flush=True)
    if not todo:
        return

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    rng = random.Random(seed)
    comparisons_path.parent.mkdir(parents=True, exist_ok=True)

    async def worker(pair: tuple[str, str], idx: int) -> None:
        item_i, item_j = pair
        async with semaphore:
            record = await judge_one(
                client, item_i, item_j, transcripts, dimension, model,
                random.Random(rng.randint(0, 2**31)), timeout_s,
            )
        async with lock:
            with comparisons_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        tag = record.get("winner", record.get("error"))
        print(f"[{idx + 1}/{len(todo)}] {item_i} vs {item_j} -> {tag}", flush=True)

    await asyncio.gather(*(worker(pair, i) for i, pair in enumerate(todo)))

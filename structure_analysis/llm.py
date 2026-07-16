"""Thin OpenAI-compatible client for the OSU LiteLLM proxy, plus cost accounting.

One client for both chat (extraction) and embeddings calls. Pricing is a small,
hand-maintained per-model table (USD per 1M tokens) snapshotted from the proxy's
`/public/model_hub` on 2026-07-16 — treat totals as *labeled estimates*, not an
invoice; re-check `model_hub` if the proxy's pricing changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# {model: (input_$_per_million_tokens, output_$_per_million_tokens)}
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "llama3-3-70b-instruct": (0.72, 0.72),
    "titan-embed-text-v2:0": (0.02, 0.00),
    # local sentence-transformers inference: no proxy call, no token cost.
    "all-MiniLM-L6-v2 (local)": (0.0, 0.0),
}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LITELLM_URL"].rstrip("/") + "/v1",
            api_key=os.environ["LITELLM_KEY"],
        )
    return _client


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Labeled estimate — see module docstring. Unknown models price at $0."""
    in_price, out_price = PRICES_PER_MILLION.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


@dataclass
class CostLedger:
    """Accumulates token/cost totals per model across a run. Thread-safe."""

    _lock: Lock = field(default_factory=Lock, repr=False)
    totals: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        with self._lock:
            row = self.totals.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0}
            )
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["calls"] += 1
            row["cost_usd"] += cost
        return cost

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(row["cost_usd"] for row in self.totals.values())

    def summary_lines(self) -> list[str]:
        with self._lock:
            lines = []
            total = 0.0
            for model, row in sorted(self.totals.items()):
                total += row["cost_usd"]
                lines.append(
                    f"  {model:30s} {row['calls']:5d} calls  "
                    f"{row['input_tokens']:>9,} in  {row['output_tokens']:>9,} out  "
                    f"${row['cost_usd']:.4f}"
                )
            # Compute the total from within this same lock — do NOT call the
            # `total_cost_usd` property here: it re-acquires `self._lock`, which
            # is a plain (non-reentrant) Lock, so it would deadlock the thread
            # against itself and hang the whole run right after "Cost:" prints.
            lines.append(f"  {'TOTAL':30s} {'':>5s}        "
                          f"{'':>9s}    {'':>9s}      ${total:.4f}")
            return lines

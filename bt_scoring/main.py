"""bt_scoring: pairwise LLM-as-judge over State of the Union address texts ->
a Bradley-Terry latent scale with standard errors, validated against party.

See README.md for the archetype (podcasts' political-orientation BT scale)
and the dimension being judged here. Pipeline:

  1. load each SOTU address's plain text (fetched by
     ../data_acquisition/fetch_sotu.py; no ASR). Every item carries a
     president and party label the judge never sees.
  2. sample pairwise matchups and judge the IDENTICAL pair set with every
     model in --judges via the LiteLLM proxy (judge.py) on some dimension
     (default: economic left/right), checkpointed to comparisons.jsonl keyed
     by (pair, judge, order) for per-judge resume.
  3. fit a separate Bradley-Terry scale with SEs per judge (bt_fit.py),
     validate each against party (point-biserial / Mann-Whitney) and year,
     and compare judges via a Spearman rank-correlation matrix of their scales.
  4. render Plotly figures + an HTML index (viz.py).

The judge is deliberately given no rubric for "economic left/right": how each
model resolves that contested axis -- and how much the judges then disagree --
is the point. Party is the ground-truth check on whatever scale emerges.

Run: uv run --package bt_scoring python bt_scoring/main.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from bt_fit import fit_bt_choix, fit_bt_logit, judge_rank_correlation, validate_scale
from judge import run_judging, sample_matchups
from viz import render_all

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "data_acquisition" / "data" / "sotu" / "manifest.jsonl"

# The dimension is written "left-pole <-> right-pole" and handed to the judge
# WITHOUT a rubric: how each model resolves an ambiguous, contested axis like
# "economic left/right" -- and how much the judges then disagree -- is part of
# what the branch surfaces. The party label (never shown to the judge) is the
# ground-truth check on whatever scale emerges.
DEFAULT_DIMENSION = (
    "economic orientation: left (more government intervention, spending, "
    "regulation, redistribution) <-> right (freer markets, lower taxes, "
    "less regulation)"
)

# Default judge roster: a diverse mix of families/tiers all present in judge.py's
# price table (so cost is accounted, not "n/a") -- two Gemini tiers, a GPT mini,
# and Claude Haiku + Opus. Add/remove models via --judges.
DEFAULT_JUDGES = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
    "gpt-5.4-mini-2026-03-17",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-8",
]


def load_items(manifest_path: Path, num_items: int) -> list[dict]:
    """Load SOTU manifest rows. Each row already carries president, party,
    year and a text_path pointing at the fetched speech text."""
    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if num_items > 0:
        items = items[:num_items]
    for item in items:
        item["year"] = float(item["year"]) if item.get("year") is not None else None
    return items


def build_scores_table(bt: pd.DataFrame, choix_est: pd.Series, items: list[dict]) -> pd.DataFrame:
    meta = pd.DataFrame(items).set_index("identifier")[["year", "president", "party", "title"]]
    scores = bt.join(meta).join(choix_est)
    scores["rank"] = scores["strength"].rank(ascending=False).astype(int)
    return scores.sort_values("rank")


def summarize_cost_by_judge(comparisons_path: Path) -> dict[str, dict]:
    """Cost/usage/status breakdown per judge model, keyed by the "model"
    field recorded in each comparisons.jsonl row."""
    per_judge: dict[str, dict] = {}
    if comparisons_path.exists():
        with comparisons_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                model = rec.get("model", "unknown")
                d = per_judge.setdefault(model, {
                    "n_ok": 0, "n_error": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "cost_usd_est": 0.0,
                })
                if rec.get("status") == "ok":
                    d["n_ok"] += 1
                else:
                    d["n_error"] += 1
                d["prompt_tokens"] += rec.get("usage", {}).get("prompt_tokens", 0)
                d["completion_tokens"] += rec.get("usage", {}).get("completion_tokens", 0)
                if rec.get("cost_usd_est"):
                    d["cost_usd_est"] += rec["cost_usd_est"]
    return per_judge


def summarize_cost_total(per_judge: dict[str, dict]) -> dict:
    total = {"n_ok": 0, "n_error": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd_est": 0.0}
    for d in per_judge.values():
        for k in total:
            total[k] += d[k]
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=BRANCH_DIR)
    p.add_argument("--num-items", type=int, default=0, help="0 = all items in the manifest")
    p.add_argument("--min-matchups", type=int, default=6, help="minimum comparisons per item")
    p.add_argument("--char-budget", type=int, default=0,
                   help="max characters of each speech sent to a judge (0 = full text)")
    p.add_argument(
        "--judges", type=str, nargs="+", default=DEFAULT_JUDGES,
        help="judge models; the IDENTICAL pair set is judged by every model for comparability",
    )
    p.add_argument("--dimension", type=str, default=DEFAULT_DIMENSION)
    p.add_argument("--concurrency", type=int, default=10, help="parallel judge calls (shared across judges)")
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=11111)
    p.add_argument("--skip-judge", action="store_true", help="reuse cached comparisons only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    comparisons_path = args.out_dir / "comparisons.jsonl"
    output_dir = args.out_dir / "output"
    scores_by_judge_dir = output_dir / "scores_by_judge"
    scores_by_judge_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.manifest, args.num_items)
    print(f"Loaded {len(items)} items from {args.manifest}", flush=True)

    # --- 1. load speech texts (no ASR: the fetcher already wrote plain text) ---
    t0 = time.time()
    texts: dict[str, str] = {}
    for item in items:
        p = Path(item["text_path"])
        if p.exists():
            texts[item["identifier"]] = p.read_text(encoding="utf-8")
    print(f"Loaded {len(texts)} speech texts ({time.time() - t0:.0f}s)", flush=True)

    usable_items = [it for it in items if it["identifier"] in texts]
    identifiers = [it["identifier"] for it in usable_items]
    if len(identifiers) < 4:
        raise SystemExit(f"Only {len(identifiers)} speech texts available; need >= 4 to fit a BT scale.")

    # --- 2. pairwise judging, identical pair set across every judge --------
    matchups = sample_matchups(identifiers, args.min_matchups, args.seed)
    print(f"Sampled {len(matchups)} matchups (min {args.min_matchups}/item, {len(identifiers)} items) "
          f"-- judged by {len(args.judges)} models: {args.judges}", flush=True)
    if not args.skip_judge:
        # DEFAULTS to OSU's LiteLLM proxy via LITELLM_URL/LITELLM_KEY (repo-root .env).
        # Standard OpenAI-compatible client: repoint those env vars for a different
        # proxy, or set base_url/api_key to a vendor's own OpenAI-compatible endpoint
        # to call the vendor directly.
        client = AsyncOpenAI(
            base_url=os.environ["LITELLM_URL"].rstrip("/") + "/v1",
            api_key=os.environ["LITELLM_KEY"],
        )
        asyncio.run(run_judging(
            client, matchups, args.judges, texts, args.dimension,
            comparisons_path, args.seed, concurrency=args.concurrency, timeout_s=args.timeout_s,
            max_chars=args.char_budget,
        ))

    cost_by_judge = summarize_cost_by_judge(comparisons_path)
    cost_total = summarize_cost_total(cost_by_judge)
    print("\n=== Judging cost/usage by judge ===", flush=True)
    for judge in args.judges:
        c = cost_by_judge.get(judge, {"n_ok": 0, "n_error": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd_est": 0.0})
        print(f"  {judge}: {c['n_ok']} ok, {c['n_error']} err, "
              f"{c['prompt_tokens'] + c['completion_tokens']} tokens, ~${c['cost_usd_est']:.4f} est.", flush=True)
    print(f"  TOTAL: {cost_total['n_ok']} ok, {cost_total['n_error']} err, "
          f"~${cost_total['cost_usd_est']:.4f} estimated across all judges", flush=True)

    # --- 3. fit a separate BT scale per judge + validate --------------------
    all_rows = []
    with comparisons_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                all_rows.append(rec)
    comparisons_all = pd.DataFrame(all_rows)

    strengths_by_judge: dict[str, pd.Series] = {}
    scores_by_judge: dict[str, pd.DataFrame] = {}
    year_corr_by_judge: dict[str, dict | None] = {}
    party_corr_by_judge: dict[str, dict | None] = {}

    for judge in args.judges:
        sub = comparisons_all[comparisons_all["model"] == judge][["item_i", "item_j", "winner"]]
        if len(sub) < 10:
            print(f"\n[skip] {judge}: only {len(sub)} 'ok' comparisons, need >= 10 to fit", flush=True)
            continue
        bt = fit_bt_logit(sub, identifiers)
        choix_est = fit_bt_choix(sub, identifiers)
        scores = build_scores_table(bt, choix_est, usable_items)
        judge_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", judge)
        scores.to_csv(scores_by_judge_dir / f"{judge_slug}.csv")

        scores_by_judge[judge] = scores
        strengths_by_judge[judge] = scores["strength"]

        validation = validate_scale(scores.reset_index().rename(columns={"index": "identifier"}))
        year_corr_by_judge[judge] = validation["spearman_strength_vs_year"]
        party_corr_by_judge[judge] = validation["strength_vs_party"]

        rank_corr_choix = scores["strength"].corr(scores["choix_strength"], method="spearman")
        print(f"\n=== Bradley-Terry scale: {judge} ({len(sub)} comparisons) ===", flush=True)
        print(scores[["strength", "se", "n_matchups", "rank", "year", "president", "party"]].to_string(), flush=True)
        print(f"Spearman(logit strength, choix strength) = {rank_corr_choix:.3f} "
              f"(regularized={bool(bt['regularized'].iloc[0])})", flush=True)
        sp = party_corr_by_judge[judge]
        if sp:
            print(f"Party alignment (right=high): point-biserial r={sp['pointbiserial_r']:.3f} "
                  f"(p={sp['pointbiserial_p']:.4f}); Mann-Whitney p={sp['mannwhitney_p']:.4f}; "
                  f"mean R={sp['mean_republican']:.3f} vs D={sp['mean_democrat']:.3f}", flush=True)
        else:
            print("Party alignment: not enough items in both parties", flush=True)
        sy = year_corr_by_judge[judge]
        if sy:
            print(f"Spearman(strength, year): rho={sy['rho']:.3f}, p={sy['p']:.4f}, n={sy['n']}", flush=True)

    if not scores_by_judge:
        raise SystemExit("No judge produced enough comparisons to fit a BT scale.")

    # Long-format combined table across judges, for viz + spot-checking.
    combined = pd.concat(
        [df.reset_index().rename(columns={"index": "identifier"}).assign(judge=judge)
         for judge, df in scores_by_judge.items()],
        ignore_index=True,
    )
    combined.to_csv(output_dir / "scores_all_judges.csv", index=False)

    # --- 4. judge x judge agreement -----------------------------------------
    rank_corr_matrix = judge_rank_correlation(strengths_by_judge)
    rank_corr_matrix.to_csv(output_dir / "judge_rank_correlation.csv")
    print("\n=== Judge x judge Spearman rank-correlation matrix (BT scales) ===", flush=True)
    print(rank_corr_matrix.round(3).to_string(), flush=True)

    print("\n=== Party alignment per judge (does the scale put Republicans to the 'right'?) ===", flush=True)
    for judge, sp in party_corr_by_judge.items():
        if sp:
            print(f"  {judge}: point-biserial r={sp['pointbiserial_r']:.3f} (p={sp['pointbiserial_p']:.4f}), "
                  f"mean R={sp['mean_republican']:.3f} vs D={sp['mean_democrat']:.3f}", flush=True)
        else:
            print(f"  {judge}: not enough items in both parties", flush=True)

    # --- 5. visualizations ---------------------------------------------------
    viz_summary = render_all(
        output_dir=output_dir,
        scores_by_judge=scores_by_judge,
        rank_corr_matrix=rank_corr_matrix,
        cost_by_judge=cost_by_judge,
        party_corr_by_judge=party_corr_by_judge,
        year_corr_by_judge=year_corr_by_judge,
        dimension=args.dimension,
    )
    print(f"\nViz index: {viz_summary['index_html']}", flush=True)

    print(f"\nDimension judged: {args.dimension}", flush=True)
    print(f"Wrote: {output_dir}/, {comparisons_path}", flush=True)


if __name__ == "__main__":
    main()

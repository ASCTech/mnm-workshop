"""bt_scoring: pairwise LLM-as-judge over presidential-audio transcripts ->
a Bradley-Terry latent scale with standard errors, validated against year.

See README.md for the archetype (podcasts' political-orientation BT scale)
and the dimension being judged here. Pipeline:

  1. transcribe a bounded leading segment of each clip with Granite-Speech
     (transcribe.py), cached to transcripts/<id>.txt.
  2. sample pairwise matchups and judge the IDENTICAL pair set with every
     model in --judges via the LiteLLM proxy (judge.py), checkpointed to
     comparisons.jsonl keyed by (pair, judge, order) for per-judge resume.
  3. fit a separate Bradley-Terry scale with SEs per judge (bt_fit.py),
     validate each against year/speaker, and compare judges via a Spearman
     rank-correlation matrix of their BT scales.
  4. render Plotly figures + an HTML index (viz.py).

Run: uv run --no-sync --package bt_scoring python bt_scoring/main.py
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
from transcribe import transcribe_manifest
from viz import render_all

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "data_acquisition" / "data" / "presidential_audio" / "manifest.jsonl"

DEFAULT_DIMENSION = (
    "rhetorical register: measured/formal <-> emotionally intense/populist"
)

# Default judge roster for the Round-2 multi-judge comparison: a diverse mix
# of families/tiers plus one frontier contrast, all confirmed against the
# LiteLLM proxy. kimi-k2.5 (slow, ~26s/call) and deepseek-r1 (dropped from the
# workshop) are deliberately left out of the default but can be added via
# --judges.
DEFAULT_JUDGES = [
    "gemini-3.1-flash-lite",
    "gpt-5.4-mini-2026-03-17",
    "claude-haiku-4-5-20251001",
    "llama3-3-70b-instruct",
    "claude-opus-4-8",
]

_NIXON_DATE_RE = re.compile(r"(\d{4})$")
_LBJ_ID_RE = re.compile(r"^lbj(\d{2})\d{4}$")


def derive_year(item: dict) -> float | None:
    """manifest `year` is null for the nixon/lbj items (their `date` field is
    a placeholder). Recover a year from the identifier/title where possible."""
    if item.get("year"):
        return float(item["year"])
    identifier = item["identifier"]
    if identifier.startswith("nixon"):
        m = _NIXON_DATE_RE.search(item["title"])
        if m:
            return float(m.group(1))
    if identifier.startswith("lbj"):
        m = _LBJ_ID_RE.match(identifier)
        if m:
            return 1900.0 + int(m.group(1))
    return None


def derive_speaker(identifier: str) -> str:
    if identifier.startswith("fdr"):
        return "FDR"
    if identifier.startswith("nixon"):
        return "Nixon"
    if identifier.startswith("lbj"):
        return "LBJ"
    return "Unknown"


def load_items(manifest_path: Path, num_items: int) -> list[dict]:
    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if num_items > 0:
        items = items[:num_items]
    for item in items:
        item["year"] = derive_year(item)
        item["speaker"] = derive_speaker(item["identifier"])
    return items


def build_scores_table(bt: pd.DataFrame, choix_est: pd.Series, items: list[dict]) -> pd.DataFrame:
    meta = pd.DataFrame(items).set_index("identifier")[["year", "speaker", "title"]]
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
    p.add_argument("--segment-seconds", type=float, default=90.0, help="leading audio segment to transcribe")
    p.add_argument("--chunk-seconds", type=float, default=30.0, help="ASR chunk window within the segment")
    p.add_argument("--min-matchups", type=int, default=8, help="minimum comparisons per item")
    p.add_argument(
        "--judges", type=str, nargs="+", default=DEFAULT_JUDGES,
        help="judge models; the IDENTICAL pair set is judged by every model for comparability",
    )
    p.add_argument("--dimension", type=str, default=DEFAULT_DIMENSION)
    p.add_argument("--concurrency", type=int, default=10, help="parallel judge calls (shared across judges)")
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=11111)
    p.add_argument("--skip-transcribe", action="store_true", help="reuse cached transcripts only")
    p.add_argument("--skip-judge", action="store_true", help="reuse cached comparisons only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    transcripts_dir = args.out_dir / "transcripts"
    comparisons_path = args.out_dir / "comparisons.jsonl"
    output_dir = args.out_dir / "output"
    scores_by_judge_dir = output_dir / "scores_by_judge"
    scores_by_judge_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.manifest, args.num_items)
    print(f"Loaded {len(items)} items from {args.manifest}", flush=True)

    # --- 1. transcribe -----------------------------------------------------
    t0 = time.time()
    if args.skip_transcribe:
        transcripts = {}
        for item in items:
            p = transcripts_dir / f"{item['identifier']}.txt"
            if p.exists():
                transcripts[item["identifier"]] = p.read_text(encoding="utf-8")
        transcribe_failures = {}
    else:
        transcripts, transcribe_failures = transcribe_manifest(
            items, transcripts_dir, args.segment_seconds, args.chunk_seconds,
        )
    print(f"Transcription stage: {len(transcripts)} ok, {len(transcribe_failures)} failed "
          f"({time.time() - t0:.0f}s)", flush=True)

    usable_items = [it for it in items if it["identifier"] in transcripts]
    identifiers = [it["identifier"] for it in usable_items]
    if len(identifiers) < 4:
        raise SystemExit(f"Only {len(identifiers)} transcripts available; need >= 4 to fit a BT scale.")

    # --- 2. pairwise judging, identical pair set across every judge --------
    matchups = sample_matchups(identifiers, args.min_matchups, args.seed)
    print(f"Sampled {len(matchups)} matchups (min {args.min_matchups}/item, {len(identifiers)} items) "
          f"-- judged by {len(args.judges)} models: {args.judges}", flush=True)
    if not args.skip_judge:
        client = AsyncOpenAI(
            base_url=os.environ["LITELLM_URL"].rstrip("/") + "/v1",
            api_key=os.environ["LITELLM_KEY"],
        )
        asyncio.run(run_judging(
            client, matchups, args.judges, transcripts, args.dimension,
            comparisons_path, args.seed, concurrency=args.concurrency, timeout_s=args.timeout_s,
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

        rank_corr_choix = scores["strength"].corr(scores["choix_strength"], method="spearman")
        print(f"\n=== Bradley-Terry scale: {judge} ({len(sub)} comparisons) ===", flush=True)
        print(scores[["strength", "se", "n_matchups", "rank", "year", "speaker"]].to_string(), flush=True)
        print(f"Spearman(logit strength, choix strength) = {rank_corr_choix:.3f} "
              f"(regularized={bool(bt['regularized'].iloc[0])})", flush=True)
        sy = year_corr_by_judge[judge]
        if sy:
            print(f"Spearman(strength, year): rho={sy['rho']:.3f}, p={sy['p']:.4f}, n={sy['n']}", flush=True)
        else:
            print("Spearman(strength, year): not enough items with a known year", flush=True)

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

    print("\n=== BT-vs-year Spearman correlation per judge ===", flush=True)
    for judge, sy in year_corr_by_judge.items():
        if sy:
            print(f"  {judge}: rho={sy['rho']:.3f}, p={sy['p']:.4f}, n={sy['n']}", flush=True)
        else:
            print(f"  {judge}: not enough items with a known year", flush=True)

    # --- 5. visualizations ---------------------------------------------------
    viz_summary = render_all(
        output_dir=output_dir,
        scores_by_judge=scores_by_judge,
        rank_corr_matrix=rank_corr_matrix,
        cost_by_judge=cost_by_judge,
        year_corr_by_judge=year_corr_by_judge,
        dimension=args.dimension,
    )
    print(f"\nViz index: {viz_summary['index_html']}", flush=True)

    print(f"\nDimension judged: {args.dimension}", flush=True)
    print(f"Wrote: {output_dir}/, {comparisons_path}, {transcripts_dir}/", flush=True)
    if transcribe_failures:
        print(f"Transcription failures: {transcribe_failures}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # avoid HF/torch background-thread shutdown hangs

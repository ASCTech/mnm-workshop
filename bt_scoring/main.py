"""bt_scoring: pairwise LLM-as-judge over presidential-audio transcripts ->
a Bradley-Terry latent scale with standard errors, validated against year.

See README.md for the archetype (podcasts' political-orientation BT scale)
and the dimension being judged here. Pipeline:

  1. transcribe a bounded leading segment of each clip with Granite-Speech
     (transcribe.py), cached to transcripts/<id>.txt.
  2. sample pairwise matchups and judge them with a cheap LLM via the LiteLLM
     proxy (judge.py), checkpointed to comparisons.jsonl.
  3. fit a Bradley-Terry scale with SEs (bt_fit.py) and validate it against
     year and speaker.

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

from bt_fit import fit_bt_choix, fit_bt_logit, validate_scale
from judge import run_judging, sample_matchups
from transcribe import transcribe_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "data_acquisition" / "data" / "presidential_audio" / "manifest.jsonl"

DEFAULT_DIMENSION = (
    "rhetorical register: measured/formal <-> emotionally intense/populist"
)

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


def summarize_cost(comparisons_path: Path) -> dict:
    n_ok, n_err = 0, 0
    prompt_tok, completion_tok, cost = 0, 0, 0.0
    if comparisons_path.exists():
        with comparisons_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                prompt_tok += rec.get("usage", {}).get("prompt_tokens", 0)
                completion_tok += rec.get("usage", {}).get("completion_tokens", 0)
                if rec.get("cost_usd_est"):
                    cost += rec["cost_usd_est"]
    return {
        "n_ok": n_ok, "n_error": n_err,
        "prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
        "cost_usd_est": cost,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=BRANCH_DIR)
    p.add_argument("--num-items", type=int, default=0, help="0 = all items in the manifest")
    p.add_argument("--segment-seconds", type=float, default=90.0, help="leading audio segment to transcribe")
    p.add_argument("--chunk-seconds", type=float, default=30.0, help="ASR chunk window within the segment")
    p.add_argument("--min-matchups", type=int, default=8, help="minimum comparisons per item")
    p.add_argument("--judge-model", type=str, default="gemini-2.5-flash-lite")
    p.add_argument("--dimension", type=str, default=DEFAULT_DIMENSION)
    p.add_argument("--concurrency", type=int, default=6, help="parallel judge calls")
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
    scores_path = args.out_dir / "scores.csv"

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

    # --- 2. pairwise judging ------------------------------------------------
    if not args.skip_judge:
        matchups = sample_matchups(identifiers, args.min_matchups, args.seed)
        print(f"Sampled {len(matchups)} matchups (min {args.min_matchups}/item, {len(identifiers)} items)", flush=True)
        client = AsyncOpenAI(
            base_url=os.environ["LITELLM_URL"].rstrip("/") + "/v1",
            api_key=os.environ["LITELLM_KEY"],
        )
        asyncio.run(run_judging(
            client, matchups, transcripts, args.dimension, args.judge_model,
            comparisons_path, args.seed, concurrency=args.concurrency, timeout_s=args.timeout_s,
        ))

    cost = summarize_cost(comparisons_path)
    print(f"Judging: {cost['n_ok']} ok, {cost['n_error']} errors, "
          f"{cost['prompt_tokens'] + cost['completion_tokens']} tokens, "
          f"~${cost['cost_usd_est']:.4f} estimated ({args.judge_model})", flush=True)

    # --- 3. fit + validate ---------------------------------------------------
    rows = []
    with comparisons_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                rows.append(rec)
    comparisons = pd.DataFrame(rows)[["item_i", "item_j", "winner"]]

    bt = fit_bt_logit(comparisons, identifiers)
    choix_est = fit_bt_choix(comparisons, identifiers)
    scores = build_scores_table(bt, choix_est, usable_items)
    scores.to_csv(scores_path)

    validation = validate_scale(scores.reset_index().rename(columns={"index": "identifier"}))

    print("\n=== Bradley-Terry scale ===", flush=True)
    print(scores[["strength", "se", "n_matchups", "rank", "year", "speaker"]].to_string(), flush=True)

    print("\n=== Cross-check: statsmodels-logit vs choix rank correlation ===", flush=True)
    rank_corr = scores["strength"].corr(scores["choix_strength"], method="spearman")
    print(f"Spearman(logit strength, choix strength) = {rank_corr:.3f}", flush=True)

    print("\n=== Validation ===", flush=True)
    sy = validation["spearman_strength_vs_year"]
    if sy:
        print(f"Spearman(strength, year): rho={sy['rho']:.3f}, p={sy['p']:.4f}, n={sy['n']}", flush=True)
    else:
        print("Spearman(strength, year): not enough items with a known year", flush=True)
    print("\nMean strength by speaker:", flush=True)
    print(validation["by_speaker"].to_string(), flush=True)

    print(f"\nDimension judged: {args.dimension}", flush=True)
    print(f"Wrote: {scores_path}, {comparisons_path}, {transcripts_dir}/", flush=True)
    if transcribe_failures:
        print(f"Transcription failures: {transcribe_failures}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)  # avoid HF/torch background-thread shutdown hangs

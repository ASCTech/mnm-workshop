"""Multi-model codebook coding of the PMC-OA demo corpus + agreement scoring.

Pipeline: read the pmc_oa manifest -> code each article with several LLMs (M
runs each, temperature > 0) through the OSU LiteLLM proxy -> persist every
(article, model, run) result to a resumable results.jsonl -> score inter-model
agreement and intra-model consistency -> emit CSVs, a printed summary, and an
optional HTML report.

Run: uv run --package corpus_coding python corpus_coding/main.py [args]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

import llm
import scoring
from schema import CodedRecord, QUESTIONS, Usage

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data_acquisition" / "data" / "pmc_oa" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                     help="pmc_oa manifest.jsonl (run data_acquisition/fetch_pmc_oa.py first)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-articles", type=int, default=12,
                     help="Number of articles to code, taken from the front of the manifest")
    ap.add_argument("--models", nargs="*", default=llm.DEFAULT_MODELS,
                     help="LiteLLM proxy model ids to compare")
    ap.add_argument("--n-runs", type=int, default=2, help="Repeated runs per model-article pair")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--char-budget", type=int, default=8000,
                     help="Max characters of article text sent to the model")
    ap.add_argument("--max-workers", type=int, default=6, help="Parallel LLM calls")
    ap.add_argument("--html", dest="html", action=argparse.BooleanOptionalAction, default=True,
                     help="Also write an HTML summary report")
    return ap.parse_args()


def load_articles(manifest_path: Path, n_articles: int, char_budget: int) -> list[tuple[str, str]]:
    """Read the first n_articles manifest rows and truncate their .txt to char_budget."""
    articles: list[tuple[str, str]] = []
    with manifest_path.open() as f:
        for line in f:
            if len(articles) >= n_articles:
                break
            row = json.loads(line)
            pmcid = row["pmcid"]
            txt_path = Path(row["files"]["txt"])
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                print(f"  skipping {pmcid}: cannot read {txt_path} ({e})", flush=True)
                continue
            if len(text) > char_budget:
                text = text[:char_budget] + "\n...[truncated]"
            articles.append((pmcid, text))
    return articles


def load_existing_records(results_path: Path) -> list[CodedRecord]:
    if not results_path.exists():
        return []
    records = []
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(CodedRecord.model_validate_json(line))
    return records


def run_pending(
    client: OpenAI,
    articles: list[tuple[str, str]],
    models: list[str],
    n_runs: int,
    temperature: float,
    results_path: Path,
    done_keys: set[tuple[str, str, int]],
    max_workers: int,
) -> None:
    """Run every (article, model, run) not already in done_keys; append as completed."""
    tasks = [
        (pmcid, text, model, run)
        for pmcid, text in articles
        for model in models
        for run in range(1, n_runs + 1)
        if (pmcid, model, run) not in done_keys
    ]
    if not tasks:
        print("Nothing to do: all requested (article, model, run) triples are already coded.", flush=True)
        return

    print(f"Running {len(tasks)} pending LLM call(s) with {len(models)} model(s), "
          f"{n_runs} run(s) each, over {len(articles)} article(s)...", flush=True)

    write_lock = threading.Lock()
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def _do_one(pmcid: str, text: str, model: str, run: int) -> CodedRecord:
        result = llm.call_model(client, model, text, temperature)
        cost = llm.estimate_cost(model, result.prompt_tokens, result.completion_tokens)
        return CodedRecord(
            pmcid=pmcid,
            model=model,
            run=run,
            response=result.response,
            usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens),
            cost_usd=cost,
            latency_s=result.latency_s,
            error=result.error,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), max_workers))) as pool:
        futures = {pool.submit(_do_one, *task): task for task in tasks}
        for future in as_completed(futures):
            pmcid, _text, model, run = futures[future]
            try:
                record = future.result()
            except Exception as e:  # a bug in our own code, not an API error: still don't abort the batch
                record = CodedRecord(
                    pmcid=pmcid, model=model, run=run, error=f"unexpected: {e}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            with write_lock:
                with results_path.open("a") as f:
                    f.write(record.model_dump_json() + "\n")
            done += 1
            status = "ok" if record.error is None else f"ERROR: {record.error[:80]}"
            print(f"  [{done}/{len(tasks)}] {pmcid} {model} run={run}: "
                  f"${record.cost_usd:.5f} {record.latency_s:.1f}s {status}", flush=True)


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty:
        print("(no data)")
    else:
        print(df.to_string())


def write_html_report(
    output_dir: Path,
    consistency_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    cost_by_model: pd.DataFrame,
    total_cost: float,
    n_articles: int,
    n_errors: int,
) -> None:
    template = Template("""
<!doctype html>
<html><head><meta charset="utf-8"><title>corpus_coding report</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; }
table { border-collapse: collapse; margin-bottom: 2rem; }
th, td { border: 1px solid #ccc; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
caption { font-weight: bold; text-align: left; margin-bottom: .5rem; }
</style></head><body>
<h1>corpus_coding: multi-model codebook coding</h1>
<p>{{ n_articles }} article(s) coded; {{ n_errors }} failed coding attempt(s) excluded from scoring.
Total cost: ${{ "%.4f"|format(total_cost) }}.</p>

<h2>Inter-model agreement (per question)</h2>
{{ agreement_html | safe }}

<h2>Intra-model consistency (per model, % of runs matching that model's own majority)</h2>
{{ consistency_html | safe }}

<h2>Cost by model</h2>
{{ cost_html | safe }}
</body></html>
""")
    html = template.render(
        n_articles=n_articles,
        n_errors=n_errors,
        total_cost=total_cost,
        agreement_html=agreement_df.to_html(na_rep="-"),
        consistency_html=consistency_df.to_html(na_rep="-"),
        cost_html=cost_by_model.to_html(na_rep="-"),
    )
    (output_dir / "report.html").write_text(html)


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(
        base_url=os.environ["LITELLM_URL"].rstrip("/") + "/v1",
        api_key=os.environ["LITELLM_KEY"],
    )

    if not args.manifest.exists():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run: uv run --package data_acquisition python data_acquisition/fetch_pmc_oa.py --limit 50"
        )

    articles = load_articles(args.manifest, args.n_articles, args.char_budget)
    if not articles:
        raise SystemExit("No articles could be read from the manifest.")
    print(f"Loaded {len(articles)} article(s) from {args.manifest}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"

    existing = load_existing_records(results_path)
    done_keys = {(r.pmcid, r.model, r.run) for r in existing}

    run_pending(
        client=client,
        articles=articles,
        models=args.models,
        n_runs=args.n_runs,
        temperature=args.temperature,
        results_path=results_path,
        done_keys=done_keys,
        max_workers=args.max_workers,
    )

    # Re-read everything (old + new) and restrict to this run's scope, so a
    # rerun with a bigger --n-articles/--n-runs still scores only what was asked.
    pmcids = [p for p, _ in articles]
    all_records = load_existing_records(results_path)
    in_scope = [
        r for r in all_records
        if r.pmcid in pmcids and r.model in args.models and 1 <= r.run <= args.n_runs
    ]
    ok_records = [r for r in in_scope if r.response is not None]
    err_records = [r for r in in_scope if r.error is not None]

    print(f"\n{len(ok_records)} successful coding(s), {len(err_records)} error(s), "
          f"out of {len(in_scope)} in scope.", flush=True)
    if err_records:
        for r in err_records[:10]:
            print(f"  ERROR {r.pmcid} {r.model} run={r.run}: {r.error}", flush=True)

    # --- cost accounting ---
    cost_rows = {}
    for r in in_scope:
        agg = cost_rows.setdefault(r.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        agg["calls"] += 1
        agg["prompt_tokens"] += r.usage.prompt_tokens
        agg["completion_tokens"] += r.usage.completion_tokens
        agg["cost_usd"] += r.cost_usd
    cost_df = pd.DataFrame.from_dict(cost_rows, orient="index").sort_index()
    total_cost = float(cost_df["cost_usd"].sum()) if not cost_df.empty else 0.0

    # --- scoring ---
    groups = scoring.group_by_model_article(ok_records)
    consistency_df = scoring.intra_model_consistency(groups)
    majorities = scoring.model_majorities(groups)
    agreement_df = scoring.inter_model_agreement(majorities, models=args.models, pmcids=pmcids)

    consistency_df.to_csv(args.output_dir / "consistency.csv")
    agreement_df.to_csv(args.output_dir / "agreement.csv")
    cost_df.to_csv(args.output_dir / "cost_by_model.csv")

    print_table("Inter-model agreement (per question)", agreement_df)
    print_table("Intra-model consistency (per model, % of runs matching majority)", consistency_df)
    print_table("Cost by model", cost_df)
    print(f"\nTotal cost: ${total_cost:.4f}  ({len(in_scope)} calls, "
          f"{cost_df['prompt_tokens'].sum() if not cost_df.empty else 0} prompt + "
          f"{cost_df['completion_tokens'].sum() if not cost_df.empty else 0} completion tokens)")

    if args.html:
        write_html_report(
            args.output_dir, consistency_df, agreement_df, cost_df, total_cost,
            n_articles=len(articles), n_errors=len(err_records),
        )
        print(f"\nWrote {args.output_dir / 'report.html'}")

    print(f"\nOutputs in {args.output_dir}/:")
    print("  - results.jsonl   (every article/model/run, resumable)")
    print("  - agreement.csv   (per-question inter-model agreement)")
    print("  - consistency.csv (per-model intra-model consistency)")
    print("  - cost_by_model.csv")
    if args.html:
        print("  - report.html")


if __name__ == "__main__":
    main()

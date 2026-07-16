"""structure_analysis: extract structured fields, embed, cluster into a topic
hierarchy, validate against ground-truth categories.

    uv run --package structure_analysis python structure_analysis/main.py

Pipeline (see README.md for the archetype this stands in for):
  1. EXTRACT  — LLM call per abstract -> AbstractFields JSON (schema.py),
               checkpointed to output/<source>/extractions.jsonl.
  2. EMBED    — join(domain, methods, techniques, contribution) -> embedding
               (local sentence-transformers all-MiniLM-L6-v2 by default, or
               the proxy's titan-embed-text-v2:0 via --embed-backend titan),
               cached to output/<source>/embeddings_<backend>.npy.
  3. CLUSTER  — BERTopic (UMAP + HDBSCAN) on precomputed embeddings.
  4. VALIDATE — ARI / NMI / homogeneity / completeness / V-measure of the
               recovered topics against the held-out ground-truth label
               (arXiv primary_category / NSF fundProgramName).
  5. VIZ      — interactive Plotly figures + an HTML index (viz.py), under
               output/<source>/viz/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pin numba/OpenMP to a single, fork-safe thread BEFORE importing anything that
# pulls in numba (UMAP, via cluster). Without this the BERTopic UMAP fit
# intermittently deadlocks at exit/compile (main thread parked in sigsuspend) on
# some machines — fatal for a workshop whose whole point is reliable re-runs.
# `workqueue` is numba's fork-safe threading layer; single-threaded is plenty for
# this corpus size and makes runs deterministic. Must be set before import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster import (  # noqa: E402
    build_embedding_text,
    dominant_label_per_topic,
    embed_texts,
    fit_topics,
    topic_label_crosstab,
    topic_table,
    validate,
)
from data import load_corpus  # noqa: E402
from extract import load_extractions, run_extraction  # noqa: E402
from llm import CostLedger  # noqa: E402
from viz import render_all  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["arxiv", "nsf"], default="arxiv")
    p.add_argument("--limit", type=int, default=400, help="corpus size (default 400)")
    p.add_argument(
        "--extract-model",
        default="gemini-3.1-flash-lite",
        help="chat model for stage-1 extraction. Other roster options: "
             "gpt-5.4-mini-2026-03-17, claude-haiku-4-5-20251001, llama3-3-70b-instruct.",
    )
    p.add_argument(
        "--embed-backend",
        choices=["minilm", "titan"],
        default="minilm",
        help="'minilm' (default): local sentence-transformers all-MiniLM-L6-v2, "
             "384-dim, no network call, no cost — anecdotally clusters more cleanly "
             "for this short structured-field text. 'titan': proxy embedding model "
             "titan-embed-text-v2:0, 1024-dim, costed.",
    )
    p.add_argument(
        "--embed-model",
        default=None,
        help="override the embed model name for the chosen --embed-backend "
             "(defaults: all-MiniLM-L6-v2 for minilm, titan-embed-text-v2:0 for titan)",
    )
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--n-components", type=int, default=5, help="UMAP n_components")
    p.add_argument("--label-level", choices=["top", "full"], default="full",
                    help="ground-truth label granularity for validation. 'top' collapses "
                         "arXiv categories to their top-level part (e.g. cs.CV -> cs); with "
                         "this corpus's cs-heavy mix that lumps ~85%% of docs into one label, "
                         "so 'full' (the un-collapsed category) is the more informative default.")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="defaults to structure_analysis/output/<source>/")
    p.add_argument("--max-workers", type=int, default=8, help="extraction concurrency")
    p.add_argument("--force-embed", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="retry extractions that previously errored")
    p.add_argument("--top-k-labels", type=int, default=12, help="labels shown individually in the crosstab")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (THIS_DIR / "output" / args.source)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = CostLedger()

    print(f"=== structure_analysis: source={args.source} n={args.limit} ===")
    docs = load_corpus(args.source, args.limit)
    print(f"Loaded {len(docs)} docs from {args.source}")

    # --- Stage 1: extract ---
    extractions_path = out_dir / "extractions.jsonl"
    run_extraction(
        docs,
        model=args.extract_model,
        out_path=extractions_path,
        ledger=ledger,
        max_workers=args.max_workers,
        retry_failed=args.retry_failed,
    )
    extractions = load_extractions(extractions_path)

    usable = []
    for doc in docs:
        rec = extractions.get(doc["doc_id"])
        if rec and rec.get("parsed"):
            usable.append({**doc, "parsed": rec["parsed"]})
    n_failed = len(docs) - len(usable)
    print(f"Stage 1 done: {len(usable)}/{len(docs)} usable extractions ({n_failed} failed/skipped)")

    # --- Stage 2: embed ---
    embed_texts_list = [build_embedding_text(d["parsed"]) for d in usable]
    doc_ids = [d["doc_id"] for d in usable]
    embed_model = args.embed_model or (
        "all-MiniLM-L6-v2" if args.embed_backend == "minilm" else "titan-embed-text-v2:0"
    )
    # Backend/model-keyed cache filenames: a 384-dim MiniLM cache and a 1024-dim
    # Titan cache must never collide or be silently loaded into the wrong shape.
    npy_path = out_dir / f"embeddings_{args.embed_backend}.npy"
    index_path = out_dir / f"embeddings_{args.embed_backend}_index.json"
    embeddings = embed_texts(
        doc_ids,
        embed_texts_list,
        backend=args.embed_backend,
        model=embed_model,
        npy_path=npy_path,
        index_path=index_path,
        ledger=ledger,
        force=args.force_embed,
    )

    # --- Stage 3: cluster ---
    print(f"Fitting BERTopic (min_cluster_size={args.min_cluster_size}, n={len(usable)})...")
    model, topics = fit_topics(
        embed_texts_list, embeddings, args.min_cluster_size, args.n_neighbors, args.n_components
    )

    ttable = topic_table(model)
    ttable_path = out_dir / "topics.csv"
    ttable.to_csv(ttable_path, index=False)

    labels = [d["label_top" if args.label_level == "top" else "label_full"] for d in usable]

    metrics = validate(topics, labels)
    metrics_path = out_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    crosstab = topic_label_crosstab(topics, labels, top_k_labels=args.top_k_labels)
    crosstab_path = out_dir / "topic_label_crosstab.csv"
    crosstab.to_csv(crosstab_path)
    dominant = dominant_label_per_topic(crosstab)
    dominant_path = out_dir / "topic_dominant_label.csv"
    dominant.to_csv(dominant_path, index=False)

    # Representative docs + per-doc topic assignment, via BERTopic's own doc-info table
    doc_info = model.get_document_info(
        embed_texts_list,
        metadata={
            "doc_id": doc_ids,
            "title": [d["title"] for d in usable],
            "label": labels,
        },
    )
    doc_info_path = out_dir / "doc_topics.csv"
    doc_info.to_csv(doc_info_path, index=False)

    # Topic hierarchy
    hierarchy_df = None
    try:
        hierarchy_df = model.hierarchical_topics(embed_texts_list)
        hierarchy_df.to_csv(out_dir / "hierarchy.csv", index=False)
        tree_text = model.get_topic_tree(hierarchy_df)
        (out_dir / "topic_tree.txt").write_text(tree_text)
        n_merges = len(hierarchy_df)
    except Exception as e:
        print(f"  [warn] hierarchy failed: {type(e).__name__}: {e}")
        n_merges = 0

    ledger_path = out_dir / "cost_ledger.json"
    ledger_path.write_text(json.dumps(ledger.totals, indent=2))

    # --- Stage 5: viz ---
    print("Building visualizations...")
    viz_meta = {
        "source": args.source,
        "extract_model": args.extract_model,
        "embed_backend": args.embed_backend,
        "embed_model": embed_model,
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "min_cluster_size": args.min_cluster_size,
        "label_level": args.label_level,
        "total_cost_usd": ledger.total_cost_usd,
    }
    try:
        index_html_path = render_all(
            out_dir=out_dir,
            model=model,
            embeddings=embeddings,
            topics=topics,
            labels=labels,
            doc_ids=doc_ids,
            titles=[d["title"] for d in usable],
            metrics=metrics,
            crosstab=crosstab,
            hierarchy_df=hierarchy_df,
            meta=viz_meta,
        )
        print(f"  viz index: {index_html_path}")
    except Exception as e:
        print(f"  [warn] viz failed: {type(e).__name__}: {e}")
        index_html_path = None

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"source={args.source}  n_docs={len(usable)}  label_level={args.label_level}")
    print(f"topics: {metrics['n_topics']}  outliers: {metrics['n_outliers']} ({metrics['outlier_pct']:.1f}%)")
    print(
        f"ARI={metrics['adjusted_rand_index']:.3f}  NMI={metrics['normalized_mutual_info']:.3f}  "
        f"homogeneity={metrics['homogeneity']:.3f}  completeness={metrics['completeness']:.3f}  "
        f"v_measure={metrics['v_measure']:.3f}"
    )
    print(f"topic hierarchy: {n_merges} merges")
    print("\nTop topics by size:")
    for _, row in ttable[ttable["topic_id"] != -1].sort_values("size", ascending=False).head(10).iterrows():
        print(f"  topic {int(row['topic_id']):>3d}  n={int(row['size']):>4d}  {row['top_words']}")
    print("\nCost:")
    for line in ledger.summary_lines():
        print(line)
    print(f"\nOutputs written under {out_dir}/:")
    for p in [extractions_path, npy_path, index_path, ttable_path, metrics_path,
              crosstab_path, dominant_path, doc_info_path, ledger_path]:
        print(f"  {p.name}")
    if n_merges:
        print("  hierarchy.csv, topic_tree.txt")
    if index_html_path:
        print(f"  viz/{index_html_path.name} (+ viz/*.html figures)")


if __name__ == "__main__":
    # Note: a previous version force-exited with os._exit(0) here to dodge an
    # apparent "exit hang". The real cause was a self-deadlock in
    # CostLedger.summary_lines() (re-acquiring a non-reentrant Lock); with that
    # fixed, the interpreter shuts down normally and reaps its worker children,
    # so no hard exit is needed.
    main()

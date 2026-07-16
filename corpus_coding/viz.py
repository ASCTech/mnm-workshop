"""Plotly visualization: agreement/consistency/cost figures + an HTML index.

Renders five figures from the results of one main.py run into
``<output_dir>/viz/*.html`` (interactive, plotly-via-CDN) with a best-effort
static ``.svg`` export alongside each, plus a hand-built ``viz/index.html``
landing page linking everything. Mirrors the pattern used by the sibling
``structure_analysis`` (syllabus_pipeline) branch's viz.py/report.py.

Called from main.py after scoring; can also be re-run standalone against an
existing output dir without re-hitting the LLM proxy (see ``if __name__``).
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go

import scoring
from schema import CodedRecord, QUESTIONS


def _write(fig: go.Figure, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(str(path.with_suffix(".svg")), format="svg",
                         width=1300, height=max(500, fig.layout.height or 600))
    except Exception as e:
        print(f"  (SVG export skipped for {path.name}: {e})", flush=True)


def fig_agreement_bars(agreement_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: Cohen kappa / Krippendorff alpha / Fleiss kappa, per question."""
    questions = list(agreement_df.index)
    fig = go.Figure()
    for col, label, color in [
        ("cohen_kappa", "Cohen's κ (mean pairwise)", "#4C78A8"),
        ("krippendorff_alpha", "Krippendorff's α", "#F58518"),
        ("fleiss_kappa", "Fleiss' κ", "#54A24B"),
    ]:
        fig.add_trace(go.Bar(
            x=questions, y=agreement_df[col], name=label, marker=dict(color=color),
            hovertemplate=f"%{{x}}<br>{label}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title="Inter-model agreement per question (models as raters)",
        barmode="group",
        yaxis=dict(title="agreement coefficient", zeroline=True),
        xaxis=dict(title="question", tickangle=-30),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=60, r=30, t=90, b=120),
        height=550,
    )
    return fig


def _model_majorities_restricted(in_scope: list[CodedRecord], models: list[str]):
    ok = [r for r in in_scope if r.response is not None and r.model in models]
    groups = scoring.group_by_model_article(ok)
    return scoring.model_majorities(groups)


def fig_pairwise_heatmap(in_scope: list[CodedRecord], models: list[str]) -> go.Figure:
    """Model x model heatmap: mean %-agreement on majority answers, averaged
    over all codebook questions and over the articles each pair both coded."""
    majorities = _model_majorities_restricted(in_scope, models)
    pmcids_by_model: dict[str, set[str]] = defaultdict(set)
    for (m, p) in majorities:
        pmcids_by_model[m].add(p)

    n = len(models)
    z = [[float("nan")] * n for _ in range(n)]
    n_units_used = [[0] * n for _ in range(n)]
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            common = sorted(pmcids_by_model[m1] & pmcids_by_model[m2])
            n_units_used[i][j] = len(common)
            if not common:
                continue
            per_q_scores = []
            for q in QUESTIONS:
                matches = [majorities[(m1, p)][q] == majorities[(m2, p)][q] for p in common]
                per_q_scores.append(sum(matches) / len(matches))
            z[i][j] = 100.0 * sum(per_q_scores) / len(per_q_scores)

    text = [[f"{z[i][j]:.1f}%" if z[i][j] == z[i][j] else "—" for j in range(n)] for i in range(n)]
    hover = [[f"{models[i]} vs {models[j]}<br>agreement: {z[i][j]:.1f}%<br>"
              f"n articles: {n_units_used[i][j]}" if z[i][j] == z[i][j] else "no shared articles"
              for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z, x=models, y=models, text=text, texttemplate="%{text}",
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale="RdYlGn", zmin=0, zmax=100,
        colorbar=dict(title="% agree"),
    ))
    fig.update_layout(
        title="Model × model pairwise agreement (mean over questions, majority answers)",
        xaxis=dict(title="", tickangle=-30),
        yaxis=dict(title="", autorange="reversed"),
        plot_bgcolor="white",
        margin=dict(l=160, r=30, t=60, b=140),
        height=max(500, 90 * n),
    )
    return fig


def fig_answer_distributions(in_scope: list[CodedRecord], models: list[str]) -> go.Figure:
    """Stacked bar per model of raw answer-value counts (all runs), one
    question visible at a time via dropdown buttons."""
    ok = [r for r in in_scope if r.response is not None and r.model in models]
    palette = pcolors.qualitative.Set3 + pcolors.qualitative.Pastel

    # counts[question][model][value] = n
    counts: dict[str, dict[str, Counter]] = {q: {m: Counter() for m in models} for q in QUESTIONS}
    for r in ok:
        dumped = r.response.model_dump(mode="json")
        for q in QUESTIONS:
            v = scoring.normalize_value(q, dumped[q])
            counts[q][r.model][v] += 1

    fig = go.Figure()
    traces_per_question: dict[str, list[int]] = {}
    trace_idx = 0
    for q in QUESTIONS:
        values = sorted({v for m in models for v in counts[q][m]})
        idxs = []
        for vi, v in enumerate(values):
            y = [counts[q][m].get(v, 0) for m in models]
            fig.add_trace(go.Bar(
                x=models, y=y, name=str(v), marker=dict(color=palette[vi % len(palette)]),
                visible=(q == QUESTIONS[0]),
                hovertemplate=f"%{{x}}<br>{html.escape(q)}=%{{fullData.name}}: %{{y}}<extra></extra>",
            ))
            idxs.append(trace_idx)
            trace_idx += 1
        traces_per_question[q] = idxs

    buttons = []
    for q in QUESTIONS:
        visible = [False] * trace_idx
        for i in traces_per_question[q]:
            visible[i] = True
        buttons.append(dict(label=q, method="update",
                             args=[{"visible": visible}, {"title": f"Answer distribution — {q}"}]))

    fig.update_layout(
        title=f"Answer distribution — {QUESTIONS[0]}",
        barmode="stack",
        xaxis=dict(title="model", tickangle=-20),
        yaxis=dict(title="count (all runs)"),
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=110, b=140),
        height=600,
        updatemenus=[dict(type="dropdown", x=1.0, xanchor="right", y=1.18, buttons=buttons)],
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def fig_consistency(consistency_df: pd.DataFrame) -> go.Figure:
    """Bar of each model's overall intra-model consistency, plus a per-question heatmap."""
    if consistency_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Intra-model consistency (no data — need >=2 runs per model)")
        return fig
    models = list(consistency_df.index)
    fig = go.Figure(go.Bar(
        x=models, y=consistency_df["overall"], marker=dict(color="#B279A2"),
        text=[f"{v:.1f}%" for v in consistency_df["overall"]], textposition="outside",
        hovertemplate="%{x}<br>overall consistency: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Intra-model consistency (% of a model's own runs matching its majority answer)",
        yaxis=dict(title="% consistent", range=[0, 105]),
        xaxis=dict(title="model", tickangle=-20),
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=60, b=140),
        height=550,
    )
    return fig


def fig_consistency_heatmap(consistency_df: pd.DataFrame) -> go.Figure:
    if consistency_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Intra-model consistency per question (no data)")
        return fig
    per_q = consistency_df[QUESTIONS]
    fig = go.Figure(go.Heatmap(
        z=per_q.values, x=list(per_q.columns), y=list(per_q.index),
        colorscale="RdYlGn", zmin=0, zmax=100,
        texttemplate="%{z:.0f}", colorbar=dict(title="% consistent"),
        hovertemplate="%{y} / %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Intra-model consistency per question",
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=30, t=60, b=160),
        height=max(400, 70 * len(per_q.index)),
    )
    return fig


def fig_cost(cost_df: pd.DataFrame) -> go.Figure:
    if cost_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Cost by model (no data)")
        return fig
    models = list(cost_df.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=cost_df["cost_usd"], name="cost (USD)", marker=dict(color="#E45756"),
        text=[f"${v:.4f}" for v in cost_df["cost_usd"]], textposition="outside",
        hovertemplate="%{x}<br>$%{y:.5f} (%{customdata} calls)<extra></extra>",
        customdata=cost_df["calls"],
    ))
    fig.update_layout(
        title="Total cost by model",
        yaxis=dict(title="USD"),
        xaxis=dict(title="model", tickangle=-20),
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=60, b=140),
        height=500,
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=models, y=cost_df["prompt_tokens"], name="prompt tokens", marker=dict(color="#4C78A8")))
    fig2.add_trace(go.Bar(x=models, y=cost_df["completion_tokens"], name="completion tokens", marker=dict(color="#F58518")))
    fig2.update_layout(
        title="Tokens by model (prompt + completion, stacked)",
        barmode="stack",
        yaxis=dict(title="tokens"),
        xaxis=dict(title="model", tickangle=-20),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=60, r=30, t=90, b=140),
        height=500,
    )
    return fig, fig2


def write_index_html(viz_dir: Path, summary: dict) -> Path:
    def row(name: str, title: str) -> str:
        html_ok = (viz_dir / f"{name}.html").exists()
        svg_ok = (viz_dir / f"{name}.svg").exists()
        html_link = f"<a href='{name}.html'>open</a>" if html_ok else "—"
        svg_link = f"<a href='{name}.svg'>svg</a>" if svg_ok else "—"
        return f"<tr><td>{html.escape(title)}</td><td>{html_link}</td><td>{svg_link}</td></tr>"

    rows = "".join([
        row("agreement_bars", "Inter-model agreement per question (Cohen κ / Krippendorff α / Fleiss κ)"),
        row("pairwise_heatmap", "Model × model pairwise agreement heatmap"),
        row("answer_distributions", "Per-question answer distribution by model (all runs)"),
        row("consistency_bars", "Intra-model consistency — overall, by model"),
        row("consistency_heatmap", "Intra-model consistency — per question, by model"),
        row("cost_bars", "Cost by model (USD)"),
        row("token_bars", "Tokens by model (prompt + completion)"),
    ])

    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>corpus_coding — viz</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.4em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
th {{ background: #f0f0f0; text-align: left; }}
.muted {{ color: #777; }}
a {{ color: #0366d6; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>corpus_coding — visual summary</h1>
<p class='muted'>{summary['n_articles']} article(s) &times; {summary['n_models']} model(s)
({', '.join(html.escape(m) for m in summary['models'])}) &times; up to {summary['n_runs']} run(s) each.
{summary['n_ok']} successful coding(s), {summary['n_err']} error(s). Total cost: ${summary['total_cost']:.4f}.</p>
<table>
<thead><tr><th>figure</th><th>interactive (html)</th><th>static (svg)</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class='muted'>HTML pages use Plotly via CDN (need network to render); open the .svg files for offline static views.</p>
</body></html>
"""
    viz_dir.mkdir(parents=True, exist_ok=True)
    out = viz_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def render_all(
    output_dir: Path,
    in_scope: list[CodedRecord],
    models: list[str],
    agreement_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> Path:
    """Render every figure + the index page. Returns the index.html path."""
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("\nRendering viz...", flush=True)

    _write(fig_agreement_bars(agreement_df), viz_dir / "agreement_bars.html")
    print("  agreement_bars.html", flush=True)

    _write(fig_pairwise_heatmap(in_scope, models), viz_dir / "pairwise_heatmap.html")
    print("  pairwise_heatmap.html", flush=True)

    _write(fig_answer_distributions(in_scope, models), viz_dir / "answer_distributions.html")
    print("  answer_distributions.html", flush=True)

    _write(fig_consistency(consistency_df), viz_dir / "consistency_bars.html")
    print("  consistency_bars.html", flush=True)

    _write(fig_consistency_heatmap(consistency_df), viz_dir / "consistency_heatmap.html")
    print("  consistency_heatmap.html", flush=True)

    cost_fig, token_fig = fig_cost(cost_df)
    _write(cost_fig, viz_dir / "cost_bars.html")
    _write(token_fig, viz_dir / "token_bars.html")
    print("  cost_bars.html, token_bars.html", flush=True)

    n_ok = sum(1 for r in in_scope if r.response is not None)
    n_err = sum(1 for r in in_scope if r.error is not None)
    summary = {
        "n_articles": len({r.pmcid for r in in_scope}),
        "n_models": len(models),
        "models": models,
        "n_runs": max((r.run for r in in_scope), default=0),
        "n_ok": n_ok,
        "n_err": n_err,
        "total_cost": float(cost_df["cost_usd"].sum()) if not cost_df.empty else 0.0,
    }
    index_path = write_index_html(viz_dir, summary)
    print(f"  index.html\nViz landing page: {index_path}", flush=True)
    return index_path


if __name__ == "__main__":
    # Standalone re-render from an existing output dir, no LLM calls.
    import argparse

    from main import DEFAULT_OUTPUT_DIR, load_existing_records  # local import to avoid a cycle at module load

    ap = argparse.ArgumentParser(description="Re-render corpus_coding viz from an existing output dir.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    results_path = args.output_dir / "results.jsonl"
    all_records = load_existing_records(results_path)
    models = args.models or sorted({r.model for r in all_records})
    in_scope = [r for r in all_records if r.model in models]

    ok_records = [r for r in in_scope if r.response is not None]
    groups = scoring.group_by_model_article(ok_records)
    consistency_df = scoring.intra_model_consistency(groups)
    majorities = scoring.model_majorities(groups)
    pmcids = sorted({r.pmcid for r in in_scope})
    agreement_df = scoring.inter_model_agreement(majorities, models=models, pmcids=pmcids)

    cost_rows = {}
    for r in in_scope:
        agg = cost_rows.setdefault(r.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
        agg["calls"] += 1
        agg["prompt_tokens"] += r.usage.prompt_tokens
        agg["completion_tokens"] += r.usage.completion_tokens
        agg["cost_usd"] += r.cost_usd
    cost_df = pd.DataFrame.from_dict(cost_rows, orient="index").sort_index()

    render_all(args.output_dir, in_scope, models, agreement_df, consistency_df, cost_df)

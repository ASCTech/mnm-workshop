"""Plotly visualizations for the multi-judge Bradley-Terry comparison.

Four figures + a landing page, written to <out-dir>/output/viz/:

  - forest.html   -- BT strength +/- SE per item, small multiples (one panel
                     per judge, each independently sorted by that judge's
                     strength), points colored by party.
  - scatter.html  -- BT strength vs. year (metadata the judge never saw),
                     colored by party, with a dropdown to switch judge. Reads
                     as: does the emergent economic-left/right scale separate
                     the parties, and does it drift over time?
  - heatmap.html  -- judge x judge Spearman rank-correlation of BT scales
                     (diverging colorscale, neutral midpoint at 0).
  - summary.html  -- per-judge party-alignment / cost / consistency table.
  - index.html    -- landing page linking the above.

Color follows the dataviz-skill rules used elsewhere in the workshop: fixed,
never-cycled hue order for categorical identity (party, judge), a diverging
two-hue-plus-neutral-midpoint scale for the correlation heatmap (a polarity
measure, not a magnitude), hover tooltips on every mark, and a legend
whenever there are >= 2 series.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Fixed, never-cycled categorical palette (Okabe-Ito colorblind-safe hues):
# vermillion for Republican, blue for Democrat (conventional, and the two hues
# stay distinct under the common colorblindness types).
PARTY_ORDER = ["Democrat", "Republican"]
PARTY_COLORS = {"Democrat": "#0072B2", "Republican": "#D55E00"}
_JUDGE_HUES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

# Diverging colorscale for a polarity measure (Spearman rho in [-1, 1]):
# two hues + a neutral gray midpoint, never a rainbow.
_DIVERGING = [[0.0, "#b2182b"], [0.5, "#f7f7f7"], [1.0, "#2166ac"]]


def _judge_palette(judges: list[str]) -> dict[str, str]:
    return {j: _JUDGE_HUES[i % len(_JUDGE_HUES)] for i, j in enumerate(judges)}


def _party_colors(parties: pd.Series) -> list[str]:
    return [PARTY_COLORS.get(p, "#999999") for p in parties]


def render_forest(scores_by_judge: dict[str, pd.DataFrame], viz_dir: Path) -> Path:
    """Small multiples: one forest-plot panel per judge, each independently
    sorted by that judge's own strength (ascending, so the strongest item
    ends up at the top of the panel -- Scatter/categorical-y defaults to
    plotting first-seen categories at the bottom)."""
    from plotly.subplots import make_subplots

    judges = list(scores_by_judge.keys())
    n = len(judges)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=False, subplot_titles=judges, horizontal_spacing=0.06)

    parties_seen: set[str] = set()
    max_items = 1
    for col, judge in enumerate(judges, start=1):
        df = scores_by_judge[judge].sort_values("strength", ascending=True)
        max_items = max(max_items, len(df))
        parties_seen.update(df["party"].unique())
        colors = _party_colors(df["party"])
        customdata = np.array(
            list(zip(
                df["title"].fillna("").map(lambda s: html.escape(str(s))[:80]),
                df["president"],
                df["party"],
                df["year"].fillna(-1),
                df["n_matchups"],
                df["se"],
                df["choix_strength"],
            )),
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=df["strength"],
                y=df.index,
                mode="markers",
                marker=dict(color=colors, size=9, line=dict(width=1, color="white")),
                error_x=dict(type="data", array=df["se"], visible=True, thickness=1.2, width=3, color="#888"),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]} (%{customdata[2]})"
                    "<br>year: %{customdata[3]}<br>strength: %{x:.3f} (se=%{customdata[5]:.3f}, "
                    "n=%{customdata[4]} matchups)<br>choix cross-check: %{customdata[6]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="BT strength", zeroline=True, zerolinecolor="#ccc", row=1, col=col)

    # Party legend as dummy off-canvas traces, fixed order, added once.
    for party in PARTY_ORDER:
        if party not in parties_seen:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=PARTY_COLORS[party], size=10),
                name=party, showlegend=True,
            ),
            row=1, col=1,
        )

    fig.update_layout(
        title=f"Bradley-Terry forest plot per judge ({n} judges, each panel sorted by its own strength)",
        height=max(520, 20 * max_items + 160),
        width=max(1100, 340 * n),
        plot_bgcolor="white",
        margin=dict(l=130, r=40, t=90, b=60),
        legend=dict(title="party", orientation="h", y=-0.05, yanchor="top"),
        hoverlabel=dict(font_size=12, align="left"),
    )
    out = viz_dir / "forest.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def render_scatter(combined: pd.DataFrame, judges: list[str], viz_dir: Path) -> Path:
    """Strength-vs-year scatter, colored by party, with a dropdown to switch
    which judge's strength column is plotted (validation against party/year
    metadata the judge never saw)."""
    fig = go.Figure()
    n_judges = len(judges)
    for idx, judge in enumerate(judges):
        df = combined[combined["judge"] == judge].dropna(subset=["year"])
        colors = _party_colors(df["party"])
        customdata = np.array(
            list(zip(df["identifier"], df["title"].fillna("").map(lambda s: html.escape(str(s))[:80]),
                     df["president"], df["party"], df["n_matchups"])),
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=df["year"], y=df["strength"], mode="markers",
                marker=dict(color=colors, size=10, line=dict(width=1, color="white")),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} (%{customdata[3]})"
                    "<br>year: %{x}<br>strength: %{y:.3f} (n=%{customdata[4]} matchups)<extra></extra>"
                ),
                name=judge, visible=(idx == 0), showlegend=False,
            )
        )

    parties_present = [p for p in PARTY_ORDER if p in set(combined["party"])]
    for party in parties_present:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=PARTY_COLORS[party], size=10),
                name=party, showlegend=True,
            )
        )
    n_dummy = len(parties_present)

    buttons = []
    for idx, judge in enumerate(judges):
        visible = [i == idx for i in range(n_judges)] + [True] * n_dummy
        buttons.append(dict(label=judge, method="update", args=[{"visible": visible}]))

    fig.update_layout(
        title="BT economic strength vs. year, by judge (party/year never shown to the judge) -- select a judge",
        xaxis=dict(title="year"),
        yaxis=dict(title="BT strength (higher = more 'right')"),
        plot_bgcolor="white",
        legend=dict(title="party"),
        margin=dict(l=60, r=140, t=70, b=50),
        updatemenus=[dict(buttons=buttons, x=1.02, xanchor="left", y=1.0, direction="down", showactive=True)],
        height=650,
    )
    out = viz_dir / "scatter.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def render_heatmap(rank_corr_matrix: pd.DataFrame, viz_dir: Path) -> Path:
    """Judge x judge Spearman rank-correlation of BT scales -- a polarity
    measure, so a diverging colorscale with a neutral gray midpoint at 0."""
    judges = list(rank_corr_matrix.columns)
    z = rank_corr_matrix.values.astype(float)

    annotations = []
    for i, row_judge in enumerate(judges):
        for j, col_judge in enumerate(judges):
            val = z[i, j]
            text = "—" if np.isnan(val) else f"{val:.2f}"
            annotations.append(
                dict(
                    x=col_judge, y=row_judge, text=text, showarrow=False,
                    font=dict(color="white" if (not np.isnan(val) and abs(val) > 0.55) else "#222"),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=judges, y=judges, zmin=-1, zmax=1, zmid=0,
            colorscale=_DIVERGING, colorbar=dict(title="Spearman ρ"),
            hovertemplate="%{y} vs %{x}: ρ=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Judge x judge Spearman rank-correlation of BT scales (do judges agree on ordering?)",
        annotations=annotations,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=max(550, 140 * len(judges)),
        height=max(500, 130 * len(judges)),
        margin=dict(l=140, r=40, t=80, b=100),
    )
    out = viz_dir / "heatmap.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def render_summary_table(
    scores_by_judge: dict[str, pd.DataFrame],
    cost_by_judge: dict[str, dict],
    party_corr_by_judge: dict[str, dict | None],
    year_corr_by_judge: dict[str, dict | None],
    viz_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    rows = []
    for judge, df in scores_by_judge.items():
        c = cost_by_judge.get(judge, {})
        sp = party_corr_by_judge.get(judge)
        sy = year_corr_by_judge.get(judge)
        rows.append({
            "judge": judge,
            "n_ok": c.get("n_ok", 0),
            "n_error": c.get("n_error", 0),
            "cost_usd_est": c.get("cost_usd_est"),
            "mean_se": float(df["se"].mean()),
            "regularized": bool(df["regularized"].iloc[0]) if "regularized" in df else None,
            "party_r": sp["pointbiserial_r"] if sp else None,
            "party_p": sp["pointbiserial_p"] if sp else None,
            "year_rho": sy["rho"] if sy else None,
            "year_p": sy["p"] if sy else None,
        })
    summary_df = pd.DataFrame(rows)

    def _fmt(col: str, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if col == "cost_usd_est":
            return f"${v:.4f}"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=[c.replace("_", " ") for c in summary_df.columns],
                fill_color="#f0f0f0", align="left", font=dict(size=12),
            ),
            cells=dict(
                values=[[_fmt(c, v) for v in summary_df[c]] for c in summary_df.columns],
                align="left", font=dict(size=12),
            ),
        )]
    )
    fig.update_layout(
        title="Per-judge party-alignment / cost / consistency summary",
        height=110 + 34 * len(summary_df),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    out = viz_dir / "summary.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out, summary_df


def write_index_html(
    viz_dir: Path,
    dimension: str,
    judges: list[str],
    n_items: int,
    figure_paths: dict[str, Path],
) -> Path:
    rows_html = "".join(
        f"<tr><td>{html.escape(label)}</td><td><a href='{path.name}'>{path.name}</a></td></tr>"
        for label, path in figure_paths.items()
    )
    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>bt_scoring -- multi-judge Bradley-Terry viz</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.4em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
th {{ background: #f0f0f0; text-align: left; }}
.muted {{ color: #777; }}
a {{ color: #0366d6; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
iframe {{ width: 100%; height: 640px; border: 1px solid #ddd; border-radius: 4px; margin: 0.5em 0 1.5em; }}
</style></head><body>
<h1>bt_scoring -- multi-judge Bradley-Terry comparison</h1>
<p class='muted'>Dimension judged: <code>{html.escape(dimension)}</code><br>
Judges ({len(judges)}): <code>{html.escape(", ".join(judges))}</code><br>
Items: {n_items}</p>
<h2>Figures</h2>
<table><thead><tr><th>figure</th><th>file</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Forest plot -- BT strength per item, per judge (colored by party)</h2>
<iframe src='forest.html'></iframe>
<h2>Strength vs. year -- validation against party / year the judge never saw</h2>
<iframe src='scatter.html'></iframe>
<h2>Judge x judge rank-correlation of BT scales</h2>
<iframe src='heatmap.html'></iframe>
<h2>Per-judge summary (party alignment, cost)</h2>
<iframe src='summary.html' style='height: 320px;'></iframe>
<p class='muted'>Plotly figures load via CDN; open individual .html files directly for a full-window view.</p>
</body></html>
"""
    out = viz_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def render_all(
    output_dir: Path,
    scores_by_judge: dict[str, pd.DataFrame],
    rank_corr_matrix: pd.DataFrame,
    cost_by_judge: dict[str, dict],
    party_corr_by_judge: dict[str, dict | None],
    year_corr_by_judge: dict[str, dict | None],
    dimension: str,
) -> dict:
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    judges = list(scores_by_judge.keys())

    combined = pd.concat(
        [df.reset_index().rename(columns={"index": "identifier"}).assign(judge=judge)
         for judge, df in scores_by_judge.items()],
        ignore_index=True,
    )
    n_items = combined["identifier"].nunique()

    forest_path = render_forest(scores_by_judge, viz_dir)
    scatter_path = render_scatter(combined, judges, viz_dir)
    heatmap_path = render_heatmap(rank_corr_matrix, viz_dir)
    summary_path, summary_df = render_summary_table(
        scores_by_judge, cost_by_judge, party_corr_by_judge, year_corr_by_judge, viz_dir,
    )

    figure_paths = {
        "Forest plot (BT strength +/- SE per judge)": forest_path,
        "Strength vs. year scatter": scatter_path,
        "Judge x judge rank-correlation heatmap": heatmap_path,
        "Per-judge summary table": summary_path,
    }
    index_path = write_index_html(viz_dir, dimension, judges, n_items, figure_paths)

    return {
        "index_html": str(index_path),
        "forest": str(forest_path),
        "scatter": str(scatter_path),
        "heatmap": str(heatmap_path),
        "summary": str(summary_path),
        "summary_table": summary_df,
    }

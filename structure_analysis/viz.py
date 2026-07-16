"""Stage 5: interactive Plotly figures + an HTML index for a completed run.

Figures produced under output/<source>/viz/:
  - documents.html    custom 2D scatter (UMAP-reduced embeddings), a button
                       toggles marker color between "topic" and "true label"
                       so cluster<->ground-truth alignment is visible at a glance.
  - intertopic_map.html   BERTopic's own visualize_topics() (LDAvis-style).
  - topic_words.html      BERTopic's own visualize_barchart() (top words/topic).
  - hierarchy.html        BERTopic's own visualize_hierarchy() dendrogram.
  - crosstab.html         topic x label count heatmap (sequential single hue).
  - index.html            landing page: metrics panel + links to all of the above.

All BERTopic calls use use_ctfidf=True so nothing here ever needs a
sentence-transformers *embedding* backend loaded inside BERTopic itself (topic
words/embeddings are derived purely from the c-TF-IDF matrix); the 2D UMAP
reduction for the scatter runs on the precomputed embeddings we already have.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from bertopic import BERTopic
from umap import UMAP

UMAP_2D_SEED = 20260716

# Fixed-order categorical hues (Plotly's colorblind-safe "Safe" set), assigned
# by rank-of-first-appearance so the same topic/label always gets the same
# color across figures within a run. Outliers / long-tail "other" get grey.
_CATEGORICAL = pcolors.qualitative.Safe + pcolors.qualitative.Set2
_OUTLIER_COLOR = "rgba(140,140,140,0.55)"
_SEQUENTIAL_SCALE = "Blues"


def _truncate(s: str, n: int = 300) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _hover_safe(s: str) -> str:
    return html.escape(str(s)).replace("\n", "<br>")


def _fixed_palette(values: list) -> dict:
    """value -> color, assigned in first-seen order (never re-cycled per redraw)."""
    order: list = []
    seen = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            order.append(v)
    mapping = {}
    for v in order:
        if v in (-1, "-1", "other", "unknown"):
            mapping[v] = _OUTLIER_COLOR
        else:
            mapping[v] = _CATEGORICAL[len(mapping) % len(_CATEGORICAL)]
    return mapping


def _topic_hover_words(model: BERTopic, tid: int, n: int = 4) -> str:
    terms = model.get_topic(tid) or []
    return ", ".join(w for w, _ in terms[:n]) if terms else "(outlier)"


def render_document_scatter(
    viz_dir: Path,
    model: BERTopic,
    embeddings: np.ndarray,
    topics: list[int],
    labels: list[str],
    doc_ids: list[str],
    titles: list[str],
    n_neighbors: int = 15,
) -> Path:
    """2D UMAP scatter with a color-toggle button: topic vs. true ground-truth label."""
    reducer = UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=UMAP_2D_SEED
    )
    xy = reducer.fit_transform(embeddings)

    topic_words = {t: _topic_hover_words(model, t) for t in set(topics)}
    customdata = np.array(
        list(
            zip(
                doc_ids,
                [_hover_safe(_truncate(t)) for t in titles],
                topics,
                [_hover_safe(topic_words[t]) for t in topics],
                [_hover_safe(l) for l in labels],
            )
        ),
        dtype=object,
    )
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
        "topic: %{customdata[2]} (%{customdata[3]})<br>"
        "label: %{customdata[4]}<extra></extra>"
    )

    topic_palette = _fixed_palette(topics)
    label_palette = _fixed_palette(labels)
    topic_colors = [topic_palette[t] for t in topics]
    label_colors = [label_palette[l] for l in labels]

    fig = go.Figure(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            marker=dict(size=6, color=topic_colors, opacity=0.8, line=dict(width=0)),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="documents",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Documents — 2D UMAP of the clustering embedding (toggle color below)",
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        hoverlabel=dict(font_size=12, align="left"),
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=800,
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                x=1.0,
                xanchor="left",
                y=1.0,
                buttons=[
                    dict(label="Color: BERTopic topic", method="restyle",
                         args=[{"marker.color": [topic_colors]}]),
                    dict(label="Color: true label", method="restyle",
                         args=[{"marker.color": [label_colors]}]),
                ],
            )
        ],
    )
    out = viz_dir / "documents.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def render_crosstab_heatmap(viz_dir: Path, crosstab: pd.DataFrame) -> Path:
    """topic_id x label heatmap, sequential single-hue (magnitude, not identity)."""
    z = crosstab.values
    x = [str(c) for c in crosstab.columns]
    y = [f"topic {i}" for i in crosstab.index]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=_SEQUENTIAL_SCALE,
            hovertemplate="topic %{y} × %{x}: %{z} docs<extra></extra>",
            colorbar=dict(title="docs"),
        )
    )
    fig.update_layout(
        title="Topic × ground-truth label (document counts)",
        xaxis=dict(title="label", tickangle=-45),
        yaxis=dict(title="topic", autorange="reversed"),
        margin=dict(l=80, r=40, t=60, b=140),
        height=max(400, 40 * len(y) + 200),
    )
    out = viz_dir / "crosstab.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    return out


def render_bertopic_views(
    viz_dir: Path, model: BERTopic, n_topics: int, hierarchy_df: pd.DataFrame | None
) -> dict[str, Path | None]:
    """BERTopic's own plotly views. use_ctfidf=True throughout so none of these
    ever need a sentence-transformers embedding backend inside BERTopic."""
    out: dict[str, Path | None] = {}

    fig = model.visualize_topics(use_ctfidf=True, title="Intertopic distance map (c-TF-IDF)")
    p = viz_dir / "intertopic_map.html"
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
    out["intertopic_map"] = p

    fig = model.visualize_barchart(top_n_topics=max(n_topics, 1), n_words=8, title="Top words per topic")
    p = viz_dir / "topic_words.html"
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
    out["topic_words"] = p

    if hierarchy_df is not None and len(hierarchy_df):
        try:
            fig = model.visualize_hierarchy(hierarchical_topics=hierarchy_df, use_ctfidf=True)
            p = viz_dir / "hierarchy.html"
            fig.write_html(str(p), include_plotlyjs="cdn", full_html=True)
            out["hierarchy"] = p
        except Exception as e:
            print(f"  [warn] visualize_hierarchy failed: {type(e).__name__}: {e}")
            out["hierarchy"] = None
    else:
        out["hierarchy"] = None

    return out


def _metric_tile(label: str, value: str) -> str:
    return (
        "<div class='tile'>"
        f"<div class='tile-value'>{html.escape(value)}</div>"
        f"<div class='tile-label'>{html.escape(label)}</div>"
        "</div>"
    )


def write_index_html(
    out_dir: Path,
    source: str,
    metrics: dict,
    meta: dict,
    bertopic_paths: dict[str, Path | None],
    doc_scatter_path: Path,
    crosstab_path: Path,
) -> Path:
    viz_dir = out_dir / "viz"

    tiles = "".join(
        [
            _metric_tile("topics", str(metrics["n_topics"])),
            _metric_tile("outliers", f"{metrics['n_outliers']} ({metrics['outlier_pct']:.1f}%)"),
            _metric_tile("ARI", f"{metrics['adjusted_rand_index']:.3f}"),
            _metric_tile("NMI", f"{metrics['normalized_mutual_info']:.3f}"),
            _metric_tile("homogeneity", f"{metrics['homogeneity']:.3f}"),
            _metric_tile("completeness", f"{metrics['completeness']:.3f}"),
            _metric_tile("V-measure", f"{metrics['v_measure']:.3f}"),
            _metric_tile("docs", str(metrics["n_docs"])),
        ]
    )

    def _link(p: Path | None, label: str) -> str:
        if p is None:
            return f"<li class='muted'>{html.escape(label)} — not available</li>"
        return f"<li><a href='{p.name}'>{html.escape(label)}</a></li>"

    links = "".join(
        [
            _link(doc_scatter_path, "Documents (2D UMAP, toggle topic/label color)"),
            _link(bertopic_paths.get("intertopic_map"), "Intertopic distance map"),
            _link(bertopic_paths.get("topic_words"), "Top words per topic"),
            _link(bertopic_paths.get("hierarchy"), "Topic hierarchy (dendrogram)"),
            _link(crosstab_path, "Topic × label crosstab (heatmap)"),
        ]
    )

    page = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>structure_analysis — {html.escape(source)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2em; color: #1b1b1b; background:#fafafa; }}
h1 {{ font-size: 1.4em; }}
.muted {{ color: #767676; }}
a {{ color: #0b5fae; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.meta {{ font-size: 0.92em; color: #444; margin-bottom: 1.4em; }}
.meta code {{ background: #eee; padding: 1px 5px; border-radius: 3px; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 1em 0 1.8em; }}
.tile {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px 16px; min-width: 96px; text-align: center; }}
.tile-value {{ font-size: 1.5em; font-weight: 600; color: #0b5fae; }}
.tile-label {{ font-size: 0.78em; color: #666; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.03em; }}
ul.links {{ list-style: none; padding: 0; }}
ul.links li {{ margin: 6px 0; padding: 8px 12px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
ul.links li.muted {{ background: #f2f2f2; color: #999; }}
</style></head><body>
<h1>structure_analysis — {html.escape(source)}</h1>
<p class='meta'>
  extract model: <code>{html.escape(meta['extract_model'])}</code> &nbsp;|&nbsp;
  embed backend: <code>{html.escape(meta['embed_backend'])}</code> (<code>{html.escape(meta['embed_model'])}</code>) &nbsp;|&nbsp;
  cluster: UMAP(n_neighbors={meta['n_neighbors']}, n_components={meta['n_components']}) + HDBSCAN(min_cluster_size={meta['min_cluster_size']}) &nbsp;|&nbsp;
  ground truth: <code>{html.escape(meta['label_level'])}</code> label &nbsp;|&nbsp;
  cost: <code>${meta['total_cost_usd']:.4f}</code>
</p>
<div class='tiles'>{tiles}</div>
<h2>Figures</h2>
<ul class='links'>{links}</ul>
<p class='muted'>Figures render Plotly via CDN (need network once per page load); metrics above are computed once per run and baked into this page.</p>
</body></html>
"""
    out = viz_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def render_all(
    out_dir: Path,
    model: BERTopic,
    embeddings: np.ndarray,
    topics: list[int],
    labels: list[str],
    doc_ids: list[str],
    titles: list[str],
    metrics: dict,
    crosstab: pd.DataFrame,
    hierarchy_df: pd.DataFrame | None,
    meta: dict,
) -> Path:
    """Build every figure + the landing page. Returns the index.html path."""
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("  building document scatter (2D UMAP)...")
    doc_scatter_path = render_document_scatter(viz_dir, model, embeddings, topics, labels, doc_ids, titles)

    print("  building topic x label crosstab heatmap...")
    crosstab_path = render_crosstab_heatmap(viz_dir, crosstab)

    print("  building BERTopic views (intertopic map, barchart, hierarchy)...")
    n_topics = len(set(topics) - {-1})
    bertopic_paths = render_bertopic_views(viz_dir, model, n_topics, hierarchy_df)

    index_path = write_index_html(
        out_dir, meta["source"], metrics, meta, bertopic_paths, doc_scatter_path, crosstab_path
    )
    return index_path

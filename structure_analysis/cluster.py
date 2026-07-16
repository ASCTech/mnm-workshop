"""Stage 2 (embed) and stage 3 (BERTopic cluster + validate against ground truth)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from tqdm import tqdm
from umap import UMAP

from llm import CostLedger, get_client

UMAP_SEED = 20260716

DEFAULT_MINILM_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TITAN_MODEL = "titan-embed-text-v2:0"

_minilm_model = None  # lazy singleton, like llm.get_client()


def _get_minilm(model_name: str):
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer

        # Pin to CPU: this branch is not a GPU workload (600 short texts embed in
        # a couple seconds on CPU), and keeping it CPU-only makes it portable for
        # workshop attendees without a GPU and avoids any CUDA/fork interaction
        # with the numba-based clustering that follows.
        print(f"  loading local sentence-transformers model: {model_name} (cpu) ...")
        _minilm_model = SentenceTransformer(model_name, device="cpu")
    return _minilm_model


def build_embedding_text(parsed: dict) -> str:
    """domain + methods + techniques + contribution, per the workshop spec.

    `application` is deliberately left out of the embedding text (kept only
    as a topic-labeling/report field) so the clustering axis stays anchored
    on field/method/technique/contribution rather than downstream use.
    """
    domain = parsed.get("domain") or ""
    methods = "; ".join(parsed.get("methods") or [])
    techniques = "; ".join(parsed.get("techniques") or [])
    contribution = parsed.get("contribution") or ""
    return " | ".join(p for p in (domain, methods, techniques, contribution) if p)


def embed_texts(
    doc_ids: list[str],
    texts: list[str],
    backend: str,
    model: str,
    npy_path: Path,
    index_path: Path,
    ledger: CostLedger,
    batch_size: int = 64,
    force: bool = False,
) -> np.ndarray:
    """Embed `texts`, cached to `npy_path` (+ doc_id index json).

    `backend` is one of:
      - "minilm": local `sentence-transformers` model (default; e.g. all-MiniLM-L6-v2,
        384-dim). No network call, no token cost — anecdotally clusters MORE cleanly
        for this kind of short structured-field text than the larger proxy embedding
        model, per the workshop owner's guidance.
      - "titan": OSU LiteLLM proxy embedding model (e.g. titan-embed-text-v2:0,
        1024-dim). Costed via `ledger`.

    `npy_path`/`index_path` are expected to already be backend-specific (callers
    should name them e.g. `embeddings_minilm.npy` / `embeddings_titan.npy`) so a
    384-dim MiniLM cache can never be silently loaded where a 1024-dim Titan
    cache — or vice versa — was expected.

    Cache hit requires the same doc_id set in the same order; otherwise
    re-embeds everything.
    """
    if not force and npy_path.exists() and index_path.exists():
        cached_ids = json.loads(index_path.read_text())
        if cached_ids == doc_ids:
            print(f"  embeddings cache hit: {npy_path.name} ({len(doc_ids)} rows)")
            return np.load(npy_path)
        print("  embeddings cache stale (doc set changed) — re-embedding")

    npy_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "minilm":
        st_model = _get_minilm(model)
        vectors_arr = st_model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        arr = np.asarray(vectors_arr, dtype=np.float32)
        # Local inference: no proxy call, no tokens, no cost — record it in the
        # ledger at $0 for visibility rather than silently omitting it.
        ledger.add(f"{model} (local)", 0, 0)
    elif backend == "titan":
        client = get_client()
        vectors: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="embed"):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", 0) or 0
            ledger.add(model, in_tok, 0)
            vectors.extend(np.array(d.embedding, dtype=np.float32) for d in resp.data)
        arr = np.vstack(vectors)
    else:
        raise ValueError(f"unknown embed backend: {backend!r} (expected 'minilm' or 'titan')")

    np.save(npy_path, arr)
    index_path.write_text(json.dumps(doc_ids))
    print(f"  embedded {arr.shape[0]} docs -> {arr.shape[1]}-dim, cached to {npy_path.name}")
    return arr


def make_bertopic(min_cluster_size: int, n_neighbors: int = 15, n_components: int = 5) -> BERTopic:
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=UMAP_SEED,
    )
    # cluster_selection_method="leaf", not the HDBSCAN default "eom": on these
    # MiniLM embeddings the density has one dominant mode, so eom collapses to a
    # single giant topic + one small one (2 topics, 0 outliers) regardless of
    # min_cluster_size. "leaf" selects the fine-grained leaf clusters instead,
    # recovering the ~9-12 field-level topics that KMeans/NMI confirm are present.
    # core_dist_n_jobs=1: HDBSCAN otherwise spawns a joblib/loky worker pool for
    # the core-distance step, and those child processes intermittently fail to be
    # reaped — the parent then blocks joining them at teardown (a 3-minute hang
    # requiring a manual kill). At this corpus size the parallelism buys nothing,
    # so run it in-process for a clean, reliable exit.
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    # embedding_model left as None (default): we always pass precomputed
    # embeddings to fit_transform, so BERTopic never loads a sentence-transformers
    # backend and no model is downloaded.
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )


def fit_topics(
    texts: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int,
    n_neighbors: int = 15,
    n_components: int = 5,
) -> tuple[BERTopic, list[int]]:
    model = make_bertopic(min_cluster_size, n_neighbors, n_components)
    topics, _ = model.fit_transform(documents=texts, embeddings=embeddings)
    return model, topics


def topic_table(model: BERTopic, n_words: int = 8) -> pd.DataFrame:
    info = model.get_topic_info()
    rows = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        terms = model.get_topic(tid) or []
        top_words = ", ".join(w for w, _ in terms[:n_words])
        rows.append({"topic_id": tid, "size": int(row["Count"]), "top_words": top_words})
    return pd.DataFrame(rows)


def representative_docs(
    model: BERTopic, doc_ids: list[str], titles: list[str], topics: list[int], n: int = 3
) -> dict[int, list[str]]:
    """A few representative doc titles per topic, via BERTopic's own selection."""
    out: dict[int, list[str]] = {}
    try:
        reps = model.get_representative_docs()
    except Exception:
        reps = {}
    id_by_text = {}  # fallback if get_representative_docs isn't keyed as expected
    for tid, docs in (reps or {}).items():
        out[int(tid)] = list(docs)[:n]
    return out


def validate(topics: list[int], labels: list[str]) -> dict:
    """ARI / NMI / homogeneity / completeness / V-measure of topics vs. ground truth.

    Outlier docs (topic == -1) are included as their own "cluster" — HDBSCAN's
    noise label is itself informative to report, and dropping it would inflate
    scores by discarding the hardest cases.
    """
    ari = adjusted_rand_score(labels, topics)
    nmi = normalized_mutual_info_score(labels, topics)
    hom, comp, vmeas = homogeneity_completeness_v_measure(labels, topics)
    n_outliers = sum(1 for t in topics if t == -1)
    n_topics = len(set(topics) - {-1})
    return {
        "n_docs": len(topics),
        "n_topics": n_topics,
        "n_outliers": n_outliers,
        "outlier_pct": 100.0 * n_outliers / len(topics) if topics else 0.0,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "homogeneity": hom,
        "completeness": comp,
        "v_measure": vmeas,
    }


def topic_label_crosstab(topics: list[int], labels: list[str], top_k_labels: int = 12) -> pd.DataFrame:
    """topic_id x label crosstab, restricted to the top_k_labels most frequent labels
    (the long tail otherwise makes the table unreadable) plus an 'other' column."""
    label_counts = pd.Series(labels).value_counts()
    keep = set(label_counts.head(top_k_labels).index)
    collapsed = [l if l in keep else "other" for l in labels]
    ct = pd.crosstab(pd.Series(topics, name="topic_id"), pd.Series(collapsed, name="label"))
    return ct.sort_index()


def dominant_label_per_topic(crosstab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, row in crosstab.iterrows():
        total = row.sum()
        dominant = row.idxmax()
        rows.append(
            {
                "topic_id": tid,
                "size": int(total),
                "dominant_label": dominant,
                "dominant_frac": row[dominant] / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)

"""Inter-model agreement and intra-model consistency scoring.

Two related but distinct questions, computed per question of the codebook:

- **Intra-model consistency**: does a single model give the same answer to the
  same article across its ``M`` repeated runs? (temperature > 0, so this is
  not a given.) Reported per model, as the mean fraction of runs agreeing
  with that model's own majority answer, averaged over articles.
- **Inter-model agreement**: do different models agree with each other on the
  same article? Each model is first collapsed to its majority answer across
  its own runs, then models are compared pairwise (percent agreement, Cohen's
  kappa) and jointly (Krippendorff's alpha, Fleiss' kappa, treating models as
  raters and articles as units).
"""

from __future__ import annotations

import itertools
import warnings
from collections import Counter, defaultdict

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from schema import CodedRecord, QUESTIONS


def normalize_value(question: str, value) -> str:
    """Coerce a coded value to a comparable string label."""
    if question == "primary_field":
        return str(value).strip().lower()
    return str(value)


def group_by_model_article(records: list[CodedRecord]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """(model, pmcid) -> one normalized dict-of-answers per successful run."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in records:
        if r.response is None:
            continue
        dumped = r.response.model_dump(mode="json")
        groups[(r.model, r.pmcid)].append({q: normalize_value(q, dumped[q]) for q in QUESTIONS})
    return groups


def majority_fraction(values: list[str]) -> tuple[str, float]:
    """Most common value and the fraction of ``values`` that equal it."""
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value, count / len(values)


def intra_model_consistency(groups: dict[tuple[str, str], list[dict[str, str]]]) -> pd.DataFrame:
    """Per model x question: mean %-of-runs-agreeing-with-majority, over articles.

    Only articles with >=2 runs for that model contribute (consistency is
    undefined with a single run).
    """
    per_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, _pmcid), answers in groups.items():
        if len(answers) < 2:
            continue
        for q in QUESTIONS:
            _, frac = majority_fraction([a[q] for a in answers])
            per_cell[(model, q)].append(frac)

    models = sorted({m for m, _ in per_cell})
    if not models:
        return pd.DataFrame(columns=[*QUESTIONS, "overall"])

    df = pd.DataFrame(index=models, columns=[*QUESTIONS, "overall"], dtype=float)
    for model in models:
        per_q = []
        for q in QUESTIONS:
            fracs = per_cell.get((model, q), [])
            if fracs:
                val = float(np.mean(fracs)) * 100
                df.loc[model, q] = val
                per_q.append(val)
        df.loc[model, "overall"] = float(np.mean(per_q)) if per_q else float("nan")
    return df.round(1)


def model_majorities(
    groups: dict[tuple[str, str], list[dict[str, str]]],
) -> dict[tuple[str, str], dict[str, str]]:
    """(model, pmcid) -> {question: majority answer across that model's runs}."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for key, answers in groups.items():
        out[key] = {q: majority_fraction([a[q] for a in answers])[0] for q in QUESTIONS}
    return out


def inter_model_agreement(
    majorities: dict[tuple[str, str], dict[str, str]],
    models: list[str],
    pmcids: list[str],
) -> pd.DataFrame:
    """Per-question agreement across models, treating models as raters.

    Units are restricted to articles every model in ``models`` successfully
    coded, so Krippendorff's alpha / Fleiss' kappa see a complete rater x unit
    matrix (both are well-defined without ad hoc missing-data handling).
    """
    rows = []
    for q in QUESTIONS:
        units = [p for p in pmcids if all((m, p) in majorities for m in models)]
        n_units = len(units)
        if n_units < 2 or len(models) < 2:
            rows.append({"question": q, "n_units": n_units, "pct_agreement": np.nan,
                         "cohen_kappa": np.nan, "krippendorff_alpha": np.nan, "fleiss_kappa": np.nan})
            continue

        vals = {m: [majorities[(m, p)][q] for p in units] for m in models}
        categories = sorted({v for series in vals.values() for v in series})
        code = {v: i for i, v in enumerate(categories)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Pairwise percent agreement + Cohen's kappa, averaged over model pairs.
            pct_scores, kappa_scores = [], []
            for m1, m2 in itertools.combinations(models, 2):
                a, b = vals[m1], vals[m2]
                pct_scores.append(np.mean([x == y for x, y in zip(a, b)]))
                try:
                    k = cohen_kappa_score(a, b)
                    kappa_scores.append(k if np.isfinite(k) else np.nan)
                except Exception:
                    kappa_scores.append(np.nan)

            # Krippendorff's alpha: raters (models) x units (articles) matrix.
            reliability_data = np.array(
                [[code[v] for v in vals[m]] for m in models], dtype=float
            )
            try:
                alpha = krippendorff.alpha(
                    reliability_data=reliability_data, level_of_measurement="nominal"
                )
            except Exception:
                alpha = np.nan

            # Fleiss' kappa: subjects (articles) x raters -> category count table.
            # aggregate_raters requires integer codes (it bincounts each row).
            try:
                table, _cats = aggregate_raters(
                    reliability_data.T.astype(int), n_cat=len(categories)
                )
                fk = fleiss_kappa(table)
                fk = fk if np.isfinite(fk) else np.nan
            except Exception:
                fk = np.nan

            pct_agreement = round(float(np.mean(pct_scores)) * 100, 1) if pct_scores else np.nan
            cohen_kappa = round(float(np.nanmean(kappa_scores)), 3) if kappa_scores else np.nan

        rows.append({
            "question": q,
            "n_units": n_units,
            "pct_agreement": pct_agreement,
            "cohen_kappa": cohen_kappa,
            "krippendorff_alpha": round(float(alpha), 3) if np.isfinite(alpha) else np.nan,
            "fleiss_kappa": round(float(fk), 3) if np.isfinite(fk) else np.nan,
        })

    return pd.DataFrame(rows).set_index("question")

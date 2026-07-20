"""Fit a Bradley-Terry scale (with standard errors) from pairwise verdicts,
and validate it against party / year metadata the judge never saw.

BT-as-logistic-regression: for each judged pair (i, j) with i < j
lexicographically, code x_i = +1, x_j = -1 (all other items 0), and
y = 1{i won}. Then P(y=1) = logistic(x @ beta) = logistic(s_i - s_j), so a
no-intercept logistic regression's coefficients ARE the BT log-strengths and
statsmodels gives their standard errors for free. One item's column is
dropped entirely to fix the identifiability gauge (its strength := 0).

``choix`` (ILSR) is used as an independent point-estimate cross-check.
"""

from __future__ import annotations

import warnings

import choix
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr


def build_design(comparisons: pd.DataFrame, items: list[str], reference: str) -> tuple[pd.DataFrame, pd.Series]:
    """comparisons must have columns item_i, item_j (item_i < item_j) and
    winner. Returns (X, y) with one column per item except `reference`."""
    cols = [it for it in items if it != reference]
    X = pd.DataFrame(0.0, index=comparisons.index, columns=cols)
    y = pd.Series(0, index=comparisons.index, dtype=int)

    for idx, row in comparisons.iterrows():
        i, j, winner = row["item_i"], row["item_j"], row["winner"]
        if i in X.columns:
            X.at[idx, i] = 1.0
        if j in X.columns:
            X.at[idx, j] = -1.0
        y.at[idx] = 1 if winner == i else 0
    return X, y


def _bootstrap_se(
    comparisons: pd.DataFrame, items: list[str], reference: str,
    alpha: float, n_boot: int, seed: int,
) -> pd.Series:
    """Non-parametric bootstrap over comparison rows, refitting the same
    ridge-regularized model each draw. Gives a real standard error even when
    the unregularized fit is undefined (quasi-complete separation) -- the
    bootstrap only assumes the rows are exchangeable draws, not that the MLE
    covariance formula applies."""
    rng = np.random.default_rng(seed)
    n = len(comparisons)
    cols = [it for it in items if it != reference]
    draws = []
    for _ in range(n_boot):
        sample = comparisons.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        Xb, yb = build_design(sample, items, reference)
        try:
            res = sm.Logit(yb, Xb).fit_regularized(disp=0, alpha=alpha, L1_wt=0.0, maxiter=300)
            draws.append(res.params.reindex(cols))
        except Exception:
            continue
    if not draws:
        return pd.Series(np.nan, index=cols)
    return pd.concat(draws, axis=1).std(axis=1, ddof=1)


def fit_bt_logit(
    comparisons: pd.DataFrame, items: list[str], n_boot: int = 200, seed: int = 0,
) -> pd.DataFrame:
    """Fit the BT scale. `reference` (dropped column) is the item with the
    most comparisons, for the most stable gauge. Returns a DataFrame indexed
    by item with columns strength, se, n_matchups.

    Tries the textbook unregularized no-intercept logistic regression first
    (its coefficients' standard errors come straight from the Hessian). With
    ~24 items and only ~8-12 matchups each this commonly hits quasi-complete
    separation (a subset of items wins every comparison it's in), so we fall
    back to a light L2-ridge fit for a finite point estimate and get SEs via
    bootstrap resampling of the comparison rows instead.
    """
    n_matchups = pd.concat([comparisons["item_i"], comparisons["item_j"]]).value_counts()
    reference = n_matchups.idxmax()

    X, y = build_design(comparisons, items, reference)
    regularized = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200)
            params, bse = result.params, result.bse
            if not np.isfinite(params).all() or not np.isfinite(bse).all() or (params.abs() > 25).any():
                raise ValueError("unregularized fit diverged (quasi-complete separation)")
        except Exception:
            regularized = True
            ridge_alpha = 0.5
            result = sm.Logit(y, X).fit_regularized(disp=0, alpha=ridge_alpha, L1_wt=0.0, maxiter=500)
            params = result.params
            bse = _bootstrap_se(comparisons, items, reference, ridge_alpha, n_boot, seed)

    out = pd.DataFrame({"strength": params, "se": bse})
    out.loc[reference] = {"strength": 0.0, "se": 0.0}
    out["n_matchups"] = n_matchups.reindex(out.index).fillna(0).astype(int)
    out["reference_item"] = reference
    out["regularized"] = regularized
    return out.sort_values("strength", ascending=False)


def fit_bt_choix(comparisons: pd.DataFrame, items: list[str]) -> pd.Series:
    """Independent point-estimate cross-check via choix's ILSR."""
    idx = {it: k for k, it in enumerate(items)}
    data = []
    for _, row in comparisons.iterrows():
        winner = row["winner"]
        loser = row["item_j"] if winner == row["item_i"] else row["item_i"]
        data.append((idx[winner], idx[loser]))
    params = choix.ilsr_pairwise(len(items), data, alpha=0.01)
    return pd.Series(params, index=items, name="choix_strength")


def judge_rank_correlation(strengths_by_judge: dict[str, pd.Series]) -> pd.DataFrame:
    """Judge x judge Spearman rank-correlation matrix between per-judge BT
    strength scales (Round 2 addition; the fitting machinery above is
    untouched -- this just compares its outputs across judges)."""
    judges = list(strengths_by_judge.keys())
    mat = pd.DataFrame(index=judges, columns=judges, dtype=float)
    for a in judges:
        for b in judges:
            common = strengths_by_judge[a].index.intersection(strengths_by_judge[b].index)
            if len(common) < 3:
                mat.loc[a, b] = np.nan
                continue
            rho, _ = spearmanr(strengths_by_judge[a].loc[common], strengths_by_judge[b].loc[common])
            mat.loc[a, b] = rho
    return mat


def validate_scale(scores: pd.DataFrame) -> dict:
    """Validate a fitted BT scale against metadata the judge never saw.

    ``scores`` must have a ``strength`` column and may carry ``year``,
    ``party`` (values "Republican"/"Democrat"), and ``president``.

    The primary check for an economic-orientation scale is **party**: if the
    emergent left/right scale tracks anything real, Republican-delivered
    speeches should sit higher (more "right") than Democratic ones. We report
    a point-biserial correlation (party coded right=1/left=0, so a positive
    value means the scale aligns "right = higher strength") and a
    Mann-Whitney U rank test. ``year`` is kept as a secondary, non-causal
    trend check.
    """
    result: dict = {}

    # --- party (primary): does the scale separate Republicans from Democrats?
    if "party" in scores.columns:
        parties = scores[["strength", "party"]].dropna(subset=["party"])
        parties = parties[parties["party"].isin(["Republican", "Democrat"])]
        r_str = parties.loc[parties["party"] == "Republican", "strength"]
        d_str = parties.loc[parties["party"] == "Democrat", "strength"]
        if len(r_str) >= 2 and len(d_str) >= 2:
            party_bin = (parties["party"] == "Republican").astype(int)  # right=1, left=0
            r_pb, p_pb = pointbiserialr(party_bin, parties["strength"])
            u, p_u = mannwhitneyu(r_str, d_str, alternative="two-sided")
            result["strength_vs_party"] = {
                "pointbiserial_r": float(r_pb), "pointbiserial_p": float(p_pb),
                "mannwhitney_u": float(u), "mannwhitney_p": float(p_u),
                "n_republican": int(len(r_str)), "n_democrat": int(len(d_str)),
                "mean_republican": float(r_str.mean()), "mean_democrat": float(d_str.mean()),
            }
        else:
            result["strength_vs_party"] = None
    else:
        result["strength_vs_party"] = None

    # --- year (secondary): non-causal trend over time
    with_year = scores.dropna(subset=["year"]) if "year" in scores.columns else scores.iloc[0:0]
    if len(with_year) >= 3:
        rho, p = spearmanr(with_year["strength"], with_year["year"])
        result["spearman_strength_vs_year"] = {"rho": rho, "p": p, "n": len(with_year)}
    else:
        result["spearman_strength_vs_year"] = None

    group_col = "president" if "president" in scores.columns else None
    if group_col:
        result["by_president"] = (
            scores.groupby(group_col)["strength"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
    if "party" in scores.columns:
        result["by_party"] = (
            scores.groupby("party")["strength"]
            .agg(["mean", "std", "count"])
            .sort_values("mean", ascending=False)
        )
    return result

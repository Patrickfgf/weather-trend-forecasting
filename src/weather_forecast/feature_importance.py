"""Unique Analysis 3 — feature importance through three independent lenses.

The Track-B global panel model (XGBoost on the leakage-safe lag/calendar/static
feature matrix from :func:`engineering.feature_columns`) is a black box. This module
explains it from three angles, then makes their *disagreement* visible — which is the
whole point: a single importance number is easy to over-trust.

The three lenses, ordered by how much we trust them:

1. Permutation importance (ROBUST HEADLINE) — shuffle one feature on the VALIDATION
   split and measure the drop in score. Model-agnostic and computed on held-out data,
   so it reflects what actually helps the model *generalize*, not what it memorized.
   This is the lens the report leads with.
2. SHAP mean(|value|) — game-theoretic local attributions averaged to a global
   magnitude. Consistent and additive, but only available when ``shap`` is installed.
3. Native gain/impurity (BIASED) — XGBoost's built-in ``feature_importances_``. Free
   but structurally biased: it inflates high-cardinality / continuous splitters and is
   measured on TRAINING structure. Labeled "native gain (biased)" everywhere so nobody
   mistakes it for the headline.

CORRELATED-FEATURE DILUTION (read before trusting any single number): the lag and
rolling features here are heavily collinear (e.g. ``..._lag1`` vs ``..._rollmean3``).
Permutation importance *splits credit* across correlated copies — shuffling one while
its near-duplicate stays intact understates both, so an important-but-redundant feature
can look weak. Native gain instead *concentrates* credit on whichever correlated copy
the trees happened to split on first. Neither is wrong; they answer different
questions. Compare ranks (:func:`compare_importances`) rather than reading one lens
alone, and treat groups of collinear features as a unit.

CAVEATS baked into figure subtitles: this dataset is a rolling daily-snapshot history
(sub-annual to ~1yr), so "importance" is within-sample and seasonal, NOT a statement
about multi-decade climate drivers. Importances are associational, not causal.

LEAKAGE CONTRACT: features come from :func:`engineering.feature_columns`, which already
excludes ``config.LEAKY_OR_TWIN_COLS`` (the Fahrenheit twin IS the target;
``feels_like_*`` is a near-deterministic contemporaneous function of it). This module
re-asserts that exclusion defensively so a caller passing a hand-built ``feature_names``
list cannot smuggle a leaky column into the explanation.

All functions are PURE: arrays/model in -> DataFrame/Figure out. No global state, no
in-place mutation, no disk reads of raw data. Stochastic calls (permutation shuffles,
SHAP sampling) are pinned to ``config.SEED`` for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend BEFORE pyplot — never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, optional_deps

logger = logging.getLogger(__name__)

# Default headline scorer: negative RMSE (sklearn convention — higher is better, so a
# drop after shuffling shows up as a positive importance). RMSE matches how the
# global panel model is selected in backtesting, keeping the lens consistent with it.
DEFAULT_SCORING: str = "neg_root_mean_squared_error"
DEFAULT_TOP_N: int = 20
SHAP_MAX_SAMPLES: int = 2000


# --------------------------------------------------------------------------- #
# Leakage guard (defensive — feature_columns already excludes these)
# --------------------------------------------------------------------------- #
def _assert_no_leaky_features(feature_names: list[str]) -> None:
    """Fail loudly if a leaky/twin column reached the feature list.

    ``engineering.feature_columns`` already drops ``config.LEAKY_OR_TWIN_COLS``; this
    is a second gate so a hand-assembled ``feature_names`` cannot silently leak the
    target into the importance analysis.
    """
    leaky = set(config.LEAKY_OR_TWIN_COLS)
    offenders = [f for f in feature_names if f in leaky]
    if offenders:
        raise ValueError(
            f"Leaky/twin columns must not appear in feature importance: {offenders}. "
            "Features should come from engineering.feature_columns()."
        )


def _sorted_frame(feature_names: list[str], importance: np.ndarray,
                  lens: str, std: np.ndarray | None = None) -> pd.DataFrame:
    """Build a tidy, descending importance frame with a ``lens`` label column."""
    data: dict[str, Any] = {
        "feature": list(feature_names),
        "importance": np.asarray(importance, dtype="float64"),
        "lens": lens,
    }
    if std is not None:
        data["importance_std"] = np.asarray(std, dtype="float64")
    return (pd.DataFrame(data)
            .sort_values("importance", ascending=False, kind="stable")
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Lens 1: native gain / impurity importance (BIASED)
# --------------------------------------------------------------------------- #
def native_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Native ``feature_importances_`` (XGBoost gain / sklearn impurity), descending.

    Labeled "native gain (biased)" because it is measured on the TRAINING structure and
    over-credits high-cardinality continuous splitters — the cheapest but least
    trustworthy lens. Returns an empty frame if the estimator exposes no
    ``feature_importances_`` (e.g. a linear model), so callers can skip gracefully.
    """
    _assert_no_leaky_features(feature_names)
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        logger.info("native_importance: estimator has no feature_importances_; skipping.")
        return _sorted_frame(feature_names, np.zeros(len(feature_names)), "native gain (biased)").iloc[0:0]
    imp = np.asarray(imp, dtype="float64")
    if imp.shape[0] != len(feature_names):
        raise ValueError(
            f"feature_importances_ length {imp.shape[0]} != n feature_names {len(feature_names)}."
        )
    return _sorted_frame(feature_names, imp, "native gain (biased)")


# --------------------------------------------------------------------------- #
# Lens 2: permutation importance (ROBUST HEADLINE — validation only)
# --------------------------------------------------------------------------- #
def permutation_importance_df(model: Any, X_val: Any, y_val: Any,
                              feature_names: list[str], n_repeats: int = 10,
                              scoring: str = DEFAULT_SCORING,
                              random_state: int = config.SEED) -> pd.DataFrame:
    """Permutation importance on the VALIDATION split — the robust headline lens.

    Shuffles each feature ``n_repeats`` times and measures the mean score drop; computed
    on held-out validation rows ONLY (never the training rows used to fit ``model``), so
    it reflects generalization rather than memorized training structure. ``random_state``
    is fixed for reproducibility. Returns mean + std over repeats, sorted descending.

    Beware correlated-feature dilution (module docstring): shuffling one of two
    near-duplicate features while the other stays intact understates both.
    """
    from sklearn.inspection import permutation_importance

    _assert_no_leaky_features(feature_names)
    X = np.asarray(X_val, dtype="float64")
    y = np.asarray(y_val, dtype="float64")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X_val has {X.shape[1]} columns but {len(feature_names)} feature_names given."
        )
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=1,
    )
    return _sorted_frame(
        feature_names, result.importances_mean, "permutation (validation)",
        std=result.importances_std,
    )


# --------------------------------------------------------------------------- #
# Lens 3: SHAP mean(|value|) (optional)
# --------------------------------------------------------------------------- #
def shap_importance(model: Any, X_sample: Any, feature_names: list[str],
                    max_samples: int = SHAP_MAX_SAMPLES) -> pd.DataFrame | None:
    """Global SHAP importance = mean(|SHAP value|) per feature, or None if unavailable.

    Uses ``shap.TreeExplainer`` on a deterministically sampled subset (capped at
    ``max_samples`` for tractability; sampling pinned to ``config.SEED``). Returns
    ``None`` when ``shap`` is not installed OR the explainer fails, so the caller simply
    skips the SHAP figure instead of crashing a Restart-Run-All notebook run.

    SHAP attributions are consistent and additive, but — like every lens here — are
    associational, not causal, and split credit across correlated features.
    """
    _assert_no_leaky_features(feature_names)
    if not optional_deps.HAS_SHAP:
        logger.info("shap_importance: shap unavailable; skipping SHAP lens.")
        return None

    X = np.asarray(X_sample, dtype="float64")
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"X_sample has {X.shape[1]} columns but {len(feature_names)} feature_names given."
        )

    # Deterministic subsample: SHAP is O(samples); cap keeps the notebook fast & stable.
    if X.shape[0] > max_samples:
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[np.sort(idx)]

    shap = optional_deps.shap
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
    except Exception as exc:  # explainer can fail on exotic estimators / wheels
        logger.info("shap_importance: TreeExplainer failed (%s); skipping.", type(exc).__name__)
        return None

    arr = np.asarray(values)
    if arr.ndim == 3:  # some versions return (n, features, outputs) for single-output
        arr = arr[..., 0]
    mean_abs = np.abs(arr).mean(axis=0)
    return _sorted_frame(feature_names, mean_abs, "shap mean(|value|)")


# --------------------------------------------------------------------------- #
# Cross-lens comparison
# --------------------------------------------------------------------------- #
def compare_importances(frames: dict[str, pd.DataFrame], on: str = "feature") -> pd.DataFrame:
    """Outer-join the lenses and rank within each, putting ranks side by side.

    For each lens we add ``<lens>_rank`` (1 = most important) so gain-vs-permutation
    DISAGREEMENT is read at a glance — large rank gaps usually flag correlated-feature
    dilution or native-gain's high-cardinality bias. Sorted by the mean rank across the
    available lenses. ``None`` frames (e.g. SHAP when shap is absent) are skipped.
    """
    valid = {name: f for name, f in frames.items()
             if f is not None and not f.empty and on in f.columns}
    if not valid:
        return pd.DataFrame(columns=[on])

    merged: pd.DataFrame | None = None
    rank_cols: list[str] = []
    for name, f in valid.items():
        ranked = (f[[on, "importance"]]
                  .rename(columns={"importance": f"{name}_importance"})
                  .assign(**{f"{name}_rank": lambda d, n=name: d[f"{n}_importance"]
                             .rank(ascending=False, method="min").astype("Int64")}))
        rank_cols.append(f"{name}_rank")
        merged = ranked if merged is None else merged.merge(ranked, on=on, how="outer")

    assert merged is not None  # guaranteed: valid is non-empty
    merged["mean_rank"] = merged[rank_cols].astype("float64").mean(axis=1)
    return merged.sort_values("mean_rank", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: "plt.Figure", out: "Path | None") -> None:
    """Save only when ``out`` is given (project figure convention)."""
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")


def fig_permutation_importance(perm_df: pd.DataFrame, top: int = DEFAULT_TOP_N,
                               out: "Path | None" = None) -> "plt.Figure":
    """Horizontal bar of the top-``top`` permutation importances with std error bars.

    The headline figure: validation-split importance (mean over repeats) with the across-
    repeats std as error bars, so the reader can see which rankings are stable vs noisy.
    Returns the Figure; saves only when ``out`` is provided. Never calls ``plt.show``.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    if perm_df is None or perm_df.empty:
        ax.text(0.5, 0.5, "no permutation importance available",
                ha="center", va="center", transform=ax.transAxes)
        _save(fig, out)
        return fig

    sub = perm_df.head(top).iloc[::-1]  # reverse so the largest sits on top
    err = sub["importance_std"] if "importance_std" in sub.columns else None
    ax.barh(sub["feature"], sub["importance"], xerr=err,
            color="#2c7fb8", ecolor="#555555", capsize=3)
    ax.set_xlabel("Permutation importance (mean score drop, RMSE units)")
    ax.set_ylabel("Feature")
    ax.set_title("Permutation feature importance — global temperature model")
    ax.figure.text(
        0.5, -0.02,
        "Robust lens: validation split only, error bars = std over repeats. "
        "Within-sample & seasonal (sub-annual daily-snapshot history), associational not causal. "
        "Correlated lag/rolling features split credit — read as groups.",
        ha="center", va="top", fontsize=8, style="italic", wrap=True,
    )
    ax.margins(y=0.01)
    fig.tight_layout()
    _save(fig, out)
    return fig


def fig_importance_comparison(compare_df: pd.DataFrame, top: int = DEFAULT_TOP_N,
                              out: "Path | None" = None) -> "plt.Figure":
    """Slope chart of native-gain rank vs permutation rank — disagreement made visible.

    Connects each top feature's "native gain (biased)" rank to its "permutation
    (validation)" rank; steep lines = the two lenses disagree (often correlated-feature
    dilution or native-gain bias). Falls back to a single-lens bar / message when only
    one lens is present. Returns the Figure; saves only when ``out`` is given.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    native_col, perm_col = "native gain (biased)_rank", "permutation (validation)_rank"

    if (compare_df is None or compare_df.empty
            or native_col not in compare_df.columns or perm_col not in compare_df.columns):
        ax.text(0.5, 0.5, "need both native-gain and permutation lenses to compare ranks",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        _save(fig, out)
        return fig

    sub = (compare_df.dropna(subset=[native_col, perm_col])
           .sort_values("mean_rank", kind="stable").head(top))
    left_x, right_x = 0.0, 1.0
    for _, r in sub.iterrows():
        n_rank, p_rank = float(r[native_col]), float(r[perm_col])
        worsens = p_rank > n_rank  # permutation ranks it LESS important than gain did
        ax.plot([left_x, right_x], [n_rank, p_rank],
                marker="o", color="#d95f02" if worsens else "#1b9e77", alpha=0.8)
        ax.text(left_x - 0.02, n_rank, str(r["feature"]), ha="right", va="center", fontsize=8)
        ax.text(right_x + 0.02, p_rank, f"#{int(p_rank)}", ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([left_x, right_x])
    ax.set_xticklabels(["native gain (biased)", "permutation (validation)"])
    ax.set_ylabel("Rank (1 = most important)")
    ax.invert_yaxis()  # rank 1 at the top
    ax.set_title("Native gain vs permutation importance — rank disagreement")
    ax.figure.text(
        0.5, -0.02,
        "Orange = permutation demotes a feature native gain inflated (high-cardinality "
        "bias / correlated-feature dilution). Ranks are associational, within-sample.",
        ha="center", va="top", fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout()
    _save(fig, out)
    return fig

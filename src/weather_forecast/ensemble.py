"""Forecast ensembling — both strategies leakage-safe.

1. Inverse-error weighting (default): weight each base model by 1/backtest-error,
   normalized. Weights come from backtest metrics that PREDATE the final fit, so no
   leakage. Models worse than seasonal-naive (MASE >= 1) are dropped before weighting.

2. Stacking (offered): a non-negative linear meta-learner trained ONLY on out-of-fold
   predictions (each made by a model that never saw that row). Training a stacker on
   in-fold predictions would be leakage; we use the OOF matrix exclusively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_error_weights(error_by_model: dict[str, float], *, alpha: float = 1.0,
                          drop_above: float | None = None,
                          drop_metric: dict[str, float] | None = None) -> dict[str, float]:
    """Normalized inverse-error weights. ``alpha=2`` approximates inverse-variance.

    Optionally drop any model whose ``drop_metric`` (e.g. MASE) exceeds ``drop_above``
    (e.g. 1.0) BEFORE weighting, so a model worse than the baseline never dilutes the blend.
    """
    items = {k: v for k, v in error_by_model.items() if np.isfinite(v) and v > 0}
    if drop_above is not None and drop_metric is not None:
        # Keep only models PROVEN to beat the baseline: gating metric must be finite AND
        # <= threshold. A model with undefined (NaN) MASE was never shown to beat
        # seasonal-naive, so it must not silently enter the blend.
        items = {k: v for k, v in items.items()
                 if np.isfinite(drop_metric.get(k, np.nan)) and drop_metric[k] <= drop_above}
    if not items:
        # everything was dropped — fall back to weighting all finite-error models
        items = {k: v for k, v in error_by_model.items() if np.isfinite(v) and v > 0}
    if not items:
        return {}
    raw = {k: (1.0 / v) ** alpha for k, v in items.items()}
    total = sum(raw.values())
    return {k: w / total for k, w in raw.items()}


def weighted_average(preds_by_model: dict[str, np.ndarray],
                     weights: dict[str, float]) -> np.ndarray | None:
    """Weighted blend of aligned forecast arrays. Uses only models present in both
    ``preds_by_model`` and ``weights``; re-normalizes over that intersection."""
    keys = [k for k in weights if k in preds_by_model]
    if not keys:
        return None
    w = np.array([weights[k] for k in keys], dtype="float64")
    mat = np.vstack([np.asarray(preds_by_model[k], dtype="float64") for k in keys])
    finite = np.isfinite(mat)
    wcol = np.where(finite, w[:, None], 0.0)                 # weight only defined models, per step
    num = np.nansum(np.where(finite, mat, 0.0) * wcol, axis=0)
    den = wcol.sum(axis=0)
    # NaN (not a fictitious 0) where every model is undefined for a step; renormalize otherwise.
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def fit_stacker(oof_wide: pd.DataFrame, y, *, non_negative: bool = True):
    """Fit a meta-learner on the OOF prediction matrix (rows=obs, cols=base models).

    Non-negative linear blend by default — interpretable, can't learn pathological
    negative weights from noise on few folds.
    """
    from sklearn.linear_model import LinearRegression, Ridge

    x = oof_wide.to_numpy("float64")
    yy = np.asarray(y, dtype="float64")
    mask = np.isfinite(yy) & np.isfinite(x).all(axis=1)
    model = LinearRegression(positive=True) if non_negative else Ridge(alpha=1.0)
    model.fit(x[mask], yy[mask])
    return model


def predict_stacker(model, oof_wide: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(oof_wide.to_numpy("float64")), dtype="float64")


def build_oof_matrix(oof_by_model: dict[str, pd.DataFrame],
                     keys=("country", "location_name", "date")) -> tuple[pd.DataFrame, pd.Series]:
    """Align per-model OOF predictions into a wide matrix on ``keys`` for stacking.

    Each value is a frame with columns ``keys + [target, 'yhat']`` (from backtest_global).
    Returns ``(X_wide[model_cols], y)`` inner-joined across models.
    """
    keys = list(keys)
    merged = None
    for name, df in oof_by_model.items():
        if df is None or df.empty:
            continue
        tcol = [c for c in df.columns if c not in keys + ["yhat", "fold"]][0]
        # Guard against overlapping-fold duplicate (city, date) rows before joining.
        if "fold" in df.columns:
            df = df.sort_values("fold")
        df = df.drop_duplicates(subset=keys, keep="last")
        right = df[keys + [tcol, "yhat"]].rename(columns={"yhat": name, tcol: "_y"})
        merged = right if merged is None else merged.merge(right[keys + [name]], on=keys, how="inner")
    if merged is None:
        return pd.DataFrame(), pd.Series(dtype="float64")
    model_cols = [c for c in merged.columns if c not in keys + ["_y"]]
    return merged[model_cols], merged["_y"]

"""Forecast error metrics and naive baselines.

Metric hierarchy (documented in REPORT.md):
- MASE is PRIMARY: scale-free, comparable across cities, and answers the only question
  that matters — "did we beat the seasonal-naive benchmark?" (MASE < 1 = useful).
- MAE / RMSE are reported for absolute interpretability (degrees C).
- sMAPE is reported (bounded). MAPE is reported but GUARDED and de-emphasized:
  temperature crosses zero, so dividing by the actual explodes the percentage.

Baselines are mandatory: a model that cannot beat seasonal-naive is not worth
shipping, and the seasonal-naive in-sample error is the MASE denominator.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="float64")
    yhat = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(yhat)
    return y[mask], yhat[mask]


def mae(y_true, y_pred) -> float:
    y, yhat = _align(y_true, y_pred)
    return float(np.mean(np.abs(y - yhat))) if y.size else float("nan")


def rmse(y_true, y_pred) -> float:
    y, yhat = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if y.size else float("nan")


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; defined (=0 term) when both values are zero."""
    y, yhat = _align(y_true, y_pred)
    if not y.size:
        return float("nan")
    denom = np.abs(y) + np.abs(yhat)
    term = np.zeros_like(denom, dtype="float64")
    nz = denom != 0  # mask the divide so 0/0 never executes (avoids a spurious warning)
    term[nz] = 2.0 * np.abs(y - yhat)[nz] / denom[nz]
    return float(100.0 * np.mean(term))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    """MAPE in percent, GUARDED: only averages over |y| >= eps and warns if many
    denominators are near zero. Temperature crosses 0 C, so raw MAPE is unreliable."""
    y, yhat = _align(y_true, y_pred)
    if not y.size:
        return float("nan")
    mask = np.abs(y) >= eps
    if mask.sum() == 0:
        return float("nan")
    if mask.sum() < y.size * 0.8:
        warnings.warn("MAPE: >20% of actuals are near zero; metric unreliable for this target.",
                      stacklevel=2)
    return float(100.0 * np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def mase(y_true, y_pred, y_train, season_length: int = config.SEASON_LENGTH) -> float:
    """Mean Absolute Scaled Error vs the in-sample seasonal-naive forecast.

    Denominator = mean(|y_train[t] - y_train[t-m]|) on the TRAINING series. MASE < 1
    means the model beats seasonal-naive; >= 1 means ship the baseline instead.
    """
    yt = np.asarray(y_train, dtype="float64")
    yt = yt[np.isfinite(yt)]
    if yt.size <= season_length:
        return float("nan")
    denom = np.mean(np.abs(yt[season_length:] - yt[:-season_length]))
    if denom == 0:
        return float("nan")
    return mae(y_true, y_pred) / denom


def evaluate(y_true, y_pred, y_train=None,
             season_length: int = config.SEASON_LENGTH) -> dict[str, float]:
    """Return all metrics as a dict. MASE included only when ``y_train`` is given."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n": int(np.isfinite(np.asarray(y_true, "float64")).sum()),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, season_length)
    return out


# --------------------------------------------------------------------------- #
# Baselines (train = pd.Series with daily DatetimeIndex; return np.ndarray len h)
# --------------------------------------------------------------------------- #
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Persistence: repeat the last observed value."""
    v = train.dropna().to_numpy()
    if v.size == 0:
        return np.full(horizon, np.nan)
    return np.repeat(v[-1], horizon)


def seasonal_naive_forecast(train: pd.Series, horizon: int,
                            season_length: int = config.SEASON_LENGTH) -> np.ndarray:
    """Repeat the last seasonal cycle (period m); the primary benchmark."""
    v = train.dropna().to_numpy()
    if v.size == 0:
        return np.full(horizon, np.nan)
    m = min(season_length, v.size)
    last_cycle = v[-m:]
    reps = int(np.ceil(horizon / m))
    return np.tile(last_cycle, reps)[:horizon]


def drift_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Linear extrapolation of the average historical trend."""
    v = train.dropna().to_numpy()
    if v.size < 2:
        return naive_forecast(train, horizon)
    slope = (v[-1] - v[0]) / (v.size - 1)
    return v[-1] + slope * np.arange(1, horizon + 1)


def climatology_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Day-of-year climatological mean — the physically meaningful weather benchmark."""
    s = train.dropna()
    if s.empty:
        return np.full(horizon, np.nan)
    clim = s.groupby(s.index.dayofyear).mean()
    overall = float(s.mean())
    future = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return np.array([clim.get(d, overall) for d in future.dayofyear], dtype="float64")


BASELINES = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "drift": drift_forecast,
    "climatology": climatology_forecast,
}

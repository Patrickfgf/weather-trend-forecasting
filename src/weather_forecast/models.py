"""Forecasting models.

Track A (per-city, univariate) — statsmodels ETS and SARIMAX, exposed as
``fn(train_series, horizon) -> np.ndarray`` so they slot into ``backtest_per_city``
alongside the baselines. Each degrades GRACEFULLY: on a short series or a convergence
failure it falls back to seasonal-naive and logs, so a Restart-Run-All never crashes.

Track B (global panel) — gradient boosting on the lag/calendar/static feature matrix.
XGBoost is the required engine (robust 3.14 wheel); LightGBM is optional. Prophet is an
optional per-city extra. Determinism: fixed ``random_state`` and ``tree_method='hist'``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from . import config, metrics, optional_deps

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Track A: per-city classical models (statsmodels)
# --------------------------------------------------------------------------- #
def ets_forecast(train: pd.Series, horizon: int,
                 season_length: int = config.SEASON_LENGTH, damped: bool = True) -> np.ndarray:
    """Holt-Winters ETS (additive trend + additive weekly seasonality, damped).

    Additive (not multiplicative) because temperature in C can be near-zero/negative.
    Falls back to seasonal-naive on short series or fit failure.
    """
    s = train.dropna()
    if s.size < 2 * season_length + 2:
        return metrics.seasonal_naive_forecast(train, horizon, season_length)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                s.to_numpy(), trend="add", damped_trend=damped,
                seasonal="add", seasonal_periods=season_length,
                initialization_method="estimated",
            ).fit()
            return np.asarray(fit.forecast(horizon), dtype="float64")
    except Exception as exc:  # convergence / linalg failure on a noisy short series
        logger.debug("ETS fell back to seasonal-naive: %s", type(exc).__name__)
        return metrics.seasonal_naive_forecast(train, horizon, season_length)


def sarimax_forecast(train: pd.Series, horizon: int,
                     order: tuple[int, int, int] = (1, 1, 1),
                     seasonal_order: tuple[int, int, int, int] = (1, 0, 1, config.SEASON_LENGTH)
                     ) -> np.ndarray:
    """SARIMAX(1,1,1)(1,0,1,7). Stationarity/invertibility unenforced for convergence
    robustness; falls back to seasonal-naive on failure."""
    s = train.dropna()
    if s.size < seasonal_order[3] * 2 + 5:
        return metrics.seasonal_naive_forecast(train, horizon)
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(s.to_numpy(), order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            return np.asarray(fit.forecast(horizon), dtype="float64")
    except Exception as exc:
        logger.debug("SARIMAX fell back to seasonal-naive: %s", type(exc).__name__)
        return metrics.seasonal_naive_forecast(train, horizon)


def prophet_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Optional Prophet (weekly seasonality only — history is sub-annual). Falls back
    to seasonal-naive if Prophet is unavailable or fit fails."""
    if not optional_deps.HAS_PROPHET:
        return metrics.seasonal_naive_forecast(train, horizon)
    s = train.dropna()
    if s.size < 2 * config.SEASON_LENGTH + 2:
        return metrics.seasonal_naive_forecast(train, horizon)
    try:
        from prophet import Prophet
        dfp = pd.DataFrame({"ds": s.index, "y": s.to_numpy()})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
            logging.getLogger("prophet").setLevel(logging.CRITICAL)
            logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=horizon, freq="D")
            yhat = m.predict(future)["yhat"].to_numpy()[-horizon:]
        return np.asarray(yhat, dtype="float64")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Prophet fell back to seasonal-naive: %s", type(exc).__name__)
        return metrics.seasonal_naive_forecast(train, horizon)


def per_city_model_fns(include_prophet: bool = False) -> dict[str, callable]:
    """Roster of per-city forecasters (baselines + classical) for ``backtest_per_city``."""
    fns: dict[str, callable] = dict(metrics.BASELINES)
    fns["ets"] = ets_forecast
    fns["sarimax"] = sarimax_forecast
    if include_prophet and optional_deps.HAS_PROPHET:
        fns["prophet"] = prophet_forecast
    return fns


# --------------------------------------------------------------------------- #
# Track B: global panel gradient boosting
# --------------------------------------------------------------------------- #
def make_xgb(seed: int = config.SEED):
    """Factory for a deterministic XGBoost regressor (handles NaN features natively)."""
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, tree_method="hist", random_state=seed, n_jobs=4,
        objective="reg:squarederror",
    )


def make_lgbm(seed: int = config.SEED):
    """Factory for a LightGBM regressor, or None if LightGBM is unavailable."""
    if not optional_deps.HAS_LIGHTGBM:
        return None
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=-1, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=-1,
    )


def global_model_factories(seed: int = config.SEED) -> dict[str, callable]:
    """Available global-model factories keyed by name (LightGBM included only if present)."""
    facs: dict[str, callable] = {"xgb_global": lambda: make_xgb(seed)}
    if optional_deps.HAS_LIGHTGBM:
        facs["lgbm_global"] = lambda: make_lgbm(seed)
    return facs

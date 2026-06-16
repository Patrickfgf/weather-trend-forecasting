"""Leakage-safe feature engineering: calendar, cyclical encodings, per-city lags and
rolling statistics, and lagged exogenous weather. All functions are pure.

The two leakage traps this module prevents BY CONSTRUCTION:
1. Cross-city bleed: every lag/rolling is computed inside ``groupby(KEY_COLS)`` so one
   city's history never feeds another's row.
2. Same-row leakage: rolling windows ``shift(1)`` BEFORE ``rolling`` so the window
   covers strictly-prior days and never includes the current row's own target.

Exogenous weather (humidity, pressure, ...) is only ever exposed LAGGED — using
tomorrow's humidity to predict tomorrow's temperature would be leakage. See
:func:`feature_columns`, which excludes contemporaneous weather/air-quality columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


# --------------------------------------------------------------------------- #
# Calendar + static features
# --------------------------------------------------------------------------- #
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar parts + hemisphere-aware meteorological season + abs(latitude).

    Season is hemisphere-aware: the Southern hemisphere is shifted 6 months, so
    December is summer in Sydney and winter in London. Pooling them would cancel the
    seasonal signal.
    """
    df = df.copy()
    ts = df[config.TIME_COL]
    df["date"] = ts.dt.tz_convert("UTC").dt.normalize()
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month.astype("int16")
    df["day_of_year"] = ts.dt.dayofyear.astype("int16")
    df["week"] = ts.dt.isocalendar().week.astype("int16")
    df["dayofweek"] = ts.dt.dayofweek.astype("int16")
    df["quarter"] = ts.dt.quarter.astype("int16")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    if "latitude" in df.columns:
        df["abs_latitude"] = df["latitude"].abs()
        southern = df["latitude"] < 0
        eff_month = np.where(southern, ((df["month"] - 1 + 6) % 12) + 1, df["month"])
        df["hemisphere"] = np.where(southern, "S", "N")
        df["season"] = pd.Series(eff_month, index=df.index).map(_SEASON_BY_MONTH).astype("category")
    return df


# --------------------------------------------------------------------------- #
# Cyclical encodings
# --------------------------------------------------------------------------- #
def add_cyclical(df: pd.DataFrame, col: str, period: float) -> pd.DataFrame:
    """Encode a circular feature as sin/cos so e.g. Dec-31 and Jan-1 are adjacent."""
    df = df.copy()
    if col in df.columns:
        rad = 2.0 * np.pi * df[col].astype("float64") / period
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
    return df


def add_calendar_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encode day_of_year (365.25), month (12), dayofweek (7), wind_degree (360)."""
    df = add_cyclical(df, "day_of_year", 365.25)
    df = add_cyclical(df, "month", 12)
    df = add_cyclical(df, "dayofweek", 7)
    df = add_cyclical(df, "wind_degree", 360)
    return df


# --------------------------------------------------------------------------- #
# Lags + rollings (per-city, shifted)
# --------------------------------------------------------------------------- #
def add_lags(df: pd.DataFrame, target: str = config.TARGET,
             lags: list[int] | None = None) -> pd.DataFrame:
    """Per-city lag features. ``shift(k>=1)`` is inherently causal (uses only the past)."""
    lags = lags or config.LAG_DAYS
    df = df.sort_values(config.KEY_COLS + [config.TIME_COL]).copy()
    g = df.groupby(config.KEY_COLS, observed=True)[target]
    for k in lags:
        df[f"{target}_lag{k}"] = g.shift(k)
    return df


def add_rollings(df: pd.DataFrame, target: str = config.TARGET,
                 windows: list[int] | None = None) -> pd.DataFrame:
    """Per-city rolling mean/std/min/max on the SHIFTED series (current row excluded)."""
    windows = windows or config.ROLLING_WINDOWS
    df = df.sort_values(config.KEY_COLS + [config.TIME_COL]).copy()
    grp = df.groupby(config.KEY_COLS, observed=True)[target]
    for w in windows:
        mp = max(2, w // 2)
        df[f"{target}_rollmean{w}"] = grp.transform(
            lambda s, w=w, mp=mp: s.shift(1).rolling(w, min_periods=mp).mean())
        df[f"{target}_rollstd{w}"] = grp.transform(
            lambda s, w=w, mp=mp: s.shift(1).rolling(w, min_periods=mp).std())
        df[f"{target}_rollmin{w}"] = grp.transform(
            lambda s, w=w, mp=mp: s.shift(1).rolling(w, min_periods=mp).min())
        df[f"{target}_rollmax{w}"] = grp.transform(
            lambda s, w=w, mp=mp: s.shift(1).rolling(w, min_periods=mp).max())
    return df


def add_lagged_exog(df: pd.DataFrame, cols: list[str] | None = None, lag: int = 1) -> pd.DataFrame:
    """Add per-city lagged exogenous weather (``<col>_lag{lag}``). Never contemporaneous."""
    cols = cols or config.EXOG_WEATHER_COLS
    df = df.sort_values(config.KEY_COLS + [config.TIME_COL]).copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_lag{lag}"] = df.groupby(config.KEY_COLS, observed=True)[col].shift(lag)
    return df


def build_features(df: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Full feature build: calendar -> cyclical -> lags -> rollings -> lagged exog."""
    return (
        df.pipe(add_calendar)
          .pipe(add_calendar_cyclical)
          .pipe(add_lags, target=target)
          .pipe(add_rollings, target=target)
          .pipe(add_lagged_exog)
    )


def feature_columns(df: pd.DataFrame, target: str = config.TARGET) -> list[str]:
    """Return the model feature columns — LAGGED/rolling/cyclical/calendar/static only.

    Deliberately EXCLUDES contemporaneous weather and air-quality columns (using
    them to forecast the same-timestamp target is leakage) and the leaky/twin columns.
    """
    feats: list[str] = []
    feats += [c for c in df.columns if c.startswith(f"{target}_lag") or c.startswith(f"{target}_roll")]
    feats += [c for c in df.columns if c.endswith("_lag1") and not c.startswith(target)]
    feats += [c for c in df.columns if c.endswith(("_sin", "_cos"))]
    feats += [c for c in ["month", "day_of_year", "dayofweek", "quarter", "week", "is_weekend", "year"]
              if c in df.columns]
    feats += [c for c in ["latitude", "longitude", "abs_latitude"] if c in df.columns]

    leaky = set(config.LEAKY_OR_TWIN_COLS)
    seen: set[str] = set()
    out = []
    for f in feats:
        if f in leaky or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Per-city daily series (Track A: classical TS models)
# --------------------------------------------------------------------------- #
def to_daily_series(df: pd.DataFrame, country: str, location_name: str,
                    value: str = config.TARGET) -> pd.Series:
    """Extract a city's value as a regular daily, tz-naive Series (resample mean -> asfreq D).

    Classical statsmodels models need a tz-naive DatetimeIndex with a regular
    frequency; intra-day duplicate snapshots are collapsed by the daily mean.
    """
    mask = (df["country"] == country) & (df["location_name"] == location_name)
    sub = df.loc[mask, [config.TIME_COL, value]].dropna(subset=[config.TIME_COL])
    if sub.empty:
        return pd.Series(dtype="float64")
    s = sub.set_index(config.TIME_COL)[value].sort_index()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return s.resample("D").mean().asfreq("D")

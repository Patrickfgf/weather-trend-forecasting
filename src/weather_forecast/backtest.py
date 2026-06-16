"""Leakage-safe backtesting on a GLOBAL calendar-date axis.

Two evaluation tracks, identical folds, one comparable metrics table:
- ``backtest_per_city``  — Track A: per-city univariate models / baselines.
- ``backtest_global``    — Track B: one global panel model evaluated per (city, fold).

The split is on ``last_updated`` (a global day), applied identically to every city, so
per-city splits are aligned and no fold ever places a city's future in train. Every
fold asserts ``train.max_day <= cutoff < test.min_day``. There is NO shuffling and NO
``KFold`` anywhere — those would leak the future into training for an autocorrelated
target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: pd.Timestamp          # last training day (inclusive), tz-naive
    horizon: int
    scheme: str = "expanding"
    train_start: pd.Timestamp | None = None
    window_days: int | None = None


def _daily_naive(dates) -> pd.Series:
    """Normalize any datetime-like to tz-naive day (UTC) for comparison with cutoffs."""
    s = pd.to_datetime(pd.Series(np.asarray(dates)))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def make_time_folds(dates, n_splits: int = config.N_BACKTEST_FOLDS,
                    horizon: int = config.HORIZON,
                    min_train_days: int = config.MIN_TRAIN_DAYS,
                    scheme: str = "expanding",
                    window_days: int | None = None) -> list[Fold]:
    """Rolling-origin folds: cutoffs evenly spaced in the back of the global timeline.

    Train = days <= cutoff (expanding) or the last ``window_days`` (rolling); test =
    the ``horizon`` days after the cutoff. ``min_train_days`` guards the first fold.
    """
    days = np.array(sorted(pd.unique(_daily_naive(dates))))
    n = len(days)
    earliest = min_train_days - 1
    latest = n - 1 - horizon
    if latest < earliest:
        if latest < 1:
            logger.warning("[backtest] not enough history for any fold (n_days=%d)", n)
            return []
        earliest = max(1, latest - 1)
    idxs = list(np.unique(np.linspace(earliest, latest, n_splits).round().astype(int)))
    # Enforce non-overlapping test windows: consecutive cutoffs must be >= horizon apart,
    # so a (city, date) is never produced by two folds (which would duplicate OOF rows).
    spaced: list[int] = []
    for i in idxs:
        if not spaced or (i - spaced[-1]) >= horizon:
            spaced.append(int(i))
    idxs = spaced
    folds = []
    for k, i in enumerate(idxs):
        train_start = (pd.Timestamp(days[max(0, i - window_days + 1)])
                       if scheme == "rolling" and window_days else pd.Timestamp(days[0]))
        folds.append(Fold(index=k, cutoff=pd.Timestamp(days[i]), horizon=horizon,
                          scheme=scheme, train_start=train_start, window_days=window_days))
    return folds


def _row(model: str, country: str, loc: str, f: Fold,
         y_true, y_pred, y_train, season_length: int) -> dict:
    met = metrics.evaluate(y_true, y_pred, y_train=y_train, season_length=season_length)
    return {"model": model, "country": country, "location_name": loc,
            "fold": f.index, "cutoff": f.cutoff, "horizon": f.horizon, **met}


def backtest_per_city(series_by_city: dict[tuple[str, str], pd.Series],
                      model_fns: dict[str, callable],
                      folds: list[Fold],
                      season_length: int = config.SEASON_LENGTH) -> pd.DataFrame:
    """Evaluate per-city forecasters. ``model_fns[name](train_series, horizon)->array``.

    Returns a long metrics frame: one row per (model, city, fold).
    """
    rows: list[dict] = []
    for (country, loc), s in series_by_city.items():
        for f in folds:
            train = s[s.index <= f.cutoff]
            if f.scheme == "rolling" and f.train_start is not None:
                train = train[train.index >= f.train_start]
            train = train.dropna()
            if train.size < season_length + 2:
                continue
            test_days = pd.date_range(f.cutoff + pd.Timedelta(days=1), periods=f.horizon, freq="D")
            y_test = s.reindex(test_days).to_numpy()
            if not np.isfinite(y_test).any():
                continue
            assert train.index.max() <= f.cutoff < test_days.min(), "fold leakage: train overlaps test"
            for name, fn in model_fns.items():
                try:
                    yhat = np.asarray(fn(train, f.horizon), dtype="float64")
                except Exception as exc:  # a per-series model may fail to converge
                    logger.warning("[backtest] %s failed on %s/%s fold %d: %s",
                                   name, country, loc, f.index, type(exc).__name__)
                    yhat = np.full(f.horizon, np.nan)
                rows.append(_row(name, country, loc, f, y_test, yhat, train.to_numpy(), season_length))
    return pd.DataFrame(rows)


def backtest_global(feats: pd.DataFrame, feature_cols: list[str], model_factory: callable,
                    folds: list[Fold], *, model_name: str = "xgb_global",
                    target: str = config.TARGET, season_length: int = config.SEASON_LENGTH,
                    series_by_city: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ONE global panel model as a genuine DIRECT multi-step forecaster.

    Critical correctness point: we do NOT predict the test window row-by-row using each
    row's own lag features — that would let the row at cutoff+5 read the ACTUAL
    temperature at cutoff+4 (intra-horizon leakage that the fold-boundary assertion can't
    catch). Instead we train one direct model with ``horizon`` as a feature on labels
    shifted h days ahead, and at test time forecast cutoff+1..cutoff+H purely from features
    known AT the origin (cutoff). ``y_origin`` (the origin day's own temperature) is the
    most-recent known value, mirroring persistence. This makes Track B a true h-step
    forecaster, directly comparable to the per-city baselines.

    Returns ``(metrics_long, oof_predictions)``; OOF predictions feed leakage-free stacking.
    """
    horizon = folds[0].horizon if folds else config.HORIZON
    df = feats.sort_values(config.KEY_COLS + [config.TIME_COL]).copy()
    df["_day"] = _daily_naive(df["date"]).to_numpy()
    grp_target = df.groupby(config.KEY_COLS, observed=True)[target]

    # Direct multi-horizon long frame: one block per horizon h, label = value h days ahead
    # (per city); features are taken AS OF the origin row (all causal, all <= origin day).
    blocks = []
    for h in range(1, horizon + 1):
        block = df[config.KEY_COLS + ["_day"] + feature_cols].copy()
        block["horizon"] = h
        block["y_origin"] = df[target].to_numpy()
        block["_label"] = grp_target.shift(-h).to_numpy()
        block["_label_day"] = df["_day"].to_numpy() + np.timedelta64(h, "D")
        blocks.append(block)
    long = pd.concat(blocks, ignore_index=True)
    feat_cols_h = feature_cols + ["horizon", "y_origin"]

    rows: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    for f in folds:
        cutoff = np.datetime64(f.cutoff)
        train_mask = (long["_label_day"] <= cutoff) & long["_label"].notna() & long["y_origin"].notna()
        if f.scheme == "rolling" and f.train_start is not None:
            train_mask &= long["_day"] >= np.datetime64(f.train_start)
        # Test origins are exactly at the cutoff day -> forecast cutoff+1..cutoff+H.
        test_mask = (long["_day"] == cutoff) & long["_label"].notna()
        if int(train_mask.sum()) < 50 or int(test_mask.sum()) == 0:
            continue
        # Leakage guard: every training label lands on/before the cutoff origin.
        assert long.loc[train_mask, "_label_day"].max() <= cutoff, "direct backtest leak: label after cutoff"

        model = model_factory()
        model.fit(long.loc[train_mask, feat_cols_h].to_numpy("float64"),
                  long.loc[train_mask, "_label"].to_numpy("float64"))
        te = long.loc[test_mask].copy()
        te["yhat"] = np.asarray(model.predict(te[feat_cols_h].to_numpy("float64")), dtype="float64")

        oof_parts.append(
            te.rename(columns={"_label": target, "_label_day": "date"})[
                config.KEY_COLS + ["date", target, "yhat"]].assign(fold=f.index))

        for (country, loc), g in te.groupby(config.KEY_COLS, observed=True):
            train_vals = None
            if series_by_city is not None:
                ser = series_by_city.get((country, loc))
                if ser is not None:
                    train_vals = ser[ser.index <= f.cutoff].to_numpy()
            rows.append(_row(model_name, country, loc, f,
                             g["_label"].to_numpy(), g["yhat"].to_numpy(), train_vals, season_length))

    metrics_long = pd.DataFrame(rows)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return metrics_long, oof


def summarize(metrics_long: pd.DataFrame) -> pd.DataFrame:
    """Macro-average metrics per model across (city, fold). Sorted by MASE ascending."""
    if metrics_long.empty:
        return metrics_long
    agg = (metrics_long
           .groupby("model", observed=True)
           .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), smape=("smape", "mean"),
                mase=("mase", "mean"), mase_std=("mase", "std"), n_obs=("n", "sum"))
           .reset_index())
    sort_key = "mase" if agg["mase"].notna().any() else "mae"
    return agg.sort_values(sort_key).reset_index(drop=True)

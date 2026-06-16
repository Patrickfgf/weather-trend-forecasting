"""Metrics + leakage-safe backtest tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_forecast import backtest, config, metrics
from weather_forecast import engineering as fe


def test_mae_matches_hand_computed_value():
    assert abs(metrics.mae([1, 2, 3], [1, 2, 5]) - (2 / 3)) < 1e-9


def test_smape_handles_zero_actuals_without_error():
    assert np.isfinite(metrics.smape([0, 0, 2], [0, 1, 2]))


def test_mase_is_one_when_model_mae_matches_inseason_seasonal_naive():
    # train diffs at lag 7 are constant 7 -> in-sample seasonal-naive MAE denom = 7
    train = np.arange(20.0)
    y, yhat = np.array([10.0, 10.0]), np.array([3.0, 17.0])  # errors 7 and 7 -> mae 7
    assert abs(metrics.mase(y, yhat, train, season_length=7) - 1.0) < 1e-9


def test_mase_is_zero_for_a_perfect_forecast():
    train = np.arange(20.0)
    y = np.array([5.0, 6.0])
    assert metrics.mase(y, y, train, season_length=7) == 0.0


def test_time_folds_have_train_strictly_before_test_and_no_overlap():
    dates = pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC")
    folds = backtest.make_time_folds(pd.Series(np.repeat(dates, 2)),
                                     n_splits=3, horizon=7, min_train_days=60)
    assert len(folds) >= 1
    cutoffs = [f.cutoff for f in folds]
    assert cutoffs == sorted(cutoffs) and len(set(cutoffs)) == len(cutoffs)  # strictly increasing
    for f in folds:
        assert f.horizon == 7
        assert f.train_start <= f.cutoff


def test_backtest_per_city_produces_metric_rows(clean_df):
    keys = list(clean_df[config.KEY_COLS].drop_duplicates().itertuples(index=False, name=None))[:3]
    series = {k: fe.to_daily_series(clean_df, k[0], k[1]) for k in keys}
    folds = backtest.make_time_folds(clean_df[config.TIME_COL], n_splits=2, horizon=7)
    res = backtest.backtest_per_city(series, {"seasonal_naive": metrics.seasonal_naive_forecast}, folds)
    assert not res.empty
    assert {"model", "mae", "mase", "fold"}.issubset(res.columns)

"""Ensemble contract tests: baseline-gated weighting, NaN-safe blending, OOF dedup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_forecast import ensemble


def test_inverse_error_weights_drop_models_worse_than_baseline():
    w = ensemble.inverse_error_weights({"good": 1.0, "bad": 2.0}, drop_above=1.0,
                                       drop_metric={"good": 0.8, "bad": 1.3})
    assert set(w) == {"good"} and abs(sum(w.values()) - 1.0) < 1e-9


def test_inverse_error_weights_drop_models_with_undefined_mase():
    # undefined (NaN) MASE -> never proven to beat the baseline -> must be dropped
    w = ensemble.inverse_error_weights({"good": 1.0, "unknown": 1.0}, drop_above=1.0,
                                       drop_metric={"good": 0.8, "unknown": float("nan")})
    assert "unknown" not in w


def test_weighted_average_renormalizes_over_defined_models_per_step():
    preds = {"a": np.array([10.0, np.nan]), "b": np.array([20.0, 30.0])}
    out = ensemble.weighted_average(preds, {"a": 0.5, "b": 0.5})
    assert abs(out[0] - 15.0) < 1e-9   # both defined -> mean
    assert abs(out[1] - 30.0) < 1e-9   # only b defined -> b (no 0.0 contamination)


def test_weighted_average_all_nan_step_is_nan_not_zero():
    out = ensemble.weighted_average({"a": np.array([np.nan]), "b": np.array([np.nan])},
                                    {"a": 0.5, "b": 0.5})
    assert np.isnan(out[0])            # never a fictitious 0.0 written to predictions


def test_build_oof_matrix_dedups_overlapping_folds_and_aligns():
    a = pd.DataFrame({"country": ["A", "A"], "location_name": ["X", "X"], "date": [1, 1],
                      "temperature_celsius": [5.0, 5.0], "yhat": [4.0, 4.5], "fold": [0, 1]})
    b = pd.DataFrame({"country": ["A"], "location_name": ["X"], "date": [1],
                      "temperature_celsius": [5.0], "yhat": [4.2], "fold": [0]})
    x_wide, y = ensemble.build_oof_matrix({"m1": a, "m2": b})
    assert len(x_wide) == 1 and list(x_wide.columns) == ["m1", "m2"]  # deduped to one (city,date)
    assert float(y.iloc[0]) == 5.0

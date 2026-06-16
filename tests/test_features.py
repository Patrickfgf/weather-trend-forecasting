"""Feature-engineering tests — the leakage guards are the most important in the suite.

A model that silently sees the future scores great in backtest and fails in production;
these tests pin the causal contract of lags/rollings and the contemporaneous-exclusion
of exogenous/leaky columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_forecast import config
from weather_forecast import engineering as fe
from weather_forecast import ingest


def _toy() -> pd.DataFrame:
    """2 cities x 10 daily rows; temperature == day index (+100 offset for city 2),
    so lag/rolling values are exactly predictable."""
    rows = []
    for ci, (country, loc) in enumerate([("A", "X"), ("B", "Y")]):
        for d in range(10):
            rows.append({
                "country": country, "location_name": loc,
                "last_updated": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=d),
                "latitude": 10.0 if ci == 0 else -10.0, "longitude": 0.0,
                "temperature_celsius": float(d + 100 * ci),
            })
    df = pd.DataFrame(rows)
    df["country"] = df["country"].astype("category")
    df["location_name"] = df["location_name"].astype("category")
    return df


def test_normalize_columns_renames_and_is_idempotent():
    df = pd.DataFrame(columns=["air_quality_PM2.5", "Last Updated", "air_quality_us-epa-index"])
    once = ingest.normalize_columns(df)
    twice = ingest.normalize_columns(once)
    assert list(once.columns) == ["air_quality_pm2_5", "last_updated", "air_quality_us_epa_index"]
    assert list(once.columns) == list(twice.columns)


def test_lag_feature_uses_only_prior_value_within_city():
    df = fe.add_lags(_toy(), lags=[1])
    x = df[df.location_name == "X"].sort_values("last_updated")
    assert x["temperature_celsius_lag1"].iloc[1:].tolist() == x["temperature_celsius"].shift(1).iloc[1:].tolist()
    assert np.isnan(x["temperature_celsius_lag1"].iloc[0])  # first row has no past


def test_lag_does_not_bleed_across_city_boundaries():
    df = fe.add_lags(_toy(), lags=[1])
    y = df[df.location_name == "Y"].sort_values("last_updated")
    # City Y's first row must be NaN, NOT city X's last value (no cross-city bleed).
    assert np.isnan(y["temperature_celsius_lag1"].iloc[0])


def test_rolling_mean_is_shifted_so_current_row_is_excluded():
    df = fe.add_rollings(_toy(), windows=[3])
    x = df[df.location_name == "X"].sort_values("last_updated").reset_index(drop=True)
    # temp == day index; rollmean3 at row 3 uses days {0,1,2} -> mean 1.0 (current row 3 excluded).
    assert abs(x["temperature_celsius_rollmean3"].iloc[3] - 1.0) < 1e-9
    # very first row has no prior window -> NaN.
    assert np.isnan(x["temperature_celsius_rollmean3"].iloc[0])


def test_feature_columns_exclude_contemporaneous_and_leaky(feats_df):
    cols = fe.feature_columns(feats_df)
    assert "humidity" not in cols and "humidity_lag1" in cols      # only lagged exog
    assert not any("feels_like" in c for c in cols)                # leaky twin excluded
    assert config.TARGET not in cols                               # never the target itself


def test_cyclical_encoding_makes_year_boundary_adjacent():
    out = fe.add_cyclical(pd.DataFrame({"day_of_year": [365, 1]}), "day_of_year", 365.25)
    dist = np.hypot(out["day_of_year_sin"].iloc[0] - out["day_of_year_sin"].iloc[1],
                    out["day_of_year_cos"].iloc[0] - out["day_of_year_cos"].iloc[1])
    assert dist < 0.05  # Dec-31 and Jan-1 are neighbors on the circle


def test_season_is_hemisphere_aware(feats_df):
    # January in the Northern hemisphere is Winter; in the Southern hemisphere it's Summer.
    jan = feats_df[feats_df.month == 1]
    if not jan.empty and jan["hemisphere"].nunique() == 2:
        north = set(jan[jan.hemisphere == "N"]["season"].astype(str))
        south = set(jan[jan.hemisphere == "S"]["season"].astype(str))
        assert "Winter" in north and "Summer" in south

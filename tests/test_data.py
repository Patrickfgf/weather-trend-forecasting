"""Ingestion + cleaning contract tests (run on the synthetic fixture)."""

from __future__ import annotations

from weather_forecast import config


def test_imperial_twins_and_epoch_dropped(loaded_df):
    assert not any(c in loaded_df.columns for c in config.DROP_IMPERIAL)
    assert "last_updated_epoch" not in loaded_df.columns


def test_last_updated_is_tz_aware(loaded_df):
    assert getattr(loaded_df[config.TIME_COL].dt, "tz", None) is not None


def test_epa_index_is_nullable_int(loaded_df):
    assert str(loaded_df["air_quality_us_epa_index"].dtype) == "Int16"


def test_clean_grain_is_unique(clean_df):
    assert not clean_df.duplicated(config.KEY_COLS + [config.TIME_COL]).any()


def test_physical_bounds_hold_after_clean(clean_df):
    assert clean_df[config.TARGET].dropna().between(-90, 60).all()
    assert clean_df["humidity"].dropna().between(0, 100).all()


def test_continuous_columns_imputed(clean_df):
    # per-city time interpolation should leave no gaps in continuous weather
    assert clean_df["humidity"].isna().sum() == 0


def test_outlier_and_missing_flags_present(clean_df):
    assert f"{config.TARGET}_is_outlier" in clean_df.columns
    assert any(c.endswith("_was_missing") for c in clean_df.columns)

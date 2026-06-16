"""Shared pytest fixtures. Tests run on a SMALL synthetic frame and NEVER read the
real Kaggle CSV, so the suite is fast and hermetic (no 100MB download needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from weather_forecast import cleaning
from weather_forecast import engineering as fe
from weather_forecast import ingest, synthetic


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Synthetic RAW frame (un-normalized column names), small + deterministic."""
    return synthetic.generate_synthetic_weather(n_days=90, seed=123)


@pytest.fixture(scope="session")
def loaded_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw frame put through the ingestion edge (normalize + dtype + validate)."""
    return raw_df.pipe(ingest.normalize_columns).pipe(ingest.coerce_dtypes).pipe(ingest.validate)


@pytest.fixture(scope="session")
def clean_df(loaded_df: pd.DataFrame) -> pd.DataFrame:
    return cleaning.clean(loaded_df)


@pytest.fixture(scope="session")
def feats_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    return fe.build_features(clean_df)

"""Ingestion edge: load the raw CSV (or synthetic fallback), normalize the schema,
coerce dtypes, and validate. This is the ONLY module that reads the raw data file.

Pipeline at this edge::

    load_weather()
        -> resolve_source()                 # real Kaggle CSV in data/raw, else synthetic
        -> normalize_columns()              # air_quality_PM2.5 -> air_quality_pm2_5 ...
        -> coerce_dtypes()                  # tz-aware UTC time, drop imperial twins, set categoricals
        -> validate(check_grain=False)      # schema/range checks (grain asserted post-dedup in cleaning)

Returns ``(df, is_real)`` so the rest of the pipeline can log which source it ran on.
Grain after this stage: still one row per raw snapshot (duplicates removed later in
``weather_forecast.cleaning``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema normalization
# --------------------------------------------------------------------------- #
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized header normalization to snake_case (idempotent).

    ``air_quality_PM2.5`` -> ``air_quality_pm2_5``; ``air_quality_us-epa-index`` ->
    ``air_quality_us_epa_index``. A deterministic regex, not a hand-built rename dict
    that rots when the source reorders/renames a column.
    """
    def _norm(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[.\-\s]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        return c

    return df.rename(columns={c: _norm(c) for c in df.columns})


# --------------------------------------------------------------------------- #
# Dtype coercion
# --------------------------------------------------------------------------- #
def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set explicit dtypes; parse ``last_updated`` as tz-aware UTC; drop redundant twins.

    Returns a NEW frame (no in-place mutation). Only operates on columns that exist,
    so it is robust to minor schema drift in the source file.
    """
    df = df.copy()

    # tz-aware UTC timestamp — the only correct global ordering key across timezones.
    if config.TIME_COL in df.columns:
        df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], utc=True, errors="coerce")

    # Drop imperial unit-twins (exact transforms of metric cols) + redundant epoch.
    drop = [c for c in (config.DROP_IMPERIAL + config.DROP_REDUNDANT) if c in df.columns]
    df = df.drop(columns=drop)

    # Numerics -> float (coerce non-numeric strings to NaN).
    for col in config.METRIC_NUMERIC + config.AQI_CONCENTRATION_COLS + config.GEO_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Ordinal indices -> nullable Int16 (1..6 / 1..10; Int16 tolerates a schema-drift
    # sentinel without the silent Int8 overflow risk, at negligible memory cost).
    for col in config.AQI_INDEX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int16")

    # Low-cardinality text -> category (memory + clear semantics).
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# --------------------------------------------------------------------------- #
# Validation (lightweight STRUCTURAL checks on the raw frame)
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, *, check_grain: bool = False) -> pd.DataFrame:
    """Lightweight structural validation of the RAW frame (asserts only).

    Range/physical-bound and grain checks are intentionally deferred: the raw frame
    legitimately contains out-of-range sensor glitches and duplicate snapshots, which
    are gated / de-duplicated in :mod:`weather_forecast.cleaning`. Strict schema +
    range + grain validation (pandera when available) runs there, on the CLEAN frame
    (see ``cleaning.validate_clean``).
    """
    assert config.TIME_COL in df.columns, f"missing required column {config.TIME_COL!r}"
    assert config.TARGET in df.columns, f"missing required column {config.TARGET!r}"
    ts = df[config.TIME_COL]
    assert getattr(ts.dt, "tz", None) is not None, "last_updated must be tz-aware (UTC)"

    n_bad_ts = int(ts.isna().sum())
    if n_bad_ts:
        logger.warning("[ingest] %d rows have unparseable %s (NaT)", n_bad_ts, config.TIME_COL)

    if check_grain:
        dup = df.duplicated(subset=config.KEY_COLS + [config.TIME_COL], keep=False)
        assert not dup.any(), f"grain violated: {int(dup.sum())} duplicate key rows"
    return df


# --------------------------------------------------------------------------- #
# Source resolution + load
# --------------------------------------------------------------------------- #
def find_raw_csv() -> Path | None:
    """Return the real raw CSV path under data/raw if present, else None."""
    if not config.RAW_DIR.exists():
        return None
    for name in config.RAW_CSV_CANDIDATES:
        p = config.RAW_DIR / name
        if p.exists():
            return p
    # Fall back to any *.csv the user dropped under data/raw (excluding our synthetic file).
    csvs = sorted(c for c in config.RAW_DIR.glob("*.csv") if "synthetic" not in c.name.lower())
    return csvs[0] if csvs else None


def load_weather(*, synthetic_kwargs: dict | None = None) -> tuple[pd.DataFrame, bool]:
    """Load weather data -> ``(df, is_real)``.

    Prefers the real Kaggle CSV in ``data/raw``; otherwise generates a deterministic
    synthetic dataset so the pipeline/notebook/tests run with no manual download.
    The returned frame is schema-normalized, dtyped and schema-validated.
    """
    path = find_raw_csv()
    if path is not None:
        logger.info("[ingest] REAL Kaggle CSV: %s", path.name)
        raw = pd.read_csv(path)
        is_real = True
    else:
        from . import synthetic
        logger.warning("[ingest] no raw CSV found in %s -> SYNTHETIC fallback (dev/CI mode). "
                        "Drop the Kaggle file in data/raw to use real data.", config.RAW_DIR)
        raw = synthetic.generate_synthetic_weather(**(synthetic_kwargs or {}))
        is_real = False

    df = raw.pipe(normalize_columns).pipe(coerce_dtypes).pipe(validate, check_grain=False)
    logger.info("[ingest] loaded %d rows x %d cols (is_real=%s)", len(df), df.shape[1], is_real)
    return df, is_real

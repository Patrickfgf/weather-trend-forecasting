"""Cleaning: de-duplication, physical-bound gating, per-city time-aware imputation,
outlier flagging, and strict (pandera) validation of the CLEAN frame.

All functions are pure: DataFrame in -> NEW DataFrame out (copy-on-write safe, no
in-place mutation). The orchestrator is :func:`clean`.

Leakage stance (documented):
- The TARGET (``temperature_celsius``) is NEVER imputed — inventing labels inflates
  metrics. Rows with a missing/invalid target are kept here (EDA needs them) and
  dropped only at model-fit time.
- Exogenous weather is imputed for clean EDA/analysis, but the FORECAST only ever
  consumes it LAGGED (see :mod:`weather_forecast.engineering`), so contemporaneous
  imputation cannot leak into the temperature forecast.
- Outliers are FLAGGED, not clipped: extreme weather is the signal a forecaster must
  capture. Only physically impossible values (sensor errors) are gated to NaN.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# Continuous columns imputed by per-city time interpolation (target + structural-zero
# precip handled separately).
_CONTINUOUS_IMPUTE = [c for c in config.METRIC_NUMERIC if c not in (config.TARGET, "precip_mm")]
_STRUCTURAL_ZERO = ["precip_mm"]


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, then collapse key-collisions keeping the latest.

    Distinguishes harmless exact re-ingest dupes from genuine key collisions (same
    ``(country, location_name, last_updated)``, differing values) — the latter are
    logged and resolved by keep-last, never silently averaged.
    """
    key = config.KEY_COLS + [config.TIME_COL]
    n0 = len(df)
    df = df.drop_duplicates()
    n_exact = n0 - len(df)

    collisions = int(df.duplicated(subset=key, keep=False).sum())
    if collisions:
        logger.warning("[clean] %d key-collision rows (same %s, differing values) -> keep last",
                       collisions, key)
    df = (df.sort_values(config.KEY_COLS + [config.TIME_COL], kind="stable")
            .drop_duplicates(subset=key, keep="last"))
    if n_exact:
        logger.info("[clean] dropped %d exact duplicate rows", n_exact)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Physical-bound gating (sensor errors -> NaN, NOT clipped)
# --------------------------------------------------------------------------- #
def gate_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Set physically impossible values to NaN (e.g. temp 150 C). These are sensor
    errors, distinct from rare-but-real extremes (which are flagged, not removed)."""
    df = df.copy()
    total = 0
    for col, (lo, hi) in config.PHYSICAL_BOUNDS.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            n = int(bad.sum())
            if n:
                df.loc[bad, col] = np.nan
                total += n
    if total:
        logger.info("[clean] gated %d out-of-physical-range values to NaN", total)
    return df


# --------------------------------------------------------------------------- #
# Imputation (per-city, time-aware)
# --------------------------------------------------------------------------- #
def _interp_group(s: pd.Series) -> pd.Series:
    # linear time-interpolation with a short gap limit, then edge fill.
    return (s.interpolate(method="linear", limit=2, limit_direction="both")
             .ffill().bfill())


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city, time-aware imputation. Returns a NEW frame.

    - precip_mm: fill 0 (missing precip = 'no rain reported', a structural zero).
    - continuous weather: per-city linear time-interp (gap<=2) then edge fill.
    - air quality: add ``<col>_was_missing`` flag (missingness is informative), then
      per-city short-gap fill; long gaps stay NaN.
    - categorical: explicit 'unknown' level.
    - target: NOT imputed.
    """
    df = df.sort_values(config.KEY_COLS + [config.TIME_COL]).copy()
    gkey = config.KEY_COLS

    for col in _STRUCTURAL_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in _CONTINUOUS_IMPUTE:
        if col in df.columns:
            df[col] = df.groupby(gkey, observed=True)[col].transform(_interp_group)

    for col in config.AQI_COLS:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype("int8")
            filled = df.groupby(gkey, observed=True)[col].transform(
                lambda s: s.interpolate(method="linear", limit=2).ffill()
            )
            df[col] = filled if col in config.AQI_CONCENTRATION_COLS else filled.round().astype("Int8")

    for col in config.CATEGORICAL_COLS:
        if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
            if "unknown" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("unknown")
            df[col] = df[col].fillna("unknown")

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Outlier flagging (per city-month robust z; flag-and-keep)
# --------------------------------------------------------------------------- #
def flag_outliers(df: pd.DataFrame, cols: list[str] | None = None,
                  z_thresh: float = config.MAD_Z_THRESH) -> pd.DataFrame:
    """Add boolean ``<col>_is_outlier`` via per-(city, month) modified z-score.

    Robust (median/MAD) and season-aware: a 35 C reading is normal in Cairo, an
    outlier in Reykjavik — comparing within (city, month) prevents flagging every
    legitimately hot/cold city. Does NOT modify the underlying values.
    """
    cols = cols or [c for c in config.METRIC_NUMERIC if c in df.columns]
    df = df.copy()
    month = df[config.TIME_COL].dt.month.rename("_month")
    gcols = config.KEY_COLS + ["_month"]
    tmp = df.assign(_month=month)

    for col in cols:
        grp = tmp.groupby(gcols, observed=True)[col]
        med = grp.transform("median")
        mad = grp.transform(lambda s: (s - s.median()).abs().median())
        # 1.4826*MAD ~ sigma; modified z = 0.6745*(x-med)/MAD
        z = np.where(mad > 0, 0.6745 * (df[col] - med) / mad, 0.0)
        df[f"{col}_is_outlier"] = (np.abs(z) > z_thresh) & df[col].notna()
    return df


# --------------------------------------------------------------------------- #
# Strict validation of the CLEAN frame (pandera when available)
# --------------------------------------------------------------------------- #
def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strict schema + range + grain validation on the cleaned frame.

    Uses pandera when available (declarative, collects all failures), else explicit
    asserts. Ranges hold here because physical-bound gating already ran. Grain is
    unique here because dedup already ran.
    """
    # Grain (always): one row per (country, location_name, last_updated).
    dup = df.duplicated(subset=config.KEY_COLS + [config.TIME_COL], keep=False)
    assert not dup.any(), f"clean grain violated: {int(dup.sum())} duplicate key rows"

    from . import optional_deps
    if optional_deps.HAS_PANDERA:
        try:
            import pandera.pandas as pa
            from pandera import Check, Column
            schema = pa.DataFrameSchema(
                {
                    config.TARGET: Column(float, Check.in_range(-90, 60), nullable=True),
                    "humidity": Column(float, Check.in_range(0, 100), nullable=True, required=False),
                    "cloud": Column(float, Check.in_range(0, 100), nullable=True, required=False),
                    "wind_degree": Column(float, Check.in_range(0, 360), nullable=True, required=False),
                    "precip_mm": Column(float, Check.ge(0), nullable=True, required=False),
                },
                strict=False, coerce=False,
            )
            schema.validate(df, lazy=True)
        except Exception as exc:  # pragma: no cover - surfaces as a hard signal
            raise AssertionError(f"clean-frame schema validation failed: {exc}") from exc
    else:
        t = df[config.TARGET].dropna()
        assert t.between(-90, 60).all(), "temperature out of physical bounds after cleaning"
    return df


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedup -> gate -> impute -> flag outliers -> validate."""
    return (
        df.pipe(dedup)
          .pipe(gate_physical_bounds)
          .pipe(impute_missing)
          .pipe(flag_outliers)
          .pipe(validate_clean)
    )

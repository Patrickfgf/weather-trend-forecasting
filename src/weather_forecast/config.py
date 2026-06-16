"""Single source of truth for the Weather Trend Forecasting project.

Everything downstream (cleaning, features, models, EDA, analysis, notebook, tests)
imports its column names, grain/key, seeds, horizon and the PM Accelerator mission
from here. No other module hard-codes a column name or a path. This is what keeps
the package internally consistent and reproducible.

Design rules encoded here:
- Canonical column names are ``snake_case`` (the raw CSV's ``air_quality_PM2.5`` /
  ``air_quality_us-epa-index`` are normalized at the ingestion edge — see
  ``weather_forecast.ingest.normalize_columns``).
- Paths are relative to the project root, derived from this file's location
  (``parents[2]``) so the code runs identically from CLI, notebook or pytest CWD.
- This module is a *leaf*: it only declares constants and never touches disk on
  import (directories are created lazily by the I/O edge modules).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (anchored to project root; no absolute paths committed)
# --------------------------------------------------------------------------- #
# src/weather_forecast/config.py -> project root is two levels above the package dir
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# The real Kaggle file is probed under RAW_DIR by these candidate names (any match
# wins); if none exist, ingestion falls back to a deterministic synthetic dataset.
RAW_CSV_CANDIDATES: tuple[str, ...] = (
    "GlobalWeatherRepository.csv",
    "Global Weather Repository.csv",
    "global_weather_repository.csv",
)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Grain, key and target (LOCKED — see docs/TECHNICAL_SPEC.md)
# --------------------------------------------------------------------------- #
# Grain: 1 row = one city's weather snapshot at one ``last_updated`` instant.
KEY_COLS: list[str] = ["country", "location_name"]   # entity (series) key
TIME_COL: str = "last_updated"                        # daily snapshot timestamp (tz-aware UTC)
TARGET: str = "temperature_celsius"

# --------------------------------------------------------------------------- #
# Column contract (canonical snake_case names AFTER normalization)
# --------------------------------------------------------------------------- #
GEO_COLS: list[str] = ["latitude", "longitude"]

# Metric (SI) numeric weather columns kept for analysis/modeling.
METRIC_NUMERIC: list[str] = [
    "temperature_celsius",
    "feels_like_celsius",
    "wind_kph",
    "wind_degree",
    "pressure_mb",
    "precip_mm",
    "humidity",
    "cloud",
    "visibility_km",
    "uv_index",
    "gust_kph",
]

# Air-quality pollutant concentrations (continuous, right-skewed).
AQI_CONCENTRATION_COLS: list[str] = [
    "air_quality_carbon_monoxide",
    "air_quality_ozone",
    "air_quality_nitrogen_dioxide",
    "air_quality_sulphur_dioxide",
    "air_quality_pm2_5",
    "air_quality_pm10",
]
# Ordinal air-quality indices (1..6 / 1..N), nullable integers.
AQI_INDEX_COLS: list[str] = [
    "air_quality_us_epa_index",
    "air_quality_gb_defra_index",
]
AQI_COLS: list[str] = AQI_CONCENTRATION_COLS + AQI_INDEX_COLS

CATEGORICAL_COLS: list[str] = [
    "country",
    "location_name",
    "timezone",
    "condition_text",
    "wind_direction",
    "moon_phase",
]

# Imperial unit-twins: exact linear transforms of metric columns. Dropped at the
# ingestion edge — keeping both injects perfect multicollinearity and, for the
# Fahrenheit twin, leaks the target.
DROP_IMPERIAL: list[str] = [
    "temperature_fahrenheit",
    "feels_like_fahrenheit",
    "wind_mph",
    "pressure_in",
    "precip_in",
    "visibility_miles",
    "gust_mph",
]

# Redundant with `last_updated`; dropped on load.
DROP_REDUNDANT: list[str] = ["last_updated_epoch"]

# Columns that must NEVER enter the feature matrix / importance for a TEMPERATURE
# forecast: the Fahrenheit twin IS the target; `feels_like_*` is a near-deterministic
# contemporaneous function of temperature (using it = target leakage).
LEAKY_OR_TWIN_COLS: list[str] = [
    "temperature_fahrenheit",
    "feels_like_celsius",
    "feels_like_fahrenheit",
]

# Exogenous weather usable ONLY as lagged features (never contemporaneous) in the
# temperature forecast — you do not know tomorrow's humidity at predict time.
EXOG_WEATHER_COLS: list[str] = ["humidity", "pressure_mb", "cloud", "wind_kph", "precip_mm"]

# Columns whose missingness is structurally informative -> add `<col>_was_missing`.
MISSING_FLAG_COLS: list[str] = AQI_COLS

# Physical plausibility bounds (hard sensor-error gates -> set to NaN, not clipped).
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_celsius": (-90.0, 60.0),
    "feels_like_celsius": (-95.0, 75.0),
    "humidity": (0.0, 100.0),
    "cloud": (0.0, 100.0),
    "wind_degree": (0.0, 360.0),
    "wind_kph": (0.0, 500.0),
    "gust_kph": (0.0, 600.0),
    "pressure_mb": (850.0, 1090.0),
    "precip_mm": (0.0, 2000.0),
    "uv_index": (0.0, 20.0),
    "visibility_km": (0.0, 100.0),
}

# --------------------------------------------------------------------------- #
# Time-series / backtesting configuration
# --------------------------------------------------------------------------- #
HORIZON: int = 7                       # primary forecast horizon (days)
HORIZONS_REPORT: list[int] = [1, 7, 14]
N_BACKTEST_FOLDS: int = 4
MIN_TRAIN_DAYS: int = 60               # guard SARIMAX/lag features against thin history
SEASON_LENGTH: int = 7                 # weekly cycle: the only reliable period on short history
TEST_FRACTION: float = 0.2             # chronological holdout for the final fit

# Feature engineering knobs
LAG_DAYS: list[int] = [1, 2, 3, 7, 14]
ROLLING_WINDOWS: list[int] = [3, 7, 14]

# --------------------------------------------------------------------------- #
# Representative cities for the per-city deep-dive (Track A).
# These are a FALLBACK; the pipeline selects data-driven by coverage + climate
# diversity (see weather_forecast.eda.profiling.choose_representative_cities) and
# only uses this list if selection cannot run.
# --------------------------------------------------------------------------- #
REPRESENTATIVE_CITIES_FALLBACK: list[str] = [
    "London",     # temperate, N hemisphere
    "Moscow",     # continental, cold
    "Cairo",      # arid, subtropical
    "New Delhi",  # subtropical monsoon
    "Nairobi",    # tropical highland, ~equator
    "Tokyo",      # humid subtropical
    "Canberra",   # temperate, S hemisphere
    "Brasilia",   # tropical, S hemisphere
]
N_REPRESENTATIVE_CITIES: int = 6

# --------------------------------------------------------------------------- #
# Anomaly-detection configuration
# --------------------------------------------------------------------------- #
MAD_Z_THRESH: float = 3.5      # Iglewicz-Hoaglin modified z-score cutoff
STL_PERIOD: int = 12           # monthly-resampled STL seasonal period
STL_MIN_POINTS: int = 24       # need >= ~2 seasonal cycles for STL, else fall back to MAD
CONTAMINATION: float = 0.01    # documented anomaly prior for IsolationForest/LOF

# --------------------------------------------------------------------------- #
# PM Accelerator mission (assessment deliverable requirement).
# Single source of truth; surfaced in README, REPORT, notebook header and the
# pipeline startup log. Source: https://www.pmaccelerator.io/
# --------------------------------------------------------------------------- #
PM_ACCELERATOR_URL: str = "https://www.pmaccelerator.io/"
PM_ACCELERATOR_MISSION: str = (
    "Product Manager Accelerator (PMA), led by Dr. Nancy Li, is a leading product "
    "management professional-development community. Its mission is to support PM "
    "professionals at every stage of their careers — from aspiring and entry-level "
    "PMs to senior leaders aiming for Director and executive roles — through "
    "hands-on, real-world training (including building and launching real AI products), "
    "career coaching, and a community built on a culture of sharing and lifting others "
    "up. Through its nonprofit PMA Kids initiative, PMA also offers free product-"
    "management education to teenagers from underserved families to break down "
    "financial barriers and advance educational fairness."
)


def mission_banner() -> str:
    """Return a formatted, printable PM Accelerator mission block."""
    bar = "=" * 78
    return (
        f"{bar}\n"
        f"PM ACCELERATOR — Mission\n"
        f"{bar}\n"
        f"{PM_ACCELERATOR_MISSION}\n"
        f"Source: {PM_ACCELERATOR_URL}\n"
        f"{bar}"
    )


def ensure_dirs() -> None:
    """Create the output directory tree (idempotent). Called by the I/O edges."""
    for d in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

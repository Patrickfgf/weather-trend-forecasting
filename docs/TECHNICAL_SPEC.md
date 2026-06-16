# Technical Specification & Design Decisions

Concise record of the locked methodology. The *why* behind the code; the *what* lives in
docstrings and `REPORT.md`.

## Problem framing

- **Target:** `temperature_celsius`. **Grain:** 1 row = (`country`, `location_name`) ×
  `last_updated` daily snapshot.
- **Two tracks, identical folds:** (A) per-city univariate classical models for a
  climate-diverse representative set; (B) one global panel ML model over all cities.
  Track A is the interpretable ceiling per series; Track B is the deployable model. They
  are compared by MASE on the same folds.

## Data contract (the single source of truth = `config.py`)

- Column names normalized to `snake_case` at the ingestion edge (regex, idempotent).
- `last_updated` parsed **tz-aware UTC** — the only correct global ordering key across
  timezones.
- Imperial unit-twins (`*_fahrenheit`, `wind_mph`, …) and `last_updated_epoch` dropped on
  load: exact linear transforms inject multicollinearity and, for Fahrenheit, leak the
  target.
- Validation split in two: **lightweight structural** asserts on the RAW frame (which
  legitimately has glitches), **strict** pandera schema + grain on the CLEAN frame.

## Cleaning decisions

| Concern | Decision | Why |
|---|---|---|
| Duplicates | exact-drop; key-collisions keep-last + log | never average two conflicting snapshots |
| Impossible values | gate to NaN (physical bounds) | sensor errors ≠ rare-but-real extremes |
| Missing continuous | per-city time interpolation (short gaps) | respects autocorrelation; mean-impute flattens variance |
| Missing precip | fill 0 | structural "no rain reported" |
| Missing air quality | `<col>_was_missing` flag + short-gap fill | missingness is informative (sparse sensors) |
| Target missing | never impute | inventing labels inflates metrics |
| Outliers | flag-and-keep (`<col>_is_outlier`), per (city, month) MAD | extremes are the signal a forecaster must predict |

## Feature engineering — leakage invariants (enforced by tests)

1. Lags: `groupby(city).shift(k)`, `k>=1` (causal).
2. Rollings: `shift(1)` **before** `rolling(...)` (current row excluded) — and within
   `groupby(city)` (no cross-city bleed).
3. Exogenous weather only as `_lag1` (never contemporaneous).
4. `feature_columns()` excludes contemporaneous weather/AQI and `LEAKY_OR_TWIN_COLS`.
5. Cyclical sin/cos for circular fields (`day_of_year`, `month`, `dayofweek`,
   `wind_degree`). Hemisphere-aware `season`.

## Backtesting

- **Rolling-origin / expanding-window** CV on a **global calendar date** axis; same
  cutoff for every city → aligned panel splits. NO `KFold`, NO shuffle.
- Hard assertion per fold: `train.max_day < test.min_day`.
- XGBoost handles NaN features natively → only target-NaN rows dropped (a point in the
  tree model's favour vs linear/distance models that would need imputation).

## Models & ensemble

- **Baselines (mandatory):** naive, seasonal-naive (m=7, the MASE denominator), drift,
  climatology. A model that can't beat seasonal-naive is not shipped.
- **Classical (statsmodels):** ETS (additive, damped, weekly) and SARIMAX(1,1,1)(1,0,1,7),
  each with graceful fallback to seasonal-naive on convergence failure (Restart-Run-All
  must survive).
- **Global ML:** XGBoost (`tree_method="hist"`, fixed seed for determinism); LightGBM
  optional. **No annual seasonality** is fit (history is sub-annual).
- **Ensemble:** inverse-backtest-error weights, dropping any model with MASE ≥ 1.
- **Stacking:** non-negative linear meta-learner on **out-of-fold** predictions only
  (in-fold predictions would leak).

## Metrics

- **MASE primary** (scale-free, cross-city comparable). MAE/RMSE for absolute reading.
  sMAPE reported. **MAPE guarded** — temperature crosses 0 °C so the percentage explodes;
  shown only to demonstrate awareness.
- Metrics computed per (city, fold) then macro-averaged so large cities don't dominate.

## Graceful degradation (Python-3.14-aware)

`lightgbm`, `prophet`, `shap`, `pandera`, `kaleido`, `folium`, `pycountry-convert` are
import-guarded in `optional_deps.py`; a missing wheel becomes a logged skip, never a
crash. `statsforecast` has no 3.14 wheel and is intentionally omitted. Windows stdout is
reconfigured to UTF-8 (Python 3.14 defaults to cp1252 and would choke on `°`/`—`).

## Reproducibility

- `SEED=42` everywhere; deterministic xgboost; parquet intermediates (typed, tz-aware).
- Idempotent: same input → byte-stable outputs (no timestamps in data paths).
- `src/` layout + editable install → `import weather_forecast` works in CLI / notebook /
  pytest identically. No absolute paths.

## Discarded alternatives (and when they'd win)

- **Per-city ARIMA as the headline model** — discarded: overfits short series; the global
  model pools strength. Would win with multi-year per-city history.
- **Star-schema / wide reshaping** — unnecessary; the panel is already tidy at the chosen
  grain.
- **Clipping/winsorizing outliers** — discarded: deletes the extreme weather we must
  predict. Reserved only for a linear/NN branch, train-only, if robustness demanded it.
- **Mahalanobis anomaly detection** — discarded: the panel is multimodal (climate zones),
  so one mean/covariance is wrong; tree-based isolation needs no distributional assumption.

# Weather Trend Forecasting — Global Weather Repository

Forecasting daily **temperature** across world cities from the Kaggle
[Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
dataset, with a leakage-safe modeling pipeline, advanced anomaly detection, and five
unique analyses (climate, air quality, feature importance, spatial, geographical).

> ### PM Accelerator — Mission
> Product Manager Accelerator (PMA), led by Dr. Nancy Li, is a leading product
> management professional-development community. Its mission is to support PM
> professionals at every stage of their careers — from aspiring and entry-level PMs to
> senior leaders aiming for Director and executive roles — through hands-on, real-world
> training (including building and launching real AI products), career coaching, and a
> community built on a culture of sharing and lifting others up. Through its nonprofit
> PMA Kids initiative, PMA also offers free product-management education to teenagers
> from underserved families to break down financial barriers and advance educational
> fairness.
> _Source: <https://www.pmaccelerator.io/>_

---

## What this project does

- **Target / grain:** forecast `temperature_celsius`; one row = one city's daily
  snapshot keyed by `last_updated`.
- **Two forecasting tracks, one leakage-safe backtest:**
  - **Track A** — per-city classical models (seasonal-naive, drift, climatology,
    Holt-Winters **ETS**, **SARIMAX**) on a climate-diverse set of representative cities.
  - **Track B** — a **global panel** gradient-boosting model (XGBoost; LightGBM if
    available) that pools signal across all cities, plus an **inverse-error ensemble**
    and **OOF stacking**.
- **Advanced EDA + anomaly detection:** robust per-(city, month) z-scores, STL
  residuals, IsolationForest + LocalOutlierFactor, with anomaly analysis (condition
  lift, ingestion-artifact detection).
- **Five unique analyses:** long-run-within-sample climate patterns by climate zone,
  air-quality ↔ weather correlations, feature importance (permutation / native / SHAP),
  and spatial + cross-country/continent geography.

## Quickstart

```bash
# 1) Python 3.11+ (developed and verified on 3.14)
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional accelerators; degrade gracefully if skipped
pip install -e .                  # makes `import weather_forecast` work everywhere

# 2) (optional) get real data — otherwise a synthetic fallback runs automatically
#    Download "GlobalWeatherRepository.csv" from Kaggle and drop it in:
#    data/raw/GlobalWeatherRepository.csv

# 3) run the whole pipeline (produces every artifact under data/ and reports/)
python run_pipeline.py

# 4) open the narrative
jupyter lab notebooks/01_weather_trend_forecasting.ipynb

# 5) tests
pytest -q
```

If no CSV is present in `data/raw/`, the pipeline logs `SYNTHETIC fallback` and runs on a
deterministic, schema-faithful synthetic dataset so a reviewer can run everything with
zero manual download. Drop the real Kaggle CSV and re-run for real results.

## Project structure

```
weather-trend-forecasting/
├── run_pipeline.py              # single entrypoint -> all artifacts
├── requirements.txt             # pinned core stack
├── requirements-optional.txt    # optional extras (graceful-degrade)
├── pyproject.toml               # editable install + pytest/ruff config
├── data/{raw,interim,processed} # raw (gitignored) -> parquet intermediates/outputs
├── reports/{figures,metrics}    # PNG/HTML figures + CSV/JSON metrics
├── notebooks/                   # narrative notebook (imports from src/)
├── tests/                       # pytest suite (synthetic fixtures, leakage tests)
├── docs/TECHNICAL_SPEC.md       # methodology & design decisions
└── src/weather_forecast/
    ├── config.py                # single source of truth (columns, grain, seeds, mission)
    ├── ingest.py                # I/O edge: load + schema-normalize + validate + synth fallback
    ├── synthetic.py             # faithful GWR synthetic generator
    ├── cleaning.py              # dedup, physical-gating, per-city imputation, outlier flags
    ├── engineering.py           # calendar/cyclical + leakage-safe lags/rollings
    ├── profiling.py             # defensive EDA profiling + representative-city selection
    ├── plots.py                 # matplotlib EDA/anomaly/forecast figures
    ├── anomaly.py               # univariate + multivariate anomaly detection & analysis
    ├── metrics.py               # MAE/RMSE/sMAPE/MAPE/MASE + baselines
    ├── backtest.py              # leakage-safe rolling-origin CV (per-city + global)
    ├── models.py                # ETS/SARIMAX + XGBoost/LightGBM/Prophet
    ├── ensemble.py              # inverse-error weighting + OOF stacking
    ├── regions.py               # country->continent + latitude climate zones
    ├── climate.py | air_quality.py | feature_importance.py | spatial.py  # unique analyses
    └── pipeline.py              # orchestration
```

## How it avoids data leakage (the core correctness story)

1. **Chronological backtest only** — rolling-origin / expanding-window CV split on a
   global calendar date, applied identically to every city. No `KFold`, no shuffling.
   Every fold asserts `train.max_day < test.min_day`.
2. **Causal features** — lags use `groupby(city).shift(k)`; rolling stats `shift(1)`
   *before* `rolling` so the current row's own value is never in its window.
3. **Exogenous weather is lagged, never contemporaneous** — you don't know tomorrow's
   humidity when forecasting tomorrow's temperature.
4. **Target & leaky twins excluded** — `temperature_fahrenheit` (the target in disguise)
   and `feels_like_*` are dropped from features and importance.
5. **Stateful steps fit on train only** — scalers / stackers see out-of-fold data only.
6. **Direct multi-step** — the global model forecasts each horizon *from the origin*
   (label shifted h days; features only as of the cutoff), never reading intra-window
   actuals — a true h-step forecaster, fairly comparable to the baselines.

These are enforced by `assert`s in code and by dedicated tests in `tests/test_features.py`.

## Reproducibility

- Fixed seeds (`config.SEED = 42`), deterministic `xgboost` (`tree_method="hist"`).
- Parquet intermediates (typed, tz-preserving). Idempotent: same input → same output.
- No absolute paths; everything is anchored to the project root.

## Results snapshot (synthetic demonstration run)

| model | MASE ↓ | MAE (°C) | note |
|---|---|---|---|
| **lgbm_global** | **0.79** | 2.26 | global panel ML — best |
| xgb_global | 0.79 | 2.28 | |
| sarimax | 0.82 | 2.34 | best classical |
| seasonal_naive | 0.89 | 2.53 | primary benchmark |
| climatology | 2.27 | 6.59 | weak on sub-annual history |

MASE < 1 means "beats seasonal-naive". The global ML model wins by pooling signal across
cities. Full narrative and figures: [`REPORT.md`](REPORT.md). Numbers regenerate from the
real Kaggle CSV when present.

## License & contact

For evaluation: keep the repository **public** during review, or grant access to
`community@pmaccelerator.io` and `hr@pmaccelerator.io`.

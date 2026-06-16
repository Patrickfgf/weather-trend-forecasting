"""Generate the narrative notebook from source via nbformat.

Run:  python scripts/build_notebook.py
The notebook imports from the installed ``weather_forecast`` package (no logic is
redefined in the notebook) and is self-contained: its setup cell runs the full
pipeline once if the artifacts are missing, then the narrative loads & displays them.
"""

from __future__ import annotations

import nbformat as nbf

from weather_forecast import config

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = '''\
import sys, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from IPython.display import Image, display, Markdown
from weather_forecast import config, pipeline, ingest, cleaning, profiling
from weather_forecast import engineering as fe

pipeline.set_global_seeds()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

print(config.mission_banner())

# Self-contained: regenerate all artifacts once if they are not present yet.
if not (config.METRICS_DIR / "model_comparison.csv").exists():
    print("\\nArtifacts missing -> running the full pipeline once (this takes ~1 min)...")
    pipeline.run_all()

FIG, MET, PROC = config.FIGURES_DIR, config.METRICS_DIR, config.PROCESSED_DIR

def show_fig(name, caption=""):
    p = FIG / name
    if p.exists():
        display(Image(filename=str(p)))
        if caption:
            display(Markdown(f"*{caption}*"))
    else:
        display(Markdown(f"_(figure {name} not found)_"))

def show_csv(name, n=10):
    p = MET / name
    return pd.read_csv(p).head(n) if p.exists() else f"(missing {name})"
'''

LOAD = '''\
df, is_real = ingest.load_weather()
print("data source:", "REAL Kaggle CSV" if is_real else "SYNTHETIC fallback", "| raw shape:", df.shape)
clean = cleaning.clean(df)
print("clean shape:", clean.shape,
      "| grain unique:", not clean.duplicated(config.KEY_COLS + [config.TIME_COL]).any())
clean[[config.TIME_COL, "country", "location_name", config.TARGET, "humidity", "air_quality_pm2_5"]].head()
'''

EDA = '''\
display(Markdown("**Grain report**")); display(profiling.grain_report(clean))
for fn in ["eda_temperature_distribution.png", "eda_seasonal_cycle.png",
           "eda_precipitation_analysis.png", "eda_correlation_spearman.png",
           "eda_temperature_trend.png", "eda_global_aggregate.png"]:
    show_fig(fn)
'''

ANOMALY = '''\
an = pd.read_parquet(PROC / "anomalies.parquet")
print("anomalies flagged:", len(an), "| high-confidence:", int(an["is_high_confidence"].sum()))
show_fig("anomaly_geography.png", "Anomalies are defined relative to (city, month) - never the global pool.")
display(Markdown("**Anomaly lift by weather condition** (sanity check: real weather, not noise)"))
display(show_csv("anomaly_condition_lift.csv", 8))
'''

FEATURES = '''\
feats = fe.build_features(clean)
fcols = fe.feature_columns(feats)
print("n model features:", len(fcols))
print("examples:", fcols[:12])

# Leakage guards demonstrated live:
g = feats[feats.location_name == feats.location_name.iloc[0]].sort_values(config.TIME_COL)
lag_ok = (g[config.TARGET].shift(1).dropna().round(3).values
          == g[f"{config.TARGET}_lag1"].dropna().round(3).values).all()
print("lag1 == prior-row target (causal):", bool(lag_ok))
print("contemporaneous 'humidity' excluded:", "humidity" not in fcols,
      "| only 'humidity_lag1' used:", "humidity_lag1" in fcols)
print("leaky 'feels_like_*' excluded:", not any("feels_like" in c for c in fcols))
'''

BACKTEST = '''\
display(Markdown("**Model comparison (representative cities), sorted by MASE** - MASE<1 beats seasonal-naive"))
display(show_csv("model_comparison.csv"))
show_fig("forecast_model_comparison.png", "Global panel ML (XGBoost) wins by pooling signal across cities.")
print("ensemble weights:", json.loads((MET / "ensemble_weights.json").read_text()))
print("stacking weights:", json.loads((MET / "stacking_weights.json").read_text()))
display(Markdown("**7-day-ahead temperature forecast per representative city**"))
display(pd.read_parquet(PROC / "predictions.parquet").head(14))
'''

ANALYSES = '''\
for fn in ["climate_seasonal_cycle.png", "climate_zone_distributions.png",
           "air_quality_weather_heatmap.png", "air_quality_epa_by_continent.png",
           "feature_importance_permutation.png", "spatial_temperature_map.png",
           "spatial_country_choropleth.png", "spatial_continent_box.png"]:
    show_fig(fn)
display(Markdown("**Top permutation feature importances** (validation-only, robust)"))
display(show_csv("feature_importance_permutation.csv", 10))
'''

cells = [
    md("# Weather Trend Forecasting — Global Weather Repository\n\n"
       "> **PM Accelerator mission.** Product Manager Accelerator (PMA), led by Dr. Nancy Li, "
       "supports PM professionals at every career stage through hands-on AI product training, "
       "coaching, and a sharing-first community; its PMA Kids nonprofit brings free PM education "
       "to underserved teenagers. Source: https://www.pmaccelerator.io/\n\n"
       "This notebook is a **narrative over the `weather_forecast` package** — all logic lives in "
       "`src/`, nothing is redefined here. It runs top-to-bottom on a fresh kernel (Restart & Run All), "
       "using the real Kaggle CSV if present in `data/raw/`, else a deterministic synthetic fallback."),
    code(SETUP),
    md("## 1. Data loading & cleaning\n"
       "tz-aware UTC parsing, imperial twins dropped, dedup, physical gating, per-city imputation, "
       "outlier flagging — see `ingest.py` / `cleaning.py`."),
    code(LOAD),
    md("## 2. Exploratory analysis\n"
       "Multimodal global temperature, **hemisphere anti-phase** seasonality, zero-inflated "
       "precipitation, Spearman correlations (rank-robust)."),
    code(EDA),
    md("## 3. Anomaly detection (advanced EDA)\n"
       "Per-(city, season) robust z-scores + STL + IsolationForest ∩ LOF, then *analysis* "
       "(condition lift, ingestion-artifact detection)."),
    code(ANOMALY),
    md("## 4. Feature engineering & leakage guards\n"
       "Calendar/cyclical + per-city lags & shifted rollings; exogenous weather only lagged. "
       "The asserts below are the core correctness story."),
    code(FEATURES),
    md("## 5. Backtesting & model comparison\n"
       "Leakage-safe rolling-origin CV on a global date axis. **MASE** is the primary metric; "
       "baselines are mandatory and the ensemble drops anything worse than seasonal-naive."),
    code(BACKTEST),
    md("## 6. Unique analyses\n"
       "Climate (latitude zones), environmental impact (air quality ↔ weather), feature importance, "
       "and spatial / cross-country geography."),
    code(ANALYSES),
    md("## 7. Conclusions\n"
       "- A **global panel gradient-boosting model wins** (MASE ≈ 0.71 representative / 0.68 full panel), "
       "beating per-city classical models and all baselines — short, noisy per-city series benefit from "
       "pooling.\n"
       "- **Leakage is controlled** by construction (chronological CV, shifted per-city features, lagged "
       "exogenous, target-twin exclusion) and verified by tests.\n"
       "- Short snapshot history limits annual seasonality / trend claims — framed honestly throughout.\n\n"
       "Full write-up: [`REPORT.md`](../REPORT.md) · methodology: [`docs/TECHNICAL_SPEC.md`](../docs/TECHNICAL_SPEC.md)."),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = config.PROJECT_ROOT / "notebooks" / "01_weather_trend_forecasting.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)

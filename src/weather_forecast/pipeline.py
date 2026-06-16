"""End-to-end orchestration: ingest -> clean -> features -> EDA -> anomaly ->
forecasting (backtest + ensemble) -> unique analyses -> artifacts.

This is the only non-edge module allowed to call the I/O edges (ingest, viz, parquet
writes). The core path (ingest/clean/features/backtest) must succeed; every analysis
and figure block is wrapped so a single failure degrades to a logged warning instead
of crashing the run. Re-running with the same input is idempotent (full-overwrite of
deterministic paths).
"""

from __future__ import annotations

import json
import logging
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import (  # noqa: E402
    air_quality, anomaly, backtest, cleaning, climate, config, ensemble,
    feature_importance as fi, ingest, metrics, models, optional_deps, plots,
    profiling, regions, spatial,
)
from . import engineering as fe  # noqa: E402
from .logging_utils import setup_logging  # noqa: E402

log = logging.getLogger("weather_forecast.pipeline")


# --------------------------------------------------------------------------- #
# Determinism + IO helpers
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = config.SEED) -> None:
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _try(label: str, fn):
    try:
        return fn()
    except Exception as exc:  # best-effort analysis/figure blocks
        log.warning("[pipeline] %s failed: %s", label, repr(exc)[:180])
        return None


def _save_mpl(fig, name: str) -> None:
    if fig is None:
        return
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_plotly(fig, name: str) -> None:
    if fig is None:
        return
    out = config.FIGURES_DIR / name
    try:
        if optional_deps.HAS_KALEIDO:
            fig.write_image(str(out.with_suffix(".png")), width=1100, height=650, scale=2)
        else:
            fig.write_html(str(out.with_suffix(".html")))
    except Exception as exc:  # pragma: no cover
        log.warning("[pipeline] save plotly %s failed: %s", name, repr(exc)[:120])


def _save_csv(df: pd.DataFrame | None, name: str) -> None:
    if df is None or getattr(df, "empty", True):
        return
    df.to_csv(config.METRICS_DIR / name, index=False)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _run_eda(clean: pd.DataFrame, reg: pd.DataFrame, rep_cities: list) -> None:
    corrs = _try("correlation_matrices", lambda: profiling.correlation_matrices(clean))
    if corrs is not None:
        _save_csv(corrs.get("spearman").reset_index(), "correlation_spearman.csv")
    _save_csv(_try("series_coverage", lambda: profiling.series_coverage(clean)), "series_coverage.csv")
    _save_csv(_try("precip_zero_inflation", lambda: profiling.precip_zero_inflation(clean)),
              "precip_zero_inflation.csv")

    _save_mpl(_try("fig:temp_dist", lambda: plots.plot_temperature_distribution(reg)), "eda_temperature_distribution.png")
    _save_mpl(_try("fig:temp_trend", lambda: plots.plot_temperature_trend(reg, rep_cities)), "eda_temperature_trend.png")
    _save_mpl(_try("fig:precip", lambda: plots.plot_precipitation_analysis(clean)), "eda_precipitation_analysis.png")
    _save_mpl(_try("fig:seasonal", lambda: plots.plot_seasonal_cycle(reg)), "eda_seasonal_cycle.png")
    _save_mpl(_try("fig:global_agg", lambda: plots.plot_global_aggregate(reg)), "eda_global_aggregate.png")
    if corrs is not None:
        _save_mpl(_try("fig:corr", lambda: plots.plot_correlation(corrs, "spearman")), "eda_correlation_spearman.png")


def _run_anomaly(reg: pd.DataFrame) -> pd.DataFrame | None:
    an = _try("detect_anomalies", lambda: anomaly.detect_anomalies(reg))
    if an is None:
        return None
    _try("write_anomalies", lambda: anomaly.write_anomalies(an))
    _save_csv(_try("condition_lift", lambda: anomaly.condition_lift(an, reg)), "anomaly_condition_lift.csv")
    _save_csv(_try("anomalies_by_time", lambda: anomaly.anomalies_by_time(an)), "anomaly_by_time.csv")
    _save_mpl(_try("fig:anomaly_geo", lambda: plots.plot_anomaly_geography(an, reg)), "anomaly_geography.png")
    return an


def _run_forecasting(clean: pd.DataFrame, feats: pd.DataFrame, rep_cities: list) -> dict:
    fcols = fe.feature_columns(feats)
    keys = list(clean[config.KEY_COLS].drop_duplicates().itertuples(index=False, name=None))
    series_all = {k: fe.to_daily_series(clean, k[0], k[1]) for k in keys}
    rep_set = {tuple(c) for c in rep_cities}
    rep_series = {k: v for k, v in series_all.items() if k in rep_set} or series_all
    folds = backtest.make_time_folds(clean[config.TIME_COL])
    if not folds:
        log.warning("[pipeline] no backtest folds; skipping forecasting")
        return {}

    # Track A — per-city models on the representative deep-dive cities
    pc = backtest.backtest_per_city(rep_series, models.per_city_model_fns(), folds)

    # Track B — global panel model(s) over the full panel
    glob_parts, oof_by = [], {}
    for name, fac in models.global_model_factories().items():
        m_long, oof = backtest.backtest_global(feats, fcols, fac, folds,
                                                model_name=name, series_by_city=series_all)
        glob_parts.append(m_long)
        oof_by[name] = oof

    # Apples-to-apples comparison on the representative cities
    glob_all = pd.concat(glob_parts, ignore_index=True) if glob_parts else pd.DataFrame()
    glob_rep = glob_all[glob_all.set_index(config.KEY_COLS).index.isin(rep_set)] if not glob_all.empty else glob_all
    comparison = backtest.summarize(pd.concat([pc, glob_rep], ignore_index=True))
    _save_csv(comparison, "model_comparison.csv")
    _save_csv(backtest.summarize(glob_all), "model_comparison_global_fullpanel.csv")
    _save_mpl(_try("fig:model_cmp", lambda: plots.plot_model_comparison(comparison)), "forecast_model_comparison.png")

    # Ensemble weights (drop models worse than seasonal-naive)
    mae_by = dict(zip(comparison.model, comparison.mae))
    mase_by = dict(zip(comparison.model, comparison.mase))
    weights = ensemble.inverse_error_weights(mae_by, alpha=1.0, drop_above=1.0, drop_metric=mase_by)
    (config.METRICS_DIR / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))

    # Leakage-free stacking over the global GBM OOFs (naturally aligned on city/date)
    stack_info = {}
    x_oof, y_oof = ensemble.build_oof_matrix(oof_by)
    if not x_oof.empty and x_oof.shape[1] >= 2:
        stk = ensemble.fit_stacker(x_oof, y_oof)
        stack_info = {"models": list(x_oof.columns), "coef": [float(c) for c in stk.coef_]}
        (config.METRICS_DIR / "stacking_weights.json").write_text(json.dumps(stack_info, indent=2))

    # Deliverable forecast: next-HORIZON per representative city via inverse-error blend
    preds = _final_forecasts(rep_series, comparison)
    if preds is not None and not preds.empty:
        preds.to_parquet(config.PROCESSED_DIR / "predictions.parquet", index=False)

    return {"comparison": comparison, "weights": weights, "stacking": stack_info,
            "n_cities": len(series_all), "n_folds": len(folds), "feature_cols": fcols,
            "series_all": series_all, "oof_by": oof_by}


def _final_forecasts(series: dict, comparison: pd.DataFrame) -> pd.DataFrame | None:
    """Per-city next-HORIZON forecast = inverse-MAE blend of the per-city models."""
    per_city_models = {
        "seasonal_naive": metrics.seasonal_naive_forecast,
        "naive": metrics.naive_forecast,
        "drift": metrics.drift_forecast,
        "ets": models.ets_forecast,
        "climatology": metrics.climatology_forecast,
    }
    mae_by = dict(zip(comparison.model, comparison.mae))
    mase_by = dict(zip(comparison.model, comparison.mase))
    w = ensemble.inverse_error_weights({k: mae_by[k] for k in per_city_models if k in mae_by},
                                       drop_above=1.0,
                                       drop_metric={k: mase_by.get(k, np.nan) for k in per_city_models})
    rows = []
    for (country, loc), s in series.items():
        s = s.dropna()
        if s.size < config.SEASON_LENGTH + 2:
            continue
        preds = {name: fn(s, config.HORIZON) for name, fn in per_city_models.items() if name in w}
        blend = ensemble.weighted_average(preds, w)
        if blend is None:
            continue
        future = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=config.HORIZON, freq="D")
        rows.append(pd.DataFrame({"country": country, "location_name": loc, "date": future,
                                  "horizon_day": np.arange(1, config.HORIZON + 1),
                                  "forecast_temperature_celsius": np.round(blend, 2),
                                  "model": "inverse_error_ensemble"}))
    return pd.concat(rows, ignore_index=True) if rows else None


def _run_analyses(feats: pd.DataFrame, reg: pd.DataFrame, an: pd.DataFrame | None,
                  fc: dict) -> None:
    # Climate
    sc = _try("seasonal_cycle", lambda: climate.seasonal_cycle(reg))
    _save_csv(sc, "climate_seasonal_cycle.csv")
    _save_plotly(_try("fig:climate_cycle", lambda: climate.fig_seasonal_cycle(sc)) if sc is not None else None,
                 "climate_seasonal_cycle")
    _save_plotly(_try("fig:zone_dist", lambda: climate.fig_zone_distributions(reg)), "climate_zone_distributions")

    # Environmental impact (air quality vs weather)
    aqc = _try("aq_corr", lambda: air_quality.aq_weather_correlation(reg))
    _save_csv(aqc, "air_quality_weather_correlation.csv")  # already tidy/long; no junk index col
    _save_plotly(_try("fig:aq_heatmap", lambda: air_quality.fig_aq_weather_heatmap(aqc)) if aqc is not None else None,
                 "air_quality_weather_heatmap")
    epa = _try("epa_summary", lambda: air_quality.epa_index_summary(reg))
    _save_csv(epa, "air_quality_epa_summary.csv")
    _save_plotly(_try("fig:epa", lambda: air_quality.fig_epa_by_continent(epa)) if epa is not None else None,
                 "air_quality_epa_by_continent")

    # Spatial + geographical
    snap = _try("latest_snapshot", lambda: spatial.latest_snapshot(reg))
    _save_plotly(_try("fig:geo_temp", lambda: spatial.fig_scatter_geo_temp(snap)) if snap is not None else None,
                 "spatial_temperature_map")
    cws = _try("country_summary", lambda: spatial.country_weather_summary(reg))
    _save_csv(cws, "geo_country_summary.csv")
    _save_plotly(_try("fig:choropleth", lambda: spatial.fig_country_choropleth(cws)) if cws is not None else None,
                 "spatial_country_choropleth")
    cc = _try("continent_comparison", lambda: spatial.continent_comparison(reg))
    _save_csv(cc, "geo_continent_comparison.csv")
    _save_plotly(_try("fig:continent_box", lambda: spatial.fig_continent_box(reg)), "spatial_continent_box")

    # Feature importance on the global model (chronological val split, no leakage)
    _run_feature_importance(feats, fc)


def _run_feature_importance(feats: pd.DataFrame, fc: dict) -> None:
    fcols = fc.get("feature_cols") or fe.feature_columns(feats)
    sub = feats.dropna(subset=[config.TARGET]).sort_values(config.TIME_COL)
    if len(sub) < 100:
        return
    cut = int(len(sub) * (1 - config.TEST_FRACTION))
    tr, va = sub.iloc[:cut], sub.iloc[cut:]
    model = models.make_xgb()
    model.fit(tr[fcols].to_numpy("float64"), tr[config.TARGET].to_numpy("float64"))
    x_va, y_va = va[fcols].to_numpy("float64"), va[config.TARGET].to_numpy("float64")

    perm = _try("perm_importance",
                lambda: fi.permutation_importance_df(model, x_va, y_va, fcols, n_repeats=10))
    _save_csv(perm, "feature_importance_permutation.csv")
    _save_mpl(_try("fig:perm", lambda: fi.fig_permutation_importance(perm)) if perm is not None else None,
              "feature_importance_permutation.png")
    nat = _try("native_importance", lambda: fi.native_importance(model, fcols))
    _save_csv(nat, "feature_importance_native.csv")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_all(synthetic_kwargs: dict | None = None) -> dict:
    """Run the full pipeline, writing all artifacts. Returns a summary dict."""
    setup_logging()
    set_global_seeds()
    config.ensure_dirs()
    log.info("\n%s", config.mission_banner())
    log.info("[pipeline] optional deps: %s", optional_deps.summary())

    df, is_real = ingest.load_weather(synthetic_kwargs=synthetic_kwargs)
    clean = cleaning.clean(df)
    clean.to_parquet(config.INTERIM_DIR / "clean.parquet", index=False)

    feats = fe.build_features(clean)
    feats.to_parquet(config.PROCESSED_DIR / "features.parquet", index=False)
    reg = regions.add_region_columns(feats)

    rep_cities = profiling.choose_representative_cities(clean)
    log.info("[pipeline] representative cities: %s", rep_cities)

    _run_eda(clean, reg, rep_cities)
    an = _run_anomaly(reg)
    fc = _run_forecasting(clean, feats, rep_cities)
    _run_analyses(feats, reg, an, fc)

    comparison = fc.get("comparison")
    best = (comparison.iloc[0]["model"] if comparison is not None and not comparison.empty else None)
    log.info("[pipeline] done. data_source=%s best_model=%s",
             "REAL" if is_real else "SYNTHETIC", best)
    return {"is_real": is_real, "rep_cities": rep_cities, "best_model": best,
            "comparison": comparison, "n_anomalies": (0 if an is None else len(an))}

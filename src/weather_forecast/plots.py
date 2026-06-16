"""Matplotlib EDA, anomaly and forecast figures (headless / Agg backend).

This module is the *static-figure* surface for the project: every function takes a
DataFrame / dict / array and returns a ``matplotlib.figure.Figure`` so the same
code path works in a notebook (inline) and in a batch pipeline (``out=...`` ->
``savefig``). Nothing here reads the raw file, mutates its input, or calls
``plt.show()``.

Honesty caveats baked into titles/subtitles where relevant:
- The dataset is a rolling DAILY-SNAPSHOT history (sub-annual to ~1yr), NOT a
  multi-decade climate record. "Trend" slopes are *within-sample* artefacts of a
  short span, framed as deg C/yr but explicitly flagged as not decadal climate.
- Spearman is the primary correlation (rank-robust to the heavy tails in weather
  / air-quality data); correlations are monotonic & possibly confounded, never
  causal.
- Seasonal panels are hemisphere-aware (the S hemisphere is anti-phase), which is
  why ``plot_seasonal_cycle`` splits N vs S instead of pooling.

Every function is defensive about missing optional columns: a panel degrades to a
visible "not available" message rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # MUST precede pyplot import: no display in batch/CI/tests.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from weather_forecast import config

# --------------------------------------------------------------------------- #
# Module-level styling constants (no magic numbers scattered in the functions).
# --------------------------------------------------------------------------- #
_DPI: int = 150
_DIVERGING_CMAP: str = "RdBu_r"  # centered diverging map for signed correlations
_HEMI_COLORS: dict[str, str] = {"N": "#1f77b4", "S": "#d62728"}
_MONTH_LABELS: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
_SHORT_SPAN_NOTE: str = (
    "Daily-snapshot history (sub-annual to ~1yr) — within-sample, not a "
    "multi-decade climate trend."
)


# --------------------------------------------------------------------------- #
# Small private helpers (DRY: save, empty-panel, span string).
# --------------------------------------------------------------------------- #
def _finalize(fig: plt.Figure, out: "Path | None") -> plt.Figure:
    """Tight-layout, optionally savefig, and return the figure.

    Centralizes the savefig contract (only write when ``out`` is set; dpi=150,
    bbox_inches='tight') so no caller forgets it or drifts on dpi.
    """
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    return fig


def _empty_panel(ax: plt.Axes, message: str) -> None:
    """Render a centered 'data unavailable' note on an axis (graceful degrade)."""
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="grey", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _sample_span(df: pd.DataFrame) -> str:
    """Human-readable date span of ``config.TIME_COL`` for honest subtitles."""
    if config.TIME_COL not in df.columns:
        return "sample span unknown"
    ts = pd.to_datetime(df[config.TIME_COL], errors="coerce", utc=True).dropna()
    if ts.empty:
        return "sample span unknown"
    lo, hi = ts.min().date(), ts.max().date()
    return f"Sample span: {lo} -> {hi} ({(hi - lo).days} days)"


# --------------------------------------------------------------------------- #
# 1. Temperature distribution
# --------------------------------------------------------------------------- #
def plot_temperature_distribution(
    df: pd.DataFrame,
    target: str = config.TARGET,
    hue: str = "hemisphere",
    out: "Path | None" = None,
) -> plt.Figure:
    """Histogram + KDE of the target; overlay per-hemisphere when ``hue`` exists.

    Splitting by hemisphere surfaces the bimodal seasonal mix (N winter vs S
    summer coexist in a snapshot) that a pooled histogram hides.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    if target not in df.columns:
        _empty_panel(ax, f"column {target!r} not available")
        return _finalize(fig, out)

    vals = pd.to_numeric(df[target], errors="coerce")
    groups: list[tuple[str, pd.Series, str]] = []
    if hue in df.columns:
        for key, sub in vals.groupby(df[hue], observed=True):
            color = _HEMI_COLORS.get(str(key), None)
            groups.append((f"{hue}={key}", sub.dropna(), color))
    if not groups:  # no hue column -> single pooled distribution
        groups = [("all cities", vals.dropna(), _HEMI_COLORS["N"])]

    for label, series, color in groups:
        series = series[np.isfinite(series)]
        if series.empty:
            continue
        ax.hist(series, bins=50, density=True, alpha=0.4, color=color, label=label)
        _overlay_kde(ax, series.to_numpy(), color=color)

    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(f"{target} (deg C)")
    ax.set_ylabel("density")
    ax.legend()
    return _finalize(fig, out)


def _overlay_kde(ax: plt.Axes, sample: np.ndarray, color: str | None) -> None:
    """Overlay a gaussian KDE if scipy is available; silently skip otherwise."""
    if sample.size < 2 or np.allclose(sample, sample[0]):
        return
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(sample)
        grid = np.linspace(sample.min(), sample.max(), 200)
        ax.plot(grid, kde(grid), color=color, linewidth=2)
    except Exception:  # scipy missing or singular covariance -> hist still shown
        return


# --------------------------------------------------------------------------- #
# 2. Per-city temperature trend small-multiples
# --------------------------------------------------------------------------- #
def plot_temperature_trend(
    df: pd.DataFrame,
    cities: list[tuple[str, str]],
    freq: str = "MS",
    rolling: int = 3,
    target: str = config.TARGET,
    out: "Path | None" = None,
) -> plt.Figure:
    """Small-multiples: per-city resampled mean + rolling overlay + polyfit slope.

    The deg C/yr slope is a deg-1 ``numpy.polyfit`` on the resampled series — a
    blunt within-sample descriptor, NOT a climate trend (subtitle says so). Two
    overlays: the raw monthly mean and a ``rolling``-window smoother for the eye.
    """
    n = max(len(cities), 1)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    missing_cols = target not in df.columns or config.TIME_COL not in df.columns
    for idx, ax in enumerate(flat):
        if idx >= len(cities):
            ax.axis("off")
            continue
        country, location = cities[idx]
        if missing_cols:
            _empty_panel(ax, "time/target column not available")
            ax.set_title(f"{location}, {country}")
            continue
        _plot_one_city_trend(ax, df, country, location, freq, rolling, target)

    fig.suptitle(f"Temperature trend by city ({freq} mean)\n{_SHORT_SPAN_NOTE}",
                 fontsize=11)
    return _finalize(fig, out)


def _plot_one_city_trend(
    ax: plt.Axes,
    df: pd.DataFrame,
    country: str,
    location: str,
    freq: str,
    rolling: int,
    target: str,
) -> None:
    """Resample one city's target, draw mean+rolling, annotate polyfit deg C/yr."""
    mask = (df["country"].astype(str) == str(country)) & (
        df["location_name"].astype(str) == str(location)
    )
    sub = df.loc[mask, [config.TIME_COL, target]].copy()
    ax.set_title(f"{location}, {country}")
    if sub.empty:
        _empty_panel(ax, "no rows for this city")
        return

    ts = pd.to_datetime(sub[config.TIME_COL], errors="coerce", utc=True)
    series = (
        pd.Series(pd.to_numeric(sub[target], errors="coerce").to_numpy(), index=ts)
        .dropna()
        .sort_index()
        .resample(freq)
        .mean()
        .dropna()
    )
    if series.empty:
        _empty_panel(ax, "no usable values")
        return

    ax.plot(series.index, series.to_numpy(), marker="o", ms=3,
            color="#1f77b4", alpha=0.6, label=f"{freq} mean")
    if rolling and len(series) >= rolling:
        roll = series.rolling(rolling, min_periods=1).mean()
        ax.plot(roll.index, roll.to_numpy(), color="#ff7f0e",
                linewidth=2, label=f"rolling({rolling})")

    slope_txt = _polyfit_slope_per_year(series)
    ax.annotate(slope_txt, xy=(0.02, 0.92), xycoords="axes fraction",
                fontsize=9, color="#333333",
                bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.set_ylabel("deg C")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelrotation=30)


def _polyfit_slope_per_year(series: pd.Series) -> str:
    """Deg-1 polyfit slope converted to deg C/yr; needs >=2 points else 'n/a'."""
    if len(series) < 2:
        return "slope: n/a (<2 points)"
    # x in fractional years from the first sample -> slope is directly deg C/yr.
    days = (series.index - series.index[0]).days.to_numpy(dtype=float)
    x_years = days / 365.25
    if np.ptp(x_years) == 0:
        return "slope: n/a"
    slope = float(np.polyfit(x_years, series.to_numpy(dtype=float), 1)[0])
    return f"slope ~ {slope:+.1f} deg C/yr (within-sample)"


# --------------------------------------------------------------------------- #
# 3. Precipitation analysis (3 panels)
# --------------------------------------------------------------------------- #
def plot_precipitation_analysis(
    df: pd.DataFrame,
    out: "Path | None" = None,
    top_n: int = 10,
) -> plt.Figure:
    """Three precip views: %dry-day extremes, log1p wet-day hist, precip vs humidity.

    log1p compresses the heavy right tail of precip so the wet-day shape is
    legible; the hexbin shows the (monotonic, confounded) precip-humidity link.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if "precip_mm" not in df.columns:
        for ax in axes:
            _empty_panel(ax, "precip_mm not available")
        fig.suptitle("Precipitation analysis", fontsize=12)
        return _finalize(fig, out)

    precip = pd.to_numeric(df["precip_mm"], errors="coerce")
    _panel_dry_days(axes[0], df, precip, top_n)
    _panel_wet_hist(axes[1], precip)
    _panel_precip_humidity(axes[2], df, precip)
    fig.suptitle("Precipitation analysis", fontsize=12)
    return _finalize(fig, out)


def _panel_dry_days(ax: plt.Axes, df: pd.DataFrame, precip: pd.Series, top_n: int) -> None:
    """Bar of % dry days for the driest and wettest countries (extremes only)."""
    if "country" not in df.columns:
        _empty_panel(ax, "country not available")
        return
    is_dry = (precip.fillna(0) <= 0.0).astype(float)
    pct = is_dry.groupby(df["country"], observed=True).mean().mul(100).dropna()
    if pct.empty:
        _empty_panel(ax, "no precip data")
        return
    pct = pct.sort_values()
    k = min(top_n, len(pct))
    extremes = pd.concat([pct.head(k), pct.tail(k)])
    extremes = extremes[~extremes.index.duplicated()]
    colors = ["#2ca02c"] * min(k, len(extremes)) + ["#8c564b"] * (len(extremes) - min(k, len(extremes)))
    ax.barh(extremes.index.astype(str), extremes.to_numpy(), color=colors)
    ax.set_title("% dry days — wettest (top) vs driest (bottom) countries")
    ax.set_xlabel("% of snapshots with precip <= 0")
    ax.tick_params(axis="y", labelsize=7)


def _panel_wet_hist(ax: plt.Axes, precip: pd.Series) -> None:
    """Histogram of log1p(precip) on wet days only (precip > 0)."""
    wet = precip[precip > 0].dropna()
    if wet.empty:
        _empty_panel(ax, "no wet-day precip")
        return
    ax.hist(np.log1p(wet.to_numpy()), bins=40, color="#1f77b4", alpha=0.8)
    ax.set_title("Wet-day precipitation (log1p)")
    ax.set_xlabel("log1p(precip_mm)")
    ax.set_ylabel("count")


def _panel_precip_humidity(ax: plt.Axes, df: pd.DataFrame, precip: pd.Series) -> None:
    """Hexbin of precip vs humidity (monotonic, confounded — not causal)."""
    if "humidity" not in df.columns:
        _empty_panel(ax, "humidity not available")
        return
    humidity = pd.to_numeric(df["humidity"], errors="coerce")
    valid = np.isfinite(precip) & np.isfinite(humidity)
    if valid.sum() < 10:
        _empty_panel(ax, "too few paired points")
        return
    hb = ax.hexbin(humidity[valid], precip[valid], gridsize=40,
                   cmap="viridis", mincnt=1, bins="log")
    fig = ax.get_figure()
    fig.colorbar(hb, ax=ax, label="log10(count)")
    ax.set_title("Precip vs humidity (monotonic, confounded)")
    ax.set_xlabel("humidity (%)")
    ax.set_ylabel("precip_mm")


# --------------------------------------------------------------------------- #
# 4. Correlation heatmap
# --------------------------------------------------------------------------- #
def plot_correlation(
    corrs: dict,
    which: str = "spearman",
    out: "Path | None" = None,
) -> plt.Figure:
    """Annotated diverging heatmap of a correlation matrix from a ``corrs`` dict.

    ``corrs`` maps method -> matrix (DataFrame or array-like). Spearman is the
    default because rank correlation is robust to the heavy tails in weather /
    AQI data; the map is centered at 0 so sign reads off the color directly.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    mat = corrs.get(which) if isinstance(corrs, dict) else None
    if mat is None:
        _empty_panel(ax, f"correlation {which!r} not available")
        return _finalize(fig, out)

    cm = mat if isinstance(mat, pd.DataFrame) else pd.DataFrame(mat)
    data = cm.to_numpy(dtype=float)
    if data.size == 0:
        _empty_panel(ax, "empty correlation matrix")
        return _finalize(fig, out)

    im = ax.imshow(data, cmap=_DIVERGING_CMAP, vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{which} rho")

    labels = list(cm.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(list(cm.index), fontsize=7)

    # Annotate cells; white text on dark (|rho| high) for legibility.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(val) > 0.55 else "black")

    ax.set_title(f"{which.capitalize()} correlation (monotonic, not causal)")
    return _finalize(fig, out)


# --------------------------------------------------------------------------- #
# 5. Seasonal cycle (N vs S, anti-phase)
# --------------------------------------------------------------------------- #
def plot_seasonal_cycle(
    df: pd.DataFrame,
    target: str = config.TARGET,
    out: "Path | None" = None,
) -> plt.Figure:
    """Boxplot of target by month, split N vs S hemisphere to show anti-phase.

    Pooling hemispheres cancels the seasonal swing (they are ~6 months out of
    phase), so we split and plot side by side. Needs a month + hemisphere signal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    month = _month_series(df)
    if target not in df.columns or month is None:
        for ax in axes:
            _empty_panel(ax, "target / month not available")
        fig.suptitle("Seasonal cycle by hemisphere", fontsize=12)
        return _finalize(fig, out)

    vals = pd.to_numeric(df[target], errors="coerce")
    hemi = df["hemisphere"] if "hemisphere" in df.columns else None
    panels = [("N", "Northern hemisphere"), ("S", "Southern hemisphere")]
    for ax, (key, label) in zip(axes, panels):
        if hemi is None:
            sel = pd.Series(True, index=df.index) if key == "N" else pd.Series(False, index=df.index)
            label = "all cities" if key == "N" else "no hemisphere column"
        else:
            sel = hemi.astype(str) == key
        _boxplot_by_month(ax, vals[sel], month[sel])
        ax.set_title(label)
        ax.set_xlabel("month")
    axes[0].set_ylabel(f"{target} (deg C)")
    fig.suptitle("Seasonal cycle by hemisphere (anti-phase; hemisphere-aware)",
                 fontsize=12)
    return _finalize(fig, out)


def _month_series(df: pd.DataFrame) -> pd.Series | None:
    """Return a 1..12 month series from the ``month`` col or derived from time."""
    if "month" in df.columns:
        return pd.to_numeric(df["month"], errors="coerce")
    if config.TIME_COL in df.columns:
        ts = pd.to_datetime(df[config.TIME_COL], errors="coerce", utc=True)
        return ts.dt.month
    return None


def _boxplot_by_month(ax: plt.Axes, vals: pd.Series, month: pd.Series) -> None:
    """Draw one month-grouped boxplot; empty months are simply absent."""
    frame = pd.DataFrame({"v": vals, "m": month}).dropna()
    if frame.empty:
        _empty_panel(ax, "no data")
        return
    data = [frame.loc[frame["m"] == m, "v"].to_numpy() for m in range(1, 13)]
    positions = [m for m in range(1, 13)]
    keep = [(p, d) for p, d in zip(positions, data) if d.size > 0]
    if not keep:
        _empty_panel(ax, "no data")
        return
    pos, dat = zip(*keep)
    ax.boxplot(dat, positions=pos, widths=0.6, showfliers=False)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(_MONTH_LABELS, fontsize=8)


# --------------------------------------------------------------------------- #
# 6. Global robust aggregate over time
# --------------------------------------------------------------------------- #
def plot_global_aggregate(
    df: pd.DataFrame,
    target: str = config.TARGET,
    freq: str = "MS",
    out: "Path | None" = None,
) -> plt.Figure:
    """Median +/- IQR band of the target across all cities over time (robust).

    Median/IQR rather than mean/std because the cross-city distribution is heavy
    tailed; the band is the 25-75 percentile spread, not a confidence interval.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    if target not in df.columns or config.TIME_COL not in df.columns:
        _empty_panel(ax, "target / time column not available")
        return _finalize(fig, out)

    ts = pd.to_datetime(df[config.TIME_COL], errors="coerce", utc=True)
    frame = pd.DataFrame(
        {"v": pd.to_numeric(df[target], errors="coerce").to_numpy()}, index=ts
    ).dropna().sort_index()
    if frame.empty:
        _empty_panel(ax, "no usable values")
        return _finalize(fig, out)

    grp = frame["v"].resample(freq)
    med = grp.median()
    q1 = grp.quantile(0.25)
    q3 = grp.quantile(0.75)
    ax.fill_between(med.index, q1.to_numpy(), q3.to_numpy(),
                    color="#1f77b4", alpha=0.25, label="IQR (25-75%)")
    ax.plot(med.index, med.to_numpy(), color="#1f77b4", linewidth=2, label="median")
    ax.set_title(f"Global {target}: median +/- IQR across cities ({freq})\n{_SHORT_SPAN_NOTE}")
    ax.set_xlabel("date")
    ax.set_ylabel(f"{target} (deg C)")
    ax.legend()
    ax.tick_params(axis="x", labelrotation=30)
    return _finalize(fig, out)


# --------------------------------------------------------------------------- #
# 7. Anomaly score distribution with contamination cut
# --------------------------------------------------------------------------- #
def plot_anomaly_scores(
    scores,
    contamination: float = config.CONTAMINATION,
    out: "Path | None" = None,
) -> plt.Figure:
    """Histogram of anomaly scores with the contamination-quantile cut line.

    The cut is the lower ``contamination`` quantile (convention: lower score =
    more anomalous, e.g. IsolationForest ``decision_function``); points left of
    the line are the flagged tail.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    arr = np.asarray(pd.Series(scores).dropna(), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        _empty_panel(ax, "no scores")
        return _finalize(fig, out)

    ax.hist(arr, bins=60, color="#1f77b4", alpha=0.8)
    cut = float(np.quantile(arr, contamination))
    ax.axvline(cut, color="#d62728", linestyle="--", linewidth=2,
               label=f"contamination cut = {cut:.3g} (q={contamination:.3g})")
    n_flag = int((arr <= cut).sum())
    ax.set_title(f"Anomaly score distribution ({n_flag} of {arr.size} below cut)")
    ax.set_xlabel("anomaly score (lower = more anomalous)")
    ax.set_ylabel("count")
    ax.legend()
    return _finalize(fig, out)


# --------------------------------------------------------------------------- #
# 8. Anomaly geography (pure matplotlib lat/lon scatter)
# --------------------------------------------------------------------------- #
def plot_anomaly_geography(
    anom_df: pd.DataFrame,
    base_df: pd.DataFrame,
    out: "Path | None" = None,
) -> plt.Figure:
    """Lat/lon scatter of all cities with anomalies highlighted (no geo deps).

    A bare equirectangular scatter (longitude=x, latitude=y) — no basemap, no
    cartopy — so it renders anywhere. Base points are faint grey; anomalies are
    drawn on top in red.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    lat, lon = config.GEO_COLS[0], config.GEO_COLS[1]
    if base_df is None or lat not in base_df.columns or lon not in base_df.columns:
        _empty_panel(ax, "latitude/longitude not available")
        return _finalize(fig, out)

    bx = pd.to_numeric(base_df[lon], errors="coerce")
    by = pd.to_numeric(base_df[lat], errors="coerce")
    ax.scatter(bx, by, s=6, color="#bbbbbb", alpha=0.4, label="all cities")

    if anom_df is not None and lat in anom_df.columns and lon in anom_df.columns and len(anom_df):
        ax_ = pd.to_numeric(anom_df[lon], errors="coerce")
        ay_ = pd.to_numeric(anom_df[lat], errors="coerce")
        ax.scatter(ax_, ay_, s=28, color="#d62728", alpha=0.85,
                   edgecolor="black", linewidth=0.3,
                   label=f"anomalies (n={len(anom_df)})")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")  # equator
    ax.set_title("Geographic distribution of anomalies")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="lower left")
    return _finalize(fig, out)


# --------------------------------------------------------------------------- #
# 9. Model comparison (MASE, lower=better, baseline at 1)
# --------------------------------------------------------------------------- #
def plot_model_comparison(
    summary_df: pd.DataFrame,
    out: "Path | None" = None,
    metric: str = "MASE",
) -> plt.Figure:
    """Horizontal bar of MASE by model with a baseline reference line at MASE=1.

    MASE < 1 beats the naive seasonal baseline; the dashed line at 1 makes that
    threshold readable at a glance. Bars are sorted best (lowest) at the top.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    if summary_df is None or len(summary_df) == 0:
        _empty_panel(ax, "no model summary")
        return _finalize(fig, out)

    df = summary_df.copy()
    model_col = _first_present(df, ["model", "name", "estimator"]) or df.index.name
    metric_col = _first_present(df, [metric, metric.lower(), metric.upper()])
    if metric_col is None:
        _empty_panel(ax, f"metric {metric!r} not in summary")
        return _finalize(fig, out)

    models = (df[model_col].astype(str).to_numpy() if model_col in df.columns
              else df.index.astype(str).to_numpy())
    vals = pd.to_numeric(df[metric_col], errors="coerce").to_numpy()
    order = np.argsort(np.where(np.isfinite(vals), vals, np.inf))[::-1]  # best last->top
    models, vals = models[order], vals[order]

    colors = ["#2ca02c" if (np.isfinite(v) and v < 1.0) else "#1f77b4" for v in vals]
    ax.barh(models, vals, color=colors)
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=2,
               label="baseline (MASE=1)")
    ax.set_title(f"Model comparison — {metric} (lower = better)")
    ax.set_xlabel(metric)
    ax.legend()
    return _finalize(fig, out)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in ``df`` (case-tolerant helper)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

"""Unique Analysis 2 — environmental impact: air quality vs weather (plotly).

This module probes how meteorology *co-moves* with pollutant concentrations on the
Global Weather Repository snapshot. It is deliberately CONSERVATIVE about what can be
claimed, because the cheap version of this analysis is wrong in three ways:

1. **Correlation is not causation, and the confounder here is spatial.** Pollution
   clusters in industrial/high-traffic cities, which also share weather regimes
   (basin geography, latitude, season). A wind-PM correlation can be entirely a
   between-city artefact (dirty cities happen to be calm) rather than a within-city
   wash-out effect. We surface direction + n, never a causal verb.
2. **The us-epa-index is DERIVED from the pollutants** (it is essentially a max over
   sub-indices of PM2.5/PM10/O3/...). Correlating it against PM2.5 is near-tautological,
   so the EPA index is used for *segmentation only* (:func:`epa_index_summary`) and is
   NEVER placed on the correlation grid.
3. **Pollutants are heavy-tailed and right-skewed**, so Pearson is dominated by a few
   extreme cities. We use **Spearman** (rank) as the primary association measure and
   compute it pairwise-complete, reporting the **n** behind every cell.

Honesty caveat baked into the figures: this is a rolling DAILY-SNAPSHOT history
(sub-annual to ~1 year), not a multi-decade record — read everything as *within-sample*
co-movement, not a climate trend.

All functions are pure: DataFrame in -> DataFrame / plotly Figure out. No global state,
no I/O of the raw data, no in-place mutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import config, optional_deps

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as _go  # noqa: F401

# `regions.add_region_columns` is authored by a sibling analysis module. Import it
# defensively so this module degrades gracefully (no `continent` -> single bucket)
# instead of crashing if that module is not present yet.
try:  # pragma: no cover - availability depends on sibling module
    from . import regions as _regions  # type: ignore[attr-defined]
except Exception:  # ImportError or any import-time failure in the sibling
    _regions = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Column sets and labels
# --------------------------------------------------------------------------- #
AQ_COLS: list[str] = config.AQI_CONCENTRATION_COLS
# Weather drivers ordered roughly by the dispersion/wash-out hypotheses we test:
# ventilation (wind/gust), wet deposition (precip/humidity), and synoptic context.
WEATHER_COLS: list[str] = [
    "wind_kph",
    "gust_kph",
    "precip_mm",
    "humidity",
    "pressure_mb",
    "temperature_celsius",
    "cloud",
    "uv_index",
]
EPA_LABELS: dict[int, str] = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy (Sensitive)",
    4: "Unhealthy",
    5: "Very Unhealthy",
    6: "Hazardous",
}
# Plot ordering + a green -> maroon public-health ramp (best -> worst).
_EPA_ORDER: list[str] = [EPA_LABELS[i] for i in range(1, 7)]
_EPA_COLORS: dict[str, str] = {
    "Good": "#2ca02c",
    "Moderate": "#bcbd22",
    "Unhealthy (Sensitive)": "#ff7f0e",
    "Unhealthy": "#d62728",
    "Very Unhealthy": "#9467bd",
    "Hazardous": "#7f1d1d",
}

# Minimum overlapping observations before we trust a correlation cell.
_MIN_PAIRS: int = 30


# --------------------------------------------------------------------------- #
# Correlation grid (weather rows x pollutant cols)
# --------------------------------------------------------------------------- #
def aq_weather_correlation(
    df: pd.DataFrame,
    method: str = "spearman",
    aq_cols: list[str] | None = None,
    weather_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Rectangular weather x pollutant correlation, pairwise-complete, with n attached.

    Why Spearman by default: pollutant concentrations are heavy-tailed, so a rank
    correlation is robust to the handful of extreme-city outliers that would otherwise
    dominate Pearson. Why pairwise-complete (cell by cell) rather than one dropna over
    all columns: AQI columns have structured missingness, and a global dropna would
    silently shrink n and bias toward well-instrumented cities.

    Returns a long/tidy frame (one row per weather x pollutant pair) with columns
    ``weather, pollutant, corr, n`` so the caller can both pivot to a heatmap and audit
    the sample size behind each cell. Cells with fewer than ``_MIN_PAIRS`` overlapping
    observations are emitted as NaN corr (n still reported) — small-n correlations are
    noise, not signal.
    """
    aq_cols = aq_cols if aq_cols is not None else AQ_COLS
    weather_cols = weather_cols if weather_cols is not None else WEATHER_COLS
    present_aq = [c for c in aq_cols if c in df.columns]
    present_w = [c for c in weather_cols if c in df.columns]

    rows: list[dict[str, object]] = []
    for w in present_w:
        ws = pd.to_numeric(df[w], errors="coerce")
        for p in present_aq:
            ps = pd.to_numeric(df[p], errors="coerce")
            mask = ws.notna() & ps.notna()
            n = int(mask.sum())
            corr = _safe_corr(ws[mask], ps[mask], method=method) if n >= _MIN_PAIRS else np.nan
            rows.append({"weather": w, "pollutant": p, "corr": corr, "n": n})

    return pd.DataFrame(rows, columns=["weather", "pollutant", "corr", "n"])


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Compute a single correlation coefficient, returning NaN on degenerate input.

    Guards the zero-variance case (constant series -> correlation undefined) which
    scipy/pandas would otherwise surface as a NaN-with-warning; we just return NaN.
    """
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    # Spearman (default): rank correlation, robust to skew/outliers.
    return float(stats.spearmanr(x, y).statistic)


def correlation_matrix(corr_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the tidy correlation frame to a weather(rows) x pollutant(cols) matrix.

    Helper for the heatmap; kept separate so the long form (with n) stays the source
    of truth and the wide form is a pure view.
    """
    return corr_long.pivot(index="weather", columns="pollutant", values="corr")


# --------------------------------------------------------------------------- #
# EPA-index segmentation (NOT correlation — the index is derived from pollutants)
# --------------------------------------------------------------------------- #
def epa_index_summary(df: pd.DataFrame, group_col: str = "continent") -> pd.DataFrame:
    """Distribution of US-EPA air-quality categories per group (share + counts).

    The EPA index is used here for *segmentation only*: because it is derived from the
    same pollutants, correlating it against PM2.5 would be tautological. Mapping it to
    the human-readable :data:`EPA_LABELS` and showing its share by continent is a
    legitimate, leakage-free descriptive view.

    If ``group_col`` (default ``continent``) is missing, we try to enrich via the
    sibling ``regions.add_region_columns``; if that is unavailable we fall back to a
    single "All" bucket so the function never crashes. Caveat the caller should keep in
    mind: continents pool many cities of very different sizes, so a continent's
    distribution is sampling-density weighted, not population weighted.
    """
    work = df.copy()
    if group_col not in work.columns:
        work = _ensure_group_col(work, group_col)

    epa_col = "air_quality_us_epa_index"
    if epa_col not in work.columns:
        return pd.DataFrame(columns=[group_col, "epa_label", "count", "share"])

    cat = pd.to_numeric(work[epa_col], errors="coerce")
    work = work.assign(epa_label=cat.map(EPA_LABELS))
    work = work.dropna(subset=["epa_label"])
    if work.empty:
        return pd.DataFrame(columns=[group_col, "epa_label", "count", "share"])

    counts = (
        work.groupby([group_col, "epa_label"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(group_col, observed=True)["count"].transform("sum")
    counts = counts.assign(share=counts["count"] / totals)
    # Order EPA labels Good -> Hazardous for stable downstream stacking.
    counts["epa_label"] = pd.Categorical(counts["epa_label"], categories=_EPA_ORDER, ordered=True)
    return counts.sort_values([group_col, "epa_label"]).reset_index(drop=True)


def _ensure_group_col(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Best-effort add of a region column; fall back to a single 'All' bucket.

    Returns a new frame (never mutates input). Used so EPA/segmentation views work
    even before the sibling ``regions`` module exists.
    """
    if _regions is not None and hasattr(_regions, "add_region_columns"):
        try:
            enriched = _regions.add_region_columns(df)
            if group_col in enriched.columns:
                return enriched
        except Exception:
            pass  # degrade to single-bucket rather than crash the analysis
    return df.assign(**{group_col: "All"})


# --------------------------------------------------------------------------- #
# Wash-out hypothesis: wind / precip quantile bins vs median PM
# --------------------------------------------------------------------------- #
def wind_precip_vs_pm(df: pd.DataFrame, pm_col: str = "air_quality_pm2_5") -> pd.DataFrame:
    """Median PM per quantile-bin of wind and of precipitation (non-parametric).

    Tests the wash-out / ventilation hypotheses WITHOUT assuming linearity: instead of
    fitting a slope, we ``pd.qcut`` each driver into quantile bins and report the
    MEDIAN PM per bin (median, not mean, because PM is right-skewed). A monotone
    decrease across wind bins is consistent with ventilation; a drop in the wettest
    precip bins is consistent with wet deposition. ``duplicates="drop"`` handles
    drivers (esp. precip, which is mostly zero) whose quantile edges collide.

    Returns a tidy frame ``driver, bin, bin_rank, n, pm_median``. This is descriptive
    co-movement on a snapshot — confounded by city identity, not a controlled
    experiment.
    """
    if pm_col not in df.columns:
        return pd.DataFrame(columns=["driver", "bin", "bin_rank", "n", "pm_median"])

    pm = pd.to_numeric(df[pm_col], errors="coerce")
    frames: list[pd.DataFrame] = []
    for driver in ("wind_kph", "precip_mm"):
        if driver not in df.columns:
            continue
        frames.append(_binned_median(pd.to_numeric(df[driver], errors="coerce"), pm, driver))

    if not frames:
        return pd.DataFrame(columns=["driver", "bin", "bin_rank", "n", "pm_median"])
    return pd.concat(frames, ignore_index=True)


def _binned_median(driver: pd.Series, pm: pd.Series, driver_name: str, q: int = 5) -> pd.DataFrame:
    """Quantile-bin one driver and return median PM + n per bin (ordered low->high)."""
    mask = driver.notna() & pm.notna()
    d, p = driver[mask], pm[mask]
    if d.nunique() < 2:
        return pd.DataFrame(columns=["driver", "bin", "bin_rank", "n", "pm_median"])

    try:
        bins = pd.qcut(d, q=q, duplicates="drop")
    except (ValueError, IndexError):
        return pd.DataFrame(columns=["driver", "bin", "bin_rank", "n", "pm_median"])

    grouped = (
        pd.DataFrame({"bin": bins, "pm": p})
        .groupby("bin", observed=True)["pm"]
        .agg(pm_median="median", n="count")
        .reset_index()
    )
    grouped = grouped.assign(
        driver=driver_name,
        bin=grouped["bin"].astype(str),
        bin_rank=range(1, len(grouped) + 1),
    )
    return grouped[["driver", "bin", "bin_rank", "n", "pm_median"]]


# --------------------------------------------------------------------------- #
# Figures (plotly) — return a Figure; never .show()
# --------------------------------------------------------------------------- #
def fig_aq_weather_heatmap(corr_df: pd.DataFrame, out: "Path | None" = None) -> go.Figure:
    """Annotated diverging heatmap of weather x pollutant Spearman correlation.

    Accepts either the tidy long frame from :func:`aq_weather_correlation` or an
    already-pivoted matrix. Diverging RdBu scale centered at 0 so sign (ventilation vs
    accumulation) reads at a glance; cells annotated with the coefficient. n is not
    drawn per cell to avoid clutter — it lives in the tidy frame for audit.
    """
    matrix = corr_df if _is_matrix(corr_df) else correlation_matrix(corr_df)
    matrix = matrix.reindex(index=[w for w in WEATHER_COLS if w in matrix.index])
    z = matrix.to_numpy(dtype="float64")
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(z))

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[_pretty(c) for c in matrix.columns],
            y=[_pretty(r) for r in matrix.index],
            colorscale="RdBu_r",
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            text=text,
            texttemplate="%{text}",
            colorbar={"title": "Spearman ρ"},
            hovertemplate="weather=%{y}<br>pollutant=%{x}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={
            "text": "Weather vs air-quality co-movement (Spearman ρ)<br>"
            "<sub>Rank correlation, pairwise-complete. Snapshot co-movement, "
            "spatially confounded — not causal. EPA index excluded (derived).</sub>"
        },
        xaxis_title="Pollutant",
        yaxis_title="Weather variable",
        template="plotly_white",
    )
    fig.update_yaxes(autorange="reversed")
    _maybe_write(fig, out)
    return fig


def fig_pm_vs_wind(binned_df: pd.DataFrame, out: "Path | None" = None) -> go.Figure:
    """Line of median PM vs wind quantile bin (the ventilation/wash-out view).

    Reads the wind rows out of :func:`wind_precip_vs_pm`. A downward slope across
    increasing wind bins is consistent with ventilation dispersing particulates; the
    median (not mean) keeps a few extreme-pollution days from dominating.
    """
    wind = binned_df[binned_df["driver"] == "wind_kph"].sort_values("bin_rank")
    fig = go.Figure()
    if not wind.empty:
        fig.add_trace(
            go.Scatter(
                x=wind["bin_rank"],
                y=wind["pm_median"],
                mode="lines+markers",
                line={"color": "#2c7fb8", "width": 3},
                marker={"size": 9},
                customdata=np.stack([wind["bin"], wind["n"]], axis=-1),
                hovertemplate=(
                    "wind bin=%{customdata[0]}<br>median PM2.5=%{y:.1f}<br>"
                    "n=%{customdata[1]}<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=wind["bin_rank"].tolist(),
            ticktext=wind["bin"].tolist(),
        )
    fig.update_layout(
        title={
            "text": "Median PM2.5 across wind-speed quantiles<br>"
            "<sub>Tests ventilation/wash-out non-parametrically. Confounded by city "
            "identity — descriptive, not causal.</sub>"
        },
        xaxis_title="Wind-speed quantile bin (low -> high)",
        yaxis_title="Median PM2.5 (µg/m³)",
        template="plotly_white",
    )
    _maybe_write(fig, out)
    return fig


def fig_epa_by_continent(summary_df: pd.DataFrame, out: "Path | None" = None) -> go.Figure:
    """Stacked bar of EPA-category SHARE per continent (green -> maroon ramp).

    Reads the output of :func:`epa_index_summary`. Shares (not counts) so continents
    with very different sample sizes stay comparable; categories stacked Good ->
    Hazardous with the public-health colour ramp. Caveat: shares are sampling-density
    weighted across this city snapshot, not population weighted.
    """
    fig = go.Figure()
    if summary_df.empty:
        fig.update_layout(title="EPA index by continent (no data)", template="plotly_white")
        _maybe_write(fig, out)
        return fig

    group_col = summary_df.columns[0]
    groups = list(pd.unique(summary_df[group_col]))
    for label in _EPA_ORDER:
        sub = summary_df[summary_df["epa_label"] == label]
        share_by_group = sub.set_index(group_col)["share"] if not sub.empty else pd.Series(dtype="float64")
        y = [float(share_by_group.get(g, 0.0)) for g in groups]
        fig.add_trace(
            go.Bar(
                name=label,
                x=groups,
                y=y,
                marker_color=_EPA_COLORS[label],
                hovertemplate=f"{group_col}=%{{x}}<br>{label}: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title={
            "text": "US-EPA air-quality category share by continent<br>"
            "<sub>Segmentation only (index is derived from pollutants). "
            "Share = sampling-density weighted across city snapshot.</sub>"
        },
        xaxis_title=group_col.title(),
        yaxis_title="Share of observations",
        yaxis={"tickformat": ".0%"},
        legend_title="EPA category",
        template="plotly_white",
    )
    _maybe_write(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Small internal helpers
# --------------------------------------------------------------------------- #
def _is_matrix(df: pd.DataFrame) -> bool:
    """True if `df` is already a pivoted correlation matrix (not the tidy long form)."""
    return "corr" not in df.columns and "pollutant" not in df.columns


def _pretty(col: str) -> str:
    """Human-friendly axis label: strip the ``air_quality_`` prefix, de-snake."""
    return col.replace("air_quality_", "").replace("_", " ")


def _maybe_write(fig: go.Figure, out: "Path | None") -> None:
    """Persist a plotly figure ONLY when `out` is given.

    Uses static PNG via kaleido when available; otherwise falls back to a
    self-contained HTML next to the requested path so the artefact is never lost on an
    interpreter without the kaleido wheel.
    """
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if optional_deps.HAS_KALEIDO:
        fig.write_image(str(out), scale=2)
    else:
        fig.write_html(str(out.with_suffix(".html")), include_plotlyjs="cdn")

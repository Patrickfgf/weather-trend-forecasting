"""Unique Analysis 1 — climate patterns by region (plotly figures).

Honest framing baked in everywhere: this dataset is a rolling DAILY-SNAPSHOT
history spanning sub-annual to ~1 year, NOT a multi-decade climate record. So
"long-term" here means *seasonal / period-over-period WITHIN the sample window*,
never decadal trends. The sample span is computed from the data and stamped into
every figure subtitle so a reader can never mistake the timescale.

Two methodological choices that matter:
- **Two-stage means** (city-mean -> region-mean). The raw rows are NOT a balanced
  panel: some regions have many cities, some cities are sampled more often. A naive
  ``groupby(region).mean()`` would let data-dense cities dominate a region's mean.
  We first collapse to one value per (city, period), then average those city values
  inside the region — every city gets one vote, removing sampling-density bias.
- **Hemisphere-aware seasons** are handled upstream (``engineering.add_calendar`` /
  ``regions``); here we work on the calendar ``year_month`` axis and rely on
  ``climate_zone`` (a latitude band) as the default region so N/S cities of the same
  band are comparable.

All functions are pure (DataFrame in -> DataFrame/Figure out), never read or write
the raw file, and never call ``fig.show()``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config, optional_deps, regions

# Region columns produced by ``regions.add_region_columns``. If the requested
# region column is absent we enrich on the fly.
_REGION_COLS = ("climate_zone", "continent", "hemisphere")

# Sentinel used by regions.py for unresolved metadata; excluded from figures so a
# noisy "Unknown" band does not dominate the visual.
_UNKNOWN = "Unknown"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_region(df: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """Return a frame guaranteed to carry ``region_col`` (enrich via regions if missing).

    WHY best-effort: the analysis takes any region column name; the only ones we know
    how to synthesize are the geographic bands from :mod:`regions`. Anything else must
    already be present or we raise a clear error rather than silently mislabel.
    """
    if region_col in df.columns:
        return df
    if region_col in _REGION_COLS:
        return regions.add_region_columns(df)
    raise KeyError(
        f"region_col {region_col!r} not in frame and not derivable by regions.add_region_columns "
        f"(known derivable: {_REGION_COLS})"
    )


def _year_month(ts: pd.Series) -> pd.Series:
    """Map a tz-aware timestamp Series to a tz-naive month-start Timestamp (the period axis).

    Period-start timestamps (not strings) keep the x-axis chronologically ordered and
    plot as a real datetime axis in plotly.
    """
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()


def sample_span(df: pd.DataFrame, time_col: str = config.TIME_COL) -> str:
    """Human-readable span of the sample window, for honest figure subtitles.

    Returns e.g. ``"2024-05-12 -> 2025-05-10 (~12.0 months, daily snapshots)"`` so every
    figure states the true (sub-annual to ~1yr) timescale instead of implying climate-scale.
    """
    ts = pd.to_datetime(df[time_col], utc=True).dropna()
    if ts.empty:
        return "sample span unavailable (no valid timestamps)"
    lo, hi = ts.min(), ts.max()
    months = (hi - lo).days / 30.44
    return f"{lo.date()} -> {hi.date()} (~{months:.1f} months, daily snapshots; within-sample, not climate-scale)"


# --------------------------------------------------------------------------- #
# Aggregations (grain-explicit, two-stage)
# --------------------------------------------------------------------------- #
def seasonal_cycle(
    df: pd.DataFrame,
    region_col: str = "climate_zone",
    value_col: str = config.TARGET,
    time_col: str = config.TIME_COL,
) -> pd.DataFrame:
    """Two-stage seasonal cycle per region. Grain: 1 row = (region, year_month).

    Stage 1: collapse to one value per (region, city, year_month) — the city-month mean.
    Stage 2: across cities in the region-month, report mean/std/p10/p90/n_cities.
    Each city gets one vote per month, so data-dense cities cannot bias the region mean.

    Columns: ``region_col``, ``year_month`` (month-start Timestamp), ``mean``, ``std``,
    ``p10``, ``p90``, ``n_cities``. Rows with a NaN value or "Unknown" region are dropped.
    """
    work = _ensure_region(df, region_col)
    work = work.loc[:, [region_col, *config.KEY_COLS, time_col, value_col]].copy()
    work = work.assign(year_month=_year_month(work[time_col]))
    work = work.dropna(subset=[value_col, "year_month"])
    work = work[work[region_col].astype("string") != _UNKNOWN]
    if work.empty:
        return pd.DataFrame(
            columns=[region_col, "year_month", "mean", "std", "p10", "p90", "n_cities"]
        )

    # Stage 1 — city-month mean (collapses intra-month duplicate snapshots).
    city_month = (
        work.groupby([region_col, *config.KEY_COLS, "year_month"], observed=True)[value_col]
        .mean()
        .reset_index()
    )

    # Stage 2 — region-month summary over the city means (one vote per city).
    grouped = city_month.groupby([region_col, "year_month"], observed=True)[value_col]
    out = (
        grouped.agg(
            mean="mean",
            std="std",
            p10=lambda s: s.quantile(0.10),
            p90=lambda s: s.quantile(0.90),
            n_cities="count",
        )
        .reset_index()
        .sort_values([region_col, "year_month"])
        .reset_index(drop=True)
    )
    return out


def period_over_period(
    df: pd.DataFrame,
    region_col: str = "climate_zone",
    value_col: str = config.TARGET,
    freq: str = "W",
) -> pd.DataFrame:
    """Within-sample period-over-period CHANGE per region (Δ vs previous period).

    NOT a trend claim. With a sub-annual to ~1yr window we cannot estimate a climate
    trend; this only reports how the two-stage regional mean moved from one ``freq``
    bucket to the next inside the sample. Grain: 1 row = (region, period).

    Two-stage again: city-period mean first, then region-period mean, then
    ``groupby(region).diff()`` on the ordered periods for ``delta``. ``delta`` is NaN at
    each region's first period (no predecessor).
    """
    work = _ensure_region(df, region_col)
    work = work.loc[:, [region_col, *config.KEY_COLS, config.TIME_COL, value_col]].copy()
    ts = pd.to_datetime(work[config.TIME_COL], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    work = work.assign(period=ts.dt.to_period(freq).dt.to_timestamp())
    work = work.dropna(subset=[value_col, "period"])
    work = work[work[region_col].astype("string") != _UNKNOWN]
    cols = [region_col, "period", "mean", "delta", "n_cities"]
    if work.empty:
        return pd.DataFrame(columns=cols)

    # Stage 1 — city-period mean.
    city_period = (
        work.groupby([region_col, *config.KEY_COLS, "period"], observed=True)[value_col]
        .mean()
        .reset_index()
    )
    # Stage 2 — region-period mean + city count.
    region_period = (
        city_period.groupby([region_col, "period"], observed=True)[value_col]
        .agg(mean="mean", n_cities="count")
        .reset_index()
        .sort_values([region_col, "period"])
        .reset_index(drop=True)
    )
    region_period["delta"] = region_period.groupby(region_col, observed=True)["mean"].diff()
    return region_period.loc[:, cols]


# --------------------------------------------------------------------------- #
# Figures (plotly; never .show())
# --------------------------------------------------------------------------- #
def _maybe_export(fig: go.Figure, out: "Path | None") -> None:
    """Export a plotly figure when ``out`` is set: PNG if kaleido is present, else HTML.

    Degrades gracefully — never crashes the pipeline because a static-export wheel is
    missing; an HTML fallback preserves the interactive figure.
    """
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if optional_deps.HAS_KALEIDO:
        fig.write_image(str(out), width=1100, height=650, scale=2)
    else:
        fig.write_html(str(out.with_suffix(".html")), include_plotlyjs="cdn")


def fig_seasonal_cycle(
    cycle_df: pd.DataFrame, value_col: str = config.TARGET, out: "Path | None" = None
) -> go.Figure:
    """Line per region over ``year_month`` with a p10-p90 band. One trace pair per region.

    The shaded band is the across-city spread (p10-p90) WITHIN each region-month, so a
    wide band flags a heterogeneous region (cities disagree), not measurement noise.
    Expects the output of :func:`seasonal_cycle`.
    """
    region_col = cycle_df.columns[0]
    fig = go.Figure()
    if cycle_df.empty:
        fig.update_layout(title="Seasonal cycle — no data after region/quality filtering")
        return fig

    palette = px.colors.qualitative.Safe
    for i, (region, grp) in enumerate(cycle_df.groupby(region_col, observed=True)):
        grp = grp.sort_values("year_month")
        color = palette[i % len(palette)]
        rgba = _to_rgba(color, 0.15)
        # p10-p90 band: draw p90 up then p10 back down, filled to previous trace.
        fig.add_trace(go.Scatter(
            x=grp["year_month"], y=grp["p90"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            legendgroup=str(region),
        ))
        fig.add_trace(go.Scatter(
            x=grp["year_month"], y=grp["p10"], mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor=rgba,
            showlegend=False, hoverinfo="skip", legendgroup=str(region),
        ))
        fig.add_trace(go.Scatter(
            x=grp["year_month"], y=grp["mean"], mode="lines+markers",
            name=str(region), legendgroup=str(region),
            line=dict(color=color, width=2),
            customdata=grp["n_cities"],
            hovertemplate=(
                f"{region}<br>%{{x|%Y-%m}}<br>{value_col}=%{{y:.1f}}"
                "<br>n_cities=%{customdata}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"Seasonal cycle of {value_col} by {region_col}"
                f"<br><sup>Two-stage mean (city then region) · band = across-city p10-p90 · "
                "within-sample seasonal pattern, NOT a decadal trend</sup>"
            )
        ),
        xaxis_title="year-month", yaxis_title=value_col,
        legend_title=region_col, template="plotly_white", hovermode="x unified",
    )
    _maybe_export(fig, out)
    return fig


def fig_zone_distributions(
    df: pd.DataFrame,
    value_col: str = config.TARGET,
    region_col: str = "climate_zone",
    out: "Path | None" = None,
) -> go.Figure:
    """Box+violin of ``value_col`` by ``region_col`` over city-month means (de-densified).

    WHY plot city-month means rather than raw rows: the raw distribution is dominated by
    cities/months with more snapshots. Collapsing to one value per (city, month) gives
    every city-month equal weight, so the spread reflects geography, not sampling cadence.
    """
    work = _ensure_region(df, region_col)
    needed = [region_col, *config.KEY_COLS, config.TIME_COL, value_col]
    work = work.loc[:, needed].copy()
    work = work.assign(year_month=_year_month(work[config.TIME_COL]))
    work = work.dropna(subset=[value_col, "year_month"])
    work = work[work[region_col].astype("string") != _UNKNOWN]

    fig = go.Figure()
    if work.empty:
        fig.update_layout(title=f"Distribution of {value_col} — no data after filtering")
        return fig

    city_month = (
        work.groupby([region_col, *config.KEY_COLS, "year_month"], observed=True)[value_col]
        .mean()
        .reset_index()
    )
    order = (
        city_month.groupby(region_col, observed=True)[value_col]
        .median()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    fig = px.violin(
        city_month, x=region_col, y=value_col, color=region_col,
        box=True, points=False, category_orders={region_col: order},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        title=dict(text=(
            f"Distribution of {value_col} by {region_col}"
            "<br><sup>One value per (city, month) — equal weight per city-month, "
            "removes sampling-density bias · within-sample</sup>"
        )),
        xaxis_title=region_col, yaxis_title=value_col,
        template="plotly_white", showlegend=False,
    )
    _maybe_export(fig, out)
    return fig


def fig_climate_heatmap(
    cycle_df: pd.DataFrame, value_col: str = config.TARGET, out: "Path | None" = None
) -> go.Figure:
    """Region x calendar-month heatmap of the two-stage mean ``value_col``.

    Collapses the (region, year_month) cycle onto a 1-12 calendar-month axis (averaging
    across any repeated months in the window) so the seasonal shape reads as a compact
    grid. Expects the output of :func:`seasonal_cycle`.
    """
    region_col = cycle_df.columns[0]
    if cycle_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Climate heatmap — no data after region/quality filtering")
        return fig

    work = cycle_df.assign(month=pd.to_datetime(cycle_df["year_month"]).dt.month)
    grid = (
        work.groupby([region_col, "month"], observed=True)["mean"]
        .mean()
        .reset_index()
        .pivot(index=region_col, columns="month", values="mean")
        .reindex(columns=range(1, 13))
    )
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(),
        x=month_labels,
        y=grid.index.astype(str).tolist(),
        colorscale="RdBu_r",
        colorbar=dict(title=value_col),
        hovertemplate=f"%{{y}}<br>%{{x}}<br>{value_col}=%{{z:.1f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"Mean {value_col}: {region_col} x calendar month"
            "<br><sup>Two-stage mean · calendar month (not hemisphere-shifted) · "
            "within-sample seasonal pattern, NOT a decadal trend</sup>"
        )),
        xaxis_title="calendar month", yaxis_title=region_col, template="plotly_white",
    )
    _maybe_export(fig, out)
    return fig


# --------------------------------------------------------------------------- #
# Color utility
# --------------------------------------------------------------------------- #
def _to_rgba(color: str, alpha: float) -> str:
    """Convert a plotly hex/``rgb(...)`` color to an ``rgba(...)`` string with ``alpha``.

    Used for the translucent p10-p90 band fill; keeps the band tinted to its region's
    line color instead of a flat gray.
    """
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb(") and color.endswith(")"):
        inner = color[4:-1]
        return f"rgba({inner},{alpha})"
    return f"rgba(99,110,250,{alpha})"  # plotly default blue fallback

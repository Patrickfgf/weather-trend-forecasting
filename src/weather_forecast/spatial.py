"""Spatial visualisation & geographical-pattern analysis (Unique Analyses 4 & 5).

GDAL-free by design: every map uses Plotly's built-in *natural-earth* geometries
via ``scatter_geo`` (point markers) or ``express.choropleth`` with
``locationmode="country names"``. No GeoPandas/Shapely/Fiona, so the notebook runs
on a bare scientific-Python stack.

Two honesty constraints are baked in:

1. **One row per city, latest snapshot.** The raw frame is a rolling daily-snapshot
   history, so a city appears on many days. Mapping every row stacks markers and
   over-weights dense histories; :func:`latest_snapshot` collapses to the most recent
   reading per ``(country, location_name)`` before any map is drawn.
2. **Explicit ``range_color``.** A single sensor outlier (e.g. a spurious AQI spike)
   otherwise flattens the whole colour scale. Ranges are derived from robust
   percentiles, not raw min/max.

Region/continent/climate-zone enrichment is delegated to the sibling
``weather_forecast.regions`` module (``add_region_columns`` + ``ALIASES``); this
module imports it defensively and degrades to a latitude-band climate zone if it is
unavailable, so a missing sibling never crashes a Restart-Run-All run.

Caveats surfaced in figure subtitles: this is a sub-annual snapshot history (NOT a
multi-decade climate record), so "climate" here means *within-sample seasonal*
structure; regional means are two-stage (city-mean then region-mean) to avoid
sampling-density bias.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config

if TYPE_CHECKING:  # import only for type checkers; keeps the hot path light
    pass

logger = logging.getLogger(__name__)

# Try the sibling region-enrichment module. It is authored separately; we must not
# hard-fail if it is missing — analyses degrade to a latitude-band fallback instead.
try:  # pragma: no cover - exercised indirectly
    from . import regions as _regions  # type: ignore
    _HAS_REGIONS = True
except Exception as exc:  # ImportError or a downstream import error
    _regions = None  # type: ignore[assignment]
    _HAS_REGIONS = False
    logger.info("weather_forecast.regions unavailable (%s); using fallback enrichment", type(exc).__name__)

# Diverging map for anomalies; sequential for raw temperature; the US-EPA AQI scale
# has fixed semantic colours (1=Good green ... 6=Hazardous maroon).
_TEMP_SCALE = "RdYlBu_r"
_ANOM_SCALE = "RdBu_r"
_EPA_COLORS: dict[int, str] = {
    1: "#00e400",  # Good
    2: "#ffff00",  # Moderate
    3: "#ff7e00",  # Unhealthy for sensitive groups
    4: "#ff0000",  # Unhealthy
    5: "#8f3f97",  # Very unhealthy
    6: "#7e0023",  # Hazardous
}
_EPA_LABELS: dict[int, str] = {
    1: "1 Good", 2: "2 Moderate", 3: "3 USG",
    4: "4 Unhealthy", 5: "5 Very unhealthy", 6: "6 Hazardous",
}
_RESOLVE_MIN_FRACTION = 0.90  # warn if fewer countries than this resolve in a choropleth
_SNAPSHOT_CAVEAT = "Latest snapshot per city · rolling daily-snapshot history (sub-annual), not a climate normal"


# --------------------------------------------------------------------------- #
# Enrichment helpers (pure)
# --------------------------------------------------------------------------- #
def _enrich_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach continent/climate_zone via the regions module, with a graceful fallback.

    WHY a fallback: ``regions`` is a sibling module authored separately. If it (or any
    column it should add) is missing we still need ``climate_zone`` for peer-anomaly
    z-scores, so we synthesise a coarse latitude band rather than crash.
    """
    out = df
    if _HAS_REGIONS and hasattr(_regions, "add_region_columns"):
        try:
            out = _regions.add_region_columns(df)  # type: ignore[union-attr]
        except Exception as exc:  # never let enrichment break the analysis
            logger.warning("regions.add_region_columns failed (%s); falling back", type(exc).__name__)
            out = df.copy()
    else:
        out = df.copy()

    if "climate_zone" not in out.columns:
        out = out.assign(climate_zone=_latitude_band(out))
    if "continent" not in out.columns:
        out = out.assign(continent=pd.Series(["Unknown"] * len(out), index=out.index, dtype="object"))
    return out


def _latitude_band(df: pd.DataFrame) -> pd.Series:
    """Coarse Koppen-ish band from |latitude| — fallback when regions has no climate_zone.

    Bands group cities at a similar latitude so the peer-anomaly z-score compares
    "like with like" (a hot reading in the tropics is not anomalous; the same reading
    near the poles is). Uses absolute latitude so both hemispheres share a band.
    """
    if "latitude" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")
    abs_lat = df["latitude"].abs()
    bins = [-0.1, 23.5, 35.0, 55.0, 66.5, 90.1]
    labels = ["Tropical", "Subtropical", "Temperate", "Boreal", "Polar"]
    band = pd.cut(abs_lat, bins=bins, labels=labels, include_lowest=True)
    return band.astype("object").fillna("Unknown")


def _remap_country_names(names: pd.Series) -> pd.Series:
    """Remap dataset country names to Plotly natural-earth names via regions.ALIASES.

    Plotly's ``locationmode="country names"`` matches a fixed gazetteer; names like
    "USA" or "Russia" need remapping. We reuse the canonical alias table so the
    choropleth and the rest of the pipeline agree on names.
    """
    aliases: dict[str, str] = {}
    if _HAS_REGIONS:
        aliases = getattr(_regions, "ALIASES", {}) or {}
    return names.map(lambda c: aliases.get(c, c))


# --------------------------------------------------------------------------- #
# Grain reduction
# --------------------------------------------------------------------------- #
def latest_snapshot(
    df: pd.DataFrame,
    time_col: str = config.TIME_COL,
    city_key: tuple[str, ...] = ("country", "location_name"),
) -> pd.DataFrame:
    """Return one row per city: its most recent reading (grain = 1 row / city).

    Sort by time, group by the city key, keep the last row. This is the only honest
    frame to map — stacking every snapshot over-weights cities with longer histories.
    """
    keys = list(city_key)
    if df.empty:
        return df.copy()
    ordered = df.sort_values(time_col)
    snap = ordered.groupby(keys, observed=True, as_index=False).tail(1)
    return snap.reset_index(drop=True)


def _valid_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null or null-island (0,0) coordinates — they are missing-data sentinels."""
    if not set(config.GEO_COLS).issubset(df.columns):
        return df.copy()
    lat, lon = config.GEO_COLS
    mask = df[lat].notna() & df[lon].notna() & ~((df[lat] == 0) & (df[lon] == 0))
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# Anomaly: per-city deviation from its climate-zone peers
# --------------------------------------------------------------------------- #
def city_anomaly(df: pd.DataFrame, value_col: str = config.TARGET) -> pd.DataFrame:
    """Per-city z-score of ``value_col`` within its climate zone ("hotter than peers").

    First reduce to the latest snapshot (one row per city), enrich with climate_zone,
    then within each zone compute ``(value - zone_mean) / zone_std``. Comparing a city
    only to peers at a similar latitude band isolates "anomalous for where it is" from
    "expected because it is tropical/polar". Zone std == 0 (single-city zone) -> z = 0.
    """
    snap = _valid_geo(latest_snapshot(df))
    snap = _enrich_regions(snap)
    if value_col not in snap.columns or snap.empty:
        return snap.assign(zone_mean=np.nan, anomaly=np.nan, anomaly_z=np.nan)

    grp = snap.groupby("climate_zone", observed=True)[value_col]
    zone_mean = grp.transform("mean")
    zone_std = grp.transform("std")
    anomaly = snap[value_col] - zone_mean
    # std NaN (single member) or 0 -> undefined z; report 0 so the map renders neutral.
    z = (anomaly / zone_std.replace(0.0, np.nan)).fillna(0.0)
    return snap.assign(zone_mean=zone_mean, anomaly=anomaly, anomaly_z=z)


# --------------------------------------------------------------------------- #
# Map factories (Plotly; natural-earth, GDAL-free)
# --------------------------------------------------------------------------- #
def _base_geo_layout(fig: go.Figure, title: str, subtitle: str) -> go.Figure:
    """Apply the shared natural-earth projection + title/subtitle to a scatter_geo figure."""
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True, countrycolor="#cccccc",
        showland=True, landcolor="#f5f5f3",
        showocean=True, oceancolor="#eaf2f8",
        showcoastlines=True, coastlinecolor="#bbbbbb",
    )
    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    return fig


def fig_scatter_geo_temp(snapshot_df: pd.DataFrame, value_col: str = config.TARGET) -> go.Figure:
    """Natural-earth point map of latest-snapshot temperature, coloured by value.

    Robust ``range_color`` (5th-95th percentile) keeps one extreme reading from
    washing out the scale. ``hover_name`` is the city.
    """
    snap = _valid_geo(latest_snapshot(snapshot_df))
    lat, lon = config.GEO_COLS
    lo, hi = _robust_range(snap[value_col]) if value_col in snap.columns and not snap.empty else (0.0, 1.0)
    fig = px.scatter_geo(
        snap, lat=lat, lon=lon, color=value_col,
        hover_name="location_name" if "location_name" in snap.columns else None,
        hover_data={c: True for c in ("country", value_col) if c in snap.columns},
        color_continuous_scale=_TEMP_SCALE, range_color=(lo, hi),
        labels={value_col: value_col.replace("_", " ").title()},
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.8, "line": {"width": 0.3, "color": "#444"}})
    return _base_geo_layout(fig, f"Global {value_col.replace('_', ' ').title()}", _SNAPSHOT_CAVEAT)


def fig_scatter_geo_aqi(snapshot_df: pd.DataFrame, aqi_col: str = "air_quality_us_epa_index") -> go.Figure:
    """Natural-earth point map of an air-quality index using fixed EPA semantic colours.

    The US-EPA index is a 1..6 ordinal with standard colours (green->maroon), so it is
    mapped categorically rather than on a continuous scale.
    """
    snap = _valid_geo(latest_snapshot(snapshot_df))
    lat, lon = config.GEO_COLS
    if aqi_col not in snap.columns or snap.empty:
        return _base_geo_layout(go.Figure(), "Air-Quality Index", f"{aqi_col} unavailable")

    plot = snap.assign(_epa=snap[aqi_col].astype("Int64").astype("object").map(_EPA_LABELS))
    plot = plot.dropna(subset=["_epa"])
    order = [_EPA_LABELS[k] for k in sorted(_EPA_LABELS)]
    cmap = {_EPA_LABELS[k]: _EPA_COLORS[k] for k in _EPA_COLORS}
    fig = px.scatter_geo(
        plot, lat=lat, lon=lon, color="_epa",
        hover_name="location_name" if "location_name" in plot.columns else None,
        hover_data={c: True for c in ("country",) if c in plot.columns},
        category_orders={"_epa": order}, color_discrete_map=cmap,
        labels={"_epa": "US-EPA AQI"},
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.85, "line": {"width": 0.3, "color": "#444"}})
    return _base_geo_layout(fig, "US-EPA Air-Quality Index", _SNAPSHOT_CAVEAT)


def fig_anomaly_map(anomaly_df: pd.DataFrame) -> go.Figure:
    """Diverging (RdBu, centered at 0) point map of per-city peer-anomaly z-scores.

    Expects the output of :func:`city_anomaly` (column ``anomaly_z``). Symmetric
    ``range_color`` around 0 so equal-magnitude hot/cold anomalies read as equal saturation.
    """
    snap = _valid_geo(anomaly_df)
    lat, lon = config.GEO_COLS
    if "anomaly_z" not in snap.columns or snap.empty:
        return _base_geo_layout(go.Figure(), "Temperature Anomaly", "anomaly_z unavailable")

    bound = float(np.nanpercentile(snap["anomaly_z"].abs(), 98)) if snap["anomaly_z"].notna().any() else 1.0
    bound = max(bound, 0.5)
    fig = px.scatter_geo(
        snap, lat=lat, lon=lon, color="anomaly_z",
        hover_name="location_name" if "location_name" in snap.columns else None,
        hover_data={c: True for c in ("country", "climate_zone", config.TARGET) if c in snap.columns},
        color_continuous_scale=_ANOM_SCALE, range_color=(-bound, bound),
        color_continuous_midpoint=0.0, labels={"anomaly_z": "z vs zone peers"},
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.85, "line": {"width": 0.3, "color": "#444"}})
    return _base_geo_layout(
        fig, "Temperature Anomaly vs Climate-Zone Peers",
        "Per-city z-score within latitude-band peers · 0 = typical for its zone",
    )


# --------------------------------------------------------------------------- #
# Country / continent aggregation (two-stage means)
# --------------------------------------------------------------------------- #
def country_weather_summary(
    df: pd.DataFrame,
    value_cols: tuple[str, ...] = ("temperature_celsius", "humidity", "wind_kph", "air_quality_pm2_5"),
) -> pd.DataFrame:
    """One row per country via TWO-STAGE means (city-mean then country-mean).

    WHY two-stage: averaging raw rows lets a city with 300 snapshots outweigh a city
    with 3. Collapsing to a city-mean first gives every city equal weight in its
    country, removing sampling-density bias. Asserts the output is unique per country.
    """
    present = [c for c in value_cols if c in df.columns]
    keys = config.KEY_COLS
    if df.empty or not present:
        return pd.DataFrame(columns=["country", *present, "n_cities", "n_days"])

    # Stage 1: city-level means (+ per-city day count).
    city = (
        df.groupby(keys, observed=True)
        .agg({**{c: "mean" for c in present}, config.TIME_COL: "nunique"})
        .rename(columns={config.TIME_COL: "n_days"})
        .reset_index()
    )
    # Stage 2: country-level mean of city means.
    country = (
        city.groupby("country", observed=True)
        .agg({**{c: "mean" for c in present},
              "location_name": "nunique",
              "n_days": "sum"})
        .rename(columns={"location_name": "n_cities"})
        .reset_index()
    )
    assert country["country"].is_unique, "country_weather_summary grain violated: country not unique"
    return country


def continent_comparison(df: pd.DataFrame, value_col: str = config.TARGET) -> pd.DataFrame:
    """Per-continent distribution of ``value_col`` (mean/std + p10/p50/p90 + counts).

    Aggregates over the two-stage country summary so each country (already city-balanced)
    contributes once per continent — avoids both row-density and country-size bias.
    """
    summary = country_weather_summary(df, value_cols=(value_col,))
    if summary.empty or value_col not in summary.columns:
        return pd.DataFrame(columns=["continent", "mean", "std", "p10", "p50", "p90", "n_countries", "n_cities"])

    enriched = _enrich_regions(summary)
    out = (
        enriched.groupby("continent", observed=True)
        .agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            p10=(value_col, lambda s: s.quantile(0.10)),
            p50=(value_col, lambda s: s.quantile(0.50)),
            p90=(value_col, lambda s: s.quantile(0.90)),
            n_countries=("country", "nunique"),
            n_cities=("n_cities", "sum"),
        )
        .reset_index()
        .sort_values("p50", ascending=False)
        .reset_index(drop=True)
    )
    return out


# --------------------------------------------------------------------------- #
# Comparative figures
# --------------------------------------------------------------------------- #
def fig_continent_box(df: pd.DataFrame, value_col: str = config.TARGET) -> go.Figure:
    """Box plot of ``value_col`` by continent, ordered by median.

    Operates on the country-level two-stage means (one point per country) so the box
    reflects between-country spread within a continent, not raw row counts.
    """
    summary = country_weather_summary(df, value_cols=(value_col,))
    if summary.empty or value_col not in summary.columns:
        return go.Figure(layout={"title": {"text": f"{value_col}: no data"}})

    enriched = _enrich_regions(summary)
    order = (
        enriched.groupby("continent", observed=True)[value_col]
        .median().sort_values(ascending=False).index.tolist()
    )
    fig = px.box(
        enriched, x="continent", y=value_col, color="continent",
        category_orders={"continent": order}, points="all",
        labels={value_col: value_col.replace("_", " ").title(), "continent": "Continent"},
    )
    fig.update_layout(
        title={"text": f"{value_col.replace('_', ' ').title()} by Continent"
                       "<br><sup>One point = one country (two-stage city→country mean)</sup>"},
        showlegend=False, margin={"t": 70},
    )
    return fig


def fig_country_choropleth(country_df: pd.DataFrame, value_col: str = "temperature_celsius") -> go.Figure:
    """Choropleth (locationmode='country names') of a per-country value, GDAL-free.

    Remaps country names via regions.ALIASES and WARNS if <90% of countries resolve to
    Plotly's gazetteer (a low rate means the alias table needs more entries). Robust
    ``range_color`` keeps one extreme country from flattening the scale.
    """
    if country_df.empty or value_col not in country_df.columns or "country" not in country_df.columns:
        return go.Figure(layout={"title": {"text": f"{value_col}: no data for choropleth"}})

    plot = country_df.assign(_loc=_remap_country_names(country_df["country"]))
    _warn_unresolved_countries(plot["_loc"])

    lo, hi = _robust_range(plot[value_col])
    fig = px.choropleth(
        plot, locations="_loc", locationmode="country names",
        color=value_col, hover_name="country",
        color_continuous_scale=_TEMP_SCALE, range_color=(lo, hi),
        labels={value_col: value_col.replace("_", " ").title()},
    )
    fig.update_geos(projection_type="natural earth", showcoastlines=True, coastlinecolor="#bbbbbb")
    fig.update_layout(
        title={"text": f"Country Mean {value_col.replace('_', ' ').title()}"
                       "<br><sup>Two-stage mean (city→country) · natural-earth, GDAL-free</sup>"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )
    return fig


def fig_top_bottom_countries(country_df: pd.DataFrame, value_col: str, k: int = 15) -> go.Figure:
    """Diverging horizontal bar of the top-k and bottom-k countries by ``value_col``.

    The two tails answer "where is it most/least X" at a glance; the global mean is the
    visual zero so bars diverge around the typical value rather than around an arbitrary 0.
    """
    if country_df.empty or value_col not in country_df.columns or "country" not in country_df.columns:
        return go.Figure(layout={"title": {"text": f"{value_col}: no data"}})

    ranked = country_df.dropna(subset=[value_col]).sort_values(value_col)
    if ranked.empty:
        return go.Figure(layout={"title": {"text": f"{value_col}: no data"}})

    k = max(1, min(k, len(ranked)))
    bottom = ranked.head(k)
    top = ranked.tail(k)
    sel = pd.concat([bottom, top]).drop_duplicates(subset=["country"])

    center = float(country_df[value_col].mean())
    sel = sel.assign(_dev=sel[value_col] - center).sort_values(value_col)
    colors = np.where(sel["_dev"] >= 0, "#c0392b", "#2471a3")
    fig = go.Figure(
        go.Bar(
            x=sel["_dev"], y=sel["country"], orientation="h",
            marker={"color": colors},
            customdata=np.stack([sel[value_col]], axis=-1),
            hovertemplate="%{y}<br>" + value_col + "=%{customdata[0]:.2f}<extra></extra>",
        )
    )
    label = value_col.replace("_", " ").title()
    fig.update_layout(
        title={"text": f"Top {k} / Bottom {k} Countries — {label}"
                       f"<br><sup>Deviation from global mean ({center:.1f}) · two-stage country means</sup>"},
        xaxis_title=f"{label} − global mean",
        yaxis_title=None, margin={"l": 0, "r": 20, "t": 70, "b": 0}, bargap=0.2,
    )
    return fig


# --------------------------------------------------------------------------- #
# Internal numeric helpers
# --------------------------------------------------------------------------- #
def _robust_range(s: pd.Series, lo_q: float = 0.05, hi_q: float = 0.95) -> tuple[float, float]:
    """Return a robust (lo_q, hi_q) percentile range; guard against degenerate/empty input.

    Used for ``range_color`` so a lone outlier does not collapse the colour scale.
    """
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return (0.0, 1.0)
    lo = float(vals.quantile(lo_q))
    hi = float(vals.quantile(hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        m = float(vals.mean())
        return (m - 1.0, m + 1.0)
    return (lo, hi)


def _warn_unresolved_countries(locations: pd.Series) -> float:
    """Log a warning if <90% of country names look resolvable; return the resolved fraction.

    Plotly does not expose its gazetteer, so this is a heuristic: blank/Unknown names are
    treated as unresolved. A low fraction signals regions.ALIASES needs more entries.
    """
    total = len(locations)
    if total == 0:
        return 1.0
    bad = locations.isna() | locations.astype("string").str.strip().isin(["", "Unknown", "nan"])
    resolved = 1.0 - float(bad.sum()) / float(total)
    if resolved < _RESOLVE_MIN_FRACTION:
        logger.warning(
            "choropleth: only %.0f%% of countries resolved (< %.0f%%); extend regions.ALIASES",
            resolved * 100.0, _RESOLVE_MIN_FRACTION * 100.0,
        )
    return resolved

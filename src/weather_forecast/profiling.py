"""Defensive EDA profiling: schema, grain, per-series coverage, precip zero-inflation,
correlation matrices, and a deterministic representative-city selector.

All functions are PURE: DataFrame in -> DataFrame/dict/list out, no global state, no
print, no in-place mutation. They never read or write the raw data file. Anything
stochastic is avoided entirely so reruns are bit-for-bit identical.

Why "defensive": before any modeling we interrogate the data on its own terms — grain
(1 row = one city's snapshot at one ``last_updated``), nulls, duplicate keys, gaps,
and the precipitation zero-inflation that breaks naive Gaussian assumptions.

Honesty caveats baked in:
- This dataset is a rolling DAILY-SNAPSHOT history (sub-annual to ~1 year), NOT a
  multi-decade climate record. ``date span`` and ``span_days`` describe within-sample
  coverage, never a decadal climate trend.
- Correlations are reported with SPEARMAN as primary (rank-robust to the heavy-tailed
  precip/AQI distributions); Pearson is included only to expose where a linear read
  disagrees. Both are monotonic/linear association, possibly confounded — NOT causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Köppen-flavoured |latitude| bands used for climate-zone diversity in the
# representative-city selector. Boundaries are the conventional tropic / subtropics /
# temperate / polar cutoffs; sign(latitude) splits the hemispheres on top.
_LAT_BAND_EDGES: tuple[tuple[float, str], ...] = (
    (23.5, "Tropical"),
    (35.0, "Subtropical"),
    (55.0, "Temperate"),
    (90.0, "Polar"),
)


# --------------------------------------------------------------------------- #
# Schema profile
# --------------------------------------------------------------------------- #
def profile_schema(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, distinct count, null %, and one example value.

    The first defensive read of any frame — surfaces ``object`` columns that should be
    typed, all-null columns, and constant (n_unique==1) columns before they bite later.
    """
    n = len(df)
    rows: list[dict[str, object]] = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        example = non_null.iloc[0] if not non_null.empty else None
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique(dropna=True)),
                "pct_null": round(float(s.isna().mean() * 100.0), 4) if n else 0.0,
                "example": example,
            }
        )
    return pd.DataFrame(rows, columns=["column", "dtype", "n_unique", "pct_null", "example"])


# --------------------------------------------------------------------------- #
# Grain report
# --------------------------------------------------------------------------- #
def grain_report(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    time_col: str = config.TIME_COL,
) -> dict[str, object]:
    """Summarise the grain: row count, distinct series, duplicate-key rows, rows-per-
    series spread, and the within-sample date span.

    ``duplicate_key_rows`` counts rows sharing a full ``keys + [time_col]`` tuple — the
    grain claim is "1 row = one series at one timestamp", so any nonzero count is a
    grain violation worth flagging (cleaning resolves it by keep-last).
    """
    keys = keys or config.KEY_COLS
    n_rows = int(len(df))
    n_series = int(df.drop_duplicates(subset=keys).shape[0]) if n_rows else 0

    full_key = list(keys) + [time_col]
    dup_key_rows = int(df.duplicated(subset=full_key, keep=False).sum()) if n_rows else 0

    if n_rows:
        per_series = df.groupby(keys, observed=True).size()
        rows_per_series = {
            "min": int(per_series.min()),
            "median": float(per_series.median()),
            "max": int(per_series.max()),
        }
    else:
        rows_per_series = {"min": 0, "median": 0.0, "max": 0}

    ts = pd.to_datetime(df[time_col], errors="coerce") if time_col in df.columns else pd.Series(dtype="datetime64[ns]")
    ts = ts.dropna()
    if not ts.empty:
        date_span = {
            "start": ts.min(),
            "end": ts.max(),
            "span_days": int((ts.max() - ts.min()).days),
        }
    else:
        date_span = {"start": None, "end": None, "span_days": 0}

    return {
        "n_rows": n_rows,
        "n_series": n_series,
        "duplicate_key_rows": dup_key_rows,
        "rows_per_series": rows_per_series,
        "date_span": date_span,
        "caveat": (
            "Rolling daily-snapshot history (sub-annual to ~1yr); span is within-sample "
            "coverage, NOT a decadal climate record."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-series coverage
# --------------------------------------------------------------------------- #
def series_coverage(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    time_col: str = config.TIME_COL,
) -> pd.DataFrame:
    """Per series: n_obs, first/last date, span_days, and n_gaps (missing daily steps).

    ``n_gaps`` is the number of absent daily steps between distinct observed calendar
    days (``span_days + 1 - n_distinct_days``). It quantifies irregularity for the
    classical TS models, which assume a regular daily frequency.
    """
    keys = keys or config.KEY_COLS
    if df.empty or time_col not in df.columns:
        return pd.DataFrame(
            columns=list(keys) + ["n_obs", "first_date", "last_date", "span_days", "n_gaps"]
        )

    # Vectorized: aggregate per series in one groupby, then derive gaps from
    # distinct-day counts vs. the calendar span (no per-row Python loop).
    work = df[list(keys) + [time_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    work = work.assign(_day=work[time_col].dt.normalize())

    grouped = work.groupby(keys, observed=True)
    agg = grouped.agg(
        n_obs=(time_col, "size"),
        first_date=("_day", "min"),
        last_date=("_day", "max"),
        n_distinct_days=("_day", "nunique"),
    ).reset_index()

    span_days = (agg["last_date"] - agg["first_date"]).dt.days
    agg = agg.assign(
        span_days=span_days.astype("int64"),
        n_gaps=(span_days + 1 - agg["n_distinct_days"]).clip(lower=0).astype("int64"),
    )
    out = agg[list(keys) + ["n_obs", "first_date", "last_date", "span_days", "n_gaps"]]
    return out.sort_values("n_obs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Precipitation zero-inflation
# --------------------------------------------------------------------------- #
def precip_zero_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Per country: pct_dry_days and describe() of wet-day ``precip_mm``.

    Precip is zero-inflated (most days are dry), so a single mean is misleading. We
    split the dry mass (``pct_dry_days``) from the wet-day intensity distribution —
    the standard two-part view for a spike-at-zero variable. Fully vectorized via
    groupby; wet-day stats describe only the strictly-positive subset.
    """
    col = "precip_mm"
    out_cols = ["country", "n_days", "pct_dry_days", "wet_count", "wet_mean",
                "wet_std", "wet_min", "wet_25%", "wet_50%", "wet_75%", "wet_max"]
    if "country" not in df.columns or col not in df.columns or df.empty:
        return pd.DataFrame(columns=out_cols)

    work = df[["country", col]].copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=[col])
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    work = work.assign(_is_dry=(work[col] <= 0.0))
    g = work.groupby("country", observed=True)
    base = g.agg(n_days=(col, "size"), dry_days=("_is_dry", "sum")).reset_index()
    base = base.assign(
        pct_dry_days=(base["dry_days"] / base["n_days"] * 100.0).round(4)
    )

    wet = work.loc[~work["_is_dry"], ["country", col]]
    if wet.empty:
        desc = pd.DataFrame(columns=["country", "wet_count", "wet_mean", "wet_std",
                                     "wet_min", "wet_25%", "wet_50%", "wet_75%", "wet_max"])
    else:
        desc = (
            wet.groupby("country", observed=True)[col]
            .describe()
            .rename(
                columns={
                    "count": "wet_count", "mean": "wet_mean", "std": "wet_std",
                    "min": "wet_min", "25%": "wet_25%", "50%": "wet_50%",
                    "75%": "wet_75%", "max": "wet_max",
                }
            )
            .reset_index()
        )

    merged = base.merge(desc, on="country", how="left")
    merged = merged[[c for c in out_cols if c in merged.columns]]
    return merged.sort_values("pct_dry_days", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Correlation matrices
# --------------------------------------------------------------------------- #
def correlation_matrices(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{'spearman','pearson','abs_diff'}`` correlation matrices on the numeric
    metric subset (AQI Int8 indices coerced to float).

    SPEARMAN is primary: rank-based, so the heavy-tailed precip/AQI columns and the
    flagged-but-kept extremes don't dominate. Pearson is included to expose where a
    linear assumption diverges; ``abs_diff = |spearman - pearson|`` highlights the
    columns whose association is strongly non-linear. All correlations are monotonic/
    linear association, possibly confounded — NOT causal.
    """
    cols = cols or (config.METRIC_NUMERIC + config.AQI_COLS)
    present = [c for c in cols if c in df.columns]
    empty = pd.DataFrame(index=present, columns=present, dtype="float64")
    if not present or df.empty:
        return {"spearman": empty, "pearson": empty, "abs_diff": empty}

    # Coerce everything (incl. nullable Int8 AQI indices) to float64; pandas .corr
    # cannot operate on the nullable extension dtypes directly.
    num = df[present].apply(pd.to_numeric, errors="coerce").astype("float64")
    spearman = num.corr(method="spearman")
    pearson = num.corr(method="pearson")
    abs_diff = (spearman - pearson).abs()
    return {"spearman": spearman, "pearson": pearson, "abs_diff": abs_diff}


# --------------------------------------------------------------------------- #
# Representative-city selection (deterministic)
# --------------------------------------------------------------------------- #
def _climate_zone(abs_lat: float, lat: float) -> str:
    """Map (|lat|, sign(lat)) to a hemisphere-qualified climate-zone label."""
    if not np.isfinite(abs_lat):
        return "Unknown"
    band = next((name for edge, name in _LAT_BAND_EDGES if abs_lat <= edge), "Polar")
    hemi = "S" if lat < 0 else "N"
    return f"{band}-{hemi}"


def choose_representative_cities(
    df: pd.DataFrame,
    n: int = config.N_REPRESENTATIVE_CITIES,
) -> list[tuple[str, str]]:
    """Pick ``n`` ``(country, location_name)`` pairs maximizing coverage AND climate
    diversity. Deterministic (stable sorts + fixed tie-breaks, no randomness).

    Strategy:
    1. Score each city by data coverage (n_obs) and tag its climate zone from
       (|latitude| band, hemisphere) so we can spread the picks across zones.
    2. PREFER cities whose name is in ``config.REPRESENTATIVE_CITIES_FALLBACK`` when
       present in the data — they were curated for climate diversity.
    3. Fill remaining slots data-driven: round-robin across climate zones, taking the
       highest-coverage city in each zone, so we don't pick six temperate-N cities.

    Why coverage + diversity (not coverage alone): the per-city deep-dive is meant to
    represent distinct climates; picking only the densest series would over-represent
    one zone and hide the hemisphere-aware seasonal contrast.
    """
    keys = config.KEY_COLS
    if df.empty or not all(c in df.columns for c in keys):
        return []

    cov = series_coverage(df)
    if cov.empty:
        return []

    # Attach a representative latitude per city (median is robust to stray rows).
    geo = (
        df[keys + config.GEO_COLS]
        .groupby(keys, observed=True)[config.GEO_COLS]
        .median()
        .reset_index()
        if all(c in df.columns for c in config.GEO_COLS)
        else None
    )
    cat = cov.merge(geo, on=keys, how="left") if geo is not None else cov.assign(latitude=np.nan)

    abs_lat = cat["latitude"].abs()
    cat = cat.assign(
        zone=[_climate_zone(a, l) for a, l in zip(abs_lat, cat["latitude"])]
    )
    # Deterministic ordering: best coverage first, then stable name tie-break.
    cat = cat.sort_values(
        ["n_obs"] + keys, ascending=[False] + [True] * len(keys)
    ).reset_index(drop=True)

    selected: list[tuple[str, str]] = []
    seen_zones: set[str] = set()

    def _pair(row: pd.Series) -> tuple[str, str]:
        return (str(row["country"]), str(row["location_name"]))

    # Stage 1: preferred curated names present in the data (one per name, dedup zone-
    # aware but still allowed even if zone repeats — these are explicitly curated).
    fallback = set(config.REPRESENTATIVE_CITIES_FALLBACK)
    if "location_name" in cat.columns:
        preferred = cat[cat["location_name"].astype(str).isin(fallback)]
        for _, row in preferred.iterrows():
            if len(selected) >= n:
                break
            pair = _pair(row)
            if pair not in selected:
                selected.append(pair)
                seen_zones.add(str(row["zone"]))

    # Stage 2: data-driven round-robin across UNSEEN climate zones (highest coverage
    # city per zone), so the picks span Tropical/Subtropical/Temperate/Polar x N/S.
    if len(selected) < n:
        for _, row in cat.iterrows():
            if len(selected) >= n:
                break
            zone = str(row["zone"])
            pair = _pair(row)
            if zone in seen_zones or pair in selected:
                continue
            selected.append(pair)
            seen_zones.add(zone)

    # Stage 3: still short (few distinct zones) -> fill by raw coverage order.
    if len(selected) < n:
        for _, row in cat.iterrows():
            if len(selected) >= n:
                break
            pair = _pair(row)
            if pair not in selected:
                selected.append(pair)

    return selected[:n]

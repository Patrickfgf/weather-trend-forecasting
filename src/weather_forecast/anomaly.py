"""Advanced EDA: anomaly detection and analysis on the cleaned weather frame.

Core principle: an anomaly is "weird FOR THIS PLACE AT THIS TIME OF YEAR", never
weird relative to the global pool. A 2 C day in Reykjavik in January is a *normal*
Reykjavik winter; flagging it against the planet's distribution would just rediscover
geography. So every detector first DESEASONALIZES per ``(city, month)`` — it subtracts
the city-month median before scoring — so what remains is the local, within-season
surprise.

Three complementary signals, deliberately diverse so agreement is meaningful:
- Univariate robust z-score (Iglewicz-Hoaglin modified z, MAD-based) + an
  IQR-within-month fence — fat-tail-robust, interpretable in degrees C.
- STL residuals on a monthly-resampled series (seasonal-trend decomposition) — only
  when there are enough points; guarded and optional (statsmodels may be absent).
- Multivariate IsolationForest + LocalOutlierFactor on season/geo-aware features —
  catches joint weirdness (e.g. high temp WITH high pressure AND high humidity) that
  no single column flags.

Honesty caveat baked in: this dataset is a rolling DAILY-SNAPSHOT history (sub-annual
to ~1 year), NOT a multi-decade climate record. "Seasonal" here means within-sample
month effects, not decadal climatology. A single date where an outsized share of all
cities go anomalous at once is far more likely an INGESTION ARTIFACT (a bad batch load)
than synchronized real weather — :func:`anomalies_by_time` flags exactly that.

All functions are pure: DataFrame/Series in -> DataFrame/Series/dict out. No global
state, no printing, no in-place mutation. The only I/O is the explicit, idempotent
:func:`write_anomalies`. Stochastic estimators are seeded with ``config.SEED``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Guard STL: statsmodels is part of the core stack, but importing it can still fail on
# a fresh interpreter without a wheel. We degrade to the MAD path rather than crash.
try:  # pragma: no cover - import-guard branch
    from statsmodels.tsa.seasonal import STL  # type: ignore

    _HAS_STL = True
except Exception:  # noqa: BLE001 - any import/build error means "no STL"
    STL = None  # type: ignore
    _HAS_STL = False

# 0.6745 is the 0.75 quantile of the standard normal: it rescales MAD to be a
# consistent estimator of sigma, so the modified z-score is comparable to a classic z.
_MAD_SCALE: float = 0.6745


# --------------------------------------------------------------------------- #
# Robust building blocks
# --------------------------------------------------------------------------- #
def modified_zscore(x: pd.Series) -> pd.Series:
    """Iglewicz-Hoaglin modified z-score: ``0.6745 * (x - median) / MAD``.

    Median/MAD instead of mean/std because weather is heavy-tailed — one freak day
    would inflate std and mask everything else. MAD == 0 (a constant series) -> 0,
    not inf, so a flat city never spuriously flags.
    """
    vals = x.astype("float64")
    median = vals.median()
    mad = (vals - median).abs().median()
    if not np.isfinite(mad) or mad == 0.0:
        return pd.Series(np.zeros(len(vals)), index=x.index, dtype="float64")
    return _MAD_SCALE * (vals - median) / mad


def deseasonalize_by_month(
    df: pd.DataFrame,
    value_col: str,
    keys: list[str] | None = None,
) -> pd.Series:
    """Subtract the per-``(keys, month)`` median from ``value_col`` (vectorized).

    This is THE key step: residual = value - typical-value-for-this-city-this-month.
    Using groupby-transform (broadcasts the group median back to every row) instead
    of a row-wise apply keeps it O(n) and aligned to the original index. Median (not
    mean) so the centering itself is robust to the very outliers we're hunting.
    """
    keys = list(config.KEY_COLS if keys is None else keys)
    month = df[config.TIME_COL].dt.month.rename("__month")
    grouper = [df[k] for k in keys] + [month]
    city_month_median = df[value_col].astype("float64").groupby(grouper).transform("median")
    return (df[value_col].astype("float64") - city_month_median).rename(f"{value_col}_deseasonalized")


def stl_residuals(
    series: pd.Series,
    period: int = config.STL_PERIOD,
    robust: bool = True,
) -> pd.Series:
    """STL residual component of a (monthly-resampled) series; NaN-filled, robust fit.

    Caller is responsible for guarding ``len(series) >= config.STL_MIN_POINTS`` (need
    ~2 seasonal cycles) and for resampling to a regular monthly cadence — STL assumes a
    fixed period. ``robust=True`` down-weights outliers in the seasonal/trend fit so the
    residual isolates the anomaly instead of bending the fit toward it. Returns a zero
    series if STL is unavailable so downstream code never branches on the import.
    """
    if not _HAS_STL:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype="float64")
    clean = series.astype("float64").interpolate(limit_direction="both")
    result = STL(clean, period=period, robust=robust).fit()
    return pd.Series(result.resid, index=series.index, dtype="float64").rename("stl_resid")


# --------------------------------------------------------------------------- #
# Univariate detector
# --------------------------------------------------------------------------- #
def detect_univariate(
    df: pd.DataFrame,
    value_col: str = config.TARGET,
    keys: list[str] | None = None,
    z_thresh: float = config.MAD_Z_THRESH,
) -> pd.DataFrame:
    """Month-deseasonalized MAD z-score + IQR-within-month fence for ``value_col``.

    Two robust univariate views that rarely fail together:
    - ``mad_zscore``: modified z on the *deseasonalized* residual, computed within each
      city (so the spread is the city's own variability, not the planet's).
    - ``iqr_flag``: Tukey 1.5*IQR fence computed within ``(city, month)`` — a
      distribution-free second opinion that doesn't assume symmetry.

    Columns returned (index-aligned to ``df``): mad_zscore, mad_flag, iqr_flag.
    """
    keys = list(config.KEY_COLS if keys is None else keys)
    resid = deseasonalize_by_month(df, value_col, keys=keys)

    # MAD z-score of the residual, computed per city (transform broadcasts back).
    mad_z = resid.groupby([df[k] for k in keys], group_keys=False).transform(
        lambda s: modified_zscore(s)
    )
    mad_flag = mad_z.abs() > z_thresh

    # Tukey IQR fence within (city, month) on the raw value — second, assumption-light opinion.
    month = df[config.TIME_COL].dt.month.rename("__month")
    grouper = [df[k] for k in keys] + [month]
    vals = df[value_col].astype("float64")
    q1 = vals.groupby(grouper).transform(lambda s: s.quantile(0.25))
    q3 = vals.groupby(grouper).transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    iqr_flag = (vals < (q1 - 1.5 * iqr)) | (vals > (q3 + 1.5 * iqr))

    return pd.DataFrame(
        {
            "mad_zscore": mad_z.astype("float64"),
            "mad_flag": mad_flag.fillna(False).astype(bool),
            "iqr_flag": iqr_flag.fillna(False).astype(bool),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Multivariate detector
# --------------------------------------------------------------------------- #
def _default_feature_cols(df: pd.DataFrame) -> list[str]:
    """Season/geo-aware feature set: present metric numerics + cyclical month + latitude.

    month_sin/cos encode seasonality so the model learns "anomalous given time of year";
    latitude proxies climate zone so polar and tropical normals aren't conflated.
    """
    cols = [c for c in config.METRIC_NUMERIC if c in df.columns]
    for extra in ("month_sin", "month_cos", "latitude"):
        if extra in df.columns:
            cols.append(extra)
    return cols


def detect_multivariate(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = config.CONTAMINATION,
    random_state: int = config.SEED,
) -> pd.DataFrame:
    """StandardScaler -> IsolationForest + LocalOutlierFactor on metric features.

    Two structurally different multivariate detectors so their *agreement* is a strong
    signal: IsolationForest isolates points via random splits (global, tree-based); LOF
    compares local density to neighbours (local, distance-based). Median-impute NaN
    BEFORE scaling so missingness never masquerades as an outlier, and standardize so no
    single high-variance column (e.g. pressure_mb) dominates the distance/splits.

    Columns returned (index-aligned to ``df``): iforest_score, iforest_flag, lof_flag.
    ``iforest_score`` is the decision_function: lower = more anomalous.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    cols = list(_default_feature_cols(df) if feature_cols is None else feature_cols)
    cols = [c for c in cols if c in df.columns]
    empty = pd.DataFrame(
        {
            "iforest_score": np.zeros(len(df), dtype="float64"),
            "iforest_flag": np.zeros(len(df), dtype=bool),
            "lof_flag": np.zeros(len(df), dtype=bool),
        },
        index=df.index,
    )
    # Need at least 2 features and enough rows for neighbour-based LOF to be meaningful.
    if len(cols) < 2 or len(df) < 5:
        return empty

    raw = df[cols].astype("float64")
    medians = raw.median(numeric_only=True)
    feats = raw.fillna(medians).fillna(0.0)  # second fillna: a fully-NaN column -> 0
    scaled = StandardScaler().fit_transform(feats.to_numpy())

    iforest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    iforest_pred = iforest.fit_predict(scaled)  # -1 anomaly, 1 normal
    iforest_score = iforest.decision_function(scaled)

    # LOF n_neighbors must stay below n_samples; clamp for tiny frames.
    n_neighbors = min(20, max(2, len(df) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_pred = lof.fit_predict(scaled)

    return pd.DataFrame(
        {
            "iforest_score": iforest_score.astype("float64"),
            "iforest_flag": (iforest_pred == -1),
            "lof_flag": (lof_pred == -1),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Flag combination
# --------------------------------------------------------------------------- #
def combine_flags(uni: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    """Union the per-method flags into a consensus view (method names, count, confidence).

    Confidence philosophy: a single detector firing is a *candidate*; agreement across
    methodologically-different detectors is what we trust. ``is_high_confidence`` =
    IsolationForest AND LOF agree (two independent multivariate views), OR any two
    methods agree at all (``n_methods >= 2``). This deliberately resists the single
    noisy detector that would otherwise dominate an alert list.

    Columns returned: mad_flag, iqr_flag, iforest_flag, lof_flag, anomaly_methods,
    n_methods, is_high_confidence.
    """
    flags = pd.DataFrame(index=uni.index)
    flags["mad_flag"] = uni["mad_flag"].astype(bool)
    flags["iqr_flag"] = uni["iqr_flag"].astype(bool)
    flags["iforest_flag"] = multi["iforest_flag"].astype(bool)
    flags["lof_flag"] = multi["lof_flag"].astype(bool)

    method_map = {
        "mad": flags["mad_flag"],
        "iqr": flags["iqr_flag"],
        "iforest": flags["iforest_flag"],
        "lof": flags["lof_flag"],
    }
    # Vectorized "method1|method2" string: join the names of the True flags per row.
    methods = pd.Series([[] for _ in range(len(flags))], index=flags.index, dtype="object")
    for name, col in method_map.items():
        methods = methods.combine(col, lambda lst, hit, n=name: lst + [n] if hit else lst)
    flags["anomaly_methods"] = methods.map("|".join)
    flags["n_methods"] = methods.map(len).astype("int16")
    flags["is_high_confidence"] = (
        (flags["iforest_flag"] & flags["lof_flag"]) | (flags["n_methods"] >= 2)
    ).astype(bool)
    return flags


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors and return a tidy table — one row per FLAGGED observation.

    Orchestrates: univariate (MAD + IQR) + multivariate (IForest + LOF) + flag
    consensus, then keeps only rows flagged by >= 1 method and attaches the human
    context needed to triage each one (the city-month median it deviated from, the
    signed deviation in C, the reported condition, hemisphere). Returns an EMPTY tidy
    frame (correct columns) if nothing fires, so callers never special-case None.
    """
    target = config.TARGET
    keys = list(config.KEY_COLS)

    uni = detect_univariate(df, value_col=target, keys=keys)
    multi = detect_multivariate(df)
    flags = combine_flags(uni, multi)

    month = df[config.TIME_COL].dt.month.astype("int16")
    grouper = [df[k] for k in keys] + [month.rename("__month")]
    city_month_median = df[target].astype("float64").groupby(grouper).transform("median")
    deviation = df[target].astype("float64") - city_month_median

    ctx_cols = {
        "country": df["country"] if "country" in df.columns else pd.Series("", index=df.index),
        "location_name": df["location_name"] if "location_name" in df.columns else pd.Series("", index=df.index),
        config.TIME_COL: df[config.TIME_COL],
        target: df[target].astype("float64"),
        "condition_text": df["condition_text"] if "condition_text" in df.columns else pd.Series(pd.NA, index=df.index),
        "month": month,
        "hemisphere": df["hemisphere"] if "hemisphere" in df.columns else pd.Series(pd.NA, index=df.index),
        "temp_city_month_median": city_month_median,
        "temp_deviation_c": deviation,
        "mad_zscore": uni["mad_zscore"].astype("float64"),
        "iforest_score": multi["iforest_score"].astype("float64"),
        "mad_flag": flags["mad_flag"],
        "iqr_flag": flags["iqr_flag"],
        "iforest_flag": flags["iforest_flag"],
        "lof_flag": flags["lof_flag"],
        "anomaly_methods": flags["anomaly_methods"],
        "n_methods": flags["n_methods"],
        "is_high_confidence": flags["is_high_confidence"],
    }
    table = pd.DataFrame(ctx_cols, index=df.index)
    flagged = table.loc[flags["n_methods"] > 0].copy()
    # category dtypes don't round-trip cleanly to parquet via this path — cast to str.
    for c in ("country", "location_name", "condition_text", "hemisphere"):
        if c in flagged.columns:
            flagged[c] = flagged[c].astype("string")
    return flagged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Post-hoc analysis
# --------------------------------------------------------------------------- #
def condition_lift(
    anom_df: pd.DataFrame,
    base_df: pd.DataFrame,
    condition_col: str = "condition_text",
) -> pd.DataFrame:
    """Lift = (anomaly rate within a condition) / (that condition's baseline share).

    Answers "which reported conditions are OVER-represented among anomalies?". Lift > 1
    means a condition is more common among flagged rows than in the data at large — e.g.
    if "Blizzard" is 0.2% of all rows but 5% of anomalies, lift = 25. Lift is the right
    framing (not raw counts) because a common condition will trivially top a raw tally.

    Columns returned: condition, n_anomalies, anomaly_share, baseline_share, lift —
    sorted by lift desc. Empty inputs -> empty frame (no division by zero).
    """
    cols = ["condition", "n_anomalies", "anomaly_share", "baseline_share", "lift"]
    if condition_col not in base_df.columns or len(base_df) == 0 or len(anom_df) == 0:
        return pd.DataFrame(columns=cols)

    base_counts = base_df[condition_col].astype("string").value_counts(dropna=True)
    base_share = (base_counts / base_counts.sum()).rename("baseline_share")

    anom_counts = anom_df[condition_col].astype("string").value_counts(dropna=True)
    anom_share = (anom_counts / anom_counts.sum()).rename("anomaly_share")

    out = (
        pd.concat([anom_counts.rename("n_anomalies"), anom_share, base_share], axis=1)
        .dropna(subset=["n_anomalies"])
        .fillna({"baseline_share": 0.0, "anomaly_share": 0.0})
    )
    # Guard divide-by-zero: a condition seen only among anomalies gets NaN lift -> inf.
    out["lift"] = np.where(
        out["baseline_share"] > 0, out["anomaly_share"] / out["baseline_share"], np.inf
    )
    out = out.reset_index(names="condition")
    out["n_anomalies"] = out["n_anomalies"].astype("int64")
    return out[cols].sort_values("lift", ascending=False, ignore_index=True)


def anomalies_by_time(
    anom_df: pd.DataFrame,
    time_col: str = config.TIME_COL,
) -> pd.DataFrame:
    """Daily anomaly counts, FLAGGING dates that look like ingestion artifacts.

    On a daily-snapshot dataset, a single date where an outsized share of ALL flagged
    rows pile up (or where most cities go anomalous simultaneously) is almost never real
    synchronized weather — it's a suspect batch load / parsing glitch. We flag a date as
    a suspected artifact when its share of total anomalies is both large in absolute
    terms (> 30%) and a strong outlier vs other days (> mean + 3*std of daily share).

    Columns returned: date, n_anomalies, share_of_total, suspected_ingestion_artifact —
    sorted by date. Empty input -> empty frame.
    """
    cols = ["date", "n_anomalies", "share_of_total", "suspected_ingestion_artifact"]
    if len(anom_df) == 0 or time_col not in anom_df.columns:
        return pd.DataFrame(columns=cols)

    dates = pd.to_datetime(anom_df[time_col], utc=True).dt.normalize()
    counts = dates.value_counts().sort_index()
    total = int(counts.sum())
    share = counts / total

    # Outlier-on-a-share basis: large absolute share AND a statistical spike vs peers.
    mean_share = share.mean()
    std_share = share.std(ddof=0)
    spike_thresh = mean_share + 3.0 * (std_share if np.isfinite(std_share) else 0.0)
    suspected = (share > 0.30) & (share > spike_thresh)

    out = pd.DataFrame(
        {
            "date": counts.index,
            "n_anomalies": counts.to_numpy().astype("int64"),
            "share_of_total": share.to_numpy().astype("float64"),
            "suspected_ingestion_artifact": suspected.to_numpy().astype(bool),
        }
    )
    return out.sort_values("date", ignore_index=True)[cols]


# --------------------------------------------------------------------------- #
# Deterministic, idempotent persistence (the only I/O edge)
# --------------------------------------------------------------------------- #
def write_anomalies(anom_df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    """Write ``anomalies.parquet`` (+ ``anomalies_summary.parquet``); idempotent.

    Deterministic sort BEFORE write so re-running the pipeline yields a byte-stable file
    (reproducible diffs). Returns the path to the main parquet. The summary captures the
    counts a report needs without re-deriving them. Never reads/writes the raw data file.
    """
    out_dir = config.PROCESSED_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sort_keys = [c for c in ("country", "location_name", config.TIME_COL) if c in anom_df.columns]
    ordered = anom_df.sort_values(sort_keys, ignore_index=True) if sort_keys else anom_df.copy()

    main_path = out_dir / "anomalies.parquet"
    ordered.to_parquet(main_path, index=False)

    n_total = int(len(ordered))
    n_high = int(ordered["is_high_confidence"].sum()) if "is_high_confidence" in ordered.columns else 0
    n_cities = int(ordered["location_name"].nunique()) if "location_name" in ordered.columns else 0
    summary = pd.DataFrame(
        {
            "metric": ["n_anomalies", "n_high_confidence", "n_cities_with_anomalies"],
            "value": [n_total, n_high, n_cities],
        }
    )
    summary.to_parquet(out_dir / "anomalies_summary.parquet", index=False)
    return main_path

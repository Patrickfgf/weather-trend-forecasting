"""Deterministic synthetic generator faithful to the Global Weather Repository schema.

Purpose: let the FULL pipeline (ingest -> clean -> features -> models -> analysis)
run end-to-end and let pytest exercise every transform WITHOUT the ~100MB Kaggle CSV.
The real file, when dropped in ``data/raw/``, transparently replaces this (see
``weather_forecast.ingest.load_weather``).

The generator emits the RAW column names exactly as Kaggle ships them (mixed case,
dots, hyphens, both metric and imperial twins, ``last_updated`` as a string) so the
ingestion/normalization edge is genuinely tested rather than bypassed. It injects a
controlled amount of realism: a slight warming trend, weekly + annual seasonality
(hemisphere-aware), autocorrelated noise, zero-inflated precipitation, an
air-quality signal that worsens in low wind, plus a few missing values, duplicate
``(city, timestamp)`` rows and out-of-physical-range sensor glitches so the cleaning
layer has something to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import config

# (country, location_name, latitude, longitude, timezone) — climate/hemisphere diverse.
_CITIES: list[tuple[str, str, float, float, str]] = [
    ("United Kingdom", "London", 51.52, -0.11, "Europe/London"),
    ("Russia", "Moscow", 55.75, 37.62, "Europe/Moscow"),
    ("Egypt", "Cairo", 30.06, 31.25, "Africa/Cairo"),
    ("India", "New Delhi", 28.60, 77.20, "Asia/Kolkata"),
    ("Kenya", "Nairobi", -1.28, 36.82, "Africa/Nairobi"),
    ("Japan", "Tokyo", 35.69, 139.69, "Asia/Tokyo"),
    ("Australia", "Canberra", -35.28, 149.13, "Australia/Sydney"),
    ("Brazil", "Brasilia", -15.78, -47.93, "America/Sao_Paulo"),
    ("United States of America", "Washington", 38.90, -77.04, "America/New_York"),
    ("Iceland", "Reykjavik", 64.15, -21.94, "Atlantic/Reykjavik"),
    ("Argentina", "Buenos Aires", -34.61, -58.38, "America/Argentina/Buenos_Aires"),
    ("Indonesia", "Jakarta", -6.21, 106.85, "Asia/Jakarta"),
]

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

_MOON_PHASES = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
                "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]


def _ar1(rng: np.random.Generator, n: int, phi: float, scale: float) -> np.ndarray:
    """Autocorrelated noise via an AR(1) filter — realistic day-to-day persistence."""
    white = rng.normal(0.0, scale, n)
    return lfilter([1.0], [1.0, -phi], white)


def generate_synthetic_weather(
    n_days: int = 220,
    start: str = "2024-08-01",
    seed: int = config.SEED,
    *,
    inject_quality_issues: bool = True,
) -> pd.DataFrame:
    """Return a synthetic GWR-shaped DataFrame with RAW (un-normalized) column names.

    Grain: one row per (city, day). ``inject_quality_issues`` adds NaNs, duplicate
    keys and out-of-range glitches so the cleaning layer is exercised.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy().astype(float)
    t = np.arange(n_days, dtype=float)

    parts: list[pd.DataFrame] = []
    for ci, (country, city, lat, lon, tz) in enumerate(_CITIES):
        crng = np.random.default_rng(seed + ci + 1)
        # --- temperature: latitude baseline + hemisphere-aware seasonality + trend + AR(1) ---
        annual_mean = 30.0 - 0.45 * abs(lat)
        amp = 3.0 + 0.35 * abs(lat)
        phase = 200.0 if lat >= 0 else 200.0 + 182.0   # N peak ~mid-Jul, S shifted 6 months
        seasonal = amp * np.sin(2.0 * np.pi * (doy - phase) / 365.25)
        warming = 0.004 * t
        temp_c = annual_mean + seasonal + warming + _ar1(crng, n_days, 0.6, 2.0)

        # --- other weather, physically loosely coupled ---
        humidity = np.clip(62 - 0.4 * (temp_c - annual_mean) + crng.normal(0, 8, n_days), 8, 100)
        pressure_mb = 1013 + crng.normal(0, 6, n_days)
        wind_kph = np.clip(np.abs(_ar1(crng, n_days, 0.5, 6.0)) + 6, 0, 90)
        precip_mm = np.where(crng.random(n_days) < 0.72, 0.0,
                             crng.exponential(4.0, n_days))            # zero-inflated
        cloud = np.clip(25 + 5 * np.log1p(precip_mm) + crng.normal(0, 18, n_days), 0, 100)
        uv_index = np.clip((1 - abs(lat) / 90) * 9 + 2 * np.sin(2 * np.pi * (doy - phase) / 365.25)
                           + crng.normal(0, 1, n_days), 0, 13).round(1)
        visibility_km = np.clip(10 - 0.05 * precip_mm - 0.03 * cloud + crng.normal(0, 1, n_days), 0.5, 10)
        gust_kph = np.clip(wind_kph * 1.3 + crng.normal(0, 3, n_days), 0, 130)
        wind_degree = crng.integers(0, 360, n_days)
        feels_like_c = temp_c - 0.12 * wind_kph * (temp_c < 12) + 0.10 * (humidity - 60) * (temp_c > 24)

        # --- air quality: worse in low wind / high pressure; city base pollution ---
        base_pm = 8 + 22 * (1 / (1 + np.exp(-(ci - 4))))   # some cities dirtier
        pm2_5 = np.clip(base_pm + 60 / (wind_kph + 2) - 4 + crng.normal(0, 4, n_days), 1, 350)
        pm10 = np.clip(pm2_5 * 1.6 + crng.normal(0, 5, n_days), 1, 500)
        co = np.clip(200 + pm2_5 * 6 + crng.normal(0, 50, n_days), 50, 4000)
        ozone = np.clip(40 + 0.5 * uv_index * 8 + crng.normal(0, 10, n_days), 5, 250)
        no2 = np.clip(5 + pm2_5 * 0.4 + crng.normal(0, 3, n_days), 0.5, 200)
        so2 = np.clip(2 + pm2_5 * 0.15 + crng.normal(0, 2, n_days), 0.1, 120)
        epa = np.digitize(pm2_5, [12, 35.5, 55.5, 150.5, 250.5]) + 1     # 1..6
        defra = np.clip(np.digitize(pm2_5, np.linspace(5, 100, 9)) + 1, 1, 10)

        cond = np.select(
            [precip_mm > 6, precip_mm > 0.4, cloud > 70, cloud > 30],
            ["Heavy rain", "Light rain", "Cloudy", "Partly cloudy"],
            default="Sunny",
        )
        wdir = np.array(_COMPASS)[((wind_degree + 11.25) // 22.5).astype(int) % 16]

        hours = crng.integers(0, 24, n_days)
        minutes = crng.choice([0, 15, 30, 45], n_days)
        ts = pd.to_datetime(dates) + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

        part = pd.DataFrame({
            "country": country, "location_name": city, "latitude": lat, "longitude": lon,
            "timezone": tz,
            "last_updated_epoch": (ts.astype("int64") // 10**9),
            "last_updated": ts.strftime("%Y-%m-%d %H:%M"),
            "temperature_celsius": temp_c.round(1),
            "temperature_fahrenheit": (temp_c * 9 / 5 + 32).round(1),
            "condition_text": cond,
            "wind_mph": (wind_kph / 1.609).round(1), "wind_kph": wind_kph.round(1),
            "wind_degree": wind_degree, "wind_direction": wdir,
            "pressure_mb": pressure_mb.round(1), "pressure_in": (pressure_mb * 0.02953).round(2),
            "precip_mm": precip_mm.round(2), "precip_in": (precip_mm * 0.03937).round(3),
            "humidity": humidity.round(0), "cloud": cloud.round(0),
            "feels_like_celsius": feels_like_c.round(1),
            "feels_like_fahrenheit": (feels_like_c * 9 / 5 + 32).round(1),
            "visibility_km": visibility_km.round(1), "visibility_miles": (visibility_km / 1.609).round(1),
            "uv_index": uv_index, "gust_mph": (gust_kph / 1.609).round(1), "gust_kph": gust_kph.round(1),
            "air_quality_Carbon_Monoxide": co.round(1), "air_quality_Ozone": ozone.round(1),
            "air_quality_Nitrogen_dioxide": no2.round(1), "air_quality_Sulphur_dioxide": so2.round(1),
            "air_quality_PM2.5": pm2_5.round(1), "air_quality_PM10": pm10.round(1),
            "air_quality_us-epa-index": epa, "air_quality_gb-defra-index": defra,
            "sunrise": "06:00 AM", "sunset": "06:00 PM",
            "moonrise": "07:00 PM", "moonset": "06:30 AM",
            "moon_phase": crng.choice(_MOON_PHASES, n_days),
            "moon_illumination": crng.integers(0, 100, n_days),
        })
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)

    if inject_quality_issues:
        df = _inject_quality_issues(df, rng)
    return df


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add NaNs, duplicate-key rows, and physical-range glitches (returns a new frame)."""
    df = df.copy()
    n = len(df)
    # ~3% missing in a few sensor columns (missing-not-at-random for air quality)
    for col, frac in [("humidity", 0.02), ("pressure_mb", 0.02),
                      ("air_quality_PM2.5", 0.06), ("visibility_km", 0.03)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan
    # a handful of out-of-physical-range sensor glitches
    glitch_idx = rng.choice(n, size=4, replace=False)
    df.loc[glitch_idx[:2], "temperature_celsius"] = 150.0     # impossible temperature
    df.loc[glitch_idx[2:], "humidity"] = 250.0                # impossible humidity
    # a few exact duplicate (city, timestamp) snapshots (re-ingest artifacts)
    dups = df.sample(n=5, random_state=0)
    return pd.concat([df, dups], ignore_index=True)


def write_synthetic_csv(path=None, **kwargs) -> "object":
    """Write a synthetic raw CSV to ``path`` (default: data/raw candidate). Returns the path."""
    from pathlib import Path
    if path is None:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RAW_DIR / "synthetic_global_weather.csv"
    path = Path(path)
    generate_synthetic_weather(**kwargs).to_csv(path, index=False)
    return path

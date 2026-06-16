"""Single entrypoint for the Weather Trend Forecasting pipeline.

Produces every artifact under data/ and reports/:
  data/interim/clean.parquet, data/processed/{features,anomalies,predictions}.parquet,
  reports/metrics/*.csv|json, reports/figures/*.png|html

Usage:
    python run_pipeline.py
It uses the real Kaggle CSV in data/raw/ if present, else a deterministic synthetic
fallback so the run completes with no manual download.
"""

from __future__ import annotations

from weather_forecast import pipeline


def main() -> int:
    pipeline.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

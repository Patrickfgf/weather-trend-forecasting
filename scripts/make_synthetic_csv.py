"""CLI: write a synthetic Global-Weather-Repository-shaped CSV into data/raw/.

Lets the pipeline / notebook / CI run end-to-end without the real Kaggle download.
The real file, when present in data/raw/, always takes precedence (see
``weather_forecast.ingest.find_raw_csv``).

Usage:
    python scripts/make_synthetic_csv.py --n-days 220
"""

from __future__ import annotations

import argparse

from weather_forecast import synthetic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=220, help="days of history per city")
    parser.add_argument("--start", type=str, default="2024-08-01", help="first date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default=None, help="output path (default: data/raw/synthetic_global_weather.csv)")
    args = parser.parse_args()

    path = synthetic.write_synthetic_csv(args.out, n_days=args.n_days, start=args.start)
    print(f"wrote synthetic CSV: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

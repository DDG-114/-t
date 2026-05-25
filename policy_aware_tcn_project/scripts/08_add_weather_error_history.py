#!/usr/bin/env python3
"""Add lagged weather forecast-error features to an existing feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.weather import (
    add_weather_error_history_features,
    build_weather_actual_table,
    weather_date_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input-features", required=True)
    parser.add_argument("--output-features", required=True)
    parser.add_argument("--forecast-lead-day", type=int, default=1)
    parser.add_argument("--windows-days", type=int, nargs="+", default=[1, 3, 7])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    input_path = Path(args.input_features)
    output_path = Path(args.output_features)
    features = pd.read_csv(input_path, parse_dates=[config["columns"]["datetime"]])
    start_date, end_date = weather_date_bounds(
        features,
        datetime_col=config["columns"]["datetime"],
    )
    actual_weather = build_weather_actual_table(
        config,
        start_date=start_date,
        end_date=end_date,
    )
    enhanced = add_weather_error_history_features(
        features,
        actual_weather,
        datetime_col=config["columns"]["datetime"],
        forecast_lead_day=int(args.forecast_lead_day),
        windows_days=args.windows_days,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Input feature table shape: {features.shape}")
    print(f"Actual weather table shape: {actual_weather.shape}")
    print(f"Enhanced feature table shape: {enhanced.shape}")
    print(f"Saved weather-error features to {output_path}")


if __name__ == "__main__":
    main()

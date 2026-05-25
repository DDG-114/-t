#!/usr/bin/env python3
"""Build a weather-augmented feature table from an existing feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.weather import build_weather_table, merge_weather_features, weather_date_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--input-features", default=None)
    parser.add_argument("--output-features", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    input_path = Path(args.input_features or config["paths"]["processed_features"])
    output_path = Path(args.output_features or input_path.with_name("features_weather.csv"))
    features = pd.read_csv(input_path, parse_dates=["date"])
    start_date, end_date = weather_date_bounds(
        features,
        datetime_col=config["columns"]["datetime"],
    )
    weather = build_weather_table(config, start_date=start_date, end_date=end_date)
    merged = merge_weather_features(
        features,
        weather,
        datetime_col=config["columns"]["datetime"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Weather table shape: {weather.shape}")
    print(f"Feature table shape: {merged.shape}")
    print(f"Saved weather-augmented features to {output_path}")


if __name__ == "__main__":
    main()

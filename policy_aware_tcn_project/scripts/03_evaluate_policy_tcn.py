#!/usr/bin/env python3
"""Evaluate the standalone policy-aware TCN and produce a 24h forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.deep_train import evaluate_trained_deep_model, predict_forecast_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    evaluation = evaluate_trained_deep_model(features, config, prefix="policy_aware_tcn")

    target_day = pd.Timestamp(config.get("forecast", {}).get("target_date", features["date"].max()))
    forecast_summary = predict_forecast_day(
        features,
        config,
        target_day=target_day,
        prefix=config.get("forecast", {}).get("output_prefix", "policy_aware_tcn_future_24h"),
    )

    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(forecast_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

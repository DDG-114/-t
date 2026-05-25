#!/usr/bin/env python
"""Evaluate simple baseline forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf.baselines import baseline_daily_metrics, evaluate_baselines, make_baseline_predictions
from epf.config import ensure_dirs, load_config
from epf.metrics import monthly_metrics
from epf.split import split_by_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    _, _, test_df = split_by_date(features, config)

    predictions = make_baseline_predictions(test_df, config)
    summaries = evaluate_baselines(predictions, config)

    pred_dir = Path(config["paths"]["prediction_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    predictions.to_csv(pred_dir / "baseline_predictions.csv", index=False, encoding="utf-8-sig")

    with (report_dir / "baseline_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

    for baseline_col in [col for col in predictions.columns if col.startswith("baseline_")]:
        daily = baseline_daily_metrics(predictions, config, baseline_col)
        daily.to_csv(report_dir / f"{baseline_col}_daily_metrics.csv", index=False, encoding="utf-8-sig")
        monthly = monthly_metrics(
            daily,
            threshold=config["metrics"].get("daily_accuracy_threshold", 0.85),
        )
        monthly.to_csv(
            report_dir / f"{baseline_col}_monthly_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

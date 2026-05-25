#!/usr/bin/env python
"""Evaluate trained slot-wise LightGBM models on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf.config import ensure_dirs, load_config
from epf.data import find_forecast_frame
from epf.evaluate import evaluate_predictions, save_evaluation
from epf.split import split_by_date
from epf.train_slot_lgbm import load_slot_models, predict_with_slot_models


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

    models, feature_cols = load_slot_models(config)
    predictions = predict_with_slot_models(test_df, models, feature_cols, config)
    evaluation = evaluate_predictions(predictions, config)
    save_evaluation(predictions, evaluation, config, prefix="slot_lightgbm")

    forecast_df = find_forecast_frame(features, config)
    forecast_predictions = predict_with_slot_models(
        forecast_df,
        models,
        feature_cols,
        config,
    )
    forecast_prefix = config.get("forecast", {}).get("output_prefix", "future_24h")
    pred_dir = Path(config["paths"]["prediction_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    forecast_predictions.to_csv(
        pred_dir / f"{forecast_prefix}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    forecast_summary = {
        "target_date": str(pd.Timestamp(forecast_df["date"].iloc[0]).date()),
        "n_rows": int(forecast_predictions.shape[0]),
        "n_predicted": int(forecast_predictions["y_pred"].notna().sum()),
        "n_missing_target": int(forecast_predictions["y_true"].isna().sum()),
        "expected_slots": int(config["data"].get("expected_slots_per_day", 96)),
        "output_file": str(pred_dir / f"{forecast_prefix}_predictions.csv"),
    }
    with (report_dir / f"{forecast_prefix}_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(forecast_summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(forecast_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

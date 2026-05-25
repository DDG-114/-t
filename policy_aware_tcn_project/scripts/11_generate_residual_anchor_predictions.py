#!/usr/bin/env python3
"""Generate train/valid/test GBM-residual predictions for deep-model anchoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.residual_calibrator import load_residual_calibrator, predict_residual_calibrated
from epf_tcn.state_gbm import load_state_gbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/residual_calibrator_weather_error_q2.yaml")
    parser.add_argument("--prefix", default="gbm_residual_anchor")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    state_model = load_state_gbm(config)
    calibrator = load_residual_calibrator(config)

    summaries = {}
    pred_dir = Path(config["paths"]["prediction_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for split_name in args.splits:
        frame = predict_residual_calibrated(
            state_model,
            calibrator,
            features,
            config,
            split_name=split_name,
        )
        split_prefix = f"{args.prefix}_{split_name}"
        save_evaluation(frame, evaluate_predictions(frame, config), config, prefix=split_prefix)
        summaries[split_name] = {
            "rows": int(frame.shape[0]),
            "start": str(pd.to_datetime(frame["Date"]).min()),
            "end": str(pd.to_datetime(frame["Date"]).max()),
        }
        frames.append(frame)

    all_frame = pd.concat(frames, ignore_index=True).sort_values("Date")
    all_path = pred_dir / f"{args.prefix}_all_predictions.csv"
    all_frame.to_csv(all_path, index=False, encoding="utf-8-sig")
    summaries["all"] = {
        "rows": int(all_frame.shape[0]),
        "start": str(pd.to_datetime(all_frame["Date"]).min()),
        "end": str(pd.to_datetime(all_frame["Date"]).max()),
        "path": str(all_path),
    }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

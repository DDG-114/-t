#!/usr/bin/env python3
"""Train the standalone policy-aware TCN model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.deep_train import save_deep_model, train_deep_model
from epf_tcn.evaluate import evaluate_predictions, save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    result = train_deep_model(features, config)
    save_deep_model(result, config)

    evaluation = evaluate_predictions(result.valid_predictions, config)
    save_evaluation(result.valid_predictions, evaluation, config, prefix="policy_aware_tcn_valid")

    architecture = result.train_report["architecture"]
    print(f"Trained {architecture['class_name']}.")
    print(f"Trainable parameters: {architecture['trainable_parameters']}")
    print(f"Best validation loss: {result.train_report['training']['best_valid_loss']:.6f}")


if __name__ == "__main__":
    main()

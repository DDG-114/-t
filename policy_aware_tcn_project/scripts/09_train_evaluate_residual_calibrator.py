#!/usr/bin/env python3
"""Train and evaluate residual calibration on top of a state-aware GBM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.residual_calibrator import (
    evaluate_residual_calibrated,
    save_residual_calibrated_evaluation,
    save_residual_calibrator,
    train_residual_calibrator,
)
from epf_tcn.state_gbm import load_state_gbm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    state_model = load_state_gbm(config)
    calibrator = train_residual_calibrator(state_model, features, config)
    save_residual_calibrator(calibrator, config)

    test_result = evaluate_residual_calibrated(
        state_model,
        calibrator,
        features,
        config,
        split_name="test",
    )
    save_residual_calibrated_evaluation(
        test_result["predictions"],
        test_result["evaluation"],
        config,
        prefix="state_gbm_residual",
    )
    print(json.dumps(test_result["evaluation"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

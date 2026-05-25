#!/usr/bin/env python3
"""Train and evaluate the state-aware LightGBM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.state_gbm import (
    evaluate_state_gbm,
    save_state_gbm,
    save_state_gbm_evaluation,
    train_state_gbm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    model = train_state_gbm(features, config)
    save_state_gbm(model, config)

    test_result = evaluate_state_gbm(model, features, config, split_name="test")
    save_state_gbm_evaluation(
        test_result["predictions"],
        test_result["evaluation"],
        config,
        prefix="state_gbm",
    )

    print(json.dumps(test_result["evaluation"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

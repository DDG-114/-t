#!/usr/bin/env python
"""Train slot-wise LightGBM models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf.config import ensure_dirs, load_config
from epf.split import split_by_date
from epf.train_slot_lgbm import save_slot_models, train_slot_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    features = pd.read_csv(config["paths"]["processed_features"], parse_dates=["date"])
    train_df, valid_df, _ = split_by_date(features, config)

    target = config["columns"]["target"]
    train_df = train_df.dropna(subset=[target]).copy()
    valid_df = valid_df.dropna(subset=[target]).copy()

    models, feature_cols, report = train_slot_models(train_df, valid_df, config)
    save_slot_models(models, feature_cols, report, config)

    print(f"Trained {len(models)} slot models.")
    print(f"Feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()

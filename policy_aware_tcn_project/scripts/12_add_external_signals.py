#!/usr/bin/env python3
"""Add leakage-safe external scarcity signal features to a feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.features import add_external_signal_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/external_scarcity_template.yaml")
    parser.add_argument("--input-features", required=True)
    parser.add_argument("--output-features", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    input_path = Path(args.input_features)
    output_path = Path(args.output_features)
    datetime_col = config["columns"]["datetime"]
    features = pd.read_csv(input_path, parse_dates=[datetime_col, "date"])
    enhanced = add_external_signal_features(features, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced.to_csv(output_path, index=False, encoding="utf-8-sig")
    added_cols = [col for col in enhanced.columns if col not in features.columns]
    print(f"Input feature table shape: {features.shape}")
    print(f"Enhanced feature table shape: {enhanced.shape}")
    print(f"Added external signal columns: {len(added_cols)}")
    print(f"Saved external-signal features to {output_path}")


if __name__ == "__main__":
    main()

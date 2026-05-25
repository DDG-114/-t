#!/usr/bin/env python3
"""Build the feature table required by the standalone policy-aware TCN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.data import load_raw_data
from epf_tcn.features import build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    raw = load_raw_data(config)
    features = build_features(raw, config)
    output_path = Path(config["paths"]["processed_features"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Feature table shape: {features.shape}")
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()

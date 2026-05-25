#!/usr/bin/env python
"""Run the full first-version forecasting pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def run_step(script: str, config: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", config]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    steps = [
        "00_profile_data.py",
        "01_build_features.py",
        "02_train_baselines.py",
        "03_train_slot_lightgbm.py",
        "04_evaluate.py",
    ]
    for step in steps:
        run_step(step, args.config)


if __name__ == "__main__":
    main()

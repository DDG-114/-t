#!/usr/bin/env python3
"""Apply leakage-safe online residual correction to saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.online_calibrator import (
    evaluate_online_residual_predictions,
    save_online_residual_evaluation,
    write_online_residual_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prefix", default="online_residual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    predictions = pd.read_csv(args.predictions, parse_dates=["Date"])
    result = evaluate_online_residual_predictions(predictions, config)
    save_online_residual_evaluation(
        result["predictions"],
        result["evaluation"],
        config,
        prefix=args.prefix,
    )
    write_online_residual_report(
        result["evaluation"],
        result["online_config"],
        config,
        prefix=args.prefix,
    )
    print(json.dumps(result["evaluation"]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

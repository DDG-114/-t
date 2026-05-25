#!/usr/bin/env python3
"""Tune online residual calibrator parameters on a validation window."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import ensure_dirs, load_config
from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.online_calibrator import (
    OnlineResidualConfig,
    apply_online_residual_correction,
    write_online_residual_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/online_residual_best.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prefix", default="online_residual_tuned")
    parser.add_argument("--valid-start", default="2025-10-01")
    parser.add_argument("--valid-end", default="2025-11-30")
    parser.add_argument("--test-start", default="2025-12-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--history-start", default="2025-09-01")
    return parser.parse_args()


def _date_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    dates = pd.to_datetime(frame["date"])
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _candidate_configs(base: OnlineResidualConfig) -> Iterable[OnlineResidualConfig]:
    groups = ["global", "pred_bin", "hour", "state", "spike_state"]
    min_history_days = [3, 5, 7, 10]
    window_days = [3, 5, 7, 10, 14]
    shrink_values = [0.25, 0.5, 0.75]
    clip_values = [80.0, 120.0, 160.0, 240.0]
    min_predictions = [40.0, 60.0, 100.0]
    max_floor_probabilities = [0.4, 0.6, 0.8]
    for (
        group,
        min_history,
        window,
        shrink,
        clip,
        min_prediction,
        max_floor_probability,
    ) in itertools.product(
        groups,
        min_history_days,
        window_days,
        shrink_values,
        clip_values,
        min_predictions,
        max_floor_probabilities,
    ):
        yield replace(
            base,
            group=group,
            min_history_days=min_history,
            window_days=window,
            shrink=shrink,
            clip_value=clip,
            min_prediction=min_prediction,
            max_floor_probability=max_floor_probability,
            spike_enabled=False,
        )


def _score(
    corrected: pd.DataFrame,
    config: dict,
    start: str,
    end: str,
) -> dict:
    subset = corrected[_date_mask(corrected, start, end)].copy()
    return evaluate_predictions(subset, config)["summary"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    predictions = pd.read_csv(args.predictions, parse_dates=["Date", "date"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions[predictions["date"] >= pd.Timestamp(args.history_start)].copy()
    base_online = OnlineResidualConfig.from_config(config)

    best: tuple[float, int, OnlineResidualConfig, dict] | None = None
    tried = 0
    for candidate in _candidate_configs(base_online):
        corrected = apply_online_residual_correction(predictions, candidate)
        summary = _score(corrected, config, args.valid_start, args.valid_end)
        valid_accuracy = float(summary["overall"]["accuracy"])
        pass_days = int(summary["pass_days"])
        tried += 1
        key = (valid_accuracy, pass_days)
        if best is None or key > (best[0], best[1]):
            best = (valid_accuracy, pass_days, candidate, summary)
        if tried >= int(args.max_candidates):
            break

    if best is None:
        raise RuntimeError("No online residual candidates were evaluated.")

    _, _, best_config, valid_summary = best
    corrected = apply_online_residual_correction(predictions, best_config)
    test_mask = _date_mask(corrected, args.test_start, args.test_end)
    test_predictions = corrected[test_mask].copy()
    test_evaluation = evaluate_predictions(test_predictions, config)
    save_evaluation(test_predictions, test_evaluation, config, prefix=args.prefix)
    write_online_residual_report(
        test_evaluation,
        best_config,
        config,
        prefix=args.prefix,
    )

    report_dir = Path(config["paths"]["report_dir"])
    report = {
        "tried_candidates": tried,
        "valid_window": {"start": args.valid_start, "end": args.valid_end},
        "test_window": {"start": args.test_start, "end": args.test_end},
        "selected_online_config": best_config.to_dict(),
        "selected_valid_summary": valid_summary,
        "test_summary": test_evaluation["summary"],
    }
    with (report_dir / f"{args.prefix}_tuning_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose whether feature columns can rank high-price regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/residual_calibrator_weather_error_q2.yaml")
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--target-threshold", type=float, default=500.0)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-09-30")
    parser.add_argument("--valid-start", default="2025-10-01")
    parser.add_argument("--valid-end", default="2025-11-30")
    parser.add_argument("--test-start", default="2025-12-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--top-k", type=int, default=120)
    return parser.parse_args()


def _select_columns(columns: list[str]) -> list[str]:
    base = {"Date", "date", "slot", "Price"}
    patterns = [
        "净负荷",
        "供需裕度",
        "新能源",
        "负荷",
        "发电",
        "竞价空间",
        "抽蓄",
        "联络线",
        "price_lag",
        "price_roll",
        "prevday",
        "floor_ratio",
        "weather_",
        "weather_error_",
        "scarcity_",
        "slot_sin",
        "slot_cos",
        "month_sin",
        "month_cos",
        "day_of_week",
        "month",
        "is_weekend",
        "is_cn",
    ]
    selected = [
        col
        for col in columns
        if col in base or any(pattern in col for pattern in patterns)
    ]
    for leakage_col in ["is_floor_price", "floor_run_slots"]:
        if leakage_col in selected:
            selected.remove(leakage_col)
    return selected


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, price_floor: float) -> float:
    denom = np.maximum(np.abs(y_true), float(price_floor))
    return float(1.0 - np.mean(np.abs(y_pred - y_true) / denom))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))

    feature_path = Path(args.features)
    prediction_path = Path(args.predictions)
    header = pd.read_csv(feature_path, nrows=1)
    usecols = _select_columns(list(header.columns))
    features = pd.read_csv(feature_path, usecols=usecols, parse_dates=["Date", "date"])
    predictions = pd.read_csv(
        prediction_path,
        usecols=[
            "Date",
            "base_pred",
            "state_pred",
            "residual_pred",
            "floor_probability",
            "high_probability",
            "cap_probability",
            "y_pred",
        ],
        parse_dates=["Date"],
    )
    data = features.merge(predictions, on="Date", how="inner")
    data = data[data["date"] >= args.train_start].copy()

    y = data["Price"].to_numpy(dtype=float)
    target = (y >= float(args.target_threshold)).astype(int)
    train = (data["date"] >= args.train_start) & (data["date"] <= args.train_end)
    valid = (data["date"] >= args.valid_start) & (data["date"] <= args.valid_end)
    test = (data["date"] >= args.test_start) & (data["date"] <= args.test_end)
    feature_cols = [
        col
        for col in data.columns
        if col not in {"Date", "date", "Price"}
        and pd.api.types.is_numeric_dtype(data[col])
    ]
    x = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        min_child_samples=40,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=8,
    )
    model.fit(x[train], target[train])
    prob_valid = model.predict_proba(x[valid])[:, 1]
    prob_test = model.predict_proba(x[test])[:, 1]

    base_pred = data["y_pred"].to_numpy(dtype=float)
    y_test = y[test]
    base_test_pred = base_pred[test]
    base_accuracy = _accuracy(y_test, base_test_pred, price_floor)
    test_indices = np.flatnonzero(test)
    ranked = test_indices[np.argsort(prob_test)[-int(args.top_k) :]]
    lifted = base_pred.copy()
    lifted[ranked] = np.maximum(lifted[ranked], float(args.target_threshold))
    topk_accuracy = _accuracy(y_test, lifted[test], price_floor)
    hits = int(target[ranked].sum())

    print(f"features: {len(feature_cols)}")
    print(f"train/valid/test rows: {int(train.sum())}/{int(valid.sum())}/{int(test.sum())}")
    print(f"valid positives: {int(target[valid].sum())}")
    print(f"test positives: {int(target[test].sum())}")
    print(f"valid AUC: {roc_auc_score(target[valid], prob_valid):.6f}")
    print(f"valid AP: {average_precision_score(target[valid], prob_valid):.6f}")
    print(f"test AUC: {roc_auc_score(target[test], prob_test):.6f}")
    print(f"test AP: {average_precision_score(target[test], prob_test):.6f}")
    print(f"base test accuracy: {base_accuracy:.6f}")
    print(
        f"top-{int(args.top_k)} lift-to-{float(args.target_threshold):.0f} accuracy: "
        f"{topk_accuracy:.6f}"
    )
    print(f"top-{int(args.top_k)} hits: {hits}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose a weighted residual expert for mid/high price slots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-09-30")
    parser.add_argument("--valid-start", default="2025-10-01")
    parser.add_argument("--valid-end", default="2025-11-30")
    parser.add_argument("--test-start", default="2025-12-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--price-floor", type=float, default=40.0)
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


def _load_frame(features_path: Path, predictions_path: Path) -> pd.DataFrame:
    header = pd.read_csv(features_path, nrows=1)
    features = pd.read_csv(
        features_path,
        usecols=_select_columns(list(header.columns)),
        parse_dates=["Date", "date"],
    )
    predictions = pd.read_csv(
        predictions_path,
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
    frame = features.merge(predictions, on="Date", how="inner")
    frame["residual"] = frame["Price"] - frame["y_pred"]
    frame["abs_residual"] = frame["residual"].abs()
    frame["max_high_probability"] = np.maximum(
        frame["high_probability"].to_numpy(dtype=float),
        frame["cap_probability"].to_numpy(dtype=float),
    )
    return frame


def _mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"Date", "date", "Price", "residual", "abs_residual"}
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def main() -> None:
    args = parse_args()
    frame = _load_frame(Path(args.features), Path(args.predictions))
    frame = frame[frame["date"] >= pd.Timestamp(args.train_start)].copy()
    feature_cols = _feature_columns(frame)
    train = _mask(frame, args.train_start, args.train_end)
    valid = _mask(frame, args.valid_start, args.valid_end)
    test = _mask(frame, args.test_start, args.test_end)

    medians = frame.loc[train, feature_cols].replace([np.inf, -np.inf], np.nan).median()
    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
    y_residual = frame["residual"].to_numpy(dtype=float)
    y_price = frame["Price"].to_numpy(dtype=float)
    base_pred = frame["y_pred"].to_numpy(dtype=float)
    weights = np.ones(frame.shape[0], dtype=float)
    weights += 1.0 * (y_price >= 200.0)
    weights += 3.0 * (y_price >= 500.0)
    weights += 5.0 * (y_price >= 800.0)
    weights = weights / float(np.mean(weights[train]))

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        random_state=42,
        verbose=-1,
        n_jobs=8,
    )
    model.fit(x.loc[train], y_residual[train], sample_weight=weights[train])
    residual_hat = model.predict(x)

    base_valid_accuracy = _accuracy(y_price[valid], base_pred[valid], args.price_floor)
    base_test_accuracy = _accuracy(y_price[test], base_pred[test], args.price_floor)
    print(f"features: {len(feature_cols)}")
    print(f"base_valid_accuracy: {base_valid_accuracy:.6f}")
    print(f"base_test_accuracy: {base_test_accuracy:.6f}")

    best: tuple[float, tuple[float, float, float, bool], float] | None = None
    for shrink in [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        for clip_value in [40.0, 80.0, 120.0, 200.0, 320.0]:
            for min_prediction in [40.0, 100.0, 160.0, 240.0, 300.0]:
                for positive_only in [False, True]:
                    correction = np.clip(residual_hat, -clip_value, clip_value) * shrink
                    if positive_only:
                        correction = np.maximum(correction, 0.0)
                    eligible = (
                        (base_pred >= min_prediction)
                        & (frame["floor_probability"].to_numpy(dtype=float) <= 0.8)
                    )
                    pred = base_pred.copy()
                    pred[eligible] = np.clip(pred[eligible] + correction[eligible], 40.0, 1000.0)
                    valid_accuracy = _accuracy(y_price[valid], pred[valid], args.price_floor)
                    test_accuracy = _accuracy(y_price[test], pred[test], args.price_floor)
                    if best is None or valid_accuracy > best[0]:
                        best = (
                            valid_accuracy,
                            (shrink, clip_value, min_prediction, positive_only),
                            test_accuracy,
                        )
    if best is None:
        raise RuntimeError("No candidates evaluated.")
    print(f"best_valid_accuracy: {best[0]:.6f}")
    print(
        "best_params: "
        f"shrink={best[1][0]} clip={best[1][1]} "
        f"min_prediction={best[1][2]} positive_only={best[1][3]}"
    )
    print(f"selected_test_accuracy: {best[2]:.6f}")


if __name__ == "__main__":
    main()

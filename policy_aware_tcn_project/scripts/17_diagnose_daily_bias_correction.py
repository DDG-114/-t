#!/usr/bin/env python3
"""Diagnose whether daily bias correction can transfer to the test month."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error


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
    base = ["Date", "date", "Price"]
    patterns = [
        "净负荷",
        "供需裕度",
        "新能源",
        "负荷",
        "发电",
        "竞价空间",
        "抽蓄",
        "联络线",
        "prevday_curve",
        "floor_ratio",
        "price_roll",
        "weather_",
        "weather_error_",
        "scarcity_",
        "slot",
        "month",
        "day_of_week",
        "is_weekend",
    ]
    selected = [
        col
        for col in columns
        if col in base or any(pattern in col for pattern in patterns)
    ]
    return list(dict.fromkeys(selected))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, price_floor: float) -> float:
    denom = np.maximum(np.abs(y_true), float(price_floor))
    return float(1.0 - np.mean(np.abs(y_pred - y_true) / denom))


def _load_point_frame(feature_path: Path, prediction_path: Path) -> pd.DataFrame:
    header = pd.read_csv(feature_path, nrows=1)
    features = pd.read_csv(
        feature_path,
        usecols=_select_columns(list(header.columns)),
        parse_dates=["Date", "date"],
    )
    predictions = pd.read_csv(
        prediction_path,
        usecols=["Date", "y_pred", "floor_probability", "high_probability", "cap_probability"],
        parse_dates=["Date"],
    )
    return features.merge(predictions, on="Date", how="inner")


def _build_daily_frame(points: pd.DataFrame) -> pd.DataFrame:
    frame = points.copy()
    frame["residual"] = frame["Price"] - frame["y_pred"]
    numeric_cols = [
        col
        for col in frame.columns
        if col not in {"Date", "date", "Price"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
    aggregations = {}
    for col in numeric_cols:
        aggregations[f"{col}_mean"] = (col, "mean")
        if col in {
            "y_pred",
            "floor_probability",
            "high_probability",
            "cap_probability",
            "净负荷",
            "供需裕度",
            "统一负荷预测",
            "统一新能源预测",
            "发电总出力预测",
            "竞价空间",
        }:
            aggregations[f"{col}_max"] = (col, "max")
            aggregations[f"{col}_min"] = (col, "min")
            aggregations[f"{col}_std"] = (col, "std")

    daily = frame.groupby("date").agg(**aggregations).reset_index()
    target = (
        frame.groupby("date")
        .agg(
            day_residual_mean=("residual", "mean"),
            day_residual_median=("residual", "median"),
        )
        .reset_index()
    )
    return daily.merge(target, on="date")


def main() -> None:
    args = parse_args()
    points = _load_point_frame(Path(args.features), Path(args.predictions))
    points = points[points["date"] >= pd.Timestamp(args.train_start)].copy()
    daily = _build_daily_frame(points)
    feature_cols = [
        col
        for col in daily.columns
        if col not in {"date", "day_residual_mean", "day_residual_median"}
        and pd.api.types.is_numeric_dtype(daily[col])
    ]
    train = (daily["date"] >= args.train_start) & (daily["date"] <= args.train_end)
    valid = (daily["date"] >= args.valid_start) & (daily["date"] <= args.valid_end)
    test = (daily["date"] >= args.test_start) & (daily["date"] <= args.test_end)
    point_test = (points["date"] >= args.test_start) & (points["date"] <= args.test_end)
    y_test = points.loc[point_test, "Price"].to_numpy(dtype=float)
    base_test = points.loc[point_test, "y_pred"].to_numpy(dtype=float)
    base_accuracy = _accuracy(y_test, base_test, args.price_floor)

    print(f"daily_features: {len(feature_cols)}")
    print(f"train/valid/test_days: {int(train.sum())}/{int(valid.sum())}/{int(test.sum())}")
    print(f"base_test_accuracy: {base_accuracy:.6f}")
    for target_col in ["day_residual_mean", "day_residual_median"]:
        model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            random_state=42,
            verbose=-1,
        )
        model.fit(daily.loc[train, feature_cols], daily.loc[train, target_col])
        valid_pred = model.predict(daily.loc[valid, feature_cols])
        test_pred = model.predict(daily.loc[test, feature_cols])
        print(f"{target_col}_valid_mae: {mean_absolute_error(daily.loc[valid, target_col], valid_pred):.6f}")
        print(f"{target_col}_test_mae: {mean_absolute_error(daily.loc[test, target_col], test_pred):.6f}")
        day_correction = dict(zip(daily.loc[test, "date"], test_pred))
        for shrink in [0.25, 0.5, 0.75, 1.0]:
            corrected = points["y_pred"].to_numpy(dtype=float).copy()
            correction = points.loc[point_test, "date"].map(day_correction).to_numpy(dtype=float)
            corrected[point_test.to_numpy()] = np.clip(
                corrected[point_test.to_numpy()] + shrink * correction,
                40.0,
                1000.0,
            )
            accuracy = _accuracy(
                y_test,
                corrected[point_test.to_numpy()],
                args.price_floor,
            )
            print(f"{target_col}_shrink_{shrink:.2f}_test_accuracy: {accuracy:.6f}")


if __name__ == "__main__":
    main()

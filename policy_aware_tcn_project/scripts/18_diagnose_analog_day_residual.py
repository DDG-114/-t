#!/usr/bin/env python3
"""Diagnose analog-day residual curve correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--history-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2025-12-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 20, 45])
    parser.add_argument("--shrink", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--price-floor", type=float, default=40.0)
    return parser.parse_args()


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray, price_floor: float) -> float:
    denom = np.maximum(np.abs(y_true), float(price_floor))
    return float(1.0 - np.mean(np.abs(y_pred - y_true) / denom))


def _load_feature_frame(feature_path: Path, prediction_path: Path) -> pd.DataFrame:
    header = pd.read_csv(feature_path, nrows=1)
    base_cols = [
        "Date",
        "date",
        "slot",
        "Price",
        "发电总出力预测",
        "竞价空间",
        "统一负荷预测",
        "统一新能源预测",
        "净负荷",
        "供需裕度",
        "prevday_curve_mean",
        "prevday_curve_max",
        "prevday_curve_floor_ratio",
        "prevday_curve_high_ratio",
        "floor_ratio_1d",
        "floor_ratio_3d",
        "floor_ratio_7d",
    ]
    weather_cols = [
        col
        for col in header.columns
        if col.startswith("weather_") and col.endswith("_d1")
    ][:80]
    usecols = [col for col in [*base_cols, *weather_cols] if col in header.columns]
    features = pd.read_csv(feature_path, usecols=usecols, parse_dates=["Date", "date"])
    predictions = pd.read_csv(
        prediction_path,
        usecols=["Date", "y_pred"],
        parse_dates=["Date"],
    )
    frame = features.merge(predictions, on="Date", how="inner")
    frame["residual"] = frame["Price"] - frame["y_pred"]
    return frame.sort_values(["date", "slot"]).reset_index(drop=True)


def _daily_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in frame.columns
        if col not in {"Date", "date", "Price", "residual"}
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    aggregations = {}
    rich_cols = {
        "发电总出力预测",
        "竞价空间",
        "统一负荷预测",
        "统一新能源预测",
        "净负荷",
        "供需裕度",
        "y_pred",
    }
    for col in numeric_cols:
        aggregations[f"{col}_mean"] = (col, "mean")
        if col in rich_cols:
            aggregations[f"{col}_max"] = (col, "max")
            aggregations[f"{col}_min"] = (col, "min")
    return frame.groupby("date").agg(**aggregations).reset_index()


def main() -> None:
    args = parse_args()
    frame = _load_feature_frame(Path(args.features), Path(args.predictions))
    frame = frame[frame["date"] >= pd.Timestamp(args.history_start)].copy()
    daily = _daily_feature_table(frame)
    feature_cols = [col for col in daily.columns if col != "date"]
    history_daily = daily[daily["date"] < pd.Timestamp(args.test_start)].copy()
    fill_values = history_daily[feature_cols].median(numeric_only=True)
    scaler = StandardScaler().fit(
        history_daily[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    )
    x_all = scaler.transform(
        daily[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    )
    daily_index = {date: idx for idx, date in enumerate(daily["date"])}

    residual_curve = frame.pivot(index="date", columns="slot", values="residual").sort_index()
    pred_curve = frame.pivot(index="date", columns="slot", values="y_pred").sort_index()
    true_curve = frame.pivot(index="date", columns="slot", values="Price").sort_index()
    test_dates = list(pd.date_range(args.test_start, args.test_end, freq="D"))
    y_true = true_curve.loc[test_dates].to_numpy(dtype=float).reshape(-1)
    base_pred = pred_curve.loc[test_dates].to_numpy(dtype=float).reshape(-1)
    base_accuracy = _accuracy(y_true, base_pred, args.price_floor)

    print(f"base_test_accuracy: {base_accuracy:.6f}")
    best: tuple[float, int, float] | None = None
    for k in args.k:
        for shrink in args.shrink:
            corrected_days = []
            for date in test_dates:
                history_dates = [
                    item
                    for item in daily["date"]
                    if item < date and item >= pd.Timestamp(args.history_start)
                ]
                if not history_dates:
                    correction = np.zeros(96, dtype=float)
                else:
                    current = x_all[daily_index[date]]
                    history_x = np.vstack([x_all[daily_index[item]] for item in history_dates])
                    distance = np.linalg.norm(history_x - current, axis=1)
                    order = np.argsort(distance)[: min(int(k), len(history_dates))]
                    weights = 1.0 / (distance[order] + 1e-6)
                    weights = weights / weights.sum()
                    curves = np.vstack(
                        [
                            residual_curve.loc[history_dates[index]].to_numpy(dtype=float)
                            for index in order
                        ]
                    )
                    correction = np.average(curves, axis=0, weights=weights)
                corrected = pred_curve.loc[date].to_numpy(dtype=float) + float(shrink) * correction
                corrected_days.append(np.clip(corrected, 40.0, 1000.0))
            y_pred = np.vstack(corrected_days).reshape(-1)
            accuracy = _accuracy(y_true, y_pred, args.price_floor)
            print(f"k={int(k)} shrink={float(shrink):.3f} test_accuracy={accuracy:.6f}")
            if best is None or accuracy > best[0]:
                best = (accuracy, int(k), float(shrink))
    if best:
        print(f"best_accuracy: {best[0]:.6f}")
        print(f"best_k: {best[1]}")
        print(f"best_shrink: {best[2]:.3f}")


if __name__ == "__main__":
    main()

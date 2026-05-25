#!/usr/bin/env python3
"""Analyze where prediction error prevents reaching the target accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--price-floor", type=float, default=40.0)
    parser.add_argument("--target-accuracy", type=float, default=0.85)
    parser.add_argument("--top-days", type=int, default=12)
    return parser.parse_args()


def _price_bucket(price: pd.Series) -> pd.Categorical:
    return pd.cut(
        price,
        bins=[-np.inf, 40.1, 100.0, 200.0, 500.0, 800.0, np.inf],
        labels=["floor40", "40-100", "100-200", "200-500", "500-800", "800+"],
    )


def main() -> None:
    args = parse_args()
    path = Path(args.predictions)
    predictions = pd.read_csv(path, parse_dates=["Date", "date"])
    required = {"Date", "date", "slot", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")

    df = predictions.sort_values(["date", "slot"]).copy()
    denominator = np.maximum(np.abs(df["y_true"].to_numpy(dtype=float)), args.price_floor)
    df["abs_error"] = np.abs(
        df["y_pred"].to_numpy(dtype=float) - df["y_true"].to_numpy(dtype=float)
    )
    df["relative_error"] = df["abs_error"] / denominator
    df["hour"] = df["Date"].dt.hour
    df["price_bucket"] = _price_bucket(df["y_true"])

    mean_rel = float(df["relative_error"].mean())
    target_rel = 1.0 - float(args.target_accuracy)
    print(f"rows: {df.shape[0]}")
    print(f"accuracy: {1.0 - mean_rel:.6f}")
    print(f"mean_relative_error: {mean_rel:.6f}")
    print(f"target_mean_relative_error: {target_rel:.6f}")
    print(f"relative_error_gap_to_target: {mean_rel - target_rel:.6f}")

    daily = (
        df.groupby("date")
        .agg(
            relative_error_sum=("relative_error", "sum"),
            accuracy=("relative_error", lambda values: 1.0 - float(values.mean())),
            mae=("abs_error", "mean"),
            true_high=("y_true", lambda values: int((values >= 500.0).sum())),
            true_cap=("y_true", lambda values: int((values >= 800.0).sum())),
            floor_slots=("y_true", lambda values: int((values == 40.0).sum())),
            pred_high=("y_pred", lambda values: int((values >= 500.0).sum())),
            true_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
        )
        .reset_index()
        .sort_values("relative_error_sum", ascending=False)
    )
    print("\nWorst days:")
    print(daily.head(int(args.top_days)).to_string(index=False))

    bucket = (
        df.groupby("price_bucket", observed=True)
        .agg(
            n=("relative_error", "size"),
            relative_error_sum=("relative_error", "sum"),
            accuracy=("relative_error", lambda values: 1.0 - float(values.mean())),
            mae=("abs_error", "mean"),
            true_mean=("y_true", "mean"),
            pred_mean=("y_pred", "mean"),
        )
        .reset_index()
    )
    total_rel = float(df["relative_error"].sum())
    bucket["relative_error_share"] = bucket["relative_error_sum"] / total_rel
    print("\nPrice buckets:")
    print(bucket.to_string(index=False))

    hourly = (
        df.groupby("hour")
        .agg(
            n=("relative_error", "size"),
            relative_error_sum=("relative_error", "sum"),
            accuracy=("relative_error", lambda values: 1.0 - float(values.mean())),
            true_high=("y_true", lambda values: int((values >= 500.0).sum())),
            floor_slots=("y_true", lambda values: int((values == 40.0).sum())),
        )
        .reset_index()
        .sort_values("relative_error_sum", ascending=False)
    )
    print("\nWorst hours:")
    print(hourly.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

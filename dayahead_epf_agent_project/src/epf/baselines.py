"""Baseline forecasting methods."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from epf.metrics import daily_metrics, regression_summary


def make_baseline_predictions(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create baseline predictions from precomputed lag and rolling features."""
    target = config["columns"]["target"]
    result = df[["date", "slot", target]].copy()
    result = result.rename(columns={target: "y_true"})

    if "price_lag_1d" in df.columns:
        result["baseline_yesterday"] = df["price_lag_1d"]
    if "price_lag_7d" in df.columns:
        result["baseline_last_week"] = df["price_lag_7d"]
    if "price_roll_7d_median" in df.columns:
        result["baseline_7d_median"] = df["price_roll_7d_median"]
    if "price_roll_7d_mean" in df.columns:
        result["baseline_7d_mean"] = df["price_roll_7d_mean"]

    return result


def evaluate_baselines(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Evaluate all baseline columns."""
    price_floor = config["metrics"].get("price_floor", 40.0)
    price_range = config["metrics"].get("price_range", 1000.0)
    threshold = config["metrics"].get("daily_accuracy_threshold", 0.85)
    summaries = {}
    baseline_cols = [col for col in predictions.columns if col.startswith("baseline_")]

    for col in baseline_cols:
        valid = predictions[["y_true", col]].dropna()
        daily = baseline_daily_metrics(predictions, config, col)
        evaluated_daily = daily[daily["n_valid"] > 0]
        pass_days = (evaluated_daily["accuracy"] >= threshold).sum()
        pass_days_cap = (
            evaluated_daily["cap_normalized_accuracy"] >= threshold
        ).sum()
        summary = regression_summary(valid["y_true"], valid[col], price_floor)
        summary.update(
            {
                "cap_normalized_accuracy": (
                    float(1.0 - summary["mae"] / price_range)
                    if pd.notna(summary["mae"])
                    else None
                ),
                "daily_accuracy_threshold": threshold,
                "n_days": int(daily.shape[0]),
                "n_evaluated_days": int(evaluated_daily.shape[0]),
                "pass_days": int(pass_days),
                "pass_day_ratio": (
                    float(pass_days / evaluated_daily.shape[0])
                    if evaluated_daily.shape[0]
                    else None
                ),
                "pass_days_cap_normalized": int(pass_days_cap),
                "pass_day_ratio_cap_normalized": (
                    float(pass_days_cap / evaluated_daily.shape[0])
                    if evaluated_daily.shape[0]
                    else None
                ),
                "mean_daily_accuracy": (
                    float(evaluated_daily["accuracy"].mean())
                    if evaluated_daily.shape[0]
                    else None
                ),
                "median_daily_accuracy": (
                    float(evaluated_daily["accuracy"].median())
                    if evaluated_daily.shape[0]
                    else None
                ),
                "mean_daily_cap_normalized_accuracy": (
                    float(evaluated_daily["cap_normalized_accuracy"].mean())
                    if evaluated_daily.shape[0]
                    else None
                ),
                "median_daily_cap_normalized_accuracy": (
                    float(evaluated_daily["cap_normalized_accuracy"].median())
                    if evaluated_daily.shape[0]
                    else None
                ),
            }
        )
        summaries[col] = summary

    return summaries


def baseline_daily_metrics(predictions: pd.DataFrame, config: Dict[str, Any], baseline_col: str) -> pd.DataFrame:
    """Compute daily metrics for one baseline column."""
    tmp = predictions[["date", "slot", "y_true", baseline_col]].rename(columns={baseline_col: "y_pred"})
    return daily_metrics(
        tmp,
        y_col="y_true",
        pred_col="y_pred",
        price_floor=config["metrics"].get("price_floor", 40.0),
        price_range=config["metrics"].get("price_range", 1000.0),
        expected_slots=config["data"].get("expected_slots_per_day", 96),
    )

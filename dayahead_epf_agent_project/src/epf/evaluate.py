"""Evaluation helpers for trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from epf.metrics import daily_metrics, monthly_metrics, regression_summary


def evaluate_predictions(predictions: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate point predictions and return summary plus daily table."""
    price_floor = config["metrics"].get("price_floor", 40.0)
    price_range = config["metrics"].get("price_range", 1000.0)
    threshold = config["metrics"].get("daily_accuracy_threshold", 0.85)
    expected_slots = config["data"].get("expected_slots_per_day", 96)

    valid = predictions[["y_true", "y_pred"]].dropna()
    overall = regression_summary(valid["y_true"], valid["y_pred"], price_floor)
    daily = daily_metrics(
        predictions,
        y_col="y_true",
        pred_col="y_pred",
        price_floor=price_floor,
        price_range=price_range,
        expected_slots=expected_slots,
    )
    daily["pass_85pct"] = daily["accuracy"] >= threshold
    daily["pass_85pct_cap_normalized"] = daily["cap_normalized_accuracy"] >= threshold
    daily["pass_85pct_point_accuracy_clipped"] = (
        daily["mean_point_accuracy_clipped"] >= threshold
    )
    monthly = monthly_metrics(daily, threshold=threshold)

    valid_days = daily[daily["n_valid"] > 0] if not daily.empty else daily
    summary = {
        "overall": overall,
        "overall_cap_normalized_accuracy": (
            float(1.0 - overall["mae"] / price_range)
            if pd.notna(overall["mae"])
            else None
        ),
        "daily_accuracy_threshold": threshold,
        "price_floor": price_floor,
        "price_range": price_range,
        "n_days": int(daily.shape[0]),
        "n_complete_days": int(daily["complete_day"].sum()) if not daily.empty else 0,
        "n_evaluated_days": int(valid_days.shape[0]) if valid_days is not None else 0,
        "pass_days": int(valid_days["pass_85pct"].sum()) if not valid_days.empty else 0,
        "pass_day_ratio": float(valid_days["pass_85pct"].mean()) if not valid_days.empty else None,
        "pass_days_cap_normalized": (
            int(valid_days["pass_85pct_cap_normalized"].sum())
            if not valid_days.empty
            else 0
        ),
        "pass_day_ratio_cap_normalized": (
            float(valid_days["pass_85pct_cap_normalized"].mean())
            if not valid_days.empty
            else None
        ),
        "mean_daily_accuracy": float(valid_days["accuracy"].mean()) if not valid_days.empty else None,
        "median_daily_accuracy": float(valid_days["accuracy"].median()) if not valid_days.empty else None,
        "mean_daily_point_accuracy_clipped": (
            float(valid_days["mean_point_accuracy_clipped"].mean())
            if not valid_days.empty
            else None
        ),
        "pass_days_point_accuracy_clipped": (
            int(valid_days["pass_85pct_point_accuracy_clipped"].sum())
            if not valid_days.empty
            else 0
        ),
        "pass_day_ratio_point_accuracy_clipped": (
            float(valid_days["pass_85pct_point_accuracy_clipped"].mean())
            if not valid_days.empty
            else None
        ),
        "mean_daily_cap_normalized_accuracy": (
            float(valid_days["cap_normalized_accuracy"].mean())
            if not valid_days.empty
            else None
        ),
        "median_daily_cap_normalized_accuracy": (
            float(valid_days["cap_normalized_accuracy"].median())
            if not valid_days.empty
            else None
        ),
        "n_prediction_rows": int(predictions.shape[0]),
        "n_valid_prediction_rows": int(valid.shape[0]),
    }
    return {"summary": summary, "daily": daily, "monthly": monthly}


def save_evaluation(
    predictions: pd.DataFrame,
    evaluation: Dict[str, Any],
    config: Dict[str, Any],
    prefix: str = "slot_lightgbm",
) -> None:
    """Save predictions, daily metrics, and summary JSON."""
    pred_dir = Path(config["paths"]["prediction_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(pred_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    evaluation["daily"].to_csv(report_dir / f"{prefix}_daily_metrics.csv", index=False, encoding="utf-8-sig")
    evaluation["monthly"].to_csv(
        report_dir / f"{prefix}_monthly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (report_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation["summary"], file, ensure_ascii=False, indent=2)

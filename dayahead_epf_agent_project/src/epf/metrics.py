"""Evaluation metrics for electricity price forecasting."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def modified_relative_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_floor: float = 40.0,
) -> np.ndarray:
    """Compute relative error with a minimum denominator.

    This avoids division by zero when spot price equals zero.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(true), price_floor)
    return np.abs(pred - true) / denominator


def point_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_floor: float = 40.0,
    clip: bool = False,
) -> np.ndarray:
    """Compute per-point accuracy as 1 - modified relative error."""
    accuracy = 1.0 - modified_relative_error(y_true, y_pred, price_floor)
    if clip:
        accuracy = np.clip(accuracy, 0.0, 1.0)
    return accuracy


def modified_mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_floor: float = 40.0,
) -> float:
    """Mean modified absolute percentage error."""
    return float(np.nanmean(modified_relative_error(y_true, y_pred, price_floor)))


def daily_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_floor: float = 40.0,
) -> float:
    """Daily accuracy defined as 1 - mean modified relative error."""
    return float(1.0 - modified_mape(y_true, y_pred, price_floor))


def regression_summary(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_floor: float = 40.0,
) -> Dict[str, float]:
    """Return common regression metrics."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(true) | np.isnan(pred))
    true = true[mask]
    pred = pred[mask]

    if len(true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "modified_mape": np.nan, "accuracy": np.nan}

    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "modified_mape": modified_mape(true, pred, price_floor),
        "accuracy": daily_accuracy(true, pred, price_floor),
        "mean_point_accuracy": float(np.nanmean(point_accuracy(true, pred, price_floor))),
        "mean_point_accuracy_clipped": float(
            np.nanmean(point_accuracy(true, pred, price_floor, clip=True))
        ),
    }


def cap_normalized_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    price_range: float = 1000.0,
) -> float:
    """Accuracy based on MAE normalized by the configured market price range."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(true) | np.isnan(pred))
    true = true[mask]
    pred = pred[mask]
    if len(true) == 0:
        return np.nan
    return float(1.0 - mean_absolute_error(true, pred) / price_range)


def daily_metrics(
    predictions: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_pred",
    price_floor: float = 40.0,
    price_range: float = 1000.0,
    expected_slots: int = 96,
) -> pd.DataFrame:
    """Compute metrics grouped by prediction date."""
    rows = []
    for day, group in predictions.groupby("date"):
        complete = int(group.shape[0]) == expected_slots
        valid = group[[y_col, pred_col]].dropna()
        summary = regression_summary(valid[y_col], valid[pred_col], price_floor)
        summary["cap_normalized_accuracy"] = cap_normalized_accuracy(
            valid[y_col],
            valid[pred_col],
            price_range,
        )
        summary.update(
            {
                "date": str(pd.Timestamp(day).date()),
                "n_slots": int(group.shape[0]),
                "n_valid": int(valid.shape[0]),
                "complete_day": bool(complete),
            }
        )
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def monthly_metrics(
    daily: pd.DataFrame,
    threshold: float = 0.85,
) -> pd.DataFrame:
    """Aggregate daily metrics by natural month."""
    if daily.empty:
        return pd.DataFrame()

    result = daily.copy()
    result["month"] = pd.to_datetime(result["date"]).dt.to_period("M").astype(str)
    valid = result[result["n_valid"] > 0].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for month, group in valid.groupby("month"):
        row = {
            "month": month,
            "n_days": int(group.shape[0]),
            "mean_daily_accuracy": float(group["accuracy"].mean()),
            "median_daily_accuracy": float(group["accuracy"].median()),
            "pass_days": int((group["accuracy"] >= threshold).sum()),
            "pass_day_ratio": float((group["accuracy"] >= threshold).mean()),
        }
        if "mean_point_accuracy_clipped" in group.columns:
            row["mean_daily_point_accuracy_clipped"] = float(
                group["mean_point_accuracy_clipped"].mean()
            )
            row["pass_days_point_accuracy_clipped"] = int(
                (group["mean_point_accuracy_clipped"] >= threshold).sum()
            )
        if "cap_normalized_accuracy" in group.columns:
            row["mean_daily_cap_normalized_accuracy"] = float(
                group["cap_normalized_accuracy"].mean()
            )
            row["pass_days_cap_normalized"] = int(
                (group["cap_normalized_accuracy"] >= threshold).sum()
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

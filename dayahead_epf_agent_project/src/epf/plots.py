"""Plotting helpers.

These plots are optional and are intended for report/debugging outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_one_day_curve(
    predictions: pd.DataFrame,
    date: str,
    output_path: str | Path,
) -> None:
    """Plot true vs predicted 96-point price curve for one day."""
    day = pd.Timestamp(date)
    data = predictions[pd.to_datetime(predictions["date"]) == day].sort_values("slot")
    if data.empty:
        raise ValueError(f"No predictions found for date: {date}")

    plt.figure(figsize=(12, 5))
    plt.plot(data["slot"], data["y_true"], label="Actual")
    plt.plot(data["slot"], data["y_pred"], label="Predicted")
    plt.xlabel("15-minute slot")
    plt.ylabel("Price")
    plt.title(f"Day-ahead price forecast: {date}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

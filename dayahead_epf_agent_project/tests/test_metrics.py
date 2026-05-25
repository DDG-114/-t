import numpy as np
import pytest

from epf.metrics import (
    cap_normalized_accuracy,
    daily_accuracy,
    modified_mape,
    monthly_metrics,
    modified_relative_error,
)


def test_modified_relative_error_handles_zero_price():
    y_true = np.array([0.0, 40.0, 100.0])
    y_pred = np.array([20.0, 50.0, 110.0])
    errors = modified_relative_error(y_true, y_pred, price_floor=40.0)
    assert np.all(np.isfinite(errors))
    assert errors[0] == 0.5


def test_daily_accuracy_is_one_minus_modified_mape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([90.0, 220.0])
    assert daily_accuracy(y_true, y_pred) == 1.0 - modified_mape(y_true, y_pred)


def test_cap_normalized_accuracy_uses_mae_over_price_range():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([100.0, 200.0])
    assert cap_normalized_accuracy(y_true, y_pred, price_range=1000.0) == 0.9


def test_monthly_metrics_averages_daily_accuracy_by_month():
    import pandas as pd

    daily = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-02", "2026-01-01"],
            "accuracy": [0.8, 0.9, 0.7],
            "n_valid": [96, 96, 96],
            "mean_point_accuracy_clipped": [0.8, 0.9, 0.7],
            "cap_normalized_accuracy": [0.85, 0.95, 0.75],
        }
    )
    monthly = monthly_metrics(daily, threshold=0.85)

    dec = monthly[monthly["month"] == "2025-12"].iloc[0]
    assert dec["mean_daily_accuracy"] == pytest.approx(0.85)
    assert dec["pass_days"] == 1
    assert dec["pass_days_cap_normalized"] == 2

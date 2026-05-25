import pandas as pd
import pytest

from epf_tcn.metrics import monthly_metrics


def test_monthly_daily_accuracy_is_sum_divided_by_days():
    daily = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-02", "2025-12-03"],
            "accuracy": [0.5, 0.7, 0.9],
            "n_valid": [96, 96, 96],
            "mean_point_accuracy_clipped": [0.5, 0.7, 0.9],
            "cap_normalized_accuracy": [0.8, 0.9, 0.95],
        }
    )

    monthly = monthly_metrics(daily, threshold=0.85)
    row = monthly.iloc[0]

    assert row["daily_accuracy_sum"] == pytest.approx(2.1)
    assert row["monthly_daily_accuracy"] == pytest.approx(2.1 / 3.0)
    assert row["mean_daily_accuracy"] == pytest.approx(row["monthly_daily_accuracy"])
    assert row["pass_days"] == 1

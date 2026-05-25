import pandas as pd
import pytest

from epf_tcn.online_calibrator import (
    OnlineResidualConfig,
    apply_online_residual_correction,
)


def test_online_residual_correction_uses_only_previous_days():
    rows = []
    for day in range(8):
        for slot in range(2):
            rows.append(
                {
                    "Date": pd.Timestamp("2025-12-01") + pd.Timedelta(days=day, minutes=15 * slot),
                    "date": pd.Timestamp("2025-12-01") + pd.Timedelta(days=day),
                    "slot": slot,
                    "y_true": 200.0,
                    "y_pred": 100.0,
                    "floor_probability": 0.0,
                    "high_probability": 0.0,
                    "cap_probability": 0.0,
                }
            )
    predictions = pd.DataFrame(rows)

    corrected = apply_online_residual_correction(
        predictions,
        OnlineResidualConfig(
            min_history_days=7,
            window_days=7,
            group="pred_bin",
            shrink=0.5,
            clip_value=160.0,
            min_prediction=60.0,
            max_floor_probability=0.8,
        ),
    )

    first_week = corrected[corrected["date"] < pd.Timestamp("2025-12-08")]
    day_eight = corrected[corrected["date"] == pd.Timestamp("2025-12-08")]
    assert first_week["y_pred"].tolist() == pytest.approx([100.0] * 14)
    assert day_eight["y_pred"].tolist() == pytest.approx([150.0, 150.0])
    assert day_eight["online_residual_correction"].tolist() == pytest.approx([50.0, 50.0])


def test_spike_online_residual_can_override_high_probability_slots():
    rows = []
    for day in range(4):
        for slot in range(2):
            is_spike_slot = slot == 1
            rows.append(
                {
                    "Date": pd.Timestamp("2025-12-01")
                    + pd.Timedelta(days=day, minutes=15 * slot),
                    "date": pd.Timestamp("2025-12-01") + pd.Timedelta(days=day),
                    "slot": slot,
                    "y_true": 700.0 if is_spike_slot else 200.0,
                    "y_pred": 300.0 if is_spike_slot else 100.0,
                    "floor_probability": 0.0,
                    "high_probability": 0.2 if is_spike_slot else 0.0,
                    "cap_probability": 0.0,
                }
            )
    predictions = pd.DataFrame(rows)

    corrected = apply_online_residual_correction(
        predictions,
        OnlineResidualConfig(
            min_history_days=3,
            window_days=3,
            group="pred_bin",
            shrink=0.5,
            clip_value=160.0,
            min_prediction=60.0,
            max_floor_probability=0.8,
            spike_enabled=True,
            spike_group="spike_state",
            spike_min_probability=0.08,
            spike_shrink=1.0,
            spike_clip_value=500.0,
            spike_min_history_rows=3,
        ),
    )

    day_four = corrected[corrected["date"] == pd.Timestamp("2025-12-04")]
    assert day_four["y_pred"].tolist() == pytest.approx([150.0, 700.0])

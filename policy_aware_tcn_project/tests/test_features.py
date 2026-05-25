import numpy as np
import pandas as pd
import pytest

from epf_tcn.features import add_floor_price_features


def test_floor_price_features_track_runs_across_day_boundary():
    rows = []
    for idx, price in enumerate([100.0, 40.0, 40.0, 40.0, 40.0, 80.0]):
        timestamp = pd.Timestamp("2025-01-01 23:30") + pd.Timedelta(minutes=15 * idx)
        rows.append(
            {
                "Date": timestamp,
                "date": timestamp.floor("D"),
                "slot": (timestamp.hour * 60 + timestamp.minute) // 15,
                "Price": price,
            }
        )
    df = pd.DataFrame(rows)

    features = add_floor_price_features(
        df,
        target_col="Price",
        floor_price=40.0,
        windows=[1],
        long_run_slots=3,
        datetime_col="Date",
        slot_minutes=15,
        expected_slots=96,
    )

    assert features["floor_run_slots"].tolist() == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    )
    assert features["prev_floor_run_slots"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    )
    assert features["prev_long_floor_run"].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    )
    assert np.isfinite(features["floor_ratio_1d"].fillna(0.0)).all()

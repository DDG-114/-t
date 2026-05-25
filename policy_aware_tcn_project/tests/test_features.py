import numpy as np
import pandas as pd
import pytest

from epf_tcn.features import add_exogenous_quantile_features, add_floor_price_features


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


def test_exogenous_quantile_features_use_previous_same_slot_values_only():
    rows = []
    for day in range(5):
        for slot in range(2):
            timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(
                days=day,
                minutes=15 * slot,
            )
            rows.append(
                {
                    "Date": timestamp,
                    "date": timestamp.floor("D"),
                    "slot": slot,
                    "统一负荷预测": float(day * 10 + slot),
                    "统一新能源预测": float(day + slot),
                    "发电总出力预测": float(100 + day * 10 + slot),
                }
            )
    df = pd.DataFrame(rows)

    features = add_exogenous_quantile_features(
        df,
        source_cols=["统一负荷预测", "统一新能源预测", "发电总出力预测"],
        windows=[3],
        quantile_levels=[0.1, 0.5, 0.9],
        include_spreads=True,
        include_derived=True,
    )

    row = features[(features["date"] == pd.Timestamp("2025-01-05")) & (features["slot"] == 0)].iloc[0]

    assert row["统一负荷预测_q50_3d"] == pytest.approx(20.0)
    assert row["统一负荷预测_q10_3d"] == pytest.approx(12.0)
    assert row["统一负荷预测_q90_3d"] == pytest.approx(28.0)
    assert row["统一负荷预测_iqr_3d"] == pytest.approx(16.0)
    assert row["净负荷_q90_3d"] == pytest.approx(
        row["统一负荷预测_q90_3d"] - row["统一新能源预测_q10_3d"]
    )

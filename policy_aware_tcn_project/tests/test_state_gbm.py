import pytest
import pandas as pd

from epf_tcn.state_gbm import add_scarcity_state_features, add_state_gbm_features


def test_state_gbm_boundary_features_use_previous_same_slot_only():
    rows = []
    for day, price in enumerate([40.0, 1000.0, 300.0]):
        for slot in [0, 1]:
            timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(
                days=day,
                minutes=15 * slot,
            )
            rows.append(
                {
                    "Date": timestamp,
                    "date": timestamp.floor("D"),
                    "slot": slot,
                    "Price": price + slot,
                    "发电总出力预测": 100.0,
                    "竞价空间": 50.0,
                    "统一负荷预测": 80.0,
                    "抽蓄": 0.0,
                    "统一新能源预测": 20.0,
                    "联络线计划": 10.0,
                }
            )
    df = pd.DataFrame(rows)
    config = {"columns": {"target": "Price"}}

    features = add_state_gbm_features(df, config)
    row = features[(features["date"] == pd.Timestamp("2025-01-03")) & (features["slot"] == 0)].iloc[0]

    assert row["floor45_ratio_1d"] == 0.0
    assert row["cap900_ratio_1d"] == 1.0
    assert row["floor45_ratio_2d"] == 0.5
    assert row["cap900_ratio_2d"] == 0.5


def test_scarcity_state_features_use_previous_same_slot_forecasts_only():
    rows = []
    for day, load in enumerate([100.0, 120.0, 140.0, 1000.0]):
        for slot in [0, 1]:
            timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(
                days=day,
                minutes=15 * slot,
            )
            rows.append(
                {
                    "Date": timestamp,
                    "date": timestamp.floor("D"),
                    "slot": slot,
                    "统一负荷预测": load + slot,
                    "统一新能源预测": 20.0,
                    "净负荷": load - 20.0 + slot,
                    "竞价空间占比": 0.5,
                    "新能源占比": 0.2,
                }
            )
    df = pd.DataFrame(rows)

    features = add_scarcity_state_features(
        df,
        {
            "state_gbm": {
                "scarcity_features": {
                    "enabled": True,
                    "source_columns": ["统一负荷预测"],
                    "windows_days": [3],
                    "quantiles": [0.1, 0.5, 0.9],
                }
            }
        },
    )
    row = features[
        (features["date"] == pd.Timestamp("2025-01-04")) & (features["slot"] == 0)
    ].iloc[0]

    assert row["统一负荷预测_scarcity_mean_3d"] == pytest.approx(120.0)
    assert row["统一负荷预测_scarcity_q50_3d"] == pytest.approx(120.0)
    assert row["统一负荷预测_scarcity_q90_3d"] == pytest.approx(136.0)
    assert row["统一负荷预测_scarcity_delta_q90_3d"] == pytest.approx(864.0)
    assert row["统一负荷预测_scarcity_above_q90_3d"] == 1.0

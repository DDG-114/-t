import pandas as pd

from epf_tcn.state_gbm import add_state_gbm_features


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

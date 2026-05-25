import pandas as pd

from epf.data import clean_target_prices


def test_clean_target_prices_flags_cap_and_long_constant_runs():
    config = {
        "columns": {"datetime": "Date", "target": "Price"},
        "data": {
            "slot_minutes": 15,
            "target_cleaning": {
                "enabled": True,
                "cap_values_as_missing": [0.0, 1000.0],
                "constant_run_min_length": 4,
            },
        },
    }
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=6, freq="15min"),
            "Price": [40.0, 40.0, 40.0, 40.0, 1000.0, 0.0],
        }
    )

    cleaned = clean_target_prices(df, config)

    assert cleaned["Price"].isna().sum() == 6
    assert cleaned.loc[0, "Price_clean_reason"] == "long_constant_run"
    assert cleaned.loc[4, "Price_clean_reason"] == "cap_value"
    assert cleaned.loc[5, "Price_clean_reason"] == "cap_value"
    assert cleaned.loc[4, "Price_raw"] == 1000.0

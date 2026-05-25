import pandas as pd

from epf_tcn.data import clean_target_prices


def _config(cap_values=None, exempt_values=None, constant_run_min_length=3):
    return {
        "columns": {"datetime": "Date", "target": "Price"},
        "data": {
            "slot_minutes": 15,
            "target_cleaning": {
                "enabled": True,
                "cap_values_as_missing": cap_values or [0.0],
                "constant_run_exempt_values": exempt_values or [40.0, 1000.0],
                "constant_run_min_length": constant_run_min_length,
            },
        },
    }


def test_clean_target_prices_keeps_1000_when_not_configured_as_missing():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=4, freq="15min"),
            "Price": [40.0, 1000.0, 0.0, 1000.0],
        }
    )

    cleaned = clean_target_prices(df, _config(constant_run_min_length=None))

    assert cleaned.loc[cleaned["Price_raw"].eq(1000.0), "Price"].tolist() == [
        1000.0,
        1000.0,
    ]
    assert cleaned.loc[cleaned["Price_raw"].eq(0.0), "Price"].isna().all()


def test_clean_target_prices_does_not_delete_long_floor_or_cap_runs():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=6, freq="15min"),
            "Price": [40.0, 40.0, 40.0, 1000.0, 1000.0, 1000.0],
        }
    )

    cleaned = clean_target_prices(df, _config())

    assert cleaned["Price"].tolist() == [40.0, 40.0, 40.0, 1000.0, 1000.0, 1000.0]
    assert cleaned["Price_clean_reason"].eq("").all()

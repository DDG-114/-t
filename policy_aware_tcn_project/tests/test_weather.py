import pandas as pd

from epf_tcn.weather import (
    add_weather_error_history_features,
    merge_weather_features,
    weather_date_bounds,
)


def test_merge_weather_features_aligns_hourly_forecasts_to_slots():
    features = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01 00:15", "2025-01-01 00:45"]),
            "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "Price": [40.0, 50.0],
        }
    )
    weather = pd.DataFrame(
        {
            "Date_hour": pd.to_datetime(["2025-01-01 00:00"]),
            "weather_xian_temperature_2m_d1": [2.5],
        }
    )

    merged = merge_weather_features(features, weather)

    assert merged["weather_xian_temperature_2m_d1"].tolist() == [2.5, 2.5]
    assert "Date_hour" not in merged.columns


def test_weather_date_bounds_are_inclusive_dates():
    features = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-01-01 00:15", "2025-01-03 23:45"]),
        }
    )

    assert weather_date_bounds(features) == ("2025-01-01", "2025-01-03")


def test_weather_error_history_uses_previous_day_same_hour_only():
    rows = []
    for day in range(3):
        for hour in [0, 1]:
            for quarter in range(4):
                timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(
                    days=day,
                    hours=hour,
                    minutes=15 * quarter,
                )
                rows.append(
                    {
                        "Date": timestamp,
                        "weather_xian_temperature_2m_d1": float(day * 10 + hour),
                    }
                )
    features = pd.DataFrame(rows)
    actual = pd.DataFrame(
        {
            "Date_hour": pd.date_range("2025-01-01", periods=3 * 2, freq="h"),
            "weather_actual_xian_temperature_2m": [1.0, 2.0, 11.0, 12.0, 21.0, 22.0],
        }
    )

    enhanced = add_weather_error_history_features(
        features,
        actual,
        forecast_lead_day=1,
        windows_days=[1],
    )
    target = enhanced[enhanced["Date"].eq(pd.Timestamp("2025-01-02 00:00"))].iloc[0]

    assert target["weather_error_xian_temperature_2m_d1_lag1d"] == 1.0
    assert "weather_actual_xian_temperature_2m" not in enhanced.columns

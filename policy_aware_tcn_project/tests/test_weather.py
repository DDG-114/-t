import pandas as pd

from epf_tcn.weather import merge_weather_features, weather_date_bounds


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

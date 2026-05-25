"""Weather forecast feature utilities for EPF experiments."""

from __future__ import annotations

import json
import warnings
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd


DEFAULT_WEATHER_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
]

DEFAULT_SHAANXI_STATIONS = [
    {"name": "xian", "latitude": 34.34, "longitude": 108.94},
    {"name": "yulin", "latitude": 38.29, "longitude": 109.73},
    {"name": "yanan", "latitude": 36.59, "longitude": 109.49},
    {"name": "baoji", "latitude": 34.36, "longitude": 107.24},
    {"name": "hanzhong", "latitude": 33.07, "longitude": 107.02},
    {"name": "ankang", "latitude": 32.68, "longitude": 109.03},
    {"name": "weinan", "latitude": 34.50, "longitude": 109.51},
    {"name": "xianyang", "latitude": 34.33, "longitude": 108.71},
]


def weather_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return weather config with defaults."""
    cfg = dict(config.get("weather", {}))
    cfg.setdefault("enabled", False)
    cfg.setdefault("provider", "open_meteo_previous_runs")
    cfg.setdefault("cache_dir", "outputs/weather_cache")
    cfg.setdefault("timezone", "Asia/Shanghai")
    cfg.setdefault("variables", DEFAULT_WEATHER_VARIABLES)
    cfg.setdefault("forecast_leads_days", [1])
    cfg.setdefault("stations", DEFAULT_SHAANXI_STATIONS)
    cfg.setdefault("request_timeout_seconds", 30)
    cfg.setdefault("skip_failed_stations", True)
    return cfg


def fetch_open_meteo_previous_runs(
    station: Mapping[str, Any],
    variables: Iterable[str],
    leads_days: Iterable[int],
    start_date: str,
    end_date: str,
    timezone: str,
    cache_dir: str | Path,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """Fetch cached Open-Meteo historical forecast runs for one station.

    The requested variables use ``*_previous_dayN`` fields, which approximate
    day-ahead forecast availability better than historical reanalysis values.
    """
    name = str(station["name"])
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    lead_label = "-".join(str(int(day)) for day in leads_days)
    var_label = "-".join(str(var) for var in variables)
    path = cache_path / f"{name}_{start_date}_{end_date}_leads-{lead_label}_{abs(hash(var_label))}.json"

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        hourly = [
            f"{variable}_previous_day{int(day)}"
            for variable in variables
            for day in leads_days
        ]
        params = {
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly),
            "timezone": timezone,
        }
        url = "https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
            params
        )
        with urllib.request.urlopen(url, timeout=int(timeout_seconds)) as response:
            payload = json.load(response)
        path.write_text(json.dumps(payload), encoding="utf-8")

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"Open-Meteo response for {name} does not contain hourly time.")
    frame = pd.DataFrame({"Date_hour": pd.to_datetime(hourly["time"])})
    for variable in variables:
        for day in leads_days:
            source = f"{variable}_previous_day{int(day)}"
            frame[f"weather_{name}_{variable}_d{int(day)}"] = hourly.get(source)
        lead_values = [int(day) for day in leads_days]
        if 1 in lead_values and 2 in lead_values:
            frame[f"weather_{name}_{variable}_d1_d2"] = (
                frame[f"weather_{name}_{variable}_d1"]
                - frame[f"weather_{name}_{variable}_d2"]
            )
        if 1 in lead_values and 3 in lead_values:
            frame[f"weather_{name}_{variable}_d1_d3"] = (
                frame[f"weather_{name}_{variable}_d1"]
                - frame[f"weather_{name}_{variable}_d3"]
            )
    return frame


def fetch_open_meteo_archive(
    station: Mapping[str, Any],
    variables: Iterable[str],
    start_date: str,
    end_date: str,
    timezone: str,
    cache_dir: str | Path,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    """Fetch cached Open-Meteo historical weather reanalysis for one station."""
    name = str(station["name"])
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    var_label = "-".join(str(var) for var in variables)
    path = cache_path / f"{name}_{start_date}_{end_date}_archive_{abs(hash(var_label))}.json"

    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        params = {
            "latitude": float(station["latitude"]),
            "longitude": float(station["longitude"]),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(str(var) for var in variables),
            "timezone": timezone,
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=int(timeout_seconds)) as response:
            payload = json.load(response)
        path.write_text(json.dumps(payload), encoding="utf-8")

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"Open-Meteo archive response for {name} does not contain hourly time.")
    frame = pd.DataFrame({"Date_hour": pd.to_datetime(hourly["time"])})
    for variable in variables:
        frame[f"weather_actual_{name}_{variable}"] = hourly.get(str(variable))
    return frame


def build_weather_table(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Build a wide hourly weather forecast table for configured stations."""
    cfg = weather_config(config)
    if str(cfg.get("provider")) != "open_meteo_previous_runs":
        raise ValueError(f"Unsupported weather provider: {cfg.get('provider')}")
    variables = list(cfg["variables"])
    leads = [int(day) for day in cfg["forecast_leads_days"]]
    frames = []
    for station in cfg["stations"]:
        try:
            frames.append(
                fetch_open_meteo_previous_runs(
                    station,
                    variables=variables,
                    leads_days=leads,
                    start_date=start_date,
                    end_date=end_date,
                    timezone=str(cfg["timezone"]),
                    cache_dir=cfg["cache_dir"],
                    timeout_seconds=int(cfg.get("request_timeout_seconds", 30)),
                )
            )
        except Exception as exc:
            if not bool(cfg.get("skip_failed_stations", True)):
                raise
            warnings.warn(
                f"Skipping weather station {station.get('name', '<unknown>')}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if not frames:
        raise RuntimeError("No weather stations were fetched successfully.")
    weather = frames[0]
    for frame in frames[1:]:
        weather = weather.merge(frame, on="Date_hour", how="outer")

    station_names = [str(station["name"]) for station in cfg["stations"]]
    additions: Dict[str, pd.Series] = {}
    for variable in variables:
        for day in leads:
            cols = [
                f"weather_{name}_{variable}_d{int(day)}"
                for name in station_names
                if f"weather_{name}_{variable}_d{int(day)}" in weather.columns
            ]
            if not cols:
                continue
            values = weather[cols]
            additions[f"weather_sx_{variable}_d{int(day)}_mean"] = values.mean(axis=1)
            additions[f"weather_sx_{variable}_d{int(day)}_min"] = values.min(axis=1)
            additions[f"weather_sx_{variable}_d{int(day)}_max"] = values.max(axis=1)
            additions[f"weather_sx_{variable}_d{int(day)}_std"] = values.std(axis=1)

        for diff_suffix in ["d1_d2", "d1_d3"]:
            cols = [
                f"weather_{name}_{variable}_{diff_suffix}"
                for name in station_names
                if f"weather_{name}_{variable}_{diff_suffix}" in weather.columns
            ]
            if cols:
                additions[f"weather_sx_{variable}_{diff_suffix}_mean"] = weather[cols].mean(axis=1)

    if additions:
        weather = pd.concat([weather, pd.DataFrame(additions)], axis=1)
    return weather.sort_values("Date_hour").reset_index(drop=True)


def build_weather_actual_table(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Build a wide hourly historical-weather table for configured stations."""
    cfg = weather_config(config)
    variables = list(cfg["variables"])
    frames = []
    for station in cfg["stations"]:
        try:
            frames.append(
                fetch_open_meteo_archive(
                    station,
                    variables=variables,
                    start_date=start_date,
                    end_date=end_date,
                    timezone=str(cfg["timezone"]),
                    cache_dir=cfg["cache_dir"],
                    timeout_seconds=int(cfg.get("request_timeout_seconds", 30)),
                )
            )
        except Exception as exc:
            if not bool(cfg.get("skip_failed_stations", True)):
                raise
            warnings.warn(
                f"Skipping weather archive station {station.get('name', '<unknown>')}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    if not frames:
        raise RuntimeError("No weather archive stations were fetched successfully.")

    weather = frames[0]
    for frame in frames[1:]:
        weather = weather.merge(frame, on="Date_hour", how="outer")
    return weather.sort_values("Date_hour").reset_index(drop=True)


def merge_weather_features(
    features: pd.DataFrame,
    weather: pd.DataFrame,
    datetime_col: str = "Date",
) -> pd.DataFrame:
    """Merge hourly weather forecasts into a 15-minute feature table."""
    result = features.copy()
    result["Date_hour"] = pd.to_datetime(result[datetime_col]).dt.floor("h")
    merged = result.merge(weather, on="Date_hour", how="left")
    merged = merged.drop(columns=["Date_hour"])
    weather_cols = [col for col in merged.columns if col.startswith("weather_")]
    for col in weather_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return add_weather_derived_features(merged)


def add_weather_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add power-market weather proxies from forecast weather columns.

    These features keep the same availability as the forecast inputs. They are
    intended to expose winter heating demand, summer cooling demand, low-wind
    renewable risk, and low-radiation solar risk without requiring additional
    external data.
    """
    result = features.copy()
    for lead in [1, 2, 3]:
        temp = f"weather_sx_temperature_2m_d{lead}_mean"
        wind = f"weather_sx_wind_speed_10m_d{lead}_mean"
        radiation = f"weather_sx_shortwave_radiation_d{lead}_mean"
        cloud = f"weather_sx_cloud_cover_d{lead}_mean"
        precipitation = f"weather_sx_precipitation_d{lead}_mean"
        humidity = f"weather_sx_relative_humidity_2m_d{lead}_mean"

        if temp in result.columns:
            temp_values = pd.to_numeric(result[temp], errors="coerce")
            result[f"weather_sx_heating_degree_d{lead}"] = np.maximum(18.0 - temp_values, 0.0)
            result[f"weather_sx_cooling_degree_d{lead}"] = np.maximum(temp_values - 24.0, 0.0)
            result[f"weather_sx_cold_stress_d{lead}"] = np.maximum(5.0 - temp_values, 0.0)
        if wind in result.columns:
            wind_values = pd.to_numeric(result[wind], errors="coerce")
            result[f"weather_sx_low_wind_risk_d{lead}"] = np.maximum(3.0 - wind_values, 0.0)
            result[f"weather_sx_wind_power_proxy_d{lead}"] = np.power(
                np.clip(wind_values, 0.0, 25.0),
                3,
            )
        if radiation in result.columns:
            radiation_values = pd.to_numeric(result[radiation], errors="coerce")
            result[f"weather_sx_low_solar_risk_d{lead}"] = np.maximum(
                120.0 - radiation_values,
                0.0,
            )
            result[f"weather_sx_solar_power_proxy_d{lead}"] = np.maximum(
                radiation_values,
                0.0,
            )
        if cloud in result.columns and radiation in result.columns:
            result[f"weather_sx_cloud_solar_stress_d{lead}"] = (
                pd.to_numeric(result[cloud], errors="coerce")
                * result[f"weather_sx_low_solar_risk_d{lead}"]
            )
        if temp in result.columns and wind in result.columns:
            result[f"weather_sx_cold_low_wind_stress_d{lead}"] = (
                result[f"weather_sx_cold_stress_d{lead}"]
                * result[f"weather_sx_low_wind_risk_d{lead}"]
            )
        if temp in result.columns and humidity in result.columns:
            result[f"weather_sx_heat_humidity_stress_d{lead}"] = (
                result[f"weather_sx_cooling_degree_d{lead}"]
                * pd.to_numeric(result[humidity], errors="coerce")
                / 100.0
            )
        if precipitation in result.columns and radiation in result.columns:
            result[f"weather_sx_rain_solar_stress_d{lead}"] = (
                pd.to_numeric(result[precipitation], errors="coerce")
                * result[f"weather_sx_low_solar_risk_d{lead}"]
            )

    for col in [col for col in result.columns if col.startswith("weather_sx_")]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def add_weather_error_history_features(
    features: pd.DataFrame,
    actual_weather: pd.DataFrame,
    datetime_col: str = "Date",
    forecast_lead_day: int = 1,
    windows_days: Iterable[int] = (1, 3, 7),
) -> pd.DataFrame:
    """Add leakage-safe lagged forecast-error features from weather actuals.

    The actual weather of the target hour is merged only to construct historical
    forecast errors, then removed. Each feature is shifted by whole days at the
    same hour, so a day-ahead forecast for day d only sees weather forecast
    errors that were already observable before day d.
    """
    result = features.copy()
    result["Date_hour"] = pd.to_datetime(result[datetime_col]).dt.floor("h")
    actual = actual_weather.copy()
    actual["Date_hour"] = pd.to_datetime(actual["Date_hour"])
    merged = result.merge(actual, on="Date_hour", how="left")

    actual_cols = [col for col in merged.columns if col.startswith("weather_actual_")]
    additions: Dict[str, pd.Series] = {}
    error_base_names: List[str] = []
    for actual_col in actual_cols:
        suffix = actual_col.removeprefix("weather_actual_")
        forecast_col = f"weather_{suffix}_d{int(forecast_lead_day)}"
        if forecast_col not in merged.columns:
            continue
        error = (
            pd.to_numeric(merged[actual_col], errors="coerce")
            - pd.to_numeric(merged[forecast_col], errors="coerce")
        )
        shifted = error.groupby(merged["Date_hour"].dt.hour, group_keys=False).shift(4)
        base_name = f"weather_error_{suffix}_d{int(forecast_lead_day)}"
        error_base_names.append(base_name)
        additions[f"{base_name}_lag1d"] = shifted
        for window in windows_days:
            slots = int(window) * 4
            rolled = shifted.groupby(merged["Date_hour"].dt.hour, group_keys=False).rolling(slots)
            additions[f"{base_name}_roll{int(window)}d_mean"] = (
                rolled.mean().reset_index(level=0, drop=True)
            )
            additions[f"{base_name}_roll{int(window)}d_abs_mean"] = (
                shifted.abs()
                .groupby(merged["Date_hour"].dt.hour, group_keys=False)
                .rolling(slots)
                .mean()
                .reset_index(level=0, drop=True)
            )

    if additions:
        additions_frame = pd.DataFrame(additions, index=merged.index)
        aggregate_additions: Dict[str, pd.Series] = {}
        variable_names = sorted(
            {
                "_".join(name.removeprefix("weather_error_").split("_")[1:-1])
                for name in error_base_names
            }
        )
        for variable in variable_names:
            cols = [
                f"{name}_lag1d"
                for name in error_base_names
                if name.endswith(f"_{variable}_d{int(forecast_lead_day)}")
            ]
            if cols:
                aggregate_additions[
                    f"weather_error_sx_{variable}_d{int(forecast_lead_day)}_lag1d_mean"
                ] = additions_frame[cols].mean(axis=1)
        if aggregate_additions:
            additions_frame = pd.concat(
                [additions_frame, pd.DataFrame(aggregate_additions, index=merged.index)],
                axis=1,
            )
        for col in additions_frame.columns:
            additions_frame[col] = pd.to_numeric(additions_frame[col], errors="coerce")
        merged = pd.concat([merged, additions_frame], axis=1)

    return merged.drop(columns=["Date_hour", *actual_cols], errors="ignore")


def weather_date_bounds(features: pd.DataFrame, datetime_col: str = "Date") -> tuple[str, str]:
    """Return inclusive date strings for weather requests."""
    timestamps = pd.to_datetime(features[datetime_col])
    start = timestamps.min().date().isoformat()
    end = timestamps.max().date().isoformat()
    return start, end

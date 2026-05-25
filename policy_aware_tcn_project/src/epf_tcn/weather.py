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
    return merged


def weather_date_bounds(features: pd.DataFrame, datetime_col: str = "Date") -> tuple[str, str]:
    """Return inclusive date strings for weather requests."""
    timestamps = pd.to_datetime(features[datetime_col])
    start = timestamps.min().date().isoformat()
    end = timestamps.max().date().isoformat()
    return start, end

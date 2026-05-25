"""Feature engineering for day-ahead electricity price forecasting."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from epf.data import add_time_index_columns


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while avoiding division by zero."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add power-market meaning features."""
    result = df.copy()
    load = result["统一负荷预测"]
    renewable = result["统一新能源预测"]
    generation = result["发电总出力预测"]

    result["净负荷"] = load - renewable
    result["供需裕度"] = generation - load
    result["新能源占比"] = safe_divide(renewable, load)
    result["竞价空间占比"] = safe_divide(result["竞价空间"], load)
    result["联络线占比"] = safe_divide(result["联络线计划"], load)
    return result


def add_cyclic_features(df: pd.DataFrame, expected_slots: int = 96) -> pd.DataFrame:
    """Add sine/cosine cyclic encodings for slot and month."""
    result = df.copy()
    result["slot_sin"] = np.sin(2 * np.pi * result["slot"] / expected_slots)
    result["slot_cos"] = np.cos(2 * np.pi * result["slot"] / expected_slots)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    return result


def add_price_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_days: List[int],
) -> pd.DataFrame:
    """Add same-slot historical price lag features by date and slot."""
    result = df.copy()
    base = result[["date", "slot", target_col]].copy()

    for lag in lag_days:
        lagged = base.copy()
        lagged["date"] = lagged["date"] + pd.Timedelta(days=lag)
        lagged = lagged.rename(columns={target_col: f"price_lag_{lag}d"})
        result = result.merge(lagged, on=["date", "slot"], how="left")

    return result


def add_rolling_price_features(
    df: pd.DataFrame,
    target_col: str,
    windows: List[int],
) -> pd.DataFrame:
    """Add rolling same-slot price statistics using previous days only."""
    result = df.sort_values(["slot", "date"]).copy()
    group = result.groupby("slot", group_keys=False)[target_col]
    shifted = group.shift(1)

    for window in windows:
        rolled = shifted.groupby(result["slot"], group_keys=False).rolling(window)
        result[f"price_roll_{window}d_mean"] = rolled.mean().reset_index(level=0, drop=True)
        result[f"price_roll_{window}d_median"] = rolled.median().reset_index(level=0, drop=True)

        if window == 7:
            result[f"price_roll_{window}d_std"] = rolled.std().reset_index(level=0, drop=True)
            result[f"price_roll_{window}d_max"] = rolled.max().reset_index(level=0, drop=True)
            result[f"price_roll_{window}d_min"] = rolled.min().reset_index(level=0, drop=True)

    return result.sort_values(["date", "slot"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Build the complete feature table."""
    target = config["columns"]["target"]
    expected_slots = config["data"].get("expected_slots_per_day", 96)

    result = add_time_index_columns(df, config)

    if config["features"].get("include_market_features", True):
        result = add_market_features(result)

    if config["features"].get("include_cyclic_slot_features", True):
        result = add_cyclic_features(result, expected_slots=expected_slots)

    result = add_price_lag_features(
        result,
        target_col=target,
        lag_days=config["features"].get("price_lags_days", [1, 2, 7, 14]),
    )
    result = add_rolling_price_features(
        result,
        target_col=target,
        windows=config["features"].get("rolling_windows_days", [3, 7, 14]),
    )

    return result


def get_feature_columns(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return numeric model feature columns."""
    target = config["columns"]["target"]
    dt_col = config["columns"]["datetime"]
    excluded = {
        target,
        f"{target}_raw",
        f"{target}_clean_reason",
        dt_col,
        "date",
        "minute",
    }
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    return feature_cols

"""Feature engineering for day-ahead electricity price forecasting."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from epf_tcn.data import add_time_index_columns


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


def add_floor_price_features(
    df: pd.DataFrame,
    target_col: str,
    floor_price: float = 40.0,
    windows: List[int] | None = None,
    long_run_slots: int = 16,
    datetime_col: str | None = None,
    slot_minutes: int = 15,
    expected_slots: int = 96,
) -> pd.DataFrame:
    """Add features that describe recent floor-price persistence.

    ``floor_ratio_*`` is shifted by same-slot previous days, so it is safe for
    the target-day known-future branch. Direct current-price columns such as
    ``is_floor_price`` and run-length columns are history-only model context.
    """
    if windows is None:
        windows = [1, 3, 7]

    result = df.sort_values(["date", "slot"]).copy()
    is_floor = result[target_col].eq(float(floor_price)).astype(float)
    result["is_floor_price"] = is_floor

    if datetime_col and datetime_col in result.columns:
        continuous_slot = result[datetime_col].diff().eq(
            pd.Timedelta(minutes=int(slot_minutes))
        )
    else:
        previous_date = result["date"].shift()
        previous_slot = result["slot"].shift()
        same_day_next_slot = result["date"].eq(previous_date) & result["slot"].eq(
            previous_slot + 1
        )
        next_day_first_slot = (
            result["date"].eq(previous_date + pd.Timedelta(days=1))
            & result["slot"].eq(0)
            & previous_slot.eq(int(expected_slots) - 1)
        )
        continuous_slot = same_day_next_slot | next_day_first_slot
    same_floor = is_floor.eq(1.0) & is_floor.shift().eq(1.0)
    floor_run_id = (~(continuous_slot & same_floor)).cumsum()
    floor_run = is_floor.groupby(floor_run_id).cumsum()
    result["floor_run_slots"] = floor_run.where(is_floor.eq(1.0), 0.0)
    result["prev_floor_run_slots"] = result["floor_run_slots"].shift(1).where(
        continuous_slot,
        0.0,
    ).fillna(0.0)
    result["prev_long_floor_run"] = (
        result["prev_floor_run_slots"] >= float(long_run_slots)
    ).astype(float)

    shifted_floor = result.groupby("slot", group_keys=False)["is_floor_price"].shift(1)
    for window in windows:
        rolled = shifted_floor.groupby(result["slot"], group_keys=False).rolling(window)
        ratio = rolled.mean().reset_index(level=0, drop=True)
        result[f"floor_ratio_{window}d"] = ratio

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
    if config["features"].get("include_floor_features", True):
        result = add_floor_price_features(
            result,
            target_col=target,
            floor_price=float(config["features"].get("floor_price", 40.0)),
            windows=config["features"].get("floor_run_windows_days", [1, 3, 7]),
            long_run_slots=int(config["features"].get("long_floor_run_slots", 16)),
            datetime_col=config["columns"].get("datetime"),
            slot_minutes=int(config["data"].get("slot_minutes", 15)),
            expected_slots=int(expected_slots),
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

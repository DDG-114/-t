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


CN_HOLIDAY_RANGES = [
    ("2024-01-01", "2024-01-01"),
    ("2024-02-10", "2024-02-17"),
    ("2024-04-04", "2024-04-06"),
    ("2024-05-01", "2024-05-05"),
    ("2024-06-08", "2024-06-10"),
    ("2024-09-15", "2024-09-17"),
    ("2024-10-01", "2024-10-07"),
    ("2025-01-01", "2025-01-01"),
    ("2025-01-28", "2025-02-04"),
    ("2025-04-04", "2025-04-06"),
    ("2025-05-01", "2025-05-05"),
    ("2025-05-31", "2025-06-02"),
    ("2025-10-01", "2025-10-08"),
]
CN_MAKEUP_WORKDAYS = [
    "2024-02-04",
    "2024-02-18",
    "2024-04-07",
    "2024-04-28",
    "2024-05-11",
    "2024-09-14",
    "2024-09-29",
    "2024-10-12",
    "2025-01-26",
    "2025-02-08",
    "2025-04-27",
    "2025-09-28",
    "2025-10-11",
]


def _date_set_from_ranges(ranges: List[tuple[str, str]]) -> set[pd.Timestamp]:
    dates: set[pd.Timestamp] = set()
    for start, end in ranges:
        for date in pd.date_range(start, end, freq="D"):
            dates.add(pd.Timestamp(date).normalize())
    return dates


def add_china_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free China holiday and adjusted-workday calendar features."""
    result = df.copy()
    normalized_date = pd.to_datetime(result["date"]).dt.normalize()
    holidays = _date_set_from_ranges(CN_HOLIDAY_RANGES)
    makeup_workdays = {pd.Timestamp(date).normalize() for date in CN_MAKEUP_WORKDAYS}
    holiday_index = pd.DatetimeIndex(sorted(holidays))

    is_holiday = normalized_date.isin(holidays)
    is_makeup = normalized_date.isin(makeup_workdays)
    is_weekend = result.get("is_weekend", normalized_date.dt.dayofweek.ge(5)).astype(bool)
    is_workday = (~is_holiday) & (~is_weekend | is_makeup)

    result["is_cn_holiday"] = is_holiday.astype(float)
    result["is_cn_makeup_workday"] = is_makeup.astype(float)
    result["is_cn_workday"] = is_workday.astype(float)
    result["is_cn_rest_day"] = (~is_workday).astype(float)

    days_to_holiday = []
    days_since_holiday = []
    for date in normalized_date:
        future = holiday_index[holiday_index >= date]
        past = holiday_index[holiday_index <= date]
        days_to_holiday.append(
            float((future[0] - date).days) if len(future) else np.nan
        )
        days_since_holiday.append(
            float((date - past[-1]).days) if len(past) else np.nan
        )
    result["days_to_cn_holiday"] = np.clip(days_to_holiday, 0, 30)
    result["days_since_cn_holiday"] = np.clip(days_since_holiday, 0, 30)
    result["near_cn_holiday_3d"] = (
        (result["days_to_cn_holiday"] <= 3)
        | (result["days_since_cn_holiday"] <= 3)
    ).astype(float)
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


def _quantile_label(level: float) -> str:
    scaled = float(level) * 100
    if abs(scaled - round(scaled)) < 1e-9:
        return f"q{int(round(scaled)):02d}"
    return f"q{int(round(float(level) * 1000)):03d}"


def add_exogenous_quantile_features(
    df: pd.DataFrame,
    source_cols: List[str],
    windows: List[int],
    quantile_levels: List[float],
    include_spreads: bool = True,
    include_derived: bool = True,
) -> pd.DataFrame:
    """Add same-slot historical quantile proxies for fundamental forecasts.

    The bundled dataset contains point forecasts for fundamental variables, but
    not the realized load/renewable observations needed for true forecast-error
    quantile postprocessing. These features therefore use only previous same-slot
    forecast distributions as a leakage-safe proxy for fundamental uncertainty.
    """
    result = df.sort_values(["slot", "date"]).copy()
    existing_cols = [
        col
        for col in source_cols
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col])
    ]
    levels = sorted({float(level) for level in quantile_levels})
    labels = {level: _quantile_label(level) for level in levels}

    for col in existing_cols:
        shifted = result.groupby("slot", group_keys=False)[col].shift(1)
        for window in windows:
            rolled = shifted.groupby(result["slot"], group_keys=False).rolling(int(window))
            for level in levels:
                label = labels[level]
                result[f"{col}_{label}_{int(window)}d"] = (
                    rolled.quantile(level).reset_index(level=0, drop=True)
                )
            if include_spreads and 0.1 in labels and 0.9 in labels:
                low = f"{col}_{labels[0.1]}_{int(window)}d"
                high = f"{col}_{labels[0.9]}_{int(window)}d"
                if low in result.columns and high in result.columns:
                    result[f"{col}_iqr_{int(window)}d"] = result[high] - result[low]

    if include_derived:
        result = add_derived_quantile_features(
            result,
            windows=windows,
            quantile_levels=levels,
            labels=labels,
        )

    return result.sort_values(["date", "slot"]).reset_index(drop=True)


def add_derived_quantile_features(
    df: pd.DataFrame,
    windows: List[int],
    quantile_levels: List[float],
    labels: Dict[float, str],
) -> pd.DataFrame:
    """Add residual-load and supply-margin quantile proxy features."""
    result = df.copy()
    load_col = "统一负荷预测"
    renewable_col = "统一新能源预测"
    generation_col = "发电总出力预测"

    for window in windows:
        window = int(window)
        for level in quantile_levels:
            label = labels[level]
            inverse_label = labels.get(round(1.0 - level, 10))

            if inverse_label:
                load_q = f"{load_col}_{label}_{window}d"
                renewable_inv_q = f"{renewable_col}_{inverse_label}_{window}d"
                generation_q = f"{generation_col}_{label}_{window}d"
                load_inv_q = f"{load_col}_{inverse_label}_{window}d"

                if load_q in result.columns and renewable_inv_q in result.columns:
                    result[f"净负荷_{label}_{window}d"] = result[load_q] - result[renewable_inv_q]
                if generation_q in result.columns and load_inv_q in result.columns:
                    result[f"供需裕度_{label}_{window}d"] = result[generation_q] - result[load_inv_q]

        q10 = labels.get(0.1)
        q90 = labels.get(0.9)
        if q10 and q90:
            for base in ["净负荷", "供需裕度"]:
                low = f"{base}_{q10}_{window}d"
                high = f"{base}_{q90}_{window}d"
                if low in result.columns and high in result.columns:
                    result[f"{base}_iqr_{window}d"] = result[high] - result[low]
    return result


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

    if config["features"].get("include_china_calendar_features", False):
        result = add_china_calendar_features(result)

    if config["features"].get("include_exogenous_quantile_features", False):
        result = add_exogenous_quantile_features(
            result,
            source_cols=config["features"].get(
                "exogenous_quantile_source_columns",
                config["columns"].get("exogenous", []),
            ),
            windows=config["features"].get("exogenous_quantile_windows_days", [7, 14, 28]),
            quantile_levels=config["features"].get(
                "exogenous_quantile_levels",
                [0.1, 0.5, 0.9],
            ),
            include_spreads=bool(
                config["features"].get("include_exogenous_quantile_spreads", True)
            ),
            include_derived=bool(
                config["features"].get("include_derived_quantile_features", True)
            ),
        )

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

"""Data loading, cleaning, and profiling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw CSV data and parse timestamp column."""
    path = Path(config["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    encoding = config.get("data", {}).get("encoding", "utf-8-sig")
    dt_col = config["columns"]["datetime"]

    df = pd.read_csv(path, encoding=encoding)
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).copy()
    numeric_cols = [config["columns"]["target"], *config["columns"].get("exogenous", [])]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(dt_col).drop_duplicates(subset=[dt_col])
    df = df.reset_index(drop=True)
    df = clean_target_prices(df, config)
    return df


def clean_target_prices(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Flag abnormal target prices by setting them to NaN.

    The raw target is kept in ``<target>_raw`` and cleaning reasons are stored in
    ``<target>_clean_reason``. Feature lags and supervised training then use the
    cleaned target, preventing cap/placeholder and long constant labels from
    leaking into the model.
    """
    cleaning = config.get("data", {}).get("target_cleaning", {})
    if not cleaning.get("enabled", False):
        return df

    result = df.copy()
    dt_col = config["columns"]["datetime"]
    target = config["columns"]["target"]
    slot_minutes = config["data"].get("slot_minutes", 15)
    constant_run_min_length = cleaning.get("constant_run_min_length")
    cap_values = [float(value) for value in cleaning.get("cap_values_as_missing", [])]
    constant_run_exempt_values = {
        float(value) for value in cleaning.get("constant_run_exempt_values", [])
    }

    raw_col = f"{target}_raw"
    reason_col = f"{target}_clean_reason"
    result[raw_col] = result[target]
    result[reason_col] = ""

    if cap_values:
        cap_mask = result[target].isin(cap_values)
        result.loc[cap_mask, reason_col] = "cap_value"

    if constant_run_min_length:
        continuous = result[dt_col].diff().eq(pd.Timedelta(minutes=slot_minutes))
        same_price = result[target].eq(result[target].shift())
        same_continuous = continuous & same_price & result[target].notna()
        run_id = (~same_continuous).cumsum()
        run_length = result.groupby(run_id)[target].transform("size")
        exempt_mask = result[target].isin(constant_run_exempt_values)
        constant_mask = (
            result[target].notna()
            & ~exempt_mask
            & (run_length >= constant_run_min_length)
        )
        result.loc[constant_mask & result[reason_col].eq(""), reason_col] = (
            "long_constant_run"
        )
        result.loc[constant_mask & result[reason_col].eq("cap_value"), reason_col] = (
            "cap_value;long_constant_run"
        )

    clean_mask = result[reason_col].ne("")
    result.loc[clean_mask, target] = np.nan
    return result


def add_time_index_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add date, slot, hour, and calendar columns."""
    dt_col = config["columns"]["datetime"]
    result = df.copy()

    result["date"] = result[dt_col].dt.floor("D")
    result["hour"] = result[dt_col].dt.hour
    result["minute"] = result[dt_col].dt.minute
    result["slot"] = (result["hour"] * 60 + result["minute"]) // config["data"].get(
        "slot_minutes", 15
    )
    result["day_of_week"] = result[dt_col].dt.dayofweek
    result["month"] = result[dt_col].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result


def profile_data(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a compact data profile dictionary."""
    dt_col = config["columns"]["datetime"]
    target = config["columns"]["target"]
    expected_slots = config["data"].get("expected_slots_per_day", 96)

    by_day = df.groupby(df[dt_col].dt.floor("D")).size()
    profile = {
        "rows": int(len(df)),
        "start": str(df[dt_col].min()),
        "end": str(df[dt_col].max()),
        "missing_values": df.isna().sum().to_dict(),
        "days": int(by_day.shape[0]),
        "incomplete_days": {str(k.date()): int(v) for k, v in by_day[by_day != expected_slots].items()},
    }

    if target in df.columns:
        profile["target_describe"] = df[target].describe().to_dict()
        profile["target_top_values"] = {str(k): int(v) for k, v in df[target].value_counts(dropna=False).head(10).items()}
        reason_col = f"{target}_clean_reason"
        raw_col = f"{target}_raw"
        if reason_col in df.columns:
            profile["target_cleaning"] = {
                "cleaned_rows": int(df[reason_col].ne("").sum()),
                "reasons": {
                    str(k): int(v)
                    for k, v in df.loc[df[reason_col].ne(""), reason_col]
                    .value_counts()
                    .items()
                },
            }
        if raw_col in df.columns:
            profile["raw_target_top_values"] = {
                str(k): int(v)
                for k, v in df[raw_col].value_counts(dropna=False).head(10).items()
            }

    return profile


def find_forecast_frame(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Return the 24h forecast target day from the feature table.

    The configured date is used when present. Otherwise the latest day with at
    least one missing target value is selected. This mirrors the bundled GS.csv,
    where the final day contains known exogenous forecasts but missing prices.
    """
    target = config["columns"]["target"]
    forecast_config = config.get("forecast", {})
    configured_date = forecast_config.get("target_date")

    if configured_date:
        target_day = pd.Timestamp(configured_date)
    else:
        missing_target = df[df[target].isna()]
        if missing_target.empty:
            target_day = pd.Timestamp(df["date"].max())
        else:
            target_day = pd.Timestamp(missing_target["date"].max())

    result = df[pd.to_datetime(df["date"]) == target_day].copy()
    if result.empty:
        raise ValueError(f"No rows found for forecast target date: {target_day.date()}")
    return result.sort_values("slot").reset_index(drop=True)

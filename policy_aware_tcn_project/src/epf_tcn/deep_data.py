"""Daily-window data utilities for deep multi-horizon EPF models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyWindow:
    """One supervised day-ahead sample.

    ``x_hist`` contains only days before ``target_day``. ``x_fut`` contains
    covariates for the delivery day and must be restricted to variables known
    before market clearing.
    """

    target_day: pd.Timestamp
    x_hist: np.ndarray
    x_fut: np.ndarray
    x_static: np.ndarray
    anchor: np.ndarray
    y: np.ndarray | None
    y_mask: np.ndarray | None
    lower_bound: float
    upper_bound: float
    policy_regime: float
    sample_weight: float = 1.0
    floor_context: np.ndarray | None = None


@dataclass(frozen=True)
class WindowConfig:
    """Shape and policy settings for daily-window construction."""

    history_days: int = 14
    horizon: int = 96
    freq_per_day: int = 96
    target_col: str = "Price"
    date_col: str = "date"
    slot_col: str = "slot"


@dataclass(frozen=True)
class DeepFeatureSpec:
    """Column groups used by the sequence model."""

    hist_cols: List[str]
    fut_cols: List[str]
    static_cols: List[str]


@dataclass
class WindowNormalizer:
    """Per-feature normalizer fitted only on training windows."""

    hist_mean: np.ndarray
    hist_std: np.ndarray
    fut_mean: np.ndarray
    fut_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    anchor_fill: float
    add_missing_indicators: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "hist_mean": self.hist_mean.tolist(),
            "hist_std": self.hist_std.tolist(),
            "fut_mean": self.fut_mean.tolist(),
            "fut_std": self.fut_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "anchor_fill": float(self.anchor_fill),
            "add_missing_indicators": bool(self.add_missing_indicators),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WindowNormalizer":
        """Rehydrate a normalizer saved by :meth:`to_dict`."""
        return cls(
            hist_mean=np.asarray(payload["hist_mean"], dtype=np.float32),
            hist_std=np.asarray(payload["hist_std"], dtype=np.float32),
            fut_mean=np.asarray(payload["fut_mean"], dtype=np.float32),
            fut_std=np.asarray(payload["fut_std"], dtype=np.float32),
            static_mean=np.asarray(payload["static_mean"], dtype=np.float32),
            static_std=np.asarray(payload["static_std"], dtype=np.float32),
            anchor_fill=float(payload["anchor_fill"]),
            add_missing_indicators=bool(payload.get("add_missing_indicators", True)),
        )


MARKET_DERIVED_COLS = [
    "净负荷",
    "供需裕度",
    "新能源占比",
    "竞价空间占比",
    "联络线占比",
]

QUANTILE_CONTEXT_PREFIXES = [
    "发电总出力预测_",
    "竞价空间_",
    "统一负荷预测_",
    "抽蓄_",
    "统一新能源预测_",
    "联络线计划_",
    "净负荷_",
    "供需裕度_",
]

WEATHER_CONTEXT_PREFIXES = [
    "weather_",
    "weather_error_",
]

EXTERNAL_ANCHOR_COPY_COLS = [
    "base_pred",
    "state_pred",
    "residual_pred",
    "floor_probability",
    "high_probability",
    "cap_probability",
]

TIME_COLS = [
    "slot",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "slot_sin",
    "slot_cos",
    "month_sin",
    "month_cos",
]

FUTURE_EXCLUDED_COLS = {
    "is_floor_price",
    "floor_run_slots",
    "prev_floor_run_slots",
    "prev_long_floor_run",
}

STATIC_COLS = [
    "policy_regime",
    "price_lower_bound",
    "price_upper_bound",
    "is_zero_floor_regime",
]


def _existing_numeric_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [
        col
        for col in candidates
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]


def infer_deep_feature_spec(df: pd.DataFrame, config: Dict[str, Any]) -> DeepFeatureSpec:
    """Infer report-aligned history, future, and static feature groups.

    The default follows report2: historical prices are available in the history
    encoder, while the future branch uses known delivery-day market forecasts,
    calendar features, and same-slot historical price context.
    """
    deep_cfg = config.get("deep_model", {})
    if deep_cfg.get("hist_cols") and deep_cfg.get("fut_cols"):
        return DeepFeatureSpec(
            hist_cols=list(deep_cfg["hist_cols"]),
            fut_cols=list(deep_cfg["fut_cols"]),
            static_cols=list(deep_cfg.get("static_cols", STATIC_COLS)),
        )

    target = config["columns"]["target"]
    exogenous = config["columns"].get("exogenous", [])
    weather_cols: List[str] = []
    if bool(deep_cfg.get("include_weather_features", False)):
        weather_prefixes = list(
            deep_cfg.get("weather_feature_prefixes", WEATHER_CONTEXT_PREFIXES)
        )
        weather_cols = _existing_numeric_columns(
            df,
            [
                col
                for col in df.columns
                if any(str(col).startswith(prefix) for prefix in weather_prefixes)
                and not str(col).startswith("weather_actual_")
            ],
        )
    external_anchor_cols = _external_anchor_feature_cols(df, config)
    price_context = [
        col
        for col in df.columns
        if col.startswith("price_lag_")
        or col.startswith("price_roll_")
        or col.startswith("floor_ratio_")
        or col.startswith("prev_floor_")
        or col in {"is_floor_price", "floor_run_slots", "prev_long_floor_run"}
    ]
    market_cols = _existing_numeric_columns(df, MARKET_DERIVED_COLS)
    quantile_cols = _existing_numeric_columns(
        df,
        [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in QUANTILE_CONTEXT_PREFIXES)
            and ("_q" in col or "_iqr_" in col)
        ],
    )
    time_cols = _existing_numeric_columns(df, TIME_COLS)
    exog_cols = _existing_numeric_columns(df, exogenous)
    context_cols = _existing_numeric_columns(df, price_context)
    hist_context_cols = context_cols
    fut_context_cols = [col for col in context_cols if col not in FUTURE_EXCLUDED_COLS]

    hist_cols = _dedupe(
        [
            target,
            *exog_cols,
            *market_cols,
            *quantile_cols,
            *weather_cols,
            *external_anchor_cols,
            *time_cols,
            *hist_context_cols,
        ]
    )
    fut_cols = _dedupe(
        [
            *exog_cols,
            *market_cols,
            *quantile_cols,
            *weather_cols,
            *external_anchor_cols,
            *time_cols,
            *fut_context_cols,
        ]
    )
    return DeepFeatureSpec(hist_cols=hist_cols, fut_cols=fut_cols, static_cols=STATIC_COLS)


def _dedupe(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def external_anchor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the optional external-anchor configuration."""
    return dict(config.get("deep_model", {}).get("external_anchor", {}))


def external_anchor_output_col(config: Dict[str, Any]) -> str:
    """Return the configured feature column used as an external price anchor."""
    return str(external_anchor_config(config).get("output_col", "external_anchor_price"))


def _external_anchor_feature_cols(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return configured external-anchor columns that are available and numeric."""
    cfg = external_anchor_config(config)
    if not bool(cfg.get("enabled", False)):
        return []
    output_col = str(cfg.get("output_col", "external_anchor_price"))
    cols = cfg.get("feature_cols")
    if cols is None:
        cols = [output_col, *cfg.get("copy_columns", EXTERNAL_ANCHOR_COPY_COLS)]
    return _existing_numeric_columns(df, _dedupe([str(col) for col in cols]))


def add_external_anchor_columns(features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Merge external model predictions into the deep-model feature table.

    The merged columns are treated as day-ahead-safe model priors. Missing
    timestamps are left as NaN so the anchor path can fall back to a configured
    baseline and the feature normalizer can expose missing indicators.
    """
    cfg = external_anchor_config(config)
    if not bool(cfg.get("enabled", False)):
        return features

    prediction_files = [Path(path) for path in cfg.get("prediction_files", [])]
    if not prediction_files:
        return features

    datetime_col = config["columns"].get("datetime", "Date")
    source_datetime_col = str(cfg.get("datetime_col", "Date"))
    source_prediction_col = str(cfg.get("prediction_col", "y_pred"))
    output_col = str(cfg.get("output_col", "external_anchor_price"))
    copy_cols = [str(col) for col in cfg.get("copy_columns", EXTERNAL_ANCHOR_COPY_COLS)]
    required = bool(cfg.get("required", True))

    frames: List[pd.DataFrame] = []
    for path in prediction_files:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"External anchor prediction file not found: {path}")
            continue
        pred = pd.read_csv(path)
        missing = [
            col
            for col in [source_datetime_col, source_prediction_col]
            if col not in pred.columns
        ]
        if missing:
            raise ValueError(f"External anchor file {path} is missing columns: {missing}")

        available_copy_cols = [col for col in copy_cols if col in pred.columns]
        keep_cols = _dedupe([source_datetime_col, source_prediction_col, *available_copy_cols])
        frame = pred[keep_cols].copy()
        frame["__anchor_datetime"] = pd.to_datetime(frame[source_datetime_col])
        frame[output_col] = pd.to_numeric(frame[source_prediction_col], errors="coerce")
        for col in available_copy_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame[["__anchor_datetime", output_col, *available_copy_cols]])

    if not frames:
        return features

    anchors = pd.concat(frames, ignore_index=True)
    anchors = anchors.sort_values("__anchor_datetime").drop_duplicates(
        "__anchor_datetime",
        keep="last",
    )
    result = features.copy()
    result[datetime_col] = pd.to_datetime(result[datetime_col])
    merge_cols = [col for col in anchors.columns if col != "__anchor_datetime"]
    result = result.drop(columns=[col for col in merge_cols if col in result.columns])
    result = result.merge(
        anchors,
        left_on=datetime_col,
        right_on="__anchor_datetime",
        how="left",
    )
    return result.drop(columns=["__anchor_datetime"])


def _floor_context_for_block(block: pd.DataFrame) -> np.ndarray:
    """Return safe future floor-price context from previous-day floor ratios."""
    cols = [
        col
        for col in block.columns
        if col.startswith("floor_ratio_") and pd.api.types.is_numeric_dtype(block[col])
    ]
    if not cols:
        return np.zeros(block.shape[0], dtype=np.float32)

    values = block[cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(block.shape[0], dtype=np.float32)
    context = np.where(finite, values, 0.0).max(axis=1)
    return np.clip(context, 0.0, 1.0).astype(np.float32)


def window_config_from_project(config: Dict[str, Any]) -> WindowConfig:
    """Create a :class:`WindowConfig` from the project YAML config."""
    deep_cfg = config.get("deep_model", {})
    return WindowConfig(
        history_days=int(deep_cfg.get("history_days", 14)),
        horizon=int(deep_cfg.get("horizon", config.get("data", {}).get("expected_slots_per_day", 96))),
        freq_per_day=int(config.get("data", {}).get("expected_slots_per_day", 96)),
        target_col=config["columns"]["target"],
        date_col="date",
        slot_col="slot",
    )


def year_bounds_from_config(config: Dict[str, Any]) -> Dict[int, tuple[float, float]]:
    """Return year-specific clipping and policy bounds."""
    default_min = float(config.get("model", {}).get("clip_prediction_min", 40.0))
    default_max = float(config.get("model", {}).get("clip_prediction_max", 1000.0))
    bounds_cfg = config.get("deep_model", {}).get("year_bounds", {})
    if not bounds_cfg:
        return {2025: (default_min, default_max), 2026: (0.0, default_max)}

    result: Dict[int, tuple[float, float]] = {}
    for year, bounds in bounds_cfg.items():
        if isinstance(bounds, Mapping):
            lo = float(bounds.get("min", default_min))
            hi = float(bounds.get("max", default_max))
        else:
            lo, hi = bounds
            lo = float(lo)
            hi = float(hi)
        result[int(year)] = (lo, hi)
    return result


def policy_regime_for_day(day: pd.Timestamp, config: Dict[str, Any]) -> float:
    """Map a target day to a numeric market-policy regime token."""
    regime_cfg = config.get("deep_model", {}).get("policy_regime_by_year", {})
    default = config.get("deep_model", {}).get("default_policy_regime", 0.0)
    return float(regime_cfg.get(str(day.year), regime_cfg.get(day.year, default)))


def static_features_for_day(day: pd.Timestamp, config: Dict[str, Any]) -> Dict[str, float]:
    """Build static policy features for one target day."""
    bounds = year_bounds_from_config(config)
    lower, upper = bounds.get(day.year, bounds.get(2025, (40.0, 1000.0)))
    return {
        "policy_regime": policy_regime_for_day(day, config),
        "price_lower_bound": float(lower),
        "price_upper_bound": float(upper),
        "is_zero_floor_regime": float(lower <= 0.0),
    }


def _complete_day_map(
    df: pd.DataFrame,
    cfg: WindowConfig,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Return complete delivery days sorted by slot."""
    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for day, group in df.groupby(cfg.date_col):
        day_ts = pd.Timestamp(day).normalize()
        group = group.sort_values(cfg.slot_col)
        if group.shape[0] != cfg.freq_per_day:
            continue
        if group[cfg.slot_col].nunique() != cfg.freq_per_day:
            continue
        result[day_ts] = group
    return result


def build_daily_windows(
    df: pd.DataFrame,
    config: Dict[str, Any],
    feature_spec: DeepFeatureSpec | None = None,
    target_days: Sequence[pd.Timestamp] | None = None,
    require_target: bool = True,
    min_target_slots: int | None = None,
) -> List[DailyWindow]:
    """Build direct 96-step day-ahead windows without random sampling.

    A sample is emitted only when the target day is complete and all historical
    days are the exact preceding calendar days. This keeps the training contract
    aligned with day-ahead deployment and prevents accidental date leakage.
    """
    cfg = window_config_from_project(config)
    spec = feature_spec or infer_deep_feature_spec(df, config)
    frame = df.copy()
    frame[cfg.date_col] = pd.to_datetime(frame[cfg.date_col]).dt.normalize()
    numeric_cols = _dedupe([*spec.hist_cols, *spec.fut_cols, cfg.target_col])
    existing_numeric = [col for col in numeric_cols if col in frame.columns]
    frame[existing_numeric] = frame[existing_numeric].replace([np.inf, -np.inf], np.nan)

    complete_days = _complete_day_map(frame, cfg)
    requested_days = (
        [pd.Timestamp(day).normalize() for day in target_days]
        if target_days is not None
        else sorted(complete_days)
    )
    if min_target_slots is None:
        min_target_slots = int(config.get("deep_model", {}).get("min_target_slots", 1))

    windows: List[DailyWindow] = []
    for target_day in requested_days:
        target_group = complete_days.get(target_day)
        if target_group is None:
            continue

        hist_days = [
            target_day - pd.Timedelta(days=offset)
            for offset in range(cfg.history_days, 0, -1)
        ]
        if any(day not in complete_days for day in hist_days):
            continue

        hist_block = pd.concat([complete_days[day] for day in hist_days])
        future_block = target_group
        y = None
        y_mask = None
        y_values = future_block[cfg.target_col].to_numpy(dtype=np.float32)
        observed = np.isfinite(y_values)
        if observed.any():
            y = y_values
            y_mask = observed.astype(np.float32)
        if require_target and int(observed.sum()) < min_target_slots:
            continue

        anchor = _anchor_for_target_day(
            complete_days=complete_days,
            hist_days=hist_days,
            target_group=target_group,
            cfg=cfg,
            config=config,
        )
        static_map = static_features_for_day(target_day, config)
        static_vec = np.asarray(
            [static_map[col] for col in spec.static_cols],
            dtype=np.float32,
        )
        lower = static_map["price_lower_bound"]
        upper = static_map["price_upper_bound"]
        floor_context = _floor_context_for_block(future_block)

        windows.append(
            DailyWindow(
                target_day=target_day,
                x_hist=hist_block[spec.hist_cols].to_numpy(dtype=np.float32),
                x_fut=future_block[spec.fut_cols].to_numpy(dtype=np.float32),
                x_static=static_vec,
                anchor=anchor,
                y=y,
                y_mask=y_mask,
                lower_bound=float(lower),
                upper_bound=float(upper),
                policy_regime=float(static_map["policy_regime"]),
                floor_context=floor_context,
            )
        )
    return windows


def _anchor_for_target_day(
    complete_days: Dict[pd.Timestamp, pd.DataFrame],
    hist_days: Sequence[pd.Timestamp],
    target_group: pd.DataFrame,
    cfg: WindowConfig,
    config: Dict[str, Any],
) -> np.ndarray:
    """Return the residual anchor used by the TCN output head."""
    deep_cfg = config.get("deep_model", {})
    prior_cfg = deep_cfg.get("supply_demand_prior", {})
    anchor_source = str(
        deep_cfg.get(
            "anchor_source",
            "supply_demand_prior" if prior_cfg.get("enabled", False) else "previous_day",
        )
    )
    if anchor_source == "external_column":
        anchor_col = external_anchor_output_col(config)
        if anchor_col not in target_group.columns:
            raise ValueError(
                f"deep_model.anchor_source=external_column requires feature column: {anchor_col}"
            )
        values = target_group[anchor_col].to_numpy(dtype=np.float32)
        if np.isfinite(values).all():
            return values
        fallback_source = str(external_anchor_config(config).get("fallback_source", "previous_day"))
        fallback = _anchor_from_source(
            fallback_source,
            complete_days=complete_days,
            hist_days=hist_days,
            target_group=target_group,
            cfg=cfg,
            config=config,
        )
        return np.where(np.isfinite(values), values, fallback).astype(np.float32)
    return _anchor_from_source(
        anchor_source,
        complete_days=complete_days,
        hist_days=hist_days,
        target_group=target_group,
        cfg=cfg,
        config=config,
    )


def _anchor_from_source(
    anchor_source: str,
    complete_days: Dict[pd.Timestamp, pd.DataFrame],
    hist_days: Sequence[pd.Timestamp],
    target_group: pd.DataFrame,
    cfg: WindowConfig,
    config: Dict[str, Any],
) -> np.ndarray:
    """Return an anchor from one non-external source."""
    deep_cfg = config.get("deep_model", {})
    prior_cfg = deep_cfg.get("supply_demand_prior", {})
    if anchor_source == "previous_day":
        previous_day = complete_days[hist_days[-1]]
        return previous_day[cfg.target_col].to_numpy(dtype=np.float32)
    if anchor_source == "supply_demand_prior":
        prior_col = str(prior_cfg.get("output_col", "sd_prior_price"))
        if prior_col not in target_group.columns:
            raise ValueError(
                f"deep_model.anchor_source=supply_demand_prior requires feature column: {prior_col}"
            )
        return target_group[prior_col].to_numpy(dtype=np.float32)
    raise ValueError(f"Unsupported deep_model.anchor_source: {anchor_source}")


def filter_windows_by_date_range(
    windows: Sequence[DailyWindow],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> List[DailyWindow]:
    """Select windows by target-day range, inclusive."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    return [
        window
        for window in windows
        if start_ts <= pd.Timestamp(window.target_day).normalize() <= end_ts
    ]


def fit_window_normalizer(
    windows: Sequence[DailyWindow],
    add_missing_indicators: bool = True,
) -> WindowNormalizer:
    """Fit input normalizers from training windows only."""
    if not windows:
        raise ValueError("Cannot fit deep-window normalizer with no training windows.")

    hist = np.concatenate([window.x_hist for window in windows], axis=0)
    fut = np.concatenate([window.x_fut for window in windows], axis=0)
    static = np.stack([window.x_static for window in windows], axis=0)
    anchors = np.concatenate([window.anchor for window in windows], axis=0)
    targets = np.concatenate([window.y for window in windows if window.y is not None], axis=0)
    anchor_fill = float(np.nanmean(np.concatenate([anchors, targets])))
    if not np.isfinite(anchor_fill):
        anchor_fill = 0.0

    return WindowNormalizer(
        hist_mean=_nanmean(hist),
        hist_std=_nanstd(hist),
        fut_mean=_nanmean(fut),
        fut_std=_nanstd(fut),
        static_mean=_nanmean(static),
        static_std=_nanstd(static),
        anchor_fill=anchor_fill,
        add_missing_indicators=add_missing_indicators,
    )


def _nanmean(array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(array)
    count = valid.sum(axis=0)
    summed = np.where(valid, array, 0.0).sum(axis=0)
    mean = np.divide(
        summed,
        count,
        out=np.zeros_like(summed, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)
    return np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)


def _nanstd(array: np.ndarray) -> np.ndarray:
    mean = _nanmean(array)
    valid = np.isfinite(array)
    count = valid.sum(axis=0)
    centered = np.where(valid, array - mean, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        count,
        out=np.ones_like(mean, dtype=np.float64),
        where=count > 0,
    )
    std = np.sqrt(variance).astype(np.float32)
    return np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0).astype(np.float32)


def _normalize_with_mask(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    add_missing_indicators: bool,
) -> np.ndarray:
    missing = ~np.isfinite(array)
    normalized = (array - mean) / std
    normalized = np.where(missing, 0.0, normalized)
    if not add_missing_indicators:
        return normalized.astype(np.float32)
    return np.concatenate([normalized, missing.astype(np.float32)], axis=-1).astype(np.float32)


def transform_windows(
    windows: Sequence[DailyWindow],
    normalizer: WindowNormalizer,
    require_target: bool = True,
) -> Dict[str, np.ndarray]:
    """Transform windows into dense arrays ready for a torch Dataset."""
    selected = [window for window in windows if window.y is not None or not require_target]
    if not selected:
        raise ValueError("No windows available for transformation.")

    x_hist = np.stack(
        [
            _normalize_with_mask(
                window.x_hist,
                normalizer.hist_mean,
                normalizer.hist_std,
                normalizer.add_missing_indicators,
            )
            for window in selected
        ],
        axis=0,
    )
    x_fut = np.stack(
        [
            _normalize_with_mask(
                window.x_fut,
                normalizer.fut_mean,
                normalizer.fut_std,
                normalizer.add_missing_indicators,
            )
            for window in selected
        ],
        axis=0,
    )
    x_static = np.stack(
        [
            np.nan_to_num(
                (window.x_static - normalizer.static_mean) / normalizer.static_std,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for window in selected
        ],
        axis=0,
    ).astype(np.float32)
    anchor = np.stack(
        [
            np.nan_to_num(
                window.anchor,
                nan=normalizer.anchor_fill,
                posinf=normalizer.anchor_fill,
                neginf=normalizer.anchor_fill,
            )
            for window in selected
        ],
        axis=0,
    ).astype(np.float32)
    y = None
    y_mask = None
    if all(window.y is not None for window in selected):
        y = np.stack(
            [
                np.nan_to_num(
                    window.y,
                    nan=normalizer.anchor_fill,
                    posinf=normalizer.anchor_fill,
                    neginf=normalizer.anchor_fill,
                )
                for window in selected
            ],
            axis=0,
        ).astype(np.float32)
        y_mask = np.stack(
            [
                window.y_mask
                if window.y_mask is not None
                else np.zeros_like(window.anchor, dtype=np.float32)
                for window in selected
            ],
            axis=0,
        ).astype(np.float32)
    floor_context = np.stack(
        [
            window.floor_context
            if window.floor_context is not None
            else np.zeros_like(window.anchor, dtype=np.float32)
            for window in selected
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "x_hist": x_hist,
        "x_fut": x_fut,
        "x_static": x_static,
        "anchor": anchor,
        "y": y,
        "y_mask": y_mask,
        "target_days": np.asarray([str(window.target_day.date()) for window in selected]),
        "lower_bound": np.asarray([window.lower_bound for window in selected], dtype=np.float32),
        "upper_bound": np.asarray([window.upper_bound for window in selected], dtype=np.float32),
        "policy_regime": np.asarray([window.policy_regime for window in selected], dtype=np.float32),
        "sample_weight": np.asarray([window.sample_weight for window in selected], dtype=np.float32),
        "floor_context": floor_context,
    }


def make_synthetic_zero_floor_windows(
    windows: Sequence[DailyWindow],
    feature_spec: DeepFeatureSpec,
    config: Dict[str, Any],
) -> List[DailyWindow]:
    """Create low-weight 2026-style lower-tail samples.

    This is deliberately conservative. It only edits slots that already look
    like low-price candidates in 2025: relatively low observed prices plus high
    renewable share or low net load. The synthetic windows are marked with the
    configured zero-floor policy regime and a reduced sample weight.
    """
    augment_cfg = config.get("deep_model", {}).get("synthetic_zero_floor", {})
    if not augment_cfg.get("enabled", False):
        return []

    max_source_price = float(augment_cfg.get("max_source_price", 120.0))
    target_upper = float(augment_cfg.get("target_upper", 80.0))
    min_changed_slots = int(augment_cfg.get("min_changed_slots", 8))
    weight = float(augment_cfg.get("sample_weight", 0.35))
    regime = float(augment_cfg.get("policy_regime", 1.0))
    source_lower = float(config.get("metrics", {}).get("price_floor", 40.0))

    fut_index = {name: idx for idx, name in enumerate(feature_spec.fut_cols)}
    static_index = {name: idx for idx, name in enumerate(feature_spec.static_cols)}
    synthetic: List[DailyWindow] = []

    for window in windows:
        if window.y is None or window.y_mask is None:
            continue
        low_price = (window.y <= max_source_price) & (window.y_mask > 0)
        if not low_price.any():
            continue

        scenario = np.zeros_like(window.y, dtype=bool)
        if "新能源占比" in fut_index:
            renewable_share = window.x_fut[:, fut_index["新能源占比"]]
            scenario |= renewable_share >= np.nanpercentile(renewable_share, 65)
        if "净负荷" in fut_index:
            net_load = window.x_fut[:, fut_index["净负荷"]]
            scenario |= net_load <= np.nanpercentile(net_load, 35)
        if "抽蓄" in fut_index:
            pumped = window.x_fut[:, fut_index["抽蓄"]]
            scenario |= pumped >= np.nanpercentile(pumped, 65)

        mask = low_price & scenario
        if int(mask.sum()) < min_changed_slots:
            continue

        y = window.y.copy()
        scaled = (y[mask] - source_lower) / max(max_source_price - source_lower, 1.0)
        y[mask] = np.clip(scaled, 0.0, 1.0) * target_upper

        x_static = window.x_static.copy()
        if "policy_regime" in static_index:
            x_static[static_index["policy_regime"]] = regime
        if "price_lower_bound" in static_index:
            x_static[static_index["price_lower_bound"]] = 0.0
        if "price_upper_bound" in static_index:
            x_static[static_index["price_upper_bound"]] = 1000.0
        if "is_zero_floor_regime" in static_index:
            x_static[static_index["is_zero_floor_regime"]] = 1.0

        synthetic.append(
            replace(
                window,
                x_static=x_static,
                y=y.astype(np.float32),
                y_mask=window.y_mask.copy(),
                lower_bound=0.0,
                upper_bound=1000.0,
                policy_regime=regime,
                sample_weight=weight,
            )
        )
    return synthetic

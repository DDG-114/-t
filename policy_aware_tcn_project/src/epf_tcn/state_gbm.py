"""State-aware LightGBM baseline for day-ahead electricity price forecasting."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.metrics import regression_summary


try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    raise ImportError(
        "state_gbm requires lightgbm. Install project requirements before training it."
    ) from exc


BOUNDARY_HISTORY_SPECS = [
    ("floor45", "le", 45.0),
    ("high600", "ge", 600.0),
    ("high800", "ge", 800.0),
    ("cap900", "ge", 900.0),
    ("cap950", "ge", 950.0),
    ("cap999", "ge", 999.0),
]
BOUNDARY_WINDOWS = [1, 2, 3, 7, 14, 28]
DAY_AGG_SOURCE_COLS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "抽蓄",
    "统一新能源预测",
    "联络线计划",
    "净负荷",
    "供需裕度",
    "新能源占比",
    "竞价空间占比",
    "联络线占比",
]
PRICE_DAY_AGG_COLS = [
    "price_lag_1d",
    "price_lag_2d",
    "price_lag_7d",
    "price_lag_14d",
    "floor_ratio_1d",
    "floor_ratio_3d",
    "floor_ratio_7d",
]
SCARCITY_STATE_COLS = [
    "发电总出力预测",
    "竞价空间",
    "统一负荷预测",
    "统一新能源预测",
    "联络线计划",
    "净负荷",
    "供需裕度",
    "新能源占比",
    "竞价空间占比",
    "联络线占比",
]
SCARCITY_WINDOWS = [7, 14, 28, 56]
SCARCITY_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


@dataclass
class StateThresholds:
    """Validation-selected state override thresholds."""

    floor_threshold: float
    floor_prediction_ceiling: float | None
    cap_threshold: float
    high_threshold: float | None
    high_lift_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "floor_threshold": float(self.floor_threshold),
            "floor_prediction_ceiling": (
                None
                if self.floor_prediction_ceiling is None
                else float(self.floor_prediction_ceiling)
            ),
            "cap_threshold": float(self.cap_threshold),
            "high_threshold": (
                None if self.high_threshold is None else float(self.high_threshold)
            ),
            "high_lift_value": float(self.high_lift_value),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateThresholds":
        return cls(
            floor_threshold=float(payload["floor_threshold"]),
            floor_prediction_ceiling=(
                None
                if payload.get("floor_prediction_ceiling") is None
                else float(payload["floor_prediction_ceiling"])
            ),
            cap_threshold=float(payload["cap_threshold"]),
            high_threshold=(
                None if payload.get("high_threshold") is None else float(payload["high_threshold"])
            ),
            high_lift_value=float(payload.get("high_lift_value", 600.0)),
        )


@dataclass
class StateGBMModel:
    """State classifier plus weighted-regression model."""

    feature_cols: List[str]
    feature_medians: Dict[str, float]
    regressor: Any
    floor_classifier: Any
    cap_classifier: Any
    high_classifier: Any
    thresholds: StateThresholds
    report: Dict[str, Any]


def add_state_gbm_features(features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add day-ahead-safe boundary and daily-shape features.

    These features only use current-day fundamental forecasts and shifted price
    history, matching the rolling-calibration setup common in EPF papers.
    """
    target = config["columns"]["target"]
    result = features.sort_values(["date", "slot"]).copy()

    for name, op, threshold in BOUNDARY_HISTORY_SPECS:
        if op == "le":
            indicator = result[target].le(float(threshold)).astype(float)
        elif op == "ge":
            indicator = result[target].ge(float(threshold)).astype(float)
        else:
            raise ValueError(f"Unsupported boundary op: {op}")
        shifted = indicator.groupby(result["slot"], group_keys=False).shift(1)
        for window in BOUNDARY_WINDOWS:
            rolled = shifted.groupby(result["slot"], group_keys=False).rolling(window)
            result[f"{name}_ratio_{window}d"] = rolled.mean().reset_index(level=0, drop=True)
            if window in {7, 14, 28}:
                result[f"{name}_max_{window}d"] = rolled.max().reset_index(level=0, drop=True)

    daily = (
        result.groupby("date")[target]
        .agg(
            prev_day_price_mean="mean",
            prev_day_price_max="max",
            prev_day_price_min="min",
            prev_day_floor_ratio=lambda values: float(values.le(45.0).mean()),
            prev_day_high600_ratio=lambda values: float(values.ge(600.0).mean()),
            prev_day_cap900_ratio=lambda values: float(values.ge(900.0).mean()),
            prev_day_cap999_ratio=lambda values: float(values.ge(999.0).mean()),
        )
        .reset_index()
    )
    for lag in [1, 2, 7, 14]:
        lagged = daily.copy()
        lagged["date"] = lagged["date"] + pd.Timedelta(days=lag)
        lagged = lagged.rename(
            columns={col: f"{col}_lag{lag}d" for col in lagged.columns if col != "date"}
        )
        result = result.merge(lagged, on="date", how="left")

    for col in DAY_AGG_SOURCE_COLS:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
            continue
        group = result.groupby("date")[col]
        result[f"{col}_day_min"] = group.transform("min")
        result[f"{col}_day_max"] = group.transform("max")
        result[f"{col}_day_mean"] = group.transform("mean")
        result[f"{col}_day_std"] = group.transform("std")
        result[f"{col}_day_rank"] = group.rank(pct=True)
        result[f"{col}_slot_delta"] = group.diff().fillna(0.0)

    for col in PRICE_DAY_AGG_COLS:
        if col not in result.columns or not pd.api.types.is_numeric_dtype(result[col]):
            continue
        group = result.groupby("date")[col]
        result[f"{col}_day_min"] = group.transform("min")
        result[f"{col}_day_max"] = group.transform("max")
        result[f"{col}_day_mean"] = group.transform("mean")
        result[f"{col}_day_std"] = group.transform("std")

    if {"净负荷", "竞价空间"}.issubset(result.columns):
        result["净负荷_竞价空间比"] = result["净负荷"] / result["竞价空间"].replace(0, np.nan)
    if {"供需裕度", "统一负荷预测"}.issubset(result.columns):
        result["供需裕度_负荷比"] = result["供需裕度"] / result["统一负荷预测"].replace(0, np.nan)

    result = add_scarcity_state_features(result, config)

    result = result.copy()
    return result.sort_values(["date", "slot"]).reset_index(drop=True)


def add_scarcity_state_features(features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add leakage-safe rolling scarcity features for current forecasts.

    For each current-day fundamental forecast, these features compare the value
    with previous same-slot forecast distributions. This encodes whether the
    current supply-demand state is historically high or low without using the
    target day's realized price.
    """
    cfg = config.get("state_gbm", {}).get("scarcity_features", {})
    if not bool(cfg.get("enabled", False)):
        return features

    result = features.sort_values(["slot", "date"]).copy()
    generated: Dict[str, pd.Series] = {}
    cols = cfg.get("source_columns", SCARCITY_STATE_COLS)
    source_cols = [
        col
        for col in cols
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col])
    ]
    windows = [int(window) for window in cfg.get("windows_days", SCARCITY_WINDOWS)]
    quantiles = [float(level) for level in cfg.get("quantiles", SCARCITY_QUANTILES)]

    for col in source_cols:
        current = pd.to_numeric(result[col], errors="coerce")
        shifted = current.groupby(result["slot"], group_keys=False).shift(1)
        for window in windows:
            rolled = shifted.groupby(result["slot"], group_keys=False).rolling(window)
            mean = rolled.mean().reset_index(level=0, drop=True)
            std = rolled.std().reset_index(level=0, drop=True)
            minimum = rolled.min().reset_index(level=0, drop=True)
            maximum = rolled.max().reset_index(level=0, drop=True)

            generated[f"{col}_scarcity_mean_{window}d"] = mean
            generated[f"{col}_scarcity_delta_mean_{window}d"] = current - mean
            generated[f"{col}_scarcity_z_{window}d"] = (current - mean) / std.replace(0, np.nan)
            generated[f"{col}_scarcity_range_pos_{window}d"] = (
                (current - minimum) / (maximum - minimum).replace(0, np.nan)
            )

            quantile_values: Dict[float, pd.Series] = {}
            for level in quantiles:
                label = _quantile_label(level)
                quantile = rolled.quantile(level).reset_index(level=0, drop=True)
                quantile_values[level] = quantile
                generated[f"{col}_scarcity_{label}_{window}d"] = quantile
                generated[f"{col}_scarcity_delta_{label}_{window}d"] = current - quantile

            q10 = quantile_values.get(0.1)
            q90 = quantile_values.get(0.9)
            if q10 is not None and q90 is not None:
                generated[f"{col}_scarcity_tail_pos_{window}d"] = (
                    (current - q10) / (q90 - q10).replace(0, np.nan)
                )
                generated[f"{col}_scarcity_above_q90_{window}d"] = current.gt(q90).astype(float)
                generated[f"{col}_scarcity_below_q10_{window}d"] = current.lt(q10).astype(float)

    if generated:
        result = pd.concat([result, pd.DataFrame(generated, index=result.index)], axis=1)

    if {"净负荷", "统一新能源预测"}.issubset(result.columns):
        for window in windows:
            net_z = f"净负荷_scarcity_z_{window}d"
            renewable_z = f"统一新能源预测_scarcity_z_{window}d"
            if net_z in result.columns and renewable_z in result.columns:
                result[f"scarcity_pressure_z_{window}d"] = result[net_z] - result[renewable_z]
    if {"竞价空间占比", "新能源占比"}.issubset(result.columns):
        for window in windows:
            bid_z = f"竞价空间占比_scarcity_z_{window}d"
            renewable_share_z = f"新能源占比_scarcity_z_{window}d"
            if bid_z in result.columns and renewable_share_z in result.columns:
                result[f"scarcity_share_pressure_z_{window}d"] = (
                    result[bid_z] - result[renewable_share_z]
                )

    return result.sort_values(["date", "slot"]).reset_index(drop=True)


def _quantile_label(level: float) -> str:
    scaled = float(level) * 100
    if abs(scaled - round(scaled)) < 1e-9:
        return f"q{int(round(scaled)):02d}"
    return f"q{int(round(float(level) * 1000)):03d}"


def _date_range_days(config: Dict[str, Any], split_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    split = config["split"]
    return (
        pd.Timestamp(split[f"{split_name}_start"]).normalize(),
        pd.Timestamp(split[f"{split_name}_end"]).normalize(),
    )


def _feature_cols(features: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    target = config["columns"]["target"]
    excluded = {
        target,
        f"{target}_raw",
        f"{target}_clean_reason",
        config["columns"]["datetime"],
        "date",
        "minute",
        "is_floor_price",
        "floor_run_slots",
    }
    return [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]


def _fit_medians(frame: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    numeric = frame[list(cols)].replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(numeric_only=True).fillna(0.0)
    return {col: float(medians.get(col, 0.0)) for col in cols}


def _matrix(frame: pd.DataFrame, cols: Sequence[str], medians: Mapping[str, float]) -> pd.DataFrame:
    result = frame[list(cols)].replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(float(medians[col]))
    return result


def _split_frame(features: pd.DataFrame, config: Dict[str, Any], split_name: str) -> pd.DataFrame:
    target = config["columns"]["target"]
    start, end = _date_range_days(config, split_name)
    mask = (
        (features["date"] >= start)
        & (features["date"] <= end)
        & features[target].notna()
    )
    return features.loc[mask].copy()


def _final_train_frame(features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    target = config["columns"]["target"]
    train_start, _ = _date_range_days(config, "train")
    _, valid_end = _date_range_days(config, "valid")
    mask = (
        (features["date"] >= train_start)
        & (features["date"] <= valid_end)
        & features[target].notna()
    )
    return features.loc[mask].copy()


def _regression_weights(y: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    cfg = config.get("state_gbm", {})
    floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    weights = np.ones_like(y, dtype=float)
    weights += float(cfg.get("floor_weight_boost", 2.5)) * (y <= floor + 5.0)
    weights += float(cfg.get("high_weight_boost", 1.5)) * (y >= 600.0)
    weights += float(cfg.get("cap_weight_boost", 3.0)) * (y >= 900.0)
    mean = float(weights.mean())
    return weights if mean <= 0 else weights / mean


def _regressor(config: Dict[str, Any], n_estimators: int | None = None) -> Any:
    cfg = config.get("state_gbm", {}).get("regressor", {})
    return lgb.LGBMRegressor(
        objective=str(cfg.get("objective", "mae")),
        n_estimators=int(n_estimators or cfg.get("n_estimators", 900)),
        learning_rate=float(cfg.get("learning_rate", 0.018)),
        num_leaves=int(cfg.get("num_leaves", 31)),
        min_child_samples=int(cfg.get("min_child_samples", 25)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        random_state=int(config.get("deep_model", {}).get("seed", 42)),
        verbose=-1,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def _classifier(config: Dict[str, Any], positive_ratio: float, n_estimators: int | None = None) -> Any:
    cfg = config.get("state_gbm", {}).get("classifier", {})
    scale_pos_weight = max(1.0, (1.0 - positive_ratio) / max(positive_ratio, 1e-6))
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators or cfg.get("n_estimators", 900)),
        learning_rate=float(cfg.get("learning_rate", 0.025)),
        num_leaves=int(cfg.get("num_leaves", 31)),
        min_child_samples=int(cfg.get("min_child_samples", 20)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        scale_pos_weight=scale_pos_weight,
        random_state=int(config.get("deep_model", {}).get("seed", 42)),
        verbose=-1,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def _fit_classifier(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    config: Dict[str, Any],
    n_estimators: int | None = None,
) -> Any:
    positive_ratio = float(y_train.mean()) if y_train.size else 0.0
    model = _classifier(config, positive_ratio=positive_ratio, n_estimators=n_estimators)
    callbacks = []
    if x_valid is not None and y_valid is not None and len(np.unique(y_valid)) > 1:
        callbacks.append(
            lgb.early_stopping(
                int(config.get("state_gbm", {}).get("early_stopping_rounds", 80)),
                verbose=False,
            )
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )
    else:
        model.fit(x_train, y_train)
    return model


def _state_labels(y: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "floor": (y <= 45.0).astype(int),
        "cap": (y >= 900.0).astype(int),
        "high": (y >= 600.0).astype(int),
    }


def _predict_state_probabilities(model: StateGBMModel, x: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "floor": model.floor_classifier.predict_proba(x)[:, 1],
        "cap": model.cap_classifier.predict_proba(x)[:, 1],
        "high": model.high_classifier.predict_proba(x)[:, 1],
    }


def _apply_state_overrides(
    base_pred: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: StateThresholds,
    floor_price: float,
) -> np.ndarray:
    pred = np.asarray(base_pred, dtype=float).copy()
    floor_mask = probabilities["floor"] >= thresholds.floor_threshold
    if thresholds.floor_prediction_ceiling is not None:
        floor_mask &= pred <= thresholds.floor_prediction_ceiling
    pred[floor_mask] = floor_price

    cap_mask = probabilities["cap"] >= thresholds.cap_threshold
    pred[cap_mask] = 1000.0

    if thresholds.high_threshold is not None:
        high_mask = (
            (probabilities["high"] >= thresholds.high_threshold)
            & ~cap_mask
            & ~floor_mask
            & (pred < thresholds.high_lift_value)
        )
        pred[high_mask] = thresholds.high_lift_value

    return np.clip(pred, floor_price, 1000.0)


def _prediction_frame(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    base_pred: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    config: Dict[str, Any],
) -> pd.DataFrame:
    target = config["columns"]["target"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(frame[config["columns"]["datetime"]]),
            "date": pd.to_datetime(frame["date"]).dt.date.astype(str),
            "slot": frame["slot"].to_numpy(dtype=int),
            "y_true": frame[target].to_numpy(dtype=float),
            "base_pred": np.asarray(base_pred, dtype=float),
            "floor_probability": probabilities["floor"],
            "cap_probability": probabilities["cap"],
            "high_probability": probabilities["high"],
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def _threshold_grid(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    cfg = config.get("state_gbm", {}).get("threshold_search", {})
    return {
        "floor_thresholds": cfg.get(
            "floor_thresholds",
            [round(float(value), 2) for value in np.linspace(0.02, 0.70, 35)],
        ),
        "floor_prediction_ceilings": cfg.get(
            "floor_prediction_ceilings",
            [80, 120, 160, 220, 300, 500, 1000, None],
        ),
        "cap_thresholds": cfg.get(
            "cap_thresholds",
            [round(float(value), 2) for value in np.linspace(0.02, 0.60, 30)],
        ),
        "high_thresholds": cfg.get(
            "high_thresholds",
            [None, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        ),
        "high_lift_values": cfg.get("high_lift_values", [600, 700, 800, 900]),
    }


def _select_thresholds(
    y_valid: np.ndarray,
    base_pred: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    config: Dict[str, Any],
) -> tuple[StateThresholds, Dict[str, Any]]:
    floor_price = float(config.get("metrics", {}).get("price_floor", 40.0))
    grid = _threshold_grid(config)
    best_score = -float("inf")
    best_thresholds = None
    candidates: List[Dict[str, Any]] = []

    for floor_threshold in grid["floor_thresholds"]:
        for floor_ceiling in grid["floor_prediction_ceilings"]:
            for cap_threshold in grid["cap_thresholds"]:
                for high_threshold in grid["high_thresholds"]:
                    for high_lift in grid["high_lift_values"]:
                        thresholds = StateThresholds(
                            floor_threshold=float(floor_threshold),
                            floor_prediction_ceiling=(
                                None if floor_ceiling is None else float(floor_ceiling)
                            ),
                            cap_threshold=float(cap_threshold),
                            high_threshold=(
                                None if high_threshold is None else float(high_threshold)
                            ),
                            high_lift_value=float(high_lift),
                        )
                        pred = _apply_state_overrides(
                            base_pred,
                            probabilities,
                            thresholds,
                            floor_price=floor_price,
                        )
                        summary = regression_summary(y_valid, pred, floor_price)
                        candidate = {**thresholds.to_dict(), **summary}
                        candidates.append(candidate)
                        score = float(summary["accuracy"])
                        if score > best_score:
                            best_score = score
                            best_thresholds = thresholds

    if best_thresholds is None:
        raise RuntimeError("No state_gbm threshold candidate was evaluated.")
    report = {
        "selected": {
            **best_thresholds.to_dict(),
            "accuracy": best_score,
        },
        "top_candidates": sorted(
            candidates,
            key=lambda item: float(item["accuracy"]),
            reverse=True,
        )[:10],
    }
    return best_thresholds, report


def _classifier_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {
            "positive_ratio": float(np.mean(y_true)),
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
        }
    return {
        "positive_ratio": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def train_state_gbm(features: pd.DataFrame, config: Dict[str, Any]) -> StateGBMModel:
    """Train the state-aware GBM using train/valid, then refit through valid."""
    enhanced = add_state_gbm_features(features, config)
    cols = _feature_cols(enhanced, config)
    train = _split_frame(enhanced, config, "train")
    valid = _split_frame(enhanced, config, "valid")
    final_train = _final_train_frame(enhanced, config)
    if train.empty or valid.empty or final_train.empty:
        raise ValueError("state_gbm requires non-empty train, valid, and final-train frames.")

    target = config["columns"]["target"]
    medians = _fit_medians(train, cols)
    x_train = _matrix(train, cols, medians)
    x_valid = _matrix(valid, cols, medians)
    y_train = train[target].to_numpy(dtype=float)
    y_valid = valid[target].to_numpy(dtype=float)

    reg = _regressor(config)
    reg.fit(
        x_train,
        y_train,
        sample_weight=_regression_weights(y_train, config),
        eval_set=[(x_valid, y_valid)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(
                int(config.get("state_gbm", {}).get("early_stopping_rounds", 80)),
                verbose=False,
            )
        ],
    )
    base_valid = np.clip(reg.predict(x_valid), 40.0, 1000.0)

    labels_train = _state_labels(y_train)
    labels_valid = _state_labels(y_valid)
    classifiers = {}
    valid_probs = {}
    classifier_reports = {}
    classifier_iterations = {}
    for name in ["floor", "cap", "high"]:
        classifiers[name] = _fit_classifier(
            x_train,
            labels_train[name],
            x_valid,
            labels_valid[name],
            config,
        )
        valid_probs[name] = classifiers[name].predict_proba(x_valid)[:, 1]
        classifier_reports[name] = _classifier_report(labels_valid[name], valid_probs[name])
        classifier_iterations[name] = int(getattr(classifiers[name], "best_iteration_", None) or 300)

    thresholds, threshold_report = _select_thresholds(
        y_valid,
        base_valid,
        valid_probs,
        config,
    )

    # Refit the final model through the validation end, preserving the
    # validation-selected iteration counts and thresholds.
    final_medians = _fit_medians(final_train, cols)
    x_final = _matrix(final_train, cols, final_medians)
    y_final = final_train[target].to_numpy(dtype=float)

    final_reg = _regressor(config, n_estimators=int(getattr(reg, "best_iteration_", None) or 400))
    final_reg.fit(
        x_final,
        y_final,
        sample_weight=_regression_weights(y_final, config),
    )

    final_classifiers = {}
    final_labels = _state_labels(y_final)
    for name in ["floor", "cap", "high"]:
        final_classifiers[name] = _fit_classifier(
            x_final,
            final_labels[name],
            None,
            None,
            config,
            n_estimators=classifier_iterations[name],
        )

    valid_pred = _apply_state_overrides(
        base_valid,
        valid_probs,
        thresholds,
        floor_price=float(config.get("metrics", {}).get("price_floor", 40.0)),
    )
    report = {
        "model_type": "state_aware_lightgbm",
        "feature_cols": cols,
        "n_features": int(len(cols)),
        "train_rows": int(train.shape[0]),
        "valid_rows": int(valid.shape[0]),
        "final_train_rows": int(final_train.shape[0]),
        "regressor_best_iteration": int(getattr(reg, "best_iteration_", None) or 400),
        "classifier_best_iterations": classifier_iterations,
        "validation_base_summary": regression_summary(
            y_valid,
            base_valid,
            float(config.get("metrics", {}).get("price_floor", 40.0)),
        ),
        "validation_corrected_summary": regression_summary(
            y_valid,
            valid_pred,
            float(config.get("metrics", {}).get("price_floor", 40.0)),
        ),
        "classifiers": classifier_reports,
        "threshold_search": threshold_report,
    }
    return StateGBMModel(
        feature_cols=cols,
        feature_medians=final_medians,
        regressor=final_reg,
        floor_classifier=final_classifiers["floor"],
        cap_classifier=final_classifiers["cap"],
        high_classifier=final_classifiers["high"],
        thresholds=thresholds,
        report=report,
    )


def predict_state_gbm(
    model: StateGBMModel,
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Predict all rows in ``features`` with the fitted state-aware GBM."""
    enhanced = add_state_gbm_features(features, config)
    target = config["columns"]["target"]
    mask = enhanced[target].notna()
    frame = enhanced.loc[mask].copy()
    x = _matrix(frame, model.feature_cols, model.feature_medians)
    base_pred = np.clip(model.regressor.predict(x), 40.0, 1000.0)
    probabilities = _predict_state_probabilities(model, x)
    y_pred = _apply_state_overrides(
        base_pred,
        probabilities,
        model.thresholds,
        floor_price=float(config.get("metrics", {}).get("price_floor", 40.0)),
    )
    return _prediction_frame(frame, y_pred, base_pred, probabilities, config)


def evaluate_state_gbm(
    model: StateGBMModel,
    features: pd.DataFrame,
    config: Dict[str, Any],
    split_name: str = "test",
) -> Dict[str, Any]:
    """Evaluate a fitted state-aware GBM on one configured split."""
    enhanced = add_state_gbm_features(features, config)
    frame = _split_frame(enhanced, config, split_name)
    x = _matrix(frame, model.feature_cols, model.feature_medians)
    base_pred = np.clip(model.regressor.predict(x), 40.0, 1000.0)
    probabilities = _predict_state_probabilities(model, x)
    y_pred = _apply_state_overrides(
        base_pred,
        probabilities,
        model.thresholds,
        floor_price=float(config.get("metrics", {}).get("price_floor", 40.0)),
    )
    predictions = _prediction_frame(frame, y_pred, base_pred, probabilities, config)
    evaluation = evaluate_predictions(predictions, config)
    return {"predictions": predictions, "evaluation": evaluation}


def save_state_gbm(model: StateGBMModel, config: Dict[str, Any]) -> None:
    """Save the state-aware GBM artifact and report."""
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "state_gbm.pkl").open("wb") as file:
        pickle.dump(model, file)
    with (model_dir / "state_gbm_report.json").open("w", encoding="utf-8") as file:
        json.dump(model.report, file, ensure_ascii=False, indent=2)


def load_state_gbm(config: Dict[str, Any]) -> StateGBMModel:
    """Load a saved state-aware GBM artifact."""
    with (Path(config["paths"]["model_dir"]) / "state_gbm.pkl").open("rb") as file:
        return pickle.load(file)


def save_state_gbm_evaluation(
    predictions: pd.DataFrame,
    evaluation: Dict[str, Any],
    config: Dict[str, Any],
    prefix: str = "state_gbm",
) -> None:
    """Save state-aware GBM predictions and reports."""
    save_evaluation(predictions, evaluation, config, prefix=prefix)

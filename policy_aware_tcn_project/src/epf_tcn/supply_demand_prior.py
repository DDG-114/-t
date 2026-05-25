"""Supply-demand prior price model for residual TCN forecasting."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_COL = "sd_prior_price"
DEFAULT_FEATURE_COLS = [
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
DEFAULT_QUANTILE_PREFIXES = [
    "统一负荷预测_",
    "统一新能源预测_",
    "净负荷_",
    "供需裕度_",
    "竞价空间_",
    "竞价空间占比_",
]
DEFAULT_HINGE_COLS = [
    "净负荷",
    "供需裕度",
    "新能源占比",
    "竞价空间",
    "竞价空间占比",
]
DEFAULT_INTERACTIONS = [
    ("净负荷", "新能源占比"),
    ("净负荷", "竞价空间占比"),
    ("供需裕度", "新能源占比"),
    ("竞价空间占比", "新能源占比"),
]


@dataclass
class SupplyDemandPriorModel:
    """Piecewise-linear supply-demand prior fitted on the training split only."""

    feature_cols: List[str]
    feature_medians: Dict[str, float]
    hinge_cols: List[str]
    hinge_knots: Dict[str, List[float]]
    interaction_pairs: List[tuple[str, str]]
    scaler: StandardScaler
    regressor: Ridge
    output_col: str
    report: Dict[str, Any]

    def predict(self, frame: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
        """Predict and clip prior prices for every row in ``frame``."""
        matrix = self._make_design_matrix(frame)
        scaled = self.scaler.transform(matrix)
        pred = self.regressor.predict(scaled).astype(float)
        lower, upper = row_policy_bounds(frame, config)
        return np.clip(pred, lower, upper)

    def _make_design_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        base = _numeric_frame(frame, self.feature_cols, self.feature_medians)
        parts = [base.to_numpy(dtype=np.float32)]

        for col in self.hinge_cols:
            if col not in base.columns:
                continue
            values = base[col].to_numpy(dtype=np.float32)
            for knot in self.hinge_knots.get(col, []):
                knot_value = float(knot)
                parts.append(np.maximum(values - knot_value, 0.0).reshape(-1, 1))
                parts.append(np.maximum(knot_value - values, 0.0).reshape(-1, 1))

        for left, right in self.interaction_pairs:
            if left in base.columns and right in base.columns:
                product = base[left].to_numpy(dtype=np.float32) * base[right].to_numpy(
                    dtype=np.float32
                )
                parts.append(product.reshape(-1, 1))

        return np.concatenate(parts, axis=1).astype(np.float32)


def supply_demand_prior_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the nested prior config with defaults applied lazily."""
    return config.get("deep_model", {}).get("supply_demand_prior", {})


def supply_demand_prior_enabled(config: Dict[str, Any]) -> bool:
    """Return whether the supply-demand prior should be generated."""
    prior_cfg = supply_demand_prior_config(config)
    anchor_source = str(config.get("deep_model", {}).get("anchor_source", "previous_day"))
    return bool(prior_cfg.get("enabled", False)) or anchor_source == "supply_demand_prior"


def supply_demand_prior_output_col(config: Dict[str, Any]) -> str:
    """Return the configured prior output column name."""
    prior_cfg = supply_demand_prior_config(config)
    return str(prior_cfg.get("output_col", DEFAULT_OUTPUT_COL))


def fit_supply_demand_prior(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> SupplyDemandPriorModel:
    """Fit a piecewise-linear supply-demand prior using only train rows."""
    prior_cfg = supply_demand_prior_config(config)
    target = config["columns"]["target"]
    train_start, train_end = _date_range_days(config, "train")
    train_mask = (
        (features["date"] >= train_start)
        & (features["date"] <= train_end)
        & features[target].notna()
    )
    if not bool(train_mask.any()):
        raise ValueError("No training rows available for supply-demand prior fitting.")

    feature_cols = _configured_feature_cols(features, prior_cfg)
    if not feature_cols:
        raise ValueError("No numeric feature columns available for supply-demand prior.")
    hinge_cols = [col for col in prior_cfg.get("hinge_cols", DEFAULT_HINGE_COLS) if col in feature_cols]
    quantiles = [float(value) for value in prior_cfg.get("hinge_quantiles", [0.2, 0.5, 0.8])]
    interaction_pairs = _configured_interactions(prior_cfg, feature_cols)

    train_frame = features.loc[train_mask].copy()
    medians = {
        col: _finite_median(train_frame[col].to_numpy(dtype=float))
        for col in feature_cols
    }
    hinge_knots = {
        col: [
            float(value)
            for value in np.nanquantile(
                _numeric_frame(train_frame, [col], medians)[col].to_numpy(dtype=float),
                quantiles,
            )
        ]
        for col in hinge_cols
    }

    provisional = SupplyDemandPriorModel(
        feature_cols=feature_cols,
        feature_medians=medians,
        hinge_cols=hinge_cols,
        hinge_knots=hinge_knots,
        interaction_pairs=interaction_pairs,
        scaler=StandardScaler(),
        regressor=Ridge(
            alpha=float(prior_cfg.get("ridge_alpha", 20.0)),
            random_state=int(config.get("deep_model", {}).get("seed", 42)),
        ),
        output_col=supply_demand_prior_output_col(config),
        report={},
    )
    x_train = provisional._make_design_matrix(train_frame)
    y_train = train_frame[target].to_numpy(dtype=float)
    sample_weight = _sample_weights(y_train, config, prior_cfg)
    scaled = provisional.scaler.fit_transform(x_train)
    provisional.regressor.fit(scaled, y_train, sample_weight=sample_weight)

    report = {
        "enabled": True,
        "model_type": "piecewise_linear_supply_demand_ridge",
        "output_col": provisional.output_col,
        "feature_cols": feature_cols,
        "hinge_cols": hinge_cols,
        "hinge_quantiles": quantiles,
        "hinge_knots": hinge_knots,
        "interaction_pairs": [list(pair) for pair in interaction_pairs],
        "ridge_alpha": float(prior_cfg.get("ridge_alpha", 20.0)),
        "train_rows": int(train_mask.sum()),
        "metrics": _split_metrics(features, config, provisional),
    }
    provisional.report = report
    return provisional


def add_supply_demand_prior_column(
    features: pd.DataFrame,
    config: Dict[str, Any],
    prior: SupplyDemandPriorModel,
) -> pd.DataFrame:
    """Return a copy of ``features`` with the configured prior price column."""
    result = features.copy()
    result[prior.output_col] = prior.predict(result, config).astype(np.float32)
    return result


def save_supply_demand_prior(
    prior: SupplyDemandPriorModel | None,
    config: Dict[str, Any],
) -> None:
    """Persist the optional supply-demand prior model and JSON report."""
    if prior is None:
        return
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "supply_demand_prior.pkl").open("wb") as file:
        pickle.dump(prior, file)
    with (model_dir / "supply_demand_prior_report.json").open("w", encoding="utf-8") as file:
        json.dump(prior.report, file, ensure_ascii=False, indent=2)


def load_supply_demand_prior(config: Dict[str, Any]) -> SupplyDemandPriorModel | None:
    """Load a saved supply-demand prior model when present."""
    path = Path(config["paths"]["model_dir"]) / "supply_demand_prior.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def row_policy_bounds(
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-level lower and upper price bounds from year policy config."""
    bounds = config.get("deep_model", {}).get("year_bounds", {})
    default_min = float(config.get("model", {}).get("clip_prediction_min", 40.0))
    default_max = float(config.get("model", {}).get("clip_prediction_max", 1000.0))
    if "date" in frame.columns:
        years = pd.to_datetime(frame["date"]).dt.year.to_numpy()
    else:
        dt_col = config["columns"].get("datetime", "Date")
        years = pd.to_datetime(frame[dt_col]).dt.year.to_numpy()

    lower = np.full(frame.shape[0], default_min, dtype=float)
    upper = np.full(frame.shape[0], default_max, dtype=float)
    for idx, year in enumerate(years):
        year_bounds = bounds.get(int(year), bounds.get(str(int(year))))
        if isinstance(year_bounds, Mapping):
            lower[idx] = float(year_bounds.get("min", default_min))
            upper[idx] = float(year_bounds.get("max", default_max))
    return lower, upper


def _configured_feature_cols(frame: pd.DataFrame, prior_cfg: Dict[str, Any]) -> List[str]:
    configured = prior_cfg.get("feature_cols")
    candidates = list(configured) if configured else DEFAULT_FEATURE_COLS
    if configured is None and bool(prior_cfg.get("include_quantile_features", True)):
        candidates = [
            *candidates,
            *[
                col
                for col in frame.columns
                if any(col.startswith(prefix) for prefix in DEFAULT_QUANTILE_PREFIXES)
                and ("_q" in col or "_iqr_" in col)
            ],
        ]
    return [
        col
        for col in candidates
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _configured_interactions(
    prior_cfg: Dict[str, Any],
    feature_cols: Sequence[str],
) -> List[tuple[str, str]]:
    configured = prior_cfg.get("interaction_pairs")
    pairs = configured if configured is not None else DEFAULT_INTERACTIONS
    feature_set = set(feature_cols)
    result: List[tuple[str, str]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        if left in feature_set and right in feature_set:
            result.append((left, right))
    return result


def _numeric_frame(
    frame: pd.DataFrame,
    cols: Sequence[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    result = frame[list(cols)].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    for col in cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(medians[col])
    return result


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def _sample_weights(
    y_true: np.ndarray,
    config: Dict[str, Any],
    prior_cfg: Dict[str, Any],
) -> np.ndarray:
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    weights = np.ones_like(y_true, dtype=float)
    weights += float(prior_cfg.get("high_price_weight_boost", 0.4)) * (y_true >= 800.0)
    weights += float(prior_cfg.get("low_price_weight_boost", 0.4)) * (
        y_true <= max(price_floor * 1.5, 60.0)
    )
    weights += float(prior_cfg.get("cap_price_weight_boost", 0.6)) * (y_true >= 950.0)
    weights += float(prior_cfg.get("floor_weight_boost", 0.4)) * (
        y_true <= price_floor + 5.0
    )
    mean = float(np.mean(weights))
    if mean <= 0:
        return weights
    return weights / mean


def _date_range_days(config: Dict[str, Any], split_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    split = config["split"]
    return (
        pd.Timestamp(split[f"{split_name}_start"]).normalize(),
        pd.Timestamp(split[f"{split_name}_end"]).normalize(),
    )


def _split_metrics(
    features: pd.DataFrame,
    config: Dict[str, Any],
    prior: SupplyDemandPriorModel,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    target = config["columns"]["target"]
    for split_name in ["train", "valid", "test"]:
        start, end = _date_range_days(config, split_name)
        mask = (
            (features["date"] >= start)
            & (features["date"] <= end)
            & features[target].notna()
        )
        if not bool(mask.any()):
            continue
        y_true = features.loc[mask, target].to_numpy(dtype=float)
        y_pred = prior.predict(features.loc[mask], config)
        metrics[split_name] = _regression_metrics(
            y_true,
            y_pred,
            price_floor=float(config.get("metrics", {}).get("price_floor", 40.0)),
        )
    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_floor: float,
) -> Dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    rel = np.abs(error) / np.maximum(np.abs(y_true), float(price_floor))
    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": float(1.0 - np.mean(rel)),
    }

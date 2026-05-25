"""Training and inference utilities for report2 deep EPF models."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from epf_tcn.deep_data import (
    DeepFeatureSpec,
    DailyWindow,
    add_external_anchor_columns,
    build_daily_windows,
    external_anchor_config,
    filter_windows_by_date_range,
    fit_window_normalizer,
    infer_deep_feature_spec,
    make_synthetic_zero_floor_windows,
    transform_windows,
)
from epf_tcn.deep_models import build_policy_aware_tcn, module_summary, require_torch
from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.supply_demand_prior import (
    SupplyDemandPriorModel,
    add_supply_demand_prior_column,
    fit_supply_demand_prior,
    load_supply_demand_prior,
    save_supply_demand_prior,
    supply_demand_prior_enabled,
)

torch = require_torch()
nn = torch.nn


@dataclass
class DeepTrainingResult:
    """Artifacts produced by one training run."""

    model: nn.Module
    feature_spec: DeepFeatureSpec
    normalizer: Any
    train_report: Dict[str, Any]
    valid_predictions: pd.DataFrame
    floor_price_correction: Any | None = None
    high_price_correction: Any | None = None
    residual_blend_correction: Any | None = None
    supply_demand_prior: SupplyDemandPriorModel | None = None


@dataclass
class FloorPriceCorrection:
    """Validation-selected postprocessor for persistent floor-price regimes."""

    classifier: Any
    feature_cols: List[str]
    floor_price: float
    probability_threshold: float
    prediction_ceiling: float | None
    report: Dict[str, Any]


@dataclass
class HighPriceCorrection:
    """Validation-selected postprocessor for TCN high-price underprediction."""

    classifier: Any
    feature_cols: List[str]
    target_threshold: float
    probability_threshold: float
    prediction_floor: float
    max_floor_probability: float | None
    report: Dict[str, Any]


@dataclass
class ResidualBlendCorrection:
    """Validation-selected blend between the external anchor and neural output."""

    shrink: float
    clip_value: float | None
    min_prediction: float
    max_floor_probability: float | None
    positive_only: bool
    report: Dict[str, Any]


class WindowTensorDataset(torch.utils.data.Dataset):
    """Small in-memory Dataset for daily-window tensors."""

    def __init__(self, arrays: Mapping[str, np.ndarray], include_target: bool = True) -> None:
        self.arrays = arrays
        self.include_target = include_target

    def __len__(self) -> int:
        return int(self.arrays["x_hist"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            "x_hist": torch.as_tensor(self.arrays["x_hist"][index], dtype=torch.float32),
            "x_fut": torch.as_tensor(self.arrays["x_fut"][index], dtype=torch.float32),
            "x_static": torch.as_tensor(self.arrays["x_static"][index], dtype=torch.float32),
            "anchor": torch.as_tensor(self.arrays["anchor"][index], dtype=torch.float32),
            "lower_bound": torch.as_tensor(self.arrays["lower_bound"][index], dtype=torch.float32),
            "upper_bound": torch.as_tensor(self.arrays["upper_bound"][index], dtype=torch.float32),
            "sample_weight": torch.as_tensor(self.arrays["sample_weight"][index], dtype=torch.float32),
        }
        if "floor_context" in self.arrays:
            item["floor_context"] = torch.as_tensor(
                self.arrays["floor_context"][index],
                dtype=torch.float32,
            )
        if self.include_target and self.arrays.get("y") is not None:
            item["y"] = torch.as_tensor(self.arrays["y"][index], dtype=torch.float32)
            item["y_mask"] = torch.as_tensor(self.arrays["y_mask"][index], dtype=torch.float32)
        return item


def make_point_weights(
    y_true: torch.Tensor,
    price_floor: float,
    config: Dict[str, Any],
) -> torch.Tensor:
    """Report2-aligned point weights for high, low, and boundary prices."""
    deep_cfg = config.get("deep_model", {})
    high_boost = float(deep_cfg.get("high_price_weight_boost", 0.50))
    low_boost = float(deep_cfg.get("low_price_weight_boost", 0.70))
    cap_boost = float(deep_cfg.get("cap_price_weight_boost", 0.80))
    floor_boost = float(deep_cfg.get("floor_weight_boost", 1.20))
    weights = torch.ones_like(y_true)
    weights = weights + high_boost * (y_true >= 800.0).float()
    weights = weights + low_boost * (y_true <= max(float(price_floor) * 1.5, 60.0)).float()
    weights = weights + cap_boost * (y_true >= 950.0).float()
    weights = weights + floor_boost * (y_true <= float(price_floor) + 5.0).float()
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)


def make_floor_context_weights(
    batch: Mapping[str, torch.Tensor],
    config: Dict[str, Any],
) -> torch.Tensor | None:
    """Increase weight for target slots likely to be in long floor-price windows."""
    boost = float(config.get("deep_model", {}).get("floor_run_weight_boost", 0.0))
    if boost <= 0:
        return None
    context = batch.get("floor_context")
    if context is None:
        return None
    context = torch.clamp(context, min=0.0, max=1.0)
    return 1.0 + boost * context


def make_floor_run_target_weights(
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
    price_floor: float,
    config: Dict[str, Any],
) -> torch.Tensor | None:
    """Boost slots that belong to long observed floor-price runs."""
    boost = float(config.get("deep_model", {}).get("floor_run_weight_boost", 0.0))
    if boost <= 0:
        return None
    long_run_slots = int(config.get("features", {}).get("long_floor_run_slots", 16))
    floor = target <= float(price_floor) + 1e-6
    if target_mask is not None:
        floor = floor & (target_mask > 0)
    if not bool(floor.any()):
        return None

    weights = torch.ones_like(target)
    floor_cpu = floor.detach().cpu().numpy()
    long_mask = torch.zeros_like(target, dtype=torch.bool)
    for batch_index, row in enumerate(floor_cpu):
        start = None
        for idx, is_floor in enumerate(row.tolist() + [False]):
            if is_floor and start is None:
                start = idx
            elif not is_floor and start is not None:
                if idx - start >= long_run_slots:
                    long_mask[batch_index, start:idx] = True
                start = None
    return weights + boost * long_mask.float()


def weighted_relative_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    point_weight: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    denom_floor: float = 40.0,
    lambda_ramp: float = 0.10,
) -> torch.Tensor:
    """Business-aligned relative MAE plus ramp consistency."""
    denom = torch.clamp(target.abs(), min=float(denom_floor))
    rel_err = (pred - target).abs() / denom
    if point_weight is None:
        point_weight = torch.ones_like(rel_err)
    if sample_weight is not None:
        point_weight = point_weight * sample_weight.view(-1, 1)
    if target_mask is not None:
        point_weight = point_weight * target_mask
    base = (rel_err * point_weight).sum() / point_weight.sum().clamp_min(1.0)

    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    ramp_denom = torch.clamp(target_delta.abs(), min=float(denom_floor))
    ramp_error = (pred_delta - target_delta).abs() / ramp_denom
    if target_mask is not None:
        ramp_mask = target_mask[:, 1:] * target_mask[:, :-1]
        ramp = (ramp_error * ramp_mask).sum() / ramp_mask.sum().clamp_min(1.0)
    else:
        ramp = ramp_error.mean()
    return base + float(lambda_ramp) * ramp


def floor_overprediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
    price_floor: float,
) -> torch.Tensor:
    """Penalize high predictions when the true label is at the floor price."""
    floor_mask = (target <= float(price_floor) + 1e-6).float()
    if target_mask is not None:
        floor_mask = floor_mask * target_mask
    excess = torch.relu(pred - float(price_floor)) / float(price_floor)
    return (excess * floor_mask).sum() / floor_mask.sum().clamp_min(1.0)


def gaussian_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_sigma: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL used as a light auxiliary objective."""
    sigma = torch.exp(log_sigma).clamp_min(1e-4)
    nll = 0.5 * (((target - pred) / sigma) ** 2 + 2.0 * log_sigma + math.log(2.0 * math.pi))
    if sample_weight is not None:
        nll = nll * sample_weight.view(-1, 1)
    if target_mask is not None:
        nll = nll * target_mask
        return nll.sum() / target_mask.sum().clamp_min(1.0)
    return nll.mean()


def combined_loss(
    model_output: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    sample_weight: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    config: Dict[str, Any],
    floor_context_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine business loss, ramp regularization, and optional NLL."""
    deep_cfg = config.get("deep_model", {})
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    lambda_ramp = float(deep_cfg.get("lambda_ramp", 0.10))
    point_weight = make_point_weights(target, price_floor=price_floor, config=config)
    if floor_context_weight is not None:
        point_weight = point_weight * floor_context_weight
    floor_run_target_weight = make_floor_run_target_weights(
        target,
        target_mask=target_mask,
        price_floor=price_floor,
        config=config,
    )
    if floor_run_target_weight is not None:
        point_weight = point_weight * floor_run_target_weight

    loss = weighted_relative_mae_loss(
        model_output["mu"],
        target,
        sample_weight=sample_weight,
        point_weight=point_weight,
        target_mask=target_mask,
        denom_floor=price_floor,
        lambda_ramp=lambda_ramp,
    )
    floor_penalty_weight = float(deep_cfg.get("floor_overprediction_weight", 0.0))
    if floor_penalty_weight > 0:
        loss = loss + floor_penalty_weight * floor_overprediction_loss(
            model_output["mu"],
            target,
            target_mask=target_mask,
            price_floor=price_floor,
        )
    if "log_sigma" in model_output:
        nll_weight = float(deep_cfg.get("nll_weight", 0.05))
        if nll_weight > 0:
            loss = loss + nll_weight * gaussian_nll_loss(
                model_output["mu"],
                target,
                model_output["log_sigma"],
                sample_weight=sample_weight,
                target_mask=target_mask,
            )
    return loss


def _batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    device: torch.device,
) -> float:
    """Train one epoch and return mean batch loss."""
    model.train()
    total = 0.0
    count = 0
    grad_clip = float(config.get("deep_model", {}).get("grad_clip", 1.0))
    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()
        output = model(batch["x_hist"], batch["x_fut"], batch["x_static"], batch["anchor"])
        loss = combined_loss(
            output,
            batch["y"],
            batch.get("sample_weight"),
            batch.get("y_mask"),
            config,
            floor_context_weight=make_floor_context_weights(batch, config),
        )
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> float:
    """Evaluate validation loss."""
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        output = model(batch["x_hist"], batch["x_fut"], batch["x_static"], batch["anchor"])
        loss = combined_loss(
            output,
            batch["y"],
            batch.get("sample_weight"),
            batch.get("y_mask"),
            config,
            floor_context_weight=make_floor_context_weights(batch, config),
        )
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def predict_from_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Predict batches from an existing loader without rebuilding a Dataset."""
    model.eval()
    outputs: List[np.ndarray] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        pred = model(batch["x_hist"], batch["x_fut"], batch["x_static"], batch["anchor"])["mu"]
        lower = batch["lower_bound"].view(-1, 1)
        upper = batch["upper_bound"].view(-1, 1)
        pred = pred.clamp(min=lower, max=upper)
        outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def predict_arrays(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    config: Dict[str, Any],
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predict daily windows and clip by each sample's policy bounds."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = model.to(device)
    model.eval()
    batch_size = batch_size or int(config.get("deep_model", {}).get("batch_size", 16))
    dataset = WindowTensorDataset(arrays, include_target=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: List[np.ndarray] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        pred = model(batch["x_hist"], batch["x_fut"], batch["x_static"], batch["anchor"])["mu"]
        lower = batch["lower_bound"].view(-1, 1)
        upper = batch["upper_bound"].view(-1, 1)
        pred = pred.clamp(min=lower, max=upper)
        outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def windows_to_prediction_frame(
    arrays: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Convert ``[days, 96]`` predictions to the project's long CSV format."""
    horizon = int(config.get("data", {}).get("expected_slots_per_day", 96))
    rows = []
    for day_index, day in enumerate(arrays["target_days"]):
        target_day = pd.Timestamp(str(day))
        y_true = None
        y_mask = None
        if arrays.get("y") is not None:
            y_true = arrays["y"][day_index]
            y_mask = arrays["y_mask"][day_index]
        for slot in range(horizon):
            observed = y_true is not None and y_mask is not None and y_mask[slot] > 0
            anchor_price = float(arrays["anchor"][day_index, slot])
            rows.append(
                {
                    "Date": target_day + pd.Timedelta(minutes=15 * slot),
                    "date": target_day.date().isoformat(),
                    "slot": slot,
                    "y_true": np.nan if not observed else float(y_true[slot]),
                    "anchor_price": anchor_price,
                    "y_pred": float(predictions[day_index, slot]),
                    "tcn_residual": float(predictions[day_index, slot] - anchor_price),
                    "policy_regime": float(arrays["policy_regime"][day_index]),
                    "price_lower_bound": float(arrays["lower_bound"][day_index]),
                    "price_upper_bound": float(arrays["upper_bound"][day_index]),
                }
            )
    return pd.DataFrame(rows)


def _attach_external_anchor_prediction_columns(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Attach external-anchor diagnostic columns to a prediction frame."""
    anchor_cfg = external_anchor_config(config)
    if not bool(anchor_cfg.get("enabled", False)):
        return frame

    datetime_col = config["columns"].get("datetime", "Date")
    cols = [
        str(col)
        for col in anchor_cfg.get(
            "copy_columns",
            [
                "base_pred",
                "state_pred",
                "residual_pred",
                "floor_probability",
                "high_probability",
                "cap_probability",
            ],
        )
        if col in features.columns
    ]
    if not cols:
        return frame

    key = features[[datetime_col, *cols]].copy()
    key[datetime_col] = pd.to_datetime(key[datetime_col])
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result = result.drop(columns=[col for col in cols if col in result.columns])
    result = result.merge(key, left_on="Date", right_on=datetime_col, how="left")
    if datetime_col != "Date":
        result = result.drop(columns=[datetime_col])
    return result


def _floor_classifier_feature_cols(features: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return day-ahead safe feature columns for floor-price classification."""
    target = config["columns"]["target"]
    dt_col = config["columns"]["datetime"]
    excluded = {
        target,
        f"{target}_raw",
        f"{target}_clean_reason",
        dt_col,
        "date",
        "minute",
        "is_floor_price",
        "floor_run_slots",
        "prev_floor_run_slots",
        "prev_long_floor_run",
    }
    return [
        col
        for col in features.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(features[col])
    ]


def _postprocessor_feature_cols(features: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Return day-ahead safe feature columns for price-state postprocessors."""
    return _floor_classifier_feature_cols(features, config)


def _price_accuracy_from_frame(frame: pd.DataFrame, pred_col: str, price_floor: float) -> Dict[str, float]:
    valid = frame[["date", "y_true", pred_col]].dropna()
    if valid.empty:
        return {"mean_daily_accuracy": float("nan"), "daily_accuracy_sum": float("nan")}

    daily_scores = []
    for _, group in valid.groupby("date"):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group[pred_col].to_numpy(dtype=float)
        rel = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), float(price_floor))
        daily_scores.append(float(1.0 - np.mean(rel)))
    overall_rel = np.abs(valid[pred_col] - valid["y_true"]) / np.maximum(
        np.abs(valid["y_true"]),
        float(price_floor),
    )
    return {
        "mean_daily_accuracy": float(np.mean(daily_scores)),
        "daily_accuracy_sum": float(np.sum(daily_scores)),
        "overall_accuracy": float(1.0 - np.mean(overall_rel)),
    }


def _weighted_accuracy_from_frame(
    frame: pd.DataFrame,
    pred_col: str,
    price_floor: float,
    high_boost: float,
    cap_boost: float,
) -> float:
    """Return project accuracy with optional high/cap sample emphasis."""
    valid = frame[["y_true", pred_col]].dropna()
    if valid.empty:
        return float("nan")
    y_true = valid["y_true"].to_numpy(dtype=float)
    y_pred = valid[pred_col].to_numpy(dtype=float)
    point_accuracy = 1.0 - np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), float(price_floor))
    weights = np.ones_like(point_accuracy, dtype=float)
    weights += float(high_boost) * (y_true >= 500.0)
    weights += float(cap_boost) * (y_true >= 900.0)
    return float(np.average(point_accuracy, weights=weights))


def _apply_floor_correction_to_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    probability_threshold: float,
    prediction_ceiling: float | None,
    floor_price: float,
) -> pd.DataFrame:
    """Apply floor-price correction to a prediction frame."""
    result = frame.copy()
    result["floor_probability"] = np.asarray(probabilities, dtype=np.float32)
    mask = result["floor_probability"] >= float(probability_threshold)
    if prediction_ceiling is not None:
        mask &= result["y_pred"] <= float(prediction_ceiling)
    result["floor_price_corrected"] = mask.astype(bool)
    result.loc[mask, "y_pred"] = float(floor_price)
    return result


def _aligned_features_for_predictions(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Align feature rows to a prediction frame by timestamp."""
    dt_col = config["columns"]["datetime"]
    key = features[[dt_col, *feature_cols]].copy()
    key[dt_col] = pd.to_datetime(key[dt_col])
    aligned = frame[["Date"]].copy()
    aligned["Date"] = pd.to_datetime(aligned["Date"])
    merged = aligned.merge(key, left_on="Date", right_on=dt_col, how="left")
    return merged[list(feature_cols)]


def train_floor_price_correction(
    features: pd.DataFrame,
    config: Dict[str, Any],
    valid_frame: pd.DataFrame,
) -> FloorPriceCorrection | None:
    """Train a conservative state classifier for persistent 40-yuan intervals."""
    correction_cfg = config.get("deep_model", {}).get("floor_price_correction", {})
    if not correction_cfg.get("enabled", True):
        return None

    target = config["columns"]["target"]
    feature_cols = _floor_classifier_feature_cols(features, config)
    if not feature_cols:
        return None

    train_start, train_end = _date_range_days(config, "train")
    valid_start, valid_end = _date_range_days(config, "valid")
    train_mask = (
        (features["date"] >= train_start)
        & (features["date"] <= train_end)
        & features[target].notna()
    )
    valid_mask = (
        (features["date"] >= valid_start)
        & (features["date"] <= valid_end)
        & features[target].notna()
    )
    if not bool(train_mask.any()) or not bool(valid_mask.any()):
        return None

    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    y_train = features.loc[train_mask, target].eq(price_floor).astype(int)
    y_valid = features.loc[valid_mask, target].eq(price_floor).astype(int)
    if y_train.nunique() < 2 or y_valid.nunique() < 2:
        return None

    model_cfg = correction_cfg.get("model", {})
    classifier = HistGradientBoostingClassifier(
        max_iter=int(model_cfg.get("max_iter", 500)),
        learning_rate=float(model_cfg.get("learning_rate", 0.025)),
        max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 15)),
        l2_regularization=float(model_cfg.get("l2_regularization", 0.1)),
        random_state=int(config.get("deep_model", {}).get("seed", 42)),
    )
    positive_weight = float(correction_cfg.get("positive_sample_weight", 1.5))
    sample_weight = np.where(y_train.to_numpy() == 1, positive_weight, 1.0)
    classifier.fit(features.loc[train_mask, feature_cols], y_train, sample_weight=sample_weight)

    valid_prob = classifier.predict_proba(features.loc[valid_mask, feature_cols])[:, 1]
    aligned_valid_x = _aligned_features_for_predictions(features, valid_frame, feature_cols, config)
    aligned_valid_prob = classifier.predict_proba(aligned_valid_x)[:, 1]

    base_metric = _price_accuracy_from_frame(valid_frame, "y_pred", price_floor)
    thresholds = correction_cfg.get("probability_threshold_grid")
    if thresholds is None:
        thresholds = [round(float(value), 2) for value in np.linspace(0.10, 0.95, 18)]
    ceilings = correction_cfg.get("prediction_ceiling_grid")
    if ceilings is None:
        ceilings = [None, 80, 120, 160, 200, 300, 500, 800, 1000]

    best = None
    candidates: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for ceiling in ceilings:
            corrected = _apply_floor_correction_to_frame(
                valid_frame,
                probabilities=aligned_valid_prob,
                probability_threshold=float(threshold),
                prediction_ceiling=None if ceiling is None else float(ceiling),
                floor_price=price_floor,
            )
            metric = _price_accuracy_from_frame(corrected, "y_pred", price_floor)
            changed = int(corrected["floor_price_corrected"].sum())
            candidate = {
                "probability_threshold": float(threshold),
                "prediction_ceiling": None if ceiling is None else float(ceiling),
                "mean_daily_accuracy": metric["mean_daily_accuracy"],
                "daily_accuracy_sum": metric["daily_accuracy_sum"],
                "overall_accuracy": metric["overall_accuracy"],
                "changed_slots": changed,
            }
            candidates.append(candidate)
            key = (candidate["mean_daily_accuracy"], -changed)
            if best is None or key > (best["mean_daily_accuracy"], -best["changed_slots"]):
                best = candidate

    if best is None or best["mean_daily_accuracy"] <= base_metric["mean_daily_accuracy"]:
        return None

    train_pos = int(y_train.sum())
    valid_pos = int(y_valid.sum())
    report = {
        "enabled": True,
        "feature_cols": feature_cols,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "train_floor_rows": train_pos,
        "valid_floor_rows": valid_pos,
        "valid_auc": float(roc_auc_score(y_valid, valid_prob)),
        "valid_average_precision": float(average_precision_score(y_valid, valid_prob)),
        "base_valid_mean_daily_accuracy": base_metric["mean_daily_accuracy"],
        "base_valid_daily_accuracy_sum": base_metric["daily_accuracy_sum"],
        "selected": best,
        "top_candidates": sorted(
            candidates,
            key=lambda item: item["mean_daily_accuracy"],
            reverse=True,
        )[:10],
    }
    return FloorPriceCorrection(
        classifier=classifier,
        feature_cols=feature_cols,
        floor_price=price_floor,
        probability_threshold=float(best["probability_threshold"]),
        prediction_ceiling=best["prediction_ceiling"],
        report=report,
    )


def _apply_high_price_correction_to_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    probability_threshold: float,
    prediction_floor: float,
    max_floor_probability: float | None,
) -> pd.DataFrame:
    """Lift predictions for validation-selected likely high-price slots."""
    result = frame.copy()
    result["high_price_probability"] = np.asarray(probabilities, dtype=np.float32)
    mask = result["high_price_probability"] >= float(probability_threshold)
    if max_floor_probability is not None and "floor_probability" in result.columns:
        mask &= result["floor_probability"] <= float(max_floor_probability)
    result["high_price_corrected"] = mask.astype(bool)
    result.loc[mask, "y_pred"] = np.maximum(
        result.loc[mask, "y_pred"].to_numpy(dtype=float),
        float(prediction_floor),
    )
    return result


def _apply_residual_blend_to_frame(
    frame: pd.DataFrame,
    shrink: float,
    clip_value: float | None,
    min_prediction: float,
    max_floor_probability: float | None,
    positive_only: bool,
    price_floor: float,
    price_cap: float,
) -> pd.DataFrame:
    """Blend neural residuals back toward the external anchor."""
    result = frame.copy()
    if "anchor_price" not in result.columns:
        raise ValueError("Residual blend correction requires anchor_price in prediction frame.")
    anchor = result["anchor_price"].to_numpy(dtype=float)
    pred = result["y_pred"].to_numpy(dtype=float)
    residual = pred - anchor
    if clip_value is not None:
        residual = np.clip(residual, -float(clip_value), float(clip_value))
    if positive_only:
        residual = np.maximum(residual, 0.0)

    corrected = anchor.copy()
    mask = anchor >= float(min_prediction)
    if max_floor_probability is not None and "floor_probability" in result.columns:
        mask &= result["floor_probability"].to_numpy(dtype=float) <= float(max_floor_probability)
    corrected[mask] = anchor[mask] + float(shrink) * residual[mask]
    result["neural_raw_pred"] = pred
    result["residual_blend_applied"] = mask.astype(bool)
    result["y_pred"] = np.clip(corrected, float(price_floor), float(price_cap))
    result["tcn_residual"] = result["y_pred"].to_numpy(dtype=float) - anchor
    return result


def train_residual_blend_correction(
    valid_frame: pd.DataFrame,
    config: Dict[str, Any],
) -> ResidualBlendCorrection | None:
    """Select a conservative anchor/neural blend on validation accuracy."""
    blend_cfg = config.get("deep_model", {}).get("residual_blend_correction", {})
    if not bool(blend_cfg.get("enabled", False)):
        return None
    if "anchor_price" not in valid_frame.columns:
        return None

    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    price_cap = float(config.get("model", {}).get("clip_prediction_max", 1000.0))
    base_frame = valid_frame.copy()
    base_frame["y_pred"] = base_frame["anchor_price"]
    base_metric = _price_accuracy_from_frame(base_frame, "y_pred", price_floor)

    shrink_grid = blend_cfg.get("shrink_grid", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    clip_grid = blend_cfg.get("clip_grid", [0.0, 40.0, 80.0, 160.0, 300.0, None])
    min_prediction_grid = blend_cfg.get("min_prediction_grid", [40.0])
    max_floor_probability_grid = blend_cfg.get("max_floor_probability_grid", [None])
    positive_only_grid = blend_cfg.get("positive_only_grid", [False])

    selection_metric = str(blend_cfg.get("selection_metric", "overall_accuracy"))
    best: Dict[str, Any] | None = None
    candidates: List[Dict[str, Any]] = []
    for shrink in shrink_grid:
        for clip_value in clip_grid:
            for min_prediction in min_prediction_grid:
                for max_floor_probability in max_floor_probability_grid:
                    for positive_only in positive_only_grid:
                        corrected = _apply_residual_blend_to_frame(
                            valid_frame,
                            shrink=float(shrink),
                            clip_value=None if clip_value is None else float(clip_value),
                            min_prediction=float(min_prediction),
                            max_floor_probability=(
                                None
                                if max_floor_probability is None
                                else float(max_floor_probability)
                            ),
                            positive_only=bool(positive_only),
                            price_floor=price_floor,
                            price_cap=price_cap,
                        )
                        metric = _price_accuracy_from_frame(corrected, "y_pred", price_floor)
                        changed = int(
                            np.count_nonzero(
                                np.abs(
                                    corrected["y_pred"].to_numpy(dtype=float)
                                    - corrected["anchor_price"].to_numpy(dtype=float)
                                )
                                > 1e-6
                            )
                        )
                        candidate = {
                            "shrink": float(shrink),
                            "clip_value": None if clip_value is None else float(clip_value),
                            "min_prediction": float(min_prediction),
                            "max_floor_probability": (
                                None
                                if max_floor_probability is None
                                else float(max_floor_probability)
                            ),
                            "positive_only": bool(positive_only),
                            "mean_daily_accuracy": metric["mean_daily_accuracy"],
                            "daily_accuracy_sum": metric["daily_accuracy_sum"],
                            "overall_accuracy": metric["overall_accuracy"],
                            "changed_slots": changed,
                        }
                        candidates.append(candidate)
                        if selection_metric == "mean_daily_accuracy":
                            key = (candidate["mean_daily_accuracy"], -abs(float(shrink)))
                            best_key = (
                                best["mean_daily_accuracy"],
                                -abs(best["shrink"]),
                            ) if best is not None else None
                        elif selection_metric == "overall_accuracy":
                            key = (candidate["overall_accuracy"], -abs(float(shrink)))
                            best_key = (
                                best["overall_accuracy"],
                                -abs(best["shrink"]),
                            ) if best is not None else None
                        else:
                            raise ValueError(
                                "Unsupported deep_model.residual_blend_correction.selection_metric: "
                                f"{selection_metric}"
                            )
                        if best is None or key > best_key:
                            best = candidate

    if best is None:
        return None
    tolerance = float(blend_cfg.get("selection_tolerance", 0.0))
    base_score = float(base_metric[selection_metric])
    if best[selection_metric] < base_score + tolerance:
        best = {
            "shrink": 0.0,
            "clip_value": 0.0,
            "min_prediction": 1000.0,
            "max_floor_probability": 0.0,
            "positive_only": False,
            "mean_daily_accuracy": base_metric["mean_daily_accuracy"],
            "daily_accuracy_sum": base_metric["daily_accuracy_sum"],
            "overall_accuracy": base_metric["overall_accuracy"],
            "changed_slots": 0,
        }

    report = {
        "enabled": True,
        "selection_metric": selection_metric,
        "base_anchor_valid_mean_daily_accuracy": base_metric["mean_daily_accuracy"],
        "base_anchor_valid_daily_accuracy_sum": base_metric["daily_accuracy_sum"],
        "base_anchor_valid_overall_accuracy": base_metric["overall_accuracy"],
        "selected": best,
        "top_candidates": sorted(
            candidates,
            key=lambda item: item["mean_daily_accuracy"],
            reverse=True,
        )[:10],
    }
    return ResidualBlendCorrection(
        shrink=float(best["shrink"]),
        clip_value=None if best["clip_value"] is None else float(best["clip_value"]),
        min_prediction=float(best["min_prediction"]),
        max_floor_probability=(
            None if best["max_floor_probability"] is None else float(best["max_floor_probability"])
        ),
        positive_only=bool(best["positive_only"]),
        report=report,
    )


def apply_residual_blend_correction(
    frame: pd.DataFrame,
    config: Dict[str, Any],
    correction: ResidualBlendCorrection | None,
) -> pd.DataFrame:
    """Apply an optional validation-selected residual blend correction."""
    if correction is None:
        return frame
    return _apply_residual_blend_to_frame(
        frame,
        shrink=correction.shrink,
        clip_value=correction.clip_value,
        min_prediction=correction.min_prediction,
        max_floor_probability=correction.max_floor_probability,
        positive_only=correction.positive_only,
        price_floor=float(config.get("metrics", {}).get("price_floor", 40.0)),
        price_cap=float(config.get("model", {}).get("clip_prediction_max", 1000.0)),
    )


def train_high_price_correction(
    features: pd.DataFrame,
    config: Dict[str, Any],
    valid_frame: pd.DataFrame,
) -> HighPriceCorrection | None:
    """Train a conservative high-price classifier for TCN underprediction."""
    correction_cfg = config.get("deep_model", {}).get("high_price_correction", {})
    if not correction_cfg.get("enabled", False):
        return None

    target = config["columns"]["target"]
    feature_cols = _postprocessor_feature_cols(features, config)
    if not feature_cols:
        return None

    train_start, train_end = _date_range_days(config, "train")
    valid_start, valid_end = _date_range_days(config, "valid")
    train_mask = (
        (features["date"] >= train_start)
        & (features["date"] <= train_end)
        & features[target].notna()
    )
    valid_mask = (
        (features["date"] >= valid_start)
        & (features["date"] <= valid_end)
        & features[target].notna()
    )
    if not bool(train_mask.any()) or not bool(valid_mask.any()):
        return None

    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    target_thresholds = correction_cfg.get("target_threshold_grid", [500.0, 600.0, 800.0, 900.0])
    probability_thresholds = correction_cfg.get(
        "probability_threshold_grid",
        [round(float(value), 2) for value in np.linspace(0.05, 0.95, 19)],
    )
    prediction_floors = correction_cfg.get("prediction_floor_grid", [400.0, 500.0, 600.0, 700.0])
    max_floor_probabilities = correction_cfg.get("max_floor_probability_grid", [None])

    model_cfg = correction_cfg.get("model", {})
    base_metric = _price_accuracy_from_frame(valid_frame, "y_pred", price_floor)
    best: Dict[str, Any] | None = None
    best_model = None
    best_threshold = None
    best_probabilities = None
    reports: List[Dict[str, Any]] = []

    for target_threshold in target_thresholds:
        y_train = features.loc[train_mask, target].ge(float(target_threshold)).astype(int)
        y_valid = features.loc[valid_mask, target].ge(float(target_threshold)).astype(int)
        if y_train.nunique() < 2 or y_valid.nunique() < 2:
            continue
        classifier = HistGradientBoostingClassifier(
            max_iter=int(model_cfg.get("max_iter", 500)),
            learning_rate=float(model_cfg.get("learning_rate", 0.025)),
            max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 15)),
            l2_regularization=float(model_cfg.get("l2_regularization", 0.1)),
            random_state=int(config.get("deep_model", {}).get("seed", 42)) + int(target_threshold),
        )
        positive_weight = float(correction_cfg.get("positive_sample_weight", 3.0))
        sample_weight = np.where(y_train.to_numpy() == 1, positive_weight, 1.0)
        classifier.fit(features.loc[train_mask, feature_cols], y_train, sample_weight=sample_weight)

        valid_prob = classifier.predict_proba(features.loc[valid_mask, feature_cols])[:, 1]
        aligned_valid_x = _aligned_features_for_predictions(features, valid_frame, feature_cols, config)
        aligned_valid_prob = classifier.predict_proba(aligned_valid_x)[:, 1]
        model_report = {
            "target_threshold": float(target_threshold),
            "train_positive_rows": int(y_train.sum()),
            "valid_positive_rows": int(y_valid.sum()),
            "valid_auc": float(roc_auc_score(y_valid, valid_prob)),
            "valid_average_precision": float(average_precision_score(y_valid, valid_prob)),
        }
        reports.append(model_report)

        for probability_threshold in probability_thresholds:
            for prediction_floor in prediction_floors:
                for max_floor_probability in max_floor_probabilities:
                    corrected = _apply_high_price_correction_to_frame(
                        valid_frame,
                        probabilities=aligned_valid_prob,
                        probability_threshold=float(probability_threshold),
                        prediction_floor=float(prediction_floor),
                        max_floor_probability=(
                            None
                            if max_floor_probability is None
                            else float(max_floor_probability)
                        ),
                    )
                    metric = _price_accuracy_from_frame(corrected, "y_pred", price_floor)
                    changed = int(corrected["high_price_corrected"].sum())
                    candidate = {
                        **model_report,
                        "probability_threshold": float(probability_threshold),
                        "prediction_floor": float(prediction_floor),
                        "max_floor_probability": (
                            None
                            if max_floor_probability is None
                            else float(max_floor_probability)
                        ),
                        "mean_daily_accuracy": metric["mean_daily_accuracy"],
                        "daily_accuracy_sum": metric["daily_accuracy_sum"],
                        "overall_accuracy": metric["overall_accuracy"],
                        "changed_slots": changed,
                    }
                    key = (candidate["mean_daily_accuracy"], -changed)
                    if best is None or key > (best["mean_daily_accuracy"], -best["changed_slots"]):
                        best = candidate
                        best_model = classifier
                        best_threshold = float(target_threshold)
                        best_probabilities = aligned_valid_prob

    if best is None or best["mean_daily_accuracy"] <= base_metric["mean_daily_accuracy"]:
        return None

    report = {
        "enabled": True,
        "feature_cols": feature_cols,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()),
        "base_valid_mean_daily_accuracy": base_metric["mean_daily_accuracy"],
        "base_valid_daily_accuracy_sum": base_metric["daily_accuracy_sum"],
        "selected": best,
        "model_reports": reports,
    }
    return HighPriceCorrection(
        classifier=best_model,
        feature_cols=feature_cols,
        target_threshold=float(best_threshold),
        probability_threshold=float(best["probability_threshold"]),
        prediction_floor=float(best["prediction_floor"]),
        max_floor_probability=best["max_floor_probability"],
        report=report,
    )


def apply_floor_price_correction(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    correction: FloorPriceCorrection | None,
) -> pd.DataFrame:
    """Apply an optional trained floor-price correction to predictions."""
    if correction is None:
        return frame
    aligned_x = _aligned_features_for_predictions(
        features,
        frame,
        correction.feature_cols,
        config,
    )
    probabilities = correction.classifier.predict_proba(aligned_x)[:, 1]
    return _apply_floor_correction_to_frame(
        frame,
        probabilities=probabilities,
        probability_threshold=correction.probability_threshold,
        prediction_ceiling=correction.prediction_ceiling,
        floor_price=correction.floor_price,
    )


def apply_high_price_correction(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    correction: HighPriceCorrection | None,
) -> pd.DataFrame:
    """Apply an optional trained high-price correction to predictions."""
    if correction is None:
        return frame
    aligned_x = _aligned_features_for_predictions(
        features,
        frame,
        correction.feature_cols,
        config,
    )
    probabilities = correction.classifier.predict_proba(aligned_x)[:, 1]
    return _apply_high_price_correction_to_frame(
        frame,
        probabilities=probabilities,
        probability_threshold=correction.probability_threshold,
        prediction_floor=correction.prediction_floor,
        max_floor_probability=correction.max_floor_probability,
    )


def _date_range_days(config: Dict[str, Any], split_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    split = config["split"]
    return (
        pd.Timestamp(split[f"{split_name}_start"]).normalize(),
        pd.Timestamp(split[f"{split_name}_end"]).normalize(),
    )


def prepare_deep_windows(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[DeepFeatureSpec, List[DailyWindow], List[DailyWindow], List[DailyWindow], Dict[str, int]]:
    """Build supervised train/valid/test windows from feature rows."""
    feature_spec = infer_deep_feature_spec(features, config)
    windows = build_daily_windows(features, config, feature_spec=feature_spec, require_target=True)

    train_start, train_end = _date_range_days(config, "train")
    valid_start, valid_end = _date_range_days(config, "valid")
    test_start, test_end = _date_range_days(config, "test")
    train_windows = filter_windows_by_date_range(windows, train_start, train_end)
    valid_windows = filter_windows_by_date_range(windows, valid_start, valid_end)
    test_windows = filter_windows_by_date_range(windows, test_start, test_end)

    synthetic = make_synthetic_zero_floor_windows(train_windows, feature_spec, config)
    stats = {
        "all_supervised_windows": int(len(windows)),
        "real_train_windows": int(len(train_windows)),
        "valid_windows": int(len(valid_windows)),
        "test_windows": int(len(test_windows)),
        "synthetic_zero_floor_windows": int(len(synthetic)),
    }
    if synthetic:
        train_windows = [*train_windows, *synthetic]
    stats["total_train_windows"] = int(len(train_windows))
    return feature_spec, train_windows, valid_windows, test_windows, stats


def train_deep_model(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> DeepTrainingResult:
    """Train the default policy-aware TCN model."""
    deep_cfg = config.get("deep_model", {})
    seed = int(deep_cfg.get("seed", config.get("model", {}).get("random_state", 42)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    supply_demand_prior = None
    if supply_demand_prior_enabled(config):
        supply_demand_prior = fit_supply_demand_prior(features, config)
        features = add_supply_demand_prior_column(features, config, supply_demand_prior)
    features = add_external_anchor_columns(features, config)

    feature_spec, train_windows, valid_windows, _, window_stats = prepare_deep_windows(features, config)
    if not train_windows:
        raise ValueError("No train windows were built for deep model training.")
    if not valid_windows:
        raise ValueError("No validation windows were built for deep model training.")

    add_missing = bool(deep_cfg.get("add_missing_indicators", True))
    normalizer = fit_window_normalizer(train_windows, add_missing_indicators=add_missing)
    train_arrays = transform_windows(train_windows, normalizer, require_target=True)
    valid_arrays = transform_windows(valid_windows, normalizer, require_target=True)

    batch_size = int(deep_cfg.get("batch_size", 16))
    train_dataset = WindowTensorDataset(train_arrays, include_target=True)
    valid_dataset = WindowTensorDataset(valid_arrays, include_target=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = build_policy_aware_tcn(
        config,
        hist_dim=int(train_arrays["x_hist"].shape[-1]),
        fut_dim=int(train_arrays["x_fut"].shape[-1]),
        static_dim=int(train_arrays["x_static"].shape[-1]),
    )
    device = torch.device(deep_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(deep_cfg.get("weight_decay", 1e-4)),
    )

    max_epochs = int(deep_cfg.get("epochs", 120))
    patience = int(deep_cfg.get("patience", 20))
    min_delta = float(deep_cfg.get("min_delta", 1e-4))
    selection_metric = str(deep_cfg.get("selection_metric", "mean_daily_accuracy"))
    best_state = None
    best_loss = float("inf")
    best_metric = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        valid_loss = evaluate_loss(model, valid_loader, config, device)
        valid_pred = predict_from_loader(model, valid_loader, device)
        valid_frame = windows_to_prediction_frame(valid_arrays, valid_pred, config)
        valid_eval = evaluate_predictions(valid_frame, config)
        valid_metric = float(valid_eval["summary"].get("mean_daily_accuracy") or -float("inf"))
        valid_weighted_metric = _weighted_accuracy_from_frame(
            valid_frame,
            "y_pred",
            price_floor=float(config.get("metrics", {}).get("price_floor", 40.0)),
            high_boost=float(deep_cfg.get("selection_high_price_weight_boost", 0.0)),
            cap_boost=float(deep_cfg.get("selection_cap_price_weight_boost", 0.0)),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_mean_daily_accuracy": valid_metric,
                "valid_weighted_accuracy": valid_weighted_metric,
            }
        )

        if selection_metric == "valid_loss":
            improved = valid_loss < best_loss - min_delta
        elif selection_metric == "mean_daily_accuracy":
            improved = valid_metric > best_metric + min_delta
        elif selection_metric == "weighted_accuracy":
            improved = valid_weighted_metric > best_metric + min_delta
            valid_metric = valid_weighted_metric
        else:
            raise ValueError(f"Unsupported deep_model.selection_metric: {selection_metric}")

        if improved:
            best_loss = valid_loss
            best_metric = valid_metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_pred = predict_arrays(model, valid_arrays, config, device=device)
    valid_frame = windows_to_prediction_frame(valid_arrays, valid_pred, config)
    valid_frame = _attach_external_anchor_prediction_columns(features, valid_frame, config)
    residual_blend = train_residual_blend_correction(valid_frame, config)
    valid_frame = apply_residual_blend_correction(valid_frame, config, residual_blend)
    floor_correction = train_floor_price_correction(features, config, valid_frame)
    valid_frame = apply_floor_price_correction(features, valid_frame, config, floor_correction)
    high_correction = train_high_price_correction(features, config, valid_frame)
    valid_frame = apply_high_price_correction(features, valid_frame, config, high_correction)
    valid_eval = evaluate_predictions(valid_frame, config)
    report = {
        "model_type": "policy_aware_tcn",
        "feature_spec": {
            "hist_cols": feature_spec.hist_cols,
            "fut_cols": feature_spec.fut_cols,
            "static_cols": feature_spec.static_cols,
        },
        "normalizer": normalizer.to_dict(),
        "architecture": module_summary(model),
        "training": {
            "seed": seed,
            "epochs_ran": len(history),
            "best_valid_loss": best_loss,
            "best_validation_mean_daily_accuracy": best_metric,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "batch_size": batch_size,
            **window_stats,
            "history": history,
        },
        "validation_summary": valid_eval["summary"],
        "config_deep_model": deep_cfg,
    }
    if floor_correction is not None:
        report["floor_price_correction"] = floor_correction.report
    else:
        report["floor_price_correction"] = {"enabled": False}
    if high_correction is not None:
        report["high_price_correction"] = high_correction.report
    else:
        report["high_price_correction"] = {"enabled": False}
    if residual_blend is not None:
        report["residual_blend_correction"] = residual_blend.report
    else:
        report["residual_blend_correction"] = {"enabled": False}
    if supply_demand_prior is not None:
        report["supply_demand_prior"] = supply_demand_prior.report
    else:
        report["supply_demand_prior"] = {"enabled": False}
    return DeepTrainingResult(
        model=model,
        feature_spec=feature_spec,
        normalizer=normalizer,
        train_report=report,
        valid_predictions=valid_frame,
        floor_price_correction=floor_correction,
        high_price_correction=high_correction,
        residual_blend_correction=residual_blend,
        supply_demand_prior=supply_demand_prior,
    )


def save_deep_model(result: DeepTrainingResult, config: Dict[str, Any]) -> None:
    """Save model weights and metadata."""
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), model_dir / "policy_aware_tcn.pt")
    save_supply_demand_prior(result.supply_demand_prior, config)
    if result.floor_price_correction is not None:
        with (model_dir / "floor_price_correction.pkl").open("wb") as file:
            pickle.dump(result.floor_price_correction, file)
    if result.high_price_correction is not None:
        with (model_dir / "high_price_correction.pkl").open("wb") as file:
            pickle.dump(result.high_price_correction, file)
    if result.residual_blend_correction is not None:
        with (model_dir / "residual_blend_correction.pkl").open("wb") as file:
            pickle.dump(result.residual_blend_correction, file)
    with (model_dir / "policy_aware_tcn_report.json").open("w", encoding="utf-8") as file:
        json.dump(result.train_report, file, ensure_ascii=False, indent=2)


def load_floor_price_correction(config: Dict[str, Any]) -> FloorPriceCorrection | None:
    """Load the optional floor-price correction artifact."""
    path = Path(config["paths"]["model_dir"]) / "floor_price_correction.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def load_high_price_correction(config: Dict[str, Any]) -> HighPriceCorrection | None:
    """Load the optional high-price correction artifact."""
    path = Path(config["paths"]["model_dir"]) / "high_price_correction.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def load_residual_blend_correction(config: Dict[str, Any]) -> ResidualBlendCorrection | None:
    """Load the optional residual-blend correction artifact."""
    path = Path(config["paths"]["model_dir"]) / "residual_blend_correction.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def load_deep_model(config: Dict[str, Any]) -> tuple[nn.Module, DeepFeatureSpec, Any]:
    """Load a trained policy-aware TCN and its feature metadata."""
    from epf_tcn.deep_data import WindowNormalizer

    model_dir = Path(config["paths"]["model_dir"])
    with (model_dir / "policy_aware_tcn_report.json").open("r", encoding="utf-8") as file:
        report = json.load(file)
    spec_payload = report["feature_spec"]
    feature_spec = DeepFeatureSpec(
        hist_cols=list(spec_payload["hist_cols"]),
        fut_cols=list(spec_payload["fut_cols"]),
        static_cols=list(spec_payload["static_cols"]),
    )
    normalizer = WindowNormalizer.from_dict(report["normalizer"])
    arch = report["architecture"]["config"]
    model = build_policy_aware_tcn(
        _config_with_arch(config, arch),
        hist_dim=int(arch["hist_dim"]),
        fut_dim=int(arch["fut_dim"]),
        static_dim=int(arch["static_dim"]),
    )
    state = torch.load(model_dir / "policy_aware_tcn.pt", map_location="cpu")
    model.load_state_dict(state)
    return model, feature_spec, normalizer


def _config_with_arch(config: Dict[str, Any], arch: Mapping[str, Any]) -> Dict[str, Any]:
    patched = dict(config)
    deep_cfg = dict(patched.get("deep_model", {}))
    deep_cfg["horizon"] = int(arch["horizon"])
    deep_cfg["tcn"] = {
        "channels": int(arch["channels"]),
        "kernel_size": int(arch["kernel_size"]),
        "dilations": list(arch["dilations"]),
        "dropout": float(arch["dropout"]),
        "future_layers": int(arch["future_layers"]),
        "static_hidden": int(arch["static_hidden"]),
        "residual_scale": float(arch["residual_scale"]),
        "history_context_mode": str(arch.get("history_context_mode", "last")),
        "output_mode": str(arch.get("output_mode", "residual")),
        "direct_output_scale": float(arch.get("direct_output_scale", 1000.0)),
        "output_sigma": bool(arch["output_sigma"]),
    }
    patched["deep_model"] = deep_cfg
    return patched


def evaluate_trained_deep_model(
    features: pd.DataFrame,
    config: Dict[str, Any],
    prefix: str = "policy_aware_tcn",
) -> Dict[str, Any]:
    """Load, predict the test split, save predictions and reports."""
    model, feature_spec, normalizer = load_deep_model(config)
    prior = load_supply_demand_prior(config)
    if prior is not None:
        features = add_supply_demand_prior_column(features, config, prior)
    features = add_external_anchor_columns(features, config)
    windows = build_daily_windows(features, config, feature_spec=feature_spec, require_target=True)
    test_start, test_end = _date_range_days(config, "test")
    test_windows = filter_windows_by_date_range(windows, test_start, test_end)
    arrays = transform_windows(test_windows, normalizer, require_target=True)
    pred = predict_arrays(model, arrays, config)
    frame = windows_to_prediction_frame(arrays, pred, config)
    frame = _attach_external_anchor_prediction_columns(features, frame, config)
    frame = apply_residual_blend_correction(
        frame,
        config,
        correction=load_residual_blend_correction(config),
    )
    frame = apply_floor_price_correction(
        features,
        frame,
        config,
        correction=load_floor_price_correction(config),
    )
    frame = apply_high_price_correction(
        features,
        frame,
        config,
        correction=load_high_price_correction(config),
    )
    evaluation = evaluate_predictions(frame, config)
    save_evaluation(frame, evaluation, config, prefix=prefix)
    return evaluation


def predict_forecast_day(
    features: pd.DataFrame,
    config: Dict[str, Any],
    target_day: pd.Timestamp,
    prefix: str = "policy_aware_tcn_future_24h",
) -> Dict[str, Any]:
    """Predict one configured day that may not have target labels."""
    model, feature_spec, normalizer = load_deep_model(config)
    prior = load_supply_demand_prior(config)
    if prior is not None:
        features = add_supply_demand_prior_column(features, config, prior)
    features = add_external_anchor_columns(features, config)
    windows = build_daily_windows(
        features,
        config,
        feature_spec=feature_spec,
        target_days=[target_day],
        require_target=False,
    )
    if not windows:
        raise ValueError(f"No forecast window can be built for {target_day.date()}.")
    arrays = transform_windows(windows, normalizer, require_target=False)
    pred = predict_arrays(model, arrays, config)
    frame = windows_to_prediction_frame(arrays, pred, config)
    frame = _attach_external_anchor_prediction_columns(features, frame, config)
    frame = apply_residual_blend_correction(
        frame,
        config,
        correction=load_residual_blend_correction(config),
    )
    frame = apply_floor_price_correction(
        features,
        frame,
        config,
        correction=load_floor_price_correction(config),
    )
    frame = apply_high_price_correction(
        features,
        frame,
        config,
        correction=load_high_price_correction(config),
    )

    pred_dir = Path(config["paths"]["prediction_dir"])
    report_dir = Path(config["paths"]["report_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = pred_dir / f"{prefix}_predictions.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary = {
        "target_date": str(pd.Timestamp(target_day).date()),
        "n_rows": int(frame.shape[0]),
        "n_predicted": int(frame["y_pred"].notna().sum()),
        "n_missing_target": int(frame["y_true"].isna().sum()),
        "expected_slots": int(config["data"].get("expected_slots_per_day", 96)),
        "output_file": str(output_path),
    }
    with (report_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary

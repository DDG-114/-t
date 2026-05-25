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
    build_daily_windows,
    filter_windows_by_date_range,
    fit_window_normalizer,
    infer_deep_feature_spec,
    make_synthetic_zero_floor_windows,
    transform_windows,
)
from epf_tcn.deep_models import build_policy_aware_tcn, module_summary, require_torch
from epf_tcn.evaluate import evaluate_predictions, save_evaluation

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


@dataclass
class FloorPriceCorrection:
    """Validation-selected postprocessor for persistent floor-price regimes."""

    classifier: Any
    feature_cols: List[str]
    floor_price: float
    probability_threshold: float
    prediction_ceiling: float | None
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
            rows.append(
                {
                    "Date": target_day + pd.Timedelta(minutes=15 * slot),
                    "date": target_day.date().isoformat(),
                    "slot": slot,
                    "y_true": np.nan if not observed else float(y_true[slot]),
                    "y_pred": float(predictions[day_index, slot]),
                    "policy_regime": float(arrays["policy_regime"][day_index]),
                    "price_lower_bound": float(arrays["lower_bound"][day_index]),
                    "price_upper_bound": float(arrays["upper_bound"][day_index]),
                }
            )
    return pd.DataFrame(rows)


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
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_mean_daily_accuracy": valid_metric,
            }
        )

        if selection_metric == "valid_loss":
            improved = valid_loss < best_loss - min_delta
        elif selection_metric == "mean_daily_accuracy":
            improved = valid_metric > best_metric + min_delta
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
    floor_correction = train_floor_price_correction(features, config, valid_frame)
    valid_frame = apply_floor_price_correction(features, valid_frame, config, floor_correction)
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
    return DeepTrainingResult(
        model=model,
        feature_spec=feature_spec,
        normalizer=normalizer,
        train_report=report,
        valid_predictions=valid_frame,
        floor_price_correction=floor_correction,
    )


def save_deep_model(result: DeepTrainingResult, config: Dict[str, Any]) -> None:
    """Save model weights and metadata."""
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), model_dir / "policy_aware_tcn.pt")
    if result.floor_price_correction is not None:
        with (model_dir / "floor_price_correction.pkl").open("wb") as file:
            pickle.dump(result.floor_price_correction, file)
    with (model_dir / "policy_aware_tcn_report.json").open("w", encoding="utf-8") as file:
        json.dump(result.train_report, file, ensure_ascii=False, indent=2)


def load_floor_price_correction(config: Dict[str, Any]) -> FloorPriceCorrection | None:
    """Load the optional floor-price correction artifact."""
    path = Path(config["paths"]["model_dir"]) / "floor_price_correction.pkl"
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
    windows = build_daily_windows(features, config, feature_spec=feature_spec, require_target=True)
    test_start, test_end = _date_range_days(config, "test")
    test_windows = filter_windows_by_date_range(windows, test_start, test_end)
    arrays = transform_windows(test_windows, normalizer, require_target=True)
    pred = predict_arrays(model, arrays, config)
    frame = windows_to_prediction_frame(arrays, pred, config)
    frame = apply_floor_price_correction(
        features,
        frame,
        config,
        correction=load_floor_price_correction(config),
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
    frame = apply_floor_price_correction(
        features,
        frame,
        config,
        correction=load_floor_price_correction(config),
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

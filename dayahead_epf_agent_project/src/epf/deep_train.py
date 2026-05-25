"""Training and inference utilities for report2 deep EPF models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from epf.deep_data import (
    DeepFeatureSpec,
    DailyWindow,
    build_daily_windows,
    filter_windows_by_date_range,
    fit_window_normalizer,
    infer_deep_feature_spec,
    make_synthetic_zero_floor_windows,
    transform_windows,
)
from epf.deep_models import build_policy_aware_tcn, module_summary, require_torch
from epf.evaluate import evaluate_predictions, save_evaluation

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
        if self.include_target and self.arrays.get("y") is not None:
            item["y"] = torch.as_tensor(self.arrays["y"][index], dtype=torch.float32)
            item["y_mask"] = torch.as_tensor(self.arrays["y_mask"][index], dtype=torch.float32)
        return item


def make_point_weights(y_true: torch.Tensor) -> torch.Tensor:
    """Report2-aligned point weights for high, low, and boundary prices."""
    weights = torch.ones_like(y_true)
    weights = weights + 0.50 * (y_true >= 800.0).float()
    weights = weights + 0.40 * (y_true <= 60.0).float()
    weights = weights + 0.30 * ((y_true >= 950.0) | (y_true <= 45.0)).float()
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)


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
) -> torch.Tensor:
    """Combine business loss, ramp regularization, and optional NLL."""
    deep_cfg = config.get("deep_model", {})
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    lambda_ramp = float(deep_cfg.get("lambda_ramp", 0.10))
    loss = weighted_relative_mae_loss(
        model_output["mu"],
        target,
        sample_weight=sample_weight,
        point_weight=make_point_weights(target),
        target_mask=target_mask,
        denom_floor=price_floor,
        lambda_ramp=lambda_ramp,
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
        )
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


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
    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        valid_loss = evaluate_loss(model, valid_loader, config, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
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
            "batch_size": batch_size,
            **window_stats,
            "history": history,
        },
        "validation_summary": valid_eval["summary"],
        "config_deep_model": deep_cfg,
    }
    return DeepTrainingResult(
        model=model,
        feature_spec=feature_spec,
        normalizer=normalizer,
        train_report=report,
        valid_predictions=valid_frame,
    )


def save_deep_model(result: DeepTrainingResult, config: Dict[str, Any]) -> None:
    """Save model weights and metadata."""
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), model_dir / "policy_aware_tcn.pt")
    with (model_dir / "policy_aware_tcn_report.json").open("w", encoding="utf-8") as file:
        json.dump(result.train_report, file, ensure_ascii=False, indent=2)


def load_deep_model(config: Dict[str, Any]) -> tuple[nn.Module, DeepFeatureSpec, Any]:
    """Load a trained policy-aware TCN and its feature metadata."""
    from epf.deep_data import WindowNormalizer

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

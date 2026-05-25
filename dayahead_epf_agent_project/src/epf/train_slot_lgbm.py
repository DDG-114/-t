"""Slot-wise LightGBM training and prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from epf.features import get_feature_columns
from epf.metrics import regression_summary


def _import_lightgbm():
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "lightgbm is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return LGBMRegressor


def clean_training_frame(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Remove rows that cannot be used for supervised training.

    LightGBM can route missing feature values natively, so only the supervised
    target must be present. Infinite feature values are converted to NaN.
    """
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=[target_col]).copy()
    return result


def make_sample_weight(
    target: pd.Series,
    config: Dict[str, Any],
) -> np.ndarray | None:
    """Build sample weights aligned with the configured metric.

    ``inverse_target_floor`` approximates relative-error optimization by
    weighting absolute errors with ``1 / max(abs(y), price_floor)``.
    """
    weight_config = config.get("model", {}).get("sample_weight", {})
    if not weight_config.get("enabled", False):
        return None

    mode = weight_config.get("mode", "inverse_target_floor")
    price_floor = config.get("metrics", {}).get("price_floor", 40.0)
    if mode == "inverse_target_floor":
        denominator = np.maximum(np.abs(target.to_numpy(dtype=float)), price_floor)
        return 1.0 / denominator
    raise ValueError(f"Unsupported sample weight mode: {mode}")


def train_one_slot_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
):
    """Train one LightGBM model for a single delivery slot."""
    LGBMRegressor = _import_lightgbm()
    model = LGBMRegressor(**params)

    train_clean = clean_training_frame(train_df, feature_cols, target_col)
    valid_clean = clean_training_frame(valid_df, feature_cols, target_col)

    if train_clean.empty:
        raise ValueError("No valid training rows for this slot after dropping missing features.")

    x_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    train_weight = make_sample_weight(y_train, config)

    if valid_clean.empty:
        model.fit(x_train, y_train, sample_weight=train_weight)
    else:
        x_valid = valid_clean[feature_cols]
        y_valid = valid_clean[target_col]
        valid_weight = make_sample_weight(y_valid, config)
        fit_kwargs: Dict[str, Any] = {
            "sample_weight": train_weight,
            "eval_set": [(x_valid, y_valid)],
            "eval_metric": "l1",
        }
        if valid_weight is not None:
            fit_kwargs["eval_sample_weight"] = [valid_weight]
        model.fit(x_train, y_train, **fit_kwargs)

    return model


def train_slot_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Dict[int, Any], List[str], Dict[str, Any]]:
    """Train 96 slot-specific LightGBM models."""
    target = config["columns"]["target"]
    expected_slots = config["data"].get("expected_slots_per_day", 96)
    params = config["model"].get("lightgbm_params", {})
    feature_cols = get_feature_columns(train_df, config)

    models = {}
    train_report: Dict[str, Any] = {"slots": {}, "feature_columns": feature_cols}

    for slot in range(expected_slots):
        slot_train = train_df[train_df["slot"] == slot]
        slot_valid = valid_df[valid_df["slot"] == slot]

        if slot_train.empty:
            train_report["slots"][str(slot)] = {"status": "skipped_no_train_rows"}
            continue

        try:
            model = train_one_slot_model(
                slot_train,
                slot_valid,
                feature_cols,
                target,
                params=params,
                config=config,
            )
            models[slot] = model

            valid_clean = clean_training_frame(slot_valid, feature_cols, target)
            if not valid_clean.empty:
                pred = model.predict(valid_clean[feature_cols])
                metrics = regression_summary(
                    valid_clean[target],
                    pred,
                    price_floor=config["metrics"].get("price_floor", 40.0),
                )
            else:
                metrics = {}

            train_report["slots"][str(slot)] = {
                "status": "trained",
                "train_rows": int(slot_train.shape[0]),
                "valid_rows": int(slot_valid.shape[0]),
                "sample_weight": config.get("model", {}).get("sample_weight", {}),
                "metrics_valid": metrics,
            }
        except Exception as exc:  # Keep training other slots.
            train_report["slots"][str(slot)] = {"status": "failed", "error": str(exc)}

    return models, feature_cols, train_report


def save_slot_models(
    models: Dict[int, Any],
    feature_cols: List[str],
    report: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Save trained slot models and metadata."""
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    for slot, model in models.items():
        joblib.dump(model, model_dir / f"lgbm_slot_{slot:02d}.pkl")

    with (model_dir / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_cols, file, ensure_ascii=False, indent=2)

    with (model_dir / "train_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def load_slot_models(config: Dict[str, Any]) -> Tuple[Dict[int, Any], List[str]]:
    """Load slot models and feature column metadata."""
    model_dir = Path(config["paths"]["model_dir"])
    with (model_dir / "feature_columns.json").open("r", encoding="utf-8") as file:
        feature_cols = json.load(file)

    models = {}
    for path in sorted(model_dir.glob("lgbm_slot_*.pkl")):
        slot = int(path.stem.split("_")[-1])
        models[slot] = joblib.load(path)
    return models, feature_cols


def predict_with_slot_models(
    df: pd.DataFrame,
    models: Dict[int, Any],
    feature_cols: List[str],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Predict prices for rows using their slot-specific model."""
    target = config["columns"]["target"]
    dt_col = config["columns"]["datetime"]
    id_cols = ["date", "slot"]
    if dt_col in df.columns:
        id_cols.insert(0, dt_col)
    predictions = df[id_cols + [target]].copy().rename(columns={target: "y_true"})
    predictions["y_pred"] = np.nan

    clean_df = df.copy()
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    clean_df[numeric_cols] = clean_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    for slot, model in models.items():
        mask = clean_df["slot"] == slot
        slot_rows = clean_df.loc[mask]
        if slot_rows.empty:
            continue
        predictions.loc[mask, "y_pred"] = model.predict(clean_df.loc[mask, feature_cols])

    clip_min = config.get("model", {}).get("clip_prediction_min", 0.0)
    clip_max = config.get("model", {}).get("clip_prediction_max")
    predictions["y_pred"] = predictions["y_pred"].clip(lower=clip_min, upper=clip_max)
    return predictions

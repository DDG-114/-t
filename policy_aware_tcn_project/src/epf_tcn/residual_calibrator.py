"""Residual calibration layer for state-aware GBM predictions."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.metrics import regression_summary
from epf_tcn.state_gbm import (
    StateGBMModel,
    _apply_state_overrides,
    _date_range_days,
    _floor_override_value,
    _matrix,
    _prediction_max,
    _prediction_min,
    _predict_state_probabilities,
    _split_frame,
    add_state_gbm_features,
)

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError("residual_calibrator requires lightgbm.") from exc


@dataclass
class ResidualCalibrationParams:
    """Validation-selected residual correction controls."""

    shrink: float
    clip_value: float
    min_prediction: float
    max_floor_probability: float
    positive_only: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shrink": float(self.shrink),
            "clip_value": float(self.clip_value),
            "min_prediction": float(self.min_prediction),
            "max_floor_probability": float(self.max_floor_probability),
            "positive_only": bool(self.positive_only),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualCalibrationParams":
        return cls(
            shrink=float(payload["shrink"]),
            clip_value=float(payload["clip_value"]),
            min_prediction=float(payload["min_prediction"]),
            max_floor_probability=float(payload["max_floor_probability"]),
            positive_only=bool(payload["positive_only"]),
        )


@dataclass
class ResidualCalibratorModel:
    """A fitted residual correction model for state-aware GBM outputs."""

    feature_cols: List[str]
    feature_medians: Dict[str, float]
    residual_model: Any
    params: ResidualCalibrationParams
    report: Dict[str, Any]


def _calibrator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config.get("residual_calibrator", {}))
    cfg.setdefault("objective", "regression")
    cfg.setdefault("n_estimators", 500)
    cfg.setdefault("learning_rate", 0.03)
    cfg.setdefault("num_leaves", 15)
    cfg.setdefault("min_child_samples", 30)
    cfg.setdefault("subsample", 0.9)
    cfg.setdefault("colsample_bytree", 0.7)
    cfg.setdefault("reg_lambda", 5.0)
    cfg.setdefault("early_stopping_rounds", 50)
    cfg.setdefault("selection_tolerance", 0.0)
    cfg.setdefault("shrink_grid", [0.5, 0.75, 1.0])
    cfg.setdefault("clip_grid", [160, 220, 300])
    cfg.setdefault("min_prediction_grid", [40, 80, 150])
    cfg.setdefault("max_floor_probability_grid", [0.2, 0.4, 0.8, 1.1])
    cfg.setdefault("positive_only_grid", [False])
    cfg.setdefault("refit_after_validation", False)
    return cfg


def _prediction_feature_frame(
    frame: pd.DataFrame,
    state_model: StateGBMModel,
    config: Dict[str, Any],
) -> pd.DataFrame:
    x = _matrix(frame, state_model.feature_cols, state_model.feature_medians)
    base_pred = np.clip(
        state_model.regressor.predict(x),
        _prediction_min(config),
        _prediction_max(config),
    )
    probabilities = _predict_state_probabilities(state_model, x)
    state_pred = _apply_state_overrides(
        base_pred,
        probabilities,
        state_model.thresholds,
        floor_price=float(config.get("metrics", {}).get("price_floor", 40.0)),
        prediction_min=_prediction_min(config),
        prediction_max=_prediction_max(config),
        floor_override_value=_floor_override_value(config),
    )
    result = frame.copy()
    result["base_pred"] = base_pred
    result["state_pred"] = state_pred
    result["floor_probability"] = probabilities["floor"]
    result["high_probability"] = probabilities["high"]
    result["cap_probability"] = probabilities["cap"]
    return result


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


def _feature_cols(frame: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
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
        "residual",
    }
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def _fit_medians(frame: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    medians = frame[list(cols)].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    medians = medians.fillna(0.0)
    return {col: float(medians.get(col, 0.0)) for col in cols}


def _matrix_from_cols(
    frame: pd.DataFrame,
    cols: Sequence[str],
    medians: Mapping[str, float],
) -> pd.DataFrame:
    result = frame[list(cols)].replace([np.inf, -np.inf], np.nan).copy()
    for col in cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(float(medians[col]))
    return result


def _residual_weights(frame: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    target = config["columns"]["target"]
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    y = frame[target].to_numpy(dtype=float)
    weights = 1.0 / np.maximum(np.abs(y), price_floor)
    mean = float(np.nanmean(weights))
    return weights if mean <= 0 else weights / mean


def _fit_residual_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    sample_weight: np.ndarray,
    config: Dict[str, Any],
    n_estimators: int | None = None,
) -> Any:
    cfg = _calibrator_config(config)
    model = lgb.LGBMRegressor(
        objective=str(cfg["objective"]),
        n_estimators=int(n_estimators or cfg["n_estimators"]),
        learning_rate=float(cfg["learning_rate"]),
        num_leaves=int(cfg["num_leaves"]),
        min_child_samples=int(cfg["min_child_samples"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        random_state=int(config.get("deep_model", {}).get("seed", 42)) + 37,
        verbose=-1,
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    if x_valid is not None and y_valid is not None:
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(int(cfg["early_stopping_rounds"]), verbose=False),
            ],
        )
    else:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def _apply_residual_correction(
    frame: pd.DataFrame,
    residual_pred: np.ndarray,
    params: ResidualCalibrationParams,
    floor_price: float,
    prediction_min: float | None = None,
    prediction_max: float = 1000.0,
) -> np.ndarray:
    state_pred = frame["state_pred"].to_numpy(dtype=float)
    correction = np.clip(
        np.asarray(residual_pred, dtype=float),
        -float(params.clip_value),
        float(params.clip_value),
    )
    correction *= float(params.shrink)
    if params.positive_only:
        correction = np.maximum(correction, 0.0)
    mask = (
        (state_pred >= float(params.min_prediction))
        & (frame["floor_probability"].to_numpy(dtype=float) <= float(params.max_floor_probability))
    )
    pred = state_pred.copy()
    pred[mask] = pred[mask] + correction[mask]
    lower = float(floor_price if prediction_min is None else prediction_min)
    return np.clip(pred, lower, float(prediction_max))


def _select_params(
    valid_frame: pd.DataFrame,
    residual_pred: np.ndarray,
    config: Dict[str, Any],
) -> tuple[ResidualCalibrationParams, Dict[str, Any]]:
    cfg = _calibrator_config(config)
    target = config["columns"]["target"]
    price_floor = float(config.get("metrics", {}).get("price_floor", 40.0))
    base_summary = regression_summary(
        valid_frame[target],
        valid_frame["state_pred"],
        price_floor,
    )
    best_score = -float("inf")
    best_params: ResidualCalibrationParams | None = None
    candidates = []
    for shrink in cfg["shrink_grid"]:
        for clip_value in cfg["clip_grid"]:
            for min_prediction in cfg["min_prediction_grid"]:
                for max_floor_probability in cfg["max_floor_probability_grid"]:
                    for positive_only in cfg["positive_only_grid"]:
                        params = ResidualCalibrationParams(
                            shrink=float(shrink),
                            clip_value=float(clip_value),
                            min_prediction=float(min_prediction),
                            max_floor_probability=float(max_floor_probability),
                            positive_only=bool(positive_only),
                        )
                        pred = _apply_residual_correction(
                            valid_frame,
                            residual_pred,
                            params,
                            price_floor,
                            prediction_min=_prediction_min(config),
                            prediction_max=_prediction_max(config),
                        )
                        summary = regression_summary(valid_frame[target], pred, price_floor)
                        row = {**params.to_dict(), **summary}
                        candidates.append(row)
                        score = float(summary["accuracy"])
                        if score > best_score:
                            best_score = score
                            best_params = params
    if best_params is None:
        raise RuntimeError("No residual calibration candidate was evaluated.")
    tolerance = float(cfg.get("selection_tolerance", 0.0))
    if best_score < float(base_summary["accuracy"]) + tolerance:
        best_params = ResidualCalibrationParams(
            shrink=0.0,
            clip_value=0.0,
            min_prediction=1000.0,
            max_floor_probability=0.0,
            positive_only=False,
        )
        best_score = float(base_summary["accuracy"])
    report = {
        "base_validation_summary": base_summary,
        "selected": {**best_params.to_dict(), "accuracy": best_score},
        "top_candidates": sorted(
            candidates,
            key=lambda item: float(item["accuracy"]),
            reverse=True,
        )[:10],
    }
    return best_params, report


def train_residual_calibrator(
    state_model: StateGBMModel,
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> ResidualCalibratorModel:
    """Train residual calibrator with train split and select params on valid."""
    enhanced = add_state_gbm_features(features, config)
    train = _split_frame(enhanced, config, "train")
    valid = _split_frame(enhanced, config, "valid")
    final_train = _final_train_frame(enhanced, config)
    target = config["columns"]["target"]

    train_pred = _prediction_feature_frame(train, state_model, config)
    valid_pred = _prediction_feature_frame(valid, state_model, config)
    final_pred = _prediction_feature_frame(final_train, state_model, config)
    for frame in [train_pred, valid_pred, final_pred]:
        frame["residual"] = frame[target].to_numpy(dtype=float) - frame["state_pred"].to_numpy(dtype=float)

    cols = _feature_cols(train_pred, config)
    medians = _fit_medians(train_pred, cols)
    x_train = _matrix_from_cols(train_pred, cols, medians)
    x_valid = _matrix_from_cols(valid_pred, cols, medians)
    y_train = train_pred["residual"].to_numpy(dtype=float)
    y_valid = valid_pred["residual"].to_numpy(dtype=float)

    residual_model = _fit_residual_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        _residual_weights(train_pred, config),
        config,
    )
    valid_residual_pred = residual_model.predict(x_valid)
    params, selection_report = _select_params(valid_pred, valid_residual_pred, config)

    if bool(_calibrator_config(config).get("refit_after_validation", False)):
        final_medians = _fit_medians(final_pred, cols)
        x_final = _matrix_from_cols(final_pred, cols, final_medians)
        final_model = _fit_residual_model(
            x_final,
            final_pred["residual"].to_numpy(dtype=float),
            None,
            None,
            _residual_weights(final_pred, config),
            config,
            n_estimators=int(getattr(residual_model, "best_iteration_", None) or 300),
        )
        output_medians = final_medians
        output_rows = int(final_pred.shape[0])
    else:
        final_model = residual_model
        output_medians = medians
        output_rows = int(train_pred.shape[0])
    report = {
        "model_type": "state_gbm_residual_calibrator",
        "n_features": int(len(cols)),
        "train_rows": int(train_pred.shape[0]),
        "valid_rows": int(valid_pred.shape[0]),
        "final_train_rows": int(final_pred.shape[0]),
        "calibrator_fit_rows": output_rows,
        "refit_after_validation": bool(_calibrator_config(config).get("refit_after_validation", False)),
        "residual_best_iteration": int(getattr(residual_model, "best_iteration_", None) or 300),
        "selection": selection_report,
    }
    return ResidualCalibratorModel(
        feature_cols=cols,
        feature_medians=output_medians,
        residual_model=final_model,
        params=params,
        report=report,
    )


def predict_residual_calibrated(
    state_model: StateGBMModel,
    calibrator: ResidualCalibratorModel,
    features: pd.DataFrame,
    config: Dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    """Predict one split with state GBM plus residual calibration."""
    enhanced = add_state_gbm_features(features, config)
    frame = _split_frame(enhanced, config, split_name)
    pred_frame = _prediction_feature_frame(frame, state_model, config)
    x = _matrix_from_cols(pred_frame, calibrator.feature_cols, calibrator.feature_medians)
    residual_pred = calibrator.residual_model.predict(x)
    y_pred = _apply_residual_correction(
        pred_frame,
        residual_pred,
        calibrator.params,
        float(config.get("metrics", {}).get("price_floor", 40.0)),
        prediction_min=_prediction_min(config),
        prediction_max=_prediction_max(config),
    )
    target = config["columns"]["target"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(pred_frame[config["columns"]["datetime"]]),
            "date": pd.to_datetime(pred_frame["date"]).dt.date.astype(str),
            "slot": pred_frame["slot"].to_numpy(dtype=int),
            "y_true": pred_frame[target].to_numpy(dtype=float),
            "base_pred": pred_frame["base_pred"].to_numpy(dtype=float),
            "state_pred": pred_frame["state_pred"].to_numpy(dtype=float),
            "residual_pred": np.asarray(residual_pred, dtype=float),
            "floor_probability": pred_frame["floor_probability"].to_numpy(dtype=float),
            "high_probability": pred_frame["high_probability"].to_numpy(dtype=float),
            "cap_probability": pred_frame["cap_probability"].to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def evaluate_residual_calibrated(
    state_model: StateGBMModel,
    calibrator: ResidualCalibratorModel,
    features: pd.DataFrame,
    config: Dict[str, Any],
    split_name: str = "test",
) -> Dict[str, Any]:
    predictions = predict_residual_calibrated(
        state_model,
        calibrator,
        features,
        config,
        split_name=split_name,
    )
    return {"predictions": predictions, "evaluation": evaluate_predictions(predictions, config)}


def save_residual_calibrator(calibrator: ResidualCalibratorModel, config: Dict[str, Any]) -> None:
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "residual_calibrator.pkl").open("wb") as file:
        pickle.dump(calibrator, file)
    with (model_dir / "residual_calibrator_report.json").open("w", encoding="utf-8") as file:
        json.dump(calibrator.report, file, ensure_ascii=False, indent=2)


def load_residual_calibrator(config: Dict[str, Any]) -> ResidualCalibratorModel:
    with (Path(config["paths"]["model_dir"]) / "residual_calibrator.pkl").open("rb") as file:
        return pickle.load(file)


def save_residual_calibrated_evaluation(
    predictions: pd.DataFrame,
    evaluation: Dict[str, Any],
    config: Dict[str, Any],
    prefix: str = "state_gbm_residual",
) -> None:
    save_evaluation(predictions, evaluation, config, prefix=prefix)

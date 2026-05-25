"""Online residual adaptation for sequential day-ahead evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from epf_tcn.evaluate import evaluate_predictions, save_evaluation


@dataclass
class OnlineResidualConfig:
    """Configuration for leakage-safe online residual correction."""

    min_history_days: int = 7
    window_days: int = 7
    group: str = "pred_bin"
    shrink: float = 0.5
    clip_value: float = 160.0
    min_prediction: float = 60.0
    max_floor_probability: float = 0.8
    prediction_bins: tuple[float, ...] = (40, 60, 100, 160, 240, 360, 500, 1000)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OnlineResidualConfig":
        cfg = dict(config.get("online_residual_calibrator", {}))
        return cls(
            min_history_days=int(cfg.get("min_history_days", 7)),
            window_days=int(cfg.get("window_days", 7)),
            group=str(cfg.get("group", "pred_bin")),
            shrink=float(cfg.get("shrink", 0.5)),
            clip_value=float(cfg.get("clip_value", 160.0)),
            min_prediction=float(cfg.get("min_prediction", 60.0)),
            max_floor_probability=float(cfg.get("max_floor_probability", 0.8)),
            prediction_bins=tuple(float(value) for value in cfg.get(
                "prediction_bins",
                [40, 60, 100, 160, 240, 360, 500, 1000],
            )),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_history_days": self.min_history_days,
            "window_days": self.window_days,
            "group": self.group,
            "shrink": self.shrink,
            "clip_value": self.clip_value,
            "min_prediction": self.min_prediction,
            "max_floor_probability": self.max_floor_probability,
            "prediction_bins": list(self.prediction_bins),
        }


def apply_online_residual_correction(
    predictions: pd.DataFrame,
    config: OnlineResidualConfig,
) -> pd.DataFrame:
    """Correct predictions using only earlier days in the same evaluation stream."""
    required = {
        "date",
        "slot",
        "y_true",
        "y_pred",
        "floor_probability",
        "high_probability",
        "cap_probability",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Online correction missing columns: {sorted(missing)}")

    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["date", "slot"]).reset_index(drop=True)
    base_pred = result["y_pred"].to_numpy(dtype=float)
    corrected = base_pred.copy()
    dates = list(pd.unique(result["date"]))
    bins = np.asarray(config.prediction_bins, dtype=float)

    for day_index, date in enumerate(dates):
        day_mask = result["date"].eq(date).to_numpy()
        if day_index < config.min_history_days:
            continue

        history = result[result["date"] < date].copy()
        if config.window_days > 0:
            history = history[
                history["date"] >= date - pd.Timedelta(days=config.window_days)
            ]
        history = history[
            (history["y_pred"] >= config.min_prediction)
            & (history["floor_probability"] <= config.max_floor_probability)
        ]
        if history.empty:
            continue

        day = result.loc[day_mask].copy()
        correction = _estimate_correction(history, day, config.group, bins)
        correction = np.clip(
            correction,
            -config.clip_value,
            config.clip_value,
        ) * config.shrink

        eligible = (
            (day["y_pred"].to_numpy(dtype=float) >= config.min_prediction)
            & (
                day["floor_probability"].to_numpy(dtype=float)
                <= config.max_floor_probability
            )
        )
        day_indices = np.flatnonzero(day_mask)
        corrected[day_indices[eligible]] = np.clip(
            corrected[day_indices[eligible]] + correction[eligible],
            40.0,
            1000.0,
        )

    result["y_pred_base"] = base_pred
    result["y_pred"] = corrected
    result["online_residual_correction"] = result["y_pred"] - result["y_pred_base"]
    return result


def _estimate_correction(
    history: pd.DataFrame,
    current: pd.DataFrame,
    group: str,
    bins: np.ndarray,
) -> np.ndarray:
    residual = history["y_true"].to_numpy(dtype=float) - history["y_pred"].to_numpy(dtype=float)
    if group == "global":
        return np.full(current.shape[0], float(np.median(residual)))
    if group == "pred_bin":
        history_bins = np.digitize(history["y_pred"].to_numpy(dtype=float), bins)
        current_bins = np.digitize(current["y_pred"].to_numpy(dtype=float), bins)
        table = {
            int(bin_id): float(np.median(residual[history_bins == bin_id]))
            for bin_id in np.unique(history_bins)
        }
        return np.asarray([table.get(int(bin_id), 0.0) for bin_id in current_bins])
    if group == "state":
        history_keys = _state_keys(history)
        current_keys = _state_keys(current)
        table = {
            key: float(np.median(residual[history_keys == key]))
            for key in np.unique(history_keys)
        }
        return np.asarray([table.get(key, 0.0) for key in current_keys])
    raise ValueError(f"Unsupported online residual group: {group}")


def _state_keys(frame: pd.DataFrame) -> np.ndarray:
    return np.select(
        [
            frame["floor_probability"].to_numpy(dtype=float) >= 0.8,
            frame["cap_probability"].to_numpy(dtype=float) >= 0.08,
            frame["high_probability"].to_numpy(dtype=float) >= 0.08,
            frame["y_pred"].to_numpy(dtype=float) >= 300.0,
            frame["y_pred"].to_numpy(dtype=float) <= 80.0,
        ],
        ["floor", "cap", "high", "mid_high", "low"],
        default="normal",
    )


def save_online_residual_evaluation(
    predictions: pd.DataFrame,
    evaluation: Dict[str, Any],
    config: Dict[str, Any],
    prefix: str = "online_residual",
) -> None:
    save_evaluation(predictions, evaluation, config, prefix=prefix)


def write_online_residual_report(
    evaluation: Dict[str, Any],
    online_config: OnlineResidualConfig,
    config: Dict[str, Any],
    prefix: str = "online_residual",
) -> None:
    report_dir = Path(config["paths"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_type": "online_residual_calibrator",
        "online_residual_config": online_config.to_dict(),
        "summary": evaluation["summary"],
    }
    with (report_dir / f"{prefix}_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def evaluate_online_residual_predictions(
    predictions: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    online_config = OnlineResidualConfig.from_config(config)
    corrected = apply_online_residual_correction(predictions, online_config)
    evaluation = evaluate_predictions(corrected, config)
    return {
        "predictions": corrected,
        "evaluation": evaluation,
        "online_config": online_config,
    }

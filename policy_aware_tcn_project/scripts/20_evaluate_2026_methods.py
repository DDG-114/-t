#!/usr/bin/env python3
"""Evaluate trained GBM and TCN routes on labeled 2026 data."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import load_config
from epf_tcn.data import load_raw_data
from epf_tcn.deep_data import (
    add_external_anchor_columns,
    build_daily_windows,
    filter_windows_by_date_range,
    transform_windows,
)
from epf_tcn.deep_train import (
    _attach_external_anchor_prediction_columns,
    apply_floor_price_correction,
    apply_high_price_correction,
    apply_residual_blend_correction,
    load_deep_model,
    load_floor_price_correction,
    load_high_price_correction,
    load_residual_blend_correction,
    load_supply_demand_prior,
    predict_arrays,
    windows_to_prediction_frame,
)
from epf_tcn.evaluate import evaluate_predictions, save_evaluation
from epf_tcn.features import build_features
from epf_tcn.online_calibrator import evaluate_online_residual_predictions
from epf_tcn.residual_calibrator import load_residual_calibrator, predict_residual_calibrated
from epf_tcn.state_gbm import load_state_gbm
from epf_tcn.supply_demand_prior import add_supply_demand_prior_column
from epf_tcn.weather import (
    merge_weather_features,
    weather_date_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/residual_calibrator_weather_error_q2.yaml")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-27")
    parser.add_argument("--output-dir", default="outputs/eval_2026")
    parser.add_argument(
        "--deep-configs",
        nargs="+",
        default=[
            "config/deep_gbm_anchor_conservative.yaml",
            "config/deep_gbm_anchor_residual.yaml",
            "config/deep_weather_highcap.yaml",
        ],
    )
    return parser.parse_args()


def _with_paths(config: Dict[str, Any], output_dir: Path, name: str) -> Dict[str, Any]:
    patched = deepcopy(config)
    patched["paths"] = dict(patched.get("paths", {}))
    patched["paths"]["prediction_dir"] = str(output_dir / name / "predictions")
    patched["paths"]["report_dir"] = str(output_dir / name / "reports")
    return patched


def _with_split_and_zero_floor(
    config: Dict[str, Any],
    start: str,
    end: str,
    feature_path: Path | None = None,
) -> Dict[str, Any]:
    patched = deepcopy(config)
    patched["split"] = dict(patched.get("split", {}))
    patched["split"]["test_start"] = start
    patched["split"]["test_end"] = end
    patched["metrics"] = dict(patched.get("metrics", {}))
    patched["metrics"]["price_floor"] = 40.0
    patched["metrics"]["price_range"] = 1000.0
    patched["model"] = dict(patched.get("model", {}))
    patched["model"]["clip_prediction_min"] = 0.0
    patched["model"]["clip_prediction_max"] = 1000.0
    patched["state_gbm"] = dict(patched.get("state_gbm", {}))
    patched["state_gbm"]["floor_override_value"] = 0.0
    if feature_path is not None:
        patched["paths"] = dict(patched.get("paths", {}))
        patched["paths"]["processed_features"] = str(feature_path)
    return patched


def _build_2026_feature_table(config: Dict[str, Any], output_dir: Path) -> Path:
    """Build features for 2026 evaluation while keeping zero prices as labels."""
    feature_path = output_dir / "features_2026_zero_floor_eval.csv"
    if feature_path.exists():
        return feature_path

    feature_cfg = deepcopy(config)
    feature_cfg.setdefault("data", {}).setdefault("target_cleaning", {})
    feature_cfg["data"]["target_cleaning"]["cap_values_as_missing"] = []
    raw = load_raw_data(feature_cfg)
    features = build_features(raw, feature_cfg)
    existing_weather_path = Path(config["paths"].get("processed_features", ""))
    if existing_weather_path.exists():
        weather_features = pd.read_csv(
            existing_weather_path,
            usecols=lambda col: col == feature_cfg["columns"]["datetime"]
            or str(col).startswith("weather_"),
            parse_dates=[feature_cfg["columns"]["datetime"]],
            low_memory=False,
        )
        weather_cols = [
            col
            for col in weather_features.columns
            if col != feature_cfg["columns"]["datetime"]
        ]
        features = features.drop(columns=[col for col in weather_cols if col in features.columns])
        features[feature_cfg["columns"]["datetime"]] = pd.to_datetime(
            features[feature_cfg["columns"]["datetime"]]
        )
        features = features.merge(
            weather_features,
            on=feature_cfg["columns"]["datetime"],
            how="left",
        )
    else:
        try:
            from epf_tcn.weather import build_weather_table

            start_date, end_date = weather_date_bounds(
                features,
                datetime_col=feature_cfg["columns"]["datetime"],
            )
            forecast_weather = build_weather_table(
                feature_cfg,
                start_date=start_date,
                end_date=end_date,
            )
            features = merge_weather_features(
                features,
                forecast_weather,
                datetime_col=feature_cfg["columns"]["datetime"],
            )
        except Exception as exc:
            print(f"warning: weather feature merge skipped: {exc}", file=sys.stderr)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    return feature_path


def _save(prefix: str, frame: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = evaluate_predictions(frame, config)
    save_evaluation(frame, evaluation, config, prefix=prefix)
    return evaluation["summary"]


def _evaluate_gbm_anchor(
    config: Dict[str, Any],
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    state_model = load_state_gbm(config)
    calibrator = load_residual_calibrator(config)
    frame = predict_residual_calibrated(
        state_model,
        calibrator,
        features,
        config,
        split_name="test",
    )
    summary = _save("gbm_residual_anchor_2026", frame, config)
    return frame, summary


def _evaluate_online(
    config: Dict[str, Any],
    anchor_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    online_cfg = deepcopy(config)
    online_cfg["online_residual_calibrator"] = {
        "min_history_days": 10,
        "window_days": 5,
        "group": "pred_bin",
        "shrink": 0.5,
        "clip_value": 160.0,
        "min_prediction": 0.0,
        "max_floor_probability": 0.8,
        "prediction_floor": 0.0,
        "prediction_cap": 1000.0,
        "prediction_bins": [0, 20, 40, 60, 100, 160, 240, 360, 500, 1000],
    }
    result = evaluate_online_residual_predictions(anchor_frame, online_cfg)
    save_evaluation(result["predictions"], result["evaluation"], online_cfg, prefix="online_residual_2026")
    return result["predictions"], result["evaluation"]["summary"]


def _evaluate_deep(
    config_path: str,
    feature_path: Path,
    anchor_path: Path,
    output_dir: Path,
    start: str,
    end: str,
) -> Dict[str, Any]:
    deep_config = load_config(config_path)
    deep_config = _with_split_and_zero_floor(deep_config, start, end, feature_path=feature_path)
    name = Path(config_path).stem
    deep_config = _with_paths(deep_config, output_dir, name)
    if deep_config.get("deep_model", {}).get("external_anchor", {}).get("enabled", False):
        deep_config["deep_model"]["external_anchor"]["prediction_files"] = [str(anchor_path)]
        deep_config["deep_model"]["external_anchor"]["required"] = True

    features = pd.read_csv(feature_path, parse_dates=["date"])
    model, feature_spec, normalizer = load_deep_model(deep_config)
    prior = load_supply_demand_prior(deep_config)
    if prior is not None:
        features = add_supply_demand_prior_column(features, deep_config, prior)
    features = add_external_anchor_columns(features, deep_config)
    windows = build_daily_windows(features, deep_config, feature_spec=feature_spec, require_target=True)
    test_start = pd.Timestamp(start).normalize()
    test_end = pd.Timestamp(end).normalize()
    test_windows = filter_windows_by_date_range(windows, test_start, test_end)
    arrays = transform_windows(test_windows, normalizer, require_target=True)
    pred = predict_arrays(model, arrays, deep_config)
    frame = windows_to_prediction_frame(arrays, pred, deep_config)
    frame = _attach_external_anchor_prediction_columns(features, frame, deep_config)
    frame = apply_residual_blend_correction(
        frame,
        deep_config,
        correction=load_residual_blend_correction(deep_config),
    )
    frame = apply_floor_price_correction(
        features,
        frame,
        deep_config,
        correction=load_floor_price_correction(deep_config),
    )
    frame = apply_high_price_correction(
        features,
        frame,
        deep_config,
        correction=load_high_price_correction(deep_config),
    )
    summary = _save(f"{name}_2026", frame, deep_config)
    return summary


def _compact(summary: Dict[str, Any]) -> Dict[str, Any]:
    overall = summary.get("overall") or {}
    return {
        "accuracy": overall.get("accuracy"),
        "mean_daily_accuracy": summary.get("mean_daily_accuracy"),
        "mae": overall.get("mae"),
        "rmse": overall.get("rmse"),
        "pass_days": summary.get("pass_days"),
        "n_evaluated_days": summary.get("n_evaluated_days"),
        "n_prediction_rows": summary.get("n_prediction_rows"),
        "n_valid_prediction_rows": summary.get("n_valid_prediction_rows"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.base_config)
    feature_path = _build_2026_feature_table(base_config, output_dir)
    eval_config = _with_split_and_zero_floor(base_config, args.start, args.end, feature_path=feature_path)
    eval_config = _with_paths(eval_config, output_dir, "gbm_online")
    features = pd.read_csv(feature_path, parse_dates=["date"])

    anchor_frame, gbm_summary = _evaluate_gbm_anchor(eval_config, features)
    anchor_path = output_dir / "gbm_online" / "predictions" / "gbm_residual_anchor_2026_predictions.csv"
    _, online_summary = _evaluate_online(eval_config, anchor_frame)

    summaries = {
        "gbm_residual_anchor_2026": _compact(gbm_summary),
        "online_residual_2026": _compact(online_summary),
    }
    for deep_config_path in args.deep_configs:
        try:
            summary = _evaluate_deep(
                deep_config_path,
                feature_path,
                anchor_path,
                output_dir,
                args.start,
                args.end,
            )
            summaries[f"{Path(deep_config_path).stem}_2026"] = _compact(summary)
        except Exception as exc:
            summaries[f"{Path(deep_config_path).stem}_2026"] = {"error": str(exc)}

    summary_path = output_dir / "summary_2026_methods.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

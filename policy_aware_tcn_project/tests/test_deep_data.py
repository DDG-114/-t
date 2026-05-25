import numpy as np
import pandas as pd
import pytest

from epf_tcn.deep_data import (
    build_daily_windows,
    filter_windows_by_date_range,
    fit_window_normalizer,
    infer_deep_feature_spec,
    static_features_for_day,
    transform_windows,
)


def _config():
    return {
        "columns": {
            "target": "Price",
            "exogenous": ["load", "renewable", "space", "pumped", "tie"],
        },
        "data": {"expected_slots_per_day": 4},
        "metrics": {"price_floor": 40.0},
        "model": {"clip_prediction_min": 40.0, "clip_prediction_max": 1000.0},
        "deep_model": {
            "history_days": 2,
            "horizon": 4,
            "policy_regime_by_year": {2025: 0.0, 2026: 1.0},
            "year_bounds": {
                2025: {"min": 40.0, "max": 1000.0},
                2026: {"min": 0.0, "max": 1000.0},
            },
        },
    }


def _frame(start="2025-01-01", days=5):
    rows = []
    base = pd.Timestamp(start)
    for day in range(days):
        for slot in range(4):
            timestamp = base + pd.Timedelta(days=day, minutes=15 * slot)
            rows.append(
                {
                    "Date": timestamp,
                    "date": timestamp.normalize(),
                    "slot": slot,
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    "month": timestamp.month,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "slot_sin": np.sin(2 * np.pi * slot / 4),
                    "slot_cos": np.cos(2 * np.pi * slot / 4),
                    "month_sin": 0.5,
                    "month_cos": 0.5,
                    "Price": 100.0 + day * 10 + slot,
                    "load": 1000.0 + day,
                    "renewable": 100.0 + slot,
                    "space": 20.0,
                    "pumped": float(slot),
                    "tie": -5.0,
                    "净负荷": 900.0,
                    "供需裕度": 10.0,
                    "新能源占比": 0.1,
                    "竞价空间占比": 0.02,
                    "联络线占比": -0.005,
                    "price_lag_1d": 90.0,
                    "price_roll_3d_mean": 95.0,
                    "is_floor_price": float((100.0 + day * 10 + slot) == 40.0),
                    "floor_run_slots": 0.0,
                    "prev_floor_run_slots": float(slot),
                    "prev_long_floor_run": float(slot >= 2),
                    "floor_ratio_1d": 0.25,
                    "floor_ratio_3d": 0.50,
                    "floor_ratio_7d": 0.75,
                    "统一负荷预测_q50_7d": 1000.0 + day,
                    "统一新能源预测_q90_7d": 120.0 + slot,
                    "净负荷_q90_7d": 930.0 + day,
                    "净负荷_iqr_7d": 30.0,
                }
            )
    return pd.DataFrame(rows)


def test_static_features_use_year_specific_policy_bounds():
    cfg = _config()
    features = static_features_for_day(pd.Timestamp("2026-04-28"), cfg)
    assert features["policy_regime"] == 1.0
    assert features["price_lower_bound"] == 0.0
    assert features["price_upper_bound"] == 1000.0
    assert features["is_zero_floor_regime"] == 1.0


def test_daily_windows_use_only_previous_complete_days():
    cfg = _config()
    df = _frame()
    spec = infer_deep_feature_spec(df, cfg)
    windows = build_daily_windows(df, cfg, feature_spec=spec)

    assert [str(window.target_day.date()) for window in windows] == [
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
    ]
    first = windows[0]
    assert first.x_hist.shape[0] == 8
    assert first.x_fut.shape[0] == 4
    assert first.y.tolist() == pytest.approx([120.0, 121.0, 122.0, 123.0])
    assert first.anchor.tolist() == pytest.approx([110.0, 111.0, 112.0, 113.0])


def test_transform_windows_adds_missing_indicators_and_metadata():
    cfg = _config()
    df = _frame()
    spec = infer_deep_feature_spec(df, cfg)
    windows = build_daily_windows(df, cfg, feature_spec=spec)
    selected = filter_windows_by_date_range(windows, "2025-01-03", "2025-01-04")
    normalizer = fit_window_normalizer(selected, add_missing_indicators=True)
    arrays = transform_windows(selected, normalizer)

    assert arrays["x_hist"].shape[0] == 2
    assert arrays["x_hist"].shape[-1] == len(spec.hist_cols) * 2
    assert arrays["x_fut"].shape[-1] == len(spec.fut_cols) * 2
    assert arrays["x_static"].shape[-1] == len(spec.static_cols)
    assert arrays["y"].shape == (2, 4)
    assert arrays["lower_bound"].tolist() == pytest.approx([40.0, 40.0])
    assert arrays["floor_context"].shape == (2, 4)
    assert arrays["floor_context"].max() == pytest.approx(0.75)


def test_daily_windows_keep_partial_label_days_with_mask():
    cfg = _config()
    df = _frame()
    df.loc[(df["date"] == pd.Timestamp("2025-01-03")) & (df["slot"] == 2), "Price"] = np.nan
    spec = infer_deep_feature_spec(df, cfg)
    windows = build_daily_windows(df, cfg, feature_spec=spec)

    partial = [window for window in windows if window.target_day == pd.Timestamp("2025-01-03")]
    assert len(partial) == 1
    assert partial[0].y_mask.tolist() == pytest.approx([1.0, 1.0, 0.0, 1.0])


def test_future_feature_spec_excludes_target_day_floor_leakage_columns():
    cfg = _config()
    df = _frame()

    spec = infer_deep_feature_spec(df, cfg)

    assert "is_floor_price" in spec.hist_cols
    assert "floor_run_slots" in spec.hist_cols
    assert "prev_long_floor_run" in spec.hist_cols
    assert "floor_ratio_1d" in spec.fut_cols
    assert "floor_ratio_3d" in spec.fut_cols
    assert "floor_ratio_7d" in spec.fut_cols
    assert "统一负荷预测_q50_7d" in spec.fut_cols
    assert "统一新能源预测_q90_7d" in spec.fut_cols
    assert "净负荷_q90_7d" in spec.fut_cols
    assert "净负荷_iqr_7d" in spec.fut_cols
    assert "is_floor_price" not in spec.fut_cols
    assert "floor_run_slots" not in spec.fut_cols
    assert "prev_floor_run_slots" not in spec.fut_cols
    assert "prev_long_floor_run" not in spec.fut_cols

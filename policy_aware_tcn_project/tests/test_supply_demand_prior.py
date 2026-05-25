import numpy as np
import pandas as pd

from epf_tcn.deep_data import build_daily_windows, infer_deep_feature_spec
from epf_tcn.supply_demand_prior import (
    add_supply_demand_prior_column,
    fit_supply_demand_prior,
)


def _config():
    return {
        "columns": {
            "datetime": "Date",
            "target": "Price",
            "exogenous": [
                "发电总出力预测",
                "竞价空间",
                "统一负荷预测",
                "抽蓄",
                "统一新能源预测",
                "联络线计划",
            ],
        },
        "data": {"expected_slots_per_day": 4},
        "metrics": {"price_floor": 40.0},
        "split": {
            "train_start": "2025-01-01",
            "train_end": "2025-01-03",
            "valid_start": "2025-01-04",
            "valid_end": "2025-01-04",
            "test_start": "2025-01-05",
            "test_end": "2025-01-06",
        },
        "model": {"clip_prediction_min": 40.0, "clip_prediction_max": 1000.0},
        "deep_model": {
            "history_days": 2,
            "horizon": 4,
            "anchor_source": "supply_demand_prior",
            "supply_demand_prior": {
                "enabled": True,
                "output_col": "sd_prior_price",
                "ridge_alpha": 1.0,
                "hinge_cols": ["净负荷", "竞价空间占比"],
                "hinge_quantiles": [0.5],
            },
            "year_bounds": {
                2025: {"min": 40.0, "max": 1000.0},
            },
        },
    }


def _frame(days=6):
    rows = []
    base = pd.Timestamp("2025-01-01")
    for day in range(days):
        for slot in range(4):
            timestamp = base + pd.Timedelta(days=day, minutes=15 * slot)
            load = 1000.0 + 20.0 * day + 5.0 * slot
            renewable = 100.0 + 10.0 * slot
            generation = load + 80.0 - 3.0 * slot
            bid_space = 0.25 * load + 4.0 * day
            net_load = load - renewable
            price = 40.0 + 0.05 * net_load + 0.04 * bid_space
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
                    "Price": price,
                    "发电总出力预测": generation,
                    "竞价空间": bid_space,
                    "统一负荷预测": load,
                    "抽蓄": float(slot),
                    "统一新能源预测": renewable,
                    "联络线计划": -5.0,
                    "净负荷": net_load,
                    "供需裕度": generation - load,
                    "新能源占比": renewable / load,
                    "竞价空间占比": bid_space / load,
                    "联络线占比": -5.0 / load,
                }
            )
    return pd.DataFrame(rows)


def test_supply_demand_prior_fits_train_split_and_adds_clipped_column():
    cfg = _config()
    df = _frame()

    prior = fit_supply_demand_prior(df, cfg)
    enriched = add_supply_demand_prior_column(df, cfg, prior)

    assert prior.report["train_rows"] == 12
    assert "valid" in prior.report["metrics"]
    assert "sd_prior_price" in enriched.columns
    assert enriched["sd_prior_price"].between(40.0, 1000.0).all()


def test_supply_demand_prior_anchor_is_used_by_daily_windows():
    cfg = _config()
    df = _frame()
    prior = fit_supply_demand_prior(df, cfg)
    enriched = add_supply_demand_prior_column(df, cfg, prior)
    spec = infer_deep_feature_spec(enriched, cfg)

    windows = build_daily_windows(enriched, cfg, feature_spec=spec)
    first = windows[0]
    target_rows = enriched[enriched["date"] == first.target_day].sort_values("slot")

    assert first.anchor.tolist() == target_rows["sd_prior_price"].tolist()


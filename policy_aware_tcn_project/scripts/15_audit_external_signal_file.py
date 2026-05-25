#!/usr/bin/env python3
"""Audit whether an external signal file is sufficient for scarcity modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf_tcn.config import load_config
from epf_tcn.features import _prepare_external_signal_index


CORE_DIRECT_SIGNALS = {
    "reserve_margin_forecast",
    "available_capacity_forecast",
    "load_forecast_external",
    "renewable_forecast_external",
    "outage_capacity_declared",
    "market_scarcity_index",
}
CORE_ACTUAL_SIGNALS = {
    "actual_load",
    "actual_renewable",
    "actual_generation",
    "actual_tie_line",
    "realized_reserve_margin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/external_scarcity_template.yaml")
    parser.add_argument("--path")
    parser.add_argument("--start", default="2025-12-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--expected-slots-per-day", type=int, default=96)
    return parser.parse_args()


def _signal_path(config: dict, override: str | None) -> Path:
    if override:
        return Path(override)
    signal_cfg = config.get("features", {}).get("external_signals", {})
    return Path(signal_cfg.get("path", "data/external/scarcity_signals.csv"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    signal_cfg = config.get("features", {}).get("external_signals", {})
    path = _signal_path(config, args.path)
    print(f"external_signal_path: {path}")
    if not path.exists():
        print("status: missing")
        print("required_action: provide a CSV with Date or date+slot and scarcity columns")
        return

    raw = pd.read_csv(path, encoding=signal_cfg.get("encoding", "utf-8-sig"))
    indexed = _prepare_external_signal_index(
        raw,
        signal_cfg,
        slot_minutes=int(config.get("data", {}).get("slot_minutes", 15)),
    )
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    window = indexed[(indexed["date"] >= start) & (indexed["date"] <= end)].copy()

    direct = [col for col in signal_cfg.get("direct_columns", []) if col in indexed.columns]
    actual = [
        col
        for col in signal_cfg.get("lagged_actual_columns", [])
        if col in indexed.columns
    ]
    core_direct = sorted(CORE_DIRECT_SIGNALS.intersection(indexed.columns))
    core_actual = sorted(CORE_ACTUAL_SIGNALS.intersection(indexed.columns))

    print(f"status: present")
    print(f"rows: {indexed.shape[0]}")
    print(f"date_min: {indexed['date'].min().date() if not indexed.empty else 'NA'}")
    print(f"date_max: {indexed['date'].max().date() if not indexed.empty else 'NA'}")
    print(f"direct_columns_found: {direct}")
    print(f"lagged_actual_columns_found: {actual}")
    print(f"core_direct_signals_found: {core_direct}")
    print(f"core_actual_signals_found: {core_actual}")
    print(f"audit_window: {start.date()} to {end.date()}")
    print(f"audit_window_rows: {window.shape[0]}")

    if not window.empty:
        per_day = window.groupby("date")["slot"].nunique()
        complete_days = int((per_day >= int(args.expected_slots_per_day)).sum())
        print(f"audit_window_days: {int(per_day.shape[0])}")
        print(f"complete_96_slot_days: {complete_days}")
        print(f"min_slots_per_day: {int(per_day.min())}")
        print(f"max_slots_per_day: {int(per_day.max())}")
        missing_rates = {}
        for col in [*direct, *actual]:
            values = pd.to_numeric(window[col], errors="coerce")
            missing_rates[col] = round(float(values.isna().mean()), 4)
        print(f"missing_rates_in_window: {missing_rates}")

    has_core_signal = bool(core_direct or core_actual)
    has_window_coverage = (
        not window.empty
        and window["date"].nunique() >= (end - start).days + 1
        and window.groupby("date")["slot"].nunique().min() >= int(args.expected_slots_per_day)
    )
    if has_core_signal and has_window_coverage:
        print("scarcity_modeling_readiness: ready_for_feature_merge")
    elif has_core_signal:
        print("scarcity_modeling_readiness: has_signals_but_coverage_incomplete")
    else:
        print("scarcity_modeling_readiness: insufficient_core_signals")


if __name__ == "__main__":
    main()

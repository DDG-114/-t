#!/usr/bin/env python
"""Check whether a given year contains zero electricity prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf.config import ensure_dirs, load_config
from epf.data import load_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    dt_col = config["columns"]["datetime"]
    target_col = config["columns"]["target"]

    df = load_raw_data(config)
    year_df = df[df[dt_col].dt.year == args.year].copy()
    zero_df = year_df[year_df[target_col] == 0].copy()

    report = {
        "year": args.year,
        "total_rows": int(year_df.shape[0]),
        "zero_price_rows": int(zero_df.shape[0]),
        "has_zero_price": bool(zero_df.shape[0] > 0),
        "zero_price_ratio": (
            float(zero_df.shape[0] / year_df.shape[0]) if year_df.shape[0] else None
        ),
        "first_zero_time": (
            str(zero_df[dt_col].min()) if not zero_df.empty else None
        ),
        "last_zero_time": (
            str(zero_df[dt_col].max()) if not zero_df.empty else None
        ),
        "zero_count_by_month": {
            str(month): int(count)
            for month, count in zero_df.groupby(zero_df[dt_col].dt.month).size().items()
        },
    }

    report_dir = Path(config["paths"]["report_dir"])
    zero_rows_path = report_dir / f"zero_price_rows_{args.year}.csv"
    summary_path = report_dir / f"zero_price_summary_{args.year}.json"

    zero_df.to_csv(zero_rows_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved zero rows to {zero_rows_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

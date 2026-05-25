#!/usr/bin/env python
"""Report target-price cleaning effects."""

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    dt_col = config["columns"]["datetime"]
    target = config["columns"]["target"]
    reason_col = f"{target}_clean_reason"
    raw_col = f"{target}_raw"

    df = load_raw_data(config)
    cleaned = df[df[reason_col].ne("")].copy()

    by_reason = cleaned[reason_col].value_counts().to_dict()
    by_year = cleaned.groupby(cleaned[dt_col].dt.year).size().to_dict()
    by_year_reason = (
        cleaned.groupby([cleaned[dt_col].dt.year, reason_col]).size().to_dict()
    )
    report = {
        "total_rows": int(df.shape[0]),
        "cleaned_rows": int(cleaned.shape[0]),
        "cleaned_ratio": float(cleaned.shape[0] / df.shape[0]) if df.shape[0] else None,
        "by_reason": {str(k): int(v) for k, v in by_reason.items()},
        "by_year": {str(k): int(v) for k, v in by_year.items()},
        "by_year_reason": {
            f"{year}|{reason}": int(count)
            for (year, reason), count in by_year_reason.items()
        },
        "raw_value_counts_cleaned": {
            str(k): int(v)
            for k, v in cleaned[raw_col].value_counts(dropna=False).head(20).items()
        },
    }

    report_dir = Path(config["paths"]["report_dir"])
    cleaned_rows_path = report_dir / "price_cleaning_rows.csv"
    summary_path = report_dir / "price_cleaning_summary.json"

    cleaned.to_csv(cleaned_rows_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved cleaned rows to {cleaned_rows_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

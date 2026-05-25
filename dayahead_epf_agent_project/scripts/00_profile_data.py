#!/usr/bin/env python
"""Profile raw electricity price data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from epf.config import ensure_dirs, load_config
from epf.data import load_raw_data, profile_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    df = load_raw_data(config)
    profile = profile_data(df, config)

    output_path = Path(config["paths"]["report_dir"]) / "data_profile.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(profile, ensure_ascii=False, indent=2, default=str))
    print(f"Saved profile to {output_path}")


if __name__ == "__main__":
    main()

"""Date-based train/validation/test split utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def split_by_date(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by target date using config ranges."""
    split = config["split"]
    date = pd.to_datetime(df["date"])

    train_mask = (date >= pd.Timestamp(split["train_start"])) & (date <= pd.Timestamp(split["train_end"]))
    valid_mask = (date >= pd.Timestamp(split["valid_start"])) & (date <= pd.Timestamp(split["valid_end"]))
    test_mask = (date >= pd.Timestamp(split["test_start"])) & (date <= pd.Timestamp(split["test_end"]))

    return df.loc[train_mask].copy(), df.loc[valid_mask].copy(), df.loc[test_mask].copy()

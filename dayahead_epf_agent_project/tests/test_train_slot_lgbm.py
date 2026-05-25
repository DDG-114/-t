import numpy as np
import pandas as pd
import pytest

from epf.train_slot_lgbm import make_sample_weight


def test_make_sample_weight_inverse_target_floor():
    config = {
        "metrics": {"price_floor": 40.0},
        "model": {
            "sample_weight": {
                "enabled": True,
                "mode": "inverse_target_floor",
            }
        },
    }
    weights = make_sample_weight(pd.Series([40.0, 80.0, 400.0]), config)

    assert weights == pytest.approx(np.array([1 / 40.0, 1 / 80.0, 1 / 400.0]))


def test_make_sample_weight_disabled_returns_none():
    config = {"model": {"sample_weight": {"enabled": False}}}

    assert make_sample_weight(pd.Series([40.0]), config) is None

import numpy as np
import pandas as pd

from epf_tcn.residual_calibrator import (
    ResidualCalibrationParams,
    _apply_residual_correction,
    _calibrator_config,
    _select_params,
)


def test_residual_correction_respects_floor_probability_mask():
    frame = pd.DataFrame(
        {
            "state_pred": [40.0, 200.0, 300.0],
            "floor_probability": [0.9, 0.1, 0.6],
        }
    )
    params = ResidualCalibrationParams(
        shrink=0.5,
        clip_value=100.0,
        min_prediction=100.0,
        max_floor_probability=0.2,
        positive_only=False,
    )

    pred = _apply_residual_correction(
        frame,
        np.array([100.0, 100.0, 100.0]),
        params,
        floor_price=40.0,
    )

    assert pred.tolist() == [40.0, 250.0, 300.0]


def test_residual_correction_supports_zero_floor_policy():
    frame = pd.DataFrame(
        {
            "state_pred": [10.0],
            "floor_probability": [0.0],
        }
    )
    params = ResidualCalibrationParams(
        shrink=1.0,
        clip_value=100.0,
        min_prediction=0.0,
        max_floor_probability=0.8,
        positive_only=False,
    )

    pred = _apply_residual_correction(
        frame,
        np.array([-50.0]),
        params,
        floor_price=40.0,
        prediction_min=0.0,
    )

    assert pred.tolist() == [0.0]


def test_select_params_can_disable_unhelpful_correction():
    frame = pd.DataFrame(
        {
            "Price": [100.0, 200.0, 300.0],
            "state_pred": [100.0, 200.0, 300.0],
            "floor_probability": [0.0, 0.0, 0.0],
        }
    )
    config = {
        "columns": {"target": "Price"},
        "metrics": {"price_floor": 40.0},
        "residual_calibrator": {
            "selection_tolerance": 0.0,
            "shrink_grid": [1.0],
            "clip_grid": [100.0],
            "min_prediction_grid": [40.0],
            "max_floor_probability_grid": [1.1],
            "positive_only_grid": [False],
        },
    }

    params, report = _select_params(frame, np.array([50.0, 50.0, 50.0]), config)

    assert params.shrink == 0.0
    assert report["selected"]["accuracy"] == 1.0


def test_calibrator_does_not_refit_after_validation_by_default():
    assert _calibrator_config({})["refit_after_validation"] is False

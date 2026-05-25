import importlib

import pytest


def _has_real_torch():
    try:
        import torch
    except ImportError:
        return False
    return hasattr(torch, "nn") and hasattr(torch, "Tensor")


@pytest.mark.skipif(not _has_real_torch(), reason="real PyTorch is not installed")
def test_policy_aware_tcn_forward_shapes():
    deep_models = importlib.import_module("epf_tcn.deep_models")

    cfg = deep_models.PolicyAwareTCNConfig(
        hist_dim=6,
        fut_dim=5,
        static_dim=4,
        horizon=4,
        channels=16,
        dilations=(1, 2),
        output_sigma=True,
    )
    model = deep_models.PolicyAwareTCN(cfg)
    torch = deep_models.torch

    output = model(
        torch.randn(3, 8, 6),
        torch.randn(3, 4, 5),
        torch.randn(3, 4),
        torch.randn(3, 4),
    )

    assert output["mu"].shape == (3, 4)
    assert output["log_sigma"].shape == (3, 4)
    assert output["gate"].shape == (3, 4, 16)


@pytest.mark.skipif(not _has_real_torch(), reason="real PyTorch is not installed")
def test_policy_aware_tcn_slot_last_day_context_shapes():
    deep_models = importlib.import_module("epf_tcn.deep_models")

    cfg = deep_models.PolicyAwareTCNConfig(
        hist_dim=6,
        fut_dim=5,
        static_dim=4,
        horizon=4,
        channels=16,
        dilations=(1, 2),
        history_context_mode="slot_last_day",
        output_sigma=True,
    )
    model = deep_models.PolicyAwareTCN(cfg)
    torch = deep_models.torch

    output = model(
        torch.randn(3, 8, 6),
        torch.randn(3, 4, 5),
        torch.randn(3, 4),
        torch.randn(3, 4),
    )

    assert output["mu"].shape == (3, 4)
    assert output["gate"].shape == (3, 4, 16)


@pytest.mark.skipif(not _has_real_torch(), reason="real PyTorch is not installed")
def test_policy_aware_tcn_residual_direct_gate_shapes():
    deep_models = importlib.import_module("epf_tcn.deep_models")

    cfg = deep_models.PolicyAwareTCNConfig(
        hist_dim=6,
        fut_dim=5,
        static_dim=4,
        horizon=4,
        channels=16,
        dilations=(1, 2),
        output_mode="residual_direct_gate",
        direct_output_scale=1000.0,
        output_sigma=True,
    )
    model = deep_models.PolicyAwareTCN(cfg)
    torch = deep_models.torch

    output = model(
        torch.randn(3, 8, 6),
        torch.randn(3, 4, 5),
        torch.randn(3, 4),
        torch.randn(3, 4),
    )

    assert output["mu"].shape == (3, 4)
    assert output["direct_mu"].shape == (3, 4)
    assert output["residual_mu"].shape == (3, 4)
    assert output["output_gate"].shape == (3, 4)

"""Deep neural architectures for direct 96-step day-ahead EPF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def require_torch():
    """Import a real PyTorch installation with a clear error message."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for deep EPF models. Install it with the "
            "CPU/GPU build appropriate for this machine, then rerun the deep "
            "training script."
        ) from exc
    if not hasattr(torch, "nn") or not hasattr(torch, "Tensor"):
        raise ImportError(
            "The import name 'torch' exists, but it is not a usable PyTorch "
            "installation. Reinstall PyTorch before running deep EPF training."
        )
    return torch


torch = require_torch()
nn = torch.nn


@dataclass(frozen=True)
class PolicyAwareTCNConfig:
    """Architecture hyperparameters for the report2 TCN route."""

    hist_dim: int
    fut_dim: int
    static_dim: int
    horizon: int = 96
    channels: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.15
    future_layers: int = 2
    static_hidden: int = 32
    residual_scale: float = 200.0
    history_context_mode: str = "last"
    output_mode: str = "residual"
    direct_output_scale: float = 1000.0
    output_sigma: bool = True

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        hist_dim: int,
        fut_dim: int,
        static_dim: int,
    ) -> "PolicyAwareTCNConfig":
        """Create model config from project YAML plus inferred dimensions."""
        model_cfg = config.get("deep_model", {}).get("tcn", {})
        return cls(
            hist_dim=hist_dim,
            fut_dim=fut_dim,
            static_dim=static_dim,
            horizon=int(config.get("deep_model", {}).get("horizon", 96)),
            channels=int(model_cfg.get("channels", 64)),
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            dilations=tuple(int(v) for v in model_cfg.get("dilations", [1, 2, 4, 8, 16, 32])),
            dropout=float(model_cfg.get("dropout", 0.15)),
            future_layers=int(model_cfg.get("future_layers", 2)),
            static_hidden=int(model_cfg.get("static_hidden", 32)),
            residual_scale=float(model_cfg.get("residual_scale", 200.0)),
            history_context_mode=str(model_cfg.get("history_context_mode", "last")),
            output_mode=str(model_cfg.get("output_mode", "residual")),
            direct_output_scale=float(model_cfg.get("direct_output_scale", 1000.0)),
            output_sigma=bool(model_cfg.get("output_sigma", True)),
        )


class Chomp1d(nn.Module):
    """Remove right padding so convolutions remain causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalResidualBlock(nn.Module):
    """Causal dilated residual block used by the history encoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class FutureEncoder(nn.Module):
    """Light temporal encoder for known delivery-day covariates."""

    def __init__(self, fut_dim: int, channels: int, layers: int, dropout: float) -> None:
        super().__init__()
        modules: List[nn.Module] = [
            nn.Linear(fut_dim, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(layers - 1, 0)):
            modules.extend(
                [
                    nn.Linear(channels, channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*modules)

    def forward(self, x_fut: torch.Tensor) -> torch.Tensor:
        return self.net(x_fut)


class PolicyAwareTCN(nn.Module):
    """Compact production TCN for direct daily 96-point price prediction.

    The model follows report2's recommended route:
    - causal dilated TCN over recent history;
    - known-future market/covariate branch;
    - static policy-regime conditioning;
    - direct 96-step output;
    - residual correction over yesterday's same-slot price anchor;
    - optional heteroscedastic scale head for uncertainty-aware training.
    """

    def __init__(self, cfg: PolicyAwareTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.horizon
        self.residual_scale = cfg.residual_scale
        self.history_context_mode = cfg.history_context_mode
        self.output_mode = cfg.output_mode
        self.direct_output_scale = cfg.direct_output_scale
        self.output_sigma = cfg.output_sigma

        self.hist_projection = nn.Conv1d(cfg.hist_dim, cfg.channels, kernel_size=1)
        self.history_encoder = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=cfg.channels,
                    kernel_size=cfg.kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                )
                for dilation in cfg.dilations
            ]
        )
        self.context_norm = nn.LayerNorm(cfg.channels)

        self.future_encoder = FutureEncoder(
            fut_dim=cfg.fut_dim,
            channels=cfg.channels,
            layers=cfg.future_layers,
            dropout=cfg.dropout,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(cfg.static_dim, cfg.static_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.static_hidden, cfg.channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(cfg.channels * 3, cfg.channels),
            nn.GELU(),
            nn.Linear(cfg.channels, cfg.channels),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.channels * 3, cfg.channels),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.channels, cfg.channels),
            nn.GELU(),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(cfg.channels, cfg.channels // 2),
            nn.GELU(),
            nn.Linear(cfg.channels // 2, 1),
        )
        if cfg.output_mode == "residual_direct_gate":
            self.direct_head = nn.Sequential(
                nn.Linear(cfg.channels, cfg.channels // 2),
                nn.GELU(),
                nn.Linear(cfg.channels // 2, 1),
            )
            self.output_gate_head = nn.Sequential(
                nn.Linear(cfg.channels, cfg.channels // 2),
                nn.GELU(),
                nn.Linear(cfg.channels // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.direct_head = None
            self.output_gate_head = None
        if cfg.output_sigma:
            self.log_sigma_head = nn.Sequential(
                nn.Linear(cfg.channels, cfg.channels // 2),
                nn.GELU(),
                nn.Linear(cfg.channels // 2, 1),
            )
        else:
            self.log_sigma_head = None

    def forward(
        self,
        x_hist: torch.Tensor,
        x_fut: torch.Tensor,
        x_static: torch.Tensor,
        anchor: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Return ``mu`` and, when configured, ``log_sigma`` tensors.

        Shapes:
        - ``x_hist``: ``[batch, history_days * 96, hist_dim]``
        - ``x_fut``: ``[batch, 96, fut_dim]``
        - ``x_static``: ``[batch, static_dim]``
        - ``anchor``: yesterday same-slot price, ``[batch, 96]``
        """
        hist = x_hist.transpose(1, 2)
        hist = self.hist_projection(hist)
        hist = self.history_encoder(hist)
        if self.history_context_mode == "slot_last_day":
            if hist.shape[-1] < self.horizon:
                raise ValueError(
                    "slot_last_day history_context_mode requires at least horizon history steps."
                )
            context = hist[:, :, -self.horizon :].transpose(1, 2)
            context = self.context_norm(context)
        elif self.history_context_mode == "last":
            context = hist[:, :, -1]
            context = self.context_norm(context)
            context = context.unsqueeze(1).expand(-1, x_fut.shape[1], -1)
        else:
            raise ValueError(f"Unsupported history_context_mode: {self.history_context_mode}")

        future = self.future_encoder(x_fut)
        static = self.static_encoder(x_static).unsqueeze(1).expand_as(future)
        fused_input = torch.cat([context, future, static], dim=-1)
        gate = self.gate(fused_input)
        fused = self.fusion(fused_input)
        fused = gate * fused + (1.0 - gate) * future

        residual = torch.tanh(self.mu_head(fused).squeeze(-1)) * self.residual_scale
        if anchor is None:
            residual_mu = residual
        else:
            residual_mu = anchor + residual

        output: Dict[str, torch.Tensor] = {
            "residual": residual,
            "gate": gate,
            "residual_mu": residual_mu,
        }
        if self.output_mode == "residual_direct_gate":
            if self.direct_head is None or self.output_gate_head is None:
                raise RuntimeError("residual_direct_gate mode requires direct and output gate heads.")
            direct_mu = torch.sigmoid(self.direct_head(fused).squeeze(-1)) * self.direct_output_scale
            output_gate = self.output_gate_head(fused).squeeze(-1)
            mu = output_gate * direct_mu + (1.0 - output_gate) * residual_mu
            output["direct_mu"] = direct_mu
            output["output_gate"] = output_gate
        elif self.output_mode == "residual":
            mu = residual_mu
        else:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        output["mu"] = mu
        if self.log_sigma_head is not None:
            log_sigma = self.log_sigma_head(fused).squeeze(-1)
            output["log_sigma"] = log_sigma.clamp(min=-5.0, max=5.0)
        return output


def build_policy_aware_tcn(
    config: Dict[str, Any],
    hist_dim: int,
    fut_dim: int,
    static_dim: int,
) -> PolicyAwareTCN:
    """Construct the default report2-aligned deep model."""
    model_cfg = PolicyAwareTCNConfig.from_config(
        config,
        hist_dim=hist_dim,
        fut_dim=fut_dim,
        static_dim=static_dim,
    )
    return PolicyAwareTCN(model_cfg)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def module_summary(model: nn.Module) -> Dict[str, Any]:
    """Return a compact architecture summary for train reports."""
    cfg = getattr(model, "cfg", None)
    payload: Dict[str, Any] = {
        "class_name": model.__class__.__name__,
        "trainable_parameters": count_parameters(model),
    }
    if cfg is not None:
        payload["config"] = {
            "hist_dim": cfg.hist_dim,
            "fut_dim": cfg.fut_dim,
            "static_dim": cfg.static_dim,
            "horizon": cfg.horizon,
            "channels": cfg.channels,
            "kernel_size": cfg.kernel_size,
            "dilations": list(cfg.dilations),
            "dropout": cfg.dropout,
            "future_layers": cfg.future_layers,
            "static_hidden": cfg.static_hidden,
            "residual_scale": cfg.residual_scale,
            "history_context_mode": cfg.history_context_mode,
            "output_mode": cfg.output_mode,
            "direct_output_scale": cfg.direct_output_scale,
            "output_sigma": cfg.output_sigma,
        }
    return payload

from __future__ import annotations

import torch
from torch import nn
from .utility import bounded_attention_bias, risk_adjusted_cev


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CEVHead(nn.Module):
    """Predicts a distribution over signed counterfactual evidence value for M modalities."""
    def __init__(self, query_dim: int, modality_dim: int, hidden: int, n_modalities: int = 2,
                 predict_uncertainty: bool = True):
        super().__init__()
        self.n_modalities = n_modalities
        self.predict_uncertainty = predict_uncertainty
        out = n_modalities * (2 if predict_uncertainty else 1)
        self.net = nn.Sequential(
            nn.Linear(query_dim + n_modalities * modality_dim, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, query: torch.Tensor, modality_features: list[torch.Tensor]):
        if len(modality_features) != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} modality features")
        x = torch.cat([query, *modality_features], dim=-1)
        out = self.net(x)
        if not self.predict_uncertainty:
            return out, None
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 5.0)


class CUREAttentionController(nn.Module):
    """Detector-agnostic control head for modality-specific attention logits.

    Input logits have shape [..., M, K] where M is the modality dimension. The controller
    adds a bounded, per-query modality bias; it never removes the host residual path.
    """
    def __init__(self, scale: float = 0.5, temperature: float = 0.75, kappa: float = 0.0):
        super().__init__()
        self.scale = scale
        self.temperature = temperature
        self.kappa = kappa

    def forward(self, attention_logits: torch.Tensor, cev_mean: torch.Tensor,
                cev_logvar: torch.Tensor | None = None) -> torch.Tensor:
        value = risk_adjusted_cev(cev_mean, cev_logvar, self.kappa)
        bias = bounded_attention_bias(value, self.scale, self.temperature)
        while bias.ndim < attention_logits.ndim:
            bias = bias.unsqueeze(-1)
        return attention_logits + bias


class CUREResidualFusion(nn.Module):
    """Small standalone proxy used for unit and CPU mechanism checks."""
    def __init__(self, camera_dim: int, radar_dim: int, hidden: int, num_classes: int,
                 temperature: float = 0.75, residual_scale: float = 0.35,
                 predict_uncertainty: bool = True, kappa: float = 0.0):
        super().__init__()
        self.camera = MLPEncoder(camera_dim, hidden)
        self.radar = MLPEncoder(radar_dim, hidden)
        self.base = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU())
        self.cev = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 4 if predict_uncertainty else 2))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_classes))
        self.temperature = temperature
        self.residual_scale = residual_scale
        self.predict_uncertainty = predict_uncertainty
        self.kappa = kappa

    def forward(self, camera, radar, return_aux=False):
        hc, hr = self.camera(camera), self.radar(radar)
        joined = torch.cat([hc, hr], -1)
        base = self.base(joined)
        raw = self.cev(joined)
        if self.predict_uncertainty:
            mean, logvar = raw.chunk(2, dim=-1)
            logvar = logvar.clamp(-8.0, 5.0)
        else:
            mean, logvar = raw, None
        value = risk_adjusted_cev(mean, logvar, self.kappa)
        b = torch.tanh(value / self.temperature)
        fused = base + self.residual_scale * (b[:, :1] * hc + b[:, 1:] * hr)
        logits = self.head(fused)
        if return_aux:
            return logits, {"values": mean, "logvar": logvar, "biases": b}
        return logits


class TaskOnlyResidualFusion(CUREResidualFusion):
    """Architecture-identical control trained without CEV distillation."""
    pass


class ConfidenceGateFusion(nn.Module):
    def __init__(self, camera_dim, radar_dim, hidden, num_classes):
        super().__init__()
        self.camera = MLPEncoder(camera_dim, hidden)
        self.radar = MLPEncoder(radar_dim, hidden)
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, camera, radar, return_aux=False):
        hc, hr = self.camera(camera), self.radar(radar)
        w = torch.softmax(self.gate(torch.cat([hc, hr], dim=-1)), dim=-1)
        logits = self.head(w[:, :1] * hc + w[:, 1:] * hr)
        return (logits, {"weights": w}) if return_aux else logits


class ConcatFusion(nn.Module):
    def __init__(self, camera_dim, radar_dim, hidden, num_classes):
        super().__init__()
        self.camera = MLPEncoder(camera_dim, hidden)
        self.radar = MLPEncoder(radar_dim, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, num_classes))

    def forward(self, camera, radar, return_aux=False):
        hc, hr = self.camera(camera), self.radar(radar)
        logits = self.head(torch.cat([hc, hr], dim=-1))
        return (logits, {}) if return_aux else logits


# Compatibility alias for the earlier toy script.
CUREFusion = CUREResidualFusion

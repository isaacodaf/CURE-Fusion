from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


def per_sample_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, reduction="none")


def counterfactual_evidence_value(
    factual_loss: torch.Tensor,
    intervened_loss: torch.Tensor,
) -> torch.Tensor:
    """Signed task-loss change caused by a sensor intervention.

    Positive means factual sensor evidence helped: degrading it increased loss.
    Negative means the factual evidence was locally harmful: intervention reduced loss.
    """
    if factual_loss.shape != intervened_loss.shape:
        raise ValueError("factual_loss and intervened_loss must have the same shape")
    return intervened_loss - factual_loss


def cev_distillation_loss(
    predicted_mean: torch.Tensor,
    target: torch.Tensor,
    predicted_logvar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber CEV regression or heteroscedastic Gaussian NLL when log-variance is predicted."""
    if predicted_logvar is None:
        return F.smooth_l1_loss(predicted_mean, target)
    logvar = predicted_logvar.clamp(-8.0, 5.0)
    inv_var = torch.exp(-logvar)
    return (0.5 * (inv_var * (predicted_mean - target).pow(2) + logvar)).mean()


def pairwise_ranking_loss(predicted: torch.Tensor, target: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """Preserve which modality has larger evidence value when teacher values differ materially."""
    if predicted.ndim != 2 or target.shape != predicted.shape:
        raise ValueError("predicted and target must have shape [batch, modalities]")
    losses = []
    m = predicted.shape[1]
    for i in range(m):
        for j in range(i + 1, m):
            td = target[:, i] - target[:, j]
            keep = td.abs() > margin
            if keep.any():
                sign = td[keep].sign()
                pd = predicted[keep, i] - predicted[keep, j]
                losses.append(F.softplus(-sign * pd).mean())
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean()


def risk_adjusted_cev(mean: torch.Tensor, logvar: torch.Tensor | None, kappa: float = 0.0) -> torch.Tensor:
    """Lower-confidence value used for attention control. kappa=0 recovers signed mean CEV."""
    if logvar is None or kappa == 0.0:
        return mean
    std = torch.exp(0.5 * logvar.clamp(-8.0, 5.0))
    return mean - float(kappa) * std


def bounded_attention_bias(value: torch.Tensor, scale: float = 0.5, temperature: float = 0.75) -> torch.Tensor:
    return float(scale) * torch.tanh(value / float(temperature))


def interaction_cev(
    delta_joint: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
) -> torch.Tensor:
    """Second-order intervention interaction for analysis: synergy (>0) or redundancy (<0)."""
    return delta_joint - delta_a - delta_b


# Backward-compatible aliases used by the mechanism-check scripts.
def counterfactual_utility(factual_loss, intervened_loss, clamp=False):
    delta = counterfactual_evidence_value(factual_loss, intervened_loss)
    return delta.clamp_min(0.0) if clamp else delta


def utility_distillation_loss(predicted, target, log_space=False):
    if log_space:
        predicted = torch.sign(predicted) * torch.log1p(predicted.abs())
        target = torch.sign(target) * torch.log1p(target.abs())
    return F.smooth_l1_loss(predicted, target)


def utility_weights(utilities: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if utilities.ndim != 2:
        raise ValueError("utilities must have shape [batch, modalities]")
    return torch.softmax(utilities / float(temperature), dim=-1)

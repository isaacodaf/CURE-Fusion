from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairedSummary:
    n_scenes: int
    mean_difference: float
    sample_standard_deviation: float
    ci95_low: float
    ci95_high: float
    paired_standardized_mean_difference: float


def summarize_paired_scenes(
    candidate: Iterable[float],
    control: Iterable[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Summarize paired candidate-control differences at the scene level."""
    cand = np.asarray(list(candidate), dtype=float)
    ctrl = np.asarray(list(control), dtype=float)
    if cand.shape != ctrl.shape or cand.ndim != 1:
        raise ValueError("candidate and control must be one-dimensional paired arrays")
    if len(cand) < 2:
        raise ValueError("at least two independent scenes are required")
    if not np.isfinite(cand).all() or not np.isfinite(ctrl).all():
        raise ValueError("paired scene values must be finite")
    delta = cand - ctrl
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot_means = delta[sample_indices].mean(axis=1)
    sd = float(delta.std(ddof=1))
    effect = float(delta.mean() / sd) if sd > 0 else float("inf") if delta.mean() > 0 else 0.0
    low, high = np.quantile(boot_means, [0.025, 0.975])
    summary = PairedSummary(
        n_scenes=len(delta),
        mean_difference=float(delta.mean()),
        sample_standard_deviation=sd,
        ci95_low=float(low),
        ci95_high=float(high),
        paired_standardized_mean_difference=effect,
    )
    return asdict(summary)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a non-empty one-dimensional sequence in [0,1]")
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, float((m - rank) * p[index]))
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()

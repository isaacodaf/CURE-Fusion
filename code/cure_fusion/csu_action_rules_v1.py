"""Matched whole-output selectors for a fixed frame-action utility predictor.

These selectors choose between two complete detections. They never align or
combine unrelated query indices. All normalization is fitted on controller-fit
frames, and all sensor conditions of a source frame travel in the same batch.
"""
import hashlib
import numpy as np
import torch
from torch import nn


CONDITION_WEIGHTS = (1., 1./12, 1./12, 1./12)


def condition_average(value):
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError('Expected source-frame by four-condition values')
    w = value.new_tensor(CONDITION_WEIGHTS)
    return (value*w).sum()/(len(value)*w.sum())


def fitting_normalization(features, gain):
    if features.ndim != 3 or features.shape[1:] != (4,40) or gain.shape != features.shape[:2]:
        raise ValueError('Expected [frames,4,40] inference features and [frames,4] gains')
    if len(features) == 0 or not torch.isfinite(features).all() or not torch.isfinite(gain).all():
        raise ValueError('Nonempty finite fitting operands required')
    w = features.new_tensor(CONDITION_WEIGHTS)
    center = (features*w[None,:,None]).sum((0,1))/(len(features)*w.sum())
    variance = ((features-center).square()*w[None,:,None]).sum((0,1))/(len(features)*w.sum())
    return center.detach(), variance.sqrt().clamp_min(1e-4).detach(), condition_average(gain.square()).sqrt().clamp_min(1e-6).detach()


def permutation_indices(size, seed, condition, role):
    salt = f'csu-action-rules-v1:{seed}:{condition}:{role}'
    return np.random.default_rng(int.from_bytes(hashlib.sha256(salt.encode()).digest()[:16], 'little')).permutation(size)


def rule_inputs(features, mean, center, scale, gain_scale, mode, seed, role):
    if features.shape[:2] != mean.shape or features.shape[1:] != (4,40):
        raise ValueError('Aligned four-condition inference operands required')
    z = (features-center)/scale
    extra = mean.detach()/gain_scale
    if mode == 'zero':
        extra = torch.zeros_like(extra)
    elif mode == 'permuted':
        extra = torch.stack([extra[torch.as_tensor(permutation_indices(len(extra),seed,c,role),device=extra.device),c] for c in range(4)],1)
    elif mode != 'utility':
        raise ValueError('Undeclared selector mode')
    return torch.cat([z.detach(),extra.unsqueeze(-1)],-1)


class FrameActionRuleV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(41,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def expected_action_risk(logits, base, candidate):
    if logits.shape != base.shape or base.shape != candidate.shape:
        raise ValueError('Aligned logits and detached action risks required')
    p = logits.sigmoid()
    return condition_average((1-p)*base.detach()+p*candidate.detach())


def sequence_risk_interval(difference, sequences, seed=9132026, draws=2000):
    """Paired sequence-mean risk difference; this is not an AP interval."""
    difference = np.asarray(difference,dtype=np.float64)
    sequences = np.asarray(sequences)
    if difference.ndim != 1 or len(difference) != len(sequences) or not np.isfinite(difference).all():
        raise ValueError('Finite paired frame differences and recording names required')
    names = sorted(set(sequences.tolist()))
    means = np.asarray([difference[sequences==name].mean() for name in names])
    if len(means)<2:
        raise ValueError('At least two recordings required')
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(len(means),size=(draws,len(means)))].mean(1)
    return dict(sequence_mean=float(means.mean()),pointwise_95_interval=np.quantile(sampled,[.025,.975]).tolist(),
                recordings=len(names),draws=draws,seed=seed,estimand='equal-recording mean frame-risk difference; not AP')

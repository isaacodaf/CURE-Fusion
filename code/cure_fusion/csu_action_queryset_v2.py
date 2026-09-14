"""Prepared joint-query action representation; no detector or training runner.

Every query keeps its own 82 sigmoid scores and raw cxcywh box. The two
detectors share encoder weights but are pooled separately: query positions
across actions never imply object correspondence. Learned forward is CUDA-only.
Existing frame risks, action outputs, native top300 and evaluator stay external.
"""
import numpy as np
import torch
from torch import nn

from .csu_action_rules_v1 import CONDITION_WEIGHTS, permutation_indices


ACTION_ORDER_V2 = ('task_only', 'cure')
TOKEN_WIDTH_V2 = 86
QUERIES_V2 = 300
HIDDEN_V2 = 64
POOLED_WIDTH_V2 = 256
UTILITY_PARAMETERS_V2 = 26241
RULE_PARAMETERS_V2 = 26305
ROLES_V2 = ('controller_fit', 'inner_calibration', 'reused_development')


def require(ok, message):
    if not ok:
        raise ValueError(message)


@torch.no_grad()
def paired_query_tokens_v2(base, candidate):
    """Stateless [B,2,300,86] tokens; sigmoid is not a detector forward.

    Raw finite negative/zero dimensions are observations, not repaired boxes.
    No class projection, confidence threshold, query sorting or GT is used.
    """
    parts = []
    first = None
    for output in (base, candidate):
        require(isinstance(output, dict) and
                {'pred_logits', 'pred_boxes'} <= set(output), 'Original output arrays required')
        logits, boxes = output['pred_logits'], output['pred_boxes']
        require(isinstance(logits, torch.Tensor) and isinstance(boxes, torch.Tensor),
                'Tensor outputs required')
        require(logits.ndim == 3 and logits.shape[0] > 0 and
                logits.shape[1:] == (QUERIES_V2, 82) and
                boxes.shape == (*logits.shape[:2], 4), 'Original300-query82-class axes required')
        require(logits.dtype == boxes.dtype == torch.float32 and logits.device == boxes.device,
                'Common FP32 output/device required')
        require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all()),
                'Finite original outputs required')
        identity = (logits.shape, logits.device)
        require(first is None or identity == first, 'Two action batch/device contracts differ')
        first = identity
        parts.append(torch.cat((logits.detach().sigmoid(), boxes.detach()), dim=-1))
    return torch.stack(parts, dim=1)


def pool_query_embeddings_v2(embeddings):
    """Pure mean/max reduction, including each action's own score-box pairing."""
    require(isinstance(embeddings, torch.Tensor) and embeddings.dtype == torch.float32 and
            embeddings.ndim == 4 and embeddings.shape[0] > 0 and
            embeddings.shape[1:] == (2, QUERIES_V2, HIDDEN_V2), 'Expected [B,2,300,64] embeddings')
    require(bool(torch.isfinite(embeddings).all()), 'Finite embeddings required')
    return torch.cat((embeddings.mean(dim=2), embeddings.amax(dim=2)), dim=-1).flatten(1)


class FittingTokenMomentsV2:
    """FP64 streaming weighted moments with complete ordered fitting IDs only.

    Normalization shares all channels across both actions/queries. Conditions
    retain their original weights. Canonical identity binding belongs to the
    external, launch-disabled specification; this class checks the bound order.
    """
    def __init__(self, expected_ids):
        require(isinstance(expected_ids, (tuple, list)) and len(expected_ids) > 0 and
                all(isinstance(x, str) and x for x in expected_ids) and
                len(set(expected_ids)) == len(expected_ids), 'Unique nonempty fitting identities required')
        self.expected_ids = tuple(expected_ids)
        self.position = 0
        self.mass = 0.
        self.mean = np.zeros(TOKEN_WIDTH_V2, dtype=np.float64)
        self.m2 = np.zeros(TOKEN_WIDTH_V2, dtype=np.float64)
        self.finished = False

    def update(self, identities, tokens, *, role):
        require(not self.finished and role == 'controller_fit', 'Only unfinished fitting accumulation allowed')
        require(isinstance(identities, (tuple, list)) and len(identities) > 0 and
                tuple(identities) == self.expected_ids[self.position:self.position+len(identities)],
                'Fitting identity order/coverage differs')
        x = np.asarray(tokens)
        require(x.dtype == np.float32 and x.shape == (len(identities), 4, 2, QUERIES_V2, TOKEN_WIDTH_V2)
                and np.isfinite(x).all(), 'Finite FP32 complete condition/action/query tokens required')
        require(bool(((x[..., :82] >= 0) & (x[..., :82] <= 1)).all()), 'Sigmoid probability columns required')
        x = x.astype(np.float64)
        weight = np.asarray(CONDITION_WEIGHTS, np.float64)[None, :, None, None, None]
        mass = len(identities)*2*QUERIES_V2*float(np.sum(CONDITION_WEIGHTS))
        mean = (x*weight).sum(axis=(0, 1, 2, 3))/mass
        m2 = ((x-mean)**2*weight).sum(axis=(0, 1, 2, 3))
        total = self.mass+mass
        delta = mean-self.mean
        self.m2 += m2+delta**2*self.mass*mass/total
        self.mean += delta*mass/total
        self.mass = total
        self.position += len(identities)

    def finish(self):
        require(not self.finished and self.position == len(self.expected_ids), 'Incomplete or finalized fitting moments')
        require(self.mass > 0 and np.isfinite(self.mean).all() and np.isfinite(self.m2).all()
                and (self.m2 >= 0).all(), 'Invalid fitting moments')
        self.finished = True
        center = self.mean.astype(np.float32)
        scale = np.maximum(np.sqrt(self.m2/self.mass), 1e-4).astype(np.float32)
        require(np.isfinite(center).all() and np.isfinite(scale).all() and (scale > 0).all(),
                'Finite positive FP32 normalization required')
        return dict(center=center, scale=scale, frames=self.position,
                    weighted_query_mass=self.mass, condition_weights=list(CONDITION_WEIGHTS),
                    arithmetic='streamed FP64 weighted population variance; FP32 outputs; std floor1e-4')


class QuerySetEncoderV2(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        require(center.shape == scale.shape == (TOKEN_WIDTH_V2,) and
                bool(torch.isfinite(center).all()) and bool(torch.isfinite(scale).all()) and
                bool((scale > 0).all()), '86 fitting-derived finite feature scales required')
        self.register_buffer('center', center.detach().clone())
        self.register_buffer('scale', scale.detach().clone())
        self.phi = nn.Sequential(nn.Linear(TOKEN_WIDTH_V2, HIDDEN_V2), nn.SiLU(),
                                 nn.Linear(HIDDEN_V2, HIDDEN_V2), nn.SiLU())

    def forward(self, tokens):
        require(isinstance(tokens, torch.Tensor) and tokens.is_cuda and tokens.dtype == torch.float32 and
                tokens.ndim == 4 and tokens.shape[0] > 0 and
                tokens.shape[1:] == (2, QUERIES_V2, TOKEN_WIDTH_V2),
                'Learned encoding requires CUDA FP32 [B,2,300,86] tokens')
        require(tokens.device == self.center.device and bool(torch.isfinite(tokens).all()),
                'Finite token/normalizer device contract differs')
        require(bool(((tokens[..., :82] >= 0) & (tokens[..., :82] <= 1)).all()),
                'Native sigmoid columns required')
        # Frozen outputs never receive gradients; phi remains trainable.
        embeddings = self.phi((tokens.detach()-self.center)/self.scale)
        return pool_query_embeddings_v2(embeddings)


class FrameActionQueryUtilityV2(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        self.encoder = QuerySetEncoderV2(center, scale)
        self.net = nn.Sequential(nn.Linear(POOLED_WIDTH_V2, HIDDEN_V2), nn.SiLU(), nn.Linear(HIDDEN_V2, 1))

    def forward(self, tokens):
        return self.net(self.encoder(tokens)).squeeze(-1)


class FrameActionQueryRuleV2(nn.Module):
    """Each matched arm owns an identically initialized, separately trained encoder.

    The utility predictor's encoder is distinct and wholly frozen first. Every
    arm gets the same rich tokens; only the detached final scalar differs.
    """
    def __init__(self, center, scale):
        super().__init__()
        self.encoder = QuerySetEncoderV2(center, scale)
        self.net = nn.Sequential(nn.Linear(POOLED_WIDTH_V2+1, HIDDEN_V2), nn.SiLU(), nn.Linear(HIDDEN_V2, 1))

    def forward(self, tokens, extra):
        require(isinstance(extra, torch.Tensor) and extra.dtype == torch.float32 and
                extra.shape == tokens.shape[:1] and extra.device == tokens.device and
                bool(torch.isfinite(extra).all()), 'Aligned finite FP32 detached utility scalar required')
        return self.net(torch.cat((self.encoder(tokens), extra.detach().unsqueeze(-1)), dim=-1)).squeeze(-1)


def queryset_rule_extra_v2(mean, gain_scale, mode, seed, role):
    """Preserve the original utility/zero/permutation law without 40 summaries."""
    require(isinstance(mean, torch.Tensor) and mean.dtype == torch.float32 and mean.ndim == 2 and
            mean.shape[0] > 0 and mean.shape[1] == 4 and bool(torch.isfinite(mean).all()),
            'Finite four-condition predicted mean required')
    require(isinstance(gain_scale, torch.Tensor) and gain_scale.dtype == mean.dtype and gain_scale.ndim == 0 and
            gain_scale.device == mean.device and bool(torch.isfinite(gain_scale).all()) and
            float(gain_scale.detach()) > 0, 'Original positive fitting gain RMS required')
    require(role in ROLES_V2 and isinstance(seed, int) and not isinstance(seed, bool), 'Declared role/seed required')
    extra = mean.detach()/gain_scale.detach()
    if mode == 'zero':
        return torch.zeros_like(extra)
    if mode == 'permuted':
        return torch.stack([extra[torch.as_tensor(permutation_indices(len(extra), seed, c, role),
                                                  device=extra.device), c] for c in range(4)], dim=1)
    require(mode == 'utility', 'Undeclared selector mode')
    return extra

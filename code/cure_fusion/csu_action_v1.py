"""Frame-level detector-action utility candidate, separate from removal CSU.

The target is R(base)-R(candidate) under one common four-class frame risk.
Assignments are made independently for each whole prediction set against the
same GT. Classification includes unmatched queries and empty frames. The
focal action risk is new; it is not the historical fixed-VFL object estimand.
"""
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .dfine_cure_v1 import require


def positive_box_giou_v1(a, b):
    """Exact broadcast GIoU for representable positive cxcywh boxes.

    No epsilon floor: small valid boxes retain their geometric ratio. Reject
    collapsed/undefined corner geometry instead of silently changing its area.
    """
    require(a.shape[-1] == b.shape[-1] == 4 and a.dtype == b.dtype and a.device == b.device,
            'Common box dtype/device required')
    alo, ahi = a[..., :2]-a[..., 2:]/2, a[..., :2]+a[..., 2:]/2
    blo, bhi = b[..., :2]-b[..., 2:]/2, b[..., :2]+b[..., 2:]/2
    require(bool((ahi > alo).all()) and bool((bhi > blo).all()),
            'Positive representable box corners required')
    area_a, area_b = (ahi-alo).prod(-1), (bhi-blo).prod(-1)
    intersection = (torch.minimum(ahi, bhi)-torch.maximum(alo, blo)).clamp(min=0).prod(-1)
    union = area_a+area_b-intersection
    enclosure = (torch.maximum(ahi, bhi)-torch.minimum(alo, blo)).prod(-1)
    require(bool((union > 0).all()) and bool((enclosure > 0).all()), 'Positive union/enclosure required')
    value = intersection/union-(enclosure-union)/enclosure
    require(bool(torch.isfinite(value).all()), 'Finite geometric ratios required')
    return value


@torch.no_grad()
def frame_action_risk_v1(logits, boxes, targets):
    """Detached main-head focal(alpha=.75,gamma=2)+5L1+2GIoU per frame.

    Matching uses focal alpha=.25/gamma=2 and 2/5/2 costs. Raw invalid boxes
    are retained, but cannot match a factual GT. Their classification still
    contributes background loss. Deployment validity is a separate admission;
    this function does not repair/export boxes or compute AP.
    """
    require(logits.ndim == 3 and logits.shape[0] > 0 and logits.shape[1] > 0 and logits.shape[-1] == 4 and
            boxes.shape == (*logits.shape[:2], 4), 'Aligned batch/query/four-class outputs required')
    require(logits.is_floating_point() and boxes.dtype == logits.dtype and boxes.device == logits.device,
            'Common floating-point output dtype/device required')
    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all()), 'Finite action outputs required')
    require(len(targets) == len(logits), 'Frame/target count differs')
    risks, records = [], []
    for x, b, target in zip(logits, boxes, targets):
        labels, truth = target['labels'], target['boxes']
        require(labels.dtype == torch.long and labels.ndim == 1 and truth.shape == (len(labels), 4),
                'Complete class labels and normalized cxcywh targets required')
        require(bool(((labels >= 0) & (labels < 4)).all()) and bool(torch.isfinite(truth).all()) and
                bool((truth[:, 2:] > 0).all()), 'Invalid four-class target')
        labels, truth = labels.to(x.device), truth.to(device=x.device, dtype=x.dtype)
        hot = torch.zeros_like(x)
        valid = torch.nonzero((b[:, 2:] > 0).all(-1), as_tuple=False).flatten()
        require(len(valid) >= len(labels), 'Insufficient valid queries for every GT object')
        q = torch.empty(0, dtype=torch.long, device=x.device)
        o = q.clone()
        l1 = x.new_zeros(())
        giou = x.new_zeros(())
        if len(labels):
            prob = x[valid].sigmoid()[:, labels]
            positive = .25 * (1-prob).square() * (-(prob+1e-8).log())
            negative = .75 * prob.square() * (-(1-prob+1e-8).log())
            cost = 2*(positive-negative) + 5*torch.cdist(b[valid], truth, p=1) - 2*positive_box_giou_v1(b[valid, None], truth[None])
            require(bool(torch.isfinite(cost).all()), 'Nonfinite action assignment costs')
            qi, oi = linear_sum_assignment(cost.cpu().numpy())
            q = valid[torch.as_tensor(qi, device=x.device)]
            o = torch.as_tensor(oi, device=x.device)
            require(len(q) == len(labels), 'Incomplete action assignment')
            hot[q, labels[o]] = 1
            l1 = (b[q]-truth[o]).abs().sum()
            giou = (1-positive_box_giou_v1(b[q], truth[o])).sum()
        probability = x.sigmoid()
        pt = probability*hot + (1-probability)*(1-hot)
        alpha = .75*hot + .25*(1-hot)
        focal = (alpha*(1-pt).square()*F.binary_cross_entropy_with_logits(x, hot, reduction='none')).sum()
        denominator = max(len(labels), 1)
        risk = (focal+5*l1+2*giou)/denominator
        require(bool(torch.isfinite(risk)) and float(risk) >= 0, 'Invalid common frame risk')
        risks.append(risk)
        records.append({'queries': q, 'objects': o, 'focal': focal/denominator,
                        'L1': l1/denominator, 'GIoU': giou/denominator,
                        'GT': len(labels), 'invalid_unassigned_queries': int(len(b)-len(valid))})
    return torch.stack(risks), records


@torch.no_grad()
def action_gain_targets_v1(base, candidate, targets):
    require(base['pred_logits'].shape == candidate['pred_logits'].shape,
            'Matched action class/query capacity required')
    rb, mb = frame_action_risk_v1(base['pred_logits'], base['pred_boxes'], targets)
    rc, mc = frame_action_risk_v1(candidate['pred_logits'], candidate['pred_boxes'], targets)
    return {'gain': rb-rc, 'base_risk': rb, 'candidate_risk': rc,
            'base_assignment': mb, 'candidate_assignment': mc,
            'estimand': 'common_focal_frame_base_risk_minus_candidate_risk_v1'}


@torch.no_grad()
def action_features_v1(base, candidate):
    """40 inference-only summaries; no GT, losses or object matching input.

    Per action: classwise sigmoid max/mean/std (12), box mean/std (8).
    This simple summary is a declared baseline interface, not an optimized
    representation. Both complete detector actions must already be computed.
    """
    parts = []
    for out in (base, candidate):
        logits, boxes = out['pred_logits'], out['pred_boxes']
        require(logits.ndim == 3 and logits.shape[-1] == 4 and
                boxes.shape == (*logits.shape[:2], 4), 'Canonical four-class action outputs required')
        require(logits.shape[1] > 0 and bool(torch.isfinite(logits).all()) and
                bool(torch.isfinite(boxes).all()), 'Finite nonempty action outputs required')
        p = logits.sigmoid()
        parts.extend((p.amax(1), p.mean(1), p.std(1, unbiased=False),
                      boxes.mean(1), boxes.std(1, unbiased=False)))
    return torch.cat(parts, dim=-1)


class FrameActionUtilityV1(nn.Module):
    """Trained action-utility baseline; regression and policy roles stay distinct.

    Scaling statistics must come only from fitting inputs. Freeze this entire
    module before controller-rule comparisons or independent calibration.
    CPU use is for component tests; the admission/training runner requires CUDA.
    """
    def __init__(self, center, scale, hidden=64):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        require(center.shape == scale.shape == (40,) and bool(torch.isfinite(center).all()) and
                bool(torch.isfinite(scale).all()) and bool((scale > 0).all()), '40 fitting-derived finite feature scales required')
        self.register_buffer('center', center.clone())
        self.register_buffer('scale', scale.clone())
        self.network = nn.Sequential(nn.Linear(40, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features):
        require(features.ndim == 2 and features.shape[1] == 40 and bool(torch.isfinite(features).all()),
                'Finite frame features required')
        return self.network((features-self.center)/self.scale).squeeze(-1)


def detached_action_policy_loss_v1(probability_candidate, base_risk, candidate_risk):
    """Differentiable expected fixed-action risk; gradients only enter the rule."""
    require(probability_candidate.shape == base_risk.shape == candidate_risk.shape and
            bool(torch.isfinite(probability_candidate).all()) and
            bool(((probability_candidate >= 0) & (probability_candidate <= 1)).all()), 'Aligned action probabilities and risks required')
    require(bool(torch.isfinite(base_risk).all()) and bool(torch.isfinite(candidate_risk).all()), 'Finite action risks required')
    return ((1-probability_candidate)*base_risk.detach() + probability_candidate*candidate_risk.detach()).mean()

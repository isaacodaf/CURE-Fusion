"""Read-only factual representation witness and original CSU head control.

This is a post-hoc supervised readout of a CURE-trained representation. It
neither modifies the detector nor creates a new counterfactual target.
"""
import copy
import math
import numpy as np
import torch
from .dfine_cure_v1 import require
from .dfine_cure_losses_v1 import cev_mean_nll_from_sampling_variance_v1


class FactualUtilityInputWitnessV1:
    """Observe the one actual .cev input; never replace arguments or outputs."""
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __enter__(self):
        require(not self.model.training, 'Frozen factual model must be in eval mode')
        self.handle = self.model.cev.register_forward_pre_hook(self._observe)
        return self

    def _observe(self, module, args):
        require(len(args) == 1, 'Original utility head has one input')
        x = args[0]
        require(x.ndim == 3 and x.shape[1:] == (300, 512), 'Original factual 300x512 contract')
        self.calls.append(x.detach().clone())
        return None

    def __exit__(self, *args):
        self.handle.remove()

    def features(self):
        require(len(self.calls) == 1, 'Exactly one factual utility-head call required')
        x = self.calls[0]
        require(bool(torch.isfinite(x).all()), 'Nonfinite factual features')
        return x


def original_head_state_v1(initial):
    """The original common initialization, never the fitted CURE utility head."""
    expected = {'0.weight': (128, 512), '0.bias': (128,), '1.weight': (128,),
                '1.bias': (128,), '3.weight': (4, 128), '3.bias': (4,)}
    state = {k[4:]: v.detach().clone() for k, v in initial.items() if k.startswith('cev.')}
    require(set(state) == set(expected), 'Exact six original cev tensors required')
    for k, shape in expected.items():
        require(tuple(state[k].shape) == shape and state[k].dtype == torch.float32,
                'Original utility-head tensor contract differs: ' + k)
        require(bool(torch.isfinite(state[k]).all()), 'Nonfinite original head')
    require(sum(x.numel() for x in state.values()) == 66436, 'Original head capacity differs')
    return state


def fresh_original_head_v1(frozen_model, initial):
    require(next(frozen_model.parameters()).is_cuda, 'Control construction is CUDA-only')
    head = copy.deepcopy(frozen_model.cev)
    head.load_state_dict(original_head_state_v1(initial), strict=True)
    head.requires_grad_(True).train()
    return head


def learning_rate_v1(epoch):
    require(1 <= epoch <= 40, 'Fixed forty-epoch budget')
    return 1e-5 + .5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (epoch - 1) / 39))


def ordered_query_indices_v1(queries, objects, count):
    """Complete one-to-one object correspondence; no error-dependent selection."""
    q, o = np.asarray(queries), np.asarray(objects)
    require(q.dtype.kind in 'iu' and o.dtype.kind in 'iu' and q.shape == o.shape == (count,),
            'Complete integer object/query mapping required')
    require(np.array_equal(np.sort(o), np.arange(count)) and len(np.unique(q)) == count,
            'Duplicate or incomplete object/query correspondence')
    require(bool(((q >= 0) & (q < 300)).all()), 'Query outside original native slots')
    return q[np.argsort(o)].astype(np.int64)


def utility_batch_loss_v1(raw, target, sampling_variance, object_mask):
    """Original .1 mean NLL in physical units, with empty-only graph zeros."""
    require(raw.shape == (*object_mask.shape, 4) and target.shape == sampling_variance.shape ==
            (*object_mask.shape, 2), 'Original two-modal object moments required')
    if not bool(object_mask.any()):
        return raw.sum() * 0
    mean, lv = raw[..., :2], raw[..., 2:].clamp(-8., 5.)
    return .1 * cev_mean_nll_from_sampling_variance_v1(
        mean, lv, target, sampling_variance, object_mask[..., None].expand_as(mean))


def descriptive_metrics_v1(raw, target, sampling_variance):
    """Saved-array assessment only; no AP and no inferential interval claim."""
    raw, target, sampling_variance = [np.asarray(v, np.float64) for v in
                                     (raw, target, sampling_variance)]
    require(raw.ndim == 2 and raw.shape[1] == 4 and target.shape == sampling_variance.shape ==
            (len(raw), 2) and len(raw) > 0, 'Complete nonempty two-modal assessment required')
    require(all(np.isfinite(v).all() for v in (raw, target, sampling_variance)) and
            (sampling_variance >= 0).all(), 'Invalid assessment operands')
    error = raw[:, :2] - target
    total = np.exp(np.clip(raw[:, 2:], -8, 5)) + sampling_variance
    return dict(objects=len(raw), modality_order=['camera_removal', 'radar_thinning'],
                RMSE=np.sqrt(np.mean(error**2, 0)).tolist(), MAE=np.abs(error).mean(0).tolist(),
                bias=error.mean(0).tolist(), Gaussian_mean_NLL_without_additive_constant=
                (.5 * (error**2 / total + np.log(total))).mean(0).tolist(),
                central90_estimated_target_mean_coverage=
                (np.abs(error) <= 1.6448536269514722 * np.sqrt(total)).mean(0).tolist())

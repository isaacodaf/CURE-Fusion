"""Exact saved-array mean-target assessment; no Torch dependency."""
import numpy as np
def require(value,message):
    if not value: raise ValueError(message)

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


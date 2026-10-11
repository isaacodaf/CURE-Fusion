"""Fitting-only routing denominators; utility targets retain task-loss units."""
import numpy as np
import torch


def fit_routing_temperatures(mean, object_mask, floor=0.75):
    mean = np.asarray(mean)
    object_mask = np.asarray(object_mask)
    if mean.shape[:-1] != object_mask.shape or mean.shape[-1] != 2:
        raise ValueError("Expected object means [..., camera/radar] and an aligned mask")
    if object_mask.dtype != np.bool_ or not object_mask.any():
        raise ValueError("At least one valid fitting object is required")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("The temperature floor must be finite and positive")
    values = mean[object_mask].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite fitting target")
    rms = np.sqrt(np.mean(values ** 2, axis=0))
    return np.maximum(float(floor), rms), rms


def scaled_routing_bias(raw_risk, temperatures, bound=0.5):
    if raw_risk.shape[-1] != 2 or temperatures.shape != (2,):
        raise ValueError("Two modality temperatures are required")
    if not bool(torch.isfinite(temperatures).all()) or not bool((temperatures > 0).all()):
        raise ValueError("Routing temperatures must be finite and positive")
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("The routing bound must be finite and nonnegative")
    # CUDA scalar division and tensor division can round differently. Keep the
    # original arithmetic exactly for each unchanged 0.75-temperature channel.
    normalized = torch.where(temperatures == 0.75, raw_risk / 0.75, raw_risk / temperatures)
    return float(bound) * torch.tanh(normalized)


def routing_diagnostics(raw_risk, temperatures):
    """Descriptive response statistics, without AP or calibration claims."""
    values = np.asarray(raw_risk, dtype=np.float64).reshape(-1, 2)
    temperature = np.asarray(temperatures, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty risk observations are required")
    response = np.tanh(values / temperature)
    return {
        "observations": len(values),
        "temperature": temperature.tolist(),
        "fraction_abs_tanh_ge_099": (np.abs(response) >= 0.99).mean(0).tolist(),
        "median_relative_tanh_slope": np.median(1 - response ** 2, axis=0).tolist(),
        "modality_order": ["camera_removal", "radar_thinning"],
    }

from __future__ import annotations
import numpy as np


def binary_average_precision(target, score, sample_weight=None):
    """Tie-aware average precision with optional nonnegative sample weights."""
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    if y.shape != s.shape or len(y) == 0:
        raise ValueError("target and score must be nonempty arrays of equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("target must be binary and scores finite")
    w = (
        np.ones(len(y), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("sample weights must be finite, nonnegative, and aligned")
    positive_weight = float(np.sum(w * y))
    if positive_weight <= 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    threshold_end = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[threshold_end], fp[threshold_end]
    recall = tp / positive_weight
    precision = tp / np.maximum(tp + fp, 1e-15)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def multilabel_average_precision(target, score, sample_weight=None):
    """Return macro, micro, and per-class AP for a binary query matrix."""
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(score, dtype=float)
    if y.ndim != 2 or y.shape != s.shape:
        raise ValueError("target and score must have the same [samples, classes] shape")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if w is not None and len(w) != len(y):
        raise ValueError("sample weights must align with samples")
    per_class = np.array(
        [binary_average_precision(y[:, column], s[:, column], w) for column in range(y.shape[1])],
        dtype=float,
    )
    tiled_weight = None if w is None else np.repeat(w, y.shape[1])
    micro = binary_average_precision(y.reshape(-1), s.reshape(-1), tiled_weight)
    return {
        "macro_ap": float(np.nanmean(per_class)),
        "micro_ap": micro,
        "per_class_ap": per_class.tolist(),
    }


def sequence_bootstrap_ap_difference(
    candidate_score,
    control_score,
    target,
    sequences,
    *,
    n_boot=10_000,
    seed=0,
):
    """Paired cluster bootstrap of macro-AP difference over whole sequences."""
    candidate = np.asarray(candidate_score, dtype=float)
    control = np.asarray(control_score, dtype=float)
    y = np.asarray(target, dtype=np.int8)
    groups = np.asarray(sequences)
    if candidate.shape != control.shape or candidate.shape != y.shape or y.ndim != 2:
        raise ValueError("candidate, control, and target must share [samples, classes]")
    if groups.shape != (len(y),):
        raise ValueError("one sequence identifier is required per sample")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two independent sequences are required")
    observed = (
        multilabel_average_precision(y, candidate)["macro_ap"]
        - multilabel_average_precision(y, control)["macro_ap"]
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(n_boot), dtype=float)
    for draw in range(int(n_boot)):
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique)).astype(float)
        weight = counts[inverse]
        deltas[draw] = (
            multilabel_average_precision(y, candidate, weight)["macro_ap"]
            - multilabel_average_precision(y, control, weight)["macro_ap"]
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))),
    )
    return {
        "n_sequences": int(len(unique)),
        "replicates": int(n_boot),
        "observed_difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "two_sided_bootstrap_p": p_value,
    }


def graceful_degradation_integral(severity, performance, normalize=True):
    s = np.asarray(severity, dtype=float)
    p = np.asarray(performance, dtype=float)
    order = np.argsort(s)
    s, p = s[order], p[order]
    if normalize:
        p = p / max(p[np.argmin(np.abs(s))], 1e-12)
    return float(np.trapezoid(p, s) / max(s[-1]-s[0], 1e-12))


def utility_calibration_error(predicted, observed, n_bins=10):
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    obs = np.asarray(observed, dtype=float).reshape(-1)
    if len(pred) != len(obs):
        raise ValueError("predicted and observed lengths differ")
    if len(pred) == 0:
        return float("nan")
    lo, hi = pred.min(), pred.max()
    if hi <= lo:
        return float(abs(pred.mean()-obs.mean()))
    edges = np.linspace(lo, hi, n_bins+1)
    total = 0.0
    for i in range(n_bins):
        m = (pred >= edges[i]) & ((pred <= edges[i+1]) if i == n_bins-1 else (pred < edges[i+1]))
        if m.any():
            total += m.mean() * abs(pred[m].mean() - obs[m].mean())
    return float(total)


def signed_rank_accuracy(predicted, observed):
    pred = np.asarray(predicted)
    obs = np.asarray(observed)
    if pred.ndim != 2 or pred.shape != obs.shape:
        raise ValueError("expected [N,M] arrays")
    return float((pred.argmax(1) == obs.argmax(1)).mean())


def paired_scene_bootstrap(delta_by_scene, n_boot=10000, seed=0):
    """Return mean delta and percentile 95% CI over independent scenes."""
    x = np.asarray(delta_by_scene, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), tuple(float(v) for v in np.quantile(samples, [0.025, 0.975]))

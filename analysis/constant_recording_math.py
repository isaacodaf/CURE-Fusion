"""Exact recorded constant-comparison arithmetic; no model."""
import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError(message)


def recording_statistics(errors, group_index, groups):
    """FP64 sums; axes: recording, predictor, modality, squared/absolute error."""
    errors = np.asarray(errors, np.float64)
    group_index = np.asarray(group_index)
    require(errors.ndim == 3 and errors.shape[1:] == (4, 2), "Error shape")
    require(group_index.shape == (len(errors),) and group_index.dtype.kind in "iu", "Group index")
    require(groups > 0 and np.all((group_index >= 0) & (group_index < groups)), "Group domain")
    require(np.isfinite(errors).all(), "Nonfinite saved error")
    terms = np.stack((errors ** 2, np.abs(errors)), axis=-1)
    require(np.isfinite(terms).all(), "Overflow in error terms")
    sums = np.zeros((groups, 4, 2, 2), np.float64)
    np.add.at(sums, group_index, terms)
    counts = np.bincount(group_index.astype(np.int64), minlength=groups).astype(np.int64)
    require(np.isfinite(sums).all(), "Overflow in recording sums")
    return counts, sums


def pooled_metrics(weights, counts, sums):
    """Metrics are object-pooled after whole-recording duplication, never mean RMSE."""
    weights = np.asarray(weights)
    counts, sums = np.asarray(counts), np.asarray(sums, np.float64)
    require(weights.ndim == 2 and weights.shape[1] == len(counts), "Weight shape")
    require(weights.dtype.kind in "iu" and np.all(weights >= 0), "Nonnegative integer multiplicities")
    require(counts.dtype.kind in "iu" and np.all(counts >= 0), "Object counts")
    require(sums.shape == (len(counts), 4, 2, 2), "Sufficient-statistic shape")
    require(np.isfinite(sums).all() and np.all(sums >= 0), "Sufficient-statistic domain")
    objects = weights @ counts
    numerator = np.einsum("bg,gpmt->bpmt", weights.astype(np.float64), sums, optimize=False)
    metrics = np.full(numerator.shape, np.nan, np.float64)
    np.divide(numerator, objects[:, None, None, None], out=metrics,
              where=objects[:, None, None, None] > 0)
    metrics[..., 0] = np.sqrt(metrics[..., 0])
    return objects, metrics


def contrasts(metrics):
    """CURE minus baseline and percent error reduction; exactly-zero denominator is undefined."""
    metrics = np.asarray(metrics, np.float64)
    require(metrics.ndim == 4 and metrics.shape[1:] == (4, 2, 2), "Metric shape")
    cure, baseline = metrics[:, :1], metrics[:, 1:]
    difference = cure - baseline
    ratio = np.full(baseline.shape, np.nan, np.float64)
    np.divide(cure, baseline, out=ratio, where=np.isfinite(baseline) & (baseline > 0))
    return difference, 100.0 * (1.0 - ratio)


def paired_recording_draws(groups, draws, seed):
    require(groups > 1 and draws > 0 and type(seed) is int, "Bootstrap declaration")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, groups, size=(draws, groups), dtype=np.int64)
    weights = np.zeros((draws, groups), np.int64)
    np.add.at(weights, (np.arange(draws)[:, None], indices), 1)
    require(np.array_equal(weights.sum(axis=1), np.full(draws, groups)), "Draw size")
    return indices, weights


def pointwise_interval(values):
    """Never omit an undefined draw to manufacture a conditional interval."""
    values = np.asarray(values, np.float64)
    require(values.ndim == 1 and len(values) > 0, "Interval shape")
    good = np.isfinite(values)
    return {
        "percentile_95": (np.quantile(values, [0.025, 0.975], method="linear").tolist()
                          if good.all() else None),
        "defined_draws": int(good.sum()), "undefined_draws": int((~good).sum()),
        "policy": "All draws retained; interval unavailable if any draw is undefined",
    }


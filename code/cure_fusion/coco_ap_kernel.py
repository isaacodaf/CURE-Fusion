"""Experimental CPU AP accumulation without replicated detection matrices.

Input matches and tied blocks come from the existing COCO evaluator. This
kernel preserves block/image replication order; it does not recompute IoU.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

RECALL_THRESHOLDS = np.linspace(0., 1., 101)
EPSILON = np.spacing(1.)

@njit(cache=True, fastmath=False, nogil=True)
def category_ap(gt, tp, fp, starts, lengths, images, weights):
    positives = 0
    for i in range(len(gt)):
        positives += gt[i] * weights[i]
    if positives == 0:
        return np.nan
    sampled = np.zeros((10, 101), dtype=np.float64)
    for threshold in range(10):
        # All observed cumulative TP counts are contiguous: each detection
        # increments TP by at most one. Store their best precision directly.
        best = np.zeros(positives + 1, dtype=np.float64)
        true_count, false_count = 0, 0
        observed = False
        for group in range(len(starts)):
            for repeat in range(weights[images[group]]):
                for offset in range(lengths[group]):
                    index = starts[group] + offset
                    true_count += int(tp[threshold, index])
                    false_count += int(fp[threshold, index])
                    if true_count > positives:
                        raise ValueError('TP count exceeds ground-truth count')
                    precision = true_count / (true_count + false_count + EPSILON)
                    best[true_count] = max(best[true_count], precision)
                    observed = True
        if not observed:
            continue
        for count in range(true_count - 1, -1, -1):
            best[count] = max(best[count], best[count + 1])
        count = 0
        for j in range(101):
            # Match searchsorted(recall, threshold, side='left'), including
            # binary floating-point boundary behavior of recall division.
            while count <= true_count and count / positives < RECALL_THRESHOLDS[j]:
                count += 1
            if count <= true_count:
                sampled[threshold, j] = best[count]
    return sampled.mean()


def accumulated_ap(evaluator, image_weights=None):
    weights = np.ones(evaluator.images, dtype=np.int64) if image_weights is None else np.asarray(image_weights)
    if (weights.shape != (evaluator.images,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.equal(weights, np.floor(weights)).all()):
        raise ValueError('Nonnegative integer replication count per image required')
    weights = weights.astype(np.int64)
    values = [category_ap(c['gt'], c['tp'], c['fp'], c['starts'], c['lengths'], c['images'], weights)
              for c in evaluator.classes]
    values = [value for value in values if not np.isnan(value)]
    return float(np.mean(values)) if values else None


def parallel_paired_bootstrap(candidate, control, *, n_boot=10000, seed=20260905, workers=8, progress=None):
    """Precompute ordered AP draws, then reuse the canonical statistical summary."""
    from .coco_cluster_bootstrap import sequence_draw_weights, paired_pooled_bootstrap
    if not candidate or len(candidate) != len(control):
        raise ValueError('Paired nonempty collections required')
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 16:
        raise ValueError('One to sixteen CPU workers required')
    packs = [*candidate, *control]
    sequences = packs[0].sequences
    if any(pack.sequences != sequences for pack in packs):
        raise ValueError('Identical image/sequence order required')
    weights = list(sequence_draw_weights(sequences, n_boot, seed))
    points = [accumulated_ap(pack) for pack in packs]  # Warm JIT before workers.
    if any(value is None for value in points):
        raise ValueError('Pooled AP undefined without ground truth')
    def compute(weight):
        return [np.nan if (value := accumulated_ap(pack, weight)) is None else value for pack in packs]
    values = np.empty((n_boot, len(packs)), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(compute, weights)):
            values[i] = row
            if progress and ((i + 1) % 100 == 0 or i + 1 == n_boot):
                progress(i + 1)

    class OrderedCache:
        def __init__(self, column):
            self.sequences = sequences
            self.column, self.cursor = column, 0

        def ap(self, weight=None):
            if weight is None:
                return points[self.column]
            if self.cursor >= n_boot or not np.array_equal(weight, weights[self.cursor]):
                raise ValueError('Canonical draw order differs from precomputed AP')
            value = values[self.cursor, self.column]
            self.cursor += 1
            return None if np.isnan(value) else float(value)

    cached = [OrderedCache(i) for i in range(len(packs))]
    count = len(candidate)
    summary, draws = paired_pooled_bootstrap(cached[:count], cached[count:], n_boot=n_boot, seed=seed)
    if any(pack.cursor != n_boot for pack in cached):
        raise ValueError('Canonical summary did not consume every AP draw')
    return summary, draws

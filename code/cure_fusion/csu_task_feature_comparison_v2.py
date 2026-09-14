"""Object-identity joins for a post-hoc Task-feature utility readout control.

Pure arrays only. Prior CURE query indices are comparison provenance; they never
select a Task feature. Original annotation order defines object_index.
"""
import numpy as np


def demand(ok, message):
    if not ok:
        raise ValueError(message)


def align_cure_readouts_v2(task_records, task_joined, cure_records, cure_operands):
    """Return prior raw readouts aligned to every Task object, including empty frames."""
    def index_records(records):
        by_id = {r['identity']: (i, r) for i, r in enumerate(records)}
        demand(len(by_id) == len(records), 'Duplicate frame identity')
        return by_id
    task_index = index_records(task_records)
    cure_index = index_records(cure_records)
    demand(set(task_index) == set(cure_index), 'Different comparison frame populations')
    mask = np.asarray(task_joined['object_mask'])
    old_mask = np.asarray(cure_operands['object_mask'])
    demand(mask.dtype == old_mask.dtype == np.bool_ and mask.ndim == old_mask.ndim == 2,
           'Boolean object masks required')
    demand(len(mask) == len(task_records) and len(old_mask) == len(cure_records), 'Mask frame axis differs')
    result = {name: np.zeros((*mask.shape, 4), np.float32) for name in
              ('trained_CURE_feature_raw', 'original_saved_CURE_raw')}
    result['prior_CURE_queries'] = np.full(mask.shape, -1, np.int64)
    for tid, (ti, tr) in task_index.items():
        ci, cr = cure_index[tid]
        demand(tr['sequence'] == cr['sequence'] and tr['labels'] == cr['labels'],
               'Object identity/label/sequence carrier differs')
        n = len(tr['labels'])
        demand(np.array_equal(mask[ti], np.arange(mask.shape[1]) < n) and
               np.array_equal(old_mask[ci], np.arange(old_mask.shape[1]) < n),
               'Incomplete or reordered object mask')
        tq = np.asarray(tr['object_order_queries'], dtype=np.int64)
        cq = np.asarray(cr['object_order_queries'], dtype=np.int64)
        demand(tq.shape == cq.shape == (n,) and
               np.array_equal(tq, task_joined['queries'][ti, :n]) and
               np.array_equal(cq, cure_operands['queries'][ci, :n]), 'Recorded query carrier differs')
        demand(len(set(tq.tolist())) == len(set(cq.tolist())) == n and
               ((tq >= 0) & (tq < 300)).all() and ((cq >= 0) & (cq < 300)).all(),
               'Incomplete factual matching')
        for key in ('mean', 'mean_sampling_variance'):
            a, b = np.asarray(task_joined[key][ti, :n]), np.asarray(cure_operands[key][ci, :n])
            demand(a.shape == b.shape == (n, 2) and a.dtype == b.dtype == np.float32 and
                   np.isfinite(a).all() and np.isfinite(b).all() and np.array_equal(a, b),
                   'Original physical object target differs: ' + key)
        for dst, src in [('trained_CURE_feature_raw', 'control_raw'),
                         ('original_saved_CURE_raw', 'historical_cure_raw')]:
            raw = np.asarray(cure_operands[src][ci, :n])
            demand(raw.shape == (n, 4) and raw.dtype == np.float32 and np.isfinite(raw).all(),
                   'Invalid comparison raw values')
            result[dst][ti, :n] = raw
        result['prior_CURE_queries'][ti, :n] = cq
    return result

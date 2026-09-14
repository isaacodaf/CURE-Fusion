"""Fixed native four-class 2D fitting contracts; no detector construction.

Metric arithmetic is the previous frozen fitting audit. Only native category
transport differs: 0..3 are mapped to that serializer's task channel labels.
"""
import json
from pathlib import Path
import numpy as np
import torch
from cure_fusion.dfine_full_native_four_class_v2 import sha256_file, tensor_binding
from cure_fusion.dfine_cure_absent_radar_overfit_v1 import (
    BUDGETS, CLASSES, evaluate_mode, ground_truth, selected_records, state_digest)


def batch_schedule(size, updates, positive, seed=11):
    """Seeded full passes; pair empties with positives, never discard a frame."""
    if size not in BUDGETS or updates != BUDGETS[size] or seed != 11:
        raise ValueError('Exact fitting prefix, budget and seed required')
    positive = np.asarray(positive)
    if positive.shape != (size,) or positive.dtype != np.bool_ or positive.sum() < (~positive).sum():
        raise ValueError('Declared prefix needs at least one positive per empty frame')
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < updates:
        order = rng.permutation(size).tolist()
        nonempty = [i for i in order if positive[i]]
        empty = [i for i in order if not positive[i]]
        units = [[i] + ([empty[j]] if j < len(empty) else []) for j, i in enumerate(nonempty)]
        current = []
        for unit in units:
            if len(current) + len(unit) > 4:
                result.append(current); current = []
            current += unit
        if current: result.append(current)
    return result[:updates]


def native_selected_records(records, labels, boxes, scores):
    labels = np.asarray(labels)
    if labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels > 3):
        raise ValueError('Native four-class top300 labels required')
    # Serialization transport only; the model uses four sigmoid channels.
    transport = np.asarray((0, 1, 80, 81), np.int64)[labels]
    return selected_records(records, transport, boxes, scores)


def expected_loss_keys(denoising):
    base = ('loss_vfl', 'loss_bbox', 'loss_giou')
    keys = set(base + ('loss_fgl',))
    for suffix in ('_pre', '_enc_0'):
        keys.update(k + suffix for k in base)
    for i in range(5):
        keys.update(k + '_aux_%d' % i for k in base + ('loss_fgl', 'loss_ddf'))
    if denoising:
        keys.update(k + '_dn_pre' for k in base)
        for i in range(6):
            keys.update(k + '_dn_%d' % i for k in base + ('loss_fgl', 'loss_ddf'))
    return keys


def equal_tree(expected, actual):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and expected.dtype == actual.dtype and
                torch.equal(expected.detach().cpu(), actual.detach().cpu()))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys() and
                all(equal_tree(v, actual[k]) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(expected) is type(actual) and len(expected) == len(actual) and
                all(equal_tree(a, b) for a, b in zip(expected, actual)))
    return type(expected) is type(actual) and expected == actual


def verify_admission(root, spec, scientific):
    """Exact successful V3 report identity; no new experiment inherits its step."""
    if not isinstance(spec, dict):
        raise ValueError('Completed native V3 admission must be bound before launch')
    root = Path(root)
    for name in ('REPORT.json', 'ARTIFACT_HASHES.json', 'protocol.json'):
        if sha256_file(root/name) != spec['files'][name]:
            raise ValueError('Actual native admission identity differs: ' + name)
    local = Path(__file__).resolve().parents[2]
    peer_path = local/spec['independent_review_path']
    if sha256_file(peer_path) != spec['independent_review_sha256']:
        raise ValueError('Independent actual admission review changed')
    peer = json.loads(peer_path.read_text())
    if (peer['status'] != 'passed_full_native_four_class_saved_tensor_review_only' or peer['findings'] or
            peer['actual_report']['sha256'] != spec['files']['REPORT.json'] or
            peer['artifact_manifest']['sha256'] != spec['files']['ARTIFACT_HASHES.json'] or
            peer['protocol']['sha256'] != spec['files']['protocol.json']):
        raise ValueError('Independent review does not admit this actual result')
    report = json.loads((root/'REPORT.json').read_text())
    manifest = json.loads((root/'ARTIFACT_HASHES.json').read_text())['files']
    for name in ('REPORT.json', 'protocol.json'):
        if manifest[name]['sha256'] != spec['files'][name] or (root/name).stat().st_size != manifest[name]['bytes']:
            raise ValueError('Actual admission manifest join differs')
    parent = json.loads((root/'protocol.json').read_text())
    if (report['status'] != 'passed_full_native_dfine_four_class_supervised_admission_only' or
            report['optimizer_steps'] != 1 or len(report['backward_passes']) != 2 or
            report['final_integrity'] != 'passed' or report['CUDA_executed'] is not True or
            report['protocol_sha256'] != spec['files']['protocol.json'] or
            parent['launch_authorized'] is not True or
            parent['prepared_protocol_sha256'] != scientific['admission_prepared_sha256']):
        raise ValueError('Successful original native V3 admission required')
    for key in ('checkpoint_sha256', 'records_sha256', 'input_manifest_sha256', 'config',
                'seed', 'native_overrides', 'deterministic_transform_ops', 'clip_max_norm'):
        if parent[key] != scientific[key]:
            raise ValueError('Admission scientific identity differs: ' + key)
    if report['bound_files'] != parent['files'] or report['source_before'] != report['source_after']:
        raise ValueError('Admission source bindings differ')
    inventory = json.loads((Path(__file__).resolve().parents[2]/scientific['source_inventory_path']).read_text())
    if report['source_before'] != dict(commit=inventory['commit'], files=inventory['files']):
        raise ValueError('Actual author source identity differs')
    return dict(files=spec['files'], status=report['status'],
                initial_state=report['transfer']['loaded_state'], optimizer_steps=1)

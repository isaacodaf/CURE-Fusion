"""Pure-array data and metric contracts for the separate V3 2D fitting diagnosis."""
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CLASSES = ('person', 'bicycle', 'slidecar', 'doll')
CHANNELS = (0, 1, 80, 81)
FIELDS = ('camera', 'camera_logits', 'camera_boxes', 'radar', 'radar_mask',
          'radar_xy', 'radar_geometry_mask', 'camera_present')
BUDGETS = {1: 500, 8: 800, 32: 1200, 128: 2400}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bound(root, relative, descriptor):
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'],
            'Changed bound payload: ' + relative)
    return path


def learning_rate(update, updates):
    require(updates >= 2 and 0 <= update < updates, 'Invalid fixed cosine index')
    return .00001 + (.001-.00001) * .5 * (1+math.cos(math.pi*update/(updates-1)))


def batch_schedule(size, updates, seed=11):
    require(size in BUDGETS and updates == BUDGETS[size] and seed == 11, 'Fixed prefix recipe required')
    generator = np.random.default_rng(seed)
    batches = []
    while len(batches) < updates:
        order = generator.permutation(size)
        batches.extend(order[i:i+16].tolist() for i in range(0, size, 16))
    return batches[:updates]


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_cache(root, protocol):
    """Verify all128 cached carriers, including variable-width padded GT arrays."""
    root = Path(root)
    require(sha(root/'REPORT.json') == protocol['cache_report_sha256'], 'Cache report differs')
    require(sha(root/'ARTIFACT_HASHES.json') == protocol['cache_manifest_sha256'], 'Cache manifest differs')
    manifest = json.loads((root/'ARTIFACT_HASHES.json').read_text())
    for name in ('REPORT.json', 'protocol.json', 'selected_records.json'):
        bound(root, name, manifest[name])
    report = json.loads((root/'REPORT.json').read_text())
    require(report['status'] == 'passed_feature_cache_only' and report['mode'] == 'admission128'
            and report['images_completed'] == 128 and report['final_integrity'] == 'passed',
            'Complete frozen admission128 cache required')
    require(sha(root/'protocol.json') == report['protocol_sha256'] == protocol['cache_protocol_sha256'],
            'Cache protocol differs')
    cache_protocol = json.loads((root/'protocol.json').read_text())
    source = dict(commit=protocol['source_commit'], inventory_sha256=protocol['source_inventory_sha256'])
    require(report['source_before'] == report['source_after'] == source, 'Cache native source binding differs')
    require(cache_protocol['checkpoint_sha256'] == protocol['native_checkpoint_sha256'], 'Native checkpoint ID differs')
    require(cache_protocol['source_commit'] == source['commit'] and
            cache_protocol['source_inventory_sha256'] == source['inventory_sha256'], 'Cache source protocol differs')
    records = json.loads((root/'selected_records.json').read_text())
    require(len(records) == 128 and len({r['identity'] for r in records}) == 128
            and all(r['development_role'] == 'fit' for r in records), 'Exact128 fitting records required')
    require([r['identity'] for r in records] == protocol['sample_ids'], 'Prefix order differs')
    arrays = {key: [] for key in FIELDS}
    cursor = 0
    bindings = {str(root/name): sha(root/name) for name in
                ('REPORT.json', 'protocol.json', 'selected_records.json', 'ARTIFACT_HASHES.json')}
    for chunk in report['chunks']:
        require(manifest[chunk['path']]['sha256'] == chunk['sha256'] and
                manifest[chunk['path']]['bytes'] == chunk['bytes'], 'Chunk manifest differs')
        path = bound(root, chunk['path'], manifest[chunk['path']])
        bindings[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        require(set(values) == set(chunk['arrays']), 'Chunk array schema differs')
        for key, value in values.items():
            spec = chunk['arrays'][key]
            require(list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype'] and
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == spec['sha256'],
                    'Raw cache array differs: ' + key)
        count = len(values['identities'])
        subset = records[cursor:cursor+count]
        require(values['identities'].tolist() == chunk['identities'] == [r['identity'] for r in subset],
                'Cache frame alignment differs')
        require(values['radar_rows'].tolist() == [r['radar_row'] for r in subset], 'Radar row alignment differs')
        for i, record in enumerate(subset):
            n = len(record['target']['labels'])
            require(values['target_mask'][i].tolist() == [True]*n+[False]*(values['target_mask'].shape[1]-n),
                    'Target padding mask differs')
            require(np.array_equal(values['target_labels'][i, :n], record['target']['labels']) and
                    np.array_equal(values['target_boxes'][i, :n],
                                   np.asarray(record['target']['boxes'], np.float32).reshape(n, 4)),
                    'Original four-class targets differ')
            require((values['target_labels'][i, n:] == -1).all() and
                    (values['target_boxes'][i, n:] == 0).all(), 'Target padding contents differ')
        for key in arrays:
            arrays[key].append(values[key])
        cursor += count
    require(cursor == 128, 'Incomplete prefix cache')
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    require(result['camera_present'].all(), 'Original cache must have camera available')
    return records, result, bindings


def xyxy(boxes):
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    return np.concatenate((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), axis=1)


def pairwise_iou(a, b):
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])-
                              np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(-1)
    area_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)
    union = area_a[:, None]+area_b[None, :]-intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def ground_truth(records):
    images, annotations = [], []
    for i, record in enumerate(records):
        images.append(dict(id=i+1, width=640, height=512, file_name=record['identity']))
        labels = np.asarray(record['target']['labels'], np.int64)
        boxes = np.asarray(record['target']['boxes'], np.float32).reshape(-1, 4)
        require(((labels >= 0) & (labels < 4)).all() and np.isfinite(boxes).all()
                and (boxes[:, 2:] > 0).all(), 'Invalid original GT')
        for label, corners in zip(labels, xyxy(boxes)*[640, 512, 640, 512]):
            width, height = corners[2:]-corners[:2]
            annotations.append(dict(id=len(annotations)+1, image_id=i+1, category_id=int(label)+1,
                                    bbox=[*corners[:2].tolist(), float(width), float(height)],
                                    area=float(width*height), iscrowd=0))
    return dict(info=dict(description='Fixed fitting-only V3 2D diagnosis'), images=images,
                annotations=annotations, categories=[dict(id=i+1, name=name) for i, name in enumerate(CLASSES)])


def selected_records(records, labels, boxes, scores):
    """Keep source global-top300 order; validate task-selected raw WH before clipping."""
    labels, boxes, scores = np.asarray(labels), np.asarray(boxes), np.asarray(scores)
    require(labels.shape == scores.shape == (len(records), 300) and
            boxes.shape == (len(records), 300, 4), 'Selected output axes differ')
    require(np.issubdtype(labels.dtype, np.integer) and ((labels >= 0) & (labels < 82)).all(),
            'Invalid channel label')
    require(np.isfinite(boxes).all() and np.isfinite(scores).all() and
            ((scores >= 0) & (scores <= 1)).all(), 'Nonfinite/invalid selected output')
    predictions = []
    invalid, clipped_zero = 0, 0
    for i in range(len(records)):
        for channel, box, score in zip(labels[i], boxes[i], scores[i]):
            if int(channel) not in CHANNELS:
                continue
            valid = bool((box[2:] > box[:2]).all())
            invalid += int(not valid)
            clipped = np.clip(box, [0, 0, 0, 0], [640, 512, 640, 512])
            wh = clipped[2:]-clipped[:2]
            clipped_zero += int(valid and (wh == 0).any())
            # Preserve invalid records too; caller must not run COCO on them.
            predictions.append(dict(image_id=i+1, category_id=CHANNELS.index(int(channel))+1,
                                    bbox=[*clipped[:2].tolist(), *wh.tolist()], score=float(score)))
    return predictions, dict(invalid_raw_task_selected_boxes=invalid,
                             valid_raw_boxes_clipped_to_zero=clipped_zero)


def evaluate_mode(records, predictions, raw_finite, box_status):
    """Official COCO AP50/AP75 and independent IoU correspondence, with fixed gates."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth = ground_truth(records)
    support = [sum(a['category_id'] == c+1 for a in truth['annotations']) for c in range(4)]
    valid = bool(raw_finite and box_status['invalid_raw_task_selected_boxes'] == 0)
    per_class = {name: dict(GT=support[c], AP50=None, AP75=None, median_matched_IoU=None,
                           assigned_GT=0, missing_GT=support[c], gate=False if support[c] else None)
                 for c, name in enumerate(CLASSES)}
    matched_rows = []
    if valid:
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(); gt.dataset = truth; gt.createIndex()
            if predictions:
                dt = gt.loadRes(predictions)
            else:
                dt = COCO(); dt.dataset = dict(images=truth['images'], categories=truth['categories'], annotations=[]); dt.createIndex()
            evaluator = COCOeval(gt, dt, 'bbox')
            evaluator.params.imgIds = [x['id'] for x in truth['images']]
            evaluator.params.catIds = [i+1 for i, count in enumerate(support) if count]
            evaluator.params.iouThrs = np.array([.5, .75])
            if evaluator.params.catIds:
                evaluator.evaluate(); evaluator.accumulate()
        for index, category in enumerate(evaluator.params.catIds):
            row = per_class[CLASSES[category-1]]
            for threshold, name in enumerate(('AP50', 'AP75')):
                precision = evaluator.eval['precision'][threshold, :, index, 0, 2]
                row[name] = float(precision[precision >= 0].mean())
        for image in truth['images']:
            for category in range(1, 5):
                actual = [a for a in truth['annotations'] if a['image_id'] == image['id'] and a['category_id'] == category]
                proposed = [(i, a) for i, a in enumerate(predictions) if a['image_id'] == image['id'] and a['category_id'] == category]
                def corners(items):
                    b = np.asarray([a['bbox'] for a in items], np.float64).reshape(-1, 4)
                    return np.concatenate((b[:, :2], b[:, :2]+b[:, 2:]), axis=1)
                overlaps = pairwise_iou(corners(actual), corners([a for _, a in proposed]))
                gt_indices, pred_indices = linear_sum_assignment(-overlaps)
                assigned = dict(zip(gt_indices.tolist(), pred_indices.tolist()))
                for i, actual_box in enumerate(actual):
                    j = assigned.get(i)
                    matched_rows.append(dict(identity=image['file_name'], image_id=image['id'], category_id=category,
                                             annotation_id=actual_box['id'], prediction_index=None if j is None else proposed[j][0],
                                             iou=0. if j is None else float(overlaps[i, j])))
        for c, name in enumerate(CLASSES):
            if not support[c]:
                continue
            rows = [r for r in matched_rows if r['category_id'] == c+1]
            row = per_class[name]
            row['median_matched_IoU'] = float(np.median([r['iou'] for r in rows]))
            row['assigned_GT'] = sum(r['prediction_index'] is not None for r in rows)
            row['missing_GT'] = support[c]-row['assigned_GT']
            row['gate'] = bool(row['AP50'] >= .95 and row['AP75'] >= .90 and
                               row['median_matched_IoU'] >= .9 and row['missing_GT'] == 0)
    empty_ids = [image['id'] for image in truth['images'] if not any(a['image_id'] == image['id'] for a in truth['annotations'])]
    result = dict(status='evaluated_fixed_endpoint' if valid else 'unavailable_invalid_outputs',
                  classes=per_class, supported_classes=[CLASSES[i] for i, n in enumerate(support) if n],
                  finite_raw_outputs=bool(raw_finite), **box_status,
                  images=len(records), GT=sum(support), empty_frames=len(empty_ids),
                  empty_frame_FP_score_ge_0_5=sum(d['image_id'] in empty_ids and d['score'] >= .5 for d in predictions),
                  all_GT_assigned=bool(valid and len(matched_rows) == sum(support) and all(r['prediction_index'] is not None for r in matched_rows)),
                  gate=bool(valid and any(support) and all(row['gate'] for row in per_class.values() if row['GT'])))
    return result, matched_rows, truth

"""Hash-bound saved-output token preparation and read-only study inputs.

No detector forward, optimizer, AP computation or CUDA work. Production token
formation is CPU NumPy FP64 exp(-logaddexp(0,-logit)), cast once to FP32.
The core's Tensor.sigmoid helper is not an alternate production path here.
"""
import hashlib
import json
import platform
from pathlib import Path
import shutil
import time

import numpy as np

from .csu_action_queryset_v2 import FittingTokenMomentsV2

ROLES = ('controller_fit', 'inner_calibration', 'reused_development')
CONDITIONS = ('clean', 'camera_removed', 'radar_removed', 'radar_thinned_fresh')
ARMS = ('task_only', 'cure', 'modality_dropout')
TASK_CHANNELS = (0, 1, 80, 81)
TOKEN_ARITHMETIC = 'CPU NumPy FP64 exp(-logaddexp(0,-saved FP32 logits)); cast once FP32; raw boxes copied exactly'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def bounded_path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative artifact path required')
    root = Path(root).resolve()
    path = (root/relative).resolve()
    path.relative_to(root)
    return path


def check_file(path, entry):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
            'Artifact bytes differ: '+str(path))


def sigmoid64_v2(logits):
    require(isinstance(logits, np.ndarray) and logits.dtype == np.float32 and np.isfinite(logits).all(),
            'Finite original FP32 logits required')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        return np.exp(-np.logaddexp(np.float64(0), -logits.astype(np.float64)))


def query_tokens_numpy_v2(logits, boxes):
    require(logits.ndim == 3 and logits.shape[0] > 0 and logits.shape[1:] == (300, 82) and
            isinstance(boxes, np.ndarray) and boxes.dtype == np.float32 and
            boxes.shape == (*logits.shape[:2], 4) and np.isfinite(boxes).all(), 'Original query axes/raw boxes required')
    probability = sigmoid64_v2(logits)
    tokens = np.concatenate((probability.astype(np.float32), boxes), axis=-1)
    require(np.isfinite(tokens).all() and ((tokens[..., :82] >= 0) & (tokens[..., :82] <= 1)).all(),
            'Invalid prepared probability tokens')
    require(np.array_equal(tokens[..., 82:].view(np.uint32), boxes.view(np.uint32)), 'Raw box bytes changed')
    return tokens


def validate_chunk_v2(raw, ids, records, operands, start, *, development):
    """Complete canonical targets, risk carriers and retained native selections."""
    n = len(ids)
    require(n > 0 and raw['identities'].tolist() == ids and
            raw['sequences'].tolist() == [records[s]['sequence'] for s in ids], 'Chunk identities/sequences differ')
    labels, boxes, mask = raw['target_labels'], raw['target_boxes'], raw['target_mask']
    require(labels.dtype == np.int64 and labels.ndim == 2 and labels.shape[0] == n and labels.shape[1] >= 1 and
            boxes.dtype == np.float32 and boxes.shape == (*labels.shape, 4) and mask.dtype == np.bool_ and
            mask.shape == labels.shape and np.isfinite(boxes).all(), 'Canonical padded target axes differ')
    require((labels[~mask] == -1).all() and (boxes[~mask] == 0).all(), 'Target padding changed')
    for i, sid in enumerate(ids):
        target = records[sid]['target']
        lab = np.asarray(target['labels'], dtype=np.int64)
        box = np.asarray(target['boxes'], dtype=np.float32).reshape(-1, 4)
        require(np.array_equal(mask[i], np.arange(mask.shape[1]) < len(lab)) and
                np.array_equal(labels[i, mask[i]], lab) and np.array_equal(boxes[i, mask[i]], box),
                'Original all-four GT carrier differs: '+sid)
        require(((lab >= 0) & (lab < 4)).all() and (box[:, 2:] > 0).all(), 'Invalid canonical task GT')
    for ci, condition in enumerate(CONDITIONS):
        for name in ('gain', 'base_risk', 'candidate_risk'):
            value = raw[condition+'_'+name]
            require(value.dtype == np.float32 and value.shape == (n,) and np.isfinite(value).all() and
                    np.array_equal(value.view(np.uint32), operands[name][start:start+n, ci].view(np.uint32)),
                    'Original risk operand differs: '+name)
        require(np.array_equal(raw[condition+'_gain'], raw[condition+'_base_risk']-raw[condition+'_candidate_risk']),
                'Original signed risk difference differs')
        require((raw[condition+'_base_risk'] >= 0).all() and (raw[condition+'_candidate_risk'] >= 0).all(),
                'Negative original frame risk')
        for arm in ARMS if development else ARMS[:2]:
            prefix = condition+'_'+arm+'_'
            x, b = raw[prefix+'pred_logits'], raw[prefix+'pred_boxes']
            require(x.dtype == b.dtype == np.float32 and x.shape == (n, 300, 82) and
                    b.shape == (n, 300, 4) and np.isfinite(x).all() and np.isfinite(b).all(), 'Raw output contract differs')
            lab, box, score = raw[prefix+'labels'], raw[prefix+'boxes'], raw[prefix+'scores']
            require(lab.dtype == np.int64 and lab.shape == (n, 300) and
                    box.dtype == score.dtype == np.float32 and box.shape == (n, 300, 4) and score.shape == (n, 300) and
                    np.isfinite(box).all() and np.isfinite(score).all() and
                    ((lab >= -1) & (lab <= 3)).all() and ((score >= 0) & (score <= 1)).all(), 'Saved native selected fields differ')
            selected = lab >= 0
            require(((box[..., 2:]-box[..., :2]) > 0).all(-1)[selected].all(),
                    'Original selected-positive gate fails; no filtering or repair')


def canonical_gt_v2(records, ids):
    """Literal source build_gt arithmetic; no metric is calculated."""
    annotations, images = [], []
    for i, sid in enumerate(ids, 1):
        images.append(dict(id=i, width=640, height=512, file_name=sid))
        target = records[sid]['target']
        for label, box in zip(target['labels'], target['boxes']):
            cx, cy, w, h = np.asarray(box, dtype=float)*[640, 512, 640, 512]
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=label+1,
                                    bbox=[cx-w/2, cy-h/2, w, h], area=w*h, iscrowd=0))
    return dict(info={}, images=images, annotations=annotations,
                categories=[dict(id=i+1, name=n) for i, n in enumerate(('person','bicycle','slidecar','doll'))])


def prepare_queryset_inputs_v2(original, records_path, specification, expected_spec_sha256, output):
    """One complete hash-verified CPU preparation; failed partial files remain."""
    original, output, specification = Path(original), Path(output), Path(specification)
    require(not output.exists(), 'New output directory required')
    require(sha(specification) == expected_spec_sha256, 'Token preparation specification differs')
    spec = json.loads(specification.read_text())
    require(spec['status'] == 'declared_CPU_token_preparation_only' and spec['learned_forward'] is False and
            spec['token_arithmetic'] == TOKEN_ARITHMETIC, 'Wrong preparation scope/arithmetic')
    require(np.__version__ == spec['production_numpy_version'], 'Declared CPU NumPy version differs')
    repo = Path(__file__).resolve().parents[2]
    for path, digest in spec['files'].items(): require(sha(repo/path) == digest, 'Prepared source changed: '+path)
    check_file(records_path, spec['canonical_records'])
    require(spec['original_manifest_filename'] == 'MANIFEST.json', 'Original action manifest filename differs')
    check_file(original/'MANIFEST.json', spec['original_manifest'])
    manifest = json.loads((original/'MANIFEST.json').read_text())
    sources = {}
    def source(path):
        require(path in manifest, 'Unbound original input: '+path)
        file = bounded_path(original, path)
        check_file(file, manifest[path]); sources[path] = dict(manifest[path])
        return file
    report = json.loads(source('REPORT.json').read_text())
    require(sha(original/'REPORT.json') == spec['original_report_sha256'] and
            report['status'] == 'completed_reused_SEW_action_study' and report['source_integrity'] == 'passed' and
            report['protocol_sha256'] == spec['original_protocol_sha256'], 'Original completed study binding differs')
    rawrecords = json.loads(Path(records_path).read_text())
    records = {r['identity']: r for r in rawrecords}
    require(len(records) == len(rawrecords) == 7172, 'Complete canonical metadata required')
    roles = json.loads((repo/spec['roles_path']).read_text())
    all_ids = [s for role in ROLES for s in roles[role]['sample_ids']]
    require(len(all_ids) == len(set(all_ids)) == 7172 and set(all_ids) == set(records), 'Role partition differs')
    captured = json.loads(source('capture/CAPTURE.json').read_text())
    inventory = json.loads((repo/spec['capture_inventory_path']).read_text())
    expected_chunks = {r['path']: r for r in inventory['records']}
    require(len(expected_chunks) == 449, 'Complete449chunk inventory required')
    with np.load(source('NORMALIZATION.npz'), allow_pickle=False) as z:
        gain_scale = z['gain_scale'].copy()
    require(gain_scale.dtype == np.float32 and gain_scale.shape == () and np.isfinite(gain_scale) and gain_scale > 0,
            'Exact original positive fitting gain RMS required')
    original_gt = json.loads(source('GROUND_TRUTH_COCO.json').read_text())
    require(canonical_gt_v2(records, roles['reused_development']['sample_ids']) == original_gt,
            'Original official GT serializer identity differs')
    output.mkdir(parents=True)
    started = time.time()
    status = dict(status='preparing_csu_queryset_saved_tokens', learned_forward=False, AP_evaluated=False,
                  CUDA_used=False, token_arithmetic=TOKEN_ARITHMETIC, specification_sha256=expected_spec_sha256)
    write(output/'REPORT.json', status)
    processed = []
    try:
        shutil.copyfile(specification, output/'TOKEN_PREPARATION_SPEC.json')
        shutil.copyfile(repo/spec['roles_path'], output/'ROLES.json')
        shutil.copyfile(records_path, output/'RECORDS.json')
        shutil.copyfile(original/'GROUND_TRUTH_COCO.json', output/'GROUND_TRUTH_COCO.json')
        moments = FittingTokenMomentsV2(roles['controller_fit']['sample_ids'])
        selected, confidence = {}, np.empty((992,4,2), np.float64)
        for role in ROLES:
            ids = roles[role]['sample_ids']; n = len(ids)
            operand_file = source('capture/'+role+'_operands.npz')
            with np.load(operand_file, allow_pickle=False) as z:
                operands = {k:z[k].copy() for k in ('gain','base_risk','candidate_risk','identities','sequences')}
            require(operands['identities'].tolist() == ids and
                    operands['sequences'].tolist() == [records[s]['sequence'] for s in ids], 'Original operand identities differ')
            for name in ('gain','base_risk','candidate_risk'):
                require(operands[name].dtype == np.float32 and operands[name].shape == (n,4) and
                        np.isfinite(operands[name]).all(), 'Original full risk carrier differs')
            np.savez_compressed(output/(role+'_operands.npz'), **operands)
            token_map = np.lib.format.open_memmap(output/(role+'_tokens.npy'), mode='w+', dtype=np.float32,
                                                  shape=(n,4,2,300,86))
            offset = 0
            for chunk in captured[role]['chunks']:
                path = 'capture/'+chunk['path']; entry = expected_chunks[path]
                require(entry['role'] == role and entry['identities'] == chunk['identities'] and
                        entry['sha256'] == chunk['sha256'] == manifest[path]['sha256'] and
                        entry['bytes'] == manifest[path]['bytes'], 'External449chunk binding differs')
                chunk_ids = chunk['identities']; size = len(chunk_ids)
                require(chunk_ids == ids[offset:offset+size], 'Complete original frame order differs')
                with np.load(source(path), allow_pickle=False) as compressed:
                    # NPZ arrays are read individually; no detector or pretrained weights enter this path.
                    z = {key:compressed[key] for key in compressed.files}
                    validate_chunk_v2(z, chunk_ids, records, operands, offset, development=role == 'reused_development')
                    tokens = np.empty((size,4,2,300,86), np.float32)
                    for ci, condition in enumerate(CONDITIONS):
                        for ai, arm in enumerate(ARMS[:2]):
                            prefix = condition+'_'+arm+'_'
                            x, b = z[prefix+'pred_logits'], z[prefix+'pred_boxes']
                            tokens[:,ci,ai] = query_tokens_numpy_v2(x,b)
                            if role == 'reused_development':
                                # Literal original FP64 confidence, before any token FP32 cast.
                                confidence[offset:offset+size,ci,ai] = sigmoid64_v2(x[:,:,TASK_CHANNELS]).max(2).mean(1)
                        if role == 'reused_development':
                            for arm in ARMS:
                                for key in ('labels','boxes','scores'):
                                    name=condition+'_'+arm+'_'+key
                                    selected.setdefault(name,[]).append(z[name].copy())
                    token_map[offset:offset+size] = tokens
                    if role == 'controller_fit': moments.update(chunk_ids, tokens, role=role)
                processed.append(path); offset += size
                token_map.flush()
                write(output/'PROGRESS.json', dict(role=role, role_frames=offset, processed_chunks=len(processed)))
                if len(processed)%16==0:
                    print(json.dumps(dict(stage='CPU_saved_token_preparation',role=role,frames=offset,chunks=len(processed))),flush=True)
            require(offset == n, 'Incomplete role token preparation')
            del token_map
        require(len(processed) == len(set(processed)) == 449 and set(processed) == set(expected_chunks),
                'Missing or repeated original raw capture chunks')
        normalization = moments.finish()
        np.savez_compressed(output/'NORMALIZATION_V2.npz', center=normalization.pop('center'),
                            scale=normalization.pop('scale'), gain_scale=gain_scale)
        normalization['gain_scale_scope'] = 'Original NORMALIZATION.npz gain_scale copied byte-exact; not recalculated'
        normalization['original_normalization'] = manifest['NORMALIZATION.npz']
        write(output/'TOKEN_MOMENTS_V2.json', normalization)
        selected = {k:np.concatenate(v,axis=0) for k,v in selected.items()}
        selected['confidence'] = confidence
        np.savez_compressed(output/'DEVELOPMENT_SELECTIONS.npz', **selected)
        for path, entry in sources.items(): check_file(bounded_path(original,path),entry)
        check_file(original/'MANIFEST.json', spec['original_manifest'])
        check_file(records_path, spec['canonical_records'])
        require(sha(specification) == expected_spec_sha256, 'Specification changed during preparation')
        for path, digest in spec['files'].items(): require(sha(repo/path) == digest, 'Prepared source changed during run')
        write(output/'ORIGINAL_INPUT_BINDINGS.json', dict(files=sources, original_manifest=spec['original_manifest'],
                    canonical_records=spec['canonical_records'], source_root=str(original.resolve())))
        status.update(status='passed_csu_queryset_token_preparation_only', complete_frames=7172, chunks=449,
                      roles={r:len(roles[r]['sample_ids']) for r in ROLES}, source_integrity='passed',
                      numpy_version=np.__version__, seconds=time.time()-started,
                      python_version=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
                      feature_scope='Defined CPU64-to-FP32 representation; no CUDA-sigmoid bit-identity claim',
                      complete_targets_verified=True, native_selected_arrays_copied=True, original_risks_copied=True)
        write(output/'REPORT.json', status)
        payloads = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}
        write(output/'MANIFEST.json', payloads)
        return dict(report=status,manifest_sha256=sha(output/'MANIFEST.json'))
    except BaseException as error:
        status.update(status='failed_csu_queryset_token_preparation_only', error=repr(error),
                      processed_chunks=len(processed), seconds=time.time()-started)
        write(output/'REPORT.json',status)
        raise


class QuerySetCacheV2:
    """Read-only NumPy mmap cache; no network, fitting, sigmoid or AP calls."""
    def __init__(self, root, expected_manifest_sha256):
        self.root = Path(root)
        require(sha(self.root/'MANIFEST.json') == expected_manifest_sha256, 'Prepared cache manifest differs')
        self.manifest_sha256 = expected_manifest_sha256
        self.manifest = json.loads((self.root/'MANIFEST.json').read_text())
        required_files = {'REPORT.json','TOKEN_PREPARATION_SPEC.json','ROLES.json','RECORDS.json',
                          'GROUND_TRUTH_COCO.json','NORMALIZATION_V2.npz','TOKEN_MOMENTS_V2.json',
                          'DEVELOPMENT_SELECTIONS.npz','ORIGINAL_INPUT_BINDINGS.json'}
        required_files.update(role+suffix for role in ROLES for suffix in ('_tokens.npy','_operands.npz'))
        require(required_files <= set(self.manifest), 'Prepared manifest omits a required cache payload')
        for path, entry in self.manifest.items(): check_file(bounded_path(self.root,path),entry)
        self.report = json.loads((self.root/'REPORT.json').read_text())
        require(self.report['status'] == 'passed_csu_queryset_token_preparation_only' and
                self.report['complete_frames'] == 7172 and self.report['chunks'] == 449 and
                self.report['source_integrity'] == 'passed' and self.report['token_arithmetic'] == TOKEN_ARITHMETIC,
                'Complete declared token preparation required')
        require(sha(self.root/'TOKEN_PREPARATION_SPEC.json') == self.report['specification_sha256'],
                'Copied token preparation specification differs')
        self.preparation_spec = json.loads((self.root/'TOKEN_PREPARATION_SPEC.json').read_text())
        require(sha(self.root/'ROLES.json') == self.preparation_spec['files'][self.preparation_spec['roles_path']],
                'Copied canonical role partition differs')
        check_file(self.root/'RECORDS.json',self.preparation_spec['canonical_records'])
        self.roles = json.loads((self.root/'ROLES.json').read_text())
        record_list = json.loads((self.root/'RECORDS.json').read_text())
        self.records = {r['identity']:r for r in record_list}
        ids = [s for r in ROLES for s in self.roles[r]['sample_ids']]
        require(len(ids) == len(set(ids)) == len(self.records) == len(record_list) == 7172 and set(ids) == set(self.records),
                'Complete prepared records and role partition differ')
        self.tokens, self._operands = {}, {}
        for role in ROLES:
            n = len(self.roles[role]['sample_ids'])
            self.tokens[role] = np.load(self.root/(role+'_tokens.npy'),mmap_mode='r',allow_pickle=False)
            require(self.tokens[role].dtype == np.float32 and self.tokens[role].shape == (n,4,2,300,86), 'Prepared mmap axes differ')
            with np.load(self.root/(role+'_operands.npz'),allow_pickle=False) as z:
                self._operands[role] = {k:z[k].copy() for k in z.files}
            require(self._operands[role]['identities'].tolist() == self.roles[role]['sample_ids'] and
                    self._operands[role]['sequences'].tolist() == [self.records[s]['sequence'] for s in self.roles[role]['sample_ids']],
                    'Prepared operand identity differs')
            for name in ('gain','base_risk','candidate_risk'):
                value = self._operands[role][name]
                require(value.shape == (n,4) and value.dtype == np.float32 and np.isfinite(value).all(),
                        'Prepared risk axes or dtype differ')
            for value in self._operands[role].values(): value.flags.writeable = False
        with np.load(self.root/'NORMALIZATION_V2.npz',allow_pickle=False) as z:
            self.normalization = {k:z[k].copy() for k in z.files}
        require(self.normalization['center'].shape == self.normalization['scale'].shape == (86,) and
                self.normalization['gain_scale'].shape == () and
                all(a.dtype == np.float32 and np.isfinite(a).all() for a in self.normalization.values()) and
                (self.normalization['scale'] > 0).all() and self.normalization['gain_scale'] > 0, 'Prepared normalization differs')
        with np.load(self.root/'DEVELOPMENT_SELECTIONS.npz',allow_pickle=False) as z:
            self._selected_arrays = {k:z[k].copy() for k in z.files}
        self.confidence = self._selected_arrays['confidence']
        require(self.confidence.shape == (992,4,2) and self.confidence.dtype == np.float64 and
                np.isfinite(self.confidence).all(), 'Original FP64 confidence carrier differs')
        for value in self.normalization.values(): value.flags.writeable = False
        for value in self._selected_arrays.values(): value.flags.writeable = False
        self.selected = {c:{a:list(zip(*(self._selected_arrays[c+'_'+a+'_'+k] for k in ('labels','boxes','scores'))))
                            for a in ARMS} for c in CONDITIONS}
        self.ground_truth = json.loads((self.root/'GROUND_TRUTH_COCO.json').read_text())
        require(canonical_gt_v2(self.records,self.roles['reused_development']['sample_ids']) == self.ground_truth,
                'Prepared canonical evaluator GT differs')

    def operands(self, role):
        require(role in ROLES, 'Unknown role')
        return self._operands[role]

    def batch(self, role, indices):
        require(role in ROLES, 'Unknown role')
        indices = np.asarray(indices)
        require(indices.ndim == 1 and len(indices) > 0 and np.issubdtype(indices.dtype,np.integer) and
                (indices >= 0).all() and (indices < len(self.tokens[role])).all(), 'Valid nonempty frame indices required')
        return np.ascontiguousarray(self.tokens[role][indices])

    def verify_unchanged(self):
        require(sha(self.root/'MANIFEST.json') == self.manifest_sha256, 'Prepared manifest changed')
        for path, entry in self.manifest.items(): check_file(bounded_path(self.root,path),entry)

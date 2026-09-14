"""One post-hoc original-CSU trained readout on frozen Task-only features.

Admission: original epoch-one first64 fitting frames and one head-only update. Study: all6180
fitting frames,40epochs, then all992 reused development frames. No AP or new
interventions. Capture all300 factual512-vectors before any GT gathering.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import torch
from cure_fusion.dfine_cure_v1 import DfineCureConfigV1, require
from cure_fusion.dfine_cure_v2 import DfineNativeCureV2
from cure_fusion.dfine_cure_losses_v2 import match_four_class_v2
from cure_fusion.dfine_cure_development_v2 import Cache, sha, write, state_digest, epoch_order
from cure_fusion.csu_sensor_head_control_v1 import (
    FactualUtilityInputWitnessV1, fresh_original_head_v1, original_head_state_v1,
    ordered_query_indices_v1, utility_batch_loss_v1, learning_rate_v1, descriptive_metrics_v1)


# Reuse the exact admitted optimization implementation without copying or editing it.
import importlib.util
_spec = importlib.util.spec_from_file_location('original_csu_head_control_v1',
    ROOT / 'code/scripts/run_csu_sensor_head_control_cuda_v1.py')
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
train = _original.train
check_bound = _original.check_bound
load_moments = _original.load_moments
from cure_fusion.csu_task_feature_comparison_v2 import align_cure_readouts_v2

def capture_task(model, cache, ids, moments, out):
    out.mkdir(exist_ok=False)
    bank = np.lib.format.open_memmap(out / 'factual_all_query_features.npy', mode='w+',
                                    dtype=np.float32, shape=(len(ids), 300, 512))
    queries = np.full(moments['object_mask'].shape, -1, np.int64)
    frozen_raw = np.zeros((*queries.shape, 4), np.float32)
    before = state_digest(model)
    rows = []
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            sids = ids[start:start+16]
            x, targets = cache.batch(sids)
            require(bool(x['camera_present'].all()), 'Only factual camera-present capture is declared')
            with FactualUtilityInputWitnessV1(model) as witness:
                output = model(**x)
            features = witness.features()
            require(features.dtype == torch.float32 and features.is_cuda, 'FP32 CUDA representation required')
            # Save every raw factual query first. The matcher only determines object joins.
            bank[start:start+len(sids)] = features.cpu().numpy()
            raw = {k: output[k].detach().cpu().numpy() for k in
                   ('pred_logits', 'pred_boxes', 'cev_mean', 'cev_logvar')}
            raw['identities'] = np.asarray(sids)
            file = out / ('raw_%06d.npz' % start)
            np.savez_compressed(file, **raw)
            assignment = match_four_class_v2(output, targets)
            for i, (sid, (q, o)) in enumerate(zip(sids, assignment)):
                n = len(targets[i]['labels'])
                current = ordered_query_indices_v1(q.cpu().numpy(), o.cpu().numpy(), n)
                chosen = current
                queries[start+i, :n] = chosen
                frozen_raw[start+i, :n] = np.concatenate((raw['cev_mean'][i, chosen], raw['cev_logvar'][i, chosen]), -1)
                rows.append(dict(identity=sid, sequence=cache.records[sid]['sequence'],
                                 labels=cache.records[sid]['target']['labels'],
                                 object_order_queries=chosen.tolist(),
                                 recaptured_assignment_queries=current.tolist(),
                                 matcher_endpoint='task_only_epoch_040'))
            print(json.dumps(dict(stage='factual_feature_capture', role=out.name, completed=start+len(sids), total=len(ids))), flush=True)
    bank.flush()
    del bank
    joined = dict(queries=queries, frozen_task_raw=frozen_raw, **moments)
    np.savez_compressed(out / 'object_joins_and_targets.npz', **joined)
    write(out / 'records.json', rows)
    require(state_digest(model) == before and all(p.grad is None for p in model.parameters()),
            'Factual capture mutated frozen model or gradients')
    report = dict(status='passed_Task_factual_representation_capture_only', frames=len(ids),
                  objects=int(moments['object_mask'].sum()), empty_frames=int((~moments['object_mask'].any(1)).sum()),
                  representation_shape=[len(ids), 300, 512], source_state_before=before,
                  source_state_after=state_digest(model), all_queries_captured_before_object_gather=True,
                  new_interventions=False, AP=False, own_Task_factual_matching_in_both_roles=True,
                  changed_recaptured_matching_frames=sum(r['object_order_queries'] != r['recaptured_assignment_queries'] for r in rows))
    write(out / 'REPORT.json', report)
    return np.load(out / 'factual_all_query_features.npy', mmap_mode='r'), joined, rows


def assess_task(head, features, joined, records, prior_root, out):
    out.mkdir(exist_ok=False)
    head.eval()
    raw = np.zeros((*joined['queries'].shape, 4), np.float32)
    with torch.no_grad():
        for start in range(0, len(records), 16):
            q = joined['queries'][start:start+16]
            data = np.asarray(features[start:start+len(q)])
            x = data[np.arange(len(q))[:, None], np.maximum(q, 0)].copy()
            raw[start:start+len(q)] = head(torch.from_numpy(x).to('cuda')).cpu().numpy()
    prior_records = json.loads((prior_root / 'original/assessment/records.json').read_text())
    with np.load(prior_root / 'original/assessment/ALL_OBJECT_OPERANDS.npz', allow_pickle=False) as z:
        prior = {k: z[k] for k in z.files}
    aligned = align_cure_readouts_v2(records, joined, prior_records, prior)
    operands = dict(control_raw=raw, **joined, **aligned)
    np.savez_compressed(out / 'ALL_OBJECT_OPERANDS.npz', **operands)
    write(out / 'records.json', records)
    mask = joined['object_mask']
    names = [('fresh_head_on_Task_features', 'control_raw'),
             ('fresh_head_on_CURE_features', 'trained_CURE_feature_raw'),
             ('original_saved_CURE_head', 'original_saved_CURE_raw'),
             ('Task_head_without_utility_supervision', 'frozen_task_raw')]
    metrics = {name: descriptive_metrics_v1(operands[key][mask], joined['mean'][mask],
                                           joined['mean_sampling_variance'][mask]) for name, key in names}
    write(out / 'REPORT.json', dict(status='completed_Task_feature_object_identity_assessment_only',
          metrics=metrics, frames=len(records), objects=int(mask.sum()),
          comparison_join=['identity', 'original_annotation_object_index'],
          Task_queries_selected_by_own_factual_matcher=True,
          comparison_queries_never_select_Task_features=True, AP=False, confidence_intervals=False,
          scope='One post-hoc head seed on frozen Task-only features. Same original teacher-conditioned physical targets and reused development objects. Prior CURE-feature head uses its original saved matches. This representation-plus-own-query comparison does not isolate encoder changes from query correspondence, establish independent-teacher transfer, or establish detection/routing benefit.'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'cache', 'original_study', 'output'):
        ap.add_argument('--'+name.replace('_', '-'), type=pathlib.Path, required=True)
    ap.add_argument('--prior-control', type=pathlib.Path)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--admission-report', type=pathlib.Path)
    ap.add_argument('--admission-report-sha256')
    ap.add_argument('--admission-review', type=pathlib.Path)
    ap.add_argument('--admission-review-sha256')
    a = ap.parse_args()
    require(sha(a.protocol) == a.protocol_sha256, 'Protocol hash differs')
    p = json.loads(a.protocol.read_text())
    require(p['launch_authorized'] is True and p['status'] == 'frozen_for_declared_execution',
            'Prepared-only protocol cannot launch')
    require(p['mode'] in ('admission', 'study'), 'Unknown declared mode')
    for rel, digest in p['files'].items():
        require(sha(ROOT / rel) == digest, 'Frozen source binding differs: '+rel)
    check_bound(a.original_study, p['original_study_files'])
    admission = p['mode'] == 'admission'
    if not admission:
        require(a.prior_control is not None, 'Reviewed prior CURE-feature control required')
        check_bound(a.prior_control, p['prior_control_files'])
        prior_review = json.loads((ROOT / p['prior_control_review_path']).read_text())
        require(prior_review['status'] == 'passed_original_CSU_head_control_study_bounded_saved_review_only'
                and prior_review['findings'] == []
                and prior_review['report_sha256'] == p['prior_control_files']['original/REPORT.json']['sha256']
                and prior_review['companion_manifest_sha256'] == p['prior_control_files']['ARTIFACT_HASHES.json']['sha256'],
                'Prior comparison evidence is not covered by the independent review')
        require(all(v is not None for v in (a.admission_report, a.admission_report_sha256,
                                            a.admission_review, a.admission_review_sha256)),
                'Actual independently reviewed Task-feature single-step admission required')
        require(sha(a.admission_report) == a.admission_report_sha256 and sha(a.admission_review) ==
                a.admission_review_sha256, 'Actual admission/review external hash differs')
        actual = json.loads(a.admission_report.read_text())
        peer = json.loads(a.admission_review.read_text())
        require(actual['status'] == 'passed_Task_feature_head_control_admission_only' and
                actual['prepared_protocol_sha256'] == p['prepared_admission_protocol_sha256'] and
                actual['optimizer_updates'] == 1 and actual['final_integrity'] == 'passed',
                'Different or incomplete actual Task-feature head admission')
        require(peer['status'] == 'passed_Task_feature_head_control_actual_review_only' and
                peer['report_sha256'] == a.admission_report_sha256 and peer['findings'] == [],
                'Actual admission independent review does not cover this report')
    require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0),
            'Actual A100 required; no CPU/MPS learned execution')
    torch.manual_seed(11)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', mode=p['mode'], CUDA=True, AP=False, new_interventions=False,
                  protocol_sha256=sha(a.protocol), prepared_protocol_sha256=p['prepared_protocol_sha256'],
                  bound_files=p['files'], original_study_files=p['original_study_files'],
                  prior_control_files=p['prior_control_files'],
                  environment=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)), optimizer_updates=0)
    started = time.time()
    try:
        original = json.loads((ROOT / p['original_protocol_path']).read_text())
        splits = json.loads((ROOT / original['splits_path']).read_text())
        cache = Cache(a.cache, original['cache_protocol_sha256'], splits,
                      original['optimization_records_canonical_sha256'], p['cache_report_sha256'])
        model = DfineNativeCureV2(DfineCureConfigV1()).cuda().eval().requires_grad_(False)
        saved = torch.load(a.original_study / 'task_only/epoch_040/checkpoint.pt', map_location='cpu', weights_only=True)
        require(saved['epoch'] == 40 and saved['arm'] == 'task_only', 'Fixed Task-only endpoint required')
        model.load_state_dict(saved['model'], strict=True)
        model_before = state_digest(model)
        initial = torch.load(a.original_study / 'INITIAL.pt', map_location='cpu', weights_only=True)
        head = fresh_original_head_v1(model, initial)
        require(all(torch.equal(head.state_dict()[k].cpu(), v) for k, v in original_head_state_v1(initial).items()),
                'Fresh original head initialization differs')
        full_fit = splits['fit']['sample_ids']
        fit = [full_fit[int(i)] for i in epoch_order(len(full_fit), 11, 1)[:64]] if admission else full_fit
        if admission:
            require(fit == p['admission_sample_ids'], 'Original first training batch membership differs')
        moments = load_moments(a.original_study, 'fit', fit, cache)
        f, j, r = capture_task(model, cache, fit, moments, a.output / 'fit_capture')
        require(set(v for row in r for v in row['labels']) == {0, 1, 2, 3} and
                any(not row['labels'] for row in r), 'Actual four-class and empty-frame admission population required')
        train(head, f, j, fit, a.output / 'control_fit', admission)
        report['optimizer_updates'] = 1 if admission else 3880
        if not admission:
            write(a.output / 'ALL_TRAINING_COMPLETED.json', dict(updates=3880, epochs=40,
                  final_head_state_sha256=state_digest(head), before_development_capture_or_assessment=True))
            dev = splits['inner_development']['sample_ids']
            moments = load_moments(a.original_study, 'development', dev, cache)
            f, j, r = capture_task(model, cache, dev, moments, a.output / 'development_capture')
            assess_task(head, f, j, r, a.prior_control, a.output / 'assessment')
        require(state_digest(model) == model_before and all(v.grad is None for v in model.parameters()),
                'Frozen final Task-only model changed during separate head training')
        report.update(status='passed_Task_feature_head_control_admission_only' if admission else
                      'completed_Task_feature_original_CSU_head_control',
                      frozen_Task_state_before=model_before, frozen_Task_state_after=state_digest(model),
                      parameters=66436, head_only=True, own_Task_factual_matching_in_both_roles=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            for rel, digest in p['files'].items():
                require(sha(ROOT / rel) == digest, 'Bound source changed: '+rel)
            check_bound(a.original_study, p['original_study_files'])
            if not admission:
                check_bound(a.prior_control, p['prior_control_files'])
                require(sha(a.admission_report) == a.admission_report_sha256 and
                        sha(a.admission_review) == a.admission_review_sha256, 'Admission evidence changed')
            require(sha(a.protocol) == a.protocol_sha256, 'Protocol changed')
            if 'cache' in locals():
                cache.verify_unchanged()
            report['final_integrity'] = 'passed'
        except BaseException as exc:
            report.update(status='failed_integrity', final_integrity='failed', integrity_error=repr(exc))
        report['seconds'] = time.time()-started
        write(a.output / 'REPORT.json', report)
        write(a.output / 'ARTIFACT_HASHES.json', {str(f.relative_to(a.output)):dict(bytes=f.stat().st_size, sha256=sha(f))
              for f in sorted(a.output.rglob('*')) if f.is_file()})
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])), flush=True)
    return 0 if report['status'] in ('passed_Task_feature_head_control_admission_only',
                                   'completed_Task_feature_original_CSU_head_control') else 1


if __name__ == '__main__':
    raise SystemExit(main())

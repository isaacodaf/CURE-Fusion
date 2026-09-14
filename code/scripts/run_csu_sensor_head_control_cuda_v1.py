"""One original-CSU trained-head control on a frozen factual representation.

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


def arrays_binding(arrays):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype),
                    sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
            for k, v in arrays.items()}


def tensor_arrays(state):
    return {k: v.detach().cpu().numpy().copy() for k, v in state.items()}


def check_bound(root, entries):
    for rel, item in entries.items():
        path = (root / rel).resolve()
        path.relative_to(root.resolve())
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Bound input differs: ' + rel)


def load_moments(study, role, ids, cache):
    folder = 'fit_targets' if role == 'fit' else 'fresh_development_targets'
    with np.load(study / folder / 'moments.npz', allow_pickle=False) as z:
        positions = {s: i for i, s in enumerate(z['identities'].tolist())}
        require(len(positions) == len(z['identities']), 'Duplicate moment identities')
        index = [positions[s] for s in ids]
        data = {k: z[k][index].copy() for k in ('mean', 'mean_sampling_variance', 'object_mask')}
    for i, sid in enumerate(ids):
        n = len(cache.records[sid]['target']['labels'])
        require(np.array_equal(data['object_mask'][i], np.arange(data['object_mask'].shape[1]) < n),
                'Original object carrier differs')
    require(data['mean'].shape == data['mean_sampling_variance'].shape ==
            (*data['object_mask'].shape, 2), 'Original moment dimensions')
    require(np.isfinite(data['mean']).all() and np.isfinite(data['mean_sampling_variance']).all()
            and (data['mean_sampling_variance'] >= 0).all()
            and np.count_nonzero(data['mean_sampling_variance'][..., 0]) == 0,
            'Physical moments or deterministic-camera variance differ')
    return data


def capture(model, cache, ids, moments, out, historical=None):
    out.mkdir(exist_ok=False)
    bank = np.lib.format.open_memmap(out / 'factual_all_query_features.npy', mode='w+',
                                    dtype=np.float32, shape=(len(ids), 300, 512))
    queries = np.full(moments['object_mask'].shape, -1, np.int64)
    frozen_raw = np.zeros((*queries.shape, 4), np.float32)
    historical_raw = np.zeros_like(frozen_raw) if historical is not None else None
    if historical is not None:
        groups = {s: [] for s in ids}
        for row in historical:
            require(row['identity'] in groups, 'Historical observation outside development role')
            groups[row['identity']].append(row)
        require(len(historical) == int(moments['object_mask'].sum()), 'Complete historical object coverage')
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
                if historical is not None:
                    observed = sorted(groups[sid], key=lambda v: v['object_index'])
                    require([r['object_index'] for r in observed] == list(range(n)),
                            'Historical observations omit/duplicate an object')
                    chosen = ordered_query_indices_v1(np.array([r['query'] for r in observed], np.int64),
                                                      np.arange(n), n)
                    # No reassignment to improve control fit: the historical join is authoritative.
                    for oi, row in enumerate(observed):
                        require(row['sequence'] == cache.records[sid]['sequence'] and
                                np.array_equal(np.asarray(row['reference_mean'], np.float32), moments['mean'][start+i, oi])
                                and np.array_equal(np.asarray(row['reference_mean_sampling_variance'], np.float32),
                                                   moments['mean_sampling_variance'][start+i, oi]),
                                'Historical physical target/sequence join differs')
                        historical_raw[start+i, oi] = row['mean'] + row['logvar']
                else:
                    chosen = current
                queries[start+i, :n] = chosen
                frozen_raw[start+i, :n] = np.concatenate((raw['cev_mean'][i, chosen], raw['cev_logvar'][i, chosen]), -1)
                rows.append(dict(identity=sid, sequence=cache.records[sid]['sequence'],
                                 labels=cache.records[sid]['target']['labels'],
                                 object_order_queries=chosen.tolist(),
                                 recaptured_assignment_queries=current.tolist(),
                                 historical_join_retained=historical is not None))
            print(json.dumps(dict(stage='factual_feature_capture', role=out.name, completed=start+len(sids), total=len(ids))), flush=True)
    bank.flush()
    del bank
    joined = dict(queries=queries, frozen_cure_raw=frozen_raw, **moments)
    if historical_raw is not None:
        joined['historical_cure_raw'] = historical_raw
    np.savez_compressed(out / 'object_joins_and_targets.npz', **joined)
    write(out / 'records.json', rows)
    require(state_digest(model) == before and all(p.grad is None for p in model.parameters()),
            'Factual capture mutated frozen model or gradients')
    report = dict(status='passed_factual_representation_capture_only', frames=len(ids),
                  objects=int(moments['object_mask'].sum()), empty_frames=int((~moments['object_mask'].any(1)).sum()),
                  representation_shape=[len(ids), 300, 512], source_state_before=before,
                  source_state_after=state_digest(model), all_queries_captured_before_object_gather=True,
                  new_interventions=False, AP=False, historical_joins_authoritative=historical is not None,
                  changed_recaptured_matching_frames=sum(r['object_order_queries'] != r['recaptured_assignment_queries'] for r in rows))
    write(out / 'REPORT.json', report)
    return np.load(out / 'factual_all_query_features.npy', mmap_mode='r'), joined, rows


def gradient_receipt(head, require_positive):
    grads = {}
    for k, p in head.named_parameters():
        require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Head gradient absent/nonfinite: ' + k)
        grads[k] = p.grad.detach().cpu().numpy().copy()
    norms = {k: float(np.square(v.astype(np.float64)).sum()) for k, v in grads.items()}
    row_norms = (np.square(grads['3.weight'].astype(np.float64)).sum(1) +
                 np.square(grads['3.bias'].astype(np.float64))).tolist()
    if require_positive:
        require(all(v > 0 for v in norms.values()) and all(v > 0 for v in row_norms),
                'Initial admission requires connected hidden block and both mean/variance modalities')
    return grads, dict(parameter_squared_norms=norms, four_output_row_squared_norms=row_norms)


def train(head, features, joined, ids, out, admission):
    out.mkdir(exist_ok=False)
    require(sum(p.numel() for p in head.parameters()) == 66436, 'Original head capacity differs')
    opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, out / 'INITIAL.pt')
    initial_hash = state_digest(head)
    updates = 0
    epochs = 1 if admission else 40
    with (out / 'updates.jsonl').open('x') as log:
        for epoch in range(1, epochs+1):
            # Admission records already follow original epoch-one first64 order.
            order = np.arange(len(ids)) if admission else epoch_order(len(ids), 11, epoch)
            for group in opt.param_groups:
                group['lr'] = learning_rate_v1(epoch)
            for offset in range(0, len(order), 64):
                index = order[offset:offset+64]
                total_objects = int(joined['object_mask'][index].sum())
                opt.zero_grad(set_to_none=True)
                aggregate = 0.
                snapshot = updates == 0 or (epoch == epochs and offset+64 >= len(order))
                if snapshot:
                    before = tensor_arrays(head.state_dict())
                    probes = []
                for local in range(0, len(index), 16):
                    chunk = index[local:local+16]
                    mask = joined['object_mask'][chunk]
                    q = joined['queries'][chunk]
                    # Padded positions use query0 solely to retain a graph-connected zero.
                    # They never contribute a utility target, including actual empty frames.
                    data = np.asarray(features[chunk])
                    selected = data[np.arange(len(chunk))[:, None], np.maximum(q, 0)].copy()
                    x = torch.from_numpy(selected).to('cuda')
                    m = torch.from_numpy(mask.copy()).to('cuda')
                    truth = torch.from_numpy(joined['mean'][chunk].copy()).to('cuda')
                    u2 = torch.from_numpy(joined['mean_sampling_variance'][chunk].copy()).to('cuda')
                    raw = head(x)
                    n = int(mask.sum())
                    loss = utility_batch_loss_v1(raw, truth, u2, m) * (n/max(total_objects, 1))
                    require(bool(torch.isfinite(loss)), 'Nonfinite original-unit utility loss')
                    loss.backward()
                    aggregate += float(loss.detach())
                    if snapshot:
                        probes.append(dict(features=selected, mask=mask.copy(), target=truth.cpu().numpy(),
                                           u2=u2.cpu().numpy(), raw=raw.detach().cpu().numpy(),
                                           index=chunk.copy()))
                grads, connection = gradient_receipt(head, require_positive=updates == 0)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                require(bool(torch.isfinite(norm)), 'Nonfinite utility-head gradient norm')
                opt.step()
                updates += 1
                require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'Nonfinite utility-head parameter')
                event = dict(epoch=epoch, update=updates, offset=offset, identities=[ids[int(i)] for i in index],
                             objects=total_objects, learning_rate=learning_rate_v1(epoch), loss=aggregate,
                             gradient_norm_before_clip=float(norm), **connection)
                log.write(json.dumps(event)+'\n')
                log.flush()
                if snapshot:
                    folder = out / ('step_%04d' % updates)
                    folder.mkdir()
                    np.savez_compressed(folder / 'BEFORE.npz', **before)
                    np.savez_compressed(folder / 'GRADIENTS_BEFORE_CLIP.npz', **grads)
                    np.savez_compressed(folder / 'AFTER.npz', **tensor_arrays(head.state_dict()))
                    for j, probe in enumerate(probes):
                        np.savez_compressed(folder / ('micro_%02d.npz' % j), **probe)
                    torch.save(opt.state_dict(), folder / 'OPTIMIZER.pt')
            checkpoint = out / ('epoch_%03d.pt' % epoch)
            torch.save(dict(head={k: v.detach().cpu() for k, v in head.state_dict().items()},
                            optimizer=opt.state_dict(), epoch=epoch, updates=updates), checkpoint)
            print(json.dumps(dict(stage='fresh_utility_head_training', epoch=epoch, epochs=epochs, updates=updates)), flush=True)
    require(updates == (1 if admission else 3880), 'Original fixed update budget differs')
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, out / 'FINAL.pt')
    write(out / 'REPORT.json', dict(status='passed_one_utility_head_step_only' if admission else 'completed_original_unit_head_fit',
          initial_state_sha256=initial_hash, final_state_sha256=state_digest(head), epochs=epochs, updates=updates,
          parameters=66436, no_detector_optimizer=True, physical_target_units_unchanged=True,
          original_fresh_initialization=True, no_early_stopping=True))


def assess(head, features, joined, records, out):
    out.mkdir(exist_ok=False)
    head.eval()
    raw = np.zeros((*joined['queries'].shape, 4), np.float32)
    with torch.no_grad():
        for start in range(0, len(records), 16):
            q = joined['queries'][start:start+16]
            data = np.asarray(features[start:start+len(q)])
            x = data[np.arange(len(q))[:, None], np.maximum(q, 0)].copy()
            raw[start:start+len(q)] = head(torch.from_numpy(x).to('cuda')).cpu().numpy()
    mask = joined['object_mask']
    operands = dict(control_raw=raw, **joined)
    np.savez_compressed(out / 'ALL_OBJECT_OPERANDS.npz', **operands)
    write(out / 'records.json', records)
    metrics = {name: descriptive_metrics_v1(operands[key][mask], joined['mean'][mask],
                                           joined['mean_sampling_variance'][mask])
               for name, key in [('fresh_trained_head', 'control_raw'),
                                 ('same_capture_frozen_CURE_head', 'frozen_cure_raw'),
                                 ('original_saved_CURE_head', 'historical_cure_raw')]}
    comparison = np.abs(joined['historical_cure_raw'][mask] - joined['frozen_cure_raw'][mask])
    write(out / 'REPORT.json', dict(status='completed_same_representation_utility_assessment_only',
          metrics=metrics, original_vs_recaptured_raw_max_absolute_difference=comparison.max(0).tolist(),
          original_vs_recaptured_raw_exactly_equal=bool(np.count_nonzero(comparison) == 0),
          no_reassignment_or_row_selection=True, AP=False, confidence_intervals=False,
          scope='Post-hoc same-representation trained readout; representation already shaped by CURE supervision. Reused development objects; no independent representation, causal, routing, detector or generalization benefit claim.'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'cache', 'original_study', 'output'):
        ap.add_argument('--'+name.replace('_', '-'), type=pathlib.Path, required=True)
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
        require(all(v is not None for v in (a.admission_report, a.admission_report_sha256,
                                            a.admission_review, a.admission_review_sha256)),
                'Actual independently reviewed single-step admission required')
        require(sha(a.admission_report) == a.admission_report_sha256 and sha(a.admission_review) ==
                a.admission_review_sha256, 'Actual admission/review external hash differs')
        actual = json.loads(a.admission_report.read_text())
        peer = json.loads(a.admission_review.read_text())
        require(actual['status'] == 'passed_original_CSU_head_control_admission_only' and
                actual['prepared_protocol_sha256'] == p['prepared_admission_protocol_sha256'] and
                actual['optimizer_updates'] == 1 and actual['final_integrity'] == 'passed',
                'Different or incomplete actual head admission')
        require(peer['status'] == 'passed_original_CSU_head_control_actual_review_only' and
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
                  environment=dict(torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0)), optimizer_updates=0)
    started = time.time()
    try:
        original = json.loads((ROOT / p['original_protocol_path']).read_text())
        splits = json.loads((ROOT / original['splits_path']).read_text())
        cache = Cache(a.cache, original['cache_protocol_sha256'], splits,
                      original['optimization_records_canonical_sha256'], p['cache_report_sha256'])
        model = DfineNativeCureV2(DfineCureConfigV1()).cuda().eval().requires_grad_(False)
        saved = torch.load(a.original_study / 'cure/epoch_040/checkpoint.pt', map_location='cpu', weights_only=True)
        require(saved['epoch'] == 40 and saved['arm'] == 'cure', 'Fixed CURE endpoint required')
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
        f, j, r = capture(model, cache, fit, moments, a.output / 'fit_capture')
        require(set(v for row in r for v in row['labels']) == {0, 1, 2, 3} and
                any(not row['labels'] for row in r), 'Actual four-class and empty-frame admission population required')
        train(head, f, j, fit, a.output / 'control_fit', admission)
        report['optimizer_updates'] = 1 if admission else 3880
        if not admission:
            write(a.output / 'ALL_TRAINING_COMPLETED.json', dict(updates=3880, epochs=40,
                  final_head_state_sha256=state_digest(head), before_development_capture_or_assessment=True))
            dev = splits['inner_development']['sample_ids']
            historical = json.loads((a.original_study / 'evaluation_cure_clean/value_observations.json').read_text())
            moments = load_moments(a.original_study, 'development', dev, cache)
            f, j, r = capture(model, cache, dev, moments, a.output / 'development_capture', historical)
            assess(head, f, j, r, a.output / 'assessment')
        require(state_digest(model) == model_before and all(v.grad is None for v in model.parameters()),
                'Frozen final CURE changed during separate head training')
        report.update(status='passed_original_CSU_head_control_admission_only' if admission else
                      'completed_original_CSU_same_representation_head_control',
                      frozen_CURE_state_before=model_before, frozen_CURE_state_after=state_digest(model),
                      parameters=66436, head_only=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            for rel, digest in p['files'].items():
                require(sha(ROOT / rel) == digest, 'Bound source changed: '+rel)
            check_bound(a.original_study, p['original_study_files'])
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
    return 0 if report['status'] in ('passed_original_CSU_head_control_admission_only',
                                   'completed_original_CSU_same_representation_head_control') else 1


if __name__ == '__main__':
    raise SystemExit(main())

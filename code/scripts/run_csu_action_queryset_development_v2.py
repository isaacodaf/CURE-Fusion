"""Saved-query action utility: separate CUDA step admission and fixed20-fit study.

No detector forward, recapture, query mixing, rescoring or new AP definition.
The original20 fits/five controller seeds/one detector seed remain distinct.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import torch
from cure_fusion.csu_action_queryset_v2 import (
    FrameActionQueryUtilityV2, FrameActionQueryRuleV2, queryset_rule_extra_v2,
    UTILITY_PARAMETERS_V2, RULE_PARAMETERS_V2, require)
from cure_fusion.csu_action_rules_v1 import condition_average, expected_action_risk, sequence_risk_interval
from cure_fusion.dfine_cure_development_v2 import sha, write, state_digest, epoch_order

SPEC = importlib.util.spec_from_file_location('original_csu_action_study',
    ROOT / 'code/scripts/run_csu_action_development_cuda_v1.py')
ORIGINAL = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ORIGINAL)
ROLES, CONDITIONS, ARMS = ORIGINAL.ROLES, ORIGINAL.CONDITIONS, ORIGINAL.ARMS
MODES = ('utility', 'zero', 'permuted')
PREPARED_SPEC = 'QA/csu_queryset_preparation_20260913/PREPARED_SPEC_V2.json'
TOKEN_SPEC = 'QA/csu_queryset_preparation_20260913/TOKEN_PREPARATION_SPEC_V3.json'
ADMISSION_STATUS = 'passed_queryset_four_model_single_step_CUDA_admission_only'
STUDY_STATUS = 'completed_queryset_reused_SEW_action_study'


def cpu(value):
    return value.detach().cpu().numpy()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return value


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b): return False
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def save_tensor_state(path, value):
    state = cpu_tree(value)
    torch.save(state, path)
    reread = torch.load(path, map_location='cpu', weights_only=True)
    require(equal_tree(state, reread), 'Exact saved model/optimizer readback failed')
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size, exact_readback=True)


def schedule(size, seed, epochs):
    for epoch in range(1, epochs + 1):
        order = epoch_order(size, seed, epoch)
        require(len(set(order.tolist())) == size and set(order.tolist()) == set(range(size)),
                'Complete unique source-frame epoch required')
        yield epoch, [order[i:i + 64] for i in range(0, size, 64)]


def predict_batch(model, tokens, extra=None):
    require(tokens.ndim == 5 and tokens.shape[1:] == (4, 2, 300, 86), 'Four full conditions/actions/query tokens required')
    flat = tokens.flatten(0, 1)
    if extra is None: result = model(flat)
    else:
        require(extra.shape == tokens.shape[:2], 'Aligned detached utility scalar required')
        result = model(flat, extra.flatten(0, 1))
    return result.reshape(tokens.shape[:2])


def objective(prediction, gain, base, candidate, kind, gain_scale):
    if kind == 'predictor': return condition_average((prediction - gain).square()) / gain_scale.square()
    require(kind in ('selector_utility', 'selector_zero', 'selector_permuted'), 'Undeclared learned rule')
    return expected_action_risk(prediction, base, candidate)


def gradient_snapshot(model, require_positive):
    raw = {name: cpu(p.grad) for name, p in model.named_parameters()}
    groups = {group: sum(float(np.square(v.astype(np.float64)).sum())
                         for name, v in raw.items() if name.startswith(group))
              for group in ('encoder.phi.', 'net.')}
    require(all(np.isfinite(v) and v >= 0 for v in groups.values()), 'Nonfinite query encoder/readout gradients')
    if require_positive:
        require(all(v > 0 for v in groups.values()), 'Initial query encoder/readout connectivity failed')
    return raw, groups


def fit_model(model, cache, operands, positions, seed, kind, extra, destination, p, gain_scale, epochs):
    """Native rich-token model only; initial/final/optimizer/journal all retained."""
    destination.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['optimizer']['lr'],
                                  weight_decay=p['optimizer']['weight_decay'])
    expected_count = UTILITY_PARAMETERS_V2 if kind == 'predictor' else RULE_PARAMETERS_V2
    require(sum(x.numel() for x in model.parameters()) == expected_count, 'Prepared parameter count differs')
    model.train(); initial = state_digest(model)
    initial_file = save_tensor_state(destination / 'INITIAL.pt', model.state_dict())
    normalizers = {name: cpu(value).copy() for name, value in model.named_buffers()}
    size = len(positions); expected_steps = epochs * ((size + 63) // 64)
    steps = 0; epochs_log = []; snapshots = []; started = time.time()
    with (destination / 'UPDATES.jsonl').open('x') as journal:
        for epoch, batches in schedule(size, seed, epochs):
            seen = []; accumulated = 0.
            for local in batches:
                tick = time.time(); indices = np.asarray(positions)[local]; seen.extend(local.tolist())
                tokens = torch.from_numpy(cache.batch('controller_fit', indices)).cuda()
                require(not tokens.requires_grad, 'Frozen saved tokens cannot receive gradients')
                selected_extra = None if extra is None else extra[indices]
                gain, base, candidate = [operands[k][indices] for k in ('gain', 'base_risk', 'candidate_risk')]
                observe = steps in (0, expected_steps - 1)
                if steps == expected_steps - 1:
                    save_tensor_state(destination / 'FINAL_STEP_BEFORE.pt', dict(
                        model=model.state_dict(), optimizer=optimizer.state_dict(),
                        update=steps, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
                optimizer.zero_grad(set_to_none=True)
                prediction = predict_batch(model, tokens, selected_extra)
                loss = objective(prediction, gain, base, candidate, kind, gain_scale)
                require(prediction.is_cuda and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(loss)),
                        'Finite CUDA predictions/loss required')
                loss.backward()
                require(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()),
                        'Missing or nonfinite trainable gradients')
                if observe:
                    raw_grad, norms = gradient_snapshot(model, require_positive=steps == 0)
                    if kind == 'selector_zero':
                        require(np.count_nonzero(raw_grad['net.0.weight'][:, -1]) == 0,
                                'Zero utility input must have exact zero last-column loss derivative')
                    probe = np.asarray([0, len(indices) - 1], np.int64)
                    raw = dict(indices=indices, prediction=cpu(prediction), gain=cpu(gain),
                               base_risk=cpu(base), candidate_risk=cpu(candidate), loss=cpu(loss),
                               token_probe_local_indices=probe, token_probe=cpu(tokens[probe]),
                               extra=np.empty((0,), np.float32) if selected_extra is None else cpu(selected_extra))
                    raw.update({'gradient/' + name: value for name, value in raw_grad.items()})
                    path = destination / ('step_%04d.npz' % (steps + 1))
                    np.savez_compressed(path, **raw)
                    snapshots.append(dict(update=steps + 1, path=path.name, sha256=sha(path),
                                          gradient_groups_squared_norm=norms))
                optimizer.step(); steps += 1
                require(all(bool(torch.isfinite(v).all()) for v in model.parameters()), 'Nonfinite updated parameter')
                value = float(loss.detach()); accumulated += value * len(indices)
                journal.write(json.dumps(dict(update=steps, epoch=epoch, indices=indices.tolist(),
                    frames=len(indices), conditions=4, loss=value, seconds=time.time()-tick,
                    group_lrs=[g['lr'] for g in optimizer.param_groups]), allow_nan=False) + '\n')
                journal.flush()
                del tokens, prediction, loss
            require(len(seen) == len(set(seen)) == size, 'Full unique epoch coverage differs')
            epochs_log.append(dict(epoch=epoch, frames=size, steps=steps, mean_loss=accumulated / size))
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(json.dumps(dict(stage='queryset_training', seed=seed, model=kind, epoch=epoch,
                                      epochs=epochs, updates=steps)), flush=True)
    require(steps == expected_steps, 'Fixed optimizer budget differs')
    require(set(optimizer.state) == set(model.parameters()), 'All prepared parameters require retained optimizer state')
    for parameter, value in optimizer.state.items():
        require(value['step'].numel() == 1 and torch.isfinite(value['step']) and value['step'].item() == steps,
                'Exact optimizer step counter differs')
        require(all(value[k].shape == parameter.shape and value[k].dtype == parameter.dtype and
                    bool(torch.isfinite(value[k]).all()) for k in ('exp_avg', 'exp_avg_sq')) and
                bool((value['exp_avg_sq'] >= 0).all()), 'Invalid retained AdamW moments')
    require(all(np.array_equal(cpu(v), normalizers[n]) for n, v in model.named_buffers()),
            'Fitting normalization buffers changed')
    final_file = save_tensor_state(destination / 'FINAL.pt', model.state_dict())
    optimizer_file = save_tensor_state(destination / 'OPTIMIZER.pt', optimizer.state_dict())
    training = dict(seed=seed, kind=kind, parameter_count=expected_count, frames=size,
                    initial_state_sha256=initial, final_state_sha256=state_digest(model),
                    initial_file=initial_file, final_file=final_file, optimizer_file=optimizer_file,
                    epochs=epochs_log, steps=steps, raw_snapshots=snapshots,
                    normalization_unchanged=True, seconds=time.time()-started)
    write(destination / 'TRAINING.json', training)
    model.eval().requires_grad_(False)
    for parameter in model.parameters(): parameter.grad = None
    return model, training


@torch.no_grad()
def infer_role(model, cache, role, extra=None, positions=None):
    positions = np.arange(len(cache.roles[role]['sample_ids'])) if positions is None else np.asarray(positions)
    values = []
    for start in range(0, len(positions), 64):
        indices = positions[start:start + 64]
        tokens = torch.from_numpy(cache.batch(role, indices)).cuda()
        values.append(predict_batch(model, tokens, None if extra is None else extra[indices]).detach())
    return torch.cat(values)


def operand_tensors(cache):
    return {role: {k: torch.from_numpy(np.asarray(cache.operands(role)[k]).copy()).cuda()
                   for k in ('gain', 'base_risk', 'candidate_risk')} for role in ROLES}


def train_all(out, p, cache, admission=False):
    operands = operand_tensors(cache)
    center, scale = [np.asarray(cache.normalization[k]).copy() for k in ('center', 'scale')]
    gain_scale = torch.as_tensor(np.asarray(cache.normalization['gain_scale']).copy(),
                                 dtype=torch.float32, device='cuda').reshape(())
    require(bool(torch.isfinite(gain_scale)) and gain_scale > 0, 'Exact original fitting gain scale required')
    np.savez_compressed(out / 'NORMALIZATION.npz', center=np.asarray(center), scale=np.asarray(scale), gain_scale=cpu(gain_scale))
    positions = np.arange(64 if admission else 5050)
    seeds = [11] if admission else p['controller_seeds']
    epochs = 1 if admission else p['optimizer']['epochs']
    fits = []; fit = operands['controller_fit']
    for seed in seeds:
        folder = out / ('seed_%d' % seed); folder.mkdir()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        predictor = FrameActionQueryUtilityV2(center, scale).cuda()
        predictor, info = fit_model(predictor, cache, fit, positions, seed, 'predictor', None,
                                    folder / 'predictor', p, gain_scale, epochs)
        fits.append(dict(info, path=str((folder/'predictor').relative_to(out))))
        predictor_hash = state_digest(predictor)
        infer_roles = ('controller_fit',) if admission else ROLES
        means = {r: infer_role(predictor, cache, r, positions=positions if admission else None) for r in infer_roles}
        rms = condition_average((means['controller_fit'] - fit['gain'][positions]).square()).sqrt()
        choices = {r: dict(mean=cpu(m), positive_mean=cpu(m > 0), fitting_RMS_margin=cpu(m-rms > 0)) for r, m in means.items()}
        if not admission:
            residual = (means['inner_calibration'] - operands['inner_calibration']['gain']).abs()
            seq = np.asarray(cache.operands('inner_calibration')['sequences'])
            sequence_names = sorted(set(seq.tolist())); require(len(sequence_names) == 9, 'Nine reused calibration recordings required')
            sequence_max = np.asarray([float(residual[np.where(seq == s)[0]].max()) for s in sequence_names])
            radius = float(sequence_max.max())
            np.savez_compressed(folder / 'EMPIRICAL_CALIBRATION.npz', sequence_names=np.asarray(sequence_names),
                                sequence_max=sequence_max, radius=np.asarray(radius), fitting_rms=cpu(rms))
        initial_rule = None
        for mode in MODES:
            torch.manual_seed(seed + 1000); torch.cuda.manual_seed_all(seed + 1000)
            selector = FrameActionQueryRuleV2(center, scale).cuda()
            digest = state_digest(selector)
            if initial_rule is None: initial_rule = digest
            require(digest == initial_rule, 'Rich selector encoder/readout initial state differs across controls')
            extras = {r: queryset_rule_extra_v2(m, gain_scale, mode, seed, r) for r, m in means.items()}
            selector, info = fit_model(selector, cache, fit, positions, seed, 'selector_' + mode,
                                       extras['controller_fit'], folder / ('selector_' + mode), p, gain_scale, epochs)
            fits.append(dict(info, path=str((folder/('selector_'+mode)).relative_to(out))))
            for role in infer_roles:
                value = infer_role(selector, cache, role, extras[role], positions if admission else None)
                # Preserve the actual V1 threshold, including its exact FP32 boundary.
                choices[role]['selector_' + mode] = cpu(value > 0)
                choices[role]['selector_' + mode + '_logits'] = cpu(value)
                choices[role]['selector_' + mode + '_extra'] = cpu(extras[role])
            require(state_digest(predictor) == predictor_hash and
                    all(not v.requires_grad and v.grad is None for v in predictor.parameters()),
                    'Entire utility predictor must stay frozen during selector training')
            del selector
        for role in infer_roles:
            chosen_gain = operands[role]['gain'][positions] if admission else operands[role]['gain']
            choices[role]['gain'] = cpu(chosen_gain)
            choices[role]['identities'] = np.asarray(cache.roles[role]['sample_ids'])[positions] if admission else np.asarray(cache.roles[role]['sample_ids'])
            if not admission: choices[role]['empirical_band_covers'] = cpu((means[role]-chosen_gain).abs() <= radius)
            np.savez_compressed(folder / (role + '_CHOICES.npz'), **choices[role])
        del predictor
    expected_models, expected_steps = (4, 1) if admission else (20, 3160)
    require(len(fits) == expected_models and all(x['steps'] == expected_steps for x in fits), 'Fixed model/update count differs')
    write(out / ('ADMISSION_TRAINING_COMPLETE.json' if admission else 'ALL_TRAINING_COMPLETE.json'), dict(
        seeds=seeds, AP_calculated=False, detector_seeds=1, source_frames=len(positions),
        epochs=epochs, steps_per_model=expected_steps, learned_models_per_seed=4, total_models=expected_models,
        total_optimizer_updates=sum(x['steps'] for x in fits), models=fits,
        initial_state_policy='fresh seeded constructors; admission state never reused by study',
        choice_threshold='logit >0; exact ties select task-only'))
    return fits


def complete_training_gate(out, p):
    marker = json.loads((out / 'ALL_TRAINING_COMPLETE.json').read_text())
    require(marker['seeds'] == p['controller_seeds'] and marker['total_models'] == 20 and
            marker['steps_per_model'] == 3160 and marker['total_optimizer_updates'] == 63200,
            'All20 fixed fits must finish before any AP')
    require(len(marker['models']) == 20 and {(x['seed'], x['kind']) for x in marker['models']} ==
            {(s, k) for s in p['controller_seeds'] for k in ('predictor', 'selector_utility', 'selector_zero', 'selector_permuted')},
            'Missing or duplicate learned arm/seed')
    for row in marker['models']:
        directory = out / row['path']
        require(row['frames'] == 5050 and row['steps'] == 3160 and len(row['epochs']) == 40 and
                row['epochs'][-1]['epoch'] == 40 and row['epochs'][-1]['steps'] == 3160,
                'Each fixed model must finish its40 epochs/3160 updates')
        require(json.loads((directory / 'TRAINING.json').read_text()) == {k:v for k,v in row.items() if k != 'path'},
                'Training completion receipt differs')
        for field in ('initial_file', 'final_file', 'optimizer_file'):
            require(sha(directory / row[field]['path']) == row[field]['sha256'], 'Saved training state changed')
    return marker


def evaluate_all(out, p, cache):
    complete_training_gate(out, p)
    ids = cache.roles['reused_development']['sample_ids']
    seq = [cache.records[s]['sequence'] for s in ids]
    gt = ORIGINAL.build_gt(cache.records, ids)
    require(gt == cache.ground_truth, 'Original complete development ground truth differs')
    write(out / 'GROUND_TRUTH_COCO.json', gt)
    operands = cache.operands('reused_development')
    gain, base, candidate = [np.asarray(operands[k]) for k in ('gain', 'base_risk', 'candidate_risk')]
    selected = cache.selected; results = []; metric_cache = {}
    selections = dict(identities=np.asarray(ids))
    for ci, condition in enumerate(CONDITIONS):
        for arm in ARMS:
            result = ORIGINAL.ap_result(gt, selected[condition][arm])
            results.append(dict(condition=condition, method=arm, seed=None, **result))
        confidence = np.asarray(cache.confidence[:, ci, 1] > cache.confidence[:, ci, 0])
        selections[condition + '_confidence'] = confidence
        for seed in p['controller_seeds']:
            with np.load(out / ('seed_%d' % seed) / 'reused_development_CHOICES.npz', allow_pickle=False) as z:
                require(z['identities'].tolist() == ids, 'Complete original development identity order differs')
                choices = {k: z[k][:, ci].copy() for k in ('positive_mean', 'fitting_RMS_margin', 'selector_utility', 'selector_zero', 'selector_permuted')}
                mean = z['mean'][:, ci].copy(); coverage = z['empirical_band_covers'][:, ci].copy()
            choices['confidence'] = confidence
            for method, choice in choices.items():
                if method == 'confidence' and seed != p['controller_seeds'][0]: continue
                require(choice.dtype == np.bool_ and choice.shape == (992,), 'Complete frame-level choices required')
                selections[condition + '_' + method + '_' + str(seed)] = choice
                key = (condition, choice.tobytes())
                if key not in metric_cache:
                    metric_cache[key] = ORIGINAL.ap_result(gt, [selected[condition]['cure' if use else 'task_only'][i]
                                                               for i, use in enumerate(choice)])
                risk = np.where(choice, candidate[:, ci], base[:, ci])
                result = dict(condition=condition, method=method, seed=None if method == 'confidence' else seed,
                    **metric_cache[key], selection_fraction=float(choice.mean()),
                    harmful_selection_fraction=float((choice & (gain[:, ci] < 0)).mean()),
                    mean_frame_risk=float(risk.mean()),
                    paired_sequence_risk_delta=sequence_risk_interval(risk-base[:, ci], seq),
                    gain_weighted_wrong_action_regret=float(np.where(choice, np.maximum(-gain[:, ci], 0),
                                                                      np.maximum(gain[:, ci], 0)).mean()))
                if method in ('selector_utility', 'positive_mean', 'fitting_RMS_margin'):
                    result['utility'] = dict(RMSE=float(np.sqrt(np.mean((mean-gain[:, ci])**2))),
                        MAE=float(np.mean(np.abs(mean-gain[:, ci]))), sign_agreement=float(np.mean((mean > 0) == (gain[:, ci] > 0))),
                        empirical_band_coverage=float(coverage.mean()))
                results.append(result)
            write(out / 'RESULTS.json', results)
            print(json.dumps(dict(stage='queryset_AP_after_all_training', condition=condition, seed=seed)), flush=True)
    require(len(results) == 116, 'Every fixed rule/condition/seed row must be retained')
    summary = []
    for condition in CONDITIONS:
        for method in sorted(set(row['method'] for row in results)):
            rows = [row for row in results if row['condition'] == condition and row['method'] == method]
            summary.append(dict(condition=condition, method=method, controller_runs=len(rows),
                **{k: dict(mean=float(np.mean([r[k] for r in rows])),
                           SD=float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else None,
                           all_values=[r[k] for r in rows]) for k in ('AP_percent', 'AP50_percent', 'AP75_percent')}))
    require(len(summary) == 36, 'All9 rules and4 conditions required')
    np.savez_compressed(out / 'ALL_DEVELOPMENT_CHOICES.npz', **selections)
    write(out / 'SUMMARY.json', summary)
    return summary


def admission_gate(directory, digest, p, token_manifest_sha):
    require(directory is not None and digest is not None, 'Actual single-step admission is required')
    require(sha(directory / 'REPORT.json') == digest, 'Externally supplied admission report differs')
    report = json.loads((directory / 'REPORT.json').read_text())
    require(report['status'] == ADMISSION_STATUS and report['source_integrity'] == 'passed' and
            report['AP_evaluated'] is False and report['detector_forward_calls'] == 0 and
            report['total_models'] == report['total_optimizer_updates'] == 4,
            'Completed four-model one-step admission required')
    require(report['token_manifest_sha256'] == token_manifest_sha and report['prepared_spec_sha256'] == p['prepared_spec_sha256'] and
            report['token_preparation_spec_sha256'] == p['token_preparation_spec_sha256'],
            'Admission representation/data contract differs')
    for path in p['admission_shared_source_paths']:
        require(report['bound_files'][path] == p['files'][path], 'Admission source differs: ' + path)
    actual_manifest = json.loads((directory / 'MANIFEST.json').read_text())
    require(actual_manifest['REPORT.json']['sha256'] == digest, 'Admission manifest/report mismatch')
    for path, item in actual_manifest.items():
        require(sha(directory / path) == item['sha256'] and (directory / path).stat().st_size == item['bytes'],
                'Retained single-step admission artifact changed')
    return dict(report_sha256=digest, manifest_sha256=sha(directory / 'MANIFEST.json'),
                scope='admission state never loaded into a study model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('admission', 'study'), required=True)
    for key in ('protocol', 'tokens', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    for key in ('protocol-sha256', 'token-manifest-sha256'): parser.add_argument('--' + key, required=True)
    parser.add_argument('--admission', type=Path); parser.add_argument('--admission-report-sha256')
    args = parser.parse_args()
    require(sha(args.protocol) == args.protocol_sha256, 'Execution protocol hash differs')
    p = json.loads(args.protocol.read_text())
    require(p['launch_authorized'] is True and p['mode'] == args.mode, 'A separately frozen execution mode is required')
    require(all(sha(ROOT / path) == value for path, value in p['files'].items()), 'Bound source differs')
    require(sha(ROOT / PREPARED_SPEC) == p['prepared_spec_sha256'], 'Preserved richer representation specification differs')
    require(sha(ROOT / TOKEN_SPEC) == p['token_preparation_spec_sha256'], 'Declared token preparation differs')
    require(p['controller_seeds'] == [11, 23, 37, 53, 71] and p['optimizer']['epochs'] == 40 and
            p['optimizer']['updates_per_model'] == 3160 and p['optimizer']['lr'] == .001 and
            p['optimizer']['weight_decay'] == .0001, 'Original matched budget differs')
    require(torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0), 'A100 required; no learned CPU/MPS fallback')
    from cure_fusion.csu_queryset_inputs_v2 import QuerySetCacheV2
    cache = QuerySetCacheV2(args.tokens, args.token_manifest_sha256)
    require(cache.report['specification_sha256'] == p['token_preparation_spec_sha256'],
            'Token files belong to a different preparation recipe')
    require([len(cache.roles[r]['sample_ids']) for r in ROLES] == [5050, 1130, 992], 'Original role sizes required')
    admission = admission_gate(args.admission, args.admission_report_sha256, p, args.token_manifest_sha256) if args.mode == 'study' else None
    require(not args.output.exists(), 'Never overwrite an earlier run'); args.output.mkdir(parents=True)
    write(args.output / 'protocol.json', p)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    started = time.time()
    report = dict(status='started', mode=args.mode, protocol_sha256=args.protocol_sha256, bound_files=p['files'],
        prepared_spec_sha256=p['prepared_spec_sha256'], token_manifest_sha256=args.token_manifest_sha256,
        token_preparation_spec_sha256=p['token_preparation_spec_sha256'],
        admission=admission, CUDA_executed=False, AP_evaluated=False, detector_forward_calls=0,
        detector_seeds=1, controller_seeds=[11] if args.mode == 'admission' else p['controller_seeds'],
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
        versions={n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pycocotools')},
        choice_threshold='logit >0; ties task-only',
        scope='reused SEW action-representation development; five controller seeds around one fixed detector seed',
        utility_information='Learned deterministic representation of the same rich tokens every selector receives',
        empirical_bands='Reused empirical action-gain coverage; no distribution-free or latent CSU guarantee',
        efficiency_scope='Query-model execution time/memory only; no detector or production end-to-end latency claim')
    write(args.output / 'REPORT.json', report)
    try:
        report['CUDA_executed'] = True
        fits = train_all(args.output, p, cache, admission=args.mode == 'admission')
        report.update(total_models=len(fits), total_optimizer_updates=sum(v['steps'] for v in fits),
                      completed_models=[dict(seed=x['seed'], kind=x['kind'], steps=x['steps'], final_state_sha256=x['final_state_sha256']) for x in fits])
        if args.mode == 'admission':
            report.update(status=ADMISSION_STATUS, admission_ids=cache.roles['controller_fit']['sample_ids'][:64],
                          warm_start_allowed=False)
        else:
            summary = evaluate_all(args.output, p, cache)
            report.update(status=STUDY_STATUS, AP_evaluated=True, summary_rows=len(summary),
                          AP_interval='Withheld: slidecar occurs in only one reused development recording; point AP retained')
    except Exception:
        report.update(status='failed_queryset_' + args.mode, error=traceback.format_exc())
    finally:
        try:
            cache.verify_unchanged()
            require(sha(args.protocol) == args.protocol_sha256 and
                    all(sha(ROOT / path) == value for path, value in p['files'].items()), 'Source/protocol changed')
            if admission is not None: admission_gate(args.admission, args.admission_report_sha256, p, args.token_manifest_sha256)
            report['source_integrity'] = 'passed'
        except Exception:
            report.update(status='failed_queryset_integrity', source_integrity='failed', integrity_error=traceback.format_exc())
        report.update(seconds=time.time()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write(args.output / 'REPORT.json', report)
        write(args.output / 'MANIFEST.json', {str(f.relative_to(args.output)): dict(sha256=sha(f), bytes=f.stat().st_size)
              for f in sorted(args.output.rglob('*')) if f.is_file() and f.name != 'MANIFEST.json'})
    return 0 if report['status'] in (ADMISSION_STATUS, STUDY_STATUS) else 1


if __name__ == '__main__': raise SystemExit(main())

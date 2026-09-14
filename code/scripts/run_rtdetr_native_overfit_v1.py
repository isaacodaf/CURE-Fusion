#!/usr/bin/env python3
"""Author-native RT-DETR four-class fixed-prefix fitting; no CURE efficacy."""
import argparse
import copy
import gc
import importlib.util
import importlib.metadata
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
import numpy as np
import torch
from cure_fusion.rtdetr_native_admission_v1 import (
    CLASS_HEADS, CLASS_KEYS, EMBED_KEY, CHECKPOINT_SHA256, export_native_coco,
    native_config_overrides, native_target_reference, sha256_file,
    state_inventory, tensor_binding, transfer_coco80_to_sew4, checkpoint_state,
    expected_loss_keys, geometry_observations)
from cure_fusion.rtdetr_native_overfit_v1 import (
    BUDGETS, batch_schedule, equal_tree, evaluate_mode,
    native_selected_records, state_digest, verify_admission)
SPEC = importlib.util.spec_from_file_location('rtdetr_admission_v2',
    ROOT/'code/scripts/run_rtdetr_native_admission_v2.py')
ADMISSION = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ADMISSION)
save, source_guard, collect_tensors = ADMISSION.save, ADMISSION.source_guard, ADMISSION.arrays


def evaluator_environment():
    # Bind actual original pycocotools objects; no backend substitution.
    import importlib
    names = ('pycocotools', 'pycocotools.coco', 'pycocotools.cocoeval', 'pycocotools.mask')
    modules = {name: importlib.import_module(name) for name in names}
    coco, cocoeval, mask = (modules[name] for name in names[1:])
    modules['active_mask_backend'] = mask._mask
    if hasattr(cocoeval, '_C'):
        modules['active_evaluation_backend'] = cocoeval._C
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'torch')}
    for name in ('pycocotools', 'faster-coco-eval'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    assert coco.COCO.__module__ == 'pycocotools.coco'
    assert cocoeval.COCOeval.__module__ == 'pycocotools.cocoeval'
    return dict(versions=versions,
                files={alias: dict(module=m.__name__, path=m.__file__, sha256=sha256_file(m.__file__))
                       for alias, m in modules.items()},
                owners=dict(COCO=coco.COCO.__module__, COCOeval=cocoeval.COCOeval.__module__,
                            mask_iou=mask.iou.__module__),
                scope='Actual public aliases and their already imported native backends; no evaluator replacement')


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def checkpoint(path, model, optimizer, update, protocol_sha, schedule_sha):
    state = dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                 update=update, protocol_sha256=protocol_sha, schedule_sha256=schedule_sha,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    tmp = path.with_suffix('.partial')
    torch.save(state, tmp)
    reread = torch.load(tmp, map_location='cpu', weights_only=True)
    assert equal_tree(state, reread), 'Checkpoint serialization/readback differs'
    os.replace(tmp, path)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha256_file(path), update=update,
                model_digest=state_digest(model), exact_model_optimizer_rng_readback=True)


def gradient_groups(model):
    values = dict(backbone=[], encoder=[], decoder_attention=[], box_distribution=[], denoising_embedding=[])
    values.update({h: [] for h in CLASS_HEADS})
    for name, p in model.named_parameters():
        if p.grad is None: continue
        n = p.grad.detach().double().square().sum()
        if name.startswith('backbone.'): values['backbone'].append(n)
        if name.startswith('encoder.'): values['encoder'].append(n)
        if name.startswith('decoder.decoder.'): values['decoder_attention'].append(n)
        if 'bbox_head' in name: values['box_distribution'].append(n)
        if name == EMBED_KEY: values['denoising_embedding'].append(n)
        for h in CLASS_HEADS:
            if name.startswith(h+'.'): values[h].append(n)
    # Descriptive: absent GT/all-empty batches need not exercise every branch.
    return {k: float(torch.stack(v).sum().cpu()) if v else 0. for k, v in values.items()}


def verify_train_output(output, count, denoising):
    assert output['pred_logits'].shape == output['pred_boxes'].shape == (count, 300, 4)
    assert len(output['aux_outputs']) == 6
    assert ('dn_aux_outputs' in output) == denoising
    if denoising:
        assert len(output['dn_aux_outputs']) == 6 and 'dn_meta' in output


def fit_stage(a, protocol, records, loaded, admitted, size):
    from src.core import YAMLConfig
    out = a.output/('size_%d' % size); out.mkdir(exist_ok=False)
    started = time.time()
    updates = BUDGETS[size]
    random.seed(protocol['seed']); np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
    cfg = YAMLConfig(str(a.source/'rtdetr_pytorch'/protocol['config']))
    cfg.yaml_cfg = native_config_overrides(cfg.yaml_cfg)
    assert cfg.yaml_cfg['PResNet']['freeze_at'] == 0 and cfg.yaml_cfg['PResNet']['freeze_norm'] is True
    assert cfg.yaml_cfg['PResNet']['depth'] == 101 and cfg.clip_max_norm == .1
    save(out/'resolved_native_config.json', cfg.yaml_cfg)
    model = cfg.model
    fresh = model.state_dict()
    original = checkpoint_state(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    expected = json.loads((ROOT/protocol['checkpoint_inventory_path']).read_text())['tensors']
    assert state_inventory(original) == expected
    full = transfer_coco80_to_sew4(original, fresh)
    model.load_state_dict(full, strict=True)
    initial_inventory = state_inventory(model.state_dict())
    assert initial_inventory == admitted['initial_state'], 'Fresh stage initialization differs from admitted seed/transfer'
    del full, fresh, original
    criterion, post = cfg.criterion, cfg.postprocessor
    assert criterion.num_classes == post.num_classes == 4
    assert criterion.losses == ['vfl', 'boxes']
    assert criterion.weight_dict == dict(loss_vfl=1, loss_bbox=5, loss_giou=2)
    assert criterion.alpha == .75 and criterion.gamma == 2.
    assert criterion.matcher.use_focal_loss and criterion.matcher.alpha == .25 and criterion.matcher.gamma == 2.
    assert (criterion.matcher.cost_class, criterion.matcher.cost_bbox, criterion.matcher.cost_giou) == (2, 5, 2)
    assert post.use_focal_loss and post.num_top_queries == 300 and not post.remap_mscoco_category and not post.deploy_mode
    assert len(model.decoder.dec_score_head) == 6 and model.decoder.denoising_class_embed.padding_idx == 4
    model.cuda(); criterion.cuda(); post.cuda(); model.train(); criterion.train()
    assert all(p.device.type == 'cuda' for p in list(model.parameters())+list(model.buffers()))
    optimizer = cfg.optimizer
    assert isinstance(optimizer, torch.optim.AdamW) and not optimizer.state
    optimized = [p for g in optimizer.param_groups for p in g['params']]
    assert len(set(optimized)) == len(optimized) and set(optimized) == {p for p in model.parameters() if p.requires_grad}
    optimizer_facts = ADMISSION.optimizer_facts(optimizer, model)
    assert json.loads(json.dumps(optimizer_facts)) == admitted['optimizer_groups'], 'Admitted optimizer membership/settings differ'
    groups = json.loads(json.dumps([v['settings'] for v in optimizer_facts]))
    frozen = {n: tensor_binding(p) for n, p in model.named_parameters() if not p.requires_grad}
    schedule = batch_schedule(size, updates, np.asarray([bool(r['target']['labels']) for r in records[:size]]))
    save(out/'batch_schedule.json', schedule)
    schedule_sha = sha256_file(out/'batch_schedule.json')
    stage = dict(status='running', size=size, updates=updates, optimizer_steps=0,
                 identities=[r['identity'] for r in records[:size]], initial_state=initial_inventory,
                 optimizer_groups=groups, schedule_sha256=schedule_sha, endpoint=None,
                 scope='camera-only full-native four-class 2D fitting; no CURE or held-out evaluation')
    stage['initial_checkpoint'] = checkpoint(out/'initial.pt', model, optimizer, 0, a.protocol_sha256, schedule_sha)
    save(out/'REPORT.json', stage)
    source_loss = criterion.get_loss
    raw_losses = []
    def observer(*args, **kwargs):
        value = source_loss(*args, **kwargs)
        assert all(v.device.type == 'cuda' and v.dtype == torch.float32 and v.numel() == 1 for v in value.values())
        # Preserve actual pre-nan_to_num values without changing the graph/return.
        raw_losses.append((args[0], {k: v.detach().clone() for k, v in value.items()}))
        return value
    criterion.get_loss = observer
    parameter_steps = {id(p): 0 for p in optimized}
    journal = out/'updates.jsonl'
    try:
        with journal.open('x') as log:
            for step, slots in enumerate(schedule):
                tick = time.time()
                batch = [loaded[i] for i in slots]
                images = torch.stack([b[0] for b in batch]).cuda()
                targets = [{k: v.as_subclass(torch.Tensor).cuda() for k, v in t.items() if isinstance(v, torch.Tensor)} for _, t in batch]
                denoising = any(len(t['labels']) for t in targets)
                assert denoising, 'Declared empty-positive pairing was violated'
                optimizer.zero_grad(set_to_none=True); raw_losses.clear()
                output = model(images, targets=targets)
                verify_train_output(output, len(slots), denoising)
                observe = step in (0, updates-1)
                if observe:
                    output['pred_logits'].retain_grad(); output['pred_boxes'].retain_grad()
                    arrays = collect_tensors(output)
                losses = criterion(output, targets)
                assert set(losses) == expected_loss_keys(denoising)
                assert [r[0] for r in raw_losses] == list(criterion.losses)*(13 if denoising else 7)
                raw_flat = [(i, name, v) for i, (_, d) in enumerate(raw_losses) for name, v in d.items()]
                raw_values = torch.stack([v.reshape(()) for _, _, v in raw_flat])
                assert torch.isfinite(raw_values).all(), 'Native pre-sanitization loss nonfinite'
                values = torch.stack(list(losses.values()))
                assert torch.isfinite(values).all()
                native_total = sum(losses.values())
                geometry_details = geometry_observations(output['pred_boxes'])
                total = native_total
                assert total.requires_grad and torch.isfinite(total) and total > 0
                total.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        assert p.requires_grad and p.grad.device.type == 'cuda' and p.grad.dtype == p.dtype
                details = {}
                if observe:
                    details['gradient_groups_squared_norm'] = gradient_groups(model)
                    arrays['gradient/main_logits'] = output['pred_logits'].grad.detach().cpu().numpy()
                    arrays['gradient/main_boxes'] = output['pred_boxes'].grad.detach().cpu().numpy()
                    for name, p in model.named_parameters():
                        if name in CLASS_KEYS and p.grad is not None:
                            arrays['gradient/'+name] = p.grad.detach().cpu().numpy()
                    np.savez_compressed(out/('train_raw_update_%04d.npz' % (step+1)), **arrays)
                    del arrays
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['clip_max_norm'], error_if_nonfinite=True)
                assert torch.isfinite(norm) and norm > 0
                for p in optimized:
                    if p.grad is not None: parameter_steps[id(p)] += 1
                optimizer.step(); stage['optimizer_steps'] = step+1
                row = dict(update=step+1, slots=slots, identities=[records[i]['identity'] for i in slots],
                           GT=[len(t['labels']) for t in targets], denoising=denoising,
                           loss=float(total.detach().cpu()), gradient_norm=float(norm.cpu()),
                           native_loss=float(native_total.detach().cpu()),
                           geometry_observations=geometry_details,
                           loss_components=dict(zip(losses, values.detach().cpu().tolist())),
                           raw_loss_calls=[dict(family=family, values={k: float(v.cpu()) for k, v in d.items()}) for family, d in raw_losses],
                           group_lrs=[g['lr'] for g in optimizer.param_groups], seconds=time.time()-tick, **details)
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                assert [g['lr'] for g in optimizer.param_groups] == [g['lr'] for g in groups]
                if (step+1) % 250 == 0 and step+1 != updates:
                    stage['latest_checkpoint'] = checkpoint(out/'latest.pt', model, optimizer, step+1, a.protocol_sha256, schedule_sha)
                    save(out/'REPORT.json', stage)
                if step == 0 or (step+1) % 50 == 0 or step+1 == updates:
                    print(json.dumps(dict(size=size, update=step+1, loss=row['loss'], seconds=row['seconds'])), flush=True)
                del images, targets, output, losses, values, raw_values, total, native_total
        criterion.get_loss = source_loss
        assert stage['optimizer_steps'] == updates
        assert set(optimizer.state) == {p for p in optimized if parameter_steps[id(p)] > 0}
        stage['optimizer_parameter_steps'] = {}
        for name, p in model.named_parameters():
            if p in optimizer.state:
                v = optimizer.state[p]; counter = v['step']
                assert isinstance(counter, torch.Tensor) and counter.ndim == 0 and torch.isfinite(counter)
                assert counter.item() == parameter_steps[id(p)]
                assert all(v[k].dtype == p.dtype and v[k].shape == p.shape and torch.isfinite(v[k]).all() for k in ('exp_avg','exp_avg_sq'))
                assert (v['exp_avg_sq'] >= 0).all()
                stage['optimizer_parameter_steps'][name] = parameter_steps[id(p)]
        final = state_inventory(model.state_dict())
        assert all(final[n] == h for n, h in frozen.items())
        stage['changed_parameter_names'] = [n for n, _ in model.named_parameters() if final[n] != initial_inventory[n]]
        assert stage['changed_parameter_names']
        stage['final_state'] = final
        stage['final_checkpoint'] = checkpoint(out/'final.pt', model, optimizer, updates, a.protocol_sha256, schedule_sha)
        model.eval()
        before = state_digest(model)
        modes = {n: m.training for n, m in model.named_modules()}
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        labels_all, boxes_all, scores_all, raw_finite = [], [], [], True
        ep = out/'endpoint'; ep.mkdir()
        with torch.no_grad():
            for i in range(size):
                image = loaded[i][0][None].cuda()
                output = model(image)
                # Preserve unchanged native sigmoid outputs, including any numerical collapse.
                raw = {k: v.detach().cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
                finite = all(np.isfinite(v).all() for v in raw.values() if v.dtype.kind == 'f')
                raw_finite = raw_finite and finite
                assert all(v.device.type == 'cuda' for v in output.values() if isinstance(v, torch.Tensor))
                assert output['pred_logits'].shape == output['pred_boxes'].shape == (1, 300, 4)
                decoded = post(output, torch.tensor([[640,512]], device='cuda'))[0]
                for k, v in decoded.items(): raw['post/'+k] = v.detach().cpu().numpy()
                np.savez_compressed(ep/('%03d.npz' % i), **raw)
                labels_all.append(raw['post/labels']); boxes_all.append(raw['post/boxes']); scores_all.append(raw['post/scores'])
        assert state_digest(model) == before and modes == {n: m.training for n, m in model.named_modules()}
        assert torch.equal(cpu_rng, torch.get_rng_state()) and equal_tree(cuda_rng, torch.cuda.get_rng_state_all())
        predictions, box_status = native_selected_records(records[:size], np.stack(labels_all), np.stack(boxes_all), np.stack(scores_all))
        save(ep/'predictions.json', predictions)
        metric, correspondences, truth = evaluate_mode(records[:size], predictions, raw_finite, box_status)
        save(ep/'ground_truth.json', truth); save(ep/'correspondences.json', correspondences); save(ep/'metrics.json', metric)
        stage['endpoint'] = metric
        stage['endpoint_state_rng_modes_unchanged'] = True
        stage['status'] = 'completed_fixed_budget_endpoint'
    except Exception:
        stage['status'] = 'failed_native_fitting_execution'
        stage['error'] = traceback.format_exc()
        if 'output' in locals():
            failed = {k: v.detach().cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
            np.savez_compressed(out/'failed_step_raw_main.npz', **failed)
        raise
    finally:
        criterion.get_loss = source_loss
        stage['seconds'] = time.time()-started
        stage['process_peak_allocated_bytes_through_stage'] = torch.cuda.max_memory_allocated()
        save(out/'REPORT.json', stage)
        manifest = {str(p.relative_to(out)): dict(sha256=sha256_file(p), bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()}
        save(out/'ARTIFACT_HASHES.json', {'files': manifest})
        del optimizer, criterion, post, model
        gc.collect(); torch.cuda.empty_cache()
    return stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','inputs','admission','protocol','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    a = p.parse_args()
    assert sha256_file(a.protocol) == a.protocol_sha256
    protocol = json.loads(a.protocol.read_text())
    assert protocol['launch_authorized'] is True, 'Prepared candidate is not a launch declaration'
    assert protocol['budgets'] == {str(k): v for k,v in BUDGETS.items()}
    assert all(sha256_file(ROOT/k) == v for k,v in protocol['files'].items())
    admitted = verify_admission(a.admission, protocol['actual_admission'], protocol)
    assert sha256_file(a.checkpoint) == protocol['checkpoint_sha256'] == CHECKPOINT_SHA256
    assert sha256_file(a.inputs/'records.json') == protocol['records_sha256']
    assert sha256_file(a.inputs/'manifest.json') == protocol['input_manifest_sha256']
    all_records = json.loads((a.inputs/'records.json').read_text())
    index = {r['identity']: r for r in all_records}; assert len(index) == len(all_records) == 7172
    splits = json.loads((ROOT/protocol['splits_path']).read_text())
    ids = splits['admission128']['sample_ids']
    assert ids == protocol['sample_ids'] and len(ids) == 128 and set(ids) <= set(splits['fit']['sample_ids'])
    records = [index[k] for k in ids]
    manifest = json.loads((a.inputs/'manifest.json').read_text())['files']
    inputs = {}
    for r in records:
        assert r['split'] == 'train'
        for field, source_key in (('image_file','image'),('label_file','label')):
            q = (a.inputs/r[field]).resolve(); q.relative_to(a.inputs.resolve())
            assert sha256_file(q) == manifest[r[field]]['sha256'] == r['sources'][source_key]['sha256']
            assert q.stat().st_size == manifest[r[field]]['bytes']
            inputs[r[field]] = sha256_file(q)
    inventory = json.loads((ROOT/protocol['source_inventory_path']).read_text())
    source_before = source_guard(a.source, inventory)
    assert sys.version_info[:2] == (3, 10)
    assert torch.__version__.split('+')[0] == '2.0.1' and torch.version.cuda == '11.8'
    import torchvision
    assert torchvision.__version__.split('+')[0] == '0.15.2'
    assert {n: importlib.metadata.version(n) for n in protocol['runtime_packages']} == protocol['runtime_packages']
    assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    author = a.source/'rtdetr_pytorch'
    sys.path.insert(0, str(author.resolve()))
    import src
    assert Path(src.__file__).resolve().is_relative_to(author.resolve())
    from src.data.coco import CocoDetection
    from src.data.transforms import Compose
    a.output.mkdir(parents=True, exist_ok=False)
    save(a.output/'protocol.json', protocol)
    coco = export_native_coco(records); save(a.output/'native_coco_all128.json', coco)
    dataset = CocoDetection(str(a.inputs), str(a.output/'native_coco_all128.json'),
                            Compose(copy.deepcopy(protocol['deterministic_transform_ops'])), return_masks=False, remap_mscoco_category=False)
    loaded, targets_raw = [], {}
    for i in range(128):
        picture, target = dataset[i]
        assert picture.dtype == torch.float32 and picture.shape == (3,640,640)
        assert torch.isfinite(picture).all() and picture.min() >= 0 and picture.max() <= 1
        expected = native_target_reference([x for x in coco['annotations'] if x['image_id'] == i+1])
        assert torch.equal(target['labels'], expected['labels'])
        assert torch.allclose(target['boxes'], expected['boxes'], rtol=0, atol=2e-7)
        loaded.append((picture, target))
        targets_raw[str(i)+'/labels'] = target['labels'].numpy()
        targets_raw[str(i)+'/boxes'] = target['boxes'].as_subclass(torch.Tensor).numpy()
    np.savez_compressed(a.output/'loaded_targets_all128.npz', **targets_raw)
    report = dict(status='running', protocol_sha256=a.protocol_sha256, source_before=source_before,
                  evaluator_before=evaluator_environment(), environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),TF32=False,precision='FP32'),
                  admission=protocol['actual_admission'], input_files=inputs, stages=[], CUDA_executed=False,
                  scope='camera-only full-native fitting on original optimization prefixes; no CURE, radar, or held-out AP')
    started = time.time()
    try:
        for size in BUDGETS:
            report['CUDA_executed'] = True
            stage = fit_stage(a, protocol, records, loaded, admitted, size)
            report['stages'].append(stage)
            save(a.output/'REPORT.json', report)
        report['status'] = 'completed_all_fixed_rtdetr_native_fitting_endpoints'
        report['all_geometry_gates_passed'] = all(x['endpoint']['gate'] for x in report['stages'])
    except Exception:
        report['status'] = 'failed_native_fitting_execution'
        report['error'] = traceback.format_exc()
    finally:
        try:
            report['evaluator_after'] = evaluator_environment()
            assert report['evaluator_after'] == report['evaluator_before']
            report['source_after'] = source_guard(a.source, inventory)
            assert report['source_after'] == source_before
            assert all(sha256_file(ROOT/k) == v for k,v in protocol['files'].items())
            assert sha256_file(a.checkpoint) == CHECKPOINT_SHA256
            assert sha256_file(a.inputs/'records.json') == protocol['records_sha256']
            assert sha256_file(a.inputs/'manifest.json') == protocol['input_manifest_sha256']
            assert all(sha256_file(a.inputs/k) == v for k,v in inputs.items())
            verify_admission(a.admission, protocol['actual_admission'], protocol)
            report['final_integrity'] = 'passed'
        except Exception:
            report['final_integrity'] = 'failed'; report['integrity_error'] = traceback.format_exc()
            report['status'] = 'failed_native_fitting_execution'
        report['seconds'] = time.time()-started
        save(a.output/'REPORT.json', report)
        payloads = {str(q.relative_to(a.output)): dict(sha256=sha256_file(q), bytes=q.stat().st_size) for q in sorted(a.output.rglob('*')) if q.is_file()}
        save(a.output/'ARTIFACT_HASHES.json', {'files': payloads})
    return 0 if report['status'] == 'completed_all_fixed_rtdetr_native_fitting_endpoints' else 1


if __name__ == '__main__': raise SystemExit(main())

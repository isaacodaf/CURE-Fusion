#!/usr/bin/env python3
"""Author-native RT-DETR four-class one-step CUDA admission, never AP or fitting."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
import torch
from cure_fusion.rtdetr_native_admission_v1 import (
    CLASS_HEADS, CLASS_KEYS, EMBED_KEY, SOURCE_COMMIT, CHECKPOINT_SHA256,
    CHECKPOINT_BYTES, TRANSFORM_OPS, checkpoint_state, export_native_coco,
    native_config_overrides, native_target_reference, select_admission_batch,
    sha256_file, state_inventory, tensor_binding, transfer_coco80_to_sew4,
    assert_loss_contract, geometry_observations)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_guard(source, inventory):
    """Pinned codeload extraction; no invented git checkout prerequisite."""
    assert inventory['commit'] == SOURCE_COMMIT
    actual = {k: sha256_file(source / k) for k in inventory['files']}
    assert actual == inventory['files'], 'Author source bytes differ'
    discovered = {str(p.relative_to(source)) for p in source.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert discovered == set(actual), 'Unexpected or missing author-tree files'
    return dict(commit=SOURCE_COMMIT, archive_sha256=inventory['archive_sha256'], files=actual)


def arrays(value, prefix='output', out=None):
    out = {} if out is None else out
    if isinstance(value, torch.Tensor):
        assert value.device.type == 'cuda', 'Tensor left CUDA: ' + prefix
        if value.is_floating_point():
            assert value.dtype == torch.float32 and torch.isfinite(value).all(), prefix
        out[prefix] = value.detach().cpu().numpy()
    elif isinstance(value, dict):
        for k, v in value.items():
            arrays(v, prefix + '/' + str(k), out)
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            arrays(v, prefix + '/' + str(k), out)
    return out


def output_geometry(output):
    views = [('main', output)]
    views += [('aux%d' % i, v) for i, v in enumerate(output.get('aux_outputs', []))]
    views += [('dn%d' % i, v) for i, v in enumerate(output.get('dn_aux_outputs', []))]
    return {name: geometry_observations(v['pred_boxes']) for name, v in views}


def gradient_audit(model):
    norms, bindings, raw = {}, {}, {}
    for name, p in model.named_parameters():
        if p.grad is None:
            bindings[name] = dict(requires_grad=p.requires_grad, gradient=None)
            continue
        g = p.grad.detach()
        assert p.requires_grad and g.device.type == 'cuda' and g.dtype == p.dtype
        assert torch.isfinite(g).all(), name
        bindings[name] = tensor_binding(g)
        norms[name] = float(g.double().square().sum().cpu())
        if name in CLASS_KEYS or name == EMBED_KEY:
            raw[name] = g.cpu().numpy()
    predicates = dict(backbone=lambda n: n.startswith('backbone.'),
                      encoder=lambda n: n.startswith('encoder.'),
                      decoder_attention=lambda n: n.startswith('decoder.decoder.'),
                      box_heads=lambda n: 'bbox_head' in n,
                      denoising_embedding=lambda n: n == EMBED_KEY)
    predicates.update({h: (lambda n, h=h: n.startswith(h + '.')) for h in CLASS_HEADS})
    groups = {k: sum(v for n, v in norms.items() if pred(n)) for k, pred in predicates.items()}
    assert all(np.isfinite(v) and v > 0 for v in groups.values()), groups
    rows = {h: [float(sum(np.square(raw[h + '.' + part][c].astype(np.float64)).sum()
                          for part in ('weight', 'bias'))) for c in range(4)] for h in CLASS_HEADS}
    assert all(v > 0 for r in rows.values() for v in r), rows
    assert np.count_nonzero(raw[EMBED_KEY][4]) == 0, 'Padding received gradient'
    return dict(groups_squared_norm=groups, class_row_squared_norms=rows,
                parameters=bindings, missing_trainable_gradients=[n for n, p in model.named_parameters()
                                                                 if p.requires_grad and p.grad is None]), raw


def optimizer_facts(optimizer, model):
    assert isinstance(optimizer, torch.optim.AdamW)
    names = {id(p): n for n, p in model.named_parameters()}
    all_ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(set(all_ids)) == len(all_ids)
    assert set(all_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert len(optimizer.param_groups) == 2
    out = []
    for group in optimizer.param_groups:
        parameter_names = [names[id(p)] for p in group['params']]
        backbone = all(n.startswith('backbone.') for n in parameter_names)
        assert backbone or all(not n.startswith('backbone.') for n in parameter_names)
        assert group['lr'] == (1e-6 if backbone else 1e-4)
        assert group['weight_decay'] == 1e-4 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
        out.append(dict(settings={k: v for k, v in group.items() if k != 'params'},
                        parameter_names=parameter_names))
    return out


def optimizer_state_binding(optimizer, model):
    names = {id(p): n for n, p in model.named_parameters()}
    return {names[id(p)]: {k: tensor_binding(v) if isinstance(v, torch.Tensor) else v
                          for k, v in state.items()} for p, state in optimizer.state.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'inputs', 'protocol', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    a = p.parse_args()
    assert sha256_file(a.protocol) == a.protocol_sha256
    protocol = json.loads(a.protocol.read_text())
    assert protocol['launch_authorized'] is True, 'Prepared source is not a CUDA launch declaration'
    assert protocol['scope'] == 'native_camera_four_class_one_step_admission_only'
    assert protocol['seed'] == 11 and protocol['optimizer_steps'] == 1
    assert protocol['deterministic_transform_ops'] == TRANSFORM_OPS
    assert protocol['clip_max_norm'] == .1
    bound = protocol['files']
    assert all(sha256_file(ROOT / k) == v for k, v in bound.items()), 'Bound candidate files differ'
    assert a.checkpoint.stat().st_size == CHECKPOINT_BYTES
    assert sha256_file(a.checkpoint) == protocol['checkpoint_sha256'] == CHECKPOINT_SHA256
    source_inventory = json.loads((ROOT / protocol['source_inventory_path']).read_text())
    source_before = source_guard(a.source, source_inventory)
    assert sha256_file(a.inputs / 'manifest.json') == protocol['input_manifest_sha256']
    assert sha256_file(a.inputs / 'records.json') == protocol['records_sha256']
    records = json.loads((a.inputs / 'records.json').read_text())
    by_id = {r['identity']: r for r in records}
    assert len(by_id) == len(records) == 7172
    splits = json.loads((ROOT / protocol['splits_path']).read_text())
    ids = splits['admission128']['sample_ids']
    assert len(set(ids)) == len(ids) == 128 and set(ids) <= set(splits['fit']['sample_ids'])
    selected = [by_id[k] for k in ids]
    assert select_admission_batch(selected) == protocol['supervised_batch_ids']
    manifests = json.loads((a.inputs / 'manifest.json').read_text())['files']
    checked_inputs = {}
    for r in selected:
        assert r['split'] == 'train'
        for field, original in (('image_file', 'image'), ('label_file', 'label')):
            q = (a.inputs / r[field]).resolve()
            q.relative_to(a.inputs.resolve())
            digest = sha256_file(q)
            assert digest == manifests[r[field]]['sha256'] == r['sources'][original]['sha256']
            assert q.stat().st_size == manifests[r[field]]['bytes']
            checked_inputs[r[field]] = digest
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', CUDA_executed=False, optimizer_steps=0, AP_evaluated=False,
                  protocol_sha256=a.protocol_sha256, checkpoint_sha256=CHECKPOINT_SHA256,
                  bound_files=bound, source_before=source_before, selected_ids=ids,
                  supervised_batch_ids=protocol['supervised_batch_ids'], input_files=checked_inputs)
    save(a.output / 'protocol.json', protocol)
    started = time.time()
    try:
        assert sys.version_info[:2] == (3, 10), 'Use the isolated declared Python3.10 environment'
        assert torch.__version__.split('+')[0] == '2.0.1' and torch.version.cuda == '11.8'
        import torchvision
        assert torchvision.__version__.split('+')[0] == '0.15.2'
        assert torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(0)
        report['environment'] = dict(python=sys.version, torch=torch.__version__,
            torchvision=torchvision.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            packages={n: importlib.metadata.version(n) for n in protocol['runtime_packages']},
            precision='FP32', tf32=False)
        assert report['environment']['packages'] == protocol['runtime_packages'], 'Runtime package pins differ'
        random.seed(11); np.random.seed(11); torch.manual_seed(11); torch.cuda.manual_seed_all(11)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        author = a.source / 'rtdetr_pytorch'
        sys.path.insert(0, str(author.resolve()))
        from src.core import YAMLConfig
        from src.data.coco import CocoDetection
        from src.data.transforms import Compose
        import src
        assert Path(src.__file__).resolve().is_relative_to(author.resolve())
        cfg = YAMLConfig(str(author / protocol['config']))
        cfg.yaml_cfg = native_config_overrides(cfg.yaml_cfg)
        assert cfg.yaml_cfg['PResNet']['depth'] == 101
        assert cfg.yaml_cfg['PResNet']['freeze_at'] == 0 and cfg.yaml_cfg['PResNet']['freeze_norm'] is True
        assert cfg.clip_max_norm == .1
        save(a.output / 'resolved_native_config.json', cfg.yaml_cfg)
        # Original modules are constructed once; never deploy/fuse/replace heads.
        model = cfg.model
        constructor = model.state_dict()
        assert len(constructor) == 991 and len(model.decoder.dec_score_head) == 6
        assert model.decoder.denoising_class_embed.padding_idx == 4
        assert model.decoder.denoising_class_embed.weight.shape == (5, 256)
        assert model.multi_scale is None
        for h in CLASS_HEADS:
            assert torch.equal(constructor[h + '.bias'], torch.full((4,), -np.log(99.), dtype=torch.float32))
        original = checkpoint_state(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
        inventory = json.loads((ROOT / protocol['checkpoint_inventory_path']).read_text())
        assert state_inventory(original) == inventory['tensors']
        novel = {k: tensor_binding(constructor[k][2:4]) for k in CLASS_KEYS + (EMBED_KEY,)}
        transferred = transfer_coco80_to_sew4(original, constructor)
        model.load_state_dict(transferred, strict=True)
        transferred_inventory = state_inventory(transferred)
        assert state_inventory(model.state_dict()) == transferred_inventory
        report['transfer'] = dict(tensors=991, class_tensors=15, unchanged_tensors=976,
            novel_constructor_rows=novel, loaded_state=transferred_inventory,
            checkpoint_padding_row=80, target_padding_row=4,
            native_COCO_forward_identity_claim=False)
        torch.save({'model': transferred, 'protocol_sha256': a.protocol_sha256}, a.output / 'initial_state.pt')
        del transferred, original, constructor
        criterion, post = cfg.criterion, cfg.postprocessor
        assert criterion.num_classes == post.num_classes == 4
        assert criterion.losses == ['vfl', 'boxes']
        assert criterion.weight_dict == dict(loss_vfl=1, loss_bbox=5, loss_giou=2)
        assert criterion.alpha == .75 and criterion.gamma == 2.
        assert criterion.matcher.cost_class == 2 and criterion.matcher.cost_bbox == 5 and criterion.matcher.cost_giou == 2
        assert criterion.matcher.alpha == .25 and criterion.matcher.gamma == 2. and criterion.matcher.use_focal_loss
        assert post.use_focal_loss and post.num_top_queries == 300 and not post.remap_mscoco_category and not post.deploy_mode
        model.cuda(); criterion.cuda(); post.cuda()
        report['CUDA_executed'] = True
        assert all(v.device.type == 'cuda' for v in list(model.parameters()) + list(model.buffers()))
        optimizer = cfg.optimizer
        assert len(optimizer.state) == 0
        report['optimizer_groups'] = optimizer_facts(optimizer, model)
        coco = export_native_coco(selected)
        coco_path = a.output / 'admission128_native_coco.json'; save(coco_path, coco)
        dataset = CocoDetection(str(a.inputs), str(coco_path), Compose(copy.deepcopy(TRANSFORM_OPS)),
                                return_masks=False, remap_mscoco_category=False)
        target_arrays, loaded = {}, {}
        for i, row in enumerate(selected):
            picture, target = dataset[i]
            assert picture.shape == (3, 640, 640) and picture.dtype == torch.float32
            assert torch.isfinite(picture).all() and picture.min() >= 0 and picture.max() <= 1
            annotations = [x for x in coco['annotations'] if x['image_id'] == i + 1]
            expected = native_target_reference(annotations)
            assert torch.equal(target['labels'], expected['labels'])
            assert torch.allclose(target['boxes'], expected['boxes'], rtol=0, atol=2e-7)
            assert torch.equal(target['orig_size'], torch.tensor([640, 512]))
            target_arrays[str(i) + '/boxes'] = target['boxes'].as_subclass(torch.Tensor).numpy()
            target_arrays[str(i) + '/labels'] = target['labels'].numpy()
            loaded[row['identity']] = (picture, target)
        np.savez_compressed(a.output / 'loaded_targets_all128.npz', **target_arrays)
        report['all128_target_counts'] = [sum(int((v == c).sum()) for k, v in target_arrays.items() if k.endswith('/labels')) for c in range(4)]
        assert report['all128_target_counts'] == [118, 57, 33, 37]
        batch = [loaded[k] for k in protocol['supervised_batch_ids']]
        images = torch.stack([im.as_subclass(torch.Tensor) for im, _ in batch]).cuda()
        targets = [{k: v.as_subclass(torch.Tensor).cuda() for k, v in t.items() if isinstance(v, torch.Tensor)} for _, t in batch]
        assert any(len(t['labels']) == 0 for t in targets)
        assert all(any((t['labels'] == c).any() for t in targets) for c in range(4))
        frozen = {n: tensor_binding(p) for n, p in model.named_parameters() if not p.requires_grad}
        native_get_loss, observed = criterion.get_loss, []
        def recorded_get_loss(*args, **kwargs):
            result = native_get_loss(*args, **kwargs)
            for key, value in result.items():
                assert value.device.type == 'cuda' and value.numel() == 1 and torch.isfinite(value).all(), key
            observed.append(dict(family=args[0], values={k: float(v.detach().cpu()) for k, v in result.items()}))
            return result
        criterion.get_loss = recorded_get_loss
        model.train(); criterion.train(); optimizer.zero_grad(set_to_none=True)
        output = model(images, targets=targets)
        assert output['pred_logits'].shape == (len(batch), 300, 4) and output['pred_boxes'].shape == (len(batch), 300, 4)
        assert len(output['aux_outputs']) == 6 and len(output['dn_aux_outputs']) == 6
        raw = arrays(output)
        np.savez_compressed(a.output / 'mixed_forward_before_geometry_check.npz', **raw)
        geometry = output_geometry(output)
        # Raw sigmoid alone is not a guarantee against floating-point corner collapse.
        assert all(v['raw_nonpositive_dimensions'] == v['corner_nonpositive_dimensions'] == 0
                   and v['outside_unit_interval'] == 0 for v in geometry.values()), geometry
        output['pred_logits'].retain_grad(); output['pred_boxes'].retain_grad()
        matches = criterion.matcher(output, targets)
        for t, (_, target_ids) in zip(targets, matches):
            assert len(target_ids) == len(t['labels']) and sorted(target_ids.cpu().tolist()) == list(range(len(t['labels'])))
        losses = criterion(output, targets)
        assert_loss_contract(losses, [v['family'] for v in observed], True)
        total = sum(losses.values())
        assert total.requires_grad and torch.isfinite(total) and total > 0
        total.backward()
        ga, gradients = gradient_audit(model)
        raw.update({'gradient/' + k: v for k, v in gradients.items()})
        raw['gradient/main_logits'] = output['pred_logits'].grad.cpu().numpy()
        raw['gradient/main_boxes'] = output['pred_boxes'].grad.cpu().numpy()
        class_counts, positives, empty_rows = [], {}, []
        for c in range(4):
            count, box_abs, cls_abs = 0, 0., 0.
            for i, (query, target_ids) in enumerate(matches):
                # Native matcher returns CPU indices; move both before CUDA masks.
                query, target_ids = query.to('cuda'), target_ids.to('cuda')
                chosen = targets[i]['labels'][target_ids] == c
                query = query[chosen]
                count += len(query)
                box_abs += float(output['pred_boxes'].grad[i, query].abs().sum().cpu())
                cls_abs += float(output['pred_logits'].grad[i, query, c].abs().sum().cpu())
            assert count > 0 and box_abs > 0 and cls_abs > 0
            class_counts.append(count)
            positives[str(c)] = dict(objects=count, box_abs=box_abs, class_abs=cls_abs)
        for i, target in enumerate(targets):
            if len(target['labels']) == 0:
                g, bg = output['pred_logits'].grad[i], output['pred_boxes'].grad[i]
                assert torch.isfinite(g).all() and (g >= 0).all() and (g.sum(0) > 0).all()
                assert torch.equal(bg, torch.zeros_like(bg))
                empty_rows.append(dict(identity=protocol['supervised_batch_ids'][i], class_gradient=g.sum(0).cpu().tolist(), box_gradient_zero=True))
        save(a.output / 'mixed_gradient_inventory.json', ga)
        np.savez_compressed(a.output / 'mixed_backward.npz', **raw)
        report['mixed_supervised_backward'] = dict(loss=float(total.detach().cpu()),
            components={k: float(v.detach().cpu()) for k, v in losses.items()}, raw_loss_calls=copy.deepcopy(observed),
            geometry=geometry, class_counts=class_counts, positive_output_gradients=positives,
            empty_image_gradients=empty_rows, groups_squared_norm=ga['groups_squared_norm'],
            class_row_squared_norms=ga['class_row_squared_norms'],
            matches=[dict(query_ids=q.cpu().tolist(), target_ids=j.cpu().tolist()) for q, j in matches])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
        report['unclipped_gradient_norm'] = float(norm.cpu())
        optimizer.step(); report['optimizer_steps'] = 1
        assert set(optimizer.state) == {p for p in model.parameters() if p.grad is not None}
        assert all(s['step'].ndim == 0 and torch.isfinite(s['step']) and s['step'].item() == 1 for s in optimizer.state.values())
        after = state_inventory(model.state_dict())
        changed = [n for n, p in model.named_parameters() if after[n] != transferred_inventory[n]]
        assert changed and all(after[n] == value for n, value in frozen.items())
        report['updated_parameter_names'] = changed
        report['after_one_step_state'] = after
        report['after_one_step_optimizer_state'] = optimizer_state_binding(optimizer, model)
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(), protocol_sha256=a.protocol_sha256), a.output / 'after_one_step.pt')
        readback = torch.load(a.output / 'after_one_step.pt', map_location='cpu', weights_only=True)
        assert state_inventory(readback['model']) == after
        assert readback['protocol_sha256'] == a.protocol_sha256
        assert readback['optimizer']['param_groups'] == optimizer.state_dict()['param_groups']
        for pid, state in optimizer.state_dict()['state'].items():
            for key, value in state.items():
                other = readback['optimizer']['state'][pid][key]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value.cpu(), other) and value.dtype == other.dtype
                else:
                    assert value == other
        del output, losses, total, raw, gradients, readback
        # All128 in fixed batches decode the exact saved one-step model before
        # empty-diagnostic BN updates. No AP or threshold selection is performed.
        model.eval()
        decode_index = []
        inference_batch_size = protocol['all128_decode_batch_size']
        assert inference_batch_size == 8
        with torch.no_grad():
            for start in range(0, 128, inference_batch_size):
                batch_ids = ids[start:start + inference_batch_size]
                all_images = torch.stack([loaded[k][0].as_subclass(torch.Tensor) for k in batch_ids]).cuda()
                output = model(all_images)
                raw = arrays(output)
                raw_name = 'native128_forward_%03d.npz' % start
                np.savez_compressed(a.output / raw_name, **raw)
                assert output['pred_logits'].shape == (len(batch_ids), 300, 4)
                assert output['pred_boxes'].shape == (len(batch_ids), 300, 4)
                decoded = post(output, torch.tensor([[640, 512]] * len(batch_ids), device='cuda'))
                geo = output_geometry(output)
                assert all(v['corner_nonpositive_dimensions'] == v['raw_nonpositive_dimensions'] == 0
                           and v['outside_unit_interval'] == 0 for v in geo.values()), geo
                assert len(decoded) == len(batch_ids)
                for i, d in enumerate(decoded):
                    assert d['scores'].shape == (300,) and d['boxes'].shape == (300, 4) and d['labels'].shape == (300,)
                    assert ((d['labels'] >= 0) & (d['labels'] < 4)).all()
                    assert ((d['boxes'][:, 2:] - d['boxes'][:, :2]) > 0).all()
                    score, index = output['pred_logits'][i].sigmoid().flatten().topk(300)
                    q = index // 4
                    box = output['pred_boxes'][i]
                    xyxy = torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], 1)
                    xyxy *= torch.tensor([640, 512, 640, 512], device='cuda')
                    assert torch.equal(d['scores'], score) and torch.equal(d['labels'], index % 4)
                    assert torch.equal(d['boxes'], xyxy[q])
                    raw.update(arrays(d, 'post/%d' % i))
                    decode_index.append(dict(identity=batch_ids[i], dataset_image_id=start+i+1,
                        file='native128_decode_%03d.npz' % start, batch_row=i, detections=300,
                        raw_geometry=geometry_observations(output['pred_boxes'][i]),
                        selected_min_corner_extent=float((d['boxes'][:, 2:] - d['boxes'][:, :2]).min().cpu())))
                np.savez_compressed(a.output / ('native128_decode_%03d.npz' % start), **raw)
                del output, decoded, raw, all_images
        assert [r['identity'] for r in decode_index] == ids and len(decode_index) == 128
        assert len({r['identity'] for r in decode_index}) == 128
        save(a.output / 'native128_decode_index.json', decode_index)
        report['one_step_native_decode'] = dict(images=128, batches=16, batch_size=8,
            detections_per_image=300, exact_original_postprocessor=True, all_ids_exact=True,
            raw_and_selected_positive_geometry=True, AP_evaluated=False,
            index_sha256=sha256_file(a.output / 'native128_decode_index.json'))
        assert state_inventory(model.state_dict()) == after
        # Backward-only all-empty native training path. Original train-mode BN
        # running buffers may advance; record them. Parameters/optimizer must not.
        empty_idx = [i for i, t in enumerate(targets) if len(t['labels']) == 0]
        empty_images = images[empty_idx]
        empty_targets = [targets[i] for i in empty_idx]
        model.train(); observed.clear(); optimizer.zero_grad(set_to_none=True)
        output = model(empty_images, targets=empty_targets)
        assert len(output['aux_outputs']) == 6 and 'dn_aux_outputs' not in output
        output['pred_logits'].retain_grad(); output['pred_boxes'].retain_grad()
        raw = arrays(output)
        np.savez_compressed(a.output / 'empty_only_forward_before_loss.npz', **raw)
        losses = criterion(output, empty_targets)
        assert_loss_contract(losses, [v['family'] for v in observed], False)
        assert all(float(v.detach().cpu()) == 0 for k, v in losses.items() if 'bbox' in k or 'giou' in k)
        total = sum(losses.values())
        assert total.requires_grad and torch.isfinite(total) and total > 0
        total.backward()
        g, bg = output['pred_logits'].grad, output['pred_boxes'].grad
        assert g is not None and bg is not None and torch.isfinite(g).all() and (g >= 0).all()
        assert (g.sum((0, 1)) > 0).all() and torch.equal(bg, torch.zeros_like(bg))
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
        raw['gradient/main_logits'] = g.cpu().numpy(); raw['gradient/main_boxes'] = bg.cpu().numpy()
        np.savez_compressed(a.output / 'empty_only_backward.npz', **raw)
        final_state = state_inventory(model.state_dict())
        assert all(final_state[n] == after[n] for n, _ in model.named_parameters())
        assert all(s['step'].item() == 1 for s in optimizer.state.values())
        assert optimizer_state_binding(optimizer, model) == report['after_one_step_optimizer_state']
        report['empty_only_backward'] = dict(identities=[protocol['supervised_batch_ids'][i] for i in empty_idx],
            optimizer_steps=0, components={k: float(v.detach().cpu()) for k, v in losses.items()},
            raw_loss_calls=copy.deepcopy(observed), class_gradient=g.sum((0, 1)).cpu().tolist(),
            box_gradient_present_zero=True, parameters_unchanged=True,
            training_forward_changed_buffers=[k for k in final_state if final_state[k] != after[k]],
            post_empty_state=final_state, saved_one_step_checkpoint_precedes_empty_diagnostic=True)
        criterion.get_loss = native_get_loss
        report['status'] = 'passed_rtdetr_native_four_class_one_step_admission_only'
    except Exception:
        report['status'] = 'failed_rtdetr_native_four_class_one_step_admission'
        report['error'] = traceback.format_exc()
    finally:
        try:
            report['source_after'] = source_guard(a.source, source_inventory)
            assert report['source_after'] == report['source_before']
            assert sha256_file(a.checkpoint) == CHECKPOINT_SHA256
            assert all(sha256_file(ROOT / k) == v for k, v in bound.items())
        except Exception:
            report['status'] = 'failed_rtdetr_native_four_class_one_step_admission'
            report['source_integrity_error'] = traceback.format_exc()
        report['elapsed_seconds'] = time.time() - started
        save(a.output / 'REPORT.json', report)
        manifest = {str(f.relative_to(a.output)): dict(bytes=f.stat().st_size, sha256=sha256_file(f))
                    for f in sorted(a.output.rglob('*')) if f.is_file() and f.name != 'ARTIFACT_HASHES.json'}
        save(a.output / 'ARTIFACT_HASHES.json', manifest)
    print(json.dumps(dict(status=report['status'], optimizer_steps=report['optimizer_steps'], AP_evaluated=False)))
    if not report['status'].startswith('passed_'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

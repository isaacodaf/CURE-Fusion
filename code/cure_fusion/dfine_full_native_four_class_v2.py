"""Strict COCO80 -> SEW4 transfer and data contracts; no detector construction.

Novel rows retain the seeded native four-class constructor initialization.
The denoising padding row is distinct from the four sigmoid foreground classes.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

CLASSES = ('person', 'bicycle', 'slidecar', 'doll')
CLASS_HEADS = ('decoder.enc_score_head',) + tuple(
    'decoder.dec_score_head.%d' % i for i in range(6))
CLASS_KEYS = tuple(h + '.' + s for h in CLASS_HEADS for s in ('weight', 'bias'))
EMBED_KEY = 'decoder.denoising_class_embed.weight'
TRANSFER_KEYS = CLASS_KEYS + (EMBED_KEY,)
SOURCE_COMMIT = '956d1709314c2c6a4df6f34de232054578a7449f'
CHECKPOINT_SHA256 = '39361855c21f8c4757a9bb777a7ff134ae8aa110d0db67a374f081e44c2ce04b'


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def tensor_binding(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError('State entry is not a tensor')
    a = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def state_inventory(state):
    return {k: tensor_binding(v) for k, v in sorted(state.items())}


def transfer_coco80_to_sew4(source, initialized):
    """Return complete state; require every non-class tensor to match exactly.

    Both inputs are CPU state dictionaries, never modules. Keys, dtypes and all
    dimensions are checked before copying. No shape-only/strict=False fallback.
    """
    if set(source) != set(initialized) or not set(TRANSFER_KEYS) <= set(source):
        raise ValueError('Incomplete or unexpected checkpoint keys')
    result = {}
    for key in source:
        old, new = source[key], initialized[key]
        if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
            raise ValueError('Non-tensor checkpoint entry: ' + key)
        if old.device.type != 'cpu' or new.device.type != 'cpu' or old.dtype != new.dtype:
            raise ValueError('CPU/dtype mismatch: ' + key)
        if key == 'decoder.anchors':
            # The original generator deliberately stores +inf for invalid
            # anchors. Stored checkpoint finite values need not equal freshly
            # generated values. Load stored bytes exactly, as for other buffers.
            mask = source.get('decoder.valid_mask')
            fresh_mask = initialized.get('decoder.valid_mask')
            if (old.shape != (1, 8400, 4) or new.shape != old.shape or
                    mask is None or fresh_mask is None or mask.dtype != torch.bool or
                    mask.shape != (1, 8400, 1) or not torch.equal(mask, fresh_mask) or
                    old.dtype != torch.float32 or
                    not torch.equal(torch.isposinf(old), (~mask).expand_as(old)) or
                    not torch.equal(torch.isposinf(new), (~mask).expand_as(new)) or
                    not torch.isfinite(old[mask.expand_as(old)]).all() or
                    not torch.isfinite(new[mask.expand_as(new)]).all()):
                raise ValueError('Native anchor sentinel/generator contract differs')
        elif (old.is_floating_point() and not torch.isfinite(old).all()) or (
                new.is_floating_point() and not torch.isfinite(new).all()):
            raise ValueError('Nonfinite checkpoint entry: ' + key)
        if key in CLASS_KEYS:
            tail = (256,) if key.endswith('.weight') else ()
            if old.shape != (80,) + tail or new.shape != (4,) + tail or old.dtype != torch.float32:
                raise ValueError('Wrong native classifier shape/dtype: ' + key)
            value = new.detach().clone()
            value[:2].copy_(old[:2])
        elif key == EMBED_KEY:
            if old.shape != (81, 256) or new.shape != (5, 256) or old.dtype != torch.float32:
                raise ValueError('Wrong native denoising embedding')
            if torch.count_nonzero(old[80]) or torch.count_nonzero(new[4]):
                raise ValueError('Denoising padding must be zero')
            value = new.detach().clone()
            value[:2].copy_(old[:2])
            value[4].copy_(old[80])
        else:
            if old.shape != new.shape:
                raise ValueError('Non-class architecture mismatch: ' + key)
            value = old.detach().clone()
        result[key] = value
    return result


def native_config_overrides(base):
    """Keep authored custom-X architecture, optimizer and freeze policy."""
    cfg = copy.deepcopy(base)
    cfg['num_classes'] = 4
    cfg['remap_mscoco_category'] = False
    cfg['HGNetv2']['pretrained'] = False
    # Admission uses an explicit native FP32 step, not the full training solver.
    cfg['use_amp'] = False
    cfg['use_ema'] = False
    return cfg


def select_admission_batch(records):
    """First class-supporting members, deduplicated, then first empty member."""
    selected = []
    for c in range(4):
        row = next((r for r in records if c in r['target']['labels']), None)
        if row is None:
            raise ValueError('Missing fitting class support')
        if row['identity'] not in selected:
            selected.append(row['identity'])
    empty = next((r for r in records if not r['target']['labels']), None)
    if empty is None:
        raise ValueError('Missing fitting empty image')
    selected.append(empty['identity'])
    return selected


def export_native_coco(records, width=640, height=512):
    """Native remap=False consumes category_id verbatim: categories are 0..3.

    Preserve original normalized boxes in `source_target`; write pixel XYWH
    without preclipping. The actual native dataset clips them on load. Known
    annotations outside the image that would disappear are rejected here.
    """
    out = dict(info={'description': 'SEW fitting-only native D-FINE admission'},
               images=[], annotations=[], categories=[dict(id=i, name=n) for i, n in enumerate(CLASSES)])
    identities = set()
    for image_id, row in enumerate(records, 1):
        if row['identity'] in identities or row.get('split') != 'train':
            raise ValueError('Duplicate or non-training identity')
        identities.add(row['identity'])
        path = Path(row['image_file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe image filename')
        labels = np.asarray(row['target']['labels'])
        boxes = np.asarray(row['target']['boxes'], dtype=np.float64).reshape(-1, 4)
        if labels.shape != (len(boxes),) or (len(labels) and labels.dtype.kind not in 'iu'):
            raise ValueError('Invalid class vector')
        if np.any(labels < 0) or np.any(labels > 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
            raise ValueError('Malformed four-class targets')
        out['images'].append(dict(id=image_id, file_name=str(path), width=width, height=height,
                                  identity=row['identity'], source_target=copy.deepcopy(row['target'])))
        for label, box in zip(labels, boxes):
            x, y, w, h = (box * np.array([width, height, width, height])).tolist()
            x, y = x - w / 2, y - h / 2
            if min(x + w, width) <= max(x, 0) or min(y + h, height) <= max(y, 0):
                raise ValueError('Native clipping would erase a known object')
            out['annotations'].append(dict(id=len(out['annotations']) + 1, image_id=image_id,
                                            category_id=int(label), bbox=[x, y, w, h],
                                            area=w*h, iscrowd=0))
    return out


def native_target_reference(annotations, width=640, height=512):
    """Tensor-only replay of the pinned parser and deterministic resize/normalize.

    FP32 pixel arithmetic is retained. No tolerance or GT replacement is used
    by the training runner: this is an independent comparison carrier.
    """
    b = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32).reshape(-1, 4)
    b[:, 2:] += b[:, :2]
    b[:, 0::2].clamp_(0, width)
    b[:, 1::2].clamp_(0, height)
    if len(b) and not ((b[:, 2:] - b[:, :2]) > 0).all():
        raise ValueError('Native parser would remove a known annotation')
    # torchvision resize boxes to 640x640, then source ConvertBoxes.
    b *= torch.tensor([640/width, 640/height, 640/width, 640/height], dtype=torch.float32)
    xy = (b[:, :2] + b[:, 2:]) / 2
    wh = b[:, 2:] - b[:, :2]
    return dict(boxes=torch.cat([xy, wh], 1) / 640,
                labels=torch.tensor([a['category_id'] for a in annotations], dtype=torch.int64))

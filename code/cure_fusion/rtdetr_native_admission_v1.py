"""RT-DETR native COCO80 -> SEW4 contracts; no detector construction.

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
SOURCE_COMMIT = '29320b6fd828f8e0987a71426cf2d961b09dfed7'
CHECKPOINT_SHA256 = 'db3d85e3e9787daa63586284cba77bd98d0836aaf76e07b4b16aeb87d370ef00'
CHECKPOINT_BYTES = 307609288
NATIVE_TENSOR_COUNT = 991
TRANSFORM_OPS = [dict(type='Resize', size=[640, 640]),
                 dict(type='ToImageTensor'), dict(type='ConvertDtype'),
                 dict(type='ConvertBox', out_fmt='cxcywh', normalize=True)]


def checkpoint_state(checkpoint):
    """The downloaded author release has EMA/module, including actual padding80."""
    if not isinstance(checkpoint, dict) or set(checkpoint) != {'ema'}:
        raise ValueError('Unexpected released checkpoint container')
    ema = checkpoint['ema']
    if not isinstance(ema, dict) or set(ema) != {'module', 'updates', 'warmups'}:
        raise ValueError('Unexpected EMA checkpoint container')
    if ema['updates'] != -1 or ema['warmups'] != -1:
        raise ValueError('Bound converted-release metadata differs')
    state = ema['module']
    if not isinstance(state, dict) or len(state) != NATIVE_TENSOR_COUNT:
        raise ValueError('Bound native checkpoint tensor count differs')
    return state


def native_config_overrides(base):
    """Only class/data adaptation and fixed-size FP32 admission overrides."""
    cfg = copy.deepcopy(base)
    cfg['num_classes'] = 4
    cfg['remap_mscoco_category'] = False
    cfg['PResNet']['pretrained'] = False  # Full checkpoint supplies backbone.
    cfg['RTDETR']['multi_scale'] = None  # Declared deterministic 640 admission.
    cfg['use_amp'] = False
    cfg['use_ema'] = False
    return cfg


def expected_loss_keys(with_denoising):
    base = ('loss_vfl', 'loss_bbox', 'loss_giou')
    suffixes = [''] + ['_aux_%d' % i for i in range(6)]
    if with_denoising:
        suffixes += ['_dn_%d' % i for i in range(6)]
    return {key + suffix for suffix in suffixes for key in base}


def assert_loss_contract(losses, loss_calls, with_denoising):
    expected_calls = ['vfl', 'boxes'] * (13 if with_denoising else 7)
    if set(losses) != expected_loss_keys(with_denoising):
        raise ValueError('Native main/aux/denoising loss coverage differs')
    if loss_calls != expected_calls:
        raise ValueError('Native unsanitized loss-call sequence differs')
    for k, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError('Invalid scalar native loss: ' + k)


def geometry_observations(boxes):
    """Observe actual FP32 corners without clipping or changing raw boxes."""
    if boxes.shape[-1] != 4 or boxes.dtype != torch.float32 or not torch.isfinite(boxes).all():
        raise ValueError('Invalid finite FP32 cxcywh tensor')
    lo, hi = boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2
    extent = hi - lo
    return dict(boxes=int(boxes.numel() // 4),
                raw_nonpositive_dimensions=int((boxes[..., 2:] <= 0).sum().item()),
                corner_nonpositive_dimensions=int((extent <= 0).sum().item()),
                min_raw_dimension=float(boxes[..., 2:].min().item()),
                min_corner_extent=float(extent.min().item()),
                outside_unit_interval=int(((boxes < 0) | (boxes > 1)).sum().item()))


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
    if 'decoder.anchors' in source or 'decoder.valid_mask' in source:
        raise ValueError('RT-DETR native anchors are plain attributes, not checkpoint tensors')
    if set(source) != set(initialized) or not set(TRANSFER_KEYS) <= set(source):
        raise ValueError('Incomplete or unexpected checkpoint keys')
    result = {}
    for key in source:
        old, new = source[key], initialized[key]
        if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
            raise ValueError('Non-tensor checkpoint entry: ' + key)
        if old.device.type != 'cpu' or new.device.type != 'cpu' or old.dtype != new.dtype:
            raise ValueError('CPU/dtype mismatch: ' + key)
        if (old.is_floating_point() and not torch.isfinite(old).all()) or (
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
    out = dict(info={'description': 'SEW fitting-only native RT-DETR admission'},
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
    # torchvision resize boxes to 640x640, then source ConvertBox.
    b *= torch.tensor([640/width, 640/height, 640/width, 640/height], dtype=torch.float32)
    xy = (b[:, :2] + b[:, 2:]) / 2
    wh = b[:, 2:] - b[:, :2]
    return dict(boxes=torch.cat([xy, wh], 1) / 640,
                labels=torch.tensor([a['category_id'] for a in annotations], dtype=torch.int64))

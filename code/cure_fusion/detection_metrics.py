"""Local SEW 2D detection protocol using the reference COCO evaluator on CPU.

This is not an official SEW leaderboard protocol. Do not pool scores with the
auxiliary class-presence endpoint or with published 3D detection metrics.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.metadata
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ("person", "bicycle", "slidecar", "doll")


def decode_queries(identities, logits, boxes):
    """One top foreground class per query, retaining background probability.

    No confidence threshold or NMS; all queries remain eligible for COCO's
    score ordering. Normalized cxcywh predictions are clipped at evaluation.
    """
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    if (logits.ndim != 3 or logits.shape[-1] != len(CLASSES)+1
            or boxes.shape != (*logits.shape[:2], 4) or len(identities) != len(logits)
            or len(set(identities)) != len(identities)
            or not np.isfinite(logits).all() or not np.isfinite(boxes).all()):
        raise ValueError("invalid query predictions or image identities")
    shifted = logits.astype(np.float64) - logits.max(-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(-1, keepdims=True)
    foreground = probabilities[..., :-1]
    return [{"identity": identity, "boxes": boxes[i], "labels": foreground[i].argmax(-1),
             "scores": foreground[i].max(-1)} for i, identity in enumerate(identities)]


def box_array(boxes):
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
            or (boxes < 0).any() or (boxes > 1).any()):
        raise ValueError("normalized finite cxcywh boxes in [0,1] required")
    return boxes


def class_array(labels, count):
    labels = np.asarray(labels)
    if (labels.shape != (count,) or not np.isfinite(labels).all()
            or (labels != labels.astype(int)).any() or (labels < 0).any()
            or (labels >= len(CLASSES)).any()):
        raise ValueError("invalid detection class labels")
    return labels.astype(int)


def absolute_xywh(box, image_size, clip):
    low, high = box[:2] - box[2:]/2, box[:2] + box[2:]/2
    if clip:
        low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
    scale = np.asarray(image_size)
    return np.concatenate((low * scale, (high-low)*scale)).tolist()


def evaluate_detection(records, predictions, *, image_size=(640, 512)):
    """Compute COCO AP.50:.95/101 recalls/maxDets=100 over exactly joined images.

    Ground-truth extents are preserved; prediction corners clip to image bounds.
    Missing ground-truth classes yield None, not zero. All zero-object images
    remain in evaluation. Range slices require a separate ignore-aware protocol.
    """
    ids = [record["identity"] for record in records]
    pred_ids = [prediction["identity"] for prediction in predictions]
    if (not ids or len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids)
            or set(ids) != set(pred_ids)):
        raise ValueError("unique and exactly matching image identities required")
    if len(image_size) != 2 or not np.isfinite(image_size).all() or min(image_size) <= 0:
        raise ValueError("positive image width and height required")
    pred_by_id = {pred["identity"]: pred for pred in predictions}
    images, annotations, detections = [], [], []
    for image_id, record in enumerate(records, start=1):
        images.append({"id": image_id, "width": image_size[0], "height": image_size[1]})
        boxes = box_array(record["target"]["boxes"])
        labels = class_array(record["target"]["labels"], len(boxes))
        if len(boxes) and (boxes[:, 2:] <= 0).any():
            raise ValueError("ground truth must have positive area")
        for box, label in zip(boxes, labels, strict=True):
            xywh = absolute_xywh(box, image_size, clip=False)
            annotations.append({"id": len(annotations)+1, "image_id": image_id,
                                "category_id": int(label)+1, "bbox": xywh,
                                "area": xywh[2]*xywh[3], "iscrowd": 0})
        prediction = pred_by_id[record["identity"]]
        boxes = box_array(prediction["boxes"])
        labels = class_array(prediction["labels"], len(boxes))
        scores = np.asarray(prediction["scores"], dtype=float)
        if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("finite scores in [0,1] required for every prediction")
        for box, label, score in zip(boxes, labels, scores, strict=True):
            detections.append({"image_id": image_id, "category_id": int(label)+1,
                               "bbox": absolute_xywh(box, image_size, clip=True), "score": float(score)})
    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"info": {}, "images": images, "annotations": annotations,
                         "categories": [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]}
        truth.createIndex()
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**truth.dataset, "annotations": []}
            predicted.createIndex()
        evaluation = COCOeval(truth, predicted, "bbox")
        evaluation.params.imgIds = [image["id"] for image in images]
        evaluation.params.catIds = list(range(1, len(CLASSES)+1))
        evaluation.params.iouThrs = np.linspace(.5, .95, 10)
        evaluation.params.recThrs = np.linspace(0, 1, 101)
        evaluation.params.maxDets = [1, 10, 100]
        evaluation.evaluate()
        evaluation.accumulate()
    precision = evaluation.eval["precision"][:, :, :, 0, 2]
    def mean_valid(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else None
    return {"protocol": "local_SEW_2D_COCO_AP_v1", "device": "cpu",
            "pycocotools_version": importlib.metadata.version("pycocotools"),
            "images": len(records), "objects": len(annotations), "predictions": len(detections),
            "sequences": len({row["sequence"] for row in records}),
            "AP": mean_valid(precision), "AP50": mean_valid(precision[0]),
            "AP75": mean_valid(precision[5]),
            "per_class_AP": {name: mean_valid(precision[:, :, i]) for i, name in enumerate(CLASSES)},
            "iou_thresholds": evaluation.params.iouThrs.tolist(), "recall_points": 101,
            "max_detections_per_image_per_category": 100,
            "ground_truth_clipping": False, "prediction_clipping": True,
            "confidence_threshold": None, "nms": False, "official_SEW_leaderboard": False}

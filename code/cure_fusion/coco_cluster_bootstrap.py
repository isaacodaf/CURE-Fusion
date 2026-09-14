"""Exact COCO AP under integer image replication, with sequence-level draws.

IoU matching is local to each image and can be reused. Equal-score detections
within a repeated image must repeat as a block, not independently by detection.
This module never treats dependent frames as independent resampling units.
"""
from contextlib import redirect_stdout
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .detection_metrics import CLASSES, box_array, class_array, absolute_xywh


def replicate_tied_blocks(starts, lengths, image_ids, weights):
    """Indices of the canonical replicated score order, preserving within-image ties."""
    repetitions = lengths*weights[image_ids]
    groups = np.repeat(np.arange(len(starts)), repetitions)
    if not len(groups):
        return np.empty(0, dtype=np.int64)
    expanded_starts = np.cumsum(repetitions)-repetitions
    return starts[groups]+(np.arange(len(groups))-expanded_starts[groups]) % lengths[groups]


class ClusterCOCOAP:
    def __init__(self, records, predictions, image_size=(640, 512)):
        identities = [r["identity"] for r in records]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("unique nonempty image identities required")
        predicted_ids = [p["identity"] for p in predictions]
        if len(set(predicted_ids)) != len(predicted_ids) or set(predicted_ids) != set(identities):
            raise ValueError("predictions must join every image exactly")
        by_id = {p["identity"]: p for p in predictions}
        images, annotations, detections = [], [], []
        for i, row in enumerate(records, start=1):
            images.append({"id": i, "width": image_size[0], "height": image_size[1]})
            boxes = box_array(row["target"]["boxes"])
            labels = class_array(row["target"]["labels"], len(boxes))
            if len(boxes) and (boxes[:, 2:] <= 0).any():
                raise ValueError("positive ground-truth extents required")
            for box, label in zip(boxes, labels, strict=True):
                xywh = absolute_xywh(box, image_size, False)
                annotations.append({"id": len(annotations)+1, "image_id": i, "category_id": int(label)+1,
                                    "bbox": xywh, "area": xywh[2]*xywh[3], "iscrowd": 0})
            prediction = by_id[row["identity"]]
            boxes = box_array(prediction["boxes"])
            labels = class_array(prediction["labels"], len(boxes))
            scores = np.asarray(prediction["scores"], dtype=float)
            if scores.shape != (len(boxes),) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError("finite probability scores required")
            for box, label, score in zip(boxes, labels, scores, strict=True):
                detections.append({"image_id": i, "category_id": int(label)+1, "bbox": absolute_xywh(box, image_size, True), "score": float(score)})
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
            evaluation.params.imgIds = list(range(1, len(records)+1))
            evaluation.params.catIds = list(range(1, len(CLASSES)+1))
            evaluation.params.iouThrs = np.linspace(.5, .95, 10)
            evaluation.params.recThrs = np.linspace(0, 1, 101)
            evaluation.params.areaRng = [[0, 1e10]]
            evaluation.params.maxDets = [100]
            evaluation.evaluate()
        self.images = len(records)
        self.sequences = [str(r["sequence"]) for r in records]
        self.classes = []
        for category in range(len(CLASSES)):
            values = evaluation.evalImgs[category*self.images:(category+1)*self.images]
            gt = np.array([int((~v["gtIgnore"].astype(bool)).sum()) if v is not None else 0 for v in values])
            scores, image_ids, tp, fp = [], [], [], []
            for i, value in enumerate(values):
                if value is None:
                    continue
                scores.extend(value["dtScores"])
                image_ids.extend([i]*len(value["dtScores"]))
                tp.append((value["dtMatches"] > 0) & ~value["dtIgnore"])
                fp.append((value["dtMatches"] == 0) & ~value["dtIgnore"])
            score = np.asarray(scores)
            image_ids = np.asarray(image_ids, dtype=np.int64)
            order = np.argsort(-score, kind="stable")
            score, image_ids = score[order], image_ids[order]
            tp = np.concatenate(tp, axis=1)[:, order] if tp else np.zeros((10, 0), bool)
            fp = np.concatenate(fp, axis=1)[:, order] if fp else np.zeros((10, 0), bool)
            starts = np.flatnonzero(np.r_[True, (score[1:] != score[:-1]) | (image_ids[1:] != image_ids[:-1])]) if len(score) else np.empty(0, dtype=np.int64)
            lengths = np.diff(np.r_[starts, len(score)]) if len(score) else np.empty(0, dtype=np.int64)
            self.classes.append({"gt": gt, "tp": tp, "fp": fp, "starts": starts, "lengths": lengths, "images": image_ids[starts]})

    def subset(self, indices):
        """Reuse per-image matches for an ordered metadata subset, without new IoUs."""
        indices = np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or not len(indices) or (indices < 0).any() or (indices >= self.images).any()
                or not np.array_equal(indices, np.unique(indices))):
            raise ValueError("nonempty sorted unique image indices required")
        result = object.__new__(ClusterCOCOAP)
        result.images = len(indices)
        result.sequences = [self.sequences[i] for i in indices]
        result.classes = []
        remap = np.full(self.images, -1, dtype=np.int64)
        remap[indices] = np.arange(len(indices))
        for category in self.classes:
            keep = remap[category["images"]] >= 0
            detections = np.repeat(keep, category["lengths"])
            lengths = category["lengths"][keep]
            starts = np.cumsum(lengths)-lengths
            result.classes.append({"gt": category["gt"][indices], "tp": category["tp"][:, detections],
                                   "fp": category["fp"][:, detections], "starts": starts, "lengths": lengths,
                                   "images": remap[category["images"][keep]]})
        return result

    def ap(self, image_weights=None):
        weights = np.ones(self.images, dtype=np.int64) if image_weights is None else np.asarray(image_weights)
        if (weights.shape != (self.images,) or not np.isfinite(weights).all() or (weights < 0).any()
                or not np.equal(weights, np.floor(weights)).all()):
            raise ValueError("nonnegative integer replication count per image required")
        weights = weights.astype(np.int64)
        values = []
        for category in self.classes:
            positives = int(category["gt"] @ weights)
            if not positives:
                continue
            order = replicate_tied_blocks(category["starts"], category["lengths"], category["images"], weights)
            if not len(order):
                values.append(0.)
                continue
            tp = np.cumsum(category["tp"][:, order], axis=1, dtype=np.float64)
            fp = np.cumsum(category["fp"][:, order], axis=1, dtype=np.float64)
            recall = tp/positives
            precision = tp/(tp+fp+np.spacing(1))
            precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            sampled = np.zeros((10, 101), dtype=float)
            for i in range(10):
                indices = np.searchsorted(recall[i], np.linspace(0, 1, 101), side="left")
                valid = indices < precision.shape[1]
                sampled[i, valid] = precision[i, indices[valid]]
            values.append(float(sampled.mean()))
        return float(np.mean(values)) if values else None


def sequence_draw_weights(sequences, n_boot, seed):
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("positive integer bootstrap replicate count required")
    names, inverse = np.unique(np.asarray(sequences), return_inverse=True)
    if len(names) < 2:
        raise ValueError("at least two independent sequences required")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        yield counts[inverse]


def paired_pooled_bootstrap(candidate, control, *, n_boot=10000, seed=20260905, progress=None):
    """Paired sequence bootstrap of mean pooled AP across fixed trained seeds."""
    if not candidate or len(candidate) != len(control):
        raise ValueError("paired nonempty collections of seed-specific COCO evaluators required")
    sequences = candidate[0].sequences
    if any(value.sequences != sequences for value in (*candidate, *control)):
        raise ValueError("identical image order and sequence identity required across seeds and methods")
    candidate_points, control_points = [v.ap() for v in candidate], [v.ap() for v in control]
    if any(v is None for v in candidate_points+control_points):
        raise ValueError("pooled AP is undefined without ground truth")
    draws = []
    for i, weights in enumerate(sequence_draw_weights(sequences, n_boot, seed)):
        a, b = [v.ap(weights) for v in candidate], [v.ap(weights) for v in control]
        draws.append([np.mean(a), np.mean(b)] if all(v is not None for v in a+b) else [np.nan, np.nan])
        if progress is not None and ((i+1) % 100 == 0 or i+1 == n_boot):
            progress(i+1)
    draws = np.asarray(draws)
    valid = np.isfinite(draws).all(axis=1)
    if not valid.any():
        raise ValueError("all sequence bootstrap draws have undefined AP")
    delta = draws[valid, 0]-draws[valid, 1]
    seed_deltas = np.asarray(candidate_points)-control_points
    observed = float(seed_deltas.mean())
    low, high = np.quantile(delta, [.025, .975])
    # A centered, paired bootstrap null distribution; never report p=0.
    p = float((1+np.count_nonzero(np.abs(delta-observed) >= abs(observed)))/(len(delta)+1))
    sd = float(seed_deltas.std(ddof=1)) if len(seed_deltas) > 1 else None
    summary = {"endpoint": "mean_seed_pooled_COCO_AP_difference", "seeds": len(candidate),
               "sequences": len(set(sequences)), "images": len(sequences), "bootstrap_replicates": n_boot,
               "valid_replicates": int(valid.sum()), "undefined_replicates": int((~valid).sum()),
               "candidate_seed_AP": candidate_points, "control_seed_AP": control_points,
               "candidate_mean_AP": float(np.mean(candidate_points)), "control_mean_AP": float(np.mean(control_points)),
               "candidate_seed_SD": float(np.std(candidate_points, ddof=1)) if len(candidate) > 1 else None,
               "control_seed_SD": float(np.std(control_points, ddof=1)) if len(control) > 1 else None,
               "paired_difference": observed, "ci95_low": float(low), "ci95_high": float(high),
               "paired_seed_difference_SD": sd, "paired_seed_standardized_difference": observed/sd if sd is not None and sd > 0 else None,
               "effect_size_unit": "independent_training_seed_pooled_AP_not_frames",
               "centered_two_sided_bootstrap_p": p, "confidence_scope": "sequence_sampling_conditional_on_fixed_trained_seeds",
               "replication_order": "canonical_original_image_order_each_image_repeated_by_sequence_count"}
    return summary, draws

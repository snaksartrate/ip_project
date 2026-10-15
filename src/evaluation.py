"""
evaluation.py

Computes per-method, per-image evaluation metrics against ground truth
(object count, ground-truth object instances and ground-truth masks).

FUNCTIONS:

    calculate_iou(predicted_mask, ground_truth_mask) -> float
        Pixel-level Intersection-over-Union between two masks. Used by
        stress-testing, grid-search auto-tuning and the benchmark.

    gt_objects_from_instances(instance_map) -> list[dict]
        Builds ground-truth object dicts (same format as
        detection.detect_objects()) from a label map in which 0 is
        background and 1..N are individual objects.

    match_objects(predicted, ground_truth, iou_threshold=0.5,
                  image_shape=None) -> dict
        Greedy one-to-one matching between predicted and ground-truth
        objects by IoU. Returns matched pairs, false positives
        (unmatched predictions) and misses (unmatched ground truth).

    count_accuracy(detected_count, gt_count) -> float
        1 - |detected - true| / true, clamped to [0, 1].

    evaluate_method(result, ground_truth, iou_threshold=0.5) -> dict
        All metrics for one method on one image:
        {
            "detected_count", "gt_count", "correct", "missed",
            "false_positives", "count_accuracy", "mean_iou",
            "mask_iou", "processing_time_ms",
            "mean_measurement_error_pct",
        }

Note: the original plan named the IoU function `compute_iou`; the
advanced benchmarking blueprint specifies `calculate_iou`, which is the
name used here.
"""

from typing import List, Optional, Tuple
import numpy as np
import cv2

from src import detection


# ---------------------------------------------------------------------------
# 1. PIXEL-LEVEL IoU
# ---------------------------------------------------------------------------

def _to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert a mask to a boolean array where True = foreground.

    Why this exists:
        Segmentation masks in this project use {0, 255}, while ground
        truth masks may come as {0, 1} or {0, 255}. Treating any
        non-zero pixel as foreground handles all of these uniformly, so
        callers never need to normalize masks themselves.

    Args:
        mask: np.ndarray of any numeric dtype, (H, W) or (H, W, C).

    Returns:
        np.ndarray, bool, (H, W).
    """
    if mask.ndim == 3:
        # A ground-truth mask saved as a 3-channel image: a pixel is
        # foreground if any channel is non-zero.
        mask = mask.max(axis=2)
    return mask > 0


def calculate_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    Calculate Intersection-over-Union (IoU, a.k.a. Jaccard index)
    between a predicted mask and a ground-truth mask.

    What it does / how it works (the math):
        IoU = Area of Overlap / Area of Union
            = |P AND G| / |P OR G|
        where P is the set of predicted foreground pixels and G is the
        set of ground-truth foreground pixels.
            IoU = 1.0 -> perfect match
            IoU = 0.0 -> no overlap at all

    Why IoU rather than plain pixel accuracy:
        Foreground objects usually cover a small part of the image. A
        method that labels everything as background would still score
        high pixel accuracy, but its IoU would be 0. IoU only rewards
        correctly overlapping foreground, so it is a much more honest
        measure of segmentation quality.

    Binary handling:
        Both masks are converted to binary first (any non-zero pixel is
        foreground), so {0, 255} and {0, 1} masks both work.

    Edge case (empty union):
        If both masks are completely empty, the union is 0 and IoU is
        mathematically undefined. Both masks agree there is no object,
        so 1.0 is returned. If only one mask is empty, the intersection
        is 0 and the result is naturally 0.0.

    Args:
        predicted_mask: np.ndarray, (H, W), output of a segmentation
            method.
        ground_truth_mask: np.ndarray, (H, W), the reference mask. Must
            have the same height and width as `predicted_mask`.

    Returns:
        float in [0.0, 1.0].

    Raises:
        ValueError: if the two masks have different spatial shapes.
    """
    pred = _to_binary(predicted_mask)
    gt = _to_binary(ground_truth_mask)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Mask shapes do not match: predicted {pred.shape} vs "
            f"ground truth {gt.shape}. Resize the ground truth to the "
            f"preprocessed image size first."
        )

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


# ---------------------------------------------------------------------------
# 2. GROUND-TRUTH OBJECTS FROM AN INSTANCE MAP
# ---------------------------------------------------------------------------

def gt_objects_from_instances(instance_map: np.ndarray) -> List[dict]:
    """
    Build ground-truth object dicts from an instance label map.

    What it does / how it works:
        The instance map (written by generate_dataset.py) holds 0 for
        background and 1..N for the N drawn objects. For each label, a
        binary mask of just that object is made and passed through
        detection.detect_objects(), so every ground-truth object has
        EXACTLY the same fields as a predicted one (id, label, bbox,
        centroid, area, contour). Predictions and ground truth are
        therefore measured by identical code, which keeps the
        comparison fair.

    Why instances rather than the binary ground-truth mask:
        The binary mask merges touching objects into one blob, so
        connected components on it would give the WRONG true count.
        The instance map keeps every drawn object separate.

    Note on "label":
        In the returned dicts, "label" is the instance value from the
        map (the original object number), and "id" is a sequential
        1..N. They are normally equal.

    Args:
        instance_map: (H, W) integer array (e.g. uint16 from
            cv2.imread(..., cv2.IMREAD_UNCHANGED)); 0 = background.

    Returns:
        list[dict], one per instance, ordered by label. Empty list if
        the map has no objects.
    """
    labels = np.unique(instance_map)
    labels = labels[labels > 0]

    objects: List[dict] = []
    for label in labels:
        region = np.where(instance_map == label, 255, 0).astype(np.uint8)
        found = detection.detect_objects(region, min_area=1)
        if not found:
            continue  # defensive: a label with no pixels cannot occur
        obj = dict(max(found, key=lambda o: o["area"]))
        obj["id"] = len(objects) + 1
        obj["label"] = int(label)
        objects.append(obj)
    return objects


# ---------------------------------------------------------------------------
# 3. OBJECT MATCHING
# ---------------------------------------------------------------------------

def _infer_shape(*object_lists: List[dict]) -> Tuple[int, int]:
    """Smallest (H, W) canvas that contains every bounding box given."""
    h = w = 1
    for objects in object_lists:
        for obj in objects:
            x, y, bw, bh = obj["bbox"]
            w = max(w, x + bw + 1)
            h = max(h, y + bh + 1)
    return h, w


def _object_mask(obj: dict, shape: Tuple[int, int]) -> np.ndarray:
    """
    Rasterize one object to a boolean mask by filling its outer contour.

    Why: detection dicts store a contour, not pixels. Filling the outer
    contour gives the object's region (holes inside it are filled,
    which does not matter for solid objects like coins).
    """
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(canvas, [obj["contour"]], -1, 255, thickness=-1)
    return canvas > 0


def _bboxes_overlap(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def match_objects(predicted: List[dict], ground_truth: List[dict],
                  iou_threshold: float = 0.5,
                  image_shape: Optional[Tuple[int, int]] = None) -> dict:
    """
    Match predicted objects to ground-truth objects, one-to-one, by IoU.

    What it does / how it works (greedy matching):
        1. Every predicted object and every ground-truth object is
           rasterized to a mask.
        2. For each pair whose bounding boxes overlap, the IoU of the
           two masks is computed. Pairs below `iou_threshold` are
           discarded.
        3. The remaining pairs are sorted by IoU, highest first. Going
           down the list, a pair is accepted if neither its predicted
           object nor its ground-truth object is already used.
        Each object ends up in at most one match.

    Outcomes:
        matched ground truth     -> a correct detection
        unmatched prediction     -> false positive (detected something
                                    that is not a real object, or a
                                    merged/badly shaped blob)
        unmatched ground truth   -> miss (a real object nobody found)

    Touching objects:
        When two real objects merge into one blob, the blob usually
        matches at most one of them. The other is a miss. This is the
        expected counting failure of connected components.

    Why greedy:
        It is simple, deterministic and easy to explain. With
        well-separated objects and a threshold of 0.5, it gives the
        same result as an optimal assignment. Ties are broken by object
        order, so results are reproducible.

    Args:
        predicted: list of object dicts from detection.detect_objects().
        ground_truth: list of object dicts from
            gt_objects_from_instances().
        iou_threshold: minimum IoU for a pair to count as a match.
        image_shape: (H, W) of the image. If None, the smallest canvas
            containing every bounding box is used (IoU is unaffected,
            since only the object pixels matter).

    Returns:
        dict:
            {
                "matches": list[dict],        # {"pred_id", "gt_id", "iou",
                                              #  "pred_area", "gt_area"}
                "false_positives": list[int], # ids of unmatched predictions
                "misses": list[int],          # ids of unmatched GT objects
                "iou_threshold": float,
            }
    """
    shape = image_shape if image_shape is not None else _infer_shape(predicted, ground_truth)

    pred_masks = [_object_mask(o, shape) for o in predicted]
    gt_masks = [_object_mask(o, shape) for o in ground_truth]

    candidates = []
    for i, (p_obj, p_mask) in enumerate(zip(predicted, pred_masks)):
        for j, (g_obj, g_mask) in enumerate(zip(ground_truth, gt_masks)):
            if not _bboxes_overlap(p_obj["bbox"], g_obj["bbox"]):
                continue
            inter = np.logical_and(p_mask, g_mask).sum()
            union = np.logical_or(p_mask, g_mask).sum()
            iou = float(inter / union) if union > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, i, j))

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_pred, used_gt = set(), set()
    matches = []
    for iou, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append({
            "pred_id": int(predicted[i]["id"]),
            "gt_id": int(ground_truth[j]["id"]),
            "iou": iou,
            "pred_area": int(predicted[i]["area"]),
            "gt_area": int(ground_truth[j]["area"]),
        })

    return {
        "matches": matches,
        "false_positives": [int(o["id"]) for k, o in enumerate(predicted) if k not in used_pred],
        "misses": [int(o["id"]) for k, o in enumerate(ground_truth) if k not in used_gt],
        "iou_threshold": float(iou_threshold),
    }


# ---------------------------------------------------------------------------
# 4. COUNT ACCURACY
# ---------------------------------------------------------------------------

def count_accuracy(detected_count: int, gt_count: int) -> float:
    """
    How close the detected object count is to the true count.

    What it does / how it works (the math):
        accuracy = 1 - |detected - true| / true, clamped to [0, 1]
            5 detected, 5 true  -> 1.0
            4 detected, 5 true  -> 0.8
            10 detected, 5 true -> 0.0 (error of 100% or more)

    Why it is separate from `correct`:
        This only compares NUMBERS. A method can reach the right count
        by luck (one object missed, one false blob found). The matching
        step (`correct`, `missed`, `false_positives`) checks that the
        right objects were found. Reporting both exposes that.

    Edge case (no true objects):
        The relative error is undefined, so 1.0 is returned if nothing
        was detected and 0.0 otherwise.

    Args:
        detected_count: number of objects the method found.
        gt_count: true number of objects.

    Returns:
        float in [0.0, 1.0].
    """
    detected_count = int(detected_count)
    gt_count = int(gt_count)
    if gt_count <= 0:
        return 1.0 if detected_count == 0 else 0.0
    return float(max(0.0, 1.0 - abs(detected_count - gt_count) / gt_count))


# ---------------------------------------------------------------------------
# 5. FULL EVALUATION OF ONE METHOD ON ONE IMAGE
# ---------------------------------------------------------------------------

def evaluate_method(result: dict, ground_truth: dict,
                    iou_threshold: float = 0.5) -> dict:
    """
    Compute every metric for one method on one image.

    Inputs (plain dicts, so this module does not depend on how the
    pipeline or the dataset loader are organised):

        result:
            "objects": list of dicts from detection.detect_objects()
            "mask": (optional) the final cleaned mask the objects came
                    from, used for the pixel-level mask IoU
            "processing_time_ms": (optional) segmentation time
        ground_truth:
            "objects": list of dicts from gt_objects_from_instances()
            "count": (optional) true object count; defaults to
                     len(ground_truth["objects"])
            "mask": (optional) binary ground-truth mask

    Metrics returned:
        detected_count, gt_count
        correct:
            ground-truth objects matched by a prediction (IoU >=
            threshold).
        missed:
            ground-truth objects with no match.
        false_positives:
            predicted objects with no match.
        count_accuracy:
            see count_accuracy().
        mean_iou:
            mean IoU over MATCHED objects only (NaN if none matched).
            It measures how well found objects are outlined.
        mask_iou:
            pixel-level IoU of the whole mask against the GT mask
            (NaN if either mask is missing). Unlike mean_iou it still
            gives a score when nothing matches, e.g. when the edge
            method floods the whole image.
        processing_time_ms:
            taken from result (NaN if missing).
        mean_measurement_error_pct:
            mean of |pred_area - gt_area| / gt_area * 100 over matched
            objects (NaN if none matched). Area is the pixel count.

    Identity that always holds:
        correct + missed == gt_count_of_objects, and
        correct + false_positives == detected_count.

    Args:
        result: see above.
        ground_truth: see above.
        iou_threshold: matching threshold, passed to match_objects().

    Returns:
        dict with the keys listed above.
    """
    objects = result["objects"]
    gt_objects = ground_truth["objects"]
    gt_count = int(ground_truth.get("count", len(gt_objects)))

    gt_mask = ground_truth.get("mask")
    shape = gt_mask.shape[:2] if gt_mask is not None else None

    matching = match_objects(objects, gt_objects, iou_threshold=iou_threshold,
                             image_shape=shape)
    matches = matching["matches"]

    if matches:
        mean_iou = float(np.mean([m["iou"] for m in matches]))
        mean_err = float(np.mean([
            abs(m["pred_area"] - m["gt_area"]) / m["gt_area"] * 100.0
            for m in matches if m["gt_area"] > 0
        ]))
    else:
        mean_iou = float("nan")
        mean_err = float("nan")

    pred_mask = result.get("mask")
    if pred_mask is not None and gt_mask is not None:
        mask_iou = calculate_iou(pred_mask, gt_mask)
    else:
        mask_iou = float("nan")

    return {
        "detected_count": len(objects),
        "gt_count": gt_count,
        "correct": len(matches),
        "missed": len(matching["misses"]),
        "false_positives": len(matching["false_positives"]),
        "count_accuracy": count_accuracy(len(objects), gt_count),
        "mean_iou": mean_iou,
        "mask_iou": mask_iou,
        "processing_time_ms": float(result.get("processing_time_ms", float("nan"))),
        "mean_measurement_error_pct": mean_err,
    }
"""
evaluation.py

Computes per-method, per-image evaluation metrics, optionally against
ground truth (object count and/or ground-truth masks).

Planned functions (to be implemented in Phase 6/9):

    match_objects(predicted, ground_truth, iou_threshold=0.5) -> dict
        Greedy IoU-based matching between predicted and ground-truth
        objects. Returns matched pairs, unmatched predictions (false
        detections), and unmatched ground truth (missed objects).

    compute_iou(mask_a, mask_b) -> float
        Intersection-over-Union between two binary masks.

    count_accuracy(detected_count, gt_count) -> float

    evaluate_method(result, ground_truth) -> dict
        Aggregates all metrics for a single method on a single image:
        {
            "detected_count": ...,
            "gt_count": ...,
            "correct": ..., "missed": ..., "false_positives": ...,
            "mean_iou": ...,
            "processing_time_ms": ...,
            "mean_measurement_error_pct": ...,
        }

No algorithms are implemented yet -- this is Phase 1 (project setup) only.
"""

# Implementation begins in Phase 6/9.

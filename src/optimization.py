"""
optimization.py

Auto-tuning for the parameterized segmentation methods, using
exhaustive grid search against a ground-truth mask.

Why this module exists:
    Adaptive and edge-based segmentation depend on hyperparameters
    (block size, C, Canny thresholds). Choosing them by moving sliders
    introduces human bias: a method can look worse simply because the
    person tuning it was less careful. Grid search removes that bias by
    trying every combination in a fixed range and keeping the one with
    the highest IoU, so each method is compared at its best.

How it works:
    For every parameter combination:
        1. Call the existing segment_*() function.
        2. Score the mask with evaluation.calculate_iou().
        3. Keep the combination with the highest IoU.

Notes:
    - Ties are resolved in favor of the FIRST combination found (the
      comparison is strict ">"), which makes results deterministic.
    - Every function also returns the full list of results, so the UI
      can display a ranking or heatmap if desired.
    - Progress can be reported through an optional callback, so this
      module stays free of any Streamlit code.
"""

from typing import Callable, Optional
import numpy as np

from src import segmentation, evaluation


# ---------------------------------------------------------------------------
# Search ranges (from the blueprint)
# ---------------------------------------------------------------------------

ADAPTIVE_BLOCK_SIZES = list(range(3, 32, 2))     # 3, 5, ..., 31  (15 values)
ADAPTIVE_C_VALUES = list(range(-10, 11))         # -10 ... +10    (21 values)

EDGE_LOW_THRESHOLDS = [10, 30, 50, 70, 90]
EDGE_HIGH_THRESHOLDS = [100, 150, 200, 250]


def _report(progress_callback: Optional[Callable[[float], None]],
            done: int, total: int) -> None:
    """Call the progress callback with a fraction in [0, 1], if given."""
    if progress_callback is not None and total > 0:
        progress_callback(done / total)


# ---------------------------------------------------------------------------
# 1. ADAPTIVE THRESHOLDING GRID SEARCH
# ---------------------------------------------------------------------------

def grid_search_adaptive(image: np.ndarray, ground_truth_mask: np.ndarray,
                         object_polarity: str = "darker",
                         adaptive_method: str = "gaussian",
                         progress_callback: Optional[Callable[[float], None]] = None,
                         block_sizes: Optional[list] = None,
                         c_values: Optional[list] = None) -> dict:
    """
    Find the (block_size, C) pair that maximizes IoU for adaptive
    thresholding.

    Default search space (the blueprint ranges):
        block_size in [3, 5, 7, ..., 31]   (15 values)
        C          in [-10, -9, ..., +10]  (21 values)
        -> 315 combinations, each one a full adaptive threshold run.
        Passing block_sizes and/or c_values replaces these ranges.

    Args:
        image: (H, W) uint8 preprocessed grayscale image.
        ground_truth_mask: (H, W) reference mask, same size as `image`.
        object_polarity: "darker" or "brighter". This is not part of
            the blueprint signature; it defaults to "darker", which
            matches the rest of the codebase, and is needed because the
            right answer depends on which side counts as foreground.
        adaptive_method: "gaussian" or "mean". Held fixed during the
            search (not tuned), defaults to "gaussian".
        progress_callback: optional function called with a float in
            [0, 1] after each combination.
        block_sizes: optional list of odd block sizes to search instead
            of the default 3..31. segment_adaptive caps the block size
            at 99, so larger values are clipped to 99.
        c_values: optional list of C values to search instead of the
            default -10..+10. Used for the wide search (block 3-99,
            C -10..30) in run_tuned_benchmark.py.

    Returns:
        dict:
            {
                "method": "adaptive",
                "best_block_size": int,
                "best_C": int,
                "best_iou": float,
                "best_mask": np.ndarray,   # mask at the optimum
                "results": list[dict],     # every combination tried:
                                           #   {"block_size", "C", "iou"}
                "n_combinations": int,
            }
    """
    block_sizes = block_sizes or ADAPTIVE_BLOCK_SIZES
    c_values = c_values or ADAPTIVE_C_VALUES
    total = len(block_sizes) * len(c_values)
    done = 0

    best_iou = -1.0
    best_block_size = None
    best_c = None
    best_mask = None
    results = []

    for block_size in block_sizes:
        for c_value in c_values:
            seg = segmentation.segment_adaptive(
                image,
                block_size=block_size,
                C=c_value,
                adaptive_method=adaptive_method,
                object_polarity=object_polarity,
            )
            iou = evaluation.calculate_iou(seg["mask"], ground_truth_mask)
            results.append({"block_size": block_size, "C": c_value, "iou": iou})

            if iou > best_iou:
                best_iou = iou
                best_block_size = block_size
                best_c = c_value
                best_mask = seg["mask"]

            done += 1
            _report(progress_callback, done, total)

    return {
        "method": "adaptive",
        "best_block_size": best_block_size,
        "best_C": best_c,
        "best_iou": best_iou,
        "best_mask": best_mask,
        "results": results,
        "n_combinations": total,
    }


# ---------------------------------------------------------------------------
# 2. EDGE-BASED GRID SEARCH
# ---------------------------------------------------------------------------

def grid_search_edge(image: np.ndarray, ground_truth_mask: np.ndarray,
                     closing_kernel_size: int = 5,
                     progress_callback: Optional[Callable[[float], None]] = None) -> dict:
    """
    Find the (low_threshold, high_threshold) pair that maximizes IoU
    for edge-based segmentation.

    Search space:
        low_threshold  in [10, 30, 50, 70, 90]   (5 values)
        high_threshold in [100, 150, 200, 250]   (4 values)
        -> 20 combinations. With these ranges low < high always holds
           (max low is 90, min high is 100), but combinations with
           low >= high are skipped anyway as a safeguard, since Canny's
           hysteresis is meaningless in that case.

    Args:
        image: (H, W) uint8 preprocessed grayscale image.
        ground_truth_mask: (H, W) reference mask, same size as `image`.
        closing_kernel_size: gap-closing kernel size, held fixed during
            the search (not tuned). Not part of the blueprint
            signature; defaults to the same value as the UI default (5).
        progress_callback: optional function called with a float in
            [0, 1] after each combination.

    Returns:
        dict:
            {
                "method": "edge",
                "best_low_threshold": int,
                "best_high_threshold": int,
                "best_iou": float,
                "best_mask": np.ndarray,
                "results": list[dict],   # {"low_threshold", "high_threshold", "iou"}
                "n_combinations": int,
            }
    """
    combinations = [
        (low, high)
        for low in EDGE_LOW_THRESHOLDS
        for high in EDGE_HIGH_THRESHOLDS
        if low < high
    ]
    total = len(combinations)
    done = 0

    best_iou = -1.0
    best_low = None
    best_high = None
    best_mask = None
    results = []

    for low, high in combinations:
        seg = segmentation.segment_edge_based(
            image,
            low_threshold=low,
            high_threshold=high,
            closing_kernel_size=closing_kernel_size,
        )
        iou = evaluation.calculate_iou(seg["mask"], ground_truth_mask)
        results.append({"low_threshold": low, "high_threshold": high, "iou": iou})

        if iou > best_iou:
            best_iou = iou
            best_low = low
            best_high = high
            best_mask = seg["mask"]

        done += 1
        _report(progress_callback, done, total)

    return {
        "method": "edge",
        "best_low_threshold": best_low,
        "best_high_threshold": best_high,
        "best_iou": best_iou,
        "best_mask": best_mask,
        "results": results,
        "n_combinations": total,
    }
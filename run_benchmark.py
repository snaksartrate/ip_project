"""
run_benchmark.py

Batch benchmark: runs all four segmentation methods on every image in
the synthetic dataset and writes one results row per (image, method).

For every image, ALL methods get:
    - the same grayscale image and the same noise reduction,
    - the same morphology (opening -> closing -> small-region removal),
    - the same object detection (connected components, same min_area).
Only the segmentation step differs, which keeps the comparison fair.

Evaluation uses exact ground truth:
    - the instance label map  -> true objects (touching objects stay
                                 separate, so the true count is right),
    - the binary GT mask      -> pixel-level mask IoU,
    - counts.csv              -> the TRUE object count.

Usage (from the project/ folder):

    python run_benchmark.py
    python run_benchmark.py --noise median --kernel-size 5 --out-file results_median.csv
    python run_benchmark.py --save-masks

Output:
    results/<out-file>            one row per (image, method)
    results/masks/<method>/...    final cleaned masks (with --save-masks)

No multiprocessing is used (see the Windows spawn issue noted in
profiling.py), so no __main__ guard is strictly needed, but it is kept
as good practice.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import preprocessing, segmentation, morphology, detection, evaluation


METHOD_KEYS = ("global", "otsu", "adaptive", "edge")

COLUMNS = [
    "filename", "condition", "method",
    "gt_count", "gt_blobs", "detected_count",
    "correct", "missed", "false_positives",
    "count_accuracy", "mean_iou", "mask_iou", "raw_mask_iou",
    "processing_time_ms", "mean_measurement_error_pct",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read(path: Path, flags: int) -> np.ndarray:
    """cv2.imread that fails loudly instead of returning None."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read '{path}'. Run generate_dataset.py first.")
    return img


def load_dataset(dataset_dir: Path) -> list:
    """
    Load every dataset entry listed in counts.csv.

    Returns:
        list[dict], one per image:
            {"filename", "condition", "polarity", "n_objects",
             "image", "gt_mask", "gt_objects", "gt_blobs"}
        gt_objects come from the instance map (separate objects);
        gt_blobs is the number of connected components of the binary GT
        mask, i.e. the count a perfect segmentation would still get if
        touching objects merge.
    """
    gt_dir = dataset_dir / "ground_truth"
    with open(gt_dir / "counts.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    entries = []
    for row in rows:
        name = row["filename"]
        image = _read(dataset_dir / "images" / name, cv2.IMREAD_GRAYSCALE)
        gt_mask = _read(gt_dir / name, cv2.IMREAD_GRAYSCALE)
        inst = _read(gt_dir / "instances" / name, cv2.IMREAD_UNCHANGED)

        gt_objects = evaluation.gt_objects_from_instances(inst)
        if len(gt_objects) != int(row["n_objects"]):
            print(f"WARNING {name}: instance map has {len(gt_objects)} objects "
                  f"but counts.csv says {row['n_objects']}.")

        entries.append({
            "filename": name,
            "condition": row["condition"],
            "polarity": row["polarity"],
            "n_objects": int(row["n_objects"]),
            "image": image,
            "gt_mask": gt_mask,
            "gt_objects": gt_objects,
            "gt_blobs": int(cv2.connectedComponents(gt_mask)[0] - 1),
        })
    return entries


# ---------------------------------------------------------------------------
# One image, all methods
# ---------------------------------------------------------------------------

def method_params(global_threshold: int, polarity: str) -> dict:
    """Default parameters for the four methods (same for every image)."""
    return {
        "global": {"threshold": global_threshold, "object_polarity": polarity},
        "otsu": {"object_polarity": polarity},
        "adaptive": {"block_size": 11, "C": 2, "adaptive_method": "gaussian",
                     "object_polarity": polarity},
        "edge": {"low_threshold": 50, "high_threshold": 150, "closing_kernel_size": 5},
    }


def run_image(entry: dict, cfg: argparse.Namespace, params_override: dict = None):
    """
    Run all four methods on one dataset entry.

    Returns:
        (rows, masks): rows is a list of result dicts (one per method),
        masks is {method_key: final cleaned mask}.
    """
    # Preprocessing: noise reduction only. The dataset is 640x480, within
    # the app's default 800 px limit, so no resize happens and the GT
    # masks stay aligned with the image.
    gray = preprocessing.apply_noise_reduction(
        entry["image"], method=cfg.noise, kernel_size=cfg.kernel_size)

    params = params_override or method_params(cfg.global_threshold, entry["polarity"])
    gt = {"objects": entry["gt_objects"], "count": entry["n_objects"],
          "mask": entry["gt_mask"]}

    rows, masks = [], {}
    for key in METHOD_KEYS:
        seg = segmentation.segment(gray, key, **params[key])
        cleaned = morphology.clean_mask(
            seg["mask"], open_kernel_size=cfg.open_kernel,
            close_kernel_size=cfg.close_kernel, min_area=cfg.min_area)
        objects = detection.detect_objects(cleaned["final"], min_area=cfg.min_area)

        metrics = evaluation.evaluate_method(
            {"objects": objects, "mask": cleaned["final"],
             "processing_time_ms": seg["processing_time_ms"]},
            gt, iou_threshold=cfg.iou_threshold)

        rows.append({
            "filename": entry["filename"],
            "condition": entry["condition"],
            "method": key,
            "gt_blobs": entry["gt_blobs"],
            "raw_mask_iou": evaluation.calculate_iou(seg["mask"], entry["gt_mask"]),
            **metrics,
        })
        masks[key] = cleaned["final"]
    return rows, masks


# ---------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------

def run_benchmark(dataset_dir: Path, out_dir: Path, out_file: str,
                  cfg: argparse.Namespace, save_masks: bool = False) -> pd.DataFrame:
    """Run every image, write the CSV, return the results DataFrame."""
    entries = load_dataset(dataset_dir)
    print(f"Loaded {len(entries)} images from '{dataset_dir}'.")

    all_rows = []
    for i, entry in enumerate(entries, start=1):
        rows, masks = run_image(entry, cfg)
        all_rows.extend(rows)

        if save_masks:
            for key, mask in masks.items():
                mdir = out_dir / "masks" / key
                mdir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mdir / entry["filename"]), mask)

        print(f"  [{i:2d}/{len(entries)}] {entry['filename']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(out_dir / out_file, index=False)
    print(f"Wrote {len(df)} rows to '{out_dir / out_file}'.")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Mean metrics per method and per (condition, method). NaNs are skipped."""
    cols = ["count_accuracy", "mean_iou", "mask_iou",
            "false_positives", "missed", "processing_time_ms"]
    pd.set_option("display.width", 140)
    print("\n=== Mean by method ===")
    print(df.groupby("method")[cols].mean().round(3).reindex(METHOD_KEYS))
    print("\n=== Mean count_accuracy by condition x method ===")
    print(df.pivot_table(index="condition", columns="method",
                         values="count_accuracy", aggfunc="mean")
          .round(3)[list(METHOD_KEYS)])


def main() -> None:
    p = argparse.ArgumentParser(description="Run the segmentation benchmark.")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="results")
    p.add_argument("--out-file", default="benchmark_results.csv")
    p.add_argument("--noise", default="none", choices=["none", "gaussian", "median"])
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=5)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--global-threshold", type=int, default=127)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--save-masks", action="store_true")
    cfg = p.parse_args()

    df = run_benchmark(Path(cfg.dataset), Path(cfg.out), cfg.out_file, cfg,
                       save_masks=cfg.save_masks)
    print_summary(df)


if __name__ == "__main__":
    main()
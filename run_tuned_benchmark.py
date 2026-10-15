"""
run_tuned_benchmark.py

Second benchmark run: every method gets its BEST parameters for each
image, found by grid search against that image's ground truth, then the
usual shared cleanup, detection and evaluation run on top.

IMPORTANT, this is an ORACLE upper bound, not a fair deployment score:
    The parameters are tuned on the same image they are tested on, using
    its ground truth. A real system could not do that. The result
    answers "how good can each method possibly get?", which separates a
    method's ceiling from the weakness of its default parameters.

What is tuned (objective = raw mask IoU, before cleanup):
    global    threshold in 5, 10, ..., 250   (sweep added here; the
              blueprint's optimization.py has no global search)
    adaptive  optimization.grid_search_adaptive
              default: block 3-31 x C -10..10 (315 combinations)
              --wide-adaptive: block 3-99 x C -10..30 (2009 combinations)
    edge      optimization.grid_search_edge     (20 combinations)
    otsu      nothing to tune

Usage (from the project/ folder):
    python run_tuned_benchmark.py
    python run_tuned_benchmark.py --wide-adaptive --out-file tuned_wide_results.csv
    python run_tuned_benchmark.py --noise median --out-file tuned_median.csv

Outputs:
    results/<out-file>               same columns as benchmark_results.csv
                                     (default name: tuned_results.csv)
    comparison/tuned_params.csv      the parameters chosen per image;
                                     overwritten by every run
"""

import argparse
from pathlib import Path

import pandas as pd

from src import preprocessing, segmentation, evaluation, optimization
import run_benchmark

GLOBAL_THRESHOLDS = list(range(5, 251, 5))

WIDE_BLOCKS = list(range(3, 100, 2))   # 3..99, 99 is segment_adaptive's own cap
WIDE_C = list(range(-10, 31))          # -10..30


def tune_global(gray, gt_mask, polarity):
    """Return the fixed threshold with the highest raw mask IoU."""
    best_iou, best_t = -1.0, None
    for t in GLOBAL_THRESHOLDS:
        seg = segmentation.segment_global(gray, threshold=t, object_polarity=polarity)
        iou = evaluation.calculate_iou(seg["mask"], gt_mask)
        if iou > best_iou:       # strict > : first best wins, deterministic
            best_iou, best_t = iou, t
    return best_t


def tuned_params(entry, cfg):
    """Grid-search every tunable method on one image; return a params dict."""
    gray = preprocessing.apply_noise_reduction(
        entry["image"], method=cfg.noise, kernel_size=cfg.kernel_size)
    pol, gt = entry["polarity"], entry["gt_mask"]

    t = tune_global(gray, gt, pol)
    wide = getattr(cfg, "wide_adaptive", False)
    ad = optimization.grid_search_adaptive(
        gray, gt, object_polarity=pol, adaptive_method="gaussian",
        block_sizes=WIDE_BLOCKS if wide else None,
        c_values=WIDE_C if wide else None)
    ed = optimization.grid_search_edge(gray, gt, closing_kernel_size=5)

    return {
        "global": {"threshold": t, "object_polarity": pol},
        "otsu": {"object_polarity": pol},
        "adaptive": {"block_size": ad["best_block_size"], "C": ad["best_C"],
                     "adaptive_method": "gaussian", "object_polarity": pol},
        "edge": {"low_threshold": ed["best_low_threshold"],
                 "high_threshold": ed["best_high_threshold"],
                 "closing_kernel_size": 5},
    }


def main():
    p = argparse.ArgumentParser(description="Oracle-tuned benchmark run.")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="results")
    p.add_argument("--out-file", default="tuned_results.csv")
    p.add_argument("--noise", default="none", choices=["none", "gaussian", "median"])
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--close-kernel", type=int, default=5)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--global-threshold", type=int, default=127)  # unused here
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--wide-adaptive", action="store_true",
                   help="search block 3-99 and C -10..30 (2009 combinations)")
    cfg = p.parse_args()

    entries = run_benchmark.load_dataset(Path(cfg.dataset))
    all_rows, param_rows = [], []
    for i, entry in enumerate(entries, start=1):
        params = tuned_params(entry, cfg)
        rows, _ = run_benchmark.run_image(entry, cfg, params_override=params)
        all_rows.extend(rows)
        param_rows.append({
            "filename": entry["filename"], "condition": entry["condition"],
            "global_T": params["global"]["threshold"],
            "adaptive_block": params["adaptive"]["block_size"],
            "adaptive_C": params["adaptive"]["C"],
            "edge_low": params["edge"]["low_threshold"],
            "edge_high": params["edge"]["high_threshold"],
        })
        print(f"  [{i:2d}/{len(entries)}] {entry['filename']}")

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=run_benchmark.COLUMNS)
    df.to_csv(out / cfg.out_file, index=False)

    Path("comparison").mkdir(exist_ok=True)
    pd.DataFrame(param_rows).to_csv("comparison/tuned_params.csv", index=False)

    print(f"Wrote {len(df)} rows to '{out / cfg.out_file}'.")
    run_benchmark.print_summary(df)


if __name__ == "__main__":
    main()
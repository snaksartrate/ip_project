"""
run_comparison.py

Aggregates a benchmark results CSV (from run_benchmark.py or
run_tuned_benchmark.py) into summary tables and comparison charts,
written to comparison/.

Usage (from the project/ folder):

    python run_comparison.py
    python run_comparison.py --results results/results_median.csv --name median
    python run_comparison.py --results results/tuned_wide_results.csv --name tuned_wide

Outputs (in --out, default "comparison", filenames start with --name):

    <name>_summary_by_method.csv
    <name>_summary_by_condition.csv
    <name>_method_<metric>.png          overall bar chart per metric
    <name>_condition_<metric>.png       grouped-by-condition chart per metric

Charted metrics: count_accuracy, mask_iou, mean_iou, false_positives,
processing_time_ms (see CHARTS below). The summary CSVs also hold
raw_mask_iou, missed and mean_measurement_error_pct.

Means skip NaN. mean_iou and mean_measurement_error_pct are NaN on
images where a method matched nothing, so those means only cover the
images where the method found at least one object. The column
"images_with_no_match" counts the images that were left out, which
keeps that selection visible.
"""

import matplotlib
matplotlib.use("Agg")   # write PNGs without needing a display

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import visualization


METHODS = ["global", "otsu", "adaptive", "edge"]
CONDITIONS = ["normal", "noisy", "uneven", "touching"]

METRICS = ["count_accuracy", "mask_iou", "raw_mask_iou", "mean_iou",
           "false_positives", "missed", "mean_measurement_error_pct",
           "processing_time_ms"]

CHARTS = [
    ("count_accuracy", "Count accuracy (1 = exact count)"),
    ("mask_iou", "Mask IoU after cleanup"),
    ("mean_iou", "Mean IoU of matched objects"),
    ("false_positives", "False positives per image"),
    ("processing_time_ms", "Segmentation time (ms)"),
]


def summarize(df: pd.DataFrame):
    """Return (by_method, by_condition) summary DataFrames."""
    by_method = df.groupby("method")[METRICS].mean().reindex(METHODS)
    by_method["images_with_no_match"] = (
        df.assign(none=df["correct"] == 0).groupby("method")["none"].sum()
        .reindex(METHODS))

    by_condition = df.groupby(["condition", "method"])[METRICS].mean()
    by_condition = by_condition.reindex(
        pd.MultiIndex.from_product([CONDITIONS, METHODS],
                                   names=["condition", "method"]))
    return by_method, by_condition


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate benchmark results.")
    p.add_argument("--results", default="results/benchmark_results.csv")
    p.add_argument("--out", default="comparison")
    p.add_argument("--name", default="baseline",
                   help="prefix for output files (e.g. baseline, median)")
    args = p.parse_args()

    df = pd.read_csv(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_method, by_condition = summarize(df)
    by_method.round(4).to_csv(out / f"{args.name}_summary_by_method.csv")
    by_condition.round(4).to_csv(out / f"{args.name}_summary_by_condition.csv")

    for metric, label in CHARTS:
        fig = visualization.plot_method_comparison(df, metric, title=label)
        fig.savefig(out / f"{args.name}_method_{metric}.png", dpi=150)
        plt.close(fig)
        fig = visualization.plot_condition_comparison(df, metric, title=label)
        fig.savefig(out / f"{args.name}_condition_{metric}.png", dpi=150)
        plt.close(fig)

    pd.set_option("display.width", 160)
    print(f"Read {len(df)} rows from '{args.results}'. Wrote files to '{out}/'.\n")
    print("=== By method ===")
    print(by_method.round(3))
    print("\n=== count_accuracy by condition x method ===")
    print(by_condition["count_accuracy"].unstack("method").round(3)[METHODS])
    print("\n=== mask_iou by condition x method ===")
    print(by_condition["mask_iou"].unstack("method").round(3)[METHODS])


if __name__ == "__main__":
    main()
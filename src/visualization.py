"""
visualization.py

Drawing and plotting helpers used by the Streamlit app and the batch
scripts.

Functions:

    plot_histogram(histogram, title) -> Figure
        256-bin intensity histogram.

    plot_degradation_curve(levels, curves, x_label, title) -> Figure
        IoU vs. degradation level, one line per segmentation method.

    draw_detections(image, objects, draw_ids) -> np.ndarray
        Contours, bounding boxes, centroids and ids drawn on a copy of
        an image (BGR output).

    plot_method_comparison(metrics_df, metric, ...) -> Figure
        Bar chart: mean of one benchmark metric per method.

    plot_condition_comparison(metrics_df, metric, ...) -> Figure
        Grouped bars: the metric per method, split by dataset condition.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_histogram(histogram: np.ndarray, title: str = "Grayscale Intensity Histogram"):
    """
    Render a 256-bin intensity histogram as a Matplotlib bar/line plot.

    What it shows:
        X-axis: intensity value (0 = black .. 255 = white).
        Y-axis: number of pixels at that intensity.

    Why this view matters:
        A histogram with two clear peaks (bimodal) is exactly the
        condition Otsu's method is designed to exploit --
        it searches for the threshold in the "valley" between two such
        peaks. A flat or single-peaked (unimodal) histogram signals that
        global/Otsu thresholding may struggle on that particular image,
        which is a useful thing to be able to see directly before any
        segmentation is even run.

    Args:
        histogram: np.ndarray of shape (256,), as returned by
            preprocessing.compute_histogram().
        title: plot title.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(256), histogram, color="black", linewidth=1)
    ax.fill_between(range(256), histogram, color="gray", alpha=0.4)
    ax.set_xlabel("Intensity value")
    ax.set_ylabel("Pixel count")
    ax.set_title(title)
    ax.set_xlim(0, 255)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Advanced benchmarking: degradation curve
# ---------------------------------------------------------------------------

def plot_degradation_curve(levels, curves: dict, x_label: str,
                           title: str = "IoU vs. Degradation Level"):
    """
    Plot IoU against degradation level, one line per segmentation method.

    A line that stays high as the level increases means the method is
    robust; the level where it drops sharply is that method's failure
    point.

    Args:
        levels: sequence of degradation levels (x-axis values).
        curves: {method_key: [IoU at each level]}, keys being
            "global", "otsu", "adaptive", "edge".
        x_label: axis label, e.g. "Noise variance".
        title: plot title.

    Returns:
        matplotlib.figure.Figure
    """
    labels = {
        "global": "Global",
        "otsu": "Otsu",
        "adaptive": "Adaptive",
        "edge": "Edge-Based",
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, ious in curves.items():
        ax.plot(levels, ious, marker="o", linewidth=1.5, label=labels.get(key, key))
    ax.set_xlabel(x_label)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Phase 4: detection overlay
# ---------------------------------------------------------------------------

def draw_detections(image: np.ndarray, objects: list, draw_ids: bool = True) -> np.ndarray:
    """
    Draw every detected object on a copy of an image.

    For each object it draws:
        - the outer contour (green)
        - the bounding box (blue-orange)
        - the centroid (red dot)
        - the object id next to the box (red text), if draw_ids is True

    Args:
        image: (H, W) grayscale or (H, W, 3) BGR image. It is never
            modified; a copy is drawn on.
        objects: list of dicts from detection.detect_objects().
        draw_ids: whether to write the object id.

    Returns:
        (H, W, 3) uint8 BGR image. Convert with [:, :, ::-1] before
        passing it to st.image().
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for obj in objects:
        x, y, w, h = obj["bbox"]
        cx, cy = obj["centroid"]
        cv2.drawContours(canvas, [obj["contour"]], -1, (0, 200, 0), 2)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), 1)
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)
        if draw_ids:
            cv2.putText(canvas, str(obj["id"]), (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas

# ---------------------------------------------------------------------------
# Phase 7: benchmark comparison charts
# ---------------------------------------------------------------------------

_METHOD_ORDER = ["global", "otsu", "adaptive", "edge"]
_METHOD_LABELS = {"global": "Global", "otsu": "Otsu",
                  "adaptive": "Adaptive", "edge": "Edge-Based"}
_METHOD_COLORS = {"global": "#4C72B0", "otsu": "#55A868",
                  "adaptive": "#DD8452", "edge": "#C44E52"}
_CONDITION_ORDER = ["normal", "noisy", "uneven", "touching"]
_UNIT_INTERVAL_METRICS = {"count_accuracy", "mean_iou", "mask_iou", "raw_mask_iou"}


def _style_metric_axis(ax, metric: str, ylabel: str = None) -> None:
    """Shared axis styling: 0-1 range for ratio metrics, light grid."""
    ax.set_ylabel(ylabel or metric)
    if metric in _UNIT_INTERVAL_METRICS:
        ax.set_ylim(0, 1.1)
    else:
        ax.margins(y=0.15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def plot_method_comparison(metrics_df, metric: str = "count_accuracy",
                           title: str = None, ylabel: str = None):
    """
    Bar chart comparing the four methods on one metric.

    What it shows:
        One bar per method: the MEAN of `metric` over every row for
        that method (all images). NaN values (e.g. mean_iou on an image
        where nothing matched) are skipped by the mean.

    Why the mean over all images:
        It summarizes overall performance, but it hides the per-
        condition story. Use plot_condition_comparison() for that.

    Args:
        metrics_df: DataFrame from run_benchmark.py, with a "method"
            column and the `metric` column.
        metric: column to plot, e.g. "count_accuracy", "mask_iou",
            "mean_iou", "false_positives", "processing_time_ms".
        title: plot title (default: the metric name).
        ylabel: y-axis label (default: the metric name).

    Returns:
        matplotlib.figure.Figure
    """
    means = metrics_df.groupby("method")[metric].mean()
    methods = [m for m in _METHOD_ORDER if m in means.index]
    values = [float(means[m]) for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([_METHOD_LABELS[m] for m in methods], values,
                  color=[_METHOD_COLORS[m] for m in methods])
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", xytext=(0, 3),
                        textcoords="offset points", fontsize=9)
    ax.set_title(title or f"{metric} by method")
    _style_metric_axis(ax, metric, ylabel)
    fig.tight_layout()
    return fig


def plot_condition_comparison(metrics_df, metric: str = "count_accuracy",
                              title: str = None, ylabel: str = None):
    """
    Grouped bar chart: method performance broken down by dataset
    condition (normal / noisy / uneven / touching).

    What it shows:
        One group of bars per condition, one bar per method, each the
        mean of `metric` over the images in that condition. This is the
        chart that shows WHERE each method breaks (e.g. global/Otsu on
        uneven lighting, edge-based on noise).

    Args:
        metrics_df: DataFrame from run_benchmark.py, with "condition"
            and "method" columns and the `metric` column.
        metric: column to plot.
        title: plot title (default: the metric name).
        ylabel: y-axis label (default: the metric name).

    Returns:
        matplotlib.figure.Figure
    """
    table = metrics_df.groupby(["condition", "method"])[metric].mean().unstack("method")
    conditions = [c for c in _CONDITION_ORDER if c in table.index]
    methods = [m for m in _METHOD_ORDER if m in table.columns]
    table = table.reindex(index=conditions, columns=methods)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(methods)
    x = np.arange(len(conditions))
    for k, m in enumerate(methods):
        offsets = x + (k - (len(methods) - 1) / 2) * width
        vals = table[m].to_numpy(dtype=float)
        bars = ax.bar(offsets, vals, width, label=_METHOD_LABELS[m],
                      color=_METHOD_COLORS[m])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                            ha="center", va="bottom", xytext=(0, 2),
                            textcoords="offset points", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title(title or f"{metric} by condition and method")
    ax.legend(ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.08), frameon=False)
    _style_metric_axis(ax, metric, ylabel)
    fig.tight_layout()
    return fig
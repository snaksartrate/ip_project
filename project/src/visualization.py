"""
visualization.py

Drawing and plotting helpers used by the Streamlit app: overlaying
bounding boxes/labels on detected objects, and generating comparison
charts (bar charts of accuracy/IoU/time across methods and dataset
conditions).

Planned functions (to be implemented in Phase 4/10):

    draw_detections(image, objects) -> np.ndarray
        Draws bounding boxes, centroids, and labels for each detected
        object on a copy of the original (color) image.

    plot_histogram(histogram) -> matplotlib.figure.Figure

    plot_method_comparison(metrics_df, metric="count_accuracy") -> matplotlib.figure.Figure
        Bar chart comparing the four methods on a chosen metric.

    plot_condition_comparison(metrics_df) -> matplotlib.figure.Figure
        Grouped bar chart: method performance broken down by dataset
        condition (normal / noisy / uneven illumination / touching).

PHASE 2 NOTE:
plot_histogram() is implemented now (moved up from its original Phase
4/10 slot) because Phase 2 requires displaying the intensity histogram
in the UI. Everything else in this module remains a future-phase stub.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(histogram: np.ndarray, title: str = "Grayscale Intensity Histogram"):
    """
    Render a 256-bin intensity histogram as a Matplotlib bar/line plot.

    What it shows:
        X-axis: intensity value (0 = black .. 255 = white).
        Y-axis: number of pixels at that intensity.

    Why this view matters:
        A histogram with two clear peaks (bimodal) is exactly the
        condition Otsu's method (a later phase) is designed to exploit --
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


# Remaining functions (draw_detections, plot_method_comparison,
# plot_condition_comparison) are implemented in Phase 4/10.

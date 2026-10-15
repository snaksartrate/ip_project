"""
app.py

Streamlit entrypoint for the "Comparative Image Segmentation and Object
Measurement System" project.

app.py is a UI controller: it lays out the page, collects inputs, calls
functions from src/ and run_benchmark.py, and displays results. The only
non-UI code here is small helpers and two cached wrappers (the stress-test
sweep and the dataset loader).

Layout:
    Upload + preprocessing settings (sidebar)
    Ground-truth inputs (shared by Stress-Test, Auto-Tuning, Measurements)
    Tabs: Pipeline | Stress-Test | Auto-Tuning | Hardware Profile |
          Measurements | Benchmark

Mask convention: uint8, 255 = foreground, 0 = background.
"""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocessing, segmentation, visualization
from src import degradation, evaluation, optimization, profiling
from src import morphology, detection, features

import run_benchmark  # batch runner; imports only src/, no Streamlit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "Global Thresholding": "global",
    "Otsu Thresholding": "otsu",
    "Adaptive Thresholding": "adaptive",
    "Edge-Based Segmentation": "edge",
}

SHORT_LABELS = {
    "global": "Global",
    "otsu": "Otsu",
    "adaptive": "Adaptive",
    "edge": "Edge-Based",
}

# Per-degradation UI configuration.
#   start        -> the "no degradation" value (first point of the sweep)
#   max_lo/hi/default/step -> slider that sets the maximum severity
#   preview_step -> step of the single-level preview slider
DEGRADATION_CONFIG = {
    "Gaussian noise": {
        "axis": "Noise variance", "start": 0.0,
        "max_lo": 100.0, "max_hi": 4000.0, "max_default": 1600.0, "step": 100.0,
        "preview_step": 50.0, "is_int": False,
    },
    "Low light (gamma)": {
        "axis": "Gamma (higher = darker)", "start": 1.0,
        "max_lo": 1.5, "max_hi": 6.0, "max_default": 4.0, "step": 0.5,
        "preview_step": 0.1, "is_int": False,
    },
    "Motion blur": {
        "axis": "Blur length (px)", "start": 1,
        "max_lo": 3, "max_hi": 61, "max_default": 31, "step": 2,
        "preview_step": 2, "is_int": True,
    },
}

NOISE_SEED = 42  # fixed seed -> reproducible noise, stable IoU curves


# ---------------------------------------------------------------------------
# Helpers (small, UI-side)
# ---------------------------------------------------------------------------

def _signature(*parts) -> str:
    """Hash arrays and settings into a short key, used to detect stale results."""
    h = hashlib.md5()
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else str(p).encode())
    return h.hexdigest()


def _apply_degradation(image, kind, level, angle, seed):
    """Dispatch to the matching degradation function."""
    if kind == "Gaussian noise":
        return degradation.inject_gaussian_noise(image, float(level), seed=seed)
    if kind == "Low light (gamma)":
        return degradation.inject_low_light(image, float(level))
    return degradation.inject_motion_blur(image, int(level), float(angle))


def _build_levels(kind, max_level, n_steps):
    """Evenly spaced degradation levels from 'none' up to max_level."""
    cfg = DEGRADATION_CONFIG[kind]
    raw = np.linspace(cfg["start"], max_level, n_steps)
    if cfg["is_int"]:
        # Motion-blur lengths must be odd integers.
        return sorted(set((np.round(raw).astype(int) | 1).tolist()))
    return [round(float(v), 3) for v in raw]


def _method_params(global_threshold, polarity):
    """Default parameters for all four methods (same for every level)."""
    return {
        "global": {"threshold": global_threshold, "object_polarity": polarity},
        "otsu": {"object_polarity": polarity},
        "adaptive": {"block_size": 11, "C": 2, "adaptive_method": "gaussian",
                     "object_polarity": polarity},
        "edge": {"low_threshold": 50, "high_threshold": 150, "closing_kernel_size": 5},
    }


@st.cache_data(show_spinner=False)
def cached_stress_sweep(gray, gt, kind, levels, angle, seed, polarity, global_threshold):
    """
    Degrade the image at every level, run all four methods, and return
    {method_key: [IoU per level]}. Cached so that re-running with the
    same settings is instant.
    """
    params = _method_params(global_threshold, polarity)
    curves = {key: [] for key in params}
    for level in levels:
        degraded = _apply_degradation(gray, kind, level, angle, seed)
        for key, p in params.items():
            seg = segmentation.segment(degraded, key, **p)
            curves[key].append(evaluation.calculate_iou(seg["mask"], gt))
    return curves

DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")

BENCH_METRICS = ["count_accuracy", "mask_iou", "mean_iou", "raw_mask_iou",
                 "false_positives", "missed", "mean_measurement_error_pct",
                 "processing_time_ms"]


@st.cache_data(show_spinner=False)
def cached_dataset(dataset_dir: str):
    """Load the dataset once (images, masks, GT objects) and reuse it."""
    return run_benchmark.load_dataset(Path(dataset_dir))


def _bench_cfg(noise: str, kernel_size: int) -> argparse.Namespace:
    """Benchmark settings, same defaults as run_benchmark.py."""
    return argparse.Namespace(
        noise=noise, kernel_size=kernel_size, open_kernel=3, close_kernel=5,
        min_area=50, global_threshold=127, iou_threshold=0.5)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Comparative Image Segmentation System",
    layout="wide",
)

st.title("Comparative Image Segmentation and Object Measurement System")
st.caption(
    "Global vs. Otsu vs. Adaptive Thresholding vs. Edge-Based Segmentation "
    "-- classical image processing, no deep learning"
)

st.info(
    "**Segmentation, measurement and benchmarking**\n\n"
    "Pipeline: preprocessing and the four segmentation methods. "
    "Stress-Test, Auto-Tuning and Hardware Profile: robustness under "
    "degradation, grid-search tuned parameters, and time/RAM footprint. "
    "Measurements: shared cleanup, object detection and per-object "
    "features. Benchmark: all four methods scored against exact "
    "ground truth on the synthetic dataset."
)

with st.sidebar:
    st.header("Module status")
    for name in ["preprocessing", "segmentation", "degradation", "optimization",
                 "profiling", "morphology", "detection", "features",
                 "evaluation", "visualization"]:
        st.success(f"src.{name} -- imported OK")

    st.divider()
    st.header("Preprocessing settings")

    max_dimension = st.slider(
        "Max image dimension (px)",
        min_value=200,
        max_value=2000,
        value=800,
        step=50,
        help="Images larger than this on their longest side are downscaled "
             "(aspect ratio preserved) before processing.",
    )

    noise_method = st.radio(
        "Noise reduction method",
        options=["None", "Gaussian", "Median"],
        index=0,
        help="Gaussian smoothing: good general-purpose blur, softens edges. "
             "Median filtering: specifically effective against salt-and-pepper "
             "(impulse) noise while preserving edges better.",
    )

    kernel_size = 5
    if noise_method != "None":
        kernel_size = st.slider(
            "Filter kernel size",
            min_value=1 if noise_method == "Gaussian" else 3,
            max_value=31,
            value=5,
            step=2,
            help="Must be odd. Larger values remove more noise but blur/erode "
                 "more detail.",
        )

    show_equalization_experiment = st.checkbox(
        "Show optional histogram equalization experiment",
        value=False,
        help="Purely for viewing -- does NOT alter the standard preprocessing "
             "pipeline used for the histogram/output below.",
    )


# ---------------------------------------------------------------------------
# Upload, validate, preprocess (shared by every tab)
# ---------------------------------------------------------------------------

st.subheader("1. Upload an image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG, BMP",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is None:
    st.write("No image uploaded yet.")
    st.stop()

image_bgr, error = preprocessing.load_and_validate_image(uploaded_file)

if error is not None:
    st.error(f"Could not process this file: {error}")
    st.stop()

info = preprocessing.get_image_info(image_bgr)

result = preprocessing.preprocess(
    image_bgr,
    max_dimension=max_dimension,
    noise_reduction_method=noise_method.lower(),
    kernel_size=kernel_size,
)


# ---------------------------------------------------------------------------
# Ground-truth inputs (shared by Stress-Test, Auto-Tuning, Hardware Profile
# and Measurements)
# ---------------------------------------------------------------------------

def render_ground_truth_inputs(filtered: np.ndarray):
    """
    Collect object polarity and a ground-truth mask.

    Returns:
        (polarity, gt_mask or None, gt_source or None)
        gt_mask is uint8 {0, 255} at the same size as `filtered`.
    """
    gt_mask, gt_source = None, None

    with st.expander("Ground truth & polarity (used by Stress-Test, Auto-Tuning, "
                    "Hardware Profile and Measurements)"):
        polarity_label = st.radio(
            "Object polarity",
            options=["Objects darker than background", "Objects brighter than background"],
            index=0,
            key="shared_polarity",
            horizontal=True,
            help="Used by Global/Otsu/Adaptive in the Stress-Test, Auto-Tuning "
                 "and Hardware Profile tabs.",
        )
        gt_file = st.file_uploader(
            "Ground-truth mask (optional): white = object, black = background",
            type=["png", "jpg", "jpeg", "bmp"],
            key="gt_uploader",
            help="Should show the same scene as the uploaded image. It is "
                 "resized to the preprocessed image size automatically.",
        )
        use_pseudo = st.checkbox(
            "No mask? Use Otsu's output as a pseudo ground truth (demo only)",
            value=False,
            key="use_pseudo_gt",
            help="Only for trying the tabs out. IoU against Otsu's own mask "
                 "is biased in Otsu's favour and is not a real accuracy figure.",
        )

        polarity = "darker" if "darker" in polarity_label else "brighter"
        target_h, target_w = filtered.shape[:2]

        if gt_file is not None:
            gt_bgr, gt_error = preprocessing.load_and_validate_image(gt_file)
            if gt_error is not None:
                st.error(f"Could not read the ground-truth mask: {gt_error}")
            else:
                gt_gray = preprocessing.convert_to_grayscale(gt_bgr)
                gh, gw = gt_gray.shape[:2]
                if abs(gw / gh - target_w / target_h) > 0.02:
                    st.warning(
                        "The mask's aspect ratio differs from the image's. "
                        "Resizing will distort it, so IoU values may be unreliable."
                    )
                if (gh, gw) != (target_h, target_w):
                    gt_gray = cv2.resize(gt_gray, (target_w, target_h),
                                         interpolation=cv2.INTER_NEAREST)
                gt_mask = np.where(gt_gray > 127, 255, 0).astype(np.uint8)
                gt_source = "uploaded mask"
        elif use_pseudo:
            gt_mask = segmentation.segment_otsu(filtered, polarity)["mask"]
            gt_source = "pseudo (Otsu)"

        if gt_mask is not None:
            fg_pct = 100.0 * np.count_nonzero(gt_mask) / gt_mask.size
            st.caption(f"Ground truth: {gt_source} -- {fg_pct:.1f}% foreground.")
            st.image(gt_mask, caption="Ground-truth mask in use", width=250)

    return polarity, gt_mask, gt_source


polarity, gt_mask, gt_source = render_ground_truth_inputs(result["filtered"])


# ---------------------------------------------------------------------------
# TAB 1: existing pipeline (unchanged logic)
# ---------------------------------------------------------------------------

def render_pipeline_tab():
    # --- Original image + info ---------------------------------------------
    st.subheader("2. Original image")

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(image_bgr[:, :, ::-1], caption="Original (unmodified)", width=450)
    with col_info:
        st.markdown("**Image information**")
        st.table(
            {
                "Property": ["Height", "Width", "Channels", "Dtype", "Approx. size"],
                "Value": [
                    f"{info['height']} px",
                    f"{info['width']} px",
                    info["channels"],
                    info["dtype"],
                    f"{info['size_kb']} KB",
                ],
            }
        )

    # --- Resize step -----------------------------------------------------------
    st.subheader("3. Resizing")
    if result["was_resized"]:
        resized_info = preprocessing.get_image_info(result["resized"])
        st.write(
            f"Original size ({info['height']}x{info['width']}) exceeded the "
            f"{max_dimension}px limit on its longest side -> resized to "
            f"**{resized_info['height']}x{resized_info['width']}** (aspect ratio preserved)."
        )
        st.image(result["resized"][:, :, ::-1], caption="Resized image", width=400)
    else:
        st.write(
            f"Image ({info['height']}x{info['width']}) is already within the "
            f"{max_dimension}px limit -- no resizing applied."
        )

    # --- Grayscale step --------------------------------------------------------
    st.subheader("4. Grayscale conversion")
    st.write(
        "Converted using OpenCV's BGR->Gray weighted sum "
        "(Gray = 0.299R + 0.587G + 0.114B), which approximates human "
        "perceptual brightness sensitivity. A single-channel intensity image "
        "is required by the thresholding-based segmentation methods."
    )
    col_gray1, col_gray2 = st.columns(2)
    with col_gray1:
        st.image(result["resized"][:, :, ::-1], caption="Before (color)", width=350)
    with col_gray2:
        st.image(result["gray"], caption="After (grayscale)", width=350)

    # --- Noise reduction step ----------------------------------------------------
    st.subheader("5. Noise reduction")
    if noise_method == "None":
        st.write("No filtering applied (selected method: None).")
    else:
        st.write(
            f"Applied **{noise_method}** filtering with kernel size **{kernel_size}**."
        )
        if noise_method == "Gaussian":
            st.caption(
                "Gaussian: weighted average using a bell-curve kernel. Smooths "
                "general sensor noise well but also softens edges, and does not "
                "specifically remove isolated outlier (salt-and-pepper) pixels."
            )
        else:
            st.caption(
                "Median: replaces each pixel with the median of its neighborhood. "
                "Highly effective against salt-and-pepper noise since outlier "
                "pixels barely influence a median, while edges are preserved "
                "better than with Gaussian blur."
            )
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.image(result["gray"], caption="Before filtering", width=350)
    with col_f2:
        st.image(result["filtered"], caption="After filtering", width=350)

    # --- Histogram step ----------------------------------------------------------
    st.subheader("6. Intensity histogram")
    st.write(
        "Computed on the final preprocessed (post-filter) grayscale image -- "
        "this is the histogram later thresholding methods (e.g. Otsu) will "
        "actually operate on."
    )
    hist_fig = visualization.plot_histogram(result["histogram"])
    st.pyplot(hist_fig)

    if show_equalization_experiment:
        st.markdown("**Optional experiment: histogram equalization**")
        st.caption(
            "For viewing only -- does not feed into the pipeline above or any "
            "later step by default."
        )
        equalized = preprocessing.equalize_histogram(result["filtered"])
        eq_hist = preprocessing.compute_histogram(equalized)
        col_eq1, col_eq2 = st.columns(2)
        with col_eq1:
            st.image(equalized, caption="Equalized image", width=350)
        with col_eq2:
            st.pyplot(visualization.plot_histogram(eq_hist, title="Equalized Histogram"))

    st.divider()

    # --- Segmentation step ---------------------------------------------------------
    st.subheader("7. Segmentation")

    col_method, col_polarity = st.columns(2)
    with col_method:
        method_label = st.selectbox("Segmentation method", options=list(METHOD_LABELS.keys()))
        method_key = METHOD_LABELS[method_label]
    with col_polarity:
        if method_key in ("global", "otsu", "adaptive"):
            polarity_label = st.radio(
                "Object polarity",
                options=["Objects darker than background", "Objects brighter than background"],
                index=0,
                help="Which side of the threshold counts as the foreground object.",
            )
            object_polarity = "darker" if "darker" in polarity_label else "brighter"
        else:
            st.caption("Edge-based segmentation fills enclosed regions directly -- "
                       "no polarity choice needed.")
            object_polarity = None

    st.markdown(f"**Parameters for {method_label}**")

    seg_params = {}
    if method_key == "global":
        seg_params["threshold"] = st.slider("Threshold value", 0, 255, 127)
        seg_params["object_polarity"] = object_polarity

    elif method_key == "otsu":
        st.caption("No threshold slider -- Otsu computes it automatically from the histogram above.")
        seg_params["object_polarity"] = object_polarity

    elif method_key == "adaptive":
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            seg_params["block_size"] = st.slider("Block size (neighborhood)", 3, 51, 11, step=2)
        with col_a2:
            seg_params["C"] = st.slider("Constant C", -20, 20, 2)
        with col_a3:
            seg_params["adaptive_method"] = st.selectbox("Local statistic", ["gaussian", "mean"])
        seg_params["object_polarity"] = object_polarity

    elif method_key == "edge":
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            seg_params["low_threshold"] = st.slider("Canny low threshold", 0, 255, 50)
        with col_e2:
            seg_params["high_threshold"] = st.slider("Canny high threshold", 0, 255, 150)
        with col_e3:
            seg_params["closing_kernel_size"] = st.slider("Gap-closing kernel size", 1, 25, 5, step=2)

    run_segmentation = st.button("Run Segmentation", type="primary")

    if run_segmentation:
        seg_result = segmentation.segment(result["filtered"], method_key, **seg_params)

        st.markdown("### Segmentation input")
        st.image(result["filtered"], caption="Preprocessed grayscale image (input to segmentation)", width=400)

        st.markdown("### Segmentation output")
        st.image(seg_result["mask"], caption=f"Binary mask ({method_label}) -- white = foreground", width=400)

        st.markdown("### Method information")
        info_rows = {
            "Method": method_label,
            "Processing time": f"{seg_result['processing_time_ms']:.2f} ms",
        }
        if seg_result["threshold"] is not None:
            info_rows["Threshold used"] = f"{seg_result['threshold']:.1f}"
        for k, v in seg_result["parameters"].items():
            info_rows[k] = str(v)
        st.table({"Property": list(info_rows.keys()), "Value": list(info_rows.values())})

        if method_key == "edge":
            st.markdown("### Edge-based pipeline detail")
            st.write("Grayscale → Canny edge map → gap-closed edges → filled binary mask")
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)
            with col_e1:
                st.image(result["filtered"], caption="Grayscale", width=220)
            with col_e2:
                st.image(seg_result["intermediate"]["edges"], caption="Canny edges", width=220)
            with col_e3:
                st.image(seg_result["intermediate"]["closed_edges"], caption="Gap-closed edges", width=220)
            with col_e4:
                st.image(seg_result["mask"], caption="Filled mask", width=220)
    else:
        st.write("Set parameters above and click **Run Segmentation**.")

    st.divider()
    st.caption(
        "Cleanup, detection and measurements are in the Measurements tab; "
        "dataset benchmarking is in the Benchmark tab."
    )


# ---------------------------------------------------------------------------
# TAB 2: stress-test
# ---------------------------------------------------------------------------

def render_stress_tab():
    st.subheader("Robustness stress-test")
    st.write(
        "Degrade the preprocessed image at increasing severity, run all four "
        "methods with fixed default parameters, and plot IoU against the "
        "degradation level. The ground truth stays the same throughout."
    )

    if gt_mask is None:
        st.info("Provide a ground-truth mask (or enable the pseudo ground truth) "
                "in the **Ground truth & polarity** section above.")
        return
    if gt_source == "pseudo (Otsu)":
        st.warning("Using Otsu's output as ground truth: the curves are only a demo, "
                   "and Otsu is favoured at level 0.")

    gray = result["filtered"]
    kind = st.selectbox("Degradation type", list(DEGRADATION_CONFIG.keys()), key="stress_kind")
    cfg = DEGRADATION_CONFIG[kind]
    cast = int if cfg["is_int"] else float

    col1, col2, col3 = st.columns(3)
    with col1:
        max_level = st.slider(
            f"Maximum: {cfg['axis']}",
            cast(cfg["max_lo"]), cast(cfg["max_hi"]), cast(cfg["max_default"]),
            step=cast(cfg["step"]), key=f"stress_max_{kind}",
        )
    with col2:
        n_steps = st.slider("Number of levels", 3, 15, 8, key="stress_steps")
    with col3:
        global_t = st.slider("Global threshold (fixed)", 0, 255, 127, key="stress_global_t")

    angle = 0
    if kind == "Motion blur":
        angle = st.slider("Motion angle (degrees)", 0, 180, 0, key="stress_angle")

    # --- Single-level preview (cheap, updates live) ---------------------------
    st.markdown("#### Preview one level")
    preview_level = st.slider(
        cfg["axis"], cast(cfg["start"]), cast(max_level), cast(cfg["start"]),
        step=cast(cfg["preview_step"]), key=f"stress_preview_{kind}",
    )
    degraded = _apply_degradation(gray, kind, preview_level, angle, NOISE_SEED)
    params = _method_params(global_t, polarity)

    cols = st.columns(5)
    with cols[0]:
        st.image(degraded, caption="Degraded input", width=220)
    for col, key in zip(cols[1:], params.keys()):
        seg = segmentation.segment(degraded, key, **params[key])
        iou = evaluation.calculate_iou(seg["mask"], gt_mask)
        with col:
            st.image(seg["mask"], caption=f"{SHORT_LABELS[key]} (IoU {iou:.3f})", width=220)

    # --- Full sweep -------------------------------------------------------------
    st.markdown("#### Full sweep")
    if st.button("Run Stress Test", type="primary", key="run_stress"):
        levels = _build_levels(kind, max_level, n_steps)
        with st.spinner(f"Running {len(levels)} levels x 4 methods..."):
            curves = cached_stress_sweep(
                gray, gt_mask, kind, levels, angle, NOISE_SEED, polarity, global_t
            )
        st.session_state["stress_result"] = {
            "kind": kind, "levels": levels, "curves": curves, "axis": cfg["axis"],
        }

    stored = st.session_state.get("stress_result")
    if stored is None:
        st.write("Choose settings and click **Run Stress Test**.")
        return

    st.caption(f"Showing the last completed run: {stored['kind']}.")
    fig = visualization.plot_degradation_curve(
        stored["levels"], stored["curves"], x_label=stored["axis"],
        title=f"IoU vs. {stored['axis']}",
    )
    st.pyplot(fig)

    table = {"Level": stored["levels"]}
    for key, ious in stored["curves"].items():
        table[SHORT_LABELS[key]] = [round(v, 4) for v in ious]
    st.dataframe(pd.DataFrame(table), hide_index=True)


# ---------------------------------------------------------------------------
# TAB 3: auto-tuning (grid search)
# ---------------------------------------------------------------------------

def render_tuning_tab():
    st.subheader("Algorithmic auto-tuning (grid search)")
    st.write(
        "Tries every parameter combination in a fixed range and keeps the one "
        "with the highest IoU, removing human slider bias. "
        "Adaptive: 15 block sizes x 21 values of C = 315 runs. "
        "Edge-based: 5 low x 4 high Canny thresholds = 20 runs."
    )

    if gt_mask is None:
        st.info("Provide a ground-truth mask in the **Ground truth & polarity** "
                "section above. Grid search needs a real reference to optimise against.")
        return
    if gt_source == "pseudo (Otsu)":
        st.warning("Tuning against Otsu's output only finds the parameters that best "
                   "imitate Otsu. Use a real ground-truth mask for meaningful results.")

    gray = result["filtered"]
    col1, col2 = st.columns(2)
    with col1:
        adaptive_method = st.selectbox("Adaptive local statistic (fixed during search)",
                                       ["gaussian", "mean"], key="tune_adaptive_method")
    with col2:
        closing = st.slider("Edge gap-closing kernel size (fixed during search)",
                            1, 25, 5, step=2, key="tune_closing")

    signature = _signature(gray, gt_mask, polarity, adaptive_method, closing)

    if st.button("Run Grid Search", type="primary", key="run_tuning"):
        bar_a = st.progress(0.0, text="Adaptive grid search (315 combinations)")
        ad = optimization.grid_search_adaptive(
            gray, gt_mask, object_polarity=polarity, adaptive_method=adaptive_method,
            progress_callback=lambda f: bar_a.progress(f, text="Adaptive grid search"),
        )
        bar_e = st.progress(0.0, text="Edge-based grid search (20 combinations)")
        ed = optimization.grid_search_edge(
            gray, gt_mask, closing_kernel_size=closing,
            progress_callback=lambda f: bar_e.progress(f, text="Edge-based grid search"),
        )
        bar_a.empty()
        bar_e.empty()
        st.session_state["tuning"] = {"signature": signature, "adaptive": ad, "edge": ed}

    stored = st.session_state.get("tuning")
    if stored is None:
        st.write("Click **Run Grid Search** to start.")
        return
    if stored["signature"] != signature:
        st.info("Image, ground truth or settings changed since the last search. "
                "Click **Run Grid Search** again to refresh.")
        return

    ad, ed = stored["adaptive"], stored["edge"]

    st.markdown("#### Optimal parameters")
    st.dataframe(
        pd.DataFrame([
            {"Method": "Adaptive Thresholding",
             "Optimal parameters": f"block_size={ad['best_block_size']}, C={ad['best_C']}",
             "Max IoU": round(ad["best_iou"], 4),
             "Combinations tried": ad["n_combinations"]},
            {"Method": "Edge-Based Segmentation",
             "Optimal parameters": f"low={ed['best_low_threshold']}, high={ed['best_high_threshold']}",
             "Max IoU": round(ed["best_iou"], 4),
             "Combinations tried": ed["n_combinations"]},
        ]),
        hide_index=True,
    )

    st.markdown("#### Masks at the optimum")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(gt_mask, caption="Ground truth", width=300)
    with c2:
        st.image(ad["best_mask"], caption=f"Adaptive (IoU {ad['best_iou']:.3f})", width=300)
    with c3:
        st.image(ed["best_mask"], caption=f"Edge-based (IoU {ed['best_iou']:.3f})", width=300)

    with st.expander("Top 10 combinations per method"):
        top_a = sorted(ad["results"], key=lambda r: r["iou"], reverse=True)[:10]
        top_e = sorted(ed["results"], key=lambda r: r["iou"], reverse=True)[:10]
        st.markdown("**Adaptive**")
        st.dataframe(pd.DataFrame(top_a), hide_index=True)
        st.markdown("**Edge-based**")
        st.dataframe(pd.DataFrame(top_e), hide_index=True)


# ---------------------------------------------------------------------------
# TAB 4: hardware profile
# ---------------------------------------------------------------------------

def render_profile_tab():
    st.subheader("Hardware profile (edge-device feasibility)")
    st.write(
        "Measures execution time and RAM for each method on the current "
        "preprocessed image, using each method's default parameters."
    )

    gray = result["filtered"]
    repeats = st.slider("Runs per method", 3, 50, 10, key="profile_repeats",
                        help="Each method is run this many times inside one monitored "
                             "window; the median time is reported.")

    if st.button("Run Profiling", type="primary", key="run_profile"):
        overrides = {
            "global": {"object_polarity": polarity},
            "otsu": {"object_polarity": polarity},
            "adaptive": {"object_polarity": polarity},
        }
        try:
            with st.spinner("Profiling all four methods..."):
                rows = profiling.profile_all_methods(gray, method_params=overrides,
                                                     repeats=repeats)
            st.session_state["profile"] = {"rows": rows, "repeats": repeats}
        except Exception as exc:  # noqa: BLE001 - surface any profiler/OS issue to the user
            st.error(f"Profiling failed: {exc}")

    stored = st.session_state.get("profile")
    if stored is None:
        st.write("Click **Run Profiling** to measure.")
        return

    df = pd.DataFrame(stored["rows"])
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index("Method")[["Time (ms)"]])
    st.caption(
        f"Median of {stored['repeats']} runs per method. 'Peak RAM' is the whole "
        "Streamlit process (dominated by Python, OpenCV and Streamlit itself); "
        "'RAM increase' is the growth above the starting level and is only a "
        "rough figure. 'Alloc peak' is the peak memory allocated by Python/NumPy "
        "during one run and is the best indicator of the algorithm's own "
        "footprint (it does not include OpenCV's internal buffers, so it is a "
        "lower bound). Results are re-measured on every click (not cached)."
    )


# ---------------------------------------------------------------------------
# TAB 5: morphology, detection and measurements
# ---------------------------------------------------------------------------

def render_measure_tab():
    st.subheader("Morphology, detection and measurements")
    st.write(
        "Each method's raw mask goes through the SAME cleanup "
        "(opening -> closing -> small-region removal), then objects are "
        "detected and measured. Methods use fixed default parameters, so "
        "differences come from the segmentation itself. All measurements "
        "are in pixels."
    )

    gray = result["filtered"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        open_k = st.slider("Opening kernel", 1, 15, 3, step=2, key="meas_open")
    with c2:
        close_k = st.slider("Closing kernel", 1, 25, 5, step=2, key="meas_close")
    with c3:
        min_area = st.slider("Min object area (px)", 1, 2000, 50, key="meas_min_area")
    with c4:
        global_t = st.slider("Global threshold (fixed)", 0, 255, 127, key="meas_global_t")

    params = _method_params(global_t, polarity)

    runs = {}
    summary = []
    for key, p in params.items():
        seg = segmentation.segment(gray, key, **p)
        cleaned = morphology.clean_mask(seg["mask"], open_kernel_size=open_k,
                                        close_kernel_size=close_k, min_area=min_area)
        objects = detection.detect_objects(cleaned["final"], min_area=min_area)
        table = features.compute_all_features(objects)
        overlay = visualization.draw_detections(result["resized"], objects)
        runs[key] = {"raw": seg["mask"], "cleaned": cleaned, "objects": objects,
                     "table": table, "overlay": overlay}

        row = {
            "Method": SHORT_LABELS[key],
            "Objects": len(objects),
            "Foreground (%)": round(100.0 * np.count_nonzero(cleaned["final"])
                                    / cleaned["final"].size, 1),
            "Mean area (px)": round(float(table["area"].mean()), 1) if len(table) else 0.0,
            "Mean circularity": round(float(table["circularity"].mean()), 3) if len(table) else 0.0,
        }
        if gt_mask is not None:
            row["IoU raw"] = round(evaluation.calculate_iou(seg["mask"], gt_mask), 4)
            row["IoU cleaned"] = round(evaluation.calculate_iou(cleaned["final"], gt_mask), 4)
        summary.append(row)

    st.markdown("#### Detections per method")
    cols = st.columns(4)
    for col, key in zip(cols, params.keys()):
        with col:
            st.image(runs[key]["overlay"][:, :, ::-1],
                     caption=f"{SHORT_LABELS[key]}: {len(runs[key]['objects'])} objects",
                     use_container_width=True)

    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame(summary), hide_index=True)
    if gt_mask is None:
        st.caption("Add a ground-truth mask in the **Ground truth & polarity** "
                   "section above to also see IoU before and after cleaning.")
    elif gt_source == "pseudo (Otsu)":
        st.caption("IoU is measured against Otsu's own output (pseudo ground truth), "
                   "so it favours Otsu and is only a demo.")

    st.markdown("#### Method detail")
    detail_key = st.selectbox(
        "Method", list(params.keys()), format_func=lambda k: SHORT_LABELS[k],
        key="meas_detail_method",
    )
    run = runs[detail_key]
    steps = [("Raw mask", run["raw"]),
             ("After opening", run["cleaned"]["opened"]),
             ("After closing", run["cleaned"]["closed"]),
             ("Final (small regions removed)", run["cleaned"]["final"])]
    for col, (caption, img) in zip(st.columns(4), steps):
        with col:
            st.image(img, caption=caption, use_container_width=True)

    st.markdown("**Measurements**")
    if len(run["table"]) == 0:
        st.write("No objects detected with the current settings.")
    else:
        st.dataframe(run["table"], hide_index=True)
        st.download_button(
            "Download measurements (CSV)",
            run["table"].to_csv(index=False).encode("utf-8"),
            file_name=f"measurements_{detail_key}.csv",
            mime="text/csv",
            key="meas_download",
        )
        


# ---------------------------------------------------------------------------
# TAB 6: benchmark on the synthetic dataset
# ---------------------------------------------------------------------------

def render_benchmark_tab():
    st.subheader("Benchmark on the synthetic dataset")
    st.write(
        "All four methods run on every dataset image with identical "
        "preprocessing, cleanup and detection, scored against exact ground "
        "truth. This tab does not use the uploaded image."
    )

    if not (DATASET_DIR / "ground_truth" / "counts.csv").exists():
        st.info("Dataset not found. Run `python generate_dataset.py` from the "
                "project folder first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        bench_noise = st.selectbox("Noise reduction", ["none", "gaussian", "median"],
                                   key="bench_noise")
    with c2:
        bench_kernel = st.slider("Filter kernel size", 3, 31, 5, step=2,
                                 key="bench_kernel")
    cfg = _bench_cfg(bench_noise, bench_kernel)

    out_name = ("benchmark_results.csv" if bench_noise == "none"
                else f"results_{bench_noise}.csv")
    if st.button("Run benchmark", type="primary", key="bench_run"):
        with st.spinner("Running all images x 4 methods..."):
            run_benchmark.run_benchmark(DATASET_DIR, RESULTS_DIR, out_name, cfg)
        st.session_state["bench_file"] = out_name
        st.success(f"Wrote results/{out_name}")
    st.caption(f"Run output file: results/{out_name}. A different kernel size "
               "overwrites the file of the same noise method.")

    # --- Results ---------------------------------------------------------
    files = sorted(p.name for p in RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else []
    if not files:
        st.write("No results yet. Click **Run benchmark**.")
        return
    default = st.session_state.get("bench_file", files[0])
    chosen = st.selectbox("Results file", files,
                          index=files.index(default) if default in files else 0,
                          key="bench_choice")
    df = pd.read_csv(RESULTS_DIR / chosen)

    metric = st.selectbox("Metric to chart", BENCH_METRICS, key="bench_metric")
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(visualization.plot_method_comparison(df, metric))
    with col_b:
        st.pyplot(visualization.plot_condition_comparison(df, metric))

    st.markdown("#### Mean per method")
    summary = df.groupby("method")[BENCH_METRICS].mean().reindex(
        list(run_benchmark.METHOD_KEYS)).round(3)
    st.dataframe(summary)
    st.caption("mean_iou and mean_measurement_error_pct only cover images where "
               "the method matched at least one object. Prefer count_accuracy "
               "and mask_iou for headline comparisons.")

    # --- Browse images ---------------------------------------------------
    st.markdown("#### Browse images")
    st.caption("Recomputed live with the noise settings above, so it matches the "
               "results file only if that file was run with the same settings.")
    entries = cached_dataset(str(DATASET_DIR))
    name = st.selectbox("Image", [e["filename"] for e in entries], key="bench_image")
    entry = next(e for e in entries if e["filename"] == name)
    rows, masks = run_benchmark.run_image(entry, cfg)

    cols = st.columns(5)
    with cols[0]:
        st.image(entry["image"],
                 caption=f"{name}: {entry['n_objects']} true objects "
                         f"({entry['gt_blobs']} blobs)",
                 use_container_width=True)
    for col, key in zip(cols[1:], run_benchmark.METHOD_KEYS):
        objects = detection.detect_objects(masks[key], min_area=cfg.min_area)
        overlay = visualization.draw_detections(entry["image"], objects)
        with col:
            st.image(overlay[:, :, ::-1],
                     caption=f"{SHORT_LABELS[key]}: {len(objects)} detected",
                     use_container_width=True)

    st.dataframe(
        pd.DataFrame(rows)[["method", "gt_count", "gt_blobs", "detected_count",
                            "correct", "missed", "false_positives",
                            "count_accuracy", "mask_iou"]].round(3),
        hide_index=True)
    with st.expander("Ground-truth mask"):
        st.image(entry["gt_mask"], width=300)



# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pipeline, tab_stress, tab_tuning, tab_profile, tab_measure, tab_bench = st.tabs(
    ["Pipeline", "Stress-Test", "Auto-Tuning", "Hardware Profile",
     "Measurements", "Benchmark"]
)

with tab_pipeline:
    render_pipeline_tab()
with tab_stress:
    render_stress_tab()
with tab_tuning:
    render_tuning_tab()
with tab_profile:
    render_profile_tab()
with tab_measure:
    render_measure_tab()
with tab_bench:
    render_benchmark_tab()
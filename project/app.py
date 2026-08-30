"""
app.py

Streamlit entrypoint for the "Comparative Image Segmentation and Object
Measurement System" project.

PHASE STATUS:
    Phase 1 -- project setup                         DONE
    Phase 2 -- image input & preprocessing            DONE
    Phase 3 -- segmentation (4 methods)               DONE (this file)
    Phase 4+ -- morphology, detection, features,
                evaluation, comparison                 NOT YET IMPLEMENTED

app.py is intentionally kept thin: it handles UI layout, collects user
inputs, calls functions from src/preprocessing.py, src/segmentation.py,
and src/visualization.py, and displays results. No image-processing
logic lives in this file.
"""

import streamlit as st

from src import preprocessing, segmentation, visualization
from src import morphology, detection, features, evaluation  # noqa: F401 (imported for module-status sidebar only)

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
    "**Phase 3 -- Segmentation**\n\n"
    "Preprocessing (Phase 2) plus four segmentation methods -- Global, "
    "Otsu, Adaptive Thresholding, and Edge-Based Segmentation -- sharing "
    "the same preprocessed grayscale input. Morphology, object detection, "
    "feature extraction, and evaluation are not implemented yet -- they "
    "arrive in later phases."
)

with st.sidebar:
    st.header("Module status")
    for name in ["preprocessing", "segmentation", "morphology", "detection", "features", "evaluation", "visualization"]:
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

# --- Original image + info -------------------------------------------------
st.subheader("2. Original image")
info = preprocessing.get_image_info(image_bgr)

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

# --- Run preprocessing pipeline --------------------------------------------
result = preprocessing.preprocess(
    image_bgr,
    max_dimension=max_dimension,
    noise_reduction_method=noise_method.lower(),
    kernel_size=kernel_size,
)

# --- Resize step -------------------------------------------------------------
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

# --- Grayscale step ----------------------------------------------------------
st.subheader("4. Grayscale conversion")
st.write(
    "Converted using OpenCV's BGR->Gray weighted sum "
    "(Gray = 0.299R + 0.587G + 0.114B), which approximates human "
    "perceptual brightness sensitivity. A single-channel intensity image "
    "is required by the thresholding-based segmentation methods used in "
    "later phases."
)
col_gray1, col_gray2 = st.columns(2)
with col_gray1:
    st.image(result["resized"][:, :, ::-1], caption="Before (color)", width=350)
with col_gray2:
    st.image(result["gray"], caption="After (grayscale)", width=350)

# --- Noise reduction step -----------------------------------------------------
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

# --- Histogram step ------------------------------------------------------------
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
        "later phase by default."
    )
    equalized = preprocessing.equalize_histogram(result["filtered"])
    eq_hist = preprocessing.compute_histogram(equalized)
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.image(equalized, caption="Equalized image", width=350)
    with col_eq2:
        st.pyplot(visualization.plot_histogram(eq_hist, title="Equalized Histogram"))

st.divider()

# --- Segmentation step ------------------------------------------------------------
st.subheader("7. Segmentation")

METHOD_LABELS = {
    "Global Thresholding": "global",
    "Otsu Thresholding": "otsu",
    "Adaptive Thresholding": "adaptive",
    "Edge-Based Segmentation": "edge",
}

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
    "Morphology, object detection, measurements, and evaluation are "
    "implemented in later phases."
)

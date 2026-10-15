"""
segmentation.py

Implements four classical segmentation methods that all solve the SAME
problem -- separating foreground objects from background in a
preprocessed grayscale image -- under one common interface, so the
later comparison across methods is fair: only this stage differs,
everything before (preprocessing) and after (morphology, detection)
stays identical regardless of which method was chosen.

MASK CONVENTION (enforced by every function in this module):
    255 (white) = foreground object
    0   (black) = background
Every method normalizes to this convention explicitly rather than
relying on whichever polarity OpenCV happens to produce by default.

COMMON RESULT STRUCTURE:
    Every segment_*() function, and the segment() dispatcher, returns a
    dict with the same keys:
        {
            "mask": np.ndarray (H, W) uint8, values in {0, 255},
            "method": str,
            "parameters": dict,
            "threshold": float | None,   # None where not a single global value
            "processing_time_ms": float,
            "intermediate": dict,        # method-specific extra images, may be empty
        }
"""

from typing import Optional, Callable
import time
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_odd(value: int, min_size: int = 3, max_size: int = 99) -> int:
    """
    Force a value into a valid odd integer within [min_size, max_size].

    Why: adaptive thresholding's block size (like the filter kernel
    sizes in preprocessing.py) must be odd and >= 3 for OpenCV to accept
    it. A UI slider or manual parameter could still produce an invalid
    value; this guards the app from crashing on it.
    """
    value = int(value)
    value = max(min_size, min(max_size, value))
    if value % 2 == 0:
        value += 1
        value = min(value, max_size if max_size % 2 == 1 else max_size - 1)
    return value


def _make_result(mask: np.ndarray, method: str, parameters: dict,
                  threshold: Optional[float], processing_time_ms: float,
                  intermediate: Optional[dict] = None) -> dict:
    """Build the common result structure every segmentation method returns."""
    # Defensive normalization: guarantee the mask is strictly {0, 255} uint8,
    # regardless of what the underlying OpenCV call produced.
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return {
        "mask": mask,
        "method": method,
        "parameters": parameters,
        "threshold": threshold,
        "processing_time_ms": processing_time_ms,
        "intermediate": intermediate or {},
    }


def _polarity_to_thresh_type(object_polarity: str) -> int:
    """
    Translate a human-readable polarity choice into the OpenCV threshold
    type that yields our white-foreground convention.

    "darker"  -> objects have LOWER intensity than background
                 -> foreground = pixels BELOW threshold -> THRESH_BINARY_INV
    "brighter"-> objects have HIGHER intensity than background
                 -> foreground = pixels ABOVE threshold -> THRESH_BINARY

    This one function is the single place polarity is resolved, so all
    three thresholding-based methods (global/Otsu/adaptive) stay
    consistent with each other rather than each re-implementing the choice.
    """
    polarity = object_polarity.lower().strip()
    if polarity == "darker":
        return cv2.THRESH_BINARY_INV
    elif polarity == "brighter":
        return cv2.THRESH_BINARY
    else:
        raise ValueError(f"Unknown object_polarity: '{object_polarity}'. Expected 'darker' or 'brighter'.")


# ---------------------------------------------------------------------------
# 1. GLOBAL THRESHOLDING
# ---------------------------------------------------------------------------

def segment_global(gray_image: np.ndarray, threshold: int = 127, object_polarity: str = "darker") -> dict:
    """
    Fixed/global thresholding: every pixel is classified using the same,
    user-specified threshold value, regardless of local image content.

    Mathematical definition:
        For "darker objects":  mask(x,y) = 255 if I(x,y) <  T else 0
        For "brighter objects": mask(x,y) = 255 if I(x,y) >= T else 0
        where I(x,y) is the grayscale intensity and T is the fixed
        threshold supplied by the user.

    Why a single threshold:
        This is the simplest possible segmentation rule -- one global
        cutoff applied uniformly across the whole image. It is the
        baseline against which the automatic (Otsu) and local (adaptive)
        methods are meaningfully compared: it deliberately does NOT
        adapt to the image at all, so it isolates "what does a
        completely non-adaptive rule achieve?" as one point in the
        comparison.

    Advantages:
        - Extremely fast, trivial to compute and explain.
        - Fully predictable/deterministic given a fixed T.

    Limitations:
        - A single T can rarely fit varying lighting across an image --
          it will misclassify regions where illumination departs from
          what T was tuned for.
        - Requires manual tuning per image/lighting condition; no
          automatic adaptation.

    Works well when:
        - Illumination is uniform and foreground/background intensities
          are already well separated (e.g., a controlled, evenly-lit
          photo of coins on a plain background).

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image.
        threshold: fixed intensity cutoff, 0-255. NOT computed from the
            image -- this is intentionally a manual, fixed value.
        object_polarity: "darker" or "brighter" -- whether objects are
            darker or brighter than the background.

    Returns:
        Common result dict (see module docstring). "threshold" is
        exactly the value passed in (unchanged, since it is fixed).
    """
    threshold = int(max(0, min(255, threshold)))
    thresh_type = _polarity_to_thresh_type(object_polarity)

    start = time.perf_counter()
    _, mask = cv2.threshold(gray_image, threshold, 255, thresh_type)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return _make_result(
        mask=mask,
        method="global",
        parameters={"threshold": threshold, "object_polarity": object_polarity},
        threshold=float(threshold),
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# 2. OTSU THRESHOLDING
# ---------------------------------------------------------------------------

def segment_otsu(gray_image: np.ndarray, object_polarity: str = "darker") -> dict:
    """
    Otsu's method: automatically computes the threshold from the
    image's own histogram, rather than requiring a manual value.

    Between-class variance principle (the mathematical idea):
        Otsu searches over every possible threshold T (0-255) and picks
        the one that MAXIMIZES the between-class variance
            sigma_b^2(T) = w0(T) * w1(T) * (mu0(T) - mu1(T))^2
        where w0/w1 are the proportions of pixels below/above T, and
        mu0/mu1 are their mean intensities. Maximizing between-class
        variance is mathematically equivalent to minimizing the
        within-class (intra-class) variance of the two resulting
        groups -- i.e., it finds the T that makes the two sides of the
        histogram as internally homogeneous, and as different from each
        other, as possible.

    Relationship with the histogram:
        This search is performed directly over the intensity histogram
        computed in preprocessing (Phase 2) -- it is why that histogram
        display matters: Otsu is essentially asking "where is the best
        valley between two peaks in this histogram?"

    Why it's useful when distributions are separable:
        If foreground and background pixel intensities form two
        distinct clusters (a bimodal histogram), Otsu finds the
        statistically optimal separating point automatically -- no
        manual tuning needed, unlike global thresholding.

    Why it can fail:
        - Non-bimodal histograms (e.g., a single broad peak) have no
          clear valley for the between-class-variance search to lock
          onto, so the resulting threshold can be poorly placed.
        - Uneven illumination effectively smears what would be two
          clean peaks into a wider, less separable distribution (dark
          object pixels in a shadowed area can overlap with bright
          background pixels elsewhere in the image), which directly
          violates Otsu's core global-histogram assumption.

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image.
        object_polarity: "darker" or "brighter" -- resolves which side
            of the computed threshold becomes foreground; does NOT
            affect what threshold value Otsu computes, only how the
            two sides are labeled.

    Returns:
        Common result dict. "threshold" is the value Otsu computed
        (never manually set).
    """
    thresh_type = _polarity_to_thresh_type(object_polarity) | cv2.THRESH_OTSU

    start = time.perf_counter()
    computed_threshold, mask = cv2.threshold(gray_image, 0, 255, thresh_type)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return _make_result(
        mask=mask,
        method="otsu",
        parameters={"object_polarity": object_polarity},
        threshold=float(computed_threshold),
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# 3. ADAPTIVE THRESHOLDING
# ---------------------------------------------------------------------------

def segment_adaptive(gray_image: np.ndarray, block_size: int = 11, C: int = 2,
                      adaptive_method: str = "gaussian", object_polarity: str = "darker") -> dict:
    """
    Adaptive thresholding: computes a DIFFERENT threshold for every
    pixel, based only on its local neighborhood, instead of one value
    for the whole image.

    Why local thresholds are useful:
        A single global cutoff (global/Otsu) assumes brightness is
        roughly consistent across the image. When it isn't -- e.g. one
        corner is shadowed while another is bright -- no single T can
        correctly separate foreground from background everywhere at
        once. Adaptive thresholding sidesteps this by asking, for each
        pixel, "how does this pixel compare to *its own neighborhood*?"
        rather than "how does it compare to the whole image?"

    How it works (the math):
        For each pixel (x,y), a local threshold T(x,y) is computed from
        the `block_size` x `block_size` neighborhood around it:
            - "mean":     T(x,y) = mean(neighborhood) - C
            - "gaussian": T(x,y) = gaussian_weighted_mean(neighborhood) - C
        The pixel is then classified using this local T(x,y) instead of
        a single global value. Gaussian weighting gives closer pixels
        more influence than distant ones within the block, which
        usually gives a smoother, less noisy result than a flat mean.

    Why this handles uneven illumination better:
        Because T is recomputed locally, a shadowed region and a bright
        region can each get an appropriately different effective
        threshold, as long as objects remain distinguishable from their
        *immediate* surroundings -- even if they wouldn't be
        distinguishable from the image as a whole.

    Effect of block_size:
        - Too small: threshold follows noise too closely -> speckled,
          fragmented mask.
        - Too large: neighborhood starts behaving like a global
          threshold again, losing the local-adaptation benefit, and can
          blend across genuinely different lighting zones.

    Effect of C:
        A constant subtracted from the local mean before comparison.
        Higher C makes the threshold stricter (fewer pixels classified
        as foreground); lower/negative C makes it more permissive.

    Limitations:
        - Sensitive to block_size choice -- no single value is optimal
          for every image.
        - Can fragment large uniform regions into speckled noise if the
          neighborhood is too small relative to object size.
        - More computationally expensive than global/Otsu (a
          neighborhood computation per pixel rather than one global pass).

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image.
        block_size: neighborhood side length; must be odd and >= 3 --
            invalid values are corrected automatically.
        C: constant subtracted from the local mean/weighted-mean.
        adaptive_method: "mean" or "gaussian".
        object_polarity: "darker" or "brighter".

    Returns:
        Common result dict. "threshold" is None -- adaptive thresholding
        has no single representative threshold value, since a different
        one is computed at every pixel.
    """
    block_size = _validate_odd(block_size, min_size=3, max_size=99)
    C = int(C)

    method_key = adaptive_method.lower().strip()
    if method_key == "mean":
        cv2_adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method_key == "gaussian":
        cv2_adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError(f"Unknown adaptive_method: '{adaptive_method}'. Expected 'mean' or 'gaussian'.")

    thresh_type = _polarity_to_thresh_type(object_polarity)

    start = time.perf_counter()
    mask = cv2.adaptiveThreshold(
        gray_image, 255, cv2_adaptive_method, thresh_type, block_size, C
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return _make_result(
        mask=mask,
        method="adaptive",
        parameters={
            "block_size": block_size,
            "C": C,
            "adaptive_method": method_key,
            "object_polarity": object_polarity,
        },
        threshold=None,
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# 4. EDGE-BASED SEGMENTATION
# ---------------------------------------------------------------------------

def _fill_enclosed_regions(closed_edge_map: np.ndarray) -> np.ndarray:
    """
    Convert a closed edge map (boundaries only) into a filled binary
    mask (interiors as foreground), using the standard flood-fill hole-
    filling technique.

    NOTE ON SCOPE: this function is intentionally kept local to
    edge-based segmentation only. It is NOT the general morphology
    module (morphology.py, a separate shared stage) -- it exists here
    solely because edge-based segmentation cannot produce a comparable
    foreground/background mask without this specific fill step; a raw
    edge map is a set of boundary pixels, not a region mask, and would
    not be comparable to the other three methods' output otherwise.

    How it works:
        1. Flood-fill the edge map starting from the image corner
           (0, 0), which is assumed to be background. This marks every
           background pixel *reachable from the border* as white.
        2. Invert the result: pixels that were NOT reachable from the
           border (i.e., enclosed by a boundary -- the interior of an
           object) become white; the reachable background becomes black.
        3. OR this inverted image with the original edge map, so the
           boundary pixels themselves are also included as foreground.

    Args:
        closed_edge_map: (H, W) uint8 binary edge map after gap-closing,
            values in {0, 255}.

    Returns:
        (H, W) uint8 filled mask, values in {0, 255}, white = foreground.
    """
    h, w = closed_edge_map.shape
    flood_filled = closed_edge_map.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    filled = cv2.bitwise_or(closed_edge_map, flood_filled_inv)
    return filled


def segment_edge_based(gray_image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150,
                        closing_kernel_size: int = 5) -> dict:
    """
    Genuine edge-based segmentation: detects object boundaries via
    gradients, closes small gaps in those boundaries, then fills the
    enclosed regions to produce a foreground mask comparable to the
    three thresholding-based methods above.

    Pipeline (each step kept visible, not hidden):
        Grayscale -> Canny edge detection -> edge map
                  -> morphological closing (THIS METHOD ONLY -- gap closing)
                  -> flood-fill region filling
                  -> binary object mask

    Gradient/edge principle:
        Canny edge detection locates pixels where intensity changes
        sharply (high gradient magnitude) -- these sharp transitions
        typically occur at object boundaries, where an object's
        intensity differs from its surrounding background. Canny
        additionally applies non-maximum suppression (thinning edges to
        single-pixel width) and hysteresis thresholding using the two
        supplied thresholds: gradient points above `high_threshold` are
        immediately accepted as edges, points below `low_threshold` are
        rejected, and points in between are accepted only if connected
        to an already-accepted edge pixel.

    Why edges can represent object boundaries:
        An object's outline is, by definition, the location where the
        object's intensity meets the background's intensity -- exactly
        where a strong gradient occurs. This makes edge detection a
        viable way to locate object boundaries even in cases where the
        object and background don't have a single consistent brightness
        relationship across the whole image (which is where intensity
        thresholding methods can struggle).

    Why edge gaps are a problem:
        Real object boundaries are rarely perfectly captured by Canny --
        low-contrast sections of a boundary, noise, or a locally weak
        gradient can leave small gaps in the traced outline. An outline
        with gaps is NOT a closed curve, so there is no well-defined
        "interior" to treat as foreground -- flood-filling from the
        border would leak straight through a gap and fill the entire
        image as "background."

    Why region filling is required:
        Canny's raw output is a set of boundary pixels, not a region --
        it says nothing about which side of a boundary is "inside" the
        object. Filling (after closing the gaps) converts this boundary
        information into the same kind of foreground/background region
        mask the other three methods natively produce, which is
        necessary for a fair, comparable output.

    Sensitivity to noise:
        Gradient-based detection is inherently sensitive to noise,
        since noise itself often manifests as small, spurious local
        intensity changes that Canny can mistake for edges, producing
        false boundary fragments. This is why the noise-reduction step
        from Phase 2 (particularly for salt-and-pepper noise) matters
        more for this method than it might first appear.

    Strengths vs. intensity thresholding:
        - Can work even when foreground/background intensities are not
          cleanly separable everywhere, as long as a local gradient
          exists at the true boundary.
        - Not dependent on a bimodal histogram assumption (unlike Otsu).

    Weaknesses vs. intensity thresholding:
        - Requires an extra, non-trivial gap-closing + filling step
          just to produce a comparable region mask.
        - More parameters to tune (two Canny thresholds + closing
          kernel size), and more failure modes (leaking through
          unclosed gaps, over-closing merging separate nearby objects).
        - Generally the most noise-sensitive of the four methods.

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image.
        low_threshold: Canny's lower hysteresis threshold.
        high_threshold: Canny's upper hysteresis threshold.
        closing_kernel_size: size of the structuring element used ONLY
            to close small gaps in the edge map before filling (not the
            general morphology module).

    Returns:
        Common result dict. "threshold" is None (not applicable --
        Canny uses two thresholds, already recorded in "parameters").
        "intermediate" contains:
            "edges": raw Canny output
            "closed_edges": edge map after gap-closing
    """
    low_threshold = int(max(0, min(255, low_threshold)))
    high_threshold = int(max(0, min(255, high_threshold)))
    closing_kernel_size = _validate_odd(closing_kernel_size, min_size=1, max_size=31)

    start = time.perf_counter()

    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # Gap-closing: specific to converting an edge map into fillable
    # closed contours -- scoped to this method only, per the design note
    # in this function's docstring.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    filled_mask = _fill_enclosed_regions(closed_edges)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return _make_result(
        mask=filled_mask,
        method="edge",
        parameters={
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "closing_kernel_size": closing_kernel_size,
        },
        threshold=None,
        processing_time_ms=elapsed_ms,
        intermediate={"edges": edges, "closed_edges": closed_edges},
    )


# ---------------------------------------------------------------------------
# 5. COMMON DISPATCHER
# ---------------------------------------------------------------------------

SEGMENTATION_METHODS: dict[str, Callable[..., dict]] = {
    "global": segment_global,
    "otsu": segment_otsu,
    "adaptive": segment_adaptive,
    "edge": segment_edge_based,
}


def segment(gray_image: np.ndarray, method: str, **parameters) -> dict:
    """
    Common dispatcher: run any of the four segmentation methods through
    a single call site, so callers (the Streamlit app, and later the
    comparison engine) don't need method-specific branching logic.

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image -- the
            SAME preprocessed image must be supplied regardless of
            method, so only the segmentation strategy varies.
        method: one of "global", "otsu", "adaptive", "edge".
        **parameters: forwarded to the corresponding segment_*() function.

    Returns:
        Common result dict (see module docstring).
    """
    method_key = method.lower().strip()
    if method_key not in SEGMENTATION_METHODS:
        raise ValueError(
            f"Unknown segmentation method: '{method}'. "
            f"Expected one of {list(SEGMENTATION_METHODS.keys())}."
        )
    return SEGMENTATION_METHODS[method_key](gray_image, **parameters)

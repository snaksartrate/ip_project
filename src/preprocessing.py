"""
preprocessing.py

Implements everything that happens BEFORE segmentation:
    - loading / validating the uploaded image
    - resizing (aspect-ratio preserving)
    - grayscale conversion
    - noise reduction (Gaussian / median filtering)
    - histogram computation (+ optional histogram equalization)

Design note: every function here is intentionally implemented using the
underlying OpenCV primitive operations directly (cv2.GaussianBlur,
cv2.medianBlur, cv2.calcHist, cv2.cvtColor) rather than any wrapped/
black-box helper, and every function's docstring explains the
"what/why/how" so this stage is fully explainable in a viva. These
operations are shared identically by all four segmentation methods --
this is what keeps the four-way comparison fair, since only the
segmentation stage differs.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# 1. IMAGE INPUT / VALIDATION
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP"}


def load_and_validate_image(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load an image from a Streamlit `UploadedFile` object and validate it.

    What it does:
        Reads the uploaded bytes with PIL (which cleanly rejects corrupt/
        non-image files with a clear exception), converts to RGB to
        normalize color mode (handles palette/CMYK/grayscale-mode PNGs
        safely), then converts to a NumPy array in BGR order -- BGR is
        used because every downstream call is an OpenCV function, and
        OpenCV's native channel order is BGR, not RGB.

    Why validation matters here:
        Streamlit's file_uploader only filters by file *extension*, not
        by actually verifying the file is a decodable image. A renamed
        .txt file with a .jpg extension, or a truncated/corrupted upload,
        would otherwise crash deep inside the pipeline with a confusing
        stack trace. Catching it here, at the boundary, lets the UI show
        a clean error message instead.

    Args:
        uploaded_file: a Streamlit UploadedFile (file-like object) or
            None.

    Returns:
        (image_bgr, error_message)
        Exactly one of the two will be non-None:
            - On success: (np.ndarray of shape (H, W, 3), None)
            - On failure: (None, "human readable reason")
    """
    if uploaded_file is None:
        return None, "No file was uploaded."

    try:
        pil_image = Image.open(uploaded_file)
        pil_image.verify()  # cheap structural check; raises if corrupt
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, f"The uploaded file could not be read as an image ({exc})."

    try:
        # Re-open after verify(): PIL's verify() leaves the file object
        # in a state where it can no longer be used to load pixel data.
        uploaded_file.seek(0)
        pil_image = Image.open(uploaded_file).convert("RGB")
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001 - deliberately broad at this I/O boundary
        return None, f"The image was readable but could not be decoded into pixel data ({exc})."

    if bgr_array is None or bgr_array.size == 0:
        return None, "The decoded image is empty."

    return bgr_array, None


def get_image_info(image: np.ndarray) -> dict:
    """
    Collect basic descriptive information about a loaded image.

    What it does:
        Returns height, width, channel count, dtype, and approximate
        in-memory size -- shown to the user right after upload so they
        can see exactly what was decoded before any processing happens.

    Args:
        image: np.ndarray, either (H, W) grayscale or (H, W, C) color.

    Returns:
        dict with keys: height, width, channels, dtype, size_kb
    """
    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    size_kb = image.nbytes / 1024

    return {
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "dtype": str(image.dtype),
        "size_kb": round(size_kb, 1),
    }


# ---------------------------------------------------------------------------
# 2. RESIZING
# ---------------------------------------------------------------------------

def resize_image(image: np.ndarray, max_dimension: int = 800) -> Tuple[np.ndarray, bool]:
    """
    Resize an image so its larger side does not exceed `max_dimension`,
    preserving aspect ratio. Images already within the limit are
    returned unchanged.

    Why resizing is useful:
        - Very large uploads (e.g., a 4000x3000 phone photo) slow down
          every later stage (segmentation, morphology, contour finding)
          without adding useful information for a coin-counting task.
        - Later, when we time-compare the four segmentation methods,
          processing time is only a meaningful comparison if input size
          is controlled/consistent -- an accidentally huge image would
          skew the timing metric for reasons unrelated to the algorithm.

    What it affects (important to state honestly):
        - Downsampling *does* lose fine detail and can slightly shift
          measured object boundaries by a pixel or two, which matters
          for the area/perimeter measurements.
          This is a real trade-off, not a free operation -- we resize
          only when the image exceeds `max_dimension`, not unconditionally,
          precisely to minimize this effect on typical-sized inputs.

    How it works:
        A single scale factor = max_dimension / max(height, width) is
        computed and applied to both dimensions, which is what
        "preserving aspect ratio" means mathematically (both axes are
        scaled by the identical factor, so width/height ratio is
        unchanged). cv2.INTER_AREA is used for downsampling, which is
        the interpolation method OpenCV recommends for shrinking images
        as it reduces aliasing better than nearest/bilinear for this
        direction.

    Args:
        image: np.ndarray, (H, W) or (H, W, C).
        max_dimension: the largest allowed side length, in pixels.

    Returns:
        (resized_image, was_resized)
    """
    if image.ndim == 2:
        height, width = image.shape
    else:
        height, width = image.shape[:2]

    largest_side = max(height, width)

    if largest_side <= max_dimension:
        return image, False

    scale_factor = max_dimension / float(largest_side)
    new_width = max(1, int(round(width * scale_factor)))
    new_height = max(1, int(round(height * scale_factor)))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True


# ---------------------------------------------------------------------------
# 3. GRAYSCALE CONVERSION
# ---------------------------------------------------------------------------

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a color (BGR) image to single-channel grayscale.

    Why grayscale is appropriate for the segmentation methods used
    later in this project:
        Global, Otsu, and Adaptive thresholding are all defined on a
        single scalar *intensity* value per pixel -- they classify each
        pixel as foreground/background by comparing one number against
        a threshold. A 3-channel color image doesn't have a single
        "intensity" value per pixel by default, so these methods
        require a grayscale (or otherwise single-channel) input.
        Edge-based segmentation (Sobel/Canny) is likewise normally
        computed on a single intensity channel, since gradients are
        best defined on a scalar field.

    How it works (the math):
        OpenCV's cv2.COLOR_BGR2GRAY uses the standard luma-weighted sum:
            Gray = 0.299*R + 0.587*G + 0.114*B
        The weights are not arbitrary -- they approximate human
        perceptual sensitivity to each color channel (the eye is most
        sensitive to green, least to blue), so the resulting grayscale
        image preserves perceived brightness better than a plain
        average of the three channels would.

    Args:
        image: np.ndarray, (H, W, 3) in BGR order, or already (H, W).

    Returns:
        np.ndarray, (H, W), uint8 grayscale image.
    """
    if image.ndim == 2:
        # Already single-channel (e.g., a grayscale PNG that our RGB
        # normalization in load_and_validate_image expanded back to 3
        # identical channels would not hit this branch -- kept here as
        # a safety net for any caller that passes a raw 2D array).
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 4. NOISE REDUCTION
# ---------------------------------------------------------------------------

def _validate_kernel_size(kernel_size: int, min_size: int = 1, max_size: int = 31) -> int:
    """
    Force a user-supplied kernel size into a valid, odd, in-range value.

    Why this exists:
        OpenCV's blur functions require an odd kernel size (so there is
        a well-defined center pixel) and raise an exception otherwise.
        A UI slider or text input could easily produce an even number
        or an out-of-range value; rather than letting that crash the
        app, we clamp and round here, once, in a single shared helper.

    Args:
        kernel_size: the requested kernel size.
        min_size: smallest allowed odd kernel size.
        max_size: largest allowed odd kernel size.

    Returns:
        A valid odd integer kernel size within [min_size, max_size].
    """
    kernel_size = int(kernel_size)
    kernel_size = max(min_size, min(max_size, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
        kernel_size = min(kernel_size, max_size if max_size % 2 == 1 else max_size - 1)
    return kernel_size


def apply_gaussian_filter(gray_image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian smoothing to a grayscale image.

    What it does / how it works (the math):
        Replaces each pixel with a weighted average of its neighbors,
        where the weights follow a 2D Gaussian (bell curve) centered on
        that pixel:
            G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
        Nearby pixels get high weight, distant pixels within the kernel
        get low weight, so the result is a smooth blur that suppresses
        random per-pixel intensity fluctuations (noise) while mostly
        preserving overall structure/edges.

    Advantages:
        - Smooth, natural-looking blur; good general-purpose noise
          reduction for Gaussian-type sensor noise.
        - Mathematically well-behaved (separable, rotationally symmetric).

    Disadvantages:
        - Blurs edges too -- it does not distinguish "noise pixel" from
          "real edge pixel", so it softens object boundaries, which can
          slightly bias later perimeter/area measurements.
        - Not effective against salt-and-pepper (impulse) noise: a
          single extreme outlier pixel still contributes to the
          weighted average and just gets smeared into its neighbors
          rather than removed.

    Args:
        gray_image: (H, W) uint8 grayscale image.
        kernel_size: side length of the (square) Gaussian kernel; must
            be odd -- invalid values are corrected automatically.
        sigma: standard deviation of the Gaussian; 0 tells OpenCV to
            derive it automatically from the kernel size.

    Returns:
        (H, W) uint8 filtered image.
    """
    kernel_size = _validate_kernel_size(kernel_size)
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigmaX=sigma)


def apply_median_filter(gray_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filtering to a grayscale image.

    What it does / how it works:
        Replaces each pixel with the *median* (not the average) of the
        intensity values in its square neighborhood. Because the median
        is robust to outliers -- an extreme value in a window barely
        shifts the median, whereas it would strongly pull a mean --
        this filter is specifically effective against salt-and-pepper
        (impulse) noise, where isolated pixels are randomly forced to
        pure black or pure white.

    Why this matters for this project specifically:
        The dataset design (Phase 5) includes a "noisy images" category,
        and salt-and-pepper noise is one of the standard noise models
        used to construct it. Gaussian blur would smear such outlier
        pixels into their neighborhood rather than eliminating them;
        median filtering can remove them almost completely while
        preserving edges noticeably better than Gaussian blur does for
        this specific noise type.

    Advantages:
        - Excellent at removing salt-and-pepper / impulse noise.
        - Better edge preservation than Gaussian blur for this noise type.

    Disadvantages:
        - Computationally more expensive than Gaussian blur (requires
          sorting/selecting within each window rather than a weighted sum).
        - Can erode fine, thin details (e.g., a genuinely thin object
          feature can be treated as an "outlier" and removed) if the
          kernel is too large relative to the detail size.
        - Less effective than Gaussian blur against smooth, Gaussian-
          distributed sensor noise.

    Args:
        gray_image: (H, W) uint8 grayscale image.
        kernel_size: side length of the square neighborhood; must be
            odd and greater than 1 -- invalid values are corrected
            automatically.

    Returns:
        (H, W) uint8 filtered image.
    """
    kernel_size = _validate_kernel_size(kernel_size, min_size=3)
    return cv2.medianBlur(gray_image, kernel_size)


def apply_noise_reduction(gray_image: np.ndarray, method: str = "none", kernel_size: int = 5) -> np.ndarray:
    """
    Dispatch to the selected noise-reduction method (or none).

    This is the single function the UI calls, so the "None / Gaussian /
    Median" choice in Streamlit maps directly onto one call site rather
    than being spread across app.py as if/else branches.

    Args:
        gray_image: (H, W) uint8 grayscale image.
        method: one of "none", "gaussian", "median" (case-insensitive).
        kernel_size: passed through to the chosen filter.

    Returns:
        (H, W) uint8 image -- unchanged if method == "none".
    """
    method = method.lower().strip()
    if method == "gaussian":
        return apply_gaussian_filter(gray_image, kernel_size=kernel_size)
    elif method == "median":
        return apply_median_filter(gray_image, kernel_size=kernel_size)
    elif method == "none":
        return gray_image
    else:
        raise ValueError(f"Unknown noise reduction method: '{method}'. Expected 'none', 'gaussian', or 'median'.")


# ---------------------------------------------------------------------------
# 5. HISTOGRAM
# ---------------------------------------------------------------------------

def compute_histogram(gray_image: np.ndarray) -> np.ndarray:
    """
    Compute the intensity histogram of a grayscale image.

    What it does / how it works:
        Counts how many pixels fall into each of the 256 possible
        intensity values (0 = black .. 255 = white), i.e. it answers
        "how many pixels have intensity 0?", "...intensity 1?", etc.
        cv2.calcHist is used directly (OpenCV's histogram primitive)
        rather than a higher-level plotting shortcut, so the actual
        counting step is a single, inspectable line.

    Why this matters for the segmentation methods used later:
        Otsu's method works by analyzing this exact histogram and
        searching for the threshold that best separates it into two
        populations (assuming it is roughly bimodal -- e.g., a peak for
        "dark background" and a peak for "bright objects", or vice
        versa). Displaying the histogram now lays the groundwork for
        visually explaining, *why* Otsu succeeds on
        some images and fails on others (e.g., a flat or unimodal
        histogram has no clear valley for Otsu to find).

    Args:
        gray_image: (H, W) uint8 grayscale image.

    Returns:
        np.ndarray of shape (256,) -- pixel count per intensity level.
    """
    hist = cv2.calcHist([gray_image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    return hist.flatten()


def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
    """
    OPTIONAL contrast-enhancement experiment -- NOT part of the default
    preprocessing path. Must be explicitly requested via the UI.

    What it does / how it works:
        Histogram equalization redistributes intensity values so the
        output histogram is approximately uniform (flat), which spreads
        out the most frequent intensity values across the available
        range. This is achieved by mapping each intensity to a new
        value using the image's cumulative distribution function (CDF)
        of pixel intensities -- intensities in densely populated regions
        of the histogram get spread further apart than intensities in
        sparse regions.

    Why it's optional here:
        It is primarily useful for the uneven-illumination dataset
        category as a *potential* preprocessing aid --
        but it can also over-amplify noise and shift apparent object
        boundaries, so per the approved design it is offered as an
        explicit, opt-in experiment rather than baked into the standard
        pipeline that segmentation will consume by default.

    Args:
        gray_image: (H, W) uint8 grayscale image.

    Returns:
        (H, W) uint8 equalized image.
    """
    return cv2.equalizeHist(gray_image)


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE CONVENIENCE WRAPPER
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    max_dimension: int = 800,
    noise_reduction_method: str = "none",
    kernel_size: int = 5,
) -> dict:
    """
    Run the full Phase 2 preprocessing sequence and return every
    intermediate result, so the UI can display each step without
    recomputing anything.

    Pipeline: resize -> grayscale -> noise reduction -> histogram.

    Args:
        image: the original loaded (H, W, 3) BGR image.
        max_dimension: passed to resize_image().
        noise_reduction_method: "none" | "gaussian" | "median".
        kernel_size: passed to the chosen filter.

    Returns:
        dict with keys:
            "original": the untouched input image
            "resized": image after resize_image()
            "was_resized": bool
            "gray": grayscale version of the resized image
            "filtered": grayscale image after noise reduction
                        (identical to "gray" if method == "none")
            "histogram": 256-length intensity histogram of "filtered"
    """
    resized, was_resized = resize_image(image, max_dimension=max_dimension)
    gray = convert_to_grayscale(resized)
    filtered = apply_noise_reduction(gray, method=noise_reduction_method, kernel_size=kernel_size)
    histogram = compute_histogram(filtered)

    return {
        "original": image,
        "resized": resized,
        "was_resized": was_resized,
        "gray": gray,
        "filtered": filtered,
        "histogram": histogram,
    }

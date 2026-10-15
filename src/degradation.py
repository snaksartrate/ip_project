"""
degradation.py

Injects synthetic environmental flaws into a clean preprocessed image so
the robustness of each segmentation method can be stress-tested.

Why this module exists:
    Accuracy on a clean image says little about how a method behaves in
    the field. By degrading the SAME clean image at increasing severity
    and re-running every segmentation method, we can plot IoU against
    degradation level and see where each method breaks down.

Design notes:
    - Every function is stateless: it takes an image and returns a NEW
      array. The input is never modified.
    - Every function returns uint8 with the same shape as the input, so
      the result can be fed straight into any segment_*() function.
    - Works on grayscale (H, W) and color (H, W, C) images.
    - Severity of 0 / identity values (variance 0, gamma 1.0, kernel
      size 1) returns an unchanged copy, so a degradation sweep can
      start at its "no degradation" point.
"""

from typing import Optional
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. GAUSSIAN NOISE
# ---------------------------------------------------------------------------

def inject_gaussian_noise(image: np.ndarray, variance_level: float,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to an image.

    What it does / how it works (the math):
        For every pixel, a random value is drawn from a normal
        distribution N(0, variance_level) and added to the pixel's
        intensity:
            Output(x,y) = clip(Input(x,y) + n(x,y), 0, 255)
            n(x,y) ~ N(mean=0, variance=variance_level)
        NumPy's normal() takes a standard deviation, so
        sigma = sqrt(variance_level) is used.

    Scale reference (intensities are on the 0-255 scale):
        variance   25  -> sigma  5   (barely visible)
        variance  100  -> sigma 10   (mild)
        variance  400  -> sigma 20   (clearly visible)
        variance 1600  -> sigma 40   (heavy)

    Why this is a useful stress test:
        Sensor noise perturbs every pixel independently. Thresholding
        methods then misclassify pixels near the threshold, and Canny
        sees spurious gradients, so edge-based segmentation is expected
        to degrade fastest.

    Why the image is converted to float first:
        Adding negative noise to a uint8 array would wrap around
        (e.g. 5 + (-10) -> 251) instead of going below zero. Computing
        in float32 and clipping afterwards avoids this.

    Args:
        image: np.ndarray, uint8, (H, W) or (H, W, C).
        variance_level: variance of the noise (must be >= 0).
        seed: optional RNG seed. Passing the same seed gives identical
            noise, which keeps benchmark runs reproducible.

    Returns:
        np.ndarray, uint8, same shape as `image`.
    """
    if variance_level < 0:
        raise ValueError(f"variance_level must be >= 0, got {variance_level}.")

    if variance_level == 0:
        return image.copy()

    rng = np.random.default_rng(seed)
    sigma = float(np.sqrt(variance_level))
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)

    noisy = image.astype(np.float32) + noise.astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. LOW LIGHT (GAMMA CORRECTION)
# ---------------------------------------------------------------------------

def inject_low_light(image: np.ndarray, gamma_value: float) -> np.ndarray:
    """
    Simulate low-light conditions using gamma correction.

    What it does / how it works (the math):
        Output = 255 * (Input / 255) ^ gamma
        Input is normalized to [0, 1], raised to the power gamma, then
        scaled back to [0, 255].
            gamma = 1.0 -> unchanged
            gamma > 1.0 -> darker (mid-tones are pulled down)
            gamma < 1.0 -> brighter

    Scale reference:
        gamma 1.5 -> slightly dim
        gamma 2.0 -> clearly dark
        gamma 3.0 -> very dark

    Why this is a useful stress test:
        Darkening compresses the intensity range into the low end of
        the histogram, shrinking the gap between object and background.
        A fixed global threshold tuned on the bright image will no
        longer fit, and Otsu's histogram valley becomes harder to find.
        Adaptive thresholding compares each pixel to its own
        neighborhood, so it is expected to cope better.

    How it is computed:
        A 256-entry lookup table (LUT) is built once and applied with
        cv2.LUT. This is exact, since uint8 only has 256 possible
        inputs, and much faster than applying the power function to
        every pixel.

    Args:
        image: np.ndarray, uint8, (H, W) or (H, W, C).
        gamma_value: must be > 0. Values > 1.0 darken the image.

    Returns:
        np.ndarray, uint8, same shape as `image`.
    """
    if gamma_value <= 0:
        raise ValueError(f"gamma_value must be > 0, got {gamma_value}.")

    lut = np.array(
        [255.0 * ((i / 255.0) ** gamma_value) for i in range(256)]
    )
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)

    return cv2.LUT(image, lut)


# ---------------------------------------------------------------------------
# 3. MOTION BLUR
# ---------------------------------------------------------------------------

def _build_motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """
    Build a normalized line-shaped (directional) blur kernel.

    How it works:
        1. Start with a (k x k) zero matrix and draw a horizontal line
           of ones through the middle row.
        2. Rotate that matrix around its center by `angle` degrees, so
           the line points in the desired direction.
        3. Divide by the sum so all weights add up to 1. This keeps the
           overall brightness unchanged after filtering.

    Args:
        kernel_size: odd integer >= 1 (length of the blur line, px).
        angle: direction of motion in degrees (0 = horizontal,
            90 = vertical, counter-clockwise).

    Returns:
        np.ndarray, float32, (k, k), sums to 1.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    kernel[center, :] = 1.0

    rotation = cv2.getRotationMatrix2D(
        (kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))

    total = kernel.sum()
    if total <= 0:
        # Defensive fallback: identity kernel (no blur) if rotation
        # somehow produced an empty kernel.
        identity = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        identity[center, center] = 1.0
        return identity

    return kernel / total


def inject_motion_blur(image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    """
    Simulate camera or object motion with a directional blur.

    What it does / how it works:
        Builds a line-shaped kernel of length `kernel_size` pointing
        along `angle`, then convolves the image with it using
        cv2.filter2D. Each output pixel becomes the average of the
        pixels along that line, which smears detail in the direction of
        motion only.

    Why this is a useful stress test:
        Motion blur widens and softens object boundaries along one
        direction. This weakens the gradient that Canny relies on, so
        edge-based segmentation should suffer, while thresholding
        methods mostly shift the boundary position.

    Difference from Gaussian blur (preprocessing.py):
        Gaussian blur smooths equally in all directions. Motion blur
        smooths along one direction only, which is why it is treated
        here as an environmental flaw rather than a denoising step.

    Args:
        image: np.ndarray, uint8, (H, W) or (H, W, C).
        kernel_size: length of the blur line in pixels. Even values are
            rounded up to the next odd number. 1 means no blur.
        angle: motion direction in degrees (0 = horizontal).

    Returns:
        np.ndarray, uint8, same shape as `image`.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}.")
    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_size == 1:
        return image.copy()

    kernel = _build_motion_blur_kernel(kernel_size, angle)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
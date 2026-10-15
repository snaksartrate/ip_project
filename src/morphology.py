"""
morphology.py

Shared morphological post-processing pipeline, applied identically
regardless of which segmentation method produced the input mask. Keeping
this stage common (rather than method-specific) is part of the fair
comparison principle: only segmentation should vary between methods.

MASK CONVENTION (same as segmentation.py):
    255 (white) = foreground object
    0   (black) = background
Every function accepts any mask where non-zero means foreground, and
always returns a strict {0, 255} uint8 mask.

Functions:

    erode(mask, kernel_size=3, iterations=1) -> np.ndarray
    dilate(mask, kernel_size=3, iterations=1) -> np.ndarray
    opening(mask, kernel_size=3) -> np.ndarray
        Erosion followed by dilation -- removes small noise specks
        without significantly changing object size.
    closing(mask, kernel_size=3) -> np.ndarray
        Dilation followed by erosion -- fills small holes and gaps.
    remove_small_regions(mask, min_area=50) -> np.ndarray
        Removes connected components below `min_area`, treated as noise
        rather than real objects.
    clean_mask(mask, **params) -> dict
        Convenience wrapper running the standard sequence
        (opening -> closing -> remove_small_regions) and returning
        intermediate results for display.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize any mask to a strict {0, 255} uint8 (H, W) array.

    Why: the morphology functions must behave the same whether the mask
    came from a segmentation method ({0, 255}) or a ground-truth file
    ({0, 1}, or a 3-channel image). Any non-zero pixel is foreground.
    """
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_kernel(kernel_size: int, max_size: int = 51) -> int:
    """
    Force a kernel size into a valid odd integer in [1, max_size].

    Why: an even-sized structuring element has no center pixel, which
    shifts the mask by half a pixel after every operation. A UI slider
    could still produce an even or out-of-range value, so it is
    corrected here instead of crashing or silently shifting the result.
    """
    kernel_size = int(kernel_size)
    kernel_size = max(1, min(max_size, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
        if kernel_size > max_size:
            kernel_size = max_size if max_size % 2 == 1 else max_size - 1
    return kernel_size


def _structuring_element(kernel_size: int) -> np.ndarray:
    """
    Build an elliptical (disk-like) structuring element.

    Why elliptical rather than square: a square kernel treats diagonal
    directions as longer than horizontal/vertical ones, so it distorts
    round objects (such as coins) into rounded squares after repeated
    operations. An ellipse treats all directions equally.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


# ---------------------------------------------------------------------------
# 1. EROSION
# ---------------------------------------------------------------------------

def erode(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Shrink foreground regions.

    What it does / how it works:
        The structuring element slides over the mask. A pixel stays
        foreground only if EVERY pixel under the element is foreground
        (a local minimum). Object boundaries therefore move inward by
        about kernel_size // 2 pixels per iteration.

    Effects:
        - Removes isolated specks smaller than the element.
        - Separates objects joined by thin bridges.
        - Shrinks every object, so it is rarely used alone (see opening).

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        kernel_size: odd integer >= 1 (invalid values are corrected).
        iterations: number of times to apply the operation (>= 1).

    Returns:
        (H, W) uint8 mask, values in {0, 255}.
    """
    mask = _to_mask(mask)
    kernel = _structuring_element(_validate_kernel(kernel_size))
    return cv2.erode(mask, kernel, iterations=max(1, int(iterations)))


# ---------------------------------------------------------------------------
# 2. DILATION
# ---------------------------------------------------------------------------

def dilate(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Grow foreground regions.

    What it does / how it works:
        A pixel becomes foreground if ANY pixel under the structuring
        element is foreground (a local maximum). Object boundaries move
        outward by about kernel_size // 2 pixels per iteration.

    Effects:
        - Fills small holes and narrow gaps inside objects.
        - Can merge objects that are close together.
        - Enlarges every object, so it is rarely used alone (see closing).

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        kernel_size: odd integer >= 1 (invalid values are corrected).
        iterations: number of times to apply the operation (>= 1).

    Returns:
        (H, W) uint8 mask, values in {0, 255}.
    """
    mask = _to_mask(mask)
    kernel = _structuring_element(_validate_kernel(kernel_size))
    return cv2.dilate(mask, kernel, iterations=max(1, int(iterations)))


# ---------------------------------------------------------------------------
# 3. OPENING
# ---------------------------------------------------------------------------

def opening(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Erosion followed by dilation: removes small foreground noise.

    Why the two steps are combined:
        Erosion deletes specks smaller than the structuring element but
        also shrinks real objects. The dilation that follows grows the
        surviving objects back to roughly their original size. Specks
        removed by the erosion are gone and cannot grow back.

    Limitation:
        Fine details thinner than the kernel (a thin real object, a
        narrow protrusion) are removed too. Keep the kernel small
        relative to the objects.

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        kernel_size: odd integer >= 1 (invalid values are corrected).

    Returns:
        (H, W) uint8 mask, values in {0, 255}.
    """
    mask = _to_mask(mask)
    kernel = _structuring_element(_validate_kernel(kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ---------------------------------------------------------------------------
# 4. CLOSING
# ---------------------------------------------------------------------------

def closing(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Dilation followed by erosion: fills small holes and gaps.

    Why the two steps are combined:
        Dilation closes pinholes and narrow cracks inside objects but
        also grows them. The erosion that follows shrinks the outer
        boundary back, while the filled holes stay filled.

    Limitation:
        Objects separated by a gap narrower than the kernel are merged
        into one. Keep the kernel smaller than the gaps between objects.

    Note:
        This is the general, shared closing step. It is separate from
        the gap-closing inside segment_edge_based(), which only bridges
        breaks in a Canny edge map before flood-filling.

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        kernel_size: odd integer >= 1 (invalid values are corrected).

    Returns:
        (H, W) uint8 mask, values in {0, 255}.
    """
    mask = _to_mask(mask)
    kernel = _structuring_element(_validate_kernel(kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


# ---------------------------------------------------------------------------
# 5. REMOVE SMALL REGIONS
# ---------------------------------------------------------------------------

def remove_small_regions(mask: np.ndarray, min_area: int = 50,
                         connectivity: int = 8) -> np.ndarray:
    """
    Delete connected foreground components smaller than `min_area` pixels.

    What it does / how it works:
        cv2.connectedComponentsWithStats labels every connected
        foreground blob and reports its pixel area. Every label whose
        area is below `min_area` is dropped, and the rest are kept.
        The background (label 0) is never kept.

    Why it complements opening:
        Opening removes specks by shape (smaller than the structuring
        element). This removes them by size, so it also catches larger
        blobs of leftover noise that opening leaves behind, without
        touching object shape at all.

    Choosing min_area:
        It is in pixels, so it depends on the image size. Set it below
        the area of the smallest real object and above the typical noise
        speck. A min_area of 0 or 1 keeps everything.

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        min_area: minimum area in pixels for a component to survive.
        connectivity: 4 or 8 (which neighbors count as connected).

    Returns:
        (H, W) uint8 mask, values in {0, 255}.
    """
    mask = _to_mask(mask)
    min_area = max(0, int(min_area))
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    if n_labels <= 1:
        return mask  # background only, nothing to remove

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False                      # label 0 is the background
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 6. STANDARD CLEANING PIPELINE
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, open_kernel_size: int = 3,
               close_kernel_size: int = 3, min_area: int = 50,
               connectivity: int = 8) -> dict:
    """
    Run the standard shared cleanup:
        opening -> closing -> remove_small_regions

    Why this order:
        1. Opening first removes speckle, so closing does not seal it
           into the objects.
        2. Closing then fills the holes and cracks left inside the
           real objects.
        3. Small-region removal finally deletes any leftover blobs
           that are too small to be a real object.

    The same function and the same parameters must be used for all four
    segmentation methods, so that only the segmentation stage differs.

    Args:
        mask: (H, W) mask from any segmentation method.
        open_kernel_size: kernel size for the opening step.
        close_kernel_size: kernel size for the closing step.
        min_area: minimum component area (px) kept in the final mask.
        connectivity: 4 or 8, used by remove_small_regions.

    Returns:
        dict:
            {
                "input": np.ndarray,      # normalized input mask
                "opened": np.ndarray,     # after opening
                "closed": np.ndarray,     # after opening + closing
                "final": np.ndarray,      # after small-region removal
                "parameters": dict,       # values actually used
            }
        All masks are (H, W) uint8 with values in {0, 255}.
    """
    normalized = _to_mask(mask)
    open_k = _validate_kernel(open_kernel_size)
    close_k = _validate_kernel(close_kernel_size)

    opened = opening(normalized, open_k)
    closed = closing(opened, close_k)
    final = remove_small_regions(closed, min_area=min_area, connectivity=connectivity)

    return {
        "input": normalized,
        "opened": opened,
        "closed": closed,
        "final": final,
        "parameters": {
            "open_kernel_size": open_k,
            "close_kernel_size": close_k,
            "min_area": int(min_area),
            "connectivity": connectivity,
        },
    }
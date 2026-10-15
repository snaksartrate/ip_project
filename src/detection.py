"""
detection.py

Detects individual objects from a cleaned binary mask, using connected
components (primary, gives area/centroid natively) and contours
(for perimeter/shape). Filters out components that are too small to be
real objects (as opposed to leftover noise).

MASK CONVENTION (same as segmentation.py / morphology.py):
    255 (white) = foreground, 0 (black) = background.
    Any non-zero pixel is treated as foreground.

Functions:

    filter_small_components(stats, min_area) -> list[int]
        Labels (from cv2.connectedComponentsWithStats) whose area is
        at least `min_area`. The background label 0 is never returned.

    detect_objects(mask, min_area=50) -> list[dict]
        One dict per detected object:
        {
            "id": int,               # 1..N, sequential after filtering
            "label": int,            # label in the connected-components map
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),    # float pixel coordinates
            "area": int,             # pixel count
            "contour": np.ndarray,   # outer contour, OpenCV (K, 1, 2) int32
        }

    get_contours(mask, min_area=50) -> list[np.ndarray]
        Outer contours of the objects that survive the size filter.
"""

from typing import List
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize any mask to a strict {0, 255} uint8 (H, W) array."""
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _validate_connectivity(connectivity: int) -> int:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")
    return connectivity


# ---------------------------------------------------------------------------
# 1. SIZE FILTER
# ---------------------------------------------------------------------------

def filter_small_components(stats: np.ndarray, min_area: int) -> List[int]:
    """
    Return the labels of components that are large enough to be objects.

    Why this exists:
        Even after morphological cleanup, small blobs can survive. A
        component with fewer than `min_area` pixels is treated as noise,
        not as an object, so it must not be counted or measured.

    Args:
        stats: the (N, 5) array returned by
            cv2.connectedComponentsWithStats. Row i describes label i;
            row 0 is the background.
        min_area: minimum pixel area for a component to be kept.

    Returns:
        list[int] of labels to keep, in ascending order (never
        includes 0).
    """
    min_area = max(0, int(min_area))
    areas = stats[:, cv2.CC_STAT_AREA]
    return [int(i) for i in range(1, len(areas)) if areas[i] >= min_area]


# ---------------------------------------------------------------------------
# 2. OBJECT DETECTION
# ---------------------------------------------------------------------------

def detect_objects(mask: np.ndarray, min_area: int = 50,
                   connectivity: int = 8) -> List[dict]:
    """
    Detect every object in a binary mask.

    What it does / how it works:
        1. cv2.connectedComponentsWithStats labels each connected
           foreground blob and reports its bounding box, pixel area
           and centroid.
        2. filter_small_components() drops blobs below `min_area`.
        3. For each surviving blob, its outer contour is extracted from
           that blob alone (cropped to its bounding box). Doing it per
           blob guarantees each contour belongs to exactly the object it
           is listed with, even when objects are close together.

    Why connected components are the primary tool:
        They give area, bounding box and centroid directly, with no
        extra computation. Contours are only needed for perimeter and
        shape measurements, which the features module computes.

    Why the contour is stored per object:
        A global cv2.findContours call returns contours in an order that
        is not tied to the component labels. Keeping the contour inside
        each object's dict avoids having to match the two afterwards.

    Touching objects:
        Objects that touch after segmentation form ONE component and are
        counted as one. This is an honest limitation of connected
        components (no watershed or splitting is attempted), and it
        will show up as a counting error in the evaluation phase.

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        min_area: minimum component area in pixels.
        connectivity: 4 or 8 (which neighbors count as connected).

    Returns:
        list[dict], one per object, ordered by label (roughly
        top-to-bottom, left-to-right). Empty list if nothing is found.
        Keys: "id", "label", "bbox", "centroid", "area", "contour".
    """
    mask = _to_mask(mask)
    connectivity = _validate_connectivity(connectivity)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    objects: List[dict] = []
    for label in filter_small_components(stats, min_area):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]

        # Contour of this component only. The offset moves the contour
        # from crop coordinates back to full-image coordinates.
        crop = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        found, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE, offset=(x, y))
        if not found:
            continue  # cannot happen for a non-empty component; defensive
        contour = max(found, key=cv2.contourArea) if len(found) > 1 else found[0]

        objects.append({
            "id": len(objects) + 1,
            "label": label,
            "bbox": (x, y, w, h),
            "centroid": (float(cx), float(cy)),
            "area": area,
            "contour": contour,
        })

    return objects


# ---------------------------------------------------------------------------
# 3. CONTOURS
# ---------------------------------------------------------------------------

def get_contours(mask: np.ndarray, min_area: int = 50,
                 connectivity: int = 8) -> List[np.ndarray]:
    """
    Return the outer contour of every object that survives the size
    filter.

    Why CHAIN_APPROX_NONE:
        Every boundary pixel is kept (no simplification), so perimeter
        calculated from the contour later is as accurate as the pixel
        grid allows.

    Why RETR_EXTERNAL:
        Only outer boundaries are returned. Holes inside an object are
        not separate objects, and ignoring them keeps perimeter and
        shape measurements about the object's silhouette.

    Args:
        mask: (H, W) mask; any non-zero pixel is foreground.
        min_area: minimum component area in pixels.
        connectivity: 4 or 8.

    Returns:
        list of (K, 1, 2) int32 contours, in the same order as
        detect_objects(). Empty list if nothing is found.
    """
    return [obj["contour"] for obj in detect_objects(mask, min_area, connectivity)]
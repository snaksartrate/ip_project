"""
detection.py

Detects individual objects from a cleaned binary mask, using connected
components (primary, gives area/centroid natively) and/or contours
(for perimeter/shape). Filters out components that are too small to be
real objects (as opposed to leftover noise).

Planned functions (to be implemented in Phase 4):

    detect_objects(mask, min_area=50) -> list[dict]
        Runs cv2.connectedComponentsWithStats, filters by `min_area`,
        and returns one dict per detected object with at least:
        {
            "label": int,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "area": int,
        }

    get_contours(mask) -> list[np.ndarray]
        Runs cv2.findContours for objects that survived filtering,
        used later for perimeter/circularity calculations.

    filter_small_components(stats, min_area) -> list[int]
        Helper returning the indices/labels of components to keep.

No algorithms are implemented yet -- this is Phase 1 (project setup) only.
"""

# Implementation begins in Phase 4.

"""
features.py

Extracts per-object geometric measurements from detected objects
(output of detection.py), producing the feature table shown in the
Streamlit app and used later for evaluation against ground truth.

Planned functions (to be implemented in Phase 4/5):

    compute_features(mask, contour, stats_row) -> dict
        Returns a dict with:
        {
            "area": ...,
            "perimeter": ...,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "width": ...,
            "height": ...,
            "circularity": ...,      # 4*pi*Area / Perimeter^2
            "equiv_diameter": ...,   # sqrt(4*Area/pi)
        }

    compute_all_features(mask, objects, contours) -> pandas.DataFrame
        Runs compute_features for every detected object and returns a
        tidy DataFrame for display / export.

No algorithms are implemented yet -- this is Phase 1 (project setup) only.
"""

# Implementation begins in Phase 4/5.

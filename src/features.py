"""
features.py

Extracts per-object geometric measurements from detected objects
(output of detection.py), producing the feature table shown in the
Streamlit app and used later for evaluation against ground truth.

All measurements are in PIXELS (and pixels squared for area). Converting
to real-world units needs a scale (e.g. mm per pixel), which this
project does not have.

Functions:

    compute_features(obj) -> dict
        Measurements for one object dict from detection.detect_objects():
        {
            "id": int,
            "area": int,                 # pixel count
            "perimeter": float,          # closed contour length
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "width": int,                # bounding-box width
            "height": int,               # bounding-box height
            "circularity": float,        # 4*pi*Area / Perimeter^2, in [0, 1]
            "equiv_diameter": float,     # sqrt(4*Area/pi)
        }

    compute_all_features(objects) -> pandas.DataFrame
        One row per object, flat columns, ready for st.dataframe().
"""

from typing import List
import numpy as np
import cv2
import pandas as pd


FEATURE_COLUMNS = [
    "id", "area", "perimeter", "centroid_x", "centroid_y",
    "bbox_x", "bbox_y", "width", "height",
    "circularity", "equiv_diameter",
]


def compute_features(obj: dict) -> dict:
    """
    Compute the geometric measurements of one detected object.

    How each value is computed:
        area:
            Pixel count from connected components (not the contour
            polygon area, which is slightly smaller).
        perimeter:
            cv2.arcLength(contour, closed=True): the length of the
            outer boundary, following the contour through boundary
            pixel centers.
        width / height:
            Size of the axis-aligned bounding box.
        circularity:
            4 * pi * Area / Perimeter^2
            1.0 for a perfect circle, lower for elongated or irregular
            shapes (a square gives about 0.785).
        equiv_diameter:
            sqrt(4 * Area / pi), the diameter of a circle with the same
            area. Useful for comparing round objects such as coins.

    Why circularity is clamped to 1.0:
        Area is a pixel count while perimeter follows pixel centers, so
        on very small, round objects the formula can slightly exceed 1.
        That is a discretization artefact, not a real value above a
        perfect circle, so it is clamped.

    Args:
        obj: one dict from detection.detect_objects(), with keys
            "id", "area", "bbox", "centroid" and "contour".

    Returns:
        dict (see module docstring).
    """
    area = int(obj["area"])
    x, y, w, h = obj["bbox"]
    perimeter = float(cv2.arcLength(obj["contour"], True))

    if perimeter > 0:
        circularity = min(1.0, 4.0 * np.pi * area / (perimeter ** 2))
    else:
        circularity = 0.0  # single-pixel object: no meaningful shape

    return {
        "id": int(obj["id"]),
        "area": area,
        "perimeter": perimeter,
        "bbox": (int(x), int(y), int(w), int(h)),
        "centroid": (float(obj["centroid"][0]), float(obj["centroid"][1])),
        "width": int(w),
        "height": int(h),
        "circularity": float(circularity),
        "equiv_diameter": float(np.sqrt(4.0 * area / np.pi)),
    }


def compute_all_features(objects: List[dict]) -> pd.DataFrame:
    """
    Compute features for every detected object and return a tidy table.

    The tuple fields (bbox, centroid) are split into separate columns
    so the DataFrame displays and exports cleanly. The contour is not
    included.

    Args:
        objects: list of dicts from detection.detect_objects().

    Returns:
        pandas.DataFrame with columns FEATURE_COLUMNS, one row per
        object. If `objects` is empty, an empty DataFrame with the same
        columns is returned, so callers never need a special case.
    """
    rows = []
    for obj in objects:
        f = compute_features(obj)
        rows.append({
            "id": f["id"],
            "area": f["area"],
            "perimeter": round(f["perimeter"], 2),
            "centroid_x": round(f["centroid"][0], 1),
            "centroid_y": round(f["centroid"][1], 1),
            "bbox_x": f["bbox"][0],
            "bbox_y": f["bbox"][1],
            "width": f["width"],
            "height": f["height"],
            "circularity": round(f["circularity"], 3),
            "equiv_diameter": round(f["equiv_diameter"], 2),
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
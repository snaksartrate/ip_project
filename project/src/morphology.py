"""
morphology.py

Shared morphological post-processing pipeline, applied identically
regardless of which segmentation method produced the input mask. Keeping
this stage common (rather than method-specific) is part of the fair
comparison principle: only segmentation should vary between methods.

Planned functions (to be implemented in Phase 4):

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
        intermediate results for display, e.g.:
        {
            "opened": ...,
            "closed": ...,
            "final": ...,
        }

No algorithms are implemented yet -- this is Phase 1 (project setup) only.
"""

# Implementation begins in Phase 4.

"""
src package

Core library for the "Comparative Study of Image Segmentation Techniques
for Automatic Object Detection and Measurement" project.

Modules:
    preprocessing  -- grayscale conversion, denoising, resizing, histograms
    segmentation   -- the four segmentation methods (global, otsu, adaptive, edge-based)
    morphology     -- shared morphological post-processing pipeline
    detection      -- object detection (connected components / contours) + filtering
    features       -- per-object feature/measurement extraction
    evaluation     -- per-method metrics against ground truth (count accuracy, IoU, timing)
    visualization  -- overlay drawing and comparison plots

Each processing module is intentionally kept independent and stateless:
every function takes the data it needs as arguments and returns plain
NumPy arrays / dicts, so any stage can be called directly, tested in
isolation, or swapped in the Streamlit app without touching other stages.
"""

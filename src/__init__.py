"""
src package

Core library for the "Comparative Image Segmentation and Object
Measurement System" project.

Modules:
    preprocessing  -- load/validate, resize, grayscale, denoising, histograms
    segmentation   -- the four segmentation methods (global, otsu, adaptive, edge-based)
    morphology     -- shared morphological cleanup (opening, closing, small-region removal)
    detection      -- object detection (connected components / contours) + size filter
    features       -- per-object measurements (area, perimeter, circularity, ...)
    evaluation     -- IoU, object matching, count accuracy, per-method metrics
    degradation    -- synthetic noise, low light and motion blur for stress-testing
    optimization   -- grid search for Adaptive and Edge-Based parameters
    profiling      -- execution time and RAM measurement (psutil + tracemalloc)
    visualization  -- histogram, detection overlays, degradation and comparison charts

Each module is independent and stateless: every function takes the data
it needs as arguments and returns plain NumPy arrays / dicts / DataFrames,
and none imports Streamlit. Any stage can be called directly, tested in
isolation, or swapped in the app without touching other stages.

Mask convention everywhere: uint8, 255 = foreground, 0 = background.
"""
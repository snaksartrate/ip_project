"""
profiling.py

Measures the execution-time and RAM footprint of a segmentation
algorithm, to judge whether it is feasible on a memory-constrained
(IoT / edge) device.

Why this does NOT use memory_profiler.memory_usage():
    memory_usage() starts its monitor as a multiprocessing.Process. On
    Windows, child processes are "spawned", which re-imports the main
    script. Under Streamlit the main script is app.py, so the child
    re-ran the whole app (with no uploaded file) and crashed. Everything
    here therefore runs inside the current process, using threads.

How the measurement works:
    1. TIMING: the function is run `repeats` times and the median
       wall-clock time is reported. No tracing is active during these
       runs, so the timing is not distorted.
    2. PROCESS RAM (RSS): during those same runs, a background thread
       samples the process's resident memory with psutil. The first
       sample is the baseline; the highest is the peak.
    3. ALLOCATION PEAK: the function is then run once more under
       tracemalloc, which reports the peak bytes allocated by Python and
       NumPy during that run. This is the figure that reflects the
       algorithm itself, because process RSS is dominated by Python,
       OpenCV and Streamlit. It does not see OpenCV's internal C++
       scratch buffers, so treat it as a lower bound.

No Streamlit code lives here, so the module stays testable on its own.
"""

from typing import Callable, Optional
import threading
import time
import tracemalloc

import numpy as np
import psutil

from src import segmentation


DEFAULT_REPEATS = 10
DEFAULT_INTERVAL_S = 0.002
_MB = 1024 * 1024


class _RamSampler:
    """
    Samples this process's RSS from a background thread.

    Usage:
        sampler = _RamSampler(interval)
        sampler.start()
        ... run the code being measured ...
        sampler.stop()
        sampler.baseline_mb, sampler.peak_mb
    """

    def __init__(self, interval: float):
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    def _sample(self) -> float:
        return self._proc.memory_info().rss / _MB

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_mb = max(self.peak_mb, self._sample())

    def start(self) -> None:
        # First sample is taken synchronously so the baseline is exact.
        self.baseline_mb = self._sample()
        self.peak_mb = self.baseline_mb
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        # One last sample, in case the peak happened right at the end.
        self.peak_mb = max(self.peak_mb, self._sample())


def profile_algorithm(func: Callable, *args, **kwargs) -> dict:
    """
    Profile execution time and RAM of any callable.

    Two keyword arguments are reserved for the profiler itself and are
    removed before `func` is called:
        profile_repeats:  how many timed runs (default 10)
        profile_interval: RSS sampling interval in seconds (default 0.002)

    Example:
        profile_algorithm(segmentation.segment_otsu, gray_image,
                          object_polarity="darker")

    Args:
        func: the function to profile (e.g. a segment_*() function).
        *args, **kwargs: forwarded to `func`.

    Returns:
        dict:
            {
                "execution_time_ms": float,   # median wall-clock per run
                "peak_memory_mb": float,      # highest process RSS seen
                "memory_delta_mb": float,     # peak minus baseline (>= 0)
                "baseline_memory_mb": float,  # RSS before the runs
                "alloc_peak_mb": float,       # tracemalloc peak, one run
                "n_runs": int,
                "mask": np.ndarray | None,    # mask returned by func
                "result": Any,                # whatever func returned
            }

        "mask" is taken from result["mask"] when func returns a dict
        (as every segment_*() function does), or is the result itself
        when func returns an array. Otherwise it is None.
    """
    repeats = max(1, int(kwargs.pop("profile_repeats", DEFAULT_REPEATS)))
    interval = float(kwargs.pop("profile_interval", DEFAULT_INTERVAL_S))

    # --- Pass 1: timing + RSS sampling (no tracing, so timing is clean) ---
    times_ms = []
    result = None
    sampler = _RamSampler(interval)
    sampler.start()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            times_ms.append((time.perf_counter() - start) * 1000)
    finally:
        sampler.stop()

    # --- Pass 2: allocation peak of a single run, under tracemalloc ---
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()
        base_current, _ = tracemalloc.get_traced_memory()
        func(*args, **kwargs)
        _, peak_bytes = tracemalloc.get_traced_memory()
        alloc_peak_mb = max(0.0, (peak_bytes - base_current) / _MB)
    finally:
        if not was_tracing:
            tracemalloc.stop()

    if isinstance(result, dict) and "mask" in result:
        mask = result["mask"]
    elif isinstance(result, np.ndarray):
        mask = result
    else:
        mask = None

    return {
        "execution_time_ms": float(np.median(times_ms)),
        "peak_memory_mb": sampler.peak_mb,
        "memory_delta_mb": max(0.0, sampler.peak_mb - sampler.baseline_mb),
        "baseline_memory_mb": sampler.baseline_mb,
        "alloc_peak_mb": alloc_peak_mb,
        "n_runs": repeats,
        "mask": mask,
        "result": result,
    }


def profile_all_methods(gray_image: np.ndarray,
                        method_params: Optional[dict] = None,
                        repeats: int = DEFAULT_REPEATS) -> list:
    """
    Profile all four segmentation methods on the same image and return
    one row per method, ready for st.dataframe(pd.DataFrame(rows)).

    This is a convenience wrapper (not in the blueprint) that keeps
    app.py a thin UI controller.

    Args:
        gray_image: (H, W) uint8 preprocessed grayscale image. The SAME
            image is used for every method, so the comparison is fair.
        method_params: optional dict of per-method parameters, e.g.
            {"adaptive": {"block_size": 25, "C": 5}}. Methods not
            listed use the defaults below.
        repeats: runs per method (see profile_algorithm).

    Returns:
        list[dict], one per method:
            {
                "Method": str,
                "Time (ms)": float,
                "Peak RAM (MB)": float,
                "RAM increase (MB)": float,
                "Alloc peak (MB)": float,
                "Image size (px)": "WxH",
            }
    """
    defaults = {
        "global": {"threshold": 127, "object_polarity": "darker"},
        "otsu": {"object_polarity": "darker"},
        "adaptive": {"block_size": 11, "C": 2,
                     "adaptive_method": "gaussian",
                     "object_polarity": "darker"},
        "edge": {"low_threshold": 50, "high_threshold": 150,
                 "closing_kernel_size": 5},
    }
    labels = {
        "global": "Global Thresholding",
        "otsu": "Otsu Thresholding",
        "adaptive": "Adaptive Thresholding",
        "edge": "Edge-Based Segmentation",
    }
    method_params = method_params or {}
    height, width = gray_image.shape[:2]

    rows = []
    for key, func in segmentation.SEGMENTATION_METHODS.items():
        params = {**defaults[key], **method_params.get(key, {})}
        prof = profile_algorithm(func, gray_image,
                                 profile_repeats=repeats, **params)
        rows.append({
            "Method": labels[key],
            "Time (ms)": round(prof["execution_time_ms"], 3),
            "Peak RAM (MB)": round(prof["peak_memory_mb"], 1),
            "RAM increase (MB)": round(prof["memory_delta_mb"], 2),
            "Alloc peak (MB)": round(prof["alloc_peak_mb"], 3),
            "Image size (px)": f"{width}x{height}",
        })
    return rows
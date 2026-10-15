# Comparative Image Segmentation and Object Measurement System

A classical (no deep learning) image-processing project that compares four
segmentation methods -- **Global**, **Otsu**, **Adaptive** and **Edge-Based**
-- on one shared preprocessing, cleanup and detection pipeline, so any
difference in the results comes from the segmentation step alone.

Stack: Python 3.11, OpenCV, NumPy, pandas, matplotlib, psutil, Streamlit.

## Project status

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup | Done |
| 2 | Image input and preprocessing | Done |
| 3 | Segmentation (4 methods) | Done |
| -- | Advanced benchmarking: degradation, auto-tuning, profiling | Done |
| 4 | Morphology, object detection, feature extraction | Done |
| 5 | Synthetic dataset with exact ground truth | Done |
| 6 | Evaluation engine and batch benchmark | Done |
| 7 | Comparison engine and charts | Done |
| 8 | Benchmark tab in the app | Done |
| 9 | Optional colour-segmentation experiment | Skipped |
| 10 | Documentation | This file |

## Headline results (synthetic dataset, 20 images)

All numbers are means over 20 images (5 per condition). Metrics are defined
under "Evaluation" below. Full tables are written to `comparison/`.

**Mean mask IoU (pixel overlap with the true mask, after cleanup)**

| Run | Global | Otsu | Adaptive | Edge |
|---|---|---|---|---|
| Baseline (default parameters, no denoising) | 0.778 | 0.779 | 0.283 | 0.744 |
| Oracle-tuned, blueprint ranges | 0.985 | 0.779 | 0.389 | 0.751 |
| Oracle-tuned, wide adaptive range | 0.985 | 0.779 | 0.864 | 0.751 |
| Default parameters + 5x5 median filter | 0.777 | 0.779 | 0.275 | 0.970 |

**Mean count accuracy (1.0 = exact object count)**

| Run | Global | Otsu | Adaptive | Edge |
|---|---|---|---|---|
| Baseline | 0.810 | 0.810 | 0.629 | 0.671 |
| Oracle-tuned | 0.870 | 0.810 | 0.879 | 0.670 |
| Default parameters + median filter | 0.810 | 0.810 | 0.629 | 0.879 |

**Baseline mask IoU by condition**

| Condition | Global | Otsu | Adaptive | Edge |
|---|---|---|---|---|
| normal | 0.996 | 0.999 | 0.254 | 0.969 |
| noisy | 0.992 | 0.993 | 0.365 | 0.075 |
| uneven | 0.130 | 0.129 | 0.266 | 0.965 |
| touching | 0.993 | 0.997 | 0.248 | 0.969 |

### What fails, and why

Each item says how well the explanation is supported.

- **Touching objects: every method scores 0.514 count accuracy.** Touching
  circles form one connected component, so they are counted as one (for
  example 10 true objects become 5 blobs). This is a limit of the detection
  stage, not of segmentation. The `gt_blobs` column gives the best count a
  perfect mask could reach. Watershed splitting was not attempted.
  *Supported by the data.*
- **Edge-based on noisy images fails at default settings; denoising fixes it.**
  Noise breaks the Canny edges, the flood fill leaks through the gaps, and the
  mask covers most of the image (mask IoU 0.075). With a 5x5 median filter,
  the same edge method reaches 0.970 on noisy images. Morphology cannot repair
  it and parameter tuning does not either. *Supported by the data (the median
  run isolates the cause).*
- **Global and Otsu fail on uneven lighting at default settings (mask IoU
  about 0.13).** For Global the default threshold of 127 is the problem: an
  oracle-tuned Global reaches 0.947 on the same images. Otsu fails regardless
  of tuning (0.129), consistent with its assumption of a clean two-peak
  histogram, which a brightness ramp breaks. *Global part supported by the
  data; the Otsu explanation follows from theory and was not tested separately.*
- **Adaptive needs both a large block size and a large C on this data.** A
  parameter sweep on one image (`normal_01`) shows two separate failure modes:
  - *Small C -> speckle.* At the default C=2 the mask is mostly false
    foreground (precision 0.13 at block 11, 0.26 at block 99): sensor noise
    alone falls below the local mean minus C. Raising C removes it
    (precision 0.78-1.00 at C >= 8).
  - *Small block -> hollow objects.* With C high enough to remove speckle but
    a small block (11, against objects 44-96 px wide), only each coin's rim
    differs from its neighbourhood, so recall collapses (0.12-0.18) while
    precision stays near 1.0. A block comparable to the object size recovers
    the interior (block 99, C=8: IoU 0.86, recall 0.91, precision 0.94).
  Oracle tuning with the blueprint range (block <= 31, C <= 10) picked block
  31 on every image, the ceiling, and reached only 0.389. The wide range
  (block <= 99) lifts it to 0.864, but every image then picked block 99, which
  is again the cap of `segment_adaptive`, so 0.864 is still a lower bound.
  *Supported by the data, but the sweep was on one image only. A 5x5 median
  filter did not fix adaptive at the default C=2.*
- **Adaptive's mean IoU of matched objects (0.952) is misleading.** Matching
  rasterises each object by filling its outer contour, so a ring-shaped or
  noisy detection can still match a solid coin. Use mask IoU and measurement
  error for adaptive.

### Caveats

- Tuned runs are **oracle upper bounds**: parameters are tuned on each test
  image using that image's own ground truth, which a real system cannot do.
- The grid-search objective is raw mask IoU (before cleanup).
- Global winning overall (0.985) is specific to this clean synthetic data (flat
  dark objects, gentle illumination ramp). Do not generalise it.
- The mean of 10.55 edge false positives in the tuned run comes almost entirely
  from one image (`noisy_05`, 200 detections). It is not a trend.
- `mean_iou` and `mean_measurement_error_pct` are NaN when a method matches
  nothing on an image, so they only cover images where something was found.
  The `images_with_no_match` column in the summary shows how many were left out.
- Sub-millisecond timings vary noticeably between runs (Global measured 0.126
  ms in one run and 0.286 ms in another). Compare them as orders of magnitude,
  or use the Hardware Profile tab, which takes a median of repeated runs.
- Synthetic data only. No real photographs were tested.
- All measurements are in pixels; no real-world scale is known.
- Using Otsu's own output as a pseudo ground truth (an option in the app) is a
  demo only and is biased toward Otsu.

## Project structure

```
project/
├── app.py                    # Streamlit UI controller (no algorithms)
├── generate_dataset.py       # builds the synthetic dataset
├── run_benchmark.py          # batch benchmark (also imported by app.py)
├── run_tuned_benchmark.py    # oracle-tuned benchmark run
├── run_comparison.py         # aggregates a results CSV into tables + charts
├── requirements.txt
├── README.md
├── src/
│   ├── preprocessing.py      # load/validate, resize, grayscale, filtering, histogram
│   ├── segmentation.py       # the 4 segmentation methods + dispatcher
│   ├── morphology.py         # erode, dilate, opening, closing, clean_mask
│   ├── detection.py          # connected components, contours, size filter
│   ├── features.py           # per-object measurements
│   ├── evaluation.py         # IoU, object matching, count accuracy
│   ├── degradation.py        # noise, low light, motion blur injection
│   ├── optimization.py       # grid search for adaptive and edge parameters
│   ├── profiling.py          # time and RAM measurement
│   └── visualization.py      # histogram, overlays, degradation and comparison charts
├── dataset/                  # generated: images/ and ground_truth/
├── results/                  # generated: per-run results CSVs
└── comparison/               # generated: summary CSVs and charts
```

Architecture rules: `src/` modules are stateless and contain no Streamlit
code; `app.py` only collects input and displays results.
**Mask convention everywhere:** uint8, 255 = foreground, 0 = background; any
non-zero pixel is treated as foreground and strict {0, 255} is returned.

## Setup and running

```bash
python -m venv venv
venv\Scripts\activate            # Windows (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python generate_dataset.py                        # once; seed 42 gives the same dataset
python run_benchmark.py                           # baseline -> results/benchmark_results.csv
python run_benchmark.py --noise median --out-file results_median.csv
python run_tuned_benchmark.py --out-file tuned_results.csv
python run_tuned_benchmark.py --wide-adaptive --out-file tuned_wide_results.csv
python run_comparison.py --results results/tuned_wide_results.csv --name tuned_wide

streamlit run app.py                              # http://localhost:8501
```

`run_comparison.py` writes `<name>_summary_by_method.csv`,
`<name>_summary_by_condition.csv` and bar charts to `comparison/`.
`run_tuned_benchmark.py` also writes `comparison/tuned_params.csv` (the
parameters chosen per image); each tuned run overwrites it.

## The app

Upload any image; the tabs are:

| Tab | What it does |
|---|---|
| Pipeline | Step-by-step preprocessing (resize, grayscale, denoise, histogram) and one segmentation method with adjustable parameters |
| Stress-Test | Degrades the image (Gaussian noise, low light, motion blur) at increasing severity and plots IoU for all four methods |
| Auto-Tuning | Grid search for the best Adaptive and Edge parameters against a ground-truth mask, with progress |
| Hardware Profile | Time and RAM per method |
| Measurements | Shared morphology, detection and per-object measurements for all four methods; CSV download |
| Benchmark | Runs or loads the dataset benchmark, charts any metric, and lets you browse each dataset image with every method's detections |

A shared "Ground truth & polarity" section above the tabs provides the object
polarity and an optional ground-truth mask for Stress-Test, Auto-Tuning and
Measurements. The Benchmark tab ignores the uploaded image and uses the dataset.

## Module notes

### Preprocessing (Phase 2)

Pipeline: resize (aspect ratio kept, only if larger than the limit) ->
grayscale (0.299R + 0.587G + 0.114B) -> optional noise reduction -> 256-bin
histogram. Gaussian blur smooths general noise but softens edges; median
filtering removes salt-and-pepper noise and preserves edges better. Kernel
sizes are forced odd and in range, so a slider can never crash OpenCV.

### Segmentation (Phase 3)

All four methods take the same grayscale image and return the same result
dict (`mask`, `method`, `parameters`, `threshold`, `processing_time_ms`,
`intermediate`). Object polarity defaults to "darker".

| Method | Idea | Weak spot |
|---|---|---|
| Global | One fixed threshold T | A single T cannot follow uneven lighting |
| Otsu | Picks T that maximises between-class variance | Needs a two-peak histogram |
| Adaptive | Per-pixel threshold = local (mean or Gaussian) average minus C | Sensitive to block size and C |
| Edge-Based | Canny -> gap closing -> flood-fill hole filling | Noise breaks edges and the fill leaks |

### Morphology, detection, features (Phase 4)

- **Morphology:** `clean_mask` runs opening (removes specks), closing (fills
  holes) and small-region removal, with an elliptical structuring element.
  The same parameters are used for every method (open 3, close 5, min area 50).
- **Detection:** `detect_objects` uses connected components for area, bounding
  box and centroid, then extracts each object's outer contour separately.
  Objects that touch are one component and are counted as one.
- **Features:** area, perimeter, bounding box, centroid, circularity
  (4*pi*area/perimeter^2, clamped to 1) and equivalent diameter, in pixels.

### Dataset (Phase 5)

`generate_dataset.py` draws 640x480 grayscale images of dark coin-like objects
on a bright background, so the ground truth is exact. Per image it writes the
image, a binary mask, a 16-bit instance map (one label per object) and a row in
`counts.csv` with the true count. Four conditions, 5 images each by default:

| Condition | What is added |
|---|---|
| normal | Mild Gaussian sensor noise (sigma 3), separated objects |
| noisy | Gaussian noise (sigma 12) plus 3% salt-and-pepper |
| uneven | A linear brightness ramp falling to roughly 35-50% of full brightness, plus sigma 3 noise |
| touching | Chains of 2-3 circles overlapping by 0-3 px, plus 1-2 separate objects |

### Evaluation (Phase 6)

Per image and method (`evaluation.py`):

- **Matching:** greedy one-to-one matching of detected objects to true objects
  by IoU (threshold 0.5), using the instance map so touching objects stay
  separate. Gives `correct`, `missed` (true objects not found) and
  `false_positives` (detections matching nothing).
- **count_accuracy:** 1 - |detected - true| / true, clamped to [0, 1].
- **mean_iou:** mean IoU of matched objects only.
- **mask_iou:** pixel-level IoU of the whole mask; always defined, so it still
  scores total failures such as a flooded edge mask.
- **raw_mask_iou:** the same before cleanup.
- **mean_measurement_error_pct:** mean area error of matched objects.

### Auto-tuning and profiling

- **Grid search** (`optimization.py`): Adaptive over block size x C (default
  15 x 21 = 315 runs; the wide run uses block 3-99 and C -10..30, 2009 runs)
  and Edge over Canny low x high thresholds (20 runs). Ties keep the first
  best, so results are deterministic. `run_tuned_benchmark.py` also sweeps the
  Global threshold (5-250, step 5) so Global is tuned as fairly as the others.
- **Profiling** (`profiling.py`): median wall-clock time over repeated runs,
  process RSS sampled by a background thread (psutil), and one
  `tracemalloc` pass for the allocation peak. Peak RAM is the whole Streamlit
  process; the allocation peak is the better indicator of the algorithm itself
  but excludes OpenCV's internal C++ buffers, so it is a lower bound.

## Deviations from the original blueprint

1. **`memory_profiler` replaced by psutil + tracemalloc.** `memory_usage()`
   starts a child process; on Windows that re-imports `app.py` under Streamlit
   and crashed. Everything now runs in-process with threads.
2. `calculate_iou` is used instead of the earlier name `compute_iou`.
3. Extra arguments in `optimization.py`: `object_polarity`, `adaptive_method`,
   `closing_kernel_size`, `progress_callback`, and `block_sizes` / `c_values`
   (for the wide search).
4. Added to `evaluation.py`: `gt_objects_from_instances` (the binary mask merges
   touching objects), the `mask_iou` metric, and an optional `image_shape`
   argument on `match_objects`.
5. Added a Global threshold sweep in `run_tuned_benchmark.py` (the blueprint has
   no Global search).
6. Added `gt_blobs` and `raw_mask_iou` columns to the benchmark output.
7. The wide adaptive range is used for one tuned run only; the Auto-Tuning tab
   keeps the blueprint ranges.

## Known limitations

- Touching objects are not split (no watershed).
- Synthetic data only; real-photo robustness is untested.
- Colour segmentation (Phase 9) was not attempted.
- `use_container_width=True` in `app.py` may warn as deprecated in newer
  Streamlit versions; the replacement is `width="stretch"`.

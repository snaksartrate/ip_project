# Comparative Image Segmentation and Object Measurement System

A classical (non-ML) image-processing project comparing four segmentation
methods — **Global Thresholding**, **Otsu Thresholding**, **Adaptive
Thresholding**, and **Edge-Based Segmentation** — on a shared
preprocessing/post-processing pipeline, for a fair comparison.

## Project status

- ✅ **Phase 1 — Project Setup** — folder structure, module skeletons, minimal runnable Streamlit app
- ✅ **Phase 2 — Image Input & Preprocessing** — upload, resize, grayscale, noise reduction, histogram
- ✅ **Phase 3 — Segmentation** (current) — described below
- ⬜ Phase 4+ — Morphology, object detection, feature extraction, evaluation, comparison — not implemented yet

## Project structure

```
project/
├── app.py                  # Streamlit entrypoint (UI only — no processing logic)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # ✅ implemented: load/validate, resize, grayscale, filtering, histogram
│   ├── segmentation.py     # stub — the 4 segmentation methods (Phase 3)
│   ├── morphology.py       # stub — shared morphological post-processing (Phase 4)
│   ├── detection.py        # stub — connected components / contours + filtering (Phase 4)
│   ├── features.py         # stub — per-object measurements (Phase 4)
│   ├── evaluation.py       # stub — metrics vs. ground truth (Phase 6)
│   └── visualization.py    # plot_histogram() implemented; overlays/comparison plots are stubs (Phase 4/10)
├── dataset/
│   ├── images/              # input images (normal/noisy/uneven/touching) — Phase 5
│   └── ground_truth/        # object counts / masks for evaluation — Phase 5
├── results/                 # saved per-run outputs — later phases
├── comparison/               # aggregated comparison outputs — later phases
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the app

```bash
streamlit run app.py
```

Opens in your browser, usually at `http://localhost:8501`.

---

## Phase 2 — Image Input & Preprocessing

### What this phase implements

Given an uploaded image, the app walks through:

```
Upload → Validate → Original display
       → Resize (if needed, aspect-ratio preserved)
       → Grayscale conversion
       → Noise reduction (None / Gaussian / Median — user-selected)
       → Intensity histogram (+ optional histogram-equalization experiment)
```

Every step is shown in the UI as a labeled section with before/after images,
so each stage's effect is visible on its own — nothing later in the
pipeline (segmentation, morphology, object detection, evaluation) exists
yet; those come in later phases.

### Why grayscale conversion

Global, Otsu, and Adaptive thresholding (Phase 3) each classify a pixel by
comparing a *single* intensity number to a threshold — they need a
one-channel input. OpenCV's BGR→Gray conversion uses the perceptually
weighted sum `Gray = 0.299R + 0.587G + 0.114B` (green contributes most,
matching human brightness sensitivity), rather than a plain average of the
three channels.

### Gaussian vs. Median filtering

| | Gaussian | Median |
|---|---|---|
| **How** | Weighted average using a bell-curve kernel | Replaces pixel with the median of its neighborhood |
| **Good against** | General smooth (e.g. sensor) noise | Salt-and-pepper (impulse) noise specifically |
| **Edge preservation** | Softens edges | Preserves edges noticeably better |
| **Cost** | Cheaper | More expensive (requires sorting) |
| **Weakness** | Doesn't remove outlier pixels, just smears them | Can erode fine thin details if kernel is too large |

Median filtering matters especially for this project because the planned
"noisy images" dataset category (Phase 5) uses salt-and-pepper noise as one
of its noise models — this is the filter expected to handle it best.

### Why the histogram matters

The 256-bin intensity histogram shown in this phase is exactly what Otsu's
method (Phase 3) will search over to find an optimal threshold — a clearly
bimodal histogram (two peaks) is the condition Otsu is designed to exploit;
a flat or single-peaked histogram predicts Otsu may struggle on that image.
Showing it now sets up that explanation for later.

### Kernel size safety

All kernel-size inputs (Gaussian/Median) pass through a shared validator
that clamps them into a valid odd range — even, zero, negative, or
excessively large values are corrected automatically rather than crashing
OpenCV's blur functions.

### Histogram equalization (optional)

Available as an explicit opt-in checkbox in the sidebar. It does **not**
alter the standard pipeline or the main histogram shown by default — it's
a separate, clearly-labeled side experiment, per the approved design.

### How to run and manually test Phase 2

```bash
streamlit run app.py
```

Then, in the browser:

1. **Normal image** — upload any everyday JPG/PNG photo → confirm original
   image, dimensions, grayscale, and histogram all display correctly.
2. **Grayscale image** — upload a PNG that's already grayscale → confirm it
   still loads and displays without error (internally normalized to 3
   channels on load, then converted back to single-channel for processing).
3. **Color image with strong hues** — confirm grayscale conversion looks
   reasonable (not just a channel drop).
4. **Noisy image** — upload an image with visible speckle/salt-and-pepper
   noise, select **Median** filtering, and confirm the "after filtering"
   panel visibly cleans it up; compare against **Gaussian** on the same
   image to see the difference described above.
5. **Very large image** (e.g., a several-thousand-pixel-wide photo) —
   confirm the "Resizing" section reports the resize and shows the new
   (smaller) dimensions.
6. **Invalid upload** — rename a `.txt` file to `.jpg` and upload it →
   confirm the app shows a clean error message instead of crashing.

All six cases above were also run directly against `src/preprocessing.py`
(bypassing the UI) using synthetic test images during development, and
completed without exceptions.

---

## Phase 3 — Segmentation

### What this phase implements

Four classical segmentation methods, all consuming the identical
preprocessed grayscale image from Phase 2, under one common interface
in `src/segmentation.py`:

```
Preprocessed grayscale (Phase 2 output)
        │
        ▼
[ selected method ]
   ├── Global Thresholding
   ├── Otsu Thresholding
   ├── Adaptive Thresholding
   └── Edge-Based Segmentation
        │
        ▼
Binary mask (255 = foreground object, 0 = background)
```

Only the segmentation strategy changes between methods — this is what
keeps the eventual four-way comparison fair. Every method returns the
same result structure: `mask`, `method`, `parameters`, `threshold`,
`processing_time_ms`, and an optional `intermediate` dict.

**Mask convention:** every method is normalized so that **255/white =
foreground object** and **0/black = background** — no method is
allowed to silently return an inverted mask.

**Object polarity:** for the three thresholding-based methods, the UI
lets you specify whether objects are darker or brighter than the
background; this determines which side of the threshold is labeled
foreground, without changing what threshold value is computed.

### 1. Global Thresholding

- **Principle:** classify every pixel using one fixed, manually-chosen
  intensity cutoff `T`, applied uniformly across the whole image.
- **Math:** `mask(x,y) = 255 if I(x,y) < T else 0` (for "darker"
  objects; inverted for "brighter").
- **Parameters:** `threshold` (0–255, manual), `object_polarity`.
- **Input:** preprocessed grayscale image. **Output:** binary mask.
- **Advantages:** trivial to compute and explain; fully predictable.
- **Disadvantages:** one value can't fit varying lighting; needs manual
  re-tuning per image.
- **Works well when:** illumination is uniform and intensities are
  already well separated.
- **Expected failure case:** uneven illumination or varying object
  contrast within the same image.

### 2. Otsu Thresholding

- **Principle:** automatically computes the threshold that maximizes
  between-class variance of the image's own histogram (equivalently,
  minimizes intra-class variance of the two resulting groups).
- **Math:** searches all T for the one maximizing
  `sigma_b^2(T) = w0(T)·w1(T)·(mu0(T) - mu1(T))^2`.
- **Parameters:** `object_polarity` only — the threshold itself is
  never manually set.
- **Input:** preprocessed grayscale image (its histogram, specifically).
  **Output:** binary mask + the computed threshold.
- **Advantages:** automatic, no manual tuning; optimal for genuinely
  bimodal histograms.
- **Disadvantages:** assumes a bimodal histogram; degrades when that
  assumption breaks.
- **Works well when:** foreground/background intensities form two
  clean, separable clusters.
- **Expected failure case:** non-bimodal or uneven-illumination
  histograms, where there's no clear valley to find.

### 3. Adaptive Thresholding

- **Principle:** computes a separate local threshold per pixel from its
  own neighborhood, instead of one global value.
- **Math:** `T(x,y) = local_mean_or_gaussian_weighted_mean(neighborhood) - C`.
- **Parameters:** `block_size` (odd, ≥3), `C`, `adaptive_method`
  (mean/gaussian), `object_polarity`.
- **Input:** preprocessed grayscale image. **Output:** binary mask
  (no single representative threshold value — `threshold` is `None`).
- **Advantages:** handles uneven illumination much better than a global
  cutoff.
- **Disadvantages:** sensitive to `block_size` choice; can fragment
  uniform regions into noise if the neighborhood is too small; more
  expensive than global/Otsu.
- **Works well when:** illumination varies across the image but objects
  remain distinguishable from their immediate surroundings.
- **Expected failure case:** block size mismatched to object scale —
  too small fragments, too large behaves like a global threshold again.

### 4. Edge-Based Segmentation

- **Principle:** detects boundaries via intensity gradients (Canny),
  then converts that boundary information into a filled region mask.
- **Pipeline (specific to this method only):**
  `Grayscale → Canny edges → gap-closing (morphological close, scoped
  to this method) → flood-fill region filling → binary mask`.
- **Math:** Canny finds high-gradient pixels, thins them via
  non-maximum suppression, and keeps/rejects them via hysteresis
  thresholding between `low_threshold` and `high_threshold`. Region
  filling uses a standard flood-fill-from-border technique: flood the
  background from the corner, invert to isolate enclosed (interior)
  regions, then OR with the original edges.
- **Parameters:** `low_threshold`, `high_threshold` (Canny), 
  `closing_kernel_size` (gap-closing only — NOT the future general
  morphology module).
- **Input:** preprocessed grayscale image. **Output:** binary mask,
  plus intermediate `edges` and `closed_edges` for display.
- **Advantages:** doesn't require a bimodal histogram; can work where
  intensity thresholding can't, as long as a local gradient exists at
  the true boundary.
- **Disadvantages:** most noise-sensitive of the four; requires an
  extra gap-closing/filling step just to produce a comparable mask;
  unclosed gaps let the fill step leak across the whole image.
- **Works well when:** boundaries have a genuine local gradient even if
  global intensity separation is poor.
- **Expected failure case:** noisy images, where spurious gradient
  fragments create gaps that the closing step can't always bridge.

### Why the shared pipeline keeps the comparison fair

All four methods receive the exact same preprocessed grayscale image
(same resize, same grayscale conversion, same noise-reduction
settings). Nothing about the input differs between methods — only the
segmentation call itself. This means any difference in output mask
quality can be attributed to the segmentation strategy itself, not to
inconsistent inputs. (No claim about which method is "best" is made
yet — that requires the quantitative evaluation in later phases.)

### Testing performed

Each method was run on:
1. The approved Phase 2 five-coin test image (dark coins, bright uniform background)
2. A clean/uniform-illumination synthetic image
3. An uneven-illumination synthetic image (added intensity gradient)
4. A noisy synthetic image (salt-and-pepper)
5. A touching/closely-positioned-objects synthetic image

No accuracy numbers were computed (no ground truth or evaluation exists
yet — that's Phase 6). Observations below are qualitative only.

**Five-coin test image:** all four methods correctly separated 5
distinct foreground regions from the background, with default/
reasonable parameters (Global T=150, Otsu auto-computed T=100, Adaptive
block_size=25/C=5, Edge Canny 50/150).

**Clean image:** all four methods produced clean, well-separated
foreground blobs matching the 5 objects.

**Uneven illumination:** Global and Otsu used a single threshold across
the whole image, so their masks were visibly less consistent across the
bright vs. dim regions than Adaptive's locally-computed thresholds —
matching the expected theoretical behavior.

**Noisy image:** Global/Otsu/Adaptive masks picked up visible speckle
noise as small foreground specks (expected — no morphological cleanup
exists yet to remove them; that's Phase 4). Edge-based segmentation was
noticeably more affected: on the noisy test image, the foreground ratio
jumped to ~99% of the whole image, indicating the flood-fill leaked
through gaps left by noise-corrupted edges rather than staying confined
to the true object boundaries — a direct, observed instance of the
"edge gaps are a problem" failure mode described above, not a bug in
the implementation.

**Touching objects:** with default parameters, all four methods still
resolved the touching-object test image into separate regions in most
cases, though closely-overlapping boundaries are visually the most
fragile case across all four masks — full quantitative merge/split
behavior will be measurable once object counting exists (Phase 4+).

### Known limitations (expected, not yet addressed)

- No morphological cleanup yet — small noise specks in the masks are
  visible as-is; this is intentional and belongs to Phase 4.
- Edge-based segmentation's flood-fill can leak across large portions
  of the mask when edges are broken by noise — mitigating this further
  (larger closing kernel, prior denoising) is a tuning question for
  later phases, not solved here.
- No numerical accuracy/quality metrics exist yet — all testing in this
  phase is visual/qualitative only, per the phase scope.

## Upcoming phases

| Phase | Scope |
|---|---|
| 4 | Morphology, object detection, feature extraction |
| 5 | Dataset creation + ground truth annotation |
| 6 | Evaluation engine (count accuracy, IoU matching, timing) |
| 7 | Comparison engine + charts |
| 8 | Full Streamlit UI (all sections) |
| 9 | Optional color-segmentation experiment |
| 10 | Documentation/report polish |

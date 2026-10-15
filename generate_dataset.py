"""
generate_dataset.py

Generates a synthetic benchmark dataset with EXACT ground truth:
dark coin-like objects on a bright background, in four conditions.

    normal    -- even lighting, mild sensor noise, separated objects
    noisy     -- Gaussian noise + salt-and-pepper noise
    uneven    -- a strong brightness gradient across the whole image
    touching  -- clusters of objects that touch or slightly overlap

Output (relative to --out, default "dataset"):

    images/<condition>_<NN>.png               grayscale test image
    ground_truth/<condition>_<NN>.png         binary mask, 255 = object
    ground_truth/instances/<condition>_<NN>.png
                                              16-bit label map, 0 = background,
                                              1..N = one value per object
    ground_truth/counts.csv                   true object count per image

Because the mask is the shape that was drawn, there is no annotation
error. Objects are DARKER than the background, which matches the app's
default "Objects darker than background" polarity.

Touching objects: the binary mask is the union of the touching shapes,
so connected components will count a touching cluster as ONE object.
counts.csv holds the TRUE count, so this shows up as a counting error
in the evaluation phase. That is intended.

Usage (from the project/ folder):

    python generate_dataset.py
    python generate_dataset.py --per-condition 10 --seed 7
    python generate_dataset.py --out dataset --width 640 --height 480

Running it again overwrites the files it generates. The same seed always
produces the same dataset.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = ("normal", "noisy", "uneven", "touching")

BG_RANGE = (185, 225)     # background intensity before degradations
FG_RANGE = (45, 95)       # object intensity before degradations
MARGIN = 10               # min distance from objects to the image border (px)
GAP = 12                  # min gap between separate objects (px)


# ---------------------------------------------------------------------------
# Shape placement
# ---------------------------------------------------------------------------

def _radius(shape: dict) -> float:
    """Bounding radius of a shape (conservative for ellipses)."""
    return max(shape["a"], shape["b"])


def _fits(shape: dict, w: int, h: int) -> bool:
    r = _radius(shape)
    return (MARGIN <= shape["cx"] - r and shape["cx"] + r <= w - MARGIN
            and MARGIN <= shape["cy"] - r and shape["cy"] + r <= h - MARGIN)


def _clear_of(shape: dict, others: list, gap: float) -> bool:
    """True if `shape` keeps at least `gap` px from every shape in `others`."""
    for o in others:
        d = math.hypot(shape["cx"] - o["cx"], shape["cy"] - o["cy"])
        if d < _radius(shape) + _radius(o) + gap:
            return False
    return True


def _add_separate(rng, w, h, shapes, n, allow_ellipse=True) -> None:
    """Append up to `n` mutually separated shapes to `shapes` (in place)."""
    target = len(shapes) + n
    for _ in range(5000):
        if len(shapes) >= target:
            return
        a = rng.uniform(22, 48)
        if allow_ellipse and rng.random() < 0.3:
            b = a * rng.uniform(0.6, 0.9)
            angle = rng.uniform(0, 180)
        else:
            b, angle = a, 0.0
        r = max(a, b)
        if w - 2 * (MARGIN + r) <= 0 or h - 2 * (MARGIN + r) <= 0:
            continue
        shape = {
            "cx": rng.uniform(MARGIN + r, w - MARGIN - r),
            "cy": rng.uniform(MARGIN + r, h - MARGIN - r),
            "a": a, "b": b, "angle": angle,
        }
        if _clear_of(shape, shapes, GAP):
            shapes.append(shape)


def _build_cluster(rng, w, h, size):
    """
    Build `size` circles that touch or overlap slightly, chained together.
    Returns a list of shapes, or None if the layout did not fit.
    """
    r0 = rng.uniform(24, 40)
    first = {"cx": rng.uniform(MARGIN + 3 * r0, w - MARGIN - 3 * r0),
             "cy": rng.uniform(MARGIN + 3 * r0, h - MARGIN - 3 * r0),
             "a": r0, "b": r0, "angle": 0.0}
    cluster = [first]
    while len(cluster) < size:
        for _ in range(60):
            parent = cluster[rng.integers(len(cluster))]
            r2 = rng.uniform(24, 40)
            ang = rng.uniform(0, 2 * math.pi)
            overlap = rng.uniform(0, 3)               # px of overlap
            dist = parent["a"] + r2 - overlap
            cand = {"cx": parent["cx"] + dist * math.cos(ang),
                    "cy": parent["cy"] + dist * math.sin(ang),
                    "a": r2, "b": r2, "angle": 0.0}
            if not _fits(cand, w, h):
                continue
            # not buried inside another member of the same cluster
            if all(math.hypot(cand["cx"] - m["cx"], cand["cy"] - m["cy"])
                   >= 0.9 * (cand["a"] + m["a"]) for m in cluster):
                cluster.append(cand)
                break
        else:
            return None
    return cluster


def layout_separate(rng, w, h):
    shapes = []
    _add_separate(rng, w, h, shapes, int(rng.integers(5, 9)))
    return shapes


def layout_touching(rng, w, h):
    shapes = []
    for _ in range(int(rng.integers(2, 4))):          # 2-3 clusters
        for _ in range(300):
            cluster = _build_cluster(rng, w, h, int(rng.integers(2, 4)))
            if cluster and all(_clear_of(s, shapes, GAP) for s in cluster):
                shapes.extend(cluster)
                break
    _add_separate(rng, w, h, shapes, int(rng.integers(1, 3)), allow_ellipse=False)
    return shapes


# ---------------------------------------------------------------------------
# Rendering and ground truth
# ---------------------------------------------------------------------------

def render_scene(rng, w, h, shapes):
    """
    Paint the scene and return (image float32, binary mask, instance map).

    The mask and instance map come from the exact shapes, before any
    blur or noise is added, so they are perfect ground truth.
    """
    bg = rng.uniform(*BG_RANGE)
    img = np.full((h, w), bg, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    inst = np.zeros((h, w), dtype=np.uint16)

    for i, s in enumerate(shapes, start=1):
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(tmp, (int(round(s["cx"])), int(round(s["cy"]))),
                    (int(round(s["a"])), int(round(s["b"]))),
                    s["angle"], 0, 360, 255, -1)
        region = tmp > 0
        img[region] = rng.uniform(*FG_RANGE)
        mask[region] = 255
        inst[region] = i

    # Soften edges slightly, as a real camera would.
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return img, mask, inst


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------

def add_gaussian(img, rng, sigma):
    return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


def add_salt_pepper(img, rng, amount):
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def illumination_field(rng, w, h):
    """Linear brightness ramp at a random angle, from 1.0 down to ~0.4."""
    low = rng.uniform(0.35, 0.5)
    angle = rng.uniform(0, 2 * math.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xs - w / 2) * math.cos(angle) + (ys - h / 2) * math.sin(angle)
    t = (proj - proj.min()) / (proj.max() - proj.min())
    return (1.0 - (1.0 - low) * t).astype(np.float32)


def apply_condition(img, rng, condition):
    w, h = img.shape[1], img.shape[0]
    if condition == "noisy":
        img = add_gaussian(img, rng, sigma=12)
        img = add_salt_pepper(img, rng, amount=0.03)
    elif condition == "uneven":
        img = img * illumination_field(rng, w, h)
        img = add_gaussian(img, rng, sigma=3)
    else:                                   # normal, touching
        img = add_gaussian(img, rng, sigma=3)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(out_dir: Path, per_condition: int, seed: int, w: int, h: int) -> list:
    img_dir = out_dir / "images"
    gt_dir = out_dir / "ground_truth"
    inst_dir = gt_dir / "instances"
    for d in (img_dir, gt_dir, inst_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for ci, condition in enumerate(CONDITIONS):
        for i in range(1, per_condition + 1):
            rng = np.random.default_rng(seed + 1000 * ci + i)
            shapes = (layout_touching(rng, w, h) if condition == "touching"
                      else layout_separate(rng, w, h))
            clean, mask, inst = render_scene(rng, w, h, shapes)
            image = apply_condition(clean, rng, condition)

            name = f"{condition}_{i:02d}.png"
            cv2.imwrite(str(img_dir / name), image)
            cv2.imwrite(str(gt_dir / name), mask)
            cv2.imwrite(str(inst_dir / name), inst)
            rows.append({"filename": name, "condition": condition,
                         "n_objects": len(shapes), "polarity": "darker"})

    with open(gt_dir / "counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "condition", "n_objects", "polarity"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic benchmark dataset.")
    parser.add_argument("--out", default="dataset", help="output folder (default: dataset)")
    parser.add_argument("--per-condition", type=int, default=5,
                        help="images per condition (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    rows = generate(Path(args.out), args.per_condition, args.seed, args.width, args.height)

    print(f"Wrote {len(rows)} images to '{args.out}/images' with masks in "
          f"'{args.out}/ground_truth'.")
    for condition in CONDITIONS:
        counts = [r["n_objects"] for r in rows if r["condition"] == condition]
        print(f"  {condition:9s} {len(counts)} images, "
              f"{min(counts)}-{max(counts)} objects each")


if __name__ == "__main__":
    main()
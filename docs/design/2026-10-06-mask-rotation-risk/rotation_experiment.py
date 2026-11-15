"""Read-only experiment: how much does rotating a binary mask change it?

Masks: savinim's applied masks in the three Danionella review runs (zarr
opened mode="r"). For each mask, rotate about its centroid with three methods
and measure area change, new pieces / holes, round-trip IoU, and skeleton
endpoint (spur) change for the body.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy import ndimage
from skimage.draw import polygon2mask
from skimage.measure import find_contours
from skimage.morphology import skeletonize

RUN = "refined_subject_masks_danionella_strong_single_inframe_review_20260804"
RECS = ["2026-07-01T15-11-03Z_arena_1_DefaultScreen", "2026-07-01T14-32-13Z_arena_2_DefaultScreen", "2026-07-01T14-32-13Z_arena_4_DefaultScreen"]
ANGLES = [0, 5, 10, 20, 30, 45, 90]
PER_COMPONENT = 200
EIGHT = np.ones((3, 3), bool)


def pieces(m):
    return ndimage.label(m, structure=EIGHT)[1]


def holes(m):
    filled = ndimage.binary_fill_holes(m)
    return ndimage.label(filled & ~m.astype(bool))[1]


def endpoints(m):
    sk = skeletonize(m.astype(bool))
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    return int(np.sum(sk & (nb == 1)))


def matrix(m, angle):
    ys, xs = np.nonzero(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    return cv2.getRotationMatrix2D((cx, cy), angle, 1.0)


def rot_nn(m, angle, M):
    return (cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def rot_bilinear(m, angle, M):
    return (cv2.warpAffine(m.astype(np.float32), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)


def rot_contour(m, angle, M):
    out = np.zeros(m.shape, bool)
    for contour in find_contours(np.pad(m, 1).astype(float), 0.5):
        rc = contour - 1.0  # (row, col) in original coordinates
        xy = np.stack([rc[:, 1], rc[:, 0], np.ones(len(rc))], axis=1) @ M.T
        out ^= polygon2mask(m.shape, np.stack([xy[:, 1], xy[:, 0]], axis=1))
    return out.astype(np.uint8)


METHODS = {"pixel_nearest": rot_nn, "pixel_bilinear": rot_bilinear, "outline": rot_contour}


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def crop(mask):
    ys, xs = np.nonzero(mask)
    pad = int(np.hypot(np.ptp(ys) + 1, np.ptp(xs) + 1)) // 2 + 4
    y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    sub = mask[y0:ys.max() + pad + 1, x0:xs.max() + pad + 1]
    return np.pad(sub, pad, constant_values=0)  # room so nothing leaves the frame


def main(out_path):
    rng = np.random.default_rng(0)
    samples = defaultdict(list)
    for rec in RECS:
        run = zarr.open_group(f"/groups/johnson/johnsonlab/jeremy/recordings/{rec}/zarr/{rec}_training.zarr", mode="r", use_consolidated=False)[f"refined_subject_masks_runs/{RUN}"]
        labels = list(run.attrs["mask_labels"])
        masks = np.asarray(run["masks_roi"][:])
        for ci, name in enumerate(labels):
            for row in range(masks.shape[0]):
                m = (masks[row, ci] > 0).astype(np.uint8)
                if m.sum() >= 10:
                    samples[name].append(m)
    rows = []
    for name, ms in samples.items():
        pick = rng.choice(len(ms), size=min(PER_COMPONENT, len(ms)), replace=False)
        for idx in pick:
            m = crop(ms[idx])
            base = {"area": int(m.sum()), "pieces": pieces(m), "holes": holes(m), "endpoints": endpoints(m) if name == "subject_body" else None}
            for angle in ANGLES:
                M = matrix(m, angle)
                Minv = matrix(m, -angle)
                for method, fn in METHODS.items():
                    r = fn(m, angle, M)
                    back = fn(r, -angle, matrix(r, -angle)) if r.sum() else r
                    rows.append({
                        "component": name, "method": method, "angle": angle,
                        "area_rel": float(r.sum() / base["area"] - 1.0),
                        "new_pieces": max(0, pieces(r) - base["pieces"]),
                        "new_holes": max(0, holes(r) - base["holes"]),
                        "roundtrip_iou": iou(m, back),
                        "endpoint_delta": (endpoints(r) - base["endpoints"]) if base["endpoints"] is not None else None,
                        "area_px": base["area"],
                    })
    Path(out_path).write_text(json.dumps(rows))
    print("masks per component:", {k: min(PER_COMPONENT, len(v)) for k, v in samples.items()}, "rows:", len(rows))


if __name__ == "__main__":
    main(sys.argv[1])

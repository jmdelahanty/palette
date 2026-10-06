"""Read-only: does rotation resampling break mask-derived tail keypoints?

For tail-review archives (refined mask run + refined keypoint run on the same
crop), derive tails with the production derive_tail_seed (legacy, then the
head-anchored fallback for its eligible reasons) on the original masks, then
on masks + head points rigidly rotated by an angle (on a padded canvas so
nothing leaves the frame). Compare validity, failure reasons, and how far the
derived tail stations move once rotated back.
"""

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from scipy import ndimage
import numpy as np
import zarr

from fisheye.analysis.subject_shape_runs import HEAD_ANCHORED_CENTERLINE_METHOD
from fisheye.training.mask_tail_apply_refresh import FALLBACK_FAILURE_REASONS
from fisheye.training.mask_tail_keypoints import LEGACY_SCHEMA_NAME, SCHEMA_NAME, derive_tail_seed

STORE = "/home/delahantyj@hhmi.org/.palette/tail_review_preview.sqlite"
ANGLES = [10, 20, 45]
METHODS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("pixel_nearest", "pixel_bilinear")


def derive(masks, labels, head, schema):
    legacy = derive_tail_seed(masks, labels, head, schema_name=schema)
    valid = np.asarray(legacy["tail_valid"], bool).copy()
    reasons = [str(r) for r in legacy["tail_failure_reason"]]
    points = np.asarray(legacy["keypoints_roi"], float).copy()
    retry_rows = np.flatnonzero(~valid & np.isin(reasons, FALLBACK_FAILURE_REASONS))
    method = np.array(["legacy"] * len(masks), dtype=object)
    if retry_rows.size:
        retry = derive_tail_seed(masks[retry_rows], labels, head[retry_rows], schema_name=schema, method=HEAD_ANCHORED_CENTERLINE_METHOD)
        ok = np.asarray(retry["tail_valid"], bool)
        for j, row in enumerate(retry_rows):
            if ok[j]:
                valid[row] = True
                reasons[row] = "ok"
                points[row] = np.asarray(retry["keypoints_roi"])[j]
                method[row] = "head_anchored"
            else:
                reasons[row] = str(retry["tail_failure_reason"][j])
    return valid, reasons, points, method


def rotate_rows(masks, head, angle, method, pad):
    n, c, h, w = masks.shape
    H, W = h + 2 * pad, w + 2 * pad
    out = np.zeros((n, c, H, W), np.uint8)
    head_out = np.full_like(head, np.nan, dtype=np.float64)
    mats = []
    for i in range(n):
        body = masks[i, 0]
        ys, xs = np.nonzero(body)
        if ys.size == 0:
            mats.append(None)
            continue
        cx, cy = xs.mean() + pad, ys.mean() + pad
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mats.append(M)
        for k in range(c):
            src = np.pad(masks[i, k], pad)
            if method == "pixel_nearest":
                out[i, k] = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_NEAREST) > 0
            else:
                out[i, k] = cv2.warpAffine(src.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR) >= 0.5
        if method == "pixel_bilinear_keep_largest":
            # What "Remove stray pieces" does to the body after a rotation.
            labels_, count = ndimage.label(out[i, 0], structure=np.ones((3, 3), bool))
            if count > 1:
                sizes = np.bincount(labels_.ravel())[1:]
                out[i, 0] = (labels_ == (1 + int(np.argmax(sizes)))).astype(np.uint8)
        pts = head[i] + pad
        head_out[i] = np.c_[pts, np.ones(len(pts))] @ M.T
    return out, head_out, mats


def unrotate(points_xy, M, pad):
    Minv = cv2.invertAffineTransform(M)
    return np.c_[points_xy, np.ones(len(points_xy))] @ Minv.T - pad


def pairs():
    conn = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True)
    seen = {}
    current_mask_run = {}
    for kind, scope_json, updated in conn.execute(
        "select workflow_kind, scope_json, updated_at_utc from labeling_tasks where state != 'superseded' order by updated_at_utc"
    ):
        scope = json.loads(scope_json or "{}")
        path = scope.get("zarr_path")
        if path and kind in ("keypoints", "subject_mask_component"):
            seen.setdefault(path, set()).add(kind)
            if kind == "subject_mask_component" and scope.get("refined_run"):
                current_mask_run[path] = scope["refined_run"]
    for path in sorted(seen):
        try:
            root = zarr.open_group(path, mode="r", use_consolidated=False)
        except Exception:
            continue
        masks = root.get("refined_subject_masks_runs")
        poses = root.get("refined_keypoints_runs")
        if masks is None or poses is None:
            continue
        names = [current_mask_run[path]] if current_mask_run.get(path) in masks else sorted(masks.group_keys())[-1:]
        for mname in names:
            m = masks[mname]
            labels = tuple(str(v) for v in (m.attrs.get("mask_labels") or []))
            if not {"subject_body", "swim_bladder"}.issubset(labels) or "masks_roi" not in m:
                continue
            for pname in sorted(poses.group_keys()):
                p = poses[pname]
                if (p.attrs.get("schema_id") == m.attrs.get("schema_id")
                        and p.attrs.get("source_bindings") == m.attrs.get("source_bindings")
                        and p.attrs.get("source_bindings") is not None
                        and "keypoints_roi" in p and p["keypoints_roi"].shape[0] == m["masks_roi"].shape[0]):
                    yield path, mname, pname, labels
                    break


def main(out_path):
    results = defaultdict(lambda: {"rows": 0, "base_valid": 0, "still_valid": 0, "newly_valid": 0, "broken": 0,
                                   "broken_reasons": Counter(), "shift_px": [], "method_flip": 0})
    archives = 0
    used = set()
    for path, mname, pname, labels in pairs():
        key = (path, mname)
        if key in used:
            continue
        used.add(key)
        root = zarr.open_group(path, mode="r", use_consolidated=False)
        m, p = root[f"refined_subject_masks_runs/{mname}"], root[f"refined_keypoints_runs/{pname}"]
        masks = (np.asarray(m["masks_roi"][:]) > 0).astype(np.uint8)
        order = [labels.index("subject_body")] + [i for i in range(len(labels)) if labels[i] != "subject_body"]
        masks = masks[:, order]
        labels_o = tuple(labels[i] for i in order)
        kp = np.asarray(p["keypoints_roi"][:], float)
        head = kp[:, :3]
        keep = np.isfinite(head).all(axis=(1, 2)) & (masks[:, 0].sum(axis=(1, 2)) > 0)
        if keep.sum() == 0:
            continue
        masks, head = masks[keep], head[keep]
        schema = SCHEMA_NAME if kp.shape[1] == 19 else LEGACY_SCHEMA_NAME
        try:
            base = derive(masks, labels_o, head, schema)
        except ValueError as exc:
            print("skip", Path(path).name, mname, exc, flush=True)
            continue
        base_valid, base_reasons, base_pts, base_method = base
        archives += 1
        pad = int(np.hypot(*masks.shape[2:])) // 3
        for method in METHODS:
            for angle in ANGLES:
                rmasks, rhead, mats = rotate_rows(masks, head, angle, method, pad)
                valid, reasons, pts, rmethod = derive(rmasks, labels_o, rhead, schema)
                r = results[(method, angle)]
                r["rows"] += len(masks)
                r["base_valid"] += int(base_valid.sum())
                for i in range(len(masks)):
                    if base_valid[i] and valid[i]:
                        r["still_valid"] += 1
                        back = unrotate(pts[i, 3:14], mats[i], pad)
                        r["shift_px"].append(float(np.nanmean(np.linalg.norm(back - base_pts[i, 3:14], axis=1))))
                        r["method_flip"] += int(rmethod[i] != base_method[i])
                    elif base_valid[i] and not valid[i]:
                        r["broken"] += 1
                        r["broken_reasons"][reasons[i]] += 1
                    elif valid[i] and not base_valid[i]:
                        r["newly_valid"] += 1
        print("done", Path(path).name, mname, int(keep.sum()), "rows; baseline valid", int(base_valid.sum()), flush=True)
        Path(out_path + ".partial").write_text(json.dumps({k[0] + "@" + str(k[1]): {**v, "broken_reasons": dict(v["broken_reasons"]), "shift_px": len(v["shift_px"])} for k, v in results.items()}))
    summary = {}
    for (method, angle), r in sorted(results.items()):
        s = np.asarray(r["shift_px"])
        summary[f"{method}@{angle}"] = {
            "rows": r["rows"], "baseline_valid": r["base_valid"], "still_valid": r["still_valid"],
            "broken": r["broken"], "broken_pct_of_valid": round(100 * r["broken"] / max(1, r["base_valid"]), 2),
            "newly_valid": r["newly_valid"], "broken_reasons": dict(r["broken_reasons"].most_common()),
            "tail_shift_px_median": round(float(np.median(s)), 3) if s.size else None,
            "tail_shift_px_p95": round(float(np.percentile(s, 95)), 3) if s.size else None,
            "method_flips": r["method_flip"],
        }
    Path(out_path).write_text(json.dumps({"archives": archives, "summary": summary}, indent=2))
    print(json.dumps({"archives": archives, "summary": summary}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import zarr

from fisheye.pose.schema import resolve_skeleton_edges_from_attrs
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.shared.type_conversions import normalize_attr as _as_text


@dataclass
class BackfillResult:
    status: str
    reason: Optional[str] = None
    used_source_fallback: bool = False


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix == ".zarr":
            yield root
            continue
        if not root.exists():
            continue
        if recursive:
            yield from root.rglob("zarr/*.zarr")
        else:
            yield from root.glob("*/zarr/*.zarr")


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    purpose = root.attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _select_run_names(refined_parent: zarr.Group, all_runs: bool) -> list[str]:
    if all_runs:
        try:
            return sorted(list(refined_parent.group_keys()))
        except Exception:
            return sorted(list(refined_parent.keys()))
    latest = refined_parent.attrs.get("latest")
    if latest and str(latest) in refined_parent:
        return [str(latest)]
    try:
        names = sorted(list(refined_parent.group_keys()))
    except Exception:
        names = sorted(list(refined_parent.keys()))
    if not names:
        return []
    return [names[-1]]


def _normalize_keypoint_labels(
    source_group: zarr.Group,
    *,
    n_keypoints: int,
) -> list[str]:
    labels_raw = source_group.attrs.get("keypoint_labels")
    labels: list[str] = []
    if isinstance(labels_raw, (list, tuple)):
        labels = [str(item).strip() for item in labels_raw if str(item).strip()]
    if len(labels) != n_keypoints:
        pose_schema = source_group.attrs.get("pose_schema")
        nodes: list[str] = []
        if isinstance(pose_schema, dict):
            raw_nodes = pose_schema.get("nodes")
            if isinstance(raw_nodes, (list, tuple)):
                for node in raw_nodes:
                    if isinstance(node, dict):
                        name = _as_text(node.get("name")) or _as_text(node.get("id"))
                    else:
                        name = _as_text(node)
                    if name:
                        nodes.append(name)
        if len(nodes) == n_keypoints:
            labels = nodes
    if len(labels) != n_keypoints:
        labels = [f"k{i}" for i in range(n_keypoints)]
    return labels


def _extract_skeleton_edges(
    source_group: zarr.Group,
    *,
    n_keypoints: int,
) -> tuple[np.ndarray, str]:
    resolved = resolve_skeleton_edges_from_attrs(
        dict(source_group.attrs),
        n_keypoints=int(n_keypoints),
    )
    if not resolved.edge_pairs:
        return np.zeros((0, 2), dtype=np.int16), resolved.source

    max_index = max(max(pair) for pair in resolved.edge_pairs)
    dtype = np.int16 if max_index <= np.iinfo(np.int16).max else np.int32
    return np.asarray(resolved.edge_pairs, dtype=dtype), resolved.source


def _resolve_roi_diagonal(root: zarr.Group, source_crop_run: Optional[str]) -> Optional[float]:
    if not source_crop_run:
        return None
    crop_group = root.get(f"crop_runs/{source_crop_run}")
    if crop_group is None or "roi_images" not in crop_group:
        return None
    try:
        shape = crop_group["roi_images"].shape
        roi_h = float(shape[1])
        roi_w = float(shape[2])
    except Exception:
        return None
    diagonal = float(np.hypot(roi_h, roi_w))
    return diagonal if np.isfinite(diagonal) and diagonal > 0 else None


def _resolve_source_group(root: zarr.Group, refined_group: zarr.Group) -> tuple[zarr.Group, bool]:
    source_run = _as_text(resolve_source_keypoints_run(refined_group.attrs))
    keypoint_parent = root.get("keypoints_runs")
    if source_run and keypoint_parent is not None and source_run in keypoint_parent:
        return keypoint_parent[source_run], False
    return refined_group, True


def _edge_chunk_len(run_group: zarr.Group, rows: int) -> int:
    kp = run_group.get("keypoints_roi")
    if kp is not None and kp.chunks:
        return int(kp.chunks[0])
    refined_success = run_group.get("refined_success")
    if refined_success is not None and refined_success.chunks:
        return int(refined_success.chunks[0])
    return max(1, min(1024, rows))


def _set_edge_attrs(
    run_group: zarr.Group,
    *,
    edge_labels: Sequence[str],
    edge_source: str,
    edge_count: int,
    roi_diagonal: Optional[float],
) -> None:
    run_group.attrs["edge_distance_labels"] = list(edge_labels)
    run_group.attrs["edge_distance_source"] = edge_source
    run_group.attrs["edge_distance_count"] = int(edge_count)
    run_group.attrs["edge_distance_normalization"] = {
        "mode": "roi_diagonal",
        "roi_diagonal": roi_diagonal,
    }
    run_group.attrs["edge_distance_roi_diagonal"] = roi_diagonal


def _backfill_run_group(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    overwrite_existing: bool,
    apply: bool,
) -> BackfillResult:
    if "keypoints_roi" not in run_group:
        return BackfillResult(status="no_keypoints", reason="keypoints_roi missing")
    keypoints = run_group["keypoints_roi"]
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        return BackfillResult(status="bad_shape", reason=f"keypoints_roi has shape {keypoints.shape}")
    rows = int(keypoints.shape[0])
    n_keypoints = int(keypoints.shape[1])
    if rows <= 0:
        return BackfillResult(status="no_rows", reason="keypoints_roi empty")

    source_group, used_source_fallback = _resolve_source_group(root, run_group)
    keypoint_labels = _normalize_keypoint_labels(source_group, n_keypoints=n_keypoints)
    edge_pairs, edge_source = _extract_skeleton_edges(source_group, n_keypoints=n_keypoints)
    edge_count = int(edge_pairs.shape[0])
    if edge_count <= 0:
        return BackfillResult(
            status="no_edges",
            reason=f"no skeleton edges resolved for n_keypoints={n_keypoints}",
            used_source_fallback=used_source_fallback,
        )
    edge_labels = [f"{keypoint_labels[int(src)]}-{keypoint_labels[int(dst)]}" for src, dst in edge_pairs.tolist()]

    source_crop_run = _as_text(run_group.attrs.get("source_crop_run"))
    roi_diagonal = _resolve_roi_diagonal(root, source_crop_run)

    required_arrays = ("edge_pairs", "edge_distances", "edge_distances_norm", "edge_distance_valid")
    has_all_arrays = all(name in run_group for name in required_arrays)
    if has_all_arrays and not overwrite_existing:
        if apply:
            _set_edge_attrs(
                run_group,
                edge_labels=edge_labels,
                edge_source=edge_source,
                edge_count=edge_count,
                roi_diagonal=roi_diagonal,
            )
        return BackfillResult(status="skipped_existing", used_source_fallback=used_source_fallback)

    refined_success = (
        np.asarray(run_group["refined_success"][:], dtype=bool)
        if "refined_success" in run_group
        else np.ones((rows,), dtype=bool)
    )
    if refined_success.shape[0] != rows:
        return BackfillResult(
            status="shape_mismatch",
            reason=f"refined_success length {refined_success.shape[0]} != keypoints rows {rows}",
            used_source_fallback=used_source_fallback,
        )

    distances = np.full((rows, edge_count), np.nan, dtype=np.float32)
    distances_norm = np.full((rows, edge_count), np.nan, dtype=np.float32)
    valid = np.zeros((rows, edge_count), dtype=bool)
    roi_values = np.asarray(keypoints[:], dtype=np.float64)
    src_points = roi_values[:, edge_pairs[:, 0], :]
    dst_points = roi_values[:, edge_pairs[:, 1], :]
    valid = (
        refined_success[:, None]
        & np.all(np.isfinite(src_points), axis=2)
        & np.all(np.isfinite(dst_points), axis=2)
    )
    deltas = src_points - dst_points
    distances_64 = np.sqrt(np.sum(np.square(deltas), axis=2, dtype=np.float64))
    distances_64[~valid] = np.nan
    distances[:] = distances_64.astype(np.float32, copy=False)
    if roi_diagonal is not None and np.isfinite(roi_diagonal) and roi_diagonal > 0:
        distances_norm[:] = (distances_64 / float(roi_diagonal)).astype(np.float32, copy=False)

    if apply:
        chunk_len = _edge_chunk_len(run_group, rows)
        run_group.create_array(
            "edge_pairs",
            data=edge_pairs,
            chunks=(max(1, min(128, edge_count)), 2),
            overwrite=True,
        )
        run_group.create_array(
            "edge_distances",
            data=distances,
            chunks=(chunk_len, edge_count),
            overwrite=True,
        )
        run_group.create_array(
            "edge_distances_norm",
            data=distances_norm,
            chunks=(chunk_len, edge_count),
            overwrite=True,
        )
        run_group.create_array(
            "edge_distance_valid",
            data=valid,
            chunks=(chunk_len, edge_count),
            overwrite=True,
        )
        if "keypoint_labels" not in run_group.attrs:
            run_group.attrs["keypoint_labels"] = keypoint_labels
        _set_edge_attrs(
            run_group,
            edge_labels=edge_labels,
            edge_source=edge_source,
            edge_count=edge_count,
            roi_diagonal=roi_diagonal,
        )
    return BackfillResult(status="ok", used_source_fallback=used_source_fallback)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill per-ROI keypoint edge-distance arrays on refined keypoint runs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter zarr archives by purpose (default: analysis).",
    )
    parser.add_argument("--all-runs", action="store_true", help="Backfill all refined keypoint runs (default: latest).")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rewrite edge-distance arrays for runs where they already exist.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates (default: dry-run).")
    args = parser.parse_args(argv)

    roots = list(args.paths) if args.paths else [Path("/nvme1/recordings")]
    counts = {
        "zarr_scanned": 0,
        "runs_considered": 0,
        "ok": 0,
        "skipped_existing": 0,
        "no_keypoints": 0,
        "bad_shape": 0,
        "no_rows": 0,
        "no_edges": 0,
        "shape_mismatch": 0,
        "missing_refined": 0,
        "filtered_zarr_use": 0,
        "source_fallback": 0,
        "errors": 0,
    }

    any_zarr = False
    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        any_zarr = True
        counts["zarr_scanned"] += 1
        try:
            root = zarr.open_group(str(zarr_path), mode="a" if args.apply else "r")
            if args.zarr_use != "any":
                observed_use = _infer_zarr_use(root, zarr_path)
                if observed_use != args.zarr_use:
                    counts["filtered_zarr_use"] += 1
                    continue
            refined_parent = _pick_refined_parent(root)
            if refined_parent is None:
                counts["missing_refined"] += 1
                continue
            run_names = _select_run_names(refined_parent, all_runs=bool(args.all_runs))
            if not run_names:
                counts["missing_refined"] += 1
                continue
            for run_name in run_names:
                counts["runs_considered"] += 1
                result = _backfill_run_group(
                    root,
                    refined_parent[run_name],
                    overwrite_existing=bool(args.overwrite_existing),
                    apply=bool(args.apply),
                )
                if result.used_source_fallback:
                    counts["source_fallback"] += 1
                counts[result.status] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"error: {zarr_path}: {exc}")

    if not any_zarr:
        print("No zarr files found.")
        return 1

    mode = "Applied" if args.apply else "Dry run"
    print(
        "Keypoint edge-distance backfill: "
        f"scope={args.zarr_use} zarr_scanned={counts['zarr_scanned']} "
        f"runs_considered={counts['runs_considered']} filtered_zarr_use={counts['filtered_zarr_use']} "
        f"missing_refined={counts['missing_refined']} source_fallback={counts['source_fallback']} "
        f"errors={counts['errors']}"
    )
    print(
        f"{mode}: ok={counts['ok']} skipped_existing={counts['skipped_existing']} "
        f"no_keypoints={counts['no_keypoints']} bad_shape={counts['bad_shape']} "
        f"no_rows={counts['no_rows']} no_edges={counts['no_edges']} "
        f"shape_mismatch={counts['shape_mismatch']}"
    )
    return 0 if counts["errors"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

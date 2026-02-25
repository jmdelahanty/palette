from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.provenance_attrs import resolve_source_keypoints_run
from fisheye.utils.eye_mask_profile import EyeMaskSourceError, resolve_eye_mask_source

KEYPOINT_GROUP_CHOICES = ("refined_keypoints_runs", "keypoints_runs")
SUCCESS_DATASET_CANDIDATES = ("detection_success", "refined_success", "source_success")


@dataclass(frozen=True)
class DatasetLookup:
    dataset_id: str
    zarr_path: Path
    profile_source_eye_mask_path: Optional[str]
    profile_run: Optional[str]


@dataclass(frozen=True)
class AreaStats:
    count: int
    zeros: int
    zero_rate: float
    mean: float
    p10: float
    p50: float
    p90: float


def _is_group(value: object) -> bool:
    return hasattr(value, "attrs") and callable(getattr(value, "get", None))


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lookup_dataset(registry_path: Path, dataset_id: str) -> DatasetLookup:
    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            """
            SELECT dataset_id, zarr_path
            FROM datasets
            WHERE dataset_id = ?
            LIMIT 1;
            """,
            (str(dataset_id),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Dataset not found in registry: {dataset_id}")

        zarr_path_raw = _normalize_text(row["zarr_path"])
        if not zarr_path_raw:
            raise RuntimeError(f"Dataset row missing zarr_path: {dataset_id}")

        profile_row = registry.conn.execute(
            """
            SELECT source_eye_mask_path, profile_run
            FROM eye_mask_data_profile
            WHERE dataset_id = ?
            ORDER BY COALESCE(profile_created_utc, updated_utc) DESC, profile_run DESC
            LIMIT 1;
            """,
            (str(dataset_id),),
        ).fetchone()

        profile_source_eye_mask_path = (
            _normalize_text(profile_row["source_eye_mask_path"]) if profile_row is not None else None
        )
        profile_run = _normalize_text(profile_row["profile_run"]) if profile_row is not None else None
        return DatasetLookup(
            dataset_id=str(row["dataset_id"]),
            zarr_path=Path(zarr_path_raw),
            profile_source_eye_mask_path=profile_source_eye_mask_path,
            profile_run=profile_run,
        )
    finally:
        registry.close()


def _resolve_source_eye_mask_path(
    root: zarr.Group,
    *,
    source_eye_mask_path: Optional[str],
    refined_run: Optional[str],
    eye_run: Optional[str],
    profile_source_eye_mask_path: Optional[str],
) -> str:
    if source_eye_mask_path:
        return str(source_eye_mask_path).strip().strip("/")
    if refined_run:
        return f"refined_eye_masks_runs/{str(refined_run).strip().strip('/')}"
    if eye_run:
        return f"eye_masks_runs/{str(eye_run).strip().strip('/')}"
    if profile_source_eye_mask_path:
        return str(profile_source_eye_mask_path).strip().strip("/")
    resolved = resolve_eye_mask_source(root)
    return resolved.eye_mask_path


def _load_source_group(root: zarr.Group, source_eye_mask_path: str) -> zarr.Group:
    try:
        source = root[source_eye_mask_path]
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Source eye-mask path not found: {source_eye_mask_path}") from exc
    if not isinstance(source, zarr.Group):
        raise RuntimeError(f"Source path is not a group: {source_eye_mask_path}")
    return source


def _latest_run_name(group: zarr.Group) -> Optional[str]:
    latest = _normalize_text(group.attrs.get("latest"))
    if latest and latest in group:
        return latest
    names: list[str] = []
    for name in getattr(group, "keys", lambda: [])():
        try:
            child = group[name]
        except Exception:
            continue
        if _is_group(child):
            names.append(str(name))
    if not names:
        return None
    return sorted(names)[-1]


def _resolve_keypoint_source(
    root: zarr.Group,
    source_group: zarr.Group,
    *,
    explicit_group: Optional[str],
    explicit_run: Optional[str],
) -> tuple[str, str, zarr.Group, list[str]]:
    notes: list[str] = []
    available_groups: dict[str, zarr.Group] = {}
    for name in KEYPOINT_GROUP_CHOICES:
        group = root.get(name)
        if _is_group(group):
            available_groups[name] = group

    if explicit_group:
        explicit_group = str(explicit_group).strip()
        if explicit_group not in KEYPOINT_GROUP_CHOICES:
            raise RuntimeError(f"Invalid --keypoint-group: {explicit_group}")
        if explicit_group not in available_groups:
            raise RuntimeError(f"Keypoint group not found in zarr: {explicit_group}")
        if explicit_run:
            explicit_run = str(explicit_run).strip()
            if explicit_run not in available_groups[explicit_group]:
                raise RuntimeError(f"Keypoint run not found: {explicit_group}/{explicit_run}")
            return explicit_group, explicit_run, available_groups[explicit_group][explicit_run], notes
        latest = _latest_run_name(available_groups[explicit_group])
        if latest is None:
            raise RuntimeError(f"No runs found under keypoint group: {explicit_group}")
        notes.append(f"using_latest:{explicit_group}/{latest}")
        return explicit_group, latest, available_groups[explicit_group][latest], notes

    if explicit_run:
        run_name = str(explicit_run).strip()
        matches: list[tuple[str, zarr.Group]] = []
        for group_name in KEYPOINT_GROUP_CHOICES:
            group = available_groups.get(group_name)
            if group is None:
                continue
            if run_name in group:
                matches.append((group_name, group[run_name]))
        if not matches:
            raise RuntimeError(f"Keypoint run not found in refined/raw groups: {run_name}")
        if len(matches) > 1:
            notes.append(
                f"keypoint run '{run_name}' found in multiple groups; preferring {KEYPOINT_GROUP_CHOICES[0]}"
            )
        group_name, run_group = matches[0]
        return group_name, run_name, run_group, notes

    attrs = dict(source_group.attrs)
    source_run = _normalize_text(resolve_source_keypoints_run(attrs))
    source_group_name = _normalize_text(attrs.get("source_keypoint_group"))
    if source_run:
        if source_group_name and source_group_name in available_groups and source_run in available_groups[source_group_name]:
            notes.append(f"using_source_run:{source_group_name}/{source_run}")
            return source_group_name, source_run, available_groups[source_group_name][source_run], notes
        matches: list[tuple[str, zarr.Group]] = []
        for group_name in KEYPOINT_GROUP_CHOICES:
            group = available_groups.get(group_name)
            if group is None:
                continue
            if source_run in group:
                matches.append((group_name, group[source_run]))
        if matches:
            if len(matches) > 1:
                notes.append(
                    f"source run '{source_run}' found in multiple groups; preferring {KEYPOINT_GROUP_CHOICES[0]}"
                )
            group_name, run_group = matches[0]
            notes.append(f"using_source_run:{group_name}/{source_run}")
            return group_name, source_run, run_group, notes
        notes.append(f"source keypoint run missing:{source_run}")

    for group_name in KEYPOINT_GROUP_CHOICES:
        group = available_groups.get(group_name)
        if group is None:
            continue
        latest = _latest_run_name(group)
        if latest:
            notes.append(f"fallback_latest:{group_name}/{latest}")
            return group_name, latest, group[latest], notes
    raise RuntimeError("No keypoint runs found in refined/raw groups.")


def _resolve_success_dataset_name(kp_group: zarr.Group) -> Optional[str]:
    for name in SUCCESS_DATASET_CANDIDATES:
        if name in kp_group:
            return name
    return None


def _xy_payload(values: np.ndarray) -> list[Optional[float]]:
    x = float(values[0]) if np.isfinite(values[0]) else None
    y = float(values[1]) if np.isfinite(values[1]) else None
    return [x, y]


def _compute_zero_union_keypoint_summary(
    *,
    root: zarr.Group,
    source_group: zarr.Group,
    zero_rows: np.ndarray,
    frame_values: Optional[np.ndarray],
    explicit_group: Optional[str],
    explicit_run: Optional[str],
    max_preview: int,
) -> dict[str, Any]:
    try:
        kp_group_name, kp_run_name, kp_group, notes = _resolve_keypoint_source(
            root,
            source_group,
            explicit_group=explicit_group,
            explicit_run=explicit_run,
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    keypoints_arr = kp_group.get("keypoints_roi")
    if keypoints_arr is None:
        return {
            "status": "missing",
            "reason": f"{kp_group_name}/{kp_run_name} missing keypoints_roi",
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "notes": notes,
        }

    success_name = _resolve_success_dataset_name(kp_group)
    if success_name is None:
        return {
            "status": "missing",
            "reason": (
                f"{kp_group_name}/{kp_run_name} missing success dataset "
                f"({', '.join(SUCCESS_DATASET_CANDIDATES)})"
            ),
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "notes": notes,
        }

    success_arr = kp_group.get(success_name)
    if success_arr is None:
        return {
            "status": "missing",
            "reason": f"{kp_group_name}/{kp_run_name} missing {success_name}",
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "notes": notes,
        }

    keypoints = np.asarray(keypoints_arr[:], dtype=np.float32)
    success_flags = np.asarray(success_arr[:], dtype=np.bool_)
    if keypoints.ndim != 3 or keypoints.shape[1] < 3 or keypoints.shape[2] < 2:
        return {
            "status": "error",
            "reason": (
                f"{kp_group_name}/{kp_run_name}/keypoints_roi must be (N,>=3,>=2), "
                f"got {tuple(keypoints.shape)}"
            ),
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "success_dataset": success_name,
            "notes": notes,
        }
    if success_flags.ndim != 1 or int(success_flags.shape[0]) != int(keypoints.shape[0]):
        return {
            "status": "error",
            "reason": (
                f"{kp_group_name}/{kp_run_name}/{success_name} row mismatch with keypoints_roi "
                f"({success_flags.shape} vs {keypoints.shape})"
            ),
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "success_dataset": success_name,
            "notes": notes,
        }
    if int(keypoints.shape[0]) == 0:
        return {
            "status": "ok",
            "keypoint_group": kp_group_name,
            "keypoint_run": kp_run_name,
            "success_dataset": success_name,
            "notes": notes,
            "rows_total": 0,
            "zero_union_rows": 0,
        }

    left = np.asarray(keypoints[:, 1, :2], dtype=np.float32)
    right = np.asarray(keypoints[:, 2, :2], dtype=np.float32)
    left_finite = np.all(np.isfinite(left), axis=1)
    right_finite = np.all(np.isfinite(right), axis=1)
    finite_pair = left_finite & right_finite
    distinct = np.any(np.abs(left - right) > 1e-6, axis=1)
    keypoint_valid = success_flags & finite_pair & distinct

    zero_mask = np.zeros((keypoints.shape[0],), dtype=np.bool_)
    zero_mask[np.asarray(zero_rows, dtype=np.int64)] = True
    zero_count = int(np.sum(zero_mask))

    zero_success = int(np.sum(zero_mask & success_flags))
    zero_valid = int(np.sum(zero_mask & keypoint_valid))
    zero_nonfinite = int(np.sum(zero_mask & ~finite_pair))
    zero_nondistinct = int(np.sum(zero_mask & finite_pair & ~distinct))
    zero_success_but_invalid = int(np.sum(zero_mask & success_flags & ~keypoint_valid))

    preview_rows = np.asarray(zero_rows[: max_preview], dtype=np.int64)
    preview: list[dict[str, Any]] = []
    for roi in preview_rows:
        idx = int(roi)
        payload: dict[str, Any] = {
            "roi_idx": idx,
            "keypoint_success": bool(success_flags[idx]),
            "left_finite": bool(left_finite[idx]),
            "right_finite": bool(right_finite[idx]),
            "eyes_distinct": bool(distinct[idx]),
            "keypoint_valid": bool(keypoint_valid[idx]),
            "left_xy": _xy_payload(left[idx]),
            "right_xy": _xy_payload(right[idx]),
        }
        if frame_values is not None and int(frame_values.shape[0]) == int(keypoints.shape[0]):
            payload["frame_idx"] = int(frame_values[idx])
        preview.append(payload)

    return {
        "status": "ok",
        "keypoint_group": kp_group_name,
        "keypoint_run": kp_run_name,
        "success_dataset": success_name,
        "notes": notes,
        "rows_total": int(keypoints.shape[0]),
        "keypoint_success_rows": int(np.sum(success_flags)),
        "keypoint_success_rate": float(np.mean(success_flags)),
        "keypoint_valid_rows": int(np.sum(keypoint_valid)),
        "keypoint_valid_rate": float(np.mean(keypoint_valid)),
        "zero_union_rows": zero_count,
        "zero_union_with_keypoint_success_rows": zero_success,
        "zero_union_with_keypoint_success_rate": (
            float(zero_success / float(zero_count)) if zero_count > 0 else 0.0
        ),
        "zero_union_with_keypoint_valid_rows": zero_valid,
        "zero_union_with_keypoint_valid_rate": (
            float(zero_valid / float(zero_count)) if zero_count > 0 else 0.0
        ),
        "zero_union_nonfinite_eye_keypoints_rows": zero_nonfinite,
        "zero_union_nondistinct_eye_keypoints_rows": zero_nondistinct,
        "zero_union_keypoint_success_but_invalid_rows": zero_success_but_invalid,
        "zero_union_keypoint_preview": preview,
    }


def _area_stats(values: np.ndarray) -> AreaStats:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D area vector, got shape={arr.shape}")
    if arr.size == 0:
        return AreaStats(count=0, zeros=0, zero_rate=0.0, mean=0.0, p10=0.0, p50=0.0, p90=0.0)
    zeros = int(np.sum(arr <= 0.0))
    return AreaStats(
        count=int(arr.size),
        zeros=zeros,
        zero_rate=float(zeros / float(arr.size)),
        mean=float(np.mean(arr)),
        p10=float(np.percentile(arr, 10)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
    )


def _iter_zero_spans(zero_mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(zero_mask)
    if idx.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for current_raw in idx[1:]:
        current = int(current_raw)
        if current == prev + 1:
            prev = current
            continue
        spans.append((start, prev))
        start = current
        prev = current
    spans.append((start, prev))
    return spans


def _chunk_rows(masks_arr: zarr.Array, chunk_rows_hint: int) -> int:
    chunks = getattr(masks_arr, "chunks", None)
    if isinstance(chunks, tuple) and chunks:
        return max(1, int(chunks[0]))
    if isinstance(chunks, int):
        return max(1, int(chunks))
    return max(1, int(chunk_rows_hint))


def _compute_area_vectors(
    masks_arr: zarr.Array,
    *,
    chunk_rows_hint: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = getattr(masks_arr, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 4:
        raise RuntimeError(f"masks_roi must be (N,C,H,W), got shape={shape}")
    n_rows = int(shape[0])
    channels = int(shape[1])
    if channels < 1:
        raise RuntimeError("masks_roi has zero channels.")
    step = _chunk_rows(masks_arr, chunk_rows_hint=chunk_rows_hint)

    left = np.zeros((n_rows,), dtype=np.int64)
    right = np.zeros((n_rows,), dtype=np.int64)
    union = np.zeros((n_rows,), dtype=np.int64)

    for start in range(0, n_rows, step):
        stop = min(n_rows, start + step)
        block = np.asarray(masks_arr[start:stop], dtype=np.uint8)
        if block.ndim != 4:
            raise RuntimeError(f"masks_roi block must be rank-4, got shape={block.shape}")

        left_block = block[:, 0] > 0
        right_block = block[:, 1] > 0 if block.shape[1] > 1 else left_block.copy()
        union_block = np.logical_or(left_block, right_block)

        left[start:stop] = np.sum(left_block.reshape(stop - start, -1), axis=1, dtype=np.int64)
        right[start:stop] = np.sum(right_block.reshape(stop - start, -1), axis=1, dtype=np.int64)
        union[start:stop] = np.sum(union_block.reshape(stop - start, -1), axis=1, dtype=np.int64)

    return left, right, union


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_report(
    *,
    root: zarr.Group,
    dataset_id: Optional[str],
    zarr_path: Path,
    source_eye_mask_path: str,
    profile_run_hint: Optional[str],
    source_group: zarr.Group,
    left: np.ndarray,
    right: np.ndarray,
    union: np.ndarray,
    max_zero_preview: int,
    max_span_preview: int,
    inspect_keypoints: bool,
    keypoint_group_override: Optional[str],
    keypoint_run_override: Optional[str],
    max_keypoint_preview: int,
) -> dict[str, Any]:
    attrs = dict(source_group.attrs)
    zero_union = union <= 0
    zero_rows = np.flatnonzero(zero_union).astype(np.int64, copy=False)
    spans = _iter_zero_spans(zero_union)

    frame_values: Optional[np.ndarray] = None
    frame_indices_preview: list[int] = []
    frame_spans_preview: list[dict[str, int]] = []
    frame_idx_arr = source_group.get("frame_indices")
    if frame_idx_arr is not None:
        frame_values = np.asarray(frame_idx_arr[:], dtype=np.int64)
        if frame_values.ndim == 1 and frame_values.size == union.size:
            frame_indices_preview = frame_values[zero_rows[:max_zero_preview]].astype(int).tolist()
            for start, end in spans[:max_span_preview]:
                frame_spans_preview.append(
                    {
                        "start_row": int(start),
                        "end_row": int(end),
                        "length": int(end - start + 1),
                        "start_frame_idx": int(frame_values[start]),
                        "end_frame_idx": int(frame_values[end]),
                    }
                )

    ellipse_pair_success_rate = None
    ellipse_pair_success_count = None
    ellipse_success_arr = source_group.get("ellipse_success")
    if ellipse_success_arr is not None:
        ellipse_success = np.asarray(ellipse_success_arr[:], dtype=np.bool_)
        if ellipse_success.ndim == 1:
            pair_success = ellipse_success
        elif ellipse_success.ndim == 2:
            pair_success = np.all(ellipse_success, axis=1)
        else:
            pair_success = None
        if pair_success is not None:
            ellipse_pair_success_count = int(np.sum(pair_success))
            ellipse_pair_success_rate = float(np.mean(pair_success)) if pair_success.size else 0.0

    source_review = attrs.get("eye_mask_review_status")
    if isinstance(source_review, dict):
        review_state = _normalize_text(source_review.get("state"))
    else:
        review_state = None

    keypoint_summary: Optional[dict[str, Any]] = None
    if inspect_keypoints:
        keypoint_summary = _compute_zero_union_keypoint_summary(
            root=root,
            source_group=source_group,
            zero_rows=zero_rows,
            frame_values=frame_values,
            explicit_group=keypoint_group_override,
            explicit_run=keypoint_run_override,
            max_preview=max_keypoint_preview,
        )

    return {
        "dataset_id": dataset_id,
        "zarr_path": str(zarr_path),
        "source_eye_mask_path": source_eye_mask_path,
        "profile_run_hint": profile_run_hint,
        "rows_total": int(union.size),
        "channels": _to_int_or_none(getattr(source_group.get("masks_roi"), "shape", [None, None])[1]),
        "source_review_state": review_state,
        "attrs_successful_roi_pair_rate": _to_float_or_none(attrs.get("successful_roi_pair_rate")),
        "attrs_successful_roi_pairs": _to_int_or_none(attrs.get("successful_roi_pairs")),
        "ellipse_pair_success_rate": ellipse_pair_success_rate,
        "ellipse_pair_success_count": ellipse_pair_success_count,
        "left_area": asdict(_area_stats(left)),
        "right_area": asdict(_area_stats(right)),
        "union_area": asdict(_area_stats(union)),
        "zero_union_rows_total": int(zero_rows.size),
        "zero_union_rows_preview": zero_rows[:max_zero_preview].astype(int).tolist(),
        "zero_union_frame_indices_preview": frame_indices_preview,
        "zero_union_spans_preview": [
            {"start_row": int(start), "end_row": int(end), "length": int(end - start + 1)}
            for start, end in spans[:max_span_preview]
        ],
        "zero_union_frame_spans_preview": frame_spans_preview,
        "keypoint_summary": keypoint_summary,
    }


def _print_text_report(report: dict[str, Any]) -> None:
    print(f"Dataset ID: {report.get('dataset_id') or '-'}")
    print(f"Zarr Path: {report['zarr_path']}")
    print(f"Source Eye-Mask Path: {report['source_eye_mask_path']}")
    print(f"Profile Run Hint: {report.get('profile_run_hint') or '-'}")
    print(f"Rows Total: {report['rows_total']}")
    print(f"Channels: {report.get('channels')}")
    print(
        "Review/Success attrs: "
        f"state={report.get('source_review_state') or '-'} "
        f"attrs_successful_roi_pair_rate={report.get('attrs_successful_roi_pair_rate')} "
        f"ellipse_pair_success_rate={report.get('ellipse_pair_success_rate')}"
    )
    print("Area stats (p50 is median):")
    for key in ("left_area", "right_area", "union_area"):
        stats = report[key]
        print(
            f"  {key}: "
            f"mean={stats['mean']:.2f} p10={stats['p10']:.1f} p50={stats['p50']:.1f} p90={stats['p90']:.1f} "
            f"zeros={stats['zeros']}/{stats['count']} ({stats['zero_rate']:.1%})"
        )

    print(
        "Zero-union rows: "
        f"{report['zero_union_rows_total']}/{report['rows_total']} "
        f"({(report['zero_union_rows_total'] / max(1, report['rows_total'])):.1%})"
    )
    row_preview = report["zero_union_rows_preview"]
    if row_preview:
        print("  Zero-union row preview:", ",".join(str(v) for v in row_preview))
    frame_preview = report["zero_union_frame_indices_preview"]
    if frame_preview:
        print("  Zero-union frame preview:", ",".join(str(v) for v in frame_preview))

    spans = report["zero_union_frame_spans_preview"] or report["zero_union_spans_preview"]
    if spans:
        print("  Zero-union span preview:")
        for span in spans:
            if "start_frame_idx" in span:
                print(
                    "    "
                    f"rows {span['start_row']}..{span['end_row']} len={span['length']} "
                    f"frames {span['start_frame_idx']}..{span['end_frame_idx']}"
                )
            else:
                print(
                    "    "
                    f"rows {span['start_row']}..{span['end_row']} len={span['length']}"
                )

    keypoint_summary = report.get("keypoint_summary")
    if not isinstance(keypoint_summary, dict):
        return
    print("Keypoint summary on zero-union rows:")
    status = _normalize_text(keypoint_summary.get("status")) or "unknown"
    print(f"  status={status}")
    if status != "ok":
        print(f"  reason={keypoint_summary.get('reason') or '-'}")
        return
    print(
        "  "
        f"source={keypoint_summary.get('keypoint_group')}/{keypoint_summary.get('keypoint_run')} "
        f"success_dataset={keypoint_summary.get('success_dataset')}"
    )
    print(
        "  "
        f"keypoint_valid_rate={keypoint_summary.get('keypoint_valid_rate'):.3f} "
        f"keypoint_success_rate={keypoint_summary.get('keypoint_success_rate'):.3f}"
    )
    zero_total = int(keypoint_summary.get("zero_union_rows") or 0)
    zero_success = int(keypoint_summary.get("zero_union_with_keypoint_success_rows") or 0)
    zero_valid = int(keypoint_summary.get("zero_union_with_keypoint_valid_rows") or 0)
    print(
        "  "
        f"zero_union_with_success={zero_success}/{zero_total} "
        f"zero_union_with_valid_keypoints={zero_valid}/{zero_total}"
    )
    print(
        "  "
        f"zero_union_success_but_invalid={int(keypoint_summary.get('zero_union_keypoint_success_but_invalid_rows') or 0)} "
        f"zero_union_nonfinite_eye_keypoints={int(keypoint_summary.get('zero_union_nonfinite_eye_keypoints_rows') or 0)} "
        f"zero_union_nondistinct_eye_keypoints={int(keypoint_summary.get('zero_union_nondistinct_eye_keypoints_rows') or 0)}"
    )
    preview = keypoint_summary.get("zero_union_keypoint_preview")
    if isinstance(preview, list) and preview:
        print("  zero-union keypoint preview:")
        for row in preview:
            frame = f" frame={row.get('frame_idx')}" if row.get("frame_idx") is not None else ""
            print(
                "    "
                f"roi={row.get('roi_idx')}{frame} "
                f"success={row.get('keypoint_success')} valid={row.get('keypoint_valid')} "
                f"left_finite={row.get('left_finite')} right_finite={row.get('right_finite')} "
                f"distinct={row.get('eyes_distinct')}"
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect eye-mask source run areas and zero-area spans from masks_roi. "
            "Supports dataset-id+registry lookup or direct zarr/source path inputs."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--dataset-id", type=str, help="Registry dataset_id to inspect.")
    target_group.add_argument("--zarr-path", type=Path, help="Direct zarr path to inspect.")
    parser.add_argument("--registry", type=Path, help="Registry path (required for --dataset-id).")

    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-eye-mask-path",
        type=str,
        help="Explicit source path (e.g. refined_eye_masks_runs/<run> or eye_masks_runs/<run>).",
    )
    source_group.add_argument("--refined-run", type=str, help="Refined eye-mask run name.")
    source_group.add_argument("--eye-run", type=str, help="Traditional eye-mask run name.")
    parser.add_argument(
        "--inspect-keypoints",
        action="store_true",
        help="Inspect keypoint validity specifically on zero-union eye-mask rows.",
    )
    parser.add_argument(
        "--keypoint-group",
        choices=KEYPOINT_GROUP_CHOICES,
        help="Optional keypoint group override for --inspect-keypoints.",
    )
    parser.add_argument(
        "--keypoint-run",
        type=str,
        help="Optional keypoint run override for --inspect-keypoints.",
    )

    parser.add_argument("--chunk-rows", type=int, default=2048, help="Chunk rows for mask scanning.")
    parser.add_argument("--max-zero-preview", type=int, default=30, help="Max zero-area row indices to print.")
    parser.add_argument("--max-span-preview", type=int, default=12, help="Max zero-area spans to print.")
    parser.add_argument(
        "--max-keypoint-preview",
        type=int,
        default=15,
        help="Max zero-union rows to include in keypoint preview payload.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    dataset_id: Optional[str] = None
    zarr_path: Path
    profile_source_eye_mask_path: Optional[str] = None
    profile_run_hint: Optional[str] = None

    if args.dataset_id is not None:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        lookup = _lookup_dataset(registry_path, dataset_id=str(args.dataset_id))
        dataset_id = lookup.dataset_id
        zarr_path = lookup.zarr_path
        profile_source_eye_mask_path = lookup.profile_source_eye_mask_path
        profile_run_hint = lookup.profile_run
    else:
        zarr_path = Path(args.zarr_path)

    zarr_path = zarr_path.expanduser().resolve(strict=False)
    if not zarr_path.exists():
        raise SystemExit(f"zarr path does not exist: {zarr_path}")

    root = _open_root(zarr_path)
    try:
        source_eye_mask_path = _resolve_source_eye_mask_path(
            root,
            source_eye_mask_path=args.source_eye_mask_path,
            refined_run=args.refined_run,
            eye_run=args.eye_run,
            profile_source_eye_mask_path=profile_source_eye_mask_path,
        )
    except EyeMaskSourceError as exc:
        raise SystemExit(str(exc)) from exc

    source_group = _load_source_group(root, source_eye_mask_path=source_eye_mask_path)
    masks_arr = source_group.get("masks_roi")
    if masks_arr is None:
        raise SystemExit(f"{source_eye_mask_path} missing masks_roi")

    left, right, union = _compute_area_vectors(
        masks_arr,
        chunk_rows_hint=max(1, int(args.chunk_rows)),
    )
    report = _build_report(
        root=root,
        dataset_id=dataset_id,
        zarr_path=zarr_path,
        source_eye_mask_path=source_eye_mask_path,
        profile_run_hint=profile_run_hint,
        source_group=source_group,
        left=left,
        right=right,
        union=union,
        max_zero_preview=max(1, int(args.max_zero_preview)),
        max_span_preview=max(1, int(args.max_span_preview)),
        inspect_keypoints=bool(args.inspect_keypoints),
        keypoint_group_override=args.keypoint_group,
        keypoint_run_override=args.keypoint_run,
        max_keypoint_preview=max(1, int(args.max_keypoint_preview)),
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_text_report(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Parallel, keyed detection-quality reconciliation for recording-level runs.

Workers read complete row shards from one recording-ordered detection surface
and write compact traces to job-local storage.  The finalizer deterministically
reconciles temporal state across every shard/clip boundary, writes the three
canonical quality arrays once, validates their identity and storage contracts,
and only then promotes the completed run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr

from .detect_quality import (
    _effective_jump_threshold_pixels,
    calculate_quality_score,
    calculate_sampled_quality_score,
)
from .utils import calculate_coverage_stats, categorize_gaps, identify_gaps
from ..shared.detect_quality_contract import (
    CLIPPED_DETECT_QUALITY_SOURCE_SCHEMA,
    FULL_FRAME_GEOMETRY_SCHEMA,
)
from ..shared.detection_tables import resolve_detection_instance_table
from ..shared.experiment_setup import resolve_expected_subject_count
from ..shared.run_provenance import build_writer_run_provenance
from ..shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID,
    canonical_detection_dimensions_from_manifest,
)
from ..shared.zarr.detection_schema import CanonicalDetectionDimensions
from ..shared.zarr_io import open_zarr_root
from ..shared.zarr_run_completion import (
    RUN_PROVENANCE_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)


COLLECTION_QUALITY_SCHEMA = "palette.detect_quality_collection.v2"
TEMPORAL_POLICY_SCHEMA = "palette.detect_quality.temporal_reacquisition.v2"
LABEL_SCHEMA = "palette.detect_quality_labels.v1"
TRACE_SCHEMA = "palette.detect_quality_shard_trace.v1"

NO_DETECTION = np.int8(-1)
CLEAN = np.int8(0)
BLIP = np.int8(2)
JUMP = np.int8(3)
OVER_EXPECTED = np.int8(4)

DEFAULT_SHARD_ROWS = 131_072
DEFAULT_ROW_CHUNK_ROWS = 16_384
DEFAULT_FRAME_CHUNK_ROWS = 16_384
DEFAULT_RELOCATION_CONFIRM_COUNT = 3
DEFAULT_RELOCATION_CLUSTER_RADIUS_FRACTION = 0.5


@dataclass(frozen=True)
class ShardTask:
    zarr_path: str
    source_group_path: str
    start: int
    stop: int
    width: float
    height: float
    output_path: str
    params_hash: str


@dataclass(frozen=True)
class ShardTrace:
    schema: str
    start: int
    stop: int
    rows: int
    path: str
    params_hash: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_group_path(value: str) -> str:
    normalized = str(value or "").strip().strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Unsafe or empty Zarr group path: {value!r}.")
    return normalized


def _group_at(root: Any, group_path: str) -> Any:
    group = root
    for part in _safe_group_path(group_path).split("/"):
        group = group[part]
    return group


def _positive_int(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _mapping_int(value: object, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    return _positive_int(value.get(key))


def _geometry_pair(
    attrs: Any,
    width_key: str,
    height_key: str,
    *,
    scope: str,
) -> tuple[float, float, str] | None:
    raw_width = attrs.get(width_key)
    raw_height = attrs.get(height_key)
    if raw_width is None and raw_height is None:
        return None
    width = _positive_int(raw_width)
    height = _positive_int(raw_height)
    if width is None or height is None:
        raise ValueError(
            f"{scope} full-frame geometry is incomplete or non-positive: "
            f"{width_key}={raw_width!r}, {height_key}={raw_height!r}."
        )
    return float(width), float(height), f"{scope}:{width_key}/{height_key}"


def _geometry_from_attrs(attrs: Any, *, scope: str) -> tuple[float, float, str] | None:
    candidates: list[tuple[float, float, str]] = []
    for width_key, height_key in (
        ("source_video_width", "source_video_height"),
        ("source_full_width", "source_full_height"),
        ("video_width", "video_height"),
        ("width", "height"),
    ):
        candidate = _geometry_pair(
            attrs,
            width_key,
            height_key,
            scope=scope,
        )
        if candidate is not None:
            candidates.append(candidate)
    metadata = attrs.get("source_video_metadata")
    if isinstance(metadata, Mapping):
        candidate = _geometry_pair(
            metadata,
            "width",
            "height",
            scope=f"{scope}.source_video_metadata",
        )
        if candidate is not None:
            candidates.append(candidate)
    distinct = {(width, height) for width, height, _ in candidates}
    if len(distinct) > 1:
        details = ", ".join(
            f"{source}={int(width)}x{int(height)}" for width, height, source in candidates
        )
        raise ValueError(f"Conflicting full-frame geometry in {scope}: {details}.")
    return candidates[0] if candidates else None


def _ancestor_groups(root: Any, group_path: str) -> Iterable[tuple[str, Any]]:
    parts = _safe_group_path(group_path).split("/")
    for stop in range(len(parts), 0, -1):
        path = "/".join(parts[:stop])
        yield path, _group_at(root, path)


def _canonical_source_dimensions(
    source: Any,
    *,
    row_count: int,
) -> CanonicalDetectionDimensions | None:
    manifest = source.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        return None
    if manifest.get("schema_id") != CANONICAL_DETECTION_RUN_MANIFEST_SCHEMA_ID:
        return None
    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    if int(dimensions.n_instances) != int(row_count):
        raise ValueError(
            "Canonical detection manifest instance count differs from its "
            "persisted table."
        )
    return dimensions


def _resolve_frame_count(
    root: Any,
    source: Any,
    source_group_path: str,
    explicit: int | None,
    canonical_dimensions: CanonicalDetectionDimensions | None,
) -> int:
    if canonical_dimensions is not None:
        canonical = int(canonical_dimensions.n_frames)
        if explicit is not None and _positive_int(explicit) != canonical:
            raise ValueError(
                "Explicit recording frame count conflicts with the canonical "
                "detection manifest."
            )
        return canonical
    if explicit is not None:
        resolved = _positive_int(explicit)
        if resolved is None:
            raise ValueError("recording_frame_count must be positive.")
        return resolved
    candidates: list[int | None] = [
        _positive_int(source.attrs.get("recording_frame_count")),
        _positive_int(source.attrs.get("total_frames")),
    ]
    for _, group in _ancestor_groups(root, source_group_path):
        candidates.extend(
            [
                _positive_int(group.attrs.get("recording_frame_count")),
                _positive_int(group.attrs.get("total_frames")),
                _mapping_int(group.attrs.get("summary_statistics"), "recording_frames"),
                _mapping_int(group.attrs.get("coverage_comparison"), "recording_frame_count"),
            ]
        )
    raw = root.get("raw_video")
    if raw is not None:
        for name in ("images_full", "images_ds"):
            if name in raw:
                candidates.append(_positive_int(raw[name].shape[0]))
    candidates.append(_positive_int(root.attrs.get("total_frames")))
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise ValueError(
        "Could not resolve the complete recording frame count. Provide "
        "--recording-frame-count or stamp it on the source/ancestor run."
    )


def _resolve_geometry(
    root: Any,
    source: Any,
    *,
    explicit_width: float | None,
    explicit_height: float | None,
    canonical_dimensions: CanonicalDetectionDimensions | None,
) -> tuple[float, float, str]:
    if (explicit_width is None) != (explicit_height is None):
        raise ValueError("Explicit full-frame width and height must be provided together.")

    if canonical_dimensions is not None:
        width = float(canonical_dimensions.source_width)
        height = float(canonical_dimensions.source_height)
        if explicit_width is not None and explicit_height is not None and (
            float(explicit_width), float(explicit_height)
        ) != (width, height):
            raise ValueError(
                "Explicit full-frame geometry conflicts with the canonical "
                "detection manifest."
            )
        return width, height, "source:canonical_run_manifest"

    source_geometry = _geometry_from_attrs(source.attrs, scope="source")
    if source.attrs.get("schema_id") == CLIPPED_DETECT_QUALITY_SOURCE_SCHEMA:
        canonical_source_geometry = _geometry_pair(
            source.attrs,
            "source_video_width",
            "source_video_height",
            scope="source",
        )
        if canonical_source_geometry is None:
            raise ValueError(
                f"{CLIPPED_DETECT_QUALITY_SOURCE_SCHEMA} requires source_video_width and "
                "source_video_height attrs."
            )

    root_geometry = _geometry_from_attrs(root.attrs, scope="root")
    raw = root.get("raw_video")
    raw_geometry = (
        _geometry_from_attrs(raw.attrs, scope="raw_video") if raw is not None else None
    )
    metadata_geometries = [
        candidate
        for candidate in (source_geometry, root_geometry, raw_geometry)
        if candidate is not None
    ]
    distinct_metadata_geometry = {
        (candidate[0], candidate[1]) for candidate in metadata_geometries
    }
    if len(distinct_metadata_geometry) > 1:
        details = ", ".join(
            f"{metadata_source}={int(metadata_width)}x{int(metadata_height)}"
            for metadata_width, metadata_height, metadata_source in metadata_geometries
        )
        raise ValueError(f"Conflicting full-frame geometry across Zarr metadata: {details}.")

    if explicit_width is not None and explicit_height is not None:
        width = float(explicit_width)
        height = float(explicit_height)
        if width <= 0 or height <= 0:
            raise ValueError("Explicit full-frame width and height must be positive.")
        conflicts = [
            candidate
            for candidate in metadata_geometries
            if (candidate[0], candidate[1]) != (width, height)
        ]
        if conflicts:
            details = ", ".join(
                f"{metadata_source}={int(metadata_width)}x{int(metadata_height)}"
                for metadata_width, metadata_height, metadata_source in conflicts
            )
            raise ValueError(
                f"Explicit full-frame geometry {int(width)}x{int(height)} conflicts with "
                f"canonical metadata: {details}."
            )
        return width, height, "explicit_cli"

    if source_geometry is not None:
        return source_geometry
    if root_geometry is not None:
        return root_geometry
    if raw_geometry is not None:
        return raw_geometry
    if raw is not None:
        for name in ("images_full", "images_ds"):
            if name in raw:
                height = float(raw[name].shape[1])
                width = float(raw[name].shape[2])
                if width > 0 and height > 0:
                    return width, height, f"raw_video/{name}:array_shape"
    raise ValueError(
        "Full-frame width and height are required for temporal quality. "
        "Stamp source_video_width/source_video_height on the source group or provide "
        "--width/--height when canonical metadata is unavailable."
    )


def _resolve_source_identity(root: Any, source_group_path: str) -> dict[str, str]:
    for path, group in _ancestor_groups(root, source_group_path):
        collection = str(group.attrs.get("source_detect_collection_id") or "").strip()
        if collection:
            return {
                "kind": "collection",
                "id": collection,
                "path": str(group.attrs.get("source_detect_collection_path") or path),
            }
    parts = _safe_group_path(source_group_path).split("/")
    if len(parts) >= 2 and parts[0] == "detect_runs":
        return {"kind": "run", "id": parts[1], "path": "/".join(parts[:2])}
    for path, group in _ancestor_groups(root, source_group_path):
        run = str(group.attrs.get("source_detect_run") or "").strip()
        if run:
            return {"kind": "run", "id": run, "path": path}
    return {"kind": "group", "id": source_group_path, "path": source_group_path}


def _effective_shard_rows(requested: int, inner: int) -> int:
    if requested <= 0 or inner <= 0:
        raise ValueError("Shard and inner chunk rows must be positive.")
    return int(math.ceil(requested / inner) * inner)


def _source_shard_rows(source: Any, requested: int) -> tuple[int, tuple[int, ...]]:
    values: set[int] = set()
    for name in ("frame_indices", "bbox_norm_coords", "instance_key"):
        shards = getattr(source[name], "shards", None)
        if shards is not None:
            values.add(int(shards[0]))
    grids = tuple(sorted(values))
    return (next(iter(values)) if len(values) == 1 else int(requested), grids)


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _worker(task: ShardTask) -> ShardTrace:
    root = zarr.open_group(task.zarr_path, mode="r", use_consolidated=False)
    source = resolve_detection_instance_table(
        _group_at(root, task.source_group_path)
    )
    frames = np.asarray(source["frame_indices"][task.start : task.stop], dtype=np.int64)
    boxes = np.asarray(source["bbox_norm_coords"][task.start : task.stop])
    keys = np.asarray(source["instance_key"][task.start : task.stop], dtype=np.uint64)
    rows = task.stop - task.start
    if frames.shape != (rows,) or keys.shape != (rows,) or boxes.shape != (rows, 4):
        raise ValueError(
            f"Source shard {task.start}:{task.stop} has incompatible shapes: "
            f"frames={frames.shape}, boxes={boxes.shape}, keys={keys.shape}."
        )
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError(f"Source frames are not ordered in shard {task.start}:{task.stop}.")
    boxes64 = boxes.astype(np.float64, copy=False)
    centroids = np.column_stack(
        (boxes64[:, 0] * float(task.width), boxes64[:, 1] * float(task.height))
    ).astype(np.float32, copy=False)
    out_of_range = np.any(
        (boxes64[:, :2] < 0)
        | (boxes64[:, :2] > 1)
        | (boxes64[:, 2:] <= 0)
        | (boxes64[:, 2:] > 1),
        axis=1,
    )
    malformed = np.any(boxes64[:, 2:] <= 0, axis=1)
    bbox_sizes = np.sqrt(boxes64[:, 2] ** 2 + boxes64[:, 3] ** 2).astype(
        np.float32,
        copy=False,
    )
    output = Path(task.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            frame_indices=frames,
            instance_key=keys,
            centroids_px=centroids,
            bbox_sizes=bbox_sizes,
            bbox_out_of_range=out_of_range,
            bbox_malformed=malformed,
        )
    os.replace(temporary, output)
    return ShardTrace(
        schema=TRACE_SCHEMA,
        start=task.start,
        stop=task.stop,
        rows=rows,
        path=str(output),
        params_hash=task.params_hash,
    )


def _flush_candidates(labels: np.ndarray, candidates: list[int]) -> None:
    if candidates:
        labels[np.asarray(candidates, dtype=np.int64)] = JUMP
        candidates.clear()


def reconcile_temporal_v2(
    frame_indices: np.ndarray,
    centroids_px: np.ndarray,
    *,
    jump_threshold_pixels: float,
    blip_gap_threshold: int,
    relocation_confirm_count: int = DEFAULT_RELOCATION_CONFIRM_COUNT,
    relocation_cluster_radius_pixels: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Classify an ordered single-subject trace with bounded reacquisition.

    A long gap produces a blip and resets the baseline.  A displacement beyond
    the jump threshold starts a provisional relocation.  Isolated provisional
    rows remain jumps, while a stable cluster of ``relocation_confirm_count``
    observations is accepted and relabeled clean.
    """

    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    centroids = np.asarray(centroids_px, dtype=np.float32)
    if centroids.shape != (frames.shape[0], 2):
        raise ValueError("centroids_px must have shape (N, 2).")
    if frames.size > 1 and np.any(np.diff(frames) <= 0):
        raise ValueError("Temporal representatives must have strictly increasing frames.")
    if relocation_confirm_count < 2:
        raise ValueError("relocation_confirm_count must be at least 2.")
    if jump_threshold_pixels <= 0 or relocation_cluster_radius_pixels <= 0:
        raise ValueError("Temporal distance thresholds must be positive.")
    if blip_gap_threshold <= 0:
        raise ValueError("blip_gap_threshold must be positive.")

    labels = np.full(frames.shape[0], CLEAN, dtype=np.int8)
    baseline: np.ndarray | None = None
    previous_frame: int | None = None
    candidate_anchor: np.ndarray | None = None
    candidates: list[int] = []
    accepted_relocations = 0

    for index, (frame, position) in enumerate(zip(frames, centroids, strict=True)):
        frame_value = int(frame)
        finite = bool(np.all(np.isfinite(position)))
        gap = frame_value - previous_frame if previous_frame is not None else None
        if gap is not None and gap >= blip_gap_threshold:
            _flush_candidates(labels, candidates)
            candidate_anchor = None
            if finite:
                labels[index] = BLIP
                baseline = position.copy()
            else:
                labels[index] = JUMP
            previous_frame = frame_value
            continue
        if not finite:
            _flush_candidates(labels, candidates)
            candidate_anchor = None
            labels[index] = JUMP
            previous_frame = frame_value
            continue
        if baseline is None:
            baseline = position.copy()
            previous_frame = frame_value
            continue

        distance_from_baseline = float(np.linalg.norm(position - baseline))
        if distance_from_baseline <= jump_threshold_pixels:
            _flush_candidates(labels, candidates)
            candidate_anchor = None
            baseline = position.copy()
            previous_frame = frame_value
            continue

        if candidate_anchor is None:
            candidate_anchor = position.copy()
            candidates.append(index)
        elif float(np.linalg.norm(position - candidate_anchor)) <= relocation_cluster_radius_pixels:
            candidates.append(index)
        else:
            _flush_candidates(labels, candidates)
            candidate_anchor = position.copy()
            candidates.append(index)

        if len(candidates) >= relocation_confirm_count:
            labels[np.asarray(candidates, dtype=np.int64)] = CLEAN
            baseline = position.copy()
            candidates.clear()
            candidate_anchor = None
            accepted_relocations += 1
        previous_frame = frame_value

    _flush_candidates(labels, candidates)
    return labels, {
        "jump_frames": frames[labels == JUMP].astype(np.int64, copy=False),
        "blip_frames": frames[labels == BLIP].astype(np.int64, copy=False),
        "accepted_relocations": int(accepted_relocations),
    }


def _load_traces(
    traces: Sequence[ShardTrace],
    *,
    expected_rows: int,
    params_hash: str,
) -> dict[str, np.ndarray]:
    ordered = sorted(traces, key=lambda item: item.start)
    cursor = 0
    arrays: dict[str, list[np.ndarray]] = {
        "frame_indices": [],
        "instance_key": [],
        "centroids_px": [],
        "bbox_sizes": [],
        "bbox_out_of_range": [],
        "bbox_malformed": [],
    }
    for trace in ordered:
        if trace.schema != TRACE_SCHEMA or trace.params_hash != params_hash:
            raise ValueError("Mixed trace schema or quality parameter hashes.")
        if trace.start != cursor or trace.stop - trace.start != trace.rows:
            raise ValueError(
                f"Trace ranges are incomplete or overlapping at {cursor}: {asdict(trace)}."
            )
        with np.load(trace.path, allow_pickle=False) as payload:
            for name in arrays:
                value = np.asarray(payload[name])
                if value.shape[0] != trace.rows:
                    raise ValueError(f"Trace {trace.path} has wrong {name} row count.")
                arrays[name].append(value)
        cursor = trace.stop
    if cursor != expected_rows:
        raise ValueError(f"Trace rows end at {cursor}, expected {expected_rows}.")
    empty_values = {
        "frame_indices": np.empty((0,), dtype=np.int64),
        "instance_key": np.empty((0,), dtype=np.uint64),
        "centroids_px": np.empty((0, 2), dtype=np.float32),
        "bbox_sizes": np.empty((0,), dtype=np.float32),
        "bbox_out_of_range": np.empty((0,), dtype=bool),
        "bbox_malformed": np.empty((0,), dtype=bool),
    }
    return {
        name: np.concatenate(parts, axis=0) if parts else empty_values[name]
        for name, parts in arrays.items()
    }


def _gap_summary(presence: np.ndarray) -> dict[str, Any]:
    gaps = identify_gaps(presence)
    sizes = [int(gap.size) for gap in gaps]
    return {
        "total_count": len(gaps),
        "categories": categorize_gaps(gaps),
        "longest_gap": max(sizes) if sizes else 0,
        "mean_gap_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_gap_size": float(np.median(sizes)) if sizes else 0.0,
    }


def _aggregate_quality(
    trace: dict[str, np.ndarray],
    *,
    frame_count: int,
    expected_subject_count: int | None,
    jump_threshold_pixels: float,
    blip_gap_threshold: int,
    relocation_confirm_count: int,
    relocation_cluster_radius_pixels: float,
) -> dict[str, Any]:
    frames = np.asarray(trace["frame_indices"], dtype=np.int64).reshape(-1)
    keys = np.asarray(trace["instance_key"], dtype=np.uint64).reshape(-1)
    if frames.size > 1 and np.any(np.diff(frames) < 0):
        raise ValueError("Source detection rows are not in canonical frame order.")
    if frames.size and (int(frames[0]) < 0 or int(frames[-1]) >= frame_count):
        raise ValueError("Source detection frame indices fall outside the recording timeline.")
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise ValueError("Modern source detection instance_key values are not unique.")

    frame_counts = np.bincount(frames, minlength=frame_count).astype(np.int32, copy=False)
    if frame_counts.shape != (frame_count,):
        raise ValueError("Could not establish complete recording frame coverage.")
    quality_flags = np.full(frame_count, NO_DETECTION, dtype=np.int8)
    quality_flags[frame_counts > 0] = CLEAN

    temporal_policy = "skipped_expected_subject_count_gt_1"
    temporal_summary: dict[str, Any] = {
        "jump_frames": np.empty((0,), dtype=np.int64),
        "blip_frames": np.empty((0,), dtype=np.int64),
        "accepted_relocations": 0,
    }
    if expected_subject_count is None or expected_subject_count == 1:
        unique_frames, first_rows = np.unique(frames, return_index=True)
        single = frame_counts[unique_frames] == 1
        representative_frames = unique_frames[single]
        representative_centroids = np.asarray(trace["centroids_px"])[first_rows[single]]
        temporal_labels, temporal_summary = reconcile_temporal_v2(
            representative_frames,
            representative_centroids,
            jump_threshold_pixels=jump_threshold_pixels,
            blip_gap_threshold=blip_gap_threshold,
            relocation_confirm_count=relocation_confirm_count,
            relocation_cluster_radius_pixels=relocation_cluster_radius_pixels,
        )
        quality_flags[representative_frames] = temporal_labels
        temporal_policy = TEMPORAL_POLICY_SCHEMA

    expected_threshold = expected_subject_count or 1
    over_expected_frames = frame_counts > expected_threshold
    quality_flags[over_expected_frames] = OVER_EXPECTED
    detection_labels = quality_flags[frames]

    _, first_rows = np.unique(frames, return_index=True)
    sizes = np.asarray(trace["bbox_sizes"], dtype=np.float64)[first_rows]
    out_of_range = np.asarray(trace["bbox_out_of_range"], dtype=bool)[first_rows]
    malformed = np.asarray(trace["bbox_malformed"], dtype=bool)[first_rows]
    mean_size = float(np.mean(sizes)) if sizes.size else 0.0
    std_size = float(np.std(sizes)) if sizes.size else 0.0
    size_outliers = int(np.sum(np.abs(sizes - mean_size) > 3.0 * std_size)) if sizes.size else 0
    bbox_validation = {
        "total_bboxes": int(sizes.size),
        "out_of_range": int(np.sum(out_of_range)),
        "size_outliers": size_outliers,
        "malformed": int(np.sum(malformed)),
        "mean_size": mean_size,
        "std_size": std_size,
        "size_cv": float(std_size / mean_size) if mean_size > 0 else 0.0,
    }

    presence = frame_counts > 0
    coverage = calculate_coverage_stats(presence)
    coverage["multi_detection_frames"] = int(np.sum(over_expected_frames))
    coverage["gaps"] = _gap_summary(presence)
    if expected_subject_count is not None:
        coverage.update(
            {
                "expected_count": int(expected_subject_count),
                "frames_with_expected_count": int(
                    np.sum(frame_counts == expected_subject_count)
                ),
                "frames_under_expected": int(
                    np.sum((frame_counts > 0) & (frame_counts < expected_subject_count))
                ),
                "frames_over_expected": int(np.sum(over_expected_frames)),
            }
        )
    artifacts = {
        "total_artifacts": int(
            np.sum(quality_flags == JUMP) + np.sum(quality_flags == BLIP)
        ),
        "jump_frames": int(np.sum(quality_flags == JUMP)),
        "blip_frames": int(np.sum(quality_flags == BLIP)),
        "accepted_relocations": int(temporal_summary["accepted_relocations"]),
        "jump_threshold_pixels_effective": float(jump_threshold_pixels),
        "blip_gap_threshold": int(blip_gap_threshold),
        "temporal_artifact_policy": temporal_policy,
    }
    if expected_subject_count is not None and expected_subject_count > 1:
        quality_score = calculate_sampled_quality_score(
            coverage,
            frame_counts,
            expected_subject_count,
            bbox_validation,
            mode="expected_count",
        )
    else:
        quality_score = calculate_quality_score(coverage, artifacts, bbox_validation)
    summary = {
        "total_frames": int(frame_count),
        "empty_frames": int(np.sum(quality_flags == NO_DETECTION)),
        "frames_with_detections": int(np.sum(frame_counts > 0)),
        "clean_frames": int(np.sum(quality_flags == CLEAN)),
        "total_detections": int(frames.size),
        "clean_detections": int(np.sum(detection_labels == CLEAN)),
        "blip_detections": int(np.sum(detection_labels == BLIP)),
        "jump_detections": int(np.sum(detection_labels == JUMP)),
        "multi_detections": int(np.sum(detection_labels == OVER_EXPECTED)),
        "expected_subject_count": expected_subject_count,
        "frames_over_expected": int(np.sum(over_expected_frames)),
        "clean_percentage": (
            float(np.mean(detection_labels == CLEAN) * 100.0)
            if detection_labels.size
            else 0.0
        ),
    }
    return {
        "quality_flags": quality_flags,
        "detection_quality_labels": detection_labels,
        "instance_key": keys,
        "frame_counts": frame_counts,
        "coverage": coverage,
        "bbox_validation": bbox_validation,
        "artifacts": artifacts,
        "quality_score": quality_score,
        "summary": summary,
    }


def _write_shardwise(array: Any, values: np.ndarray) -> None:
    outer = int(array.shards[0])
    for start in range(0, int(values.shape[0]), outer):
        stop = min(start + outer, int(values.shape[0]))
        array[start:stop] = values[start:stop]


def _digest_array(
    array: Any,
    *,
    block_rows: int = DEFAULT_SHARD_ROWS,
    dtype: np.dtype[Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), block_rows):
        values = np.asarray(array[start : min(start + block_rows, int(array.shape[0]))])
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _validate_source(source: Any) -> int:
    required = ("frame_indices", "bbox_norm_coords", "instance_key")
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(f"Modern detection source is missing required arrays: {missing}.")
    rows = int(source["frame_indices"].shape[0])
    if tuple(source["bbox_norm_coords"].shape) != (rows, 4):
        raise ValueError("bbox_norm_coords must have shape (N, 4).")
    if tuple(source["instance_key"].shape) != (rows,):
        raise ValueError("instance_key must have shape (N,).")
    if np.dtype(source["instance_key"].dtype) != np.dtype(np.uint64):
        raise ValueError("Modern detection source instance_key must be uint64.")
    return rows


def run_collection_detect_quality(
    *,
    zarr_path: str | Path,
    source_group_path: str,
    output_run: str,
    recording_frame_count: int | None = None,
    width: float | None = None,
    height: float | None = None,
    expected_subject_count: int | None = None,
    jump_threshold: float = 100.0,
    threshold_mode: str = "scaled",
    threshold_reference_width: float = 640.0,
    blip_gap_threshold: int = 10,
    relocation_confirm_count: int = DEFAULT_RELOCATION_CONFIRM_COUNT,
    relocation_cluster_radius_fraction: float = DEFAULT_RELOCATION_CLUSTER_RADIUS_FRACTION,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    row_chunk_rows: int = DEFAULT_ROW_CHUNK_ROWS,
    frame_chunk_rows: int = DEFAULT_FRAME_CHUNK_ROWS,
    workers: int = 1,
    work_dir: str | Path | None = None,
    apply: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    source_group_path = _safe_group_path(source_group_path)
    if not output_run or "/" in output_run or output_run in {".", ".."}:
        raise ValueError("output_run must be a safe single group name.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if relocation_confirm_count < 2:
        raise ValueError("relocation_confirm_count must be at least 2.")
    if blip_gap_threshold <= 0:
        raise ValueError("blip_gap_threshold must be positive.")
    if relocation_cluster_radius_fraction <= 0:
        raise ValueError("relocation_cluster_radius_fraction must be positive.")
    root = open_zarr_root(archive, mode="r")
    expected_subject_count, experiment_setup = resolve_expected_subject_count(
        root,
        expected_subject_count,
        allow_legacy=True,
    )
    source_group = _group_at(root, source_group_path)
    source = resolve_detection_instance_table(source_group)
    row_count = _validate_source(source)
    canonical_dimensions = _canonical_source_dimensions(
        source_group,
        row_count=row_count,
    )
    resolved_frame_count = _resolve_frame_count(
        root,
        source_group,
        source_group_path,
        recording_frame_count,
        canonical_dimensions,
    )
    resolved_width, resolved_height, geometry_source = _resolve_geometry(
        root,
        source_group,
        explicit_width=width,
        explicit_height=height,
        canonical_dimensions=canonical_dimensions,
    )
    effective_jump = _effective_jump_threshold_pixels(
        jump_threshold,
        threshold_mode,
        resolved_width,
        resolved_height,
        threshold_reference_width,
    )
    cluster_radius = effective_jump * float(relocation_cluster_radius_fraction)
    effective_row_shard = _effective_shard_rows(int(shard_rows), int(row_chunk_rows))
    effective_frame_shard = _effective_shard_rows(int(shard_rows), int(frame_chunk_rows))
    task_rows, source_row_shard_grids = _source_shard_rows(
        source, effective_row_shard
    )
    ranges = [
        (start, min(start + task_rows, row_count))
        for start in range(0, row_count, task_rows)
    ]
    source_identity = _resolve_source_identity(root, source_group_path)
    params = {
        "schema": COLLECTION_QUALITY_SCHEMA,
        "temporal_policy": TEMPORAL_POLICY_SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "source_group_path": source_group_path,
        "source_identity": source_identity,
        "recording_frame_count": resolved_frame_count,
        "width": resolved_width,
        "height": resolved_height,
        "full_frame_geometry_schema": FULL_FRAME_GEOMETRY_SCHEMA,
        "full_frame_geometry_source": geometry_source,
        "expected_subject_count": expected_subject_count,
        "experiment_setup_path": experiment_setup.group_path,
        "experiment_setup_sha256": experiment_setup.record_sha256,
        "experiment_setup_legacy": experiment_setup.legacy,
        "jump_threshold": float(jump_threshold),
        "threshold_mode": threshold_mode,
        "threshold_reference_width": float(threshold_reference_width),
        "jump_threshold_pixels_effective": float(effective_jump),
        "blip_gap_threshold": int(blip_gap_threshold),
        "relocation_confirm_count": int(relocation_confirm_count),
        "relocation_cluster_radius_fraction": float(relocation_cluster_radius_fraction),
        "relocation_cluster_radius_pixels_effective": float(cluster_radius),
        "requested_shard_rows": int(shard_rows),
        "effective_row_shard_rows": effective_row_shard,
        "effective_frame_shard_rows": effective_frame_shard,
        "row_chunk_rows": int(row_chunk_rows),
        "frame_chunk_rows": int(frame_chunk_rows),
        "source_task_rows": int(task_rows),
        "source_row_shard_grids": list(source_row_shard_grids),
        "source_task_partition_policy": (
            "uniform_source_shard_grid"
            if len(source_row_shard_grids) == 1
            else "requested_rows_mixed_read_only_source_grids"
            if source_row_shard_grids
            else "requested_rows_unsharded_source"
        ),
    }
    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result: dict[str, Any] = {
        "status": "planned" if not apply else "running",
        "schema": COLLECTION_QUALITY_SCHEMA,
        "zarr_path": str(archive),
        "source_group_path": source_group_path,
        "source_identity": source_identity,
        "output_run": output_run,
        "output_run_path": f"detect_quality_runs/{output_run}",
        "row_count": row_count,
        "recording_frame_count": resolved_frame_count,
        "worker_tasks": len(ranges),
        "requested_workers": int(workers),
        "effective_workers": min(int(workers), max(len(ranges), 1)),
        "params": params,
        "params_hash": params_hash,
        "promote": bool(promote),
    }
    if not apply:
        return result

    trace_parent = Path(work_dir).expanduser().resolve() if work_dir else None
    if trace_parent is not None:
        trace_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="palette_detect_quality_", dir=trace_parent) as tmp:
        tasks = [
            ShardTask(
                zarr_path=str(archive),
                source_group_path=source_group_path,
                start=start,
                stop=stop,
                width=resolved_width,
                height=resolved_height,
                output_path=str(Path(tmp) / f"trace_{index:06d}.npz"),
                params_hash=params_hash,
            )
            for index, (start, stop) in enumerate(ranges)
        ]
        effective_workers = min(int(workers), max(len(tasks), 1))
        if effective_workers == 1:
            traces = [_worker(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as pool:
                traces = list(pool.map(_worker, tasks))
        trace = _load_traces(traces, expected_rows=row_count, params_hash=params_hash)
        aggregate = _aggregate_quality(
            trace,
            frame_count=resolved_frame_count,
            expected_subject_count=expected_subject_count,
            jump_threshold_pixels=effective_jump,
            blip_gap_threshold=blip_gap_threshold,
            relocation_confirm_count=relocation_confirm_count,
            relocation_cluster_radius_pixels=cluster_radius,
        )

        write_root = open_zarr_root(archive, mode="a")
        parent = require_runs_parent(write_root, "detect_quality_runs")
        if output_run in parent:
            raise ValueError(f"detect_quality_runs/{output_run} already exists.")
        target = parent.create_group(output_run)
        mark_run_started(target, run_name=output_run, stage="detect_quality_collection")
        if promote:
            note_pending_latest(parent, output_run)
        try:
            target.attrs.update(
                {
                    "schema_id": COLLECTION_QUALITY_SCHEMA,
                    "artifact_mutability": "immutable_snapshot",
                    "quality_label_schema_id": LABEL_SCHEMA,
                    "quality_label_codes": {
                        "-1": "no_detection",
                        "0": "clean",
                        "2": "blip",
                        "3": "jump",
                        "4": "over_expected_detections",
                    },
                    "temporal_artifact_policy": TEMPORAL_POLICY_SCHEMA,
                    "source_detect_identity_kind": source_identity["kind"],
                    "source_detect_run": source_identity["id"],
                    "source_detect_path": source_identity["path"],
                    "source_detection_group_path": source_group_path,
                    "source_row_count": row_count,
                    "recording_frame_count": resolved_frame_count,
                    "source_video_width": int(resolved_width),
                    "source_video_height": int(resolved_height),
                    "width": int(resolved_width),
                    "height": int(resolved_height),
                    "full_frame_geometry_schema": FULL_FRAME_GEOMETRY_SCHEMA,
                    "full_frame_geometry_source": geometry_source,
                    "expected_subject_count": expected_subject_count,
                    "experiment_setup_path": experiment_setup.group_path,
                    "experiment_setup_sha256": experiment_setup.record_sha256,
                    "experiment_setup_legacy": experiment_setup.legacy,
                    "frame_index_semantics": "recording_parent_frame_index_0_based",
                    "row_identity": "instance_key",
                    "storage_layout": "indexed_sharding_v1",
                    "row_shard_rows": effective_row_shard,
                    "frame_shard_rows": effective_frame_shard,
                    "requested_workers": int(workers),
                    "effective_workers": effective_workers,
                    "worker_task_count": len(tasks),
                    "quality_params": params,
                    "quality_params_hash": params_hash,
                    "coverage_stats": aggregate["coverage"],
                    "bbox_validation": aggregate["bbox_validation"],
                    "artifact_summary": aggregate["artifacts"],
                    "quality_score": aggregate["quality_score"],
                    "detection_quality_summary": aggregate["summary"],
                }
            )
            quality_flags = target.create_array(
                "quality_flags",
                shape=(resolved_frame_count,),
                dtype=np.int8,
                chunks=(int(frame_chunk_rows),),
                shards=(effective_frame_shard,),
            )
            labels = target.create_array(
                "detection_quality_labels",
                shape=(row_count,),
                dtype=np.int8,
                chunks=(int(row_chunk_rows),),
                shards=(effective_row_shard,),
            )
            keys = target.create_array(
                "instance_key",
                shape=(row_count,),
                dtype=np.uint64,
                chunks=(int(row_chunk_rows),),
                shards=(effective_row_shard,),
            )
            _write_shardwise(quality_flags, aggregate["quality_flags"])
            _write_shardwise(labels, aggregate["detection_quality_labels"])
            _write_shardwise(keys, aggregate["instance_key"])

            source_key_hash = _hash_arrays(aggregate["instance_key"])
            source_frame_hash = _hash_arrays(trace["frame_indices"])
            live_source = resolve_detection_instance_table(
                _group_at(write_root, source_group_path)
            )
            live_source_key_hash = _digest_array(
                live_source["instance_key"],
                block_rows=effective_row_shard,
            )
            live_source_frame_hash = _digest_array(
                live_source["frame_indices"],
                block_rows=effective_row_shard,
                dtype=np.dtype(np.int64),
            )
            if source_key_hash != live_source_key_hash:
                raise RuntimeError("Source instance_key changed after shard workers completed.")
            if source_frame_hash != live_source_frame_hash:
                raise RuntimeError("Source frame order changed after shard workers completed.")
            output_key_hash = _digest_array(keys, block_rows=effective_row_shard)
            if source_key_hash != output_key_hash:
                raise RuntimeError("Stored instance_key values differ from the source trace.")
            if not np.array_equal(np.asarray(keys[:], dtype=np.uint64), aggregate["instance_key"]):
                raise RuntimeError("Stored instance_key values failed exact reread validation.")
            if not np.array_equal(
                np.asarray(labels[:], dtype=np.int8),
                aggregate["quality_flags"][trace["frame_indices"]],
            ):
                raise RuntimeError("Detection labels do not map exactly from frame quality flags.")
            label_counts = {
                str(code): int(np.sum(aggregate["detection_quality_labels"] == code))
                for code in (NO_DETECTION, CLEAN, BLIP, JUMP, OVER_EXPECTED)
            }
            if label_counts["-1"] != 0:
                raise RuntimeError("Per-detection labels may not contain no-detection rows.")
            validation = {
                "status": "complete",
                "source_instance_key_sha256": source_key_hash,
                "live_source_instance_key_sha256": live_source_key_hash,
                "source_frame_indices_sha256": source_frame_hash,
                "live_source_frame_indices_sha256": live_source_frame_hash,
                "destination_instance_key_sha256": output_key_hash,
                "instance_key_exact": True,
                "instance_key_unique": True,
                "row_count": row_count,
                "recording_frame_count": resolved_frame_count,
                "source_video_width": int(resolved_width),
                "source_video_height": int(resolved_height),
                "full_frame_geometry_source": geometry_source,
                "label_counts": label_counts,
                "arrays_indexed_sharded": True,
                "trace_ranges_complete_nonoverlapping": True,
                "source_rows_canonical_frame_order": True,
                "quality_params_uniform": True,
                "label_schema_uniform": True,
            }
            target.attrs["collection_quality_validation"] = validation
            provenance = build_writer_run_provenance(
                command="fisheye.refinement.detect_quality_collection",
                params={**params, "output_run": output_run, "promote": bool(promote)},
                input_run_ids={
                    "source_detect_identity": source_identity,
                    "source_detection_group_path": source_group_path,
                    "experiment_setup_path": experiment_setup.group_path,
                    "experiment_setup_sha256": experiment_setup.record_sha256,
                },
                cwd=Path.cwd(),
            )
            target.attrs[RUN_PROVENANCE_ATTR] = provenance
            target.attrs["collection_quality_completed_at_utc"] = _utc_now()
            mark_run_complete(
                target,
                parent_group=parent if promote else None,
                run_name=output_run,
                run_provenance=provenance,
            )
            result.update(
                {
                    "status": "complete",
                    "validation": validation,
                    "summary": aggregate["summary"],
                    "quality_score": aggregate["quality_score"],
                }
            )
            return result
        except Exception as exc:
            mark_run_failed(
                target,
                parent_group=parent if promote else None,
                run_name=output_run,
                error=str(exc),
            )
            raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--recording-frame-count", type=int, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--expected-subject-count", type=int, default=None)
    parser.add_argument("--jump-threshold", type=float, default=100.0)
    parser.add_argument(
        "--threshold-mode",
        choices=("scaled", "pixels", "normalized"),
        default="scaled",
    )
    parser.add_argument("--threshold-reference-width", type=float, default=640.0)
    parser.add_argument("--blip-gap-threshold", type=int, default=10)
    parser.add_argument(
        "--relocation-confirm-count",
        type=int,
        default=DEFAULT_RELOCATION_CONFIRM_COUNT,
    )
    parser.add_argument(
        "--relocation-cluster-radius-fraction",
        type=float,
        default=DEFAULT_RELOCATION_CLUSTER_RADIUS_FRACTION,
    )
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--row-chunk-rows", type=int, default=DEFAULT_ROW_CHUNK_ROWS)
    parser.add_argument("--frame-chunk-rows", type=int, default=DEFAULT_FRAME_CHUNK_ROWS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_collection_detect_quality(
        zarr_path=args.zarr_path,
        source_group_path=args.source_group,
        output_run=args.output_run,
        recording_frame_count=args.recording_frame_count,
        width=args.width,
        height=args.height,
        expected_subject_count=args.expected_subject_count,
        jump_threshold=args.jump_threshold,
        threshold_mode=args.threshold_mode,
        threshold_reference_width=args.threshold_reference_width,
        blip_gap_threshold=args.blip_gap_threshold,
        relocation_confirm_count=args.relocation_confirm_count,
        relocation_cluster_radius_fraction=args.relocation_cluster_radius_fraction,
        shard_rows=args.shard_rows,
        row_chunk_rows=args.row_chunk_rows,
        frame_chunk_rows=args.frame_chunk_rows,
        workers=args.workers,
        work_dir=args.work_dir,
        apply=args.apply,
        promote=not args.no_promote,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

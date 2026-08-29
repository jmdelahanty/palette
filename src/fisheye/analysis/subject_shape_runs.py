"""Build analysis/subject_shape_runs from refined subject masks."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import heapq
import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import dask
from dask import delayed
import numpy as np
from skimage.measure import find_contours, label as label_components
from skimage.morphology import skeletonize
import zarr
from threadpoolctl import threadpool_limits

try:
    from dask.distributed import Client, LocalCluster

    HAVE_DISTRIBUTED = True
except ImportError:  # pragma: no cover - depends on optional dependency
    Client = None  # type: ignore
    LocalCluster = None  # type: ignore
    HAVE_DISTRIBUTED = False

from ..shared.detect_reason_codec import decode_reason_bytes
from ..shared.archive_identity import archive_identity
from ..shared.coordinate_record import (
    bind_persisted_coordinate_record,
    coordinate_record_sha256,
    stamp_and_bind_persisted_coordinate_record,
)
from ..shared.json_safety import json_attr_safe
from ..shared.mask_geometry import batch_mask_spatial_metrics, measure_mask_ellipse as _measure_mask
from ..shared.row_lineage import copy_row_lineage_arrays
from ..shared.proof_verification import (
    finish_proof_verification,
    proof_verification_operation,
    restart_proof_verification,
)
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)
from ..shared.refined_subject_masks_io import (
    RefinedSubjectMasksRunTables,
    load_refined_subject_masks_run_tables,
    resolve_refined_subject_masks_run,
)
from ..shared.refined_subject_mask_coordinate_publication import (
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from ..shared.mask_store import MaskStore, open_mask_store
from ..shared.subject_mask_chunks import (
    REFINED_SUBJECT_MASK_DASK_CHUNK_ALIGNMENT,
    refined_subject_mask_dask_worker_row_chunk,
    refined_subject_mask_metric_row_chunk,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.subject_shape_coordinate_publication import (
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
    CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER,
    CANONICAL_SUBJECT_SHAPE_METHOD,
    CANONICAL_SUBJECT_SHAPE_METHOD_VERSION,
    CANONICAL_SUBJECT_SHAPE_RELATION_ORDER,
    CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
    CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION,
    SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
    SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR,
    SUBJECT_SHAPE_BUNDLE_ID_ATTR,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_COMPUTING_UNBOUND_STATUS,
    SUBJECT_SHAPE_COORDINATE_CONTRACT,
    SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR,
    SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
    SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS,
    SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR,
    SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR,
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SUBJECT_SHAPE_SOURCE_BINDING_ATTR,
    SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR,
    SUBJECT_SHAPE_SOURCE_KIND_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
    SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
    DeferredSubjectShapeCoordinateActivation,
    activate_subject_shape_coordinate_publication,
    build_subject_shape_unbound_numeric_manifest_record,
    derive_canonical_subject_shape_body_frame,
    mask_invalid_subject_shape_vectors,
    project_subject_shape_bboxes,
    project_subject_shape_ellipses,
    project_subject_shape_points,
    require_subject_shape_source_camera_numeric_projection,
    stamp_subject_shape_source_camera_numeric_projection,
    load_exact_subject_shape_source,
    load_sealed_unbound_subject_shape_manifest as _load_shared_sealed_unbound_manifest,
    load_unbound_subject_shape_manifest as _load_shared_unbound_manifest,
    prepare_subject_shape_identity_and_schema,
    publish_subject_shape_coordinate_surfaces,
    selector_snapshot,
    stamp_unbound_subject_shape_manifest,
    validate_sealed_subject_shape_publication_metadata,
)
from ..shared.subject_shape_storage import (
    build_subject_shape_unbound_payload_scan_receipt,
)
from ..shared.zarr.subject_shape_bundle_source import (
    BoundSubjectShapeBundleSource,
    assignment_rebinding_run_id_from_source_record,
    load_subject_shape_bundle_source,
)
from ..shared.zarr_io import open_zarr_root
from .subject_shape_spline import (
    BSPLINE_METHOD,
    DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT,
    DEFAULT_BSPLINE_DEGREE,
    DEFAULT_BSPLINE_SMOOTHING,
    DEFAULT_TAIL_CURVATURE_SMOOTHING_PX,
    DEFAULT_TAIL_SAMPLE_COUNT,
    TAIL_CURVATURE_METHOD,
    SubjectBodySplineBatch,
    fit_subject_body_spline_batch,
    tail_sample_positions,
)

SUBJECT_SHAPE_SCHEMA_ID = CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID
SUBJECT_SHAPE_SCHEMA_VERSION = CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_VERSION
SOURCE_REFINED_SUBJECT_MASKS_SCHEMA_ID = "analysis.subject_shape.source_refined_subject_masks_v1"
SUBJECT_SHAPE_METHOD = CANONICAL_SUBJECT_SHAPE_METHOD
SUBJECT_SHAPE_METHOD_VERSION = CANONICAL_SUBJECT_SHAPE_METHOD_VERSION
SUBJECT_SHAPE_STAGE_NAME = "analysis.subject_shape_runs"
COMPONENT_ORDER = CANONICAL_SUBJECT_SHAPE_COMPONENT_ORDER
RELATION_ORDER = CANONICAL_SUBJECT_SHAPE_RELATION_ORDER
ELLIPSE_COMPONENTS = ("swim_bladder", "eye_left", "eye_right")
EYE_COMPONENTS = ("eye_left", "eye_right")
BODY_FRAME_COMPONENTS = ("swim_bladder", "eye_left", "eye_right")
BODY_FRAME_SCHEMA_ID = "fish_anatomical_body_frame"
BODY_FRAME_SCHEMA_VERSION = 1
BODY_FRAME_ESTIMATOR = "mask_component_axis"
TAIL_GEOMETRY_SCHEMA_ID = "analysis.subject_shape.tail_geometry"
TAIL_GEOMETRY_SCHEMA_VERSION = 1
SNOUT_TIP_METHOD = "subject_body_contour_max_forward_projection_v1"
TAIL_ANCHOR_METHOD = "caudal_swim_bladder_contour_min_forward_projection_v1"
CENTERLINE_METHOD = "snout_anchored_skeleton_longest_endpoint_path_v1"
CENTERLINE_SKELETON_METHOD = "skeleton_longest_endpoint_path_v1"
CENTERLINE_SNOUT_EXTENSION_METHOD = "prepend_mask_path_to_body_frame_guided_join_v1"
CENTERLINE_SNOUT_JOIN_METHOD = "body_frame_lateral_min_head_region_v1"
CENTERLINE_HEAD_ENDPOINT_SEMANTICS = "validated_snout_tip"
CENTERLINE_SAMPLE_COUNT = 64
CENTERLINE_SNOUT_CHECK_METHOD = "head_endpoint_to_snout_distance_v1"
CENTERLINE_REACHES_SNOUT_THRESHOLD_PX = 5.0
CENTERLINE_SNOUT_JOIN_MAX_ARCLENGTH_PX = 64.0
CENTERLINE_SNOUT_EXTENSION_MAX_DISTANCE_PX = 48.0
CENTERLINE_SNOUT_EXTENSION_MAX_LENGTH_RATIO = 3.0
CENTERLINE_SNOUT_EXTENSION_MAX_EXTRA_PX = 24.0
TAIL_SAMPLE_COUNT = DEFAULT_TAIL_SAMPLE_COUNT
REASON_BYTES_WIDTH = 64
SUPPORTED_SCHEDULERS = ("single-threaded", "threads", "processes", "distributed")
EXECUTION_BACKENDS = ("serial_driver", "dask_worker_chunks")
SERIAL_EXECUTION_BACKEND = "serial_driver"
DASK_WORKER_EXECUTION_BACKEND = "dask_worker_chunks"
_SUBJECT_SHAPE_WORKER_CONTEXT_CACHE_MAX_ENTRIES = 8
_SUBJECT_SHAPE_WORKER_CONTEXT_CACHE: dict[
    tuple[object, ...],
    tuple[zarr.Group, zarr.Group, MaskStore],
] = {}
_SUBJECT_SHAPE_WORKER_CONTEXT_CACHE_LOCK = threading.Lock()


def _zarr_store_identity(path: str | Path) -> tuple[object, ...]:
    root = Path(path).expanduser().resolve()
    metadata = root / "zarr.json"
    if not metadata.is_file():
        metadata = root / ".zgroup"
    stat = metadata.stat()
    return (
        str(root),
        str(metadata.name),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _clear_subject_shape_worker_context_cache() -> None:
    """Clear process-local handles; intended for bounded worker/test teardown."""

    with _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE_LOCK:
        _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE.clear()


def _subject_shape_worker_context(
    source_zarr_path: str,
    *,
    output_zarr_path: str,
    refined_run: str,
    shape_run: str,
) -> tuple[tuple[zarr.Group, zarr.Group, MaskStore], bool]:
    """Reuse exact subgroup handles inside one worker process.

    Source publications are immutable and the output path is one exclusively
    owned node-local stage.  Device/inode/size/mtime identity prevents a
    deleted-and-recreated path from inheriting a prior process-local handle.
    """

    key = (
        int(os.getpid()),
        *_zarr_store_identity(source_zarr_path),
        *_zarr_store_identity(output_zarr_path),
        str(refined_run),
        str(shape_run),
    )
    with _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE_LOCK:
        cached = _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE.get(key)
        if cached is not None:
            return cached, True
        source_root = open_zarr_root(source_zarr_path, mode="r")
        output_root = open_zarr_root(output_zarr_path, mode="a")
        refined_group = source_root["refined_subject_masks_runs"][refined_run]
        mask_store = open_mask_store(
            refined_group,
            source_path=f"refined_subject_masks_runs/{refined_run}",
            prefer="dense",
        )
        context = (
            refined_group,
            output_root["analysis"]["subject_shape_runs"][shape_run],
            mask_store,
        )
        if (
            len(_SUBJECT_SHAPE_WORKER_CONTEXT_CACHE)
            >= _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE_MAX_ENTRIES
        ):
            oldest_key = next(iter(_SUBJECT_SHAPE_WORKER_CONTEXT_CACHE))
            del _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE[oldest_key]
        _SUBJECT_SHAPE_WORKER_CONTEXT_CACHE[key] = context
        return context, False


@contextmanager
def _native_thread_limit(num_threads: Optional[int]):
    """Bound native kernels inside one subject-shape worker process."""

    if num_threads is None:
        yield
        return
    limit = max(1, int(num_threads))
    try:
        import cv2

        previous_cv2_threads = int(cv2.getNumThreads())
        cv2.setNumThreads(limit)
    except (ImportError, AttributeError):  # pragma: no cover - optional implementation detail
        cv2 = None
        previous_cv2_threads = None
    try:
        with threadpool_limits(limits=limit):
            yield
    finally:
        if cv2 is not None and previous_cv2_threads is not None:
            cv2.setNumThreads(previous_cv2_threads)


@dataclass(frozen=True)
class ComponentBatch:
    mask_present: np.ndarray
    area_px: np.ndarray
    centroid_xy: np.ndarray
    centroid_valid: np.ndarray
    bbox_xyxy: np.ndarray
    bbox_valid: np.ndarray
    principal_axis_xy: np.ndarray
    principal_axis_valid: np.ndarray
    principal_axis_length_px: np.ndarray
    secondary_axis_length_px: np.ndarray
    ellipse_params: np.ndarray
    ellipse_success: np.ndarray


@dataclass(frozen=True)
class BodyFrameBatch:
    origin_xy: np.ndarray
    forward_axis_xy: np.ndarray
    left_axis_xy: np.ndarray
    heading_deg: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class CaudalAnchorBatch:
    point_xy: np.ndarray
    projection_px: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class SourceBodyMaskQcBatch:
    available: np.ndarray
    severe_qc_failure: np.ndarray
    requires_review: np.ndarray
    reason_bytes: np.ndarray


@dataclass(frozen=True)
class SnoutTipBatch:
    point_xy: np.ndarray
    valid: np.ndarray
    failure_reason_bytes: np.ndarray


@dataclass(frozen=True)
class CenterlineBatch:
    centerline_xy: np.ndarray
    centerline_valid: np.ndarray
    centerline_failure_reason_bytes: np.ndarray
    head_endpoint_xy: np.ndarray
    tail_tip_xy: np.ndarray
    tail_base_xy: np.ndarray
    tail_base_valid: np.ndarray
    tail_base_arclength_px: np.ndarray
    tail_base_failure_reason_bytes: np.ndarray
    tail_segment_arclength_px: np.ndarray
    body_arclength_px: np.ndarray


@dataclass(frozen=True)
class CenterlineSnoutCheckBatch:
    distance_px: np.ndarray
    reaches_snout: np.ndarray
    reason_bytes: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"subject_shape_{stamp}"


def _normalize_scheduler(value: str) -> str:
    scheduler = str(value).strip().lower().replace("_", "-")
    aliases = {
        "single": "single-threaded",
        "single_threaded": "single-threaded",
        "thread": "threads",
        "process": "processes",
        "local-cluster": "distributed",
        "local_cluster": "distributed",
    }
    scheduler = aliases.get(scheduler, scheduler)
    if scheduler not in SUPPORTED_SCHEDULERS:
        raise argparse.ArgumentTypeError(
            f"scheduler must be one of {', '.join(SUPPORTED_SCHEDULERS)}; got {value!r}."
        )
    return scheduler


def _normalize_execution_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("-", "_")
    aliases = {
        "serial": SERIAL_EXECUTION_BACKEND,
        "driver": SERIAL_EXECUTION_BACKEND,
        "dask": DASK_WORKER_EXECUTION_BACKEND,
        "dask_chunks": DASK_WORKER_EXECUTION_BACKEND,
    }
    backend = aliases.get(backend, backend)
    if backend not in EXECUTION_BACKENDS:
        raise ValueError(f"execution_backend must be one of {EXECUTION_BACKENDS}; got {value!r}.")
    return backend


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    chunk = max(1, int(chunk_size))
    return [(start, min(total, start + chunk)) for start in range(0, total, chunk)]


def _worker_chunk_size_for_backend(total_rows: int, requested_chunk_size: int, execution_backend: str) -> int:
    requested = max(1, int(requested_chunk_size))
    if execution_backend == DASK_WORKER_EXECUTION_BACKEND:
        return refined_subject_mask_dask_worker_row_chunk(total_rows, requested)
    return requested


_json_safe = json_attr_safe


def _encode_reason(reason: object, *, width: int = REASON_BYTES_WIDTH) -> np.ndarray:
    data = str(reason or "").encode("utf-8", errors="replace")[: max(0, int(width) - 1)]
    out = np.zeros((int(width),), dtype=np.uint8)
    out[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    return out


def _encode_reasons(reasons: Sequence[object], *, width: int = REASON_BYTES_WIDTH) -> np.ndarray:
    out = np.zeros((len(reasons), int(width)), dtype=np.uint8)
    for idx, reason in enumerate(reasons):
        out[int(idx), :] = _encode_reason(reason, width=width)
    return out


def _set_reason_bytes_attrs(group: zarr.Group, *, width: int = REASON_BYTES_WIDTH) -> None:
    group.attrs["reason_encoding"] = "utf8-null-terminated"
    group.attrs["reason_bytes_width"] = int(width)
    group.attrs["reason_bytes_null_terminated"] = True


def _resolve_refined_run(root: zarr.Group, refined_run: Optional[str]) -> tuple[str, zarr.Group]:
    refined_group, run_name, _run_path = resolve_refined_subject_masks_run(root, refined_run)
    return run_name, refined_group


def _resolve_components_from_refined_tables(
    refined_tables: RefinedSubjectMasksRunTables,
    components: Optional[Sequence[str]],
) -> tuple[tuple[str, int], ...]:
    requested = (
        [str(value) for value in components]
        if components
        else [name for name in COMPONENT_ORDER if name in refined_tables.label_to_index]
    )
    if tuple(requested) != COMPONENT_ORDER:
        raise ValueError(
            "Canonical subject-shape publication requires the exact component "
            f"order {COMPONENT_ORDER!r}; got {tuple(requested)!r}. Use explicit "
            "historical inspection for older subset or reordered variants."
        )
    resolved = refined_tables.resolve_components(requested)
    ordered = tuple((name, idx) for name, idx in resolved if name in requested)
    if not ordered:
        raise ValueError("No available refined subject-mask components selected for subject-shape analysis.")
    return ordered


def _create_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> None:
    if name in group:
        del group[name]
    group.create_array(
        name,
        shape=tuple(int(dim) for dim in shape),
        dtype=dtype,
        chunks=tuple(int(dim) for dim in chunks),
        fill_value=fill_value,
        overwrite=True,
    )


def _metric_chunks(total_rows: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(width))


def _metric_chunks_3d(total_rows: int, middle: int, width: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows), int(middle), int(width))


def _component_review_states(refined_group: zarr.Group, components: Sequence[str]) -> dict[str, str]:
    statuses = refined_group.attrs.get("component_review_statuses")
    if not isinstance(statuses, Mapping):
        return {str(component): "unknown" for component in components}
    result: dict[str, str] = {}
    for component in components:
        payload = statuses.get(str(component))
        if isinstance(payload, Mapping):
            state = payload.get("state") or payload.get("review_state")
            result[str(component)] = str(state or "unknown")
        else:
            result[str(component)] = "unknown"
    return result


def _prepare_component_group(run_group: zarr.Group, component_name: str, *, total_rows: int) -> None:
    chunks_1d = _metric_chunks(total_rows)
    component_group = run_group.require_group("components").require_group(component_name)
    component_group.attrs["component_name"] = component_name
    component_group.attrs["source_component"] = component_name
    component_group.attrs["component_schema_id"] = "analysis.subject_shape_component_v1"
    component_group.attrs["point_coordinate_space"] = "roi_local_px"
    component_group.attrs["bbox_coordinate_space"] = "roi_local_px"
    component_group.attrs["bbox_convention"] = "xyxy_pixel_edge_half_open"
    _create_array(component_group, "mask_present", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(component_group, "area_px", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
    _create_array(
        component_group,
        "centroid_xy",
        shape=(total_rows, 2),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 2),
        fill_value=np.nan,
    )
    _create_array(component_group, "centroid_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(
        component_group,
        "bbox_xyxy",
        shape=(total_rows, 4),
        dtype=np.float32,
        chunks=_metric_chunks_lastdim(total_rows, 4),
        fill_value=np.nan,
    )
    _create_array(component_group, "bbox_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)

    if component_name == "subject_body":
        component_group.attrs["principal_axis_method"] = "pca_mask_pixels_v1"
        component_group.attrs["principal_axis_semantics"] = "unoriented_principal_axis_in_roi_xy"
        component_group.attrs["snout_tip_semantic_label"] = "snout_tip"
        component_group.attrs["snout_tip_estimator"] = SNOUT_TIP_METHOD
        component_group.attrs["snout_tip_estimator_version"] = 1
        component_group.attrs["centerline_method"] = CENTERLINE_METHOD
        component_group.attrs["centerline_skeleton_method"] = CENTERLINE_SKELETON_METHOD
        component_group.attrs["centerline_snout_extension_method"] = CENTERLINE_SNOUT_EXTENSION_METHOD
        component_group.attrs["centerline_snout_join_method"] = CENTERLINE_SNOUT_JOIN_METHOD
        component_group.attrs["head_endpoint_semantics"] = CENTERLINE_HEAD_ENDPOINT_SEMANTICS
        component_group.attrs["centerline_sample_count"] = CENTERLINE_SAMPLE_COUNT
        component_group.attrs["centerline_snout_check_method"] = CENTERLINE_SNOUT_CHECK_METHOD
        component_group.attrs["centerline_reaches_snout_threshold_px"] = CENTERLINE_REACHES_SNOUT_THRESHOLD_PX
        component_group.attrs["centerline_snout_join_max_arclength_px"] = CENTERLINE_SNOUT_JOIN_MAX_ARCLENGTH_PX
        component_group.attrs["centerline_snout_extension_max_distance_px"] = (
            CENTERLINE_SNOUT_EXTENSION_MAX_DISTANCE_PX
        )
        component_group.attrs["centerline_snout_extension_max_length_ratio"] = (
            CENTERLINE_SNOUT_EXTENSION_MAX_LENGTH_RATIO
        )
        component_group.attrs["centerline_snout_extension_max_extra_px"] = CENTERLINE_SNOUT_EXTENSION_MAX_EXTRA_PX
        component_group.attrs["bspline_method"] = BSPLINE_METHOD
        component_group.attrs["bspline_degree"] = DEFAULT_BSPLINE_DEGREE
        component_group.attrs["bspline_fit_mode"] = "interpolating" if DEFAULT_BSPLINE_SMOOTHING == 0.0 else "smoothing"
        component_group.attrs["bspline_smoothing"] = DEFAULT_BSPLINE_SMOOTHING
        component_group.attrs["bspline_arclength_sample_count"] = DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT
        # tail tangent/normal/CURVATURE come from a separate smoothing spline; positions and
        # arc length stay on the interpolating spline. See subject_shape_spline for why.
        component_group.attrs["tail_curvature_method"] = TAIL_CURVATURE_METHOD
        component_group.attrs["tail_curvature_smoothing_px"] = DEFAULT_TAIL_CURVATURE_SMOOTHING_PX
        component_group.attrs["tail_sample_domain"] = "tail_segment_normalized_arclength"
        component_group.attrs["tail_sample_count"] = TAIL_SAMPLE_COUNT
        component_group.attrs["tail_tip_semantic_label"] = "tail_tip"
        component_group.attrs["tail_tip_estimator"] = "subject_body_centerline_posterior_endpoint"
        component_group.attrs["tail_base_definition"] = (
            "body_centerline_projection_of_caudal_swim_bladder_contour_point"
        )
        component_group.attrs["source_mask_qc_semantics"] = (
            "snapshot of refined_subject_masks_runs/<run>/components/subject_body/qc at shape-run creation time"
        )
        _set_reason_bytes_attrs(component_group)
        _create_array(
            component_group,
            "principal_axis_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "principal_axis_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "principal_axis_length_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "secondary_axis_length_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "centerline_xy",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "centerline_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "centerline_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "head_endpoint_to_snout_distance_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(component_group, "centerline_reaches_snout", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "centerline_snout_check_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "bspline_control_points_xy",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "bspline_knots",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT + DEFAULT_BSPLINE_DEGREE + 1),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, CENTERLINE_SAMPLE_COUNT + DEFAULT_BSPLINE_DEGREE + 1),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "bspline_degree_used",
            shape=(total_rows,),
            dtype=np.int16,
            chunks=chunks_1d,
            fill_value=-1,
        )
        _create_array(
            component_group,
            "bspline_sample_xy",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, CENTERLINE_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "bspline_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "bspline_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "bspline_arc_length_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "centerline_curvature_px_inv",
            shape=(total_rows, CENTERLINE_SAMPLE_COUNT),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, CENTERLINE_SAMPLE_COUNT),
            fill_value=np.nan,
        )
        if "tail_sample_s" in component_group:
            del component_group["tail_sample_s"]
        component_group.create_array(
            "tail_sample_s",
            data=tail_sample_positions(TAIL_SAMPLE_COUNT),
            chunks=(TAIL_SAMPLE_COUNT,),
            overwrite=True,
        )
        _create_array(
            component_group,
            "tail_sample_xy",
            shape=(total_rows, TAIL_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, TAIL_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "tail_tangent_xy",
            shape=(total_rows, TAIL_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, TAIL_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "tail_normal_xy",
            shape=(total_rows, TAIL_SAMPLE_COUNT, 2),
            dtype=np.float32,
            chunks=_metric_chunks_3d(total_rows, TAIL_SAMPLE_COUNT, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "tail_curvature_px_inv",
            shape=(total_rows, TAIL_SAMPLE_COUNT),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, TAIL_SAMPLE_COUNT),
            fill_value=np.nan,
        )
        _create_array(component_group, "tail_sample_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "tail_sample_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(component_group, "source_mask_qc_available", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "source_mask_qc_severe_failure",
            shape=(total_rows,),
            dtype=bool,
            chunks=chunks_1d,
        )
        _create_array(
            component_group,
            "source_mask_qc_requires_review",
            shape=(total_rows,),
            dtype=bool,
            chunks=chunks_1d,
        )
        _create_array(
            component_group,
            "source_mask_qc_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "snout_tip_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(component_group, "snout_tip_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "snout_tip_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        for name in ("head_endpoint_xy", "tail_tip_xy", "tail_base_xy"):
            _create_array(
                component_group,
                name,
                shape=(total_rows, 2),
                dtype=np.float32,
                chunks=_metric_chunks_lastdim(total_rows, 2),
                fill_value=np.nan,
            )
        _create_array(component_group, "tail_base_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "tail_base_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "tail_base_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )
        _create_array(
            component_group,
            "tail_segment_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "body_arclength_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )

    if component_name in ELLIPSE_COMPONENTS:
        component_group.attrs["ellipse_method"] = "cv2.fitEllipse_component_contour_v1"
        _create_array(
            component_group,
            "ellipse_params",
            shape=(total_rows, 5),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 5),
            fill_value=np.nan,
        )
        _create_array(component_group, "ellipse_success", shape=(total_rows,), dtype=bool, chunks=chunks_1d)

    if component_name == "swim_bladder":
        component_group.attrs["caudal_anchor_method"] = TAIL_ANCHOR_METHOD
        component_group.attrs["caudal_anchor_definition"] = "min_projection_on_body_forward_axis"
        _set_reason_bytes_attrs(component_group)
        _create_array(
            component_group,
            "caudal_contour_point_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(
            component_group,
            "caudal_contour_projection_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(component_group, "caudal_contour_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            component_group,
            "caudal_contour_failure_reason_bytes",
            shape=(total_rows, REASON_BYTES_WIDTH),
            dtype=np.uint8,
            chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
        )


def _prepare_body_frame_group(run_group: zarr.Group, *, total_rows: int) -> None:
    chunks_1d = _metric_chunks(total_rows)
    group = run_group.require_group("body_frame")
    _set_reason_bytes_attrs(group)
    group.attrs.update(
        {
            "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
            "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
            "body_frame_estimator": BODY_FRAME_ESTIMATOR,
            "body_frame_coordinate_space": "roi_pixels",
            "body_frame_angle_convention": "math_ccw_degrees_after_y_flip",
            "origin_definition": "midpoint_eye_left_eye_right",
            "forward_axis_definition": "swim_bladder_centroid_to_eye_pair_midpoint",
            "left_axis_definition": "eye_right_to_eye_left_projected_perpendicular_to_forward",
        }
    )
    for name in ("origin_xy", "forward_axis_xy", "left_axis_xy"):
        _create_array(
            group,
            name,
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
    _create_array(group, "heading_deg", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
    _create_array(group, "valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    _create_array(
        group,
        "failure_reason_bytes",
        shape=(total_rows, REASON_BYTES_WIDTH),
        dtype=np.uint8,
        chunks=_metric_chunks_lastdim(total_rows, REASON_BYTES_WIDTH),
    )


def _prepare_relation_groups(run_group: zarr.Group, components: Sequence[str], *, total_rows: int) -> tuple[str, ...]:
    chunks_1d = _metric_chunks(total_rows)
    relation_names: list[str] = []
    relations = run_group.require_group("relations")
    component_set = set(components)
    if set(EYE_COMPONENTS).issubset(component_set):
        relation_names.append("eye_pair")
        group = relations.require_group("eye_pair")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.eye_pair_v1"
        group.attrs["relation_components"] = list(EYE_COMPONENTS)
        _create_array(group, "separation_px", shape=(total_rows,), dtype=np.float32, chunks=chunks_1d, fill_value=np.nan)
        _create_array(group, "separation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            group,
            "midpoint_xy",
            shape=(total_rows, 2),
            dtype=np.float32,
            chunks=_metric_chunks_lastdim(total_rows, 2),
            fill_value=np.nan,
        )
        _create_array(group, "midpoint_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
    if {"subject_body", "swim_bladder"}.issubset(component_set):
        relation_names.append("swim_bladder_to_body")
        group = relations.require_group("swim_bladder_to_body")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.swim_bladder_to_body_v1"
        group.attrs["relation_components"] = ["swim_bladder", "subject_body"]
        group.attrs["axis_semantics"] = "unoriented_body_principal_axis"
        _create_array(group, "relation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
        _create_array(
            group,
            "distance_to_body_centroid_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            group,
            "longitudinal_offset_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
        _create_array(
            group,
            "lateral_offset_px",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=chunks_1d,
            fill_value=np.nan,
        )
    if {"subject_body", *EYE_COMPONENTS}.issubset(component_set):
        relation_names.append("eyes_to_body")
        group = relations.require_group("eyes_to_body")
        group.attrs["relation_schema_id"] = "analysis.subject_shape.eyes_to_body_v1"
        group.attrs["relation_components"] = ["eye_left", "eye_right", "subject_body"]
        group.attrs["angle_semantics"] = "unoriented_eye_major_axis_relative_to_unoriented_body_principal_axis"
        for eye in EYE_COMPONENTS:
            prefix = "left" if eye == "eye_left" else "right"
            _create_array(group, f"{prefix}_eye_relation_valid", shape=(total_rows,), dtype=bool, chunks=chunks_1d)
            _create_array(
                group,
                f"{prefix}_eye_offset_xy",
                shape=(total_rows, 2),
                dtype=np.float32,
                chunks=_metric_chunks_lastdim(total_rows, 2),
                fill_value=np.nan,
            )
            _create_array(
                group,
                f"{prefix}_eye_distance_to_body_centroid_px",
                shape=(total_rows,),
                dtype=np.float32,
                chunks=chunks_1d,
                fill_value=np.nan,
            )
            _create_array(
                group,
                f"{prefix}_eye_axis_angle_to_body_rad",
                shape=(total_rows,),
                dtype=np.float32,
                chunks=chunks_1d,
                fill_value=np.nan,
            )
    resolved_relations = tuple(relation_names)
    if resolved_relations != RELATION_ORDER:
        raise ValueError(
            "Canonical subject-shape relation bundle differs from the exact "
            f"maintained order {RELATION_ORDER!r}; got {resolved_relations!r}."
        )
    return resolved_relations


def _read_refined_component_row_revision(
    refined_group: zarr.Group,
    component_name: str,
    *,
    total_rows: int,
) -> tuple[np.ndarray, bool]:
    components_group = refined_group.get("components")
    if not isinstance(components_group, zarr.Group):
        return np.zeros((int(total_rows),), dtype=np.int64), False
    component_group = components_group.get(str(component_name))
    if not isinstance(component_group, zarr.Group):
        return np.zeros((int(total_rows),), dtype=np.int64), False
    revision = component_group.get("row_revision")
    if revision is None:
        return np.zeros((int(total_rows),), dtype=np.int64), False
    if tuple(revision.shape) != (int(total_rows),):
        raise ValueError(
            f"{component_group.name}/row_revision shape mismatch: expected {(int(total_rows),)}, "
            f"got {tuple(revision.shape)}"
        )
    return np.asarray(revision[:], dtype=np.int64), True


def _write_source_refined_subject_mask_revisions(
    run_group: zarr.Group,
    refined_group: zarr.Group,
    *,
    refined_run_name: str,
    components: Sequence[str],
    total_rows: int,
) -> None:
    source_group = run_group.require_group("source_refined_subject_masks")
    component_names = [str(component) for component in components]
    component_count = int(len(component_names))
    row_revision = np.zeros((int(total_rows), component_count), dtype=np.int64)
    revision_available = np.zeros((component_count,), dtype=bool)
    for component_idx, component_name in enumerate(component_names):
        values, available = _read_refined_component_row_revision(
            refined_group,
            component_name,
            total_rows=int(total_rows),
        )
        row_revision[:, int(component_idx)] = values
        revision_available[int(component_idx)] = bool(available)

    source_group.attrs.update(
        {
            "schema_id": SOURCE_REFINED_SUBJECT_MASKS_SCHEMA_ID,
            "schema_version": 1,
            "source_stage": "refined_subject_masks_runs",
            "source_run": str(refined_run_name),
            "source_path": f"refined_subject_masks_runs/{refined_run_name}",
            "component_names": component_names,
            "row_revision_semantics": (
                "per-component refined mask row-local generation copied at subject-shape run creation"
            ),
        }
    )
    _create_array(
        source_group,
        "row_revision",
        shape=(int(total_rows), component_count),
        dtype=np.int64,
        chunks=(refined_subject_mask_metric_row_chunk(total_rows), max(1, component_count)),
    )
    source_group["row_revision"][:, :] = row_revision
    _create_array(
        source_group,
        "row_revision_available",
        shape=(component_count,),
        dtype=bool,
        chunks=(max(1, component_count),),
    )
    source_group["row_revision_available"][:] = revision_available


def audit_subject_shape_source_revisions_group(
    root: zarr.Group,
    *,
    shape_run: Optional[str] = None,
    refined_run: Optional[str] = None,
) -> dict[str, object]:
    """Compare stored subject-shape source row revisions against current refined masks."""

    parent = root.get("analysis/subject_shape_runs")
    if not isinstance(parent, zarr.Group):
        return {"status": "unknown", "reason": "missing_subject_shape_runs"}
    shape_run_name = str(shape_run or parent.attrs.get("latest") or "")
    if not shape_run_name:
        return {"status": "unknown", "reason": "missing_shape_run"}
    if shape_run_name not in parent:
        return {"status": "unknown", "reason": f"shape_run_not_found:{shape_run_name}", "shape_run": shape_run_name}
    shape_group = parent[shape_run_name]
    source_group = shape_group.get("source_refined_subject_masks")
    if not isinstance(source_group, zarr.Group):
        return {"status": "unknown", "reason": "missing_source_revision_group", "shape_run": shape_run_name}
    stored_revision_arr = source_group.get("row_revision")
    if stored_revision_arr is None:
        return {"status": "unknown", "reason": "missing_stored_row_revision", "shape_run": shape_run_name}

    source_run_name = str(refined_run or source_group.attrs.get("source_run") or shape_group.attrs.get("source_refined_subject_masks_run") or "")
    if not source_run_name:
        return {"status": "unknown", "reason": "missing_source_refined_run", "shape_run": shape_run_name}
    refined_parent = root.get("refined_subject_masks_runs")
    if not isinstance(refined_parent, zarr.Group) or source_run_name not in refined_parent:
        return {
            "status": "unknown",
            "reason": f"source_refined_run_not_found:{source_run_name}",
            "shape_run": shape_run_name,
            "source_refined_subject_masks_run": source_run_name,
        }

    refined_group = refined_parent[source_run_name]
    stored_revision = np.asarray(stored_revision_arr[:], dtype=np.int64)
    if stored_revision.ndim != 2:
        return {"status": "unknown", "reason": "stored_row_revision_shape_invalid", "shape_run": shape_run_name}
    total_rows = int(stored_revision.shape[0])
    component_names = [str(value) for value in (source_group.attrs.get("component_names") or [])]
    if not component_names:
        component_names = [str(value) for value in (shape_group.attrs.get("component_names") or [])]
    if int(stored_revision.shape[1]) != len(component_names):
        return {
            "status": "unknown",
            "reason": "component_count_mismatch",
            "shape_run": shape_run_name,
            "stored_component_count": int(stored_revision.shape[1]),
            "component_name_count": int(len(component_names)),
        }

    stale_rows_by_component: dict[str, list[int]] = {}
    unavailable_components: list[str] = []
    for component_idx, component_name in enumerate(component_names):
        try:
            current_revision, available = _read_refined_component_row_revision(
                refined_group,
                component_name,
                total_rows=total_rows,
            )
        except ValueError as exc:
            return {
                "status": "unknown",
                "reason": str(exc),
                "shape_run": shape_run_name,
                "source_refined_subject_masks_run": source_run_name,
            }
        if not available:
            unavailable_components.append(component_name)
            current_revision = np.zeros((total_rows,), dtype=np.int64)
        changed = np.flatnonzero(current_revision != stored_revision[:, int(component_idx)])
        if int(changed.size) > 0:
            stale_rows_by_component[component_name] = [int(value) for value in changed.tolist()]

    stale_rows = sorted({row for rows in stale_rows_by_component.values() for row in rows})
    status = "stale" if stale_rows else "current"
    return {
        "status": status,
        "shape_run": shape_run_name,
        "source_refined_subject_masks_run": source_run_name,
        "component_names": component_names,
        "stale_row_count": int(len(stale_rows)),
        "stale_rows": stale_rows,
        "stale_component_count": int(len(stale_rows_by_component)),
        "stale_rows_by_component": stale_rows_by_component,
        "revision_unavailable_components": unavailable_components,
    }


def _prepare_subject_shape_run(
    root: zarr.Group,
    *,
    target_run: str,
    refined_run_name: str,
    refined_group: zarr.Group,
    source_mask_store: MaskStore,
    source_mask_store_path: str | None,
    component_indices: Sequence[tuple[str, int]],
    requested_chunk_size: int,
    worker_chunk_size: int,
    execution_backend: str,
    scheduler: str,
    num_workers: Optional[int],
    centerline_crop_to_foreground: bool,
    native_threads: Optional[int],
    stage_command: str,
    publication_owner: str,
    write_best_effort_lineage: bool,
    overwrite: bool,
    bundle_source: BoundSubjectShapeBundleSource | None = None,
) -> zarr.Group:
    total_rows = int(source_mask_store.n_rows)
    components = tuple(name for name, _idx in component_indices)
    analysis_group = root.require_group("analysis")
    parent = require_runs_parent(analysis_group, "subject_shape_runs")
    if target_run in parent:
        raise ValueError(
            f"analysis/subject_shape_runs/{target_run} already exists. Canonical "
            "subject-shape runs are immutable; choose a new run name."
        )
    if overwrite:
        raise ValueError(
            "overwrite=True is unsupported for canonical subject-shape publication; "
            "choose a new immutable run name."
        )
    run_group = parent.create_group(target_run)
    mark_run_started(run_group, run_name=target_run, stage="subject_shape")
    run_group.attrs[SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR] = str(publication_owner)
    run_group.attrs["stage_selector_eligible"] = False
    run_group.attrs[SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR] = (
        SUBJECT_SHAPE_COMPUTING_UNBOUND_STATUS
    )

    row_index = run_group.require_group("row_index")
    copy_result = copy_row_lineage_arrays(
        row_index,
        refined_group,
        names=(
            "frame_indices",
            "detection_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
            "source_crop_row_ids",
            "instance_key",
        ),
        total_rois=total_rows,
        overwrite=True,
    )
    run_group.attrs["row_lineage_copied"] = list(copy_result.copied)
    run_group.attrs["row_lineage_missing"] = list(copy_result.missing)

    for component_name in components:
        _prepare_component_group(run_group, component_name, total_rows=total_rows)
    relation_names = _prepare_relation_groups(run_group, components, total_rows=total_rows)
    if set(BODY_FRAME_COMPONENTS).issubset(set(components)):
        _prepare_body_frame_group(run_group, total_rows=total_rows)
    _write_source_refined_subject_mask_revisions(
        run_group,
        refined_group,
        refined_run_name=refined_run_name,
        components=components,
        total_rows=total_rows,
    )

    created = _utc_now()
    dask_metadata = {
        "execution_backend": execution_backend,
        "dask_execution_enabled": execution_backend == DASK_WORKER_EXECUTION_BACKEND,
        "dask_scheduler": scheduler,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_requested_chunk_size": max(1, int(requested_chunk_size)),
        "dask_chunk_size": max(1, int(worker_chunk_size)),
        "dask_chunk_alignment": (
            REFINED_SUBJECT_MASK_DASK_CHUNK_ALIGNMENT
            if execution_backend == DASK_WORKER_EXECUTION_BACKEND
            else "requested_chunk_size"
        ),
        "dask_version": getattr(dask, "__version__", "unknown"),
    }
    source_labels = list(refined_group.attrs.get("mask_labels") or [])
    mask_store_path = str(source_mask_store_path or source_mask_store.storage_path)
    source_refs = {
        "refined_subject_masks": f"refined_subject_masks_runs/{refined_run_name}",
        "refined_subject_masks_mask_store": mask_store_path,
    }
    if "masks_roi" in refined_group:
        source_refs["refined_subject_masks_masks_roi"] = f"refined_subject_masks_runs/{refined_run_name}/masks_roi"
    if "mask_rle" in refined_group:
        source_refs["refined_subject_masks_mask_rle"] = f"refined_subject_masks_runs/{refined_run_name}/mask_rle"
    source_components_group = refined_group.get("components")
    source_body_qc_available = bool(
        source_components_group is not None
        and "subject_body" in source_components_group
        and "qc" in source_components_group["subject_body"]
    )
    if source_body_qc_available:
        source_refs["source_body_mask_qc"] = (
            f"refined_subject_masks_runs/{refined_run_name}/components/subject_body/qc"
        )
    if "source_subject_mask_run" in refined_group.attrs:
        source_refs["source_subject_mask_run"] = str(refined_group.attrs["source_subject_mask_run"])

    bundle_bound = bundle_source is not None
    schema_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION
        if bundle_bound
        else SUBJECT_SHAPE_SCHEMA_VERSION
    )
    method = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD
        if bundle_bound
        else SUBJECT_SHAPE_METHOD
    )
    method_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION
        if bundle_bound
        else SUBJECT_SHAPE_METHOD_VERSION
    )
    source_binding_attrs: dict[str, object] = {}
    if bundle_source is not None:
        source_binding_attrs = {
            SUBJECT_SHAPE_SOURCE_KIND_ATTR: SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
            SUBJECT_SHAPE_BUNDLE_ID_ATTR: bundle_source.bundle_id,
            SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR: bundle_source.active,
            SUBJECT_SHAPE_SOURCE_BINDING_ATTR: json_attr_safe(
                dict(bundle_source.source_record)
            ),
            SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR: bundle_source.source_digest,
        }
    run_group.attrs.update(
        {
            "schema_id": SUBJECT_SHAPE_SCHEMA_ID,
            "schema_version": schema_version,
            "method": method,
            "method_version": method_version,
            "created_at_utc": created,
            "created_utc": created,
            "row_axis": (
                "recording_subject_mask_bundle_rows"
                if bundle_bound
                else "refined_subject_mask_rows"
            ),
            "source_refined_subject_masks_run": refined_run_name,
            "source_refined_subject_masks_stage": "refined_subject_masks_runs",
            "source_mask_labels": source_labels,
            "source_mask_label_schema_id": refined_group.attrs.get("label_schema_id"),
            "source_mask_geometry_schema_id": refined_group.attrs.get("component_metrics_schema_id"),
            "source_mask_store_encoding": source_mask_store.encoding,
            "source_mask_storage_surface": source_mask_store.storage_surface,
            "source_mask_store_path": mask_store_path,
            "source_body_mask_qc_available": bool(source_body_qc_available),
            "source_body_mask_qc_schema_id": (
                refined_group["components"]["subject_body"]["qc"].attrs.get("schema_id")
                if source_body_qc_available
                else None
            ),
            "source_component_review_states": _component_review_states(refined_group, components),
            "source_refs": source_refs,
            "component_names": list(components),
            "relation_names": list(relation_names),
            "body_frame_schema_id": BODY_FRAME_SCHEMA_ID,
            "body_frame_schema_version": BODY_FRAME_SCHEMA_VERSION,
            "body_frame_estimator": (
                BODY_FRAME_ESTIMATOR if set(BODY_FRAME_COMPONENTS).issubset(set(components)) else None
            ),
            "body_frame_source_refs": {
                "refined_subject_masks_run": f"refined_subject_masks_runs/{refined_run_name}",
                "swim_bladder_component": "refined_subject_masks_runs/{}/components/swim_bladder".format(
                    refined_run_name
                ),
                "eye_left_component": f"refined_subject_masks_runs/{refined_run_name}/components/eye_left",
                "eye_right_component": f"refined_subject_masks_runs/{refined_run_name}/components/eye_right",
            }
            if set(BODY_FRAME_COMPONENTS).issubset(set(components))
            else None,
            "tail_geometry_schema_id": TAIL_GEOMETRY_SCHEMA_ID,
            "tail_geometry_schema_version": TAIL_GEOMETRY_SCHEMA_VERSION,
            "snout_tip_semantic_label": "snout_tip",
            "snout_tip_estimator": SNOUT_TIP_METHOD,
            "snout_tip_estimator_version": 1,
            "tail_anchor_method": TAIL_ANCHOR_METHOD,
            "centerline_method": CENTERLINE_METHOD,
            "centerline_skeleton_method": CENTERLINE_SKELETON_METHOD,
            "centerline_snout_extension_method": CENTERLINE_SNOUT_EXTENSION_METHOD,
            "centerline_snout_join_method": CENTERLINE_SNOUT_JOIN_METHOD,
            "head_endpoint_semantics": CENTERLINE_HEAD_ENDPOINT_SEMANTICS,
            "centerline_sample_count": CENTERLINE_SAMPLE_COUNT,
            "centerline_snout_check_method": CENTERLINE_SNOUT_CHECK_METHOD,
            "centerline_reaches_snout_threshold_px": CENTERLINE_REACHES_SNOUT_THRESHOLD_PX,
            "centerline_snout_join_max_arclength_px": CENTERLINE_SNOUT_JOIN_MAX_ARCLENGTH_PX,
            "centerline_snout_extension_max_distance_px": CENTERLINE_SNOUT_EXTENSION_MAX_DISTANCE_PX,
            "centerline_snout_extension_max_length_ratio": CENTERLINE_SNOUT_EXTENSION_MAX_LENGTH_RATIO,
            "centerline_snout_extension_max_extra_px": CENTERLINE_SNOUT_EXTENSION_MAX_EXTRA_PX,
            "bspline_method": BSPLINE_METHOD,
            "bspline_degree": DEFAULT_BSPLINE_DEGREE,
            "bspline_fit_mode": "interpolating" if DEFAULT_BSPLINE_SMOOTHING == 0.0 else "smoothing",
            "bspline_smoothing": DEFAULT_BSPLINE_SMOOTHING,
            "bspline_arclength_sample_count": DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT,
            "tail_curvature_method": TAIL_CURVATURE_METHOD,
            "tail_curvature_smoothing_px": DEFAULT_TAIL_CURVATURE_SMOOTHING_PX,
            "tail_sample_count": TAIL_SAMPLE_COUNT,
            "tail_sample_domain": "tail_segment_normalized_arclength",
            "chunk_size": max(1, int(requested_chunk_size)),
            "worker_chunk_size": max(1, int(worker_chunk_size)),
            "chunk_count": len(_row_chunks(total_rows, max(1, int(worker_chunk_size)))),
            "centerline_crop_to_foreground": bool(centerline_crop_to_foreground),
            "native_threads_per_worker": (
                max(1, int(native_threads)) if native_threads is not None else None
            ),
            **dask_metadata,
            **source_binding_attrs,
        }
    )

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage=SUBJECT_SHAPE_STAGE_NAME,
        command=stage_command,
        created_at_utc=created,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        scheduler=dask_metadata,
        parameters={
            "method": method,
            "components": list(components),
            "relations": list(relation_names),
            "body_frame_estimator": (
                BODY_FRAME_ESTIMATOR if set(BODY_FRAME_COMPONENTS).issubset(set(components)) else None
            ),
            "tail_anchor_method": TAIL_ANCHOR_METHOD,
            "centerline_method": CENTERLINE_METHOD,
            "centerline_sample_count": CENTERLINE_SAMPLE_COUNT,
            "bspline_method": BSPLINE_METHOD,
            "bspline_degree": DEFAULT_BSPLINE_DEGREE,
            "bspline_fit_mode": "interpolating" if DEFAULT_BSPLINE_SMOOTHING == 0.0 else "smoothing",
            "bspline_smoothing": DEFAULT_BSPLINE_SMOOTHING,
            "bspline_arclength_sample_count": DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT,
            "tail_curvature_method": TAIL_CURVATURE_METHOD,
            "tail_curvature_smoothing_px": DEFAULT_TAIL_CURVATURE_SMOOTHING_PX,
            "tail_sample_count": TAIL_SAMPLE_COUNT,
            "tail_sample_domain": "tail_segment_normalized_arclength",
            "chunk_size": max(1, int(requested_chunk_size)),
            "worker_chunk_size": max(1, int(worker_chunk_size)),
            "centerline_crop_to_foreground": bool(centerline_crop_to_foreground),
            "native_threads_per_worker": (
                max(1, int(native_threads)) if native_threads is not None else None
            ),
        },
        inputs={
            "source_refined_subject_masks_run": refined_run_name,
            "source_refined_subject_masks_stage": "refined_subject_masks_runs",
            "source_mask_store_encoding": source_mask_store.encoding,
            "source_mask_storage_surface": source_mask_store.storage_surface,
            "source_mask_store_path": mask_store_path,
            "source_refs": source_refs,
            **(
                {
                    "source_subject_mask_bundle_id": bundle_source.bundle_id,
                    "source_subject_mask_bundle_digest": bundle_source.source_digest,
                }
                if bundle_source is not None
                else {}
            ),
        },
    )
    write_stage_provenance(run_group, provenance)
    if write_best_effort_lineage:
        write_best_effort_run_lineage_attrs(
            run_group,
            run_family="subject_shape_run",
        )
    return run_group


def _compute_principal_axis_metrics(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks_bool = np.asarray(masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    axis_valid = np.zeros((row_count,), dtype=bool)
    major_length = np.full((row_count,), np.nan, dtype=np.float32)
    minor_length = np.full((row_count,), np.nan, dtype=np.float32)
    for row_idx in range(row_count):
        ys, xs = np.nonzero(masks_bool[row_idx])
        if int(xs.size) < 2:
            continue
        coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        centered = coords - coords.mean(axis=0, keepdims=True)
        try:
            cov = centered.T @ centered / max(1, int(coords.shape[0]) - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eigvals)
        major_vec = np.asarray(eigvecs[:, int(order[-1])], dtype=np.float64)
        if not np.all(np.isfinite(major_vec)):
            continue
        norm = float(np.linalg.norm(major_vec))
        if norm <= 0.0:
            continue
        major_vec = major_vec / norm
        if major_vec[0] < 0.0 or (major_vec[0] == 0.0 and major_vec[1] < 0.0):
            major_vec *= -1.0
        minor_vec = np.asarray([-major_vec[1], major_vec[0]], dtype=np.float64)
        major_projection = centered @ major_vec
        minor_projection = centered @ minor_vec
        axis_xy[row_idx] = major_vec.astype(np.float32)
        axis_valid[row_idx] = True
        major_length[row_idx] = np.float32(float(major_projection.max() - major_projection.min() + 1.0))
        minor_length[row_idx] = np.float32(float(minor_projection.max() - minor_projection.min() + 1.0))
    return axis_xy, axis_valid, major_length, minor_length


def _compute_ellipse_metrics(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    row_count = int(masks_u8.shape[0])
    ellipse_params = np.full((row_count, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((row_count,), dtype=bool)
    for row_idx in range(row_count):
        success, ellipse, _centroid, _contour, _reason = _measure_mask(masks_u8[row_idx])
        ellipse_params[row_idx] = np.asarray(ellipse, dtype=np.float32)
        ellipse_success[row_idx] = bool(success)
    return ellipse_params, ellipse_success


def _compute_component_batch(masks: np.ndarray, component_name: str) -> ComponentBatch:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    spatial_metrics = batch_mask_spatial_metrics(masks_u8)
    if component_name == "subject_body":
        axis_xy, axis_valid, major_length, minor_length = _compute_principal_axis_metrics(masks_u8)
    else:
        row_count = int(masks_u8.shape[0])
        axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
        axis_valid = np.zeros((row_count,), dtype=bool)
        major_length = np.full((row_count,), np.nan, dtype=np.float32)
        minor_length = np.full((row_count,), np.nan, dtype=np.float32)
    if component_name in ELLIPSE_COMPONENTS:
        ellipse_params, ellipse_success = _compute_ellipse_metrics(masks_u8)
    else:
        row_count = int(masks_u8.shape[0])
        ellipse_params = np.full((row_count, 5), np.nan, dtype=np.float32)
        ellipse_success = np.zeros((row_count,), dtype=bool)
    return ComponentBatch(
        mask_present=np.asarray(spatial_metrics["mask_present"], dtype=bool),
        area_px=np.asarray(spatial_metrics["area_px"], dtype=np.float32),
        centroid_xy=np.asarray(spatial_metrics["centroid_xy"], dtype=np.float32),
        centroid_valid=np.asarray(spatial_metrics["centroid_valid"], dtype=bool),
        bbox_xyxy=np.asarray(spatial_metrics["bbox_xyxy"], dtype=np.float32),
        bbox_valid=np.asarray(spatial_metrics["bbox_valid"], dtype=bool),
        principal_axis_xy=axis_xy,
        principal_axis_valid=axis_valid,
        principal_axis_length_px=major_length,
        secondary_axis_length_px=minor_length,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
    )


def _projection_offsets(
    offsets_xy: np.ndarray | None,
    *,
    row_count: int,
) -> np.ndarray | None:
    if offsets_xy is None:
        return None
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    if offsets.shape != (int(row_count), 2) or not np.isfinite(offsets).all():
        raise ValueError(
            "Subject-shape source-camera projection offsets must be finite [row,xy]."
        )
    return offsets


def _project_component_batch_for_write(
    batch: ComponentBatch,
    offsets_xy: np.ndarray | None,
) -> ComponentBatch:
    offsets = _projection_offsets(
        offsets_xy,
        row_count=int(batch.mask_present.shape[0]),
    )
    if offsets is None:
        return batch
    return replace(
        batch,
        centroid_xy=project_subject_shape_points(
            batch.centroid_xy,
            offsets,
            batch.centroid_valid,
        ),
        bbox_xyxy=project_subject_shape_bboxes(
            batch.bbox_xyxy,
            offsets,
            batch.bbox_valid,
        ),
        principal_axis_xy=mask_invalid_subject_shape_vectors(
            batch.principal_axis_xy,
            batch.principal_axis_valid,
        ),
        ellipse_params=project_subject_shape_ellipses(
            batch.ellipse_params,
            offsets,
            batch.ellipse_success,
        ),
    )


def _write_component_batch(
    run_group: zarr.Group,
    component_name: str,
    row_slice: slice,
    batch: ComponentBatch,
) -> None:
    group = run_group["components"][component_name]
    group["mask_present"][row_slice] = batch.mask_present
    group["area_px"][row_slice] = batch.area_px
    group["centroid_xy"][row_slice, :] = batch.centroid_xy
    group["centroid_valid"][row_slice] = batch.centroid_valid
    group["bbox_xyxy"][row_slice, :] = batch.bbox_xyxy
    group["bbox_valid"][row_slice] = batch.bbox_valid
    if component_name == "subject_body":
        group["principal_axis_xy"][row_slice, :] = batch.principal_axis_xy
        group["principal_axis_valid"][row_slice] = batch.principal_axis_valid
        group["principal_axis_length_px"][row_slice] = batch.principal_axis_length_px
        group["secondary_axis_length_px"][row_slice] = batch.secondary_axis_length_px
    if component_name in ELLIPSE_COMPONENTS:
        group["ellipse_params"][row_slice, :] = batch.ellipse_params
        group["ellipse_success"][row_slice] = batch.ellipse_success


def _decode_reason_rows(reason_bytes: np.ndarray) -> np.ndarray:
    data = np.asarray(reason_bytes, dtype=np.uint8)
    if data.ndim != 2:
        return np.full((int(data.shape[0]) if data.ndim else 0,), "", dtype=object)
    try:
        return decode_reason_bytes(data)
    except Exception:
        labels: list[str] = []
        for row in data:
            labels.append(bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode("utf-8", "ignore"))
        return np.asarray(labels, dtype=object)


def _read_source_body_mask_qc_batch(refined_group: zarr.Group, row_slice: slice) -> SourceBodyMaskQcBatch:
    row_count = int(row_slice.stop or 0) - int(row_slice.start or 0)
    available = np.zeros((row_count,), dtype=bool)
    severe = np.zeros((row_count,), dtype=bool)
    requires = np.zeros((row_count,), dtype=bool)
    reason_labels = np.full((row_count,), "not_available", dtype=object)

    components = refined_group.get("components")
    body_group = components.get("subject_body") if components is not None else None
    qc_group = body_group.get("qc") if body_group is not None else None
    if qc_group is not None:
        available[:] = True
        if "severe_qc_failure" in qc_group:
            severe[:] = np.asarray(qc_group["severe_qc_failure"][row_slice], dtype=bool)
        if "requires_review" in qc_group:
            requires[:] = np.asarray(qc_group["requires_review"][row_slice], dtype=bool)
        if "reason_bytes" in qc_group:
            reason_labels[:] = _decode_reason_rows(np.asarray(qc_group["reason_bytes"][row_slice], dtype=np.uint8))
        elif "reason" in qc_group:
            reason_labels[:] = np.asarray(qc_group["reason"][row_slice], dtype=object)
        else:
            reason_labels[:] = np.where(requires, "requires_review", "ok")

    return SourceBodyMaskQcBatch(
        available=available,
        severe_qc_failure=severe,
        requires_review=requires,
        reason_bytes=_encode_reasons(reason_labels),
    )


def _write_source_body_mask_qc_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: SourceBodyMaskQcBatch,
) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "source_mask_qc_available" not in group:
        return
    group["source_mask_qc_available"][row_slice] = batch.available
    group["source_mask_qc_severe_failure"][row_slice] = batch.severe_qc_failure
    group["source_mask_qc_requires_review"][row_slice] = batch.requires_review
    group["source_mask_qc_reason_bytes"][row_slice, :] = batch.reason_bytes


def _unit_vector_xy(vector: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(vector, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return None
    return vec / norm


def _compute_body_frame_batch(batches: Mapping[str, ComponentBatch]) -> BodyFrameBatch:
    any_batch = next(iter(batches.values()))
    row_count = int(any_batch.mask_present.shape[0])
    origin_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    forward_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    left_axis_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    heading_deg = np.full((row_count,), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_source_component"] * row_count

    if not set(BODY_FRAME_COMPONENTS).issubset(batches):
        return BodyFrameBatch(
            origin_xy=origin_xy,
            forward_axis_xy=forward_axis_xy,
            left_axis_xy=left_axis_xy,
            heading_deg=heading_deg,
            valid=valid,
            failure_reason_bytes=_encode_reasons(reasons),
        )

    swim = batches["swim_bladder"]
    left = batches["eye_left"]
    right = batches["eye_right"]
    for row_idx in range(row_count):
        if not (swim.centroid_valid[row_idx] and left.centroid_valid[row_idx] and right.centroid_valid[row_idx]):
            reasons[row_idx] = "missing_source_anchor"
            continue
        swim_xy = swim.centroid_xy[row_idx].astype(np.float64)
        left_xy = left.centroid_xy[row_idx].astype(np.float64)
        right_xy = right.centroid_xy[row_idx].astype(np.float64)
        eye_midpoint = (left_xy + right_xy) * 0.5
        forward = _unit_vector_xy(eye_midpoint - swim_xy)
        if forward is None:
            reasons[row_idx] = "degenerate_forward_axis"
            continue
        left_candidate = left_xy - right_xy
        left_candidate = left_candidate - float(np.dot(left_candidate, forward)) * forward
        left_axis = _unit_vector_xy(left_candidate)
        if left_axis is None:
            reasons[row_idx] = "left_right_unresolved"
            continue
        origin_xy[row_idx] = eye_midpoint.astype(np.float32)
        forward_axis_xy[row_idx] = forward.astype(np.float32)
        left_axis_xy[row_idx] = left_axis.astype(np.float32)
        heading_deg[row_idx] = np.float32(math.degrees(math.atan2(-float(forward[1]), float(forward[0]))))
        valid[row_idx] = True
        reasons[row_idx] = "ok"
    return BodyFrameBatch(
        origin_xy=origin_xy,
        forward_axis_xy=forward_axis_xy,
        left_axis_xy=left_axis_xy,
        heading_deg=heading_deg,
        valid=valid,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _compute_canonical_camera_body_frame_batch(
    batches: Mapping[str, ComponentBatch],
) -> BodyFrameBatch:
    labels = tuple(str(value) for value in batches)
    centroids = np.stack(
        [np.asarray(batches[name].centroid_xy, dtype=np.float32) for name in labels],
        axis=1,
    )
    validity = np.stack(
        [np.asarray(batches[name].centroid_valid, dtype=bool) for name in labels],
        axis=1,
    )
    derived = derive_canonical_subject_shape_body_frame(
        labels,
        centroids,
        validity,
    )
    return BodyFrameBatch(
        origin_xy=derived["origin_xy"],
        forward_axis_xy=derived["forward_axis_xy"],
        left_axis_xy=derived["left_axis_xy"],
        heading_deg=derived["heading_deg"],
        valid=derived["valid"],
        failure_reason_bytes=derived["failure_reason_bytes"],
    )


def _write_body_frame_batch(run_group: zarr.Group, row_slice: slice, batch: BodyFrameBatch) -> None:
    if "body_frame" not in run_group:
        return
    group = run_group["body_frame"]
    group["origin_xy"][row_slice, :] = batch.origin_xy
    group["forward_axis_xy"][row_slice, :] = batch.forward_axis_xy
    group["left_axis_xy"][row_slice, :] = batch.left_axis_xy
    group["heading_deg"][row_slice] = batch.heading_deg
    group["valid"][row_slice] = batch.valid
    group["failure_reason_bytes"][row_slice, :] = batch.failure_reason_bytes


def _single_component_count(mask: np.ndarray) -> int:
    labels = label_components(np.asarray(mask, dtype=bool), connectivity=2)
    return int(labels.max()) if labels.size else 0


def _contour_points_xy(mask: np.ndarray) -> Optional[np.ndarray]:
    contours = find_contours(np.asarray(mask, dtype=bool).astype(np.float32), level=0.5)
    if not contours:
        return None
    points: list[np.ndarray] = []
    for contour in contours:
        if int(contour.shape[0]) < 2:
            continue
        xy = np.stack([contour[:, 1], contour[:, 0]], axis=1).astype(np.float64, copy=False)
        points.append(xy)
    if not points:
        return None
    return np.concatenate(points, axis=0)


def _compute_caudal_anchor_batch(
    swim_masks: np.ndarray,
    body_frame: BodyFrameBatch,
) -> CaudalAnchorBatch:
    masks_bool = np.asarray(swim_masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    point_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    projection_px = np.full((row_count,), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_swim_bladder_mask"] * row_count
    for row_idx in range(row_count):
        if not bool(body_frame.valid[row_idx]):
            reasons[row_idx] = "missing_body_frame"
            continue
        mask = masks_bool[row_idx]
        if int(np.count_nonzero(mask)) == 0:
            reasons[row_idx] = "missing_swim_bladder_mask"
            continue
        if _single_component_count(mask) != 1:
            reasons[row_idx] = "fragmented_swim_bladder_mask"
            continue
        contour_xy = _contour_points_xy(mask)
        if contour_xy is None or int(contour_xy.shape[0]) == 0:
            reasons[row_idx] = "empty_swim_bladder_contour"
            continue
        origin = body_frame.origin_xy[row_idx].astype(np.float64)
        forward = body_frame.forward_axis_xy[row_idx].astype(np.float64)
        projections = (contour_xy - origin[None, :]) @ forward
        if not np.any(np.isfinite(projections)):
            reasons[row_idx] = "caudal_projection_failed"
            continue
        caudal_idx = int(np.nanargmin(projections))
        point_xy[row_idx] = contour_xy[caudal_idx].astype(np.float32)
        projection_px[row_idx] = np.float32(float(projections[caudal_idx]))
        valid[row_idx] = True
        reasons[row_idx] = "ok"
    return CaudalAnchorBatch(
        point_xy=point_xy,
        projection_px=projection_px,
        valid=valid,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _write_caudal_anchor_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: CaudalAnchorBatch,
    *,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> None:
    components = run_group.get("components")
    if components is None or "swim_bladder" not in components:
        return
    group = components["swim_bladder"]
    if "caudal_contour_point_xy" not in group:
        return
    point_xy = (
        project_subject_shape_points(
            batch.point_xy,
            source_camera_offsets_xy,
            batch.valid,
        )
        if source_camera_offsets_xy is not None
        else batch.point_xy
    )
    group["caudal_contour_point_xy"][row_slice, :] = point_xy
    group["caudal_contour_projection_px"][row_slice] = batch.projection_px
    group["caudal_contour_valid"][row_slice] = batch.valid
    group["caudal_contour_failure_reason_bytes"][row_slice, :] = batch.failure_reason_bytes


def _compute_snout_tip_batch(
    body_masks: np.ndarray,
    body_frame: BodyFrameBatch,
    *,
    source_body_qc: SourceBodyMaskQcBatch | None = None,
    projection_tolerance_px: float = 1.0,
) -> SnoutTipBatch:
    masks_bool = np.asarray(body_masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    point_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_subject_body_mask"] * row_count
    for row_idx in range(row_count):
        if source_body_qc is not None and bool(source_body_qc.severe_qc_failure[row_idx]):
            reasons[row_idx] = "source_body_mask_qc_failed"
            continue
        if not bool(body_frame.valid[row_idx]):
            reasons[row_idx] = "missing_body_frame"
            continue
        mask = masks_bool[row_idx]
        if int(np.count_nonzero(mask)) == 0:
            reasons[row_idx] = "missing_subject_body_mask"
            continue
        if _single_component_count(mask) != 1:
            reasons[row_idx] = "fragmented_subject_body_mask"
            continue
        contour_xy = _contour_points_xy(mask)
        if contour_xy is None or int(contour_xy.shape[0]) == 0:
            reasons[row_idx] = "missing_subject_body_contour"
            continue
        origin = body_frame.origin_xy[row_idx].astype(np.float64)
        forward = body_frame.forward_axis_xy[row_idx].astype(np.float64)
        left = body_frame.left_axis_xy[row_idx].astype(np.float64)
        projections = (contour_xy - origin[None, :]) @ forward
        if not np.any(np.isfinite(projections)):
            reasons[row_idx] = "rostral_projection_failed"
            continue
        max_projection = float(np.nanmax(projections))
        near_rostral = np.isfinite(projections) & (projections >= max_projection - float(projection_tolerance_px))
        candidates = contour_xy[near_rostral]
        if int(candidates.shape[0]) == 0:
            reasons[row_idx] = "rostral_projection_failed"
            continue
        if left.shape == (2,) and np.all(np.isfinite(left)):
            lateral = np.abs((candidates - origin[None, :]) @ left)
            chosen_idx = int(np.nanargmin(lateral)) if np.any(np.isfinite(lateral)) else 0
        else:
            chosen_idx = 0
        point_xy[row_idx] = candidates[chosen_idx].astype(np.float32)
        valid[row_idx] = True
        reasons[row_idx] = "ok"
    return SnoutTipBatch(
        point_xy=point_xy,
        valid=valid,
        failure_reason_bytes=_encode_reasons(reasons),
    )


def _write_snout_tip_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: SnoutTipBatch,
    *,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "snout_tip_xy" not in group:
        return
    point_xy = (
        project_subject_shape_points(
            batch.point_xy,
            source_camera_offsets_xy,
            batch.valid,
        )
        if source_camera_offsets_xy is not None
        else batch.point_xy
    )
    group["snout_tip_xy"][row_slice, :] = point_xy
    group["snout_tip_valid"][row_slice] = batch.valid
    group["snout_tip_failure_reason_bytes"][row_slice, :] = batch.failure_reason_bytes


def _skeleton_neighbors(coords_yx: np.ndarray) -> list[list[tuple[int, float]]]:
    index_by_coord = {(int(y), int(x)): int(idx) for idx, (y, x) in enumerate(coords_yx)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(int(coords_yx.shape[0]))]
    for idx, (y, x) in enumerate(coords_yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = index_by_coord.get((int(y) + dy, int(x) + dx))
                if other is None:
                    continue
                weight = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
                neighbors[int(idx)].append((int(other), float(weight)))
    return neighbors


def _dijkstra_path_tree(
    neighbors: Sequence[Sequence[tuple[int, float]]],
    start_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(neighbors)
    dist = np.full((node_count,), np.inf, dtype=np.float64)
    prev = np.full((node_count,), -1, dtype=np.int64)
    dist[int(start_idx)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(start_idx))]
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > float(dist[current]):
            continue
        for other, weight in neighbors[current]:
            next_dist = current_dist + float(weight)
            if next_dist < float(dist[other]):
                dist[other] = next_dist
                prev[other] = current
                heapq.heappush(heap, (next_dist, int(other)))
    return dist, prev


def _farthest_endpoint(dist: np.ndarray, endpoints: Sequence[int]) -> Optional[int]:
    best_idx: Optional[int] = None
    best_dist = -np.inf
    for endpoint in endpoints:
        value = float(dist[int(endpoint)])
        if np.isfinite(value) and value > best_dist:
            best_dist = value
            best_idx = int(endpoint)
    return best_idx


def _reconstruct_path(prev: np.ndarray, start_idx: int, end_idx: int) -> Optional[list[int]]:
    path = [int(end_idx)]
    current = int(end_idx)
    seen = {current}
    while current != int(start_idx):
        current = int(prev[current])
        if current < 0 or current in seen:
            return None
        path.append(current)
        seen.add(current)
    path.reverse()
    return path


def _longest_skeleton_endpoint_path_xy(
    mask: np.ndarray,
    *,
    crop_to_foreground: bool = False,
) -> tuple[Optional[np.ndarray], str]:
    mask_bool = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask_bool)) == 0:
        return None, "missing_subject_body_mask"
    coordinate_offset_xy = np.zeros((2,), dtype=np.float64)
    working_mask = mask_bool
    if bool(crop_to_foreground):
        foreground_yx = np.argwhere(mask_bool)
        y0, x0 = np.min(foreground_yx, axis=0).astype(np.int64)
        y1, x1 = (np.max(foreground_yx, axis=0) + 1).astype(np.int64)
        # A one-pixel zero border preserves the full-frame outside-mask
        # boundary condition even when the foreground touches an ROI edge.
        working_mask = np.pad(
            mask_bool[int(y0) : int(y1), int(x0) : int(x1)],
            pad_width=1,
            mode="constant",
            constant_values=False,
        )
        coordinate_offset_xy = np.asarray([float(x0) - 1.0, float(y0) - 1.0])
    if _single_component_count(working_mask) != 1:
        return None, "fragmented_subject_body_mask"
    skeleton = skeletonize(working_mask)
    coords_yx = np.argwhere(skeleton)
    if int(coords_yx.shape[0]) < 2:
        return None, "skeleton_empty"
    neighbors = _skeleton_neighbors(coords_yx)
    endpoints = [idx for idx, items in enumerate(neighbors) if len(items) == 1]
    if len(endpoints) < 2:
        return None, "skeleton_endpoint_ambiguous"
    dist0, _prev0 = _dijkstra_path_tree(neighbors, endpoints[0])
    start = _farthest_endpoint(dist0, endpoints)
    if start is None:
        return None, "centerline_order_failed"
    dist1, prev1 = _dijkstra_path_tree(neighbors, start)
    end = _farthest_endpoint(dist1, endpoints)
    if end is None or end == start:
        return None, "centerline_order_failed"
    path_indices = _reconstruct_path(prev1, start, end)
    if path_indices is None or len(path_indices) < 2:
        return None, "centerline_order_failed"
    path_yx = coords_yx[np.asarray(path_indices, dtype=np.int64)]
    path_xy = np.stack([path_yx[:, 1], path_yx[:, 0]], axis=1).astype(np.float64)
    path_xy += coordinate_offset_xy.reshape(1, 2)
    return path_xy, "ok"


def _polyline_arclength(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if int(points.shape[0]) < 2:
        return np.zeros((int(points.shape[0]),), dtype=np.float64), 0.0
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg)])
    return cumulative, float(cumulative[-1])


def _resample_polyline(points_xy: np.ndarray, sample_count: int) -> tuple[Optional[np.ndarray], float]:
    points = np.asarray(points_xy, dtype=np.float64)
    cumulative, total = _polyline_arclength(points)
    if total <= 1e-6:
        return None, total
    targets = np.linspace(0.0, total, int(sample_count), dtype=np.float64)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1), total


def _mask_contains_point_near(mask: np.ndarray, point_xy: np.ndarray, *, radius_px: int = 1) -> bool:
    mask_bool = np.asarray(mask, dtype=bool)
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return False
    height, width = mask_bool.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    radius = max(0, int(radius_px))
    y0 = max(0, y - radius)
    y1 = min(int(height), y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(int(width), x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    return bool(np.any(mask_bool[y0:y1, x0:x1]))


def _straight_segment_supported_by_mask(
    mask: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    *,
    sample_count: int = 32,
    radius_px: int = 2,
) -> bool:
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        return False
    count = max(2, int(sample_count))
    for fraction in np.linspace(0.0, 1.0, count, dtype=np.float64):
        point = start + (end - start) * float(fraction)
        if not _mask_contains_point_near(mask, point, radius_px=radius_px):
            return False
    return True


def _nearest_mask_pixel_yx(mask: np.ndarray, point_xy: np.ndarray, *, radius_px: int = 4) -> Optional[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    height, width = mask_bool.shape
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    radius = max(0, int(radius_px))
    y0 = max(0, y - radius)
    y1 = min(int(height), y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(int(width), x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    coords = np.argwhere(mask_bool[y0:y1, x0:x1])
    if int(coords.shape[0]) == 0:
        return None
    coords[:, 0] += y0
    coords[:, 1] += x0
    coords_xy = np.stack([coords[:, 1], coords[:, 0]], axis=1).astype(np.float64)
    distances = np.linalg.norm(coords_xy - point.reshape(1, 2), axis=1)
    return coords[int(np.argmin(distances))].astype(np.int64)


def _shortest_mask_path_xy(mask: np.ndarray, start_yx: np.ndarray, end_yx: np.ndarray, *, margin_px: int) -> Optional[np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    start = np.asarray(start_yx, dtype=np.int64).reshape(2)
    end = np.asarray(end_yx, dtype=np.int64).reshape(2)
    height, width = mask_bool.shape
    margin = max(0, int(margin_px))
    y0 = max(0, int(min(start[0], end[0])) - margin)
    y1 = min(int(height), int(max(start[0], end[0])) + margin + 1)
    x0 = max(0, int(min(start[1], end[1])) - margin)
    x1 = min(int(width), int(max(start[1], end[1])) + margin + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    local = mask_bool[y0:y1, x0:x1]
    start_local = (int(start[0] - y0), int(start[1] - x0))
    end_local = (int(end[0] - y0), int(end[1] - x0))
    if not (local[start_local] and local[end_local]):
        return None

    dist = np.full(local.shape, np.inf, dtype=np.float64)
    prev_y = np.full(local.shape, -1, dtype=np.int32)
    prev_x = np.full(local.shape, -1, dtype=np.int32)
    dist[start_local] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, int(start_local[0]), int(start_local[1]))]
    neighbor_steps = (
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    )
    while heap:
        current_dist, y, x = heapq.heappop(heap)
        if current_dist > float(dist[y, x]):
            continue
        if (y, x) == end_local:
            break
        for dy, dx, weight in neighbor_steps:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= local.shape[0] or xx < 0 or xx >= local.shape[1] or not bool(local[yy, xx]):
                continue
            next_dist = current_dist + float(weight)
            if next_dist < float(dist[yy, xx]):
                dist[yy, xx] = next_dist
                prev_y[yy, xx] = y
                prev_x[yy, xx] = x
                heapq.heappush(heap, (next_dist, yy, xx))
    if not np.isfinite(dist[end_local]):
        return None

    coords: list[tuple[int, int]] = []
    y, x = end_local
    seen: set[tuple[int, int]] = set()
    while True:
        if (y, x) in seen:
            return None
        seen.add((y, x))
        coords.append((y + y0, x + x0))
        if (y, x) == start_local:
            break
        py = int(prev_y[y, x])
        px = int(prev_x[y, x])
        if py < 0 or px < 0:
            return None
        y, x = py, px
    coords.reverse()
    coords_yx = np.asarray(coords, dtype=np.float64)
    return np.stack([coords_yx[:, 1], coords_yx[:, 0]], axis=1)


def _snout_join_index(
    path_xy: np.ndarray,
    body_frame_origin_xy: np.ndarray,
    body_frame_left_axis_xy: np.ndarray,
    *,
    max_arclength_px: float = CENTERLINE_SNOUT_JOIN_MAX_ARCLENGTH_PX,
) -> int:
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or int(path.shape[0]) < 2:
        return 0
    origin = np.asarray(body_frame_origin_xy, dtype=np.float64)
    left = np.asarray(body_frame_left_axis_xy, dtype=np.float64)
    if origin.shape != (2,) or left.shape != (2,) or not (np.all(np.isfinite(origin)) and np.all(np.isfinite(left))):
        return 0
    cumulative, _total = _polyline_arclength(path)
    eligible = np.flatnonzero(cumulative <= float(max_arclength_px))
    eligible = eligible[eligible < int(path.shape[0]) - 1]
    if int(eligible.shape[0]) == 0:
        return 0
    lateral = np.abs((path[eligible] - origin.reshape(1, 2)) @ left)
    if not np.any(np.isfinite(lateral)):
        return 0
    score = lateral + 0.02 * cumulative[eligible]
    return int(eligible[int(np.nanargmin(score))])


def _snout_bridge_path_xy(
    mask: np.ndarray,
    snout_xy: np.ndarray,
    skeleton_head_xy: np.ndarray,
    *,
    max_distance_px: float = CENTERLINE_SNOUT_EXTENSION_MAX_DISTANCE_PX,
    max_length_ratio: float = CENTERLINE_SNOUT_EXTENSION_MAX_LENGTH_RATIO,
    max_extra_px: float = CENTERLINE_SNOUT_EXTENSION_MAX_EXTRA_PX,
) -> tuple[Optional[np.ndarray], str]:
    snout = np.asarray(snout_xy, dtype=np.float64)
    head = np.asarray(skeleton_head_xy, dtype=np.float64)
    if snout.shape != (2,) or head.shape != (2,) or not (np.all(np.isfinite(snout)) and np.all(np.isfinite(head))):
        return None, "missing_snout_tip"
    straight_distance = float(np.linalg.norm(head - snout))
    if straight_distance <= 1e-6:
        return snout.reshape(1, 2), "ok"
    if straight_distance > float(max_distance_px):
        return None, "snout_extension_too_long"
    if _straight_segment_supported_by_mask(mask, snout, head, sample_count=32, radius_px=2):
        return np.vstack([snout.reshape(1, 2), head.reshape(1, 2)]), "ok"
    start_yx = _nearest_mask_pixel_yx(mask, snout, radius_px=4)
    end_yx = _nearest_mask_pixel_yx(mask, head, radius_px=2)
    if start_yx is None or end_yx is None:
        return None, "snout_extension_endpoint_outside_mask"
    margin_px = int(math.ceil(max(8.0, min(24.0, straight_distance * 0.75))))
    mask_path = _shortest_mask_path_xy(mask, start_yx, end_yx, margin_px=margin_px)
    if mask_path is None or int(mask_path.shape[0]) == 0:
        return None, "snout_extension_no_mask_path"
    bridge = np.vstack([snout.reshape(1, 2), mask_path, head.reshape(1, 2)])
    keep = [0]
    for idx in range(1, int(bridge.shape[0])):
        if float(np.linalg.norm(bridge[idx] - bridge[keep[-1]])) > 1e-6:
            keep.append(idx)
    bridge = bridge[np.asarray(keep, dtype=np.int64)]
    _cumulative, bridge_length = _polyline_arclength(bridge)
    max_allowed = max(straight_distance * float(max_length_ratio), straight_distance + float(max_extra_px))
    if bridge_length > max_allowed:
        return None, "snout_extension_path_too_indirect"
    return bridge, "ok"


def _project_point_to_polyline(point_xy: np.ndarray, polyline_xy: np.ndarray) -> tuple[Optional[np.ndarray], float]:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    polyline = np.asarray(polyline_xy, dtype=np.float64)
    if int(polyline.shape[0]) < 2:
        return None, float("nan")
    cumulative, _total = _polyline_arclength(polyline)
    best_point: Optional[np.ndarray] = None
    best_arclength = float("nan")
    best_dist = np.inf
    for idx in range(int(polyline.shape[0]) - 1):
        a = polyline[idx]
        b = polyline[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        t = min(1.0, max(0.0, float(np.dot(point - a, ab) / denom)))
        candidate = a + t * ab
        dist = float(np.linalg.norm(point - candidate))
        if dist < best_dist:
            best_dist = dist
            best_point = candidate
            best_arclength = float(cumulative[idx] + t * math.sqrt(denom))
    return best_point, best_arclength


def _compute_centerline_batch(
    body_masks: np.ndarray,
    body_frame: BodyFrameBatch,
    caudal_anchor: CaudalAnchorBatch,
    *,
    snout_tip: Optional[SnoutTipBatch] = None,
    source_body_qc: Optional[SourceBodyMaskQcBatch] = None,
    sample_count: int = CENTERLINE_SAMPLE_COUNT,
    crop_to_foreground: bool = False,
) -> CenterlineBatch:
    masks_bool = np.asarray(body_masks, dtype=np.uint8) > 0
    row_count = int(masks_bool.shape[0])
    centerline_xy = np.full((row_count, int(sample_count), 2), np.nan, dtype=np.float32)
    centerline_valid = np.zeros((row_count,), dtype=bool)
    centerline_reasons = ["missing_subject_body_mask"] * row_count
    head_endpoint_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_tip_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_base_xy = np.full((row_count, 2), np.nan, dtype=np.float32)
    tail_base_valid = np.zeros((row_count,), dtype=bool)
    tail_base_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)
    tail_base_reasons = ["missing_centerline"] * row_count
    tail_segment_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)
    body_arclength_px = np.full((row_count,), np.nan, dtype=np.float32)

    for row_idx in range(row_count):
        if source_body_qc is not None and bool(source_body_qc.severe_qc_failure[row_idx]):
            centerline_reasons[row_idx] = "source_body_mask_qc_failed"
            tail_base_reasons[row_idx] = "source_body_mask_qc_failed"
            continue
        if not bool(body_frame.valid[row_idx]):
            centerline_reasons[row_idx] = "missing_body_frame"
            tail_base_reasons[row_idx] = "missing_body_frame"
            continue
        path_xy, reason = _longest_skeleton_endpoint_path_xy(
            masks_bool[row_idx],
            crop_to_foreground=bool(crop_to_foreground),
        )
        if path_xy is None:
            centerline_reasons[row_idx] = reason
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        origin = body_frame.origin_xy[row_idx].astype(np.float64)
        forward = body_frame.forward_axis_xy[row_idx].astype(np.float64)
        first_projection = float(np.dot(path_xy[0] - origin, forward))
        last_projection = float(np.dot(path_xy[-1] - origin, forward))
        if not (np.isfinite(first_projection) and np.isfinite(last_projection)):
            centerline_reasons[row_idx] = "endpoint_orientation_failed"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        if abs(first_projection - last_projection) <= 1e-6:
            centerline_reasons[row_idx] = "ambiguous_polarity"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        if first_projection < last_projection:
            path_xy = path_xy[::-1]
        if snout_tip is None or not bool(snout_tip.valid[row_idx]):
            reason = "missing_snout_tip"
            if snout_tip is not None:
                decoded = str(_decode_reason_rows(snout_tip.failure_reason_bytes[row_idx : row_idx + 1])[0] or "")
                if decoded and decoded != "ok":
                    reason = decoded
            centerline_reasons[row_idx] = reason
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        snout_xy = np.asarray(snout_tip.point_xy[row_idx], dtype=np.float64)
        if snout_xy.shape != (2,) or not np.all(np.isfinite(snout_xy)):
            centerline_reasons[row_idx] = "missing_snout_tip"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        join_idx = _snout_join_index(path_xy, origin, body_frame.left_axis_xy[row_idx].astype(np.float64))
        bridge_xy, bridge_reason = _snout_bridge_path_xy(masks_bool[row_idx], snout_xy, path_xy[join_idx])
        if bridge_xy is None:
            centerline_reasons[row_idx] = bridge_reason
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        if int(bridge_xy.shape[0]) <= 1:
            path_xy[0] = snout_xy
        else:
            path_xy = np.vstack([bridge_xy, path_xy[join_idx + 1 :]])
        sampled, total_length = _resample_polyline(path_xy, int(sample_count))
        if sampled is None or total_length <= 1e-6:
            centerline_reasons[row_idx] = "centerline_order_failed"
            tail_base_reasons[row_idx] = "missing_centerline"
            continue
        centerline_xy[row_idx] = sampled.astype(np.float32)
        centerline_valid[row_idx] = True
        centerline_reasons[row_idx] = "ok"
        head_endpoint_xy[row_idx] = path_xy[0].astype(np.float32)
        tail_tip_xy[row_idx] = path_xy[-1].astype(np.float32)
        body_arclength_px[row_idx] = np.float32(total_length)

        if not bool(caudal_anchor.valid[row_idx]):
            tail_base_reasons[row_idx] = "missing_tail_anchor"
            continue
        projected, arclength = _project_point_to_polyline(caudal_anchor.point_xy[row_idx], path_xy)
        if projected is None or not np.isfinite(arclength):
            tail_base_reasons[row_idx] = "tail_base_projection_failed"
            continue
        tail_length = float(total_length - arclength)
        if tail_length < 0.0:
            tail_base_reasons[row_idx] = "tail_base_projection_failed"
            continue
        tail_base_xy[row_idx] = projected.astype(np.float32)
        tail_base_valid[row_idx] = True
        tail_base_arclength_px[row_idx] = np.float32(arclength)
        tail_segment_arclength_px[row_idx] = np.float32(tail_length)
        tail_base_reasons[row_idx] = "ok"

    return CenterlineBatch(
        centerline_xy=centerline_xy,
        centerline_valid=centerline_valid,
        centerline_failure_reason_bytes=_encode_reasons(centerline_reasons),
        head_endpoint_xy=head_endpoint_xy,
        tail_tip_xy=tail_tip_xy,
        tail_base_xy=tail_base_xy,
        tail_base_valid=tail_base_valid,
        tail_base_arclength_px=tail_base_arclength_px,
        tail_base_failure_reason_bytes=_encode_reasons(tail_base_reasons),
        tail_segment_arclength_px=tail_segment_arclength_px,
        body_arclength_px=body_arclength_px,
    )


def _write_centerline_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: CenterlineBatch,
    *,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "centerline_xy" not in group:
        return
    offsets = source_camera_offsets_xy
    centerline_xy = (
        project_subject_shape_points(
            batch.centerline_xy,
            offsets,
            batch.centerline_valid,
        )
        if offsets is not None
        else batch.centerline_xy
    )
    head_endpoint_xy = (
        project_subject_shape_points(
            batch.head_endpoint_xy,
            offsets,
            batch.centerline_valid,
        )
        if offsets is not None
        else batch.head_endpoint_xy
    )
    tail_tip_xy = (
        project_subject_shape_points(
            batch.tail_tip_xy,
            offsets,
            batch.centerline_valid,
        )
        if offsets is not None
        else batch.tail_tip_xy
    )
    tail_base_xy = (
        project_subject_shape_points(
            batch.tail_base_xy,
            offsets,
            batch.tail_base_valid,
        )
        if offsets is not None
        else batch.tail_base_xy
    )
    group["centerline_xy"][row_slice, :, :] = centerline_xy
    group["centerline_valid"][row_slice] = batch.centerline_valid
    group["centerline_failure_reason_bytes"][row_slice, :] = batch.centerline_failure_reason_bytes
    group["head_endpoint_xy"][row_slice, :] = head_endpoint_xy
    group["tail_tip_xy"][row_slice, :] = tail_tip_xy
    group["tail_base_xy"][row_slice, :] = tail_base_xy
    group["tail_base_valid"][row_slice] = batch.tail_base_valid
    group["tail_base_arclength_px"][row_slice] = batch.tail_base_arclength_px
    group["tail_base_failure_reason_bytes"][row_slice, :] = batch.tail_base_failure_reason_bytes
    group["tail_segment_arclength_px"][row_slice] = batch.tail_segment_arclength_px
    group["body_arclength_px"][row_slice] = batch.body_arclength_px


def _compute_centerline_snout_check_batch(
    snout_tip: SnoutTipBatch,
    centerline: CenterlineBatch,
    *,
    threshold_px: float = CENTERLINE_REACHES_SNOUT_THRESHOLD_PX,
) -> CenterlineSnoutCheckBatch:
    row_count = int(centerline.centerline_valid.shape[0])
    distance_px = np.full((row_count,), np.nan, dtype=np.float32)
    reaches_snout = np.zeros((row_count,), dtype=bool)
    reasons = ["missing_snout_tip"] * row_count
    for row_idx in range(row_count):
        if not bool(snout_tip.valid[row_idx]):
            reasons[row_idx] = "missing_snout_tip"
            continue
        if not bool(centerline.centerline_valid[row_idx]):
            reasons[row_idx] = "missing_centerline"
            continue
        head = np.asarray(centerline.head_endpoint_xy[row_idx], dtype=np.float64)
        snout = np.asarray(snout_tip.point_xy[row_idx], dtype=np.float64)
        if head.shape != (2,) or snout.shape != (2,) or not (np.all(np.isfinite(head)) and np.all(np.isfinite(snout))):
            reasons[row_idx] = "missing_head_endpoint"
            continue
        distance = float(np.linalg.norm(head - snout))
        distance_px[row_idx] = np.float32(distance)
        if distance <= float(threshold_px):
            reaches_snout[row_idx] = True
            reasons[row_idx] = "ok"
        else:
            reasons[row_idx] = "centerline_does_not_reach_snout"
    return CenterlineSnoutCheckBatch(
        distance_px=distance_px,
        reaches_snout=reaches_snout,
        reason_bytes=_encode_reasons(reasons),
    )


def _write_centerline_snout_check_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: CenterlineSnoutCheckBatch,
) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "head_endpoint_to_snout_distance_px" not in group:
        return
    group["head_endpoint_to_snout_distance_px"][row_slice] = batch.distance_px
    group["centerline_reaches_snout"][row_slice] = batch.reaches_snout
    group["centerline_snout_check_reason_bytes"][row_slice, :] = batch.reason_bytes


def _write_subject_body_spline_batch(
    run_group: zarr.Group,
    row_slice: slice,
    batch: SubjectBodySplineBatch,
    *,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> None:
    components = run_group.get("components")
    if components is None or "subject_body" not in components:
        return
    group = components["subject_body"]
    if "bspline_sample_xy" not in group:
        return
    offsets = source_camera_offsets_xy
    bspline_control_points_xy = (
        project_subject_shape_points(
            batch.bspline_control_points_xy,
            offsets,
            batch.bspline_valid,
        )
        if offsets is not None
        else batch.bspline_control_points_xy
    )
    bspline_sample_xy = (
        project_subject_shape_points(
            batch.bspline_sample_xy,
            offsets,
            batch.bspline_valid,
        )
        if offsets is not None
        else batch.bspline_sample_xy
    )
    tail_sample_xy = (
        project_subject_shape_points(
            batch.tail_sample_xy,
            offsets,
            batch.tail_sample_valid,
        )
        if offsets is not None
        else batch.tail_sample_xy
    )
    tail_tangent_xy = (
        mask_invalid_subject_shape_vectors(
            batch.tail_tangent_xy,
            batch.tail_sample_valid,
        )
        if offsets is not None
        else batch.tail_tangent_xy
    )
    tail_normal_xy = (
        mask_invalid_subject_shape_vectors(
            batch.tail_normal_xy,
            batch.tail_sample_valid,
        )
        if offsets is not None
        else batch.tail_normal_xy
    )
    group["bspline_control_points_xy"][row_slice, :, :] = (
        bspline_control_points_xy
    )
    group["bspline_knots"][row_slice, :] = batch.bspline_knots
    group["bspline_degree_used"][row_slice] = batch.bspline_degree_used
    group["bspline_sample_xy"][row_slice, :, :] = bspline_sample_xy
    group["bspline_valid"][row_slice] = batch.bspline_valid
    group["bspline_failure_reason_bytes"][row_slice, :] = _encode_reasons(batch.bspline_failure_reasons)
    group["bspline_arc_length_px"][row_slice] = batch.bspline_arc_length_px
    group["centerline_curvature_px_inv"][row_slice, :] = batch.centerline_curvature_px_inv
    group["tail_sample_xy"][row_slice, :, :] = tail_sample_xy
    group["tail_tangent_xy"][row_slice, :, :] = tail_tangent_xy
    group["tail_normal_xy"][row_slice, :, :] = tail_normal_xy
    group["tail_curvature_px_inv"][row_slice, :] = batch.tail_curvature_px_inv
    group["tail_sample_valid"][row_slice] = batch.tail_sample_valid
    group["tail_sample_failure_reason_bytes"][row_slice, :] = _encode_reasons(batch.tail_sample_failure_reasons)


def _angle_between_unoriented_axes(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    dot = float(np.dot(axis_a, axis_b))
    cross = float(axis_a[0] * axis_b[1] - axis_a[1] * axis_b[0])
    angle = math.atan2(cross, dot)
    if angle > math.pi / 2.0:
        angle -= math.pi
    elif angle < -math.pi / 2.0:
        angle += math.pi
    return float(angle)


def _eye_axis_from_ellipse_params(params: np.ndarray) -> np.ndarray:
    theta = math.radians(float(params[4]))
    return np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)


def _write_relations(
    run_group: zarr.Group,
    row_slice: slice,
    batches: Mapping[str, ComponentBatch],
    *,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> None:
    relations = run_group.get("relations")
    if relations is None:
        return
    if "eye_pair" in relations and set(EYE_COMPONENTS).issubset(batches):
        left = batches["eye_left"]
        right = batches["eye_right"]
        valid = left.centroid_valid & right.centroid_valid
        separation = np.full(valid.shape, np.nan, dtype=np.float32)
        midpoint = np.full((valid.shape[0], 2), np.nan, dtype=np.float32)
        if np.any(valid):
            delta = left.centroid_xy[valid] - right.centroid_xy[valid]
            separation[valid] = np.linalg.norm(delta, axis=1).astype(np.float32, copy=False)
            midpoint[valid] = ((left.centroid_xy[valid] + right.centroid_xy[valid]) * 0.5).astype(np.float32)
        persisted_midpoint = (
            project_subject_shape_points(
                midpoint,
                source_camera_offsets_xy,
                valid,
            )
            if source_camera_offsets_xy is not None
            else midpoint
        )
        group = relations["eye_pair"]
        group["separation_px"][row_slice] = separation
        group["separation_valid"][row_slice] = valid
        group["midpoint_xy"][row_slice, :] = persisted_midpoint
        group["midpoint_valid"][row_slice] = valid

    if "swim_bladder_to_body" in relations and {"subject_body", "swim_bladder"}.issubset(batches):
        body = batches["subject_body"]
        swim = batches["swim_bladder"]
        valid = body.centroid_valid & body.principal_axis_valid & swim.centroid_valid
        distance = np.full(valid.shape, np.nan, dtype=np.float32)
        longitudinal = np.full(valid.shape, np.nan, dtype=np.float32)
        lateral = np.full(valid.shape, np.nan, dtype=np.float32)
        for row_idx in np.flatnonzero(valid):
            axis = body.principal_axis_xy[int(row_idx)].astype(np.float64)
            delta = (swim.centroid_xy[int(row_idx)] - body.centroid_xy[int(row_idx)]).astype(np.float64)
            perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)
            distance[int(row_idx)] = np.float32(np.linalg.norm(delta))
            longitudinal[int(row_idx)] = np.float32(np.dot(delta, axis))
            lateral[int(row_idx)] = np.float32(np.dot(delta, perp))
        group = relations["swim_bladder_to_body"]
        group["relation_valid"][row_slice] = valid
        group["distance_to_body_centroid_px"][row_slice] = distance
        group["longitudinal_offset_px"][row_slice] = longitudinal
        group["lateral_offset_px"][row_slice] = lateral

    if "eyes_to_body" in relations and {"subject_body", *EYE_COMPONENTS}.issubset(batches):
        body = batches["subject_body"]
        group = relations["eyes_to_body"]
        for eye in EYE_COMPONENTS:
            prefix = "left" if eye == "eye_left" else "right"
            eye_batch = batches[eye]
            valid = body.centroid_valid & body.principal_axis_valid & eye_batch.centroid_valid
            offset = np.full((valid.shape[0], 2), np.nan, dtype=np.float32)
            distance = np.full(valid.shape, np.nan, dtype=np.float32)
            angle = np.full(valid.shape, np.nan, dtype=np.float32)
            if np.any(valid):
                offset[valid] = (eye_batch.centroid_xy[valid] - body.centroid_xy[valid]).astype(np.float32)
                distance[valid] = np.linalg.norm(offset[valid], axis=1).astype(np.float32, copy=False)
            angle_valid = valid & eye_batch.ellipse_success
            for row_idx in np.flatnonzero(angle_valid):
                body_axis = body.principal_axis_xy[int(row_idx)]
                eye_axis = _eye_axis_from_ellipse_params(eye_batch.ellipse_params[int(row_idx)])
                angle[int(row_idx)] = np.float32(_angle_between_unoriented_axes(body_axis, eye_axis))
            persisted_offset = (
                mask_invalid_subject_shape_vectors(offset, valid)
                if source_camera_offsets_xy is not None
                else offset
            )
            group[f"{prefix}_eye_relation_valid"][row_slice] = valid
            group[f"{prefix}_eye_offset_xy"][row_slice, :] = persisted_offset
            group[f"{prefix}_eye_distance_to_body_centroid_px"][row_slice] = distance
            group[f"{prefix}_eye_axis_angle_to_body_rad"][row_slice] = angle


def _process_and_write_subject_shape_chunk_groups(
    refined_group: zarr.Group,
    run_group: zarr.Group,
    *,
    mask_store: MaskStore,
    component_indices: Sequence[tuple[str, int]],
    start_row: int,
    stop_row: int,
    chunk_index: int,
    execution_backend: str,
    output_start_row: Optional[int] = None,
    centerline_crop_to_foreground: bool = False,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> dict[str, object]:
    source_row_slice = slice(int(start_row), int(stop_row))
    output_start = int(start_row) if output_start_row is None else int(output_start_row)
    output_stop = output_start + int(stop_row) - int(start_row)
    output_row_slice = slice(output_start, output_stop)
    row_indices = np.arange(int(start_row), int(stop_row), dtype=np.int64)
    chunk_start = time.perf_counter()
    chunk_timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "output_start_row": output_start,
        "output_stop_row": output_stop,
        "row_count": int(stop_row) - int(start_row),
        "execution_backend": execution_backend,
        "centerline_crop_to_foreground": bool(centerline_crop_to_foreground),
    }
    source_read_seconds = 0.0
    compute_seconds = 0.0
    write_seconds = 0.0
    batches: dict[str, ComponentBatch] = {}
    component_masks_by_name: dict[str, np.ndarray] = {}
    rows_with_component: dict[str, int] = {}
    source_body_qc: SourceBodyMaskQcBatch | None = None
    chunk_offsets = _projection_offsets(
        source_camera_offsets_xy,
        row_count=int(stop_row) - int(start_row),
    )
    persisted_batches: dict[str, ComponentBatch] = {}
    for component_name, component_idx in component_indices:
        phase_start = time.perf_counter()
        read_start = time.perf_counter()
        component_masks = np.asarray(
            mask_store.read_dense(rows=row_indices, channels=[int(component_idx)])[:, 0],
            dtype=np.uint8,
        )
        read_elapsed = float(time.perf_counter() - read_start)
        source_read_seconds += read_elapsed
        compute_start = time.perf_counter()
        batch = _compute_component_batch(component_masks, str(component_name))
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        persisted_batch = _project_component_batch_for_write(batch, chunk_offsets)
        _write_component_batch(
            run_group,
            str(component_name),
            output_row_slice,
            persisted_batch,
        )
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        batches[str(component_name)] = batch
        persisted_batches[str(component_name)] = persisted_batch
        if str(component_name) in {"subject_body", "swim_bladder"}:
            component_masks_by_name[str(component_name)] = component_masks
        if str(component_name) == "subject_body":
            read_start = time.perf_counter()
            source_body_qc = _read_source_body_mask_qc_batch(refined_group, source_row_slice)
            qc_read_elapsed = float(time.perf_counter() - read_start)
            source_read_seconds += qc_read_elapsed
            write_start = time.perf_counter()
            _write_source_body_mask_qc_batch(run_group, output_row_slice, source_body_qc)
            qc_write_elapsed = float(time.perf_counter() - write_start)
            write_seconds += qc_write_elapsed
            chunk_timing["read_source_body_mask_qc_seconds"] = qc_read_elapsed
            chunk_timing["write_source_body_mask_qc_seconds"] = qc_write_elapsed
            rows_with_component["source_body_mask_qc_severe"] = int(
                np.count_nonzero(source_body_qc.severe_qc_failure)
            )
        rows_with_component[str(component_name)] = int(np.count_nonzero(batch.mask_present))
        chunk_timing[f"write_{component_name}_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing[f"read_{component_name}_seconds"] = read_elapsed
        chunk_timing[f"compute_{component_name}_seconds"] = compute_elapsed
        chunk_timing[f"persist_{component_name}_seconds"] = write_elapsed

    body_frame: Optional[BodyFrameBatch] = None
    if set(BODY_FRAME_COMPONENTS).issubset(batches):
        phase_start = time.perf_counter()
        compute_start = time.perf_counter()
        body_frame = _compute_body_frame_batch(batches)
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        persisted_body_frame = (
            _compute_canonical_camera_body_frame_batch(persisted_batches)
            if chunk_offsets is not None
            else body_frame
        )
        _write_body_frame_batch(run_group, output_row_slice, persisted_body_frame)
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        chunk_timing["write_body_frame_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing["compute_body_frame_seconds"] = compute_elapsed
        chunk_timing["persist_body_frame_seconds"] = write_elapsed
        rows_with_component["body_frame_valid"] = int(np.count_nonzero(body_frame.valid))

    snout_tip: Optional[SnoutTipBatch] = None
    if body_frame is not None and "subject_body" in component_masks_by_name:
        phase_start = time.perf_counter()
        compute_start = time.perf_counter()
        snout_tip = _compute_snout_tip_batch(
            component_masks_by_name["subject_body"],
            body_frame,
            source_body_qc=source_body_qc,
        )
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        _write_snout_tip_batch(
            run_group,
            output_row_slice,
            snout_tip,
            source_camera_offsets_xy=chunk_offsets,
        )
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        chunk_timing["write_snout_tip_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing["compute_snout_tip_seconds"] = compute_elapsed
        chunk_timing["persist_snout_tip_seconds"] = write_elapsed
        rows_with_component["snout_tip_valid"] = int(np.count_nonzero(snout_tip.valid))

    caudal_anchor: Optional[CaudalAnchorBatch] = None
    if body_frame is not None and "swim_bladder" in component_masks_by_name:
        phase_start = time.perf_counter()
        compute_start = time.perf_counter()
        caudal_anchor = _compute_caudal_anchor_batch(component_masks_by_name["swim_bladder"], body_frame)
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        _write_caudal_anchor_batch(
            run_group,
            output_row_slice,
            caudal_anchor,
            source_camera_offsets_xy=chunk_offsets,
        )
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        chunk_timing["write_caudal_anchor_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing["compute_caudal_anchor_seconds"] = compute_elapsed
        chunk_timing["persist_caudal_anchor_seconds"] = write_elapsed
        rows_with_component["caudal_contour_valid"] = int(np.count_nonzero(caudal_anchor.valid))

    if body_frame is not None and caudal_anchor is not None and "subject_body" in component_masks_by_name:
        phase_start = time.perf_counter()
        compute_start = time.perf_counter()
        centerline = _compute_centerline_batch(
            component_masks_by_name["subject_body"],
            body_frame,
            caudal_anchor,
            snout_tip=snout_tip,
            source_body_qc=source_body_qc,
            sample_count=CENTERLINE_SAMPLE_COUNT,
            crop_to_foreground=bool(centerline_crop_to_foreground),
        )
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        _write_centerline_batch(
            run_group,
            output_row_slice,
            centerline,
            source_camera_offsets_xy=chunk_offsets,
        )
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        chunk_timing["write_centerline_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing["compute_centerline_seconds"] = compute_elapsed
        chunk_timing["persist_centerline_seconds"] = write_elapsed
        rows_with_component["centerline_valid"] = int(np.count_nonzero(centerline.centerline_valid))
        rows_with_component["tail_base_valid"] = int(np.count_nonzero(centerline.tail_base_valid))

        if snout_tip is not None:
            phase_start = time.perf_counter()
            compute_start = time.perf_counter()
            snout_check = _compute_centerline_snout_check_batch(snout_tip, centerline)
            compute_elapsed = float(time.perf_counter() - compute_start)
            compute_seconds += compute_elapsed
            write_start = time.perf_counter()
            _write_centerline_snout_check_batch(run_group, output_row_slice, snout_check)
            write_elapsed = float(time.perf_counter() - write_start)
            write_seconds += write_elapsed
            chunk_timing["write_centerline_snout_check_seconds"] = float(time.perf_counter() - phase_start)
            chunk_timing["compute_centerline_snout_check_seconds"] = compute_elapsed
            chunk_timing["persist_centerline_snout_check_seconds"] = write_elapsed
            rows_with_component["centerline_reaches_snout"] = int(np.count_nonzero(snout_check.reaches_snout))

        phase_start = time.perf_counter()
        compute_start = time.perf_counter()
        spline = fit_subject_body_spline_batch(
            centerline.centerline_xy,
            centerline.centerline_valid,
            centerline.tail_base_valid,
            centerline.tail_base_arclength_px,
            centerline_failure_reasons=_decode_reason_rows(centerline.centerline_failure_reason_bytes),
            tail_base_failure_reasons=_decode_reason_rows(centerline.tail_base_failure_reason_bytes),
            centerline_sample_count=CENTERLINE_SAMPLE_COUNT,
            tail_sample_count=TAIL_SAMPLE_COUNT,
            degree=DEFAULT_BSPLINE_DEGREE,
            smoothing=DEFAULT_BSPLINE_SMOOTHING,
            arclength_sample_count=DEFAULT_BSPLINE_ARCLENGTH_SAMPLE_COUNT,
            curvature_smoothing_px=DEFAULT_TAIL_CURVATURE_SMOOTHING_PX,
        )
        compute_elapsed = float(time.perf_counter() - compute_start)
        compute_seconds += compute_elapsed
        write_start = time.perf_counter()
        _write_subject_body_spline_batch(
            run_group,
            output_row_slice,
            spline,
            source_camera_offsets_xy=chunk_offsets,
        )
        write_elapsed = float(time.perf_counter() - write_start)
        write_seconds += write_elapsed
        chunk_timing["write_subject_body_spline_seconds"] = float(time.perf_counter() - phase_start)
        chunk_timing["compute_subject_body_spline_seconds"] = compute_elapsed
        chunk_timing["persist_subject_body_spline_seconds"] = write_elapsed
        rows_with_component["bspline_valid"] = int(np.count_nonzero(spline.bspline_valid))
        rows_with_component["tail_sample_valid"] = int(np.count_nonzero(spline.tail_sample_valid))

    phase_start = time.perf_counter()
    _write_relations(
        run_group,
        output_row_slice,
        batches,
        source_camera_offsets_xy=chunk_offsets,
    )
    relations_elapsed = float(time.perf_counter() - phase_start)
    chunk_timing["write_relations_seconds"] = relations_elapsed
    chunk_timing["relations_compute_write_seconds"] = relations_elapsed
    chunk_timing["source_read_seconds"] = source_read_seconds
    chunk_timing["compute_seconds"] = compute_seconds
    chunk_timing["persist_seconds"] = write_seconds
    chunk_timing["combined_compute_write_seconds"] = (
        compute_seconds + write_seconds + relations_elapsed
    )
    chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {
        "chunk_timing": chunk_timing,
        "rows_with_component": rows_with_component,
    }


def _process_and_write_subject_shape_chunk(
    source_zarr_path: str,
    *,
    output_zarr_path: str,
    refined_run: str,
    shape_run: str,
    component_indices: Sequence[tuple[str, int]],
    start_row: int,
    stop_row: int,
    chunk_index: int,
    centerline_crop_to_foreground: bool = False,
    native_threads: Optional[int] = None,
    source_camera_offsets_xy: np.ndarray | None = None,
) -> dict[str, object]:
    context_start = time.perf_counter()
    (refined_group, run_group, mask_store), cache_hit = (
        _subject_shape_worker_context(
            source_zarr_path,
            output_zarr_path=output_zarr_path,
            refined_run=refined_run,
            shape_run=shape_run,
        )
    )
    context_seconds = float(time.perf_counter() - context_start)
    with _native_thread_limit(native_threads):
        result = _process_and_write_subject_shape_chunk_groups(
            refined_group,
            run_group,
            mask_store=mask_store,
            component_indices=component_indices,
            start_row=start_row,
            stop_row=stop_row,
            chunk_index=chunk_index,
            execution_backend=DASK_WORKER_EXECUTION_BACKEND,
            centerline_crop_to_foreground=bool(centerline_crop_to_foreground),
            source_camera_offsets_xy=source_camera_offsets_xy,
        )
    chunk_timing = dict(result["chunk_timing"])
    chunk_timing["worker_context_lookup_seconds"] = context_seconds
    chunk_timing["worker_context_cache_hit"] = cache_hit
    chunk_timing["total_seconds"] = (
        float(chunk_timing.get("total_seconds") or 0.0) + context_seconds
    )
    return {
        **result,
        "chunk_timing": chunk_timing,
    }


def _compute_dask_tasks(
    tasks: Sequence[object],
    *,
    scheduler_key: str,
    num_workers: Optional[int],
) -> list[dict[str, object]]:
    if not tasks:
        return []
    cluster = None
    client = None
    try:
        if scheduler_key == "distributed":
            if not HAVE_DISTRIBUTED:
                raise RuntimeError(
                    "Dask distributed is not available. Install dask[distributed] or choose a different scheduler."
                )
            cluster_kwargs: dict[str, object] = {}
            if num_workers is not None:
                cluster_kwargs["n_workers"] = int(num_workers)
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            results = list(client.gather(client.compute(list(tasks))))
        else:
            compute_kwargs: dict[str, object] = {"scheduler": scheduler_key}
            if num_workers is not None and scheduler_key != "single-threaded":
                compute_kwargs["num_workers"] = int(num_workers)
            results = list(dask.compute(*tasks, **compute_kwargs))
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()
    return [dict(result) for result in results]


def _summarize_subject_shape_chunk_timings(
    chunk_timings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    totals = [float(item.get("total_seconds") or 0.0) for item in chunk_timings]
    numeric_keys = sorted(
        {
            str(key)
            for item in chunk_timings
            for key, value in item.items()
            if str(key).endswith("_seconds") and isinstance(value, (int, float))
        }
    )
    return {
        "chunk_count": len(chunk_timings),
        "worker_context_cache_hits": sum(
            item.get("worker_context_cache_hit") is True for item in chunk_timings
        ),
        "worker_context_cache_misses": sum(
            item.get("worker_context_cache_hit") is False for item in chunk_timings
        ),
        "mean_chunk_seconds": float(np.mean(totals)) if totals else 0.0,
        "median_chunk_seconds": float(np.median(totals)) if totals else 0.0,
        "p95_chunk_seconds": float(np.percentile(totals, 95.0)) if totals else 0.0,
        "summed_timing_seconds": {
            key: float(sum(float(item.get(key) or 0.0) for item in chunk_timings))
            for key in numeric_keys
        },
    }


def _iter_subject_shape_arrays(
    group: zarr.Group,
    prefix: str = "",
):
    for name in sorted(str(value) for value in group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(str(value) for value in group.group_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield from _iter_subject_shape_arrays(group[name], path)


def _unbound_numeric_manifest_record(run_group: zarr.Group) -> dict[str, object]:
    return build_subject_shape_unbound_numeric_manifest_record(run_group)


def _stamp_unbound_numeric_manifest(
    run_group: zarr.Group,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
):
    return stamp_unbound_subject_shape_manifest(
        run_group,
        array_content_sha256=array_content_sha256,
    )


def refresh_unbound_subject_shape_manifest_after_storage_materialization(
    run_group: zarr.Group,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Restamp the exact unbound receipt after a physical-only rewrite.

    Candidate materialization preserves every decoded logical value but adds
    planner-owned array metadata.  The retained unbound receipt must describe
    that exact staged tree before authoritative final-path binding consumes it.
    """

    state = (
        run_group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR),
        run_group.attrs.get("palette_run_completion_status"),
    )
    if (
        state
        not in {
            (SUBJECT_SHAPE_UNBOUND_STAGE_STATUS, "complete"),
            (SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS, "running"),
        }
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Storage rematerialization requires one complete, unbound, "
            "selector-ineligible subject-shape stage."
        )
    source_link = bind_persisted_coordinate_record(
        run_group,
        attr_name=SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR,
    ).record
    source_manifest = source_link.get("source_manifest")
    source_manifest_sha256 = source_link.get("source_manifest_sha256")
    if (
        source_link.get("schema_id")
        != SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_SCHEMA_ID
        or source_link.get("schema_version") != 1
        or not isinstance(source_manifest, Mapping)
        or not isinstance(source_manifest_sha256, str)
        or coordinate_record_sha256(source_manifest) != source_manifest_sha256
        or run_group.attrs.get(SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR)
        != source_manifest
        or run_group.attrs.get(f"{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}_sha256")
        != source_manifest_sha256
    ):
        raise ValueError(
            "Storage rematerialization lacks its exact original producer seal."
        )
    for name in (
        SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
        f"{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}_sha256",
    ):
        if name in run_group.attrs:
            del run_group.attrs[name]
    manifest = stamp_unbound_subject_shape_manifest(
        run_group,
        array_content_sha256=array_content_sha256,
    )
    return {
        "valid": True,
        "status": SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
        "unbound_manifest_sha256": manifest.record_sha256,
    }


def _load_unbound_numeric_manifest(
    run_group: zarr.Group,
    *,
    array_content_sha256: Mapping[str, str] | None = None,
):
    return _load_shared_unbound_manifest(
        run_group,
        array_content_sha256=array_content_sha256,
    )


def load_sealed_unbound_subject_shape_manifest(run_group: zarr.Group):
    """Bind the producer seal only when it still matches every live array.

    Physical rematerializers must call this before deriving a plan from live
    metadata.  The check closes the gap where a changed dtype, shape, payload,
    schema path, or authoritative attribute could otherwise be restamped as a
    new apparently valid unbound stage.
    """

    return _load_shared_sealed_unbound_manifest(run_group)


def _validate_unbound_subject_shape_payload(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    expected_refined_run: str,
    expected_run_name: str,
    expected_binding_status: str,
    require_complete: bool,
    expected_subject_mask_bundle_id: str | None = None,
    array_content_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    expected_path = f"analysis/subject_shape_runs/{expected_run_name}"
    if str(run_group.path) != expected_path:
        raise ValueError(
            f"Unbound subject-shape run path differs from {expected_path!r}."
        )
    expected_completion = "complete" if require_complete else "running"
    if (
        run_group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != expected_binding_status
        or run_group.attrs.get("palette_run_completion_status")
        != expected_completion
        or run_group.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Unbound subject-shape lifecycle/binding state is not exact."
        )
    owner = run_group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
    try:
        owner_is_uuid = isinstance(owner, str) and len(owner) == 32 and int(owner, 16) >= 0
    except ValueError:
        owner_is_uuid = False
    if not owner_is_uuid:
        raise ValueError("Unbound subject-shape run lacks a canonical owner UUID.")
    bundle_bound = expected_subject_mask_bundle_id is not None
    expected_schema_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION
        if bundle_bound
        else SUBJECT_SHAPE_SCHEMA_VERSION
    )
    expected_method = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD
        if bundle_bound
        else SUBJECT_SHAPE_METHOD
    )
    expected_method_version = (
        CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION
        if bundle_bound
        else SUBJECT_SHAPE_METHOD_VERSION
    )
    if (
        run_group.attrs.get("schema_id") != SUBJECT_SHAPE_SCHEMA_ID
        or run_group.attrs.get("schema_version") != expected_schema_version
        or run_group.attrs.get("method") != expected_method
        or run_group.attrs.get("method_version") != expected_method_version
        or run_group.attrs.get("source_refined_subject_masks_run")
        != expected_refined_run
    ):
        raise ValueError("Unbound subject-shape identity/configuration is invalid.")
    if bundle_bound:
        if (
            run_group.attrs.get(SUBJECT_SHAPE_SOURCE_KIND_ATTR)
            != SUBJECT_SHAPE_BUNDLE_SOURCE_KIND
            or run_group.attrs.get(SUBJECT_SHAPE_BUNDLE_ID_ATTR)
            != expected_subject_mask_bundle_id
        ):
            raise ValueError("Unbound subject-shape bundle source identity is invalid.")
        archive = archive_identity(authoritative_root)
        if archive.kind != "local_store_root":
            raise ValueError(
                "Bundle-bound unbound validation requires a local authoritative archive."
            )
        source = load_subject_shape_bundle_source(
            Path(str(archive.key[0])),
            bundle_id=expected_subject_mask_bundle_id,
            allow_inactive=True,
            assignment_keypoint_rebinding_run_id=(
                assignment_rebinding_run_id_from_source_record(
                    run_group.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_ATTR)
                )
            ),
        )
        if (
            run_group.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_ATTR)
            != source.source_record
            or run_group.attrs.get(SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR)
            != source.source_digest
            or run_group.attrs.get(SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR)
            is not source.active
        ):
            raise ValueError(
                "Unbound subject-shape source-binding receipt differs from the live bundle."
            )
    else:
        source = load_persisted_refined_subject_mask_coordinate_surfaces(
            authoritative_root,
            f"refined_subject_masks_runs/{expected_refined_run}",
        )
    projection_present = (
        SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR in run_group.attrs
        or SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR in run_group.attrs
    )
    if projection_present:
        require_subject_shape_source_camera_numeric_projection(
            run_group,
            source,
        )
    component_names = tuple(run_group.attrs.get("component_names") or ())
    if (
        len(component_names) != len(COMPONENT_ORDER)
        or len(set(component_names)) != len(component_names)
        or set(component_names) != set(COMPONENT_ORDER)
    ):
        raise ValueError("Unbound subject-shape stage lacks the full component set.")
    forbidden_children = {
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "coordinate_records",
        "component_centroid_xy",
        "component_centroid_valid",
    }
    if any(run_group.get(name) is not None for name in forbidden_children):
        raise ValueError(
            "Unbound subject-shape stage contains final-path canonical children."
        )
    forbidden_attrs = {
        "coordinate_contract",
        "subject_shape_coordinate_derivation",
        "subject_shape_publication_manifest",
        "source_row_temporal_authority",
        "row_identity_contract",
    }
    if any(name in run_group.attrs for name in forbidden_attrs):
        raise ValueError(
            "Unbound subject-shape stage contains canonical coordinate records."
        )
    for _path, node in _iter_subject_shape_arrays(run_group):
        if "coordinate_descriptor" in node.attrs:
            raise ValueError(
                "Unbound subject-shape stage contains a coordinate descriptor."
            )

    row_count = (
        int(source.row_count)
        if isinstance(source, BoundSubjectShapeBundleSource)
        else int(source.context.row_identity.leading_dimension)
    )
    for name in ("instance_key", "source_crop_row_ids"):
        staged = run_group.get(f"row_index/{name}")
        source_node = (
            source.instance_key_node
            if isinstance(source, BoundSubjectShapeBundleSource)
            and name == "instance_key"
            else source.source_crop_row_ids_node
            if isinstance(source, BoundSubjectShapeBundleSource)
            else source.context._run_group.get(name)
        )
        if staged is None or source_node is None or not np.array_equal(
            np.asarray(staged[:]),
            np.asarray(source_node[:]),
        ):
            raise ValueError(
                f"Unbound subject-shape row_index/{name} differs from exact source."
            )
    core_shapes = {
        "row_index/instance_key": (row_count,),
        "components/subject_body/centerline_xy": (
            row_count,
            CENTERLINE_SAMPLE_COUNT,
            2,
        ),
        "components/subject_body/tail_sample_xy": (
            row_count,
            TAIL_SAMPLE_COUNT,
            2,
        ),
        "body_frame/heading_deg": (row_count,),
    }
    for path, shape in core_shapes.items():
        node = run_group.get(path)
        if node is None or tuple(int(value) for value in node.shape) != shape:
            raise ValueError(f"Unbound subject-shape array {path!r} has wrong shape.")
    manifest = _load_unbound_numeric_manifest(
        run_group,
        array_content_sha256=array_content_sha256,
    )
    return {
        "valid": True,
        "status": expected_binding_status,
        "run_name": expected_run_name,
        "row_count": row_count,
        "unbound_manifest_sha256": manifest.record_sha256,
    }


def validate_unbound_subject_shape_run(
    authoritative_root: zarr.Group,
    run_group: zarr.Group,
    *,
    expected_refined_run: str,
    expected_run_name: str,
    expected_binding_status: str = SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
    require_complete: bool = True,
    expected_subject_mask_bundle_id: str | None = None,
    array_content_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate one numeric-only ROI-local stage against its exact source."""

    return _validate_unbound_subject_shape_payload(
        authoritative_root,
        run_group,
        expected_refined_run=expected_refined_run,
        expected_run_name=expected_run_name,
        expected_binding_status=expected_binding_status,
        require_complete=require_complete,
        expected_subject_mask_bundle_id=expected_subject_mask_bundle_id,
        array_content_sha256=array_content_sha256,
    )


@proof_verification_operation
def bind_staged_subject_shape_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_refined_run: str,
    expected_run_name: str,
    expected_subject_mask_bundle_id: str | None = None,
    payload_run_path: str | Path | None = None,
    payload_hash_workers: int = 4,
    unbound_array_content_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Bind and transform an unbound stage only at its authoritative path."""

    if archive_identity(authoritative_root) != archive_identity(final_run_group):
        raise ValueError(
            "Final subject-shape run is not inside the authoritative archive."
        )
    validation = _validate_unbound_subject_shape_payload(
        authoritative_root,
        final_run_group,
        expected_refined_run=expected_refined_run,
        expected_run_name=expected_run_name,
        expected_binding_status=SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS,
        require_complete=False,
        expected_subject_mask_bundle_id=expected_subject_mask_bundle_id,
        array_content_sha256=unbound_array_content_sha256,
    )
    source_revision_audit = audit_subject_shape_source_revisions_group(
        authoritative_root,
        shape_run=expected_run_name,
        refined_run=expected_refined_run,
    )
    if source_revision_audit.get("status") != "current":
        raise ValueError(
            "Refined subject-mask revisions changed before final-path binding: "
            f"{source_revision_audit!r}."
        )
    source = load_exact_subject_shape_source(authoritative_root, final_run_group)
    unbound_manifest = _load_unbound_numeric_manifest(
        final_run_group,
        array_content_sha256=unbound_array_content_sha256,
    )
    manifest_sha256 = str(validation["unbound_manifest_sha256"])
    if unbound_manifest.record_sha256 != manifest_sha256:
        raise ValueError(
            "Subject-shape unbound receipt changed between validation and consumption."
        )

    # Close the producer-sealed numeric proof before final-path authority
    # stamping. Legacy v1 stages are transformed after this boundary; projected
    # v2 stages have already written source-camera numerics and are admitted by
    # their freshly revalidated private projection receipt. In both cases, the
    # bound publication starts a distinct proof phase.
    finish_proof_verification()
    restart_proof_verification()

    records = final_run_group.require_group("coordinate_records")
    consumed_node = records.require_group("consumed_unbound_stage")
    consumed = stamp_and_bind_persisted_coordinate_record(
        consumed_node,
        unbound_manifest.record,
        attr_name=SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
    )
    if consumed.record_sha256 != manifest_sha256:
        raise ValueError(
            "Retained subject-shape unbound receipt differs from its validated digest."
        )
    del final_run_group.attrs[SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR]
    del final_run_group.attrs[f"{SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR}_sha256"]
    final_run_group.attrs["unbound_numeric_stage_manifest_sha256_consumed"] = (
        manifest_sha256
    )
    component_names = tuple(
        str(value) for value in final_run_group.attrs["component_names"]
    )
    identity, component_schema = prepare_subject_shape_identity_and_schema(
        final_run_group,
        source,
        component_names=component_names,
    )
    publication = publish_subject_shape_coordinate_surfaces(
        authoritative_root,
        final_run_group,
        source,
        component_names=component_names,
        identity=identity,
        component_schema=component_schema,
        payload_run_path=payload_run_path,
        payload_hash_workers=payload_hash_workers,
    )
    final_run_group.attrs[SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR] = (
        SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
    )
    if (
        publication.run_path
        != f"analysis/subject_shape_runs/{expected_run_name}"
        or publication.publication_owner
        != final_run_group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
        or final_run_group.attrs.get("coordinate_contract")
        != SUBJECT_SHAPE_COORDINATE_CONTRACT
    ):
        raise ValueError("Final-path subject-shape binding did not persist exactly.")
    return {
        "valid": True,
        "status": SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
        "run_name": expected_run_name,
        "row_count": int(identity.leading_dimension),
        "publication_manifest_sha256": publication.manifest.record_sha256,
        "unbound_manifest_sha256": manifest_sha256,
        "source_revision_audit": source_revision_audit,
    }


@proof_verification_operation
def complete_and_activate_bound_subject_shape_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_run_name: str,
    publication_owner: str,
) -> dict[str, object]:
    """Complete, strictly reload, and selector-last activate one bound child."""

    summary, activation = _complete_bound_subject_shape_run(
        authoritative_root,
        final_run_group,
        expected_run_name=expected_run_name,
        publication_owner=publication_owner,
        defer_eligibility=False,
    )
    if activation is not None:
        raise RuntimeError("Immediate subject-shape activation was unexpectedly deferred.")
    return summary


@proof_verification_operation
def complete_bound_subject_shape_run_for_deferred_activation(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_run_name: str,
    publication_owner: str,
) -> tuple[dict[str, object], DeferredSubjectShapeCoordinateActivation]:
    """Complete and select a child while deferring its eligibility commit."""

    summary, activation = _complete_bound_subject_shape_run(
        authoritative_root,
        final_run_group,
        expected_run_name=expected_run_name,
        publication_owner=publication_owner,
        defer_eligibility=True,
    )
    if activation is None:
        raise RuntimeError("Deferred subject-shape completion lacks a receipt.")
    return summary, activation


@proof_verification_operation
def complete_bound_subject_shape_candidate_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_run_name: str,
    publication_owner: str,
) -> dict[str, object]:
    """Complete one exact candidate without changing parent selection state."""

    if archive_identity(authoritative_root) != archive_identity(final_run_group):
        raise ValueError(
            "Bound subject-shape candidate completion is outside the authoritative archive."
        )
    if (
        str(final_run_group.path)
        != f"analysis/subject_shape_runs/{expected_run_name}"
        or final_run_group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
        or final_run_group.attrs.get("palette_run_completion_status") != "running"
        or final_run_group.attrs.get("stage_selector_eligible") is not False
        or final_run_group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
        != publication_owner
    ):
        raise ValueError(
            "Subject-shape candidate completion requires one exact bound "
            "running/ineligible child."
        )
    finish_proof_verification()
    mark_run_complete(
        final_run_group,
        parent_group=None,
        run_name=expected_run_name,
        run_provenance=build_run_provenance_from_stage_record(
            final_run_group.attrs.get("provenance", {}),
            fallback_command="subject_shape_runs",
        ),
    )
    restart_proof_verification()
    proof = validate_sealed_subject_shape_publication_metadata(
        authoritative_root,
        f"analysis/subject_shape_runs/{expected_run_name}",
        expected_selector_eligible=False,
        expected_publication_owner=publication_owner,
    )
    return {
        "valid": True,
        "status": SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
        "run_name": expected_run_name,
        "row_count": proof.row_count,
        "publication_manifest_sha256": proof.manifest.record_sha256,
        "selector_state": "unchanged_candidate_ineligible",
    }


def _complete_bound_subject_shape_run(
    authoritative_root: zarr.Group,
    final_run_group: zarr.Group,
    *,
    expected_run_name: str,
    publication_owner: str,
    defer_eligibility: bool,
) -> tuple[
    dict[str, object],
    DeferredSubjectShapeCoordinateActivation | None,
]:
    """Complete one bound child and prepare its guarded selector epoch."""

    if archive_identity(authoritative_root) != archive_identity(final_run_group):
        raise ValueError(
            "Bound subject-shape completion is outside the authoritative archive."
        )
    if (
        str(final_run_group.path)
        != f"analysis/subject_shape_runs/{expected_run_name}"
        or final_run_group.attrs.get(SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR)
        != SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
        or final_run_group.attrs.get("palette_run_completion_status") != "running"
        or final_run_group.attrs.get("stage_selector_eligible") is not False
        or final_run_group.attrs.get(SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR)
        != publication_owner
    ):
        raise ValueError(
            "Subject-shape completion requires one exact bound running/ineligible child."
        )

    # Close every source/output proof collected while the child was running
    # before changing its lifecycle state.  A completed-child publication is
    # then established in a fresh phase and activation closes that phase before
    # touching parent selectors.
    finish_proof_verification()
    parent = authoritative_root["analysis/subject_shape_runs"]
    mark_run_complete(
        final_run_group,
        parent_group=None,
        run_name=expected_run_name,
        run_provenance=build_run_provenance_from_stage_record(
            final_run_group.attrs.get("provenance", {}),
            fallback_command="subject_shape_runs",
        ),
    )
    restart_proof_verification()
    proof = validate_sealed_subject_shape_publication_metadata(
        authoritative_root,
        f"analysis/subject_shape_runs/{expected_run_name}",
        expected_selector_eligible=False,
        expected_publication_owner=publication_owner,
    )
    activation_snapshot = selector_snapshot(parent)
    activation = activate_subject_shape_coordinate_publication(
        authoritative_root,
        parent,
        proof,
        run_name=expected_run_name,
        owner=publication_owner,
        snapshot=activation_snapshot,
        defer_eligibility=defer_eligibility,
    )
    return (
        {
            "valid": True,
            "status": SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
            "run_name": expected_run_name,
            "row_count": proof.row_count,
            "publication_manifest_sha256": proof.manifest.record_sha256,
        },
        activation,
    )


@proof_verification_operation
def write_subject_shape_run_group(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    output_root: zarr.Group | None = None,
    output_zarr_path: str | Path | None = None,
    refined_run: Optional[str] = None,
    run_name: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    execution_backend: str = SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    include_chunk_timings: bool = False,
    centerline_crop_to_foreground: bool = False,
    native_threads: Optional[int] = None,
    stage_command: Optional[str] = None,
    subject_mask_bundle_id: str | None = None,
    allow_inactive_subject_mask_bundle: bool = False,
    assignment_keypoint_rebinding_run_id: str | None = None,
    payload_scan_workers: int = 1,
    payload_scan_block_rows: int | None = None,
    _unbound_coordinate_stage: bool = False,
) -> dict[str, object]:
    """Write one row-aligned subject-shape analysis run."""

    scheduler_key = _normalize_scheduler(scheduler)
    backend = _normalize_execution_backend(execution_backend)
    destination_root = output_root if output_root is not None else root
    destination_zarr_path = output_zarr_path if output_zarr_path is not None else zarr_path
    if destination_root is not root and not _unbound_coordinate_stage:
        raise ValueError(
            "Canonical subject-shape publication must remain in the same archive "
            "as its exact refined-mask authority; cross-archive output is unsupported."
        )
    if _unbound_coordinate_stage and destination_root is root:
        raise ValueError(
            "An unbound subject-shape numeric stage must use a separate scratch archive."
        )
    if backend == DASK_WORKER_EXECUTION_BACKEND and (zarr_path is None or destination_zarr_path is None):
        raise ValueError(
            "execution_backend='dask_worker_chunks' requires filesystem source and output Zarr paths."
        )
    if type(allow_inactive_subject_mask_bundle) is not bool:
        raise TypeError("allow_inactive_subject_mask_bundle must be an exact bool.")
    if type(payload_scan_workers) is not int or payload_scan_workers <= 0:
        raise ValueError("payload_scan_workers must be a positive integer.")
    if payload_scan_block_rows is not None and (
        type(payload_scan_block_rows) is not int or payload_scan_block_rows <= 0
    ):
        raise ValueError("payload_scan_block_rows must be a positive integer.")
    if subject_mask_bundle_id is not None and not _unbound_coordinate_stage:
        raise ValueError(
            "Recording-bundle subject-shape v5 must be finalized through the "
            "access-aware materializer; direct legacy-layout publication is forbidden."
        )
    bundle_source: BoundSubjectShapeBundleSource | None = None
    if subject_mask_bundle_id is not None:
        archive = archive_identity(root)
        if archive.kind != "local_store_root":
            raise ValueError(
                "Recording-bundle subject-shape computation requires a local Zarr archive."
            )
        bundle_source = load_subject_shape_bundle_source(
            Path(str(archive.key[0])),
            bundle_id=str(subject_mask_bundle_id),
            allow_inactive=allow_inactive_subject_mask_bundle,
            assignment_keypoint_rebinding_run_id=(
                assignment_keypoint_rebinding_run_id
            ),
        )
        refined_run_path = bundle_source.authority.refined_run_path
        prefix = "refined_subject_masks_runs/"
        if not refined_run_path.startswith(prefix) or "/" in refined_run_path[len(prefix) :]:
            raise ValueError("Subject-mask bundle refined member path is invalid.")
        refined_run_name = refined_run_path[len(prefix) :]
        if refined_run is not None and str(refined_run) != refined_run_name:
            raise ValueError(
                "Explicit refined_run differs from the selected subject-mask bundle member."
            )
        refined_group = bundle_source.authority.refined_run
        coordinate_source = bundle_source
    else:
        if allow_inactive_subject_mask_bundle:
            raise ValueError(
                "allow_inactive_subject_mask_bundle requires subject_mask_bundle_id."
            )
        refined_run_name, refined_group = _resolve_refined_run(root, refined_run)
        refined_coordinate_source = (
            load_persisted_refined_subject_mask_coordinate_surfaces(
                root,
                f"refined_subject_masks_runs/{refined_run_name}",
            )
        )
        if refined_coordinate_source.context._run_group.path != refined_group.path:
            raise ValueError(
                "Logical refined-mask selection differs from canonical coordinate authority."
            )
        coordinate_source = refined_coordinate_source
    refined_tables = load_refined_subject_masks_run_tables(
        root,
        run_name=refined_run_name,
        component_names=components,
        include_masks_roi=True,
        include_metrics=False,
        include_components=False,
        include_relations=False,
    )
    component_indices = _resolve_components_from_refined_tables(refined_tables, components)
    selected_component_names = tuple(name for name, _idx in component_indices)
    missing_required = sorted(set(COMPONENT_ORDER) - set(selected_component_names))
    if missing_required:
        raise ValueError(
            "Canonical subject-shape publication requires the full component/body-frame "
            f"anchor set; missing {missing_required!r}."
        )
    mask_store = refined_tables.require_mask_store()
    total_rows = int(mask_store.n_rows)
    target_run = str(run_name or _default_run_name())
    requested_chunk_size = max(1, int(chunk_size))
    worker_chunk_size = _worker_chunk_size_for_backend(total_rows, requested_chunk_size, backend)
    chunks = _row_chunks(total_rows, worker_chunk_size)
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "source_refined_subject_masks_run": refined_run_name,
        "subject_shape_run": target_run,
        "component_names": [name for name, _idx in component_indices],
        "roi_count": total_rows,
        "chunk_size": requested_chunk_size,
        "worker_chunk_size": worker_chunk_size,
        "chunk_count": len(chunks),
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_requested_chunk_size": requested_chunk_size,
        "dask_chunk_size": worker_chunk_size,
        "dask_chunk_alignment": (
            REFINED_SUBJECT_MASK_DASK_CHUNK_ALIGNMENT
            if backend == DASK_WORKER_EXECUTION_BACKEND
            else "requested_chunk_size"
        ),
        "worker_context_cache_policy": (
            "exact_store_metadata_identity_process_local_v1"
            if backend == DASK_WORKER_EXECUTION_BACKEND
            else "not_used_serial_driver"
        ),
        "centerline_crop_to_foreground": bool(centerline_crop_to_foreground),
        "native_threads_per_worker": (
            max(1, int(native_threads)) if native_threads is not None else None
        ),
        "payload_scan_workers": int(payload_scan_workers),
        "payload_scan_block_rows": (
            int(payload_scan_block_rows)
            if payload_scan_block_rows is not None
            else int(worker_chunk_size)
        ),
        "separate_output_root": destination_root is not root,
        "mutates_archive": not bool(dry_run),
        "coordinate_contract": (
            "unbound_numeric_stage"
            if _unbound_coordinate_stage
            else "canonical_v2"
        ),
        "subject_mask_bundle_id": (
            bundle_source.bundle_id if bundle_source is not None else None
        ),
        "subject_mask_bundle_active": (
            bundle_source.active if bundle_source is not None else None
        ),
        "assignment_keypoint_rebinding_run_id": (
            bundle_source.assignment_keypoint_rebinding_run_id
            if bundle_source is not None
            else None
        ),
        "point_coordinate_space": (
            "source_camera_image_px_precanonical_numeric"
            if _unbound_coordinate_stage
            else "source_camera_image_px"
        ),
    }
    if dry_run:
        return summary

    stage_start = time.perf_counter()
    command = stage_command or (" ".join(sys.argv) if sys.argv else "unknown")
    publication_owner = uuid.uuid4().hex
    run_group = _prepare_subject_shape_run(
        destination_root,
        target_run=target_run,
        refined_run_name=refined_run_name,
        refined_group=refined_group,
        source_mask_store=mask_store,
        source_mask_store_path=refined_tables.source_paths.get("mask_store"),
        component_indices=component_indices,
        requested_chunk_size=requested_chunk_size,
        worker_chunk_size=worker_chunk_size,
        execution_backend=backend,
        scheduler=scheduler_key,
        num_workers=num_workers,
        centerline_crop_to_foreground=bool(centerline_crop_to_foreground),
        native_threads=native_threads,
        stage_command=command,
        publication_owner=publication_owner,
        write_best_effort_lineage=not _unbound_coordinate_stage,
        overwrite=overwrite,
        bundle_source=bundle_source,
    )
    source_camera_offsets_xy: np.ndarray | None = None
    if _unbound_coordinate_stage:
        source_camera_offsets_xy = (
            stamp_subject_shape_source_camera_numeric_projection(
                run_group,
                coordinate_source,
                component_names=selected_component_names,
            )
        )
    chunk_timings: list[dict[str, object]] = []
    rows_with_component: dict[str, int] = {name: 0 for name, _idx in component_indices}
    if backend == DASK_WORKER_EXECUTION_BACKEND:
        assert zarr_path is not None
        tasks = [
            delayed(_process_and_write_subject_shape_chunk)(
                str(zarr_path),
                output_zarr_path=str(destination_zarr_path),
                refined_run=refined_run_name,
                shape_run=target_run,
                component_indices=tuple(component_indices),
                start_row=start_row,
                stop_row=stop_row,
                chunk_index=chunk_index,
                centerline_crop_to_foreground=bool(centerline_crop_to_foreground),
                native_threads=native_threads,
                source_camera_offsets_xy=(
                    source_camera_offsets_xy[start_row:stop_row]
                    if source_camera_offsets_xy is not None
                    else None
                ),
            )
            for chunk_index, (start_row, stop_row) in enumerate(chunks)
        ]
        results = _compute_dask_tasks(tasks, scheduler_key=scheduler_key, num_workers=num_workers)
        for result in sorted(results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
            chunk_timings.append(dict(result["chunk_timing"]))
            for component_name, count in dict(result.get("rows_with_component") or {}).items():
                rows_with_component[str(component_name)] = int(rows_with_component.get(str(component_name), 0)) + int(count)
    else:
        for chunk_index, (start_row, stop_row) in enumerate(chunks):
            with _native_thread_limit(native_threads):
                result = _process_and_write_subject_shape_chunk_groups(
                    refined_group,
                    run_group,
                    mask_store=mask_store,
                    component_indices=tuple(component_indices),
                    start_row=start_row,
                    stop_row=stop_row,
                    chunk_index=chunk_index,
                    execution_backend=SERIAL_EXECUTION_BACKEND,
                    centerline_crop_to_foreground=bool(centerline_crop_to_foreground),
                    source_camera_offsets_xy=(
                        source_camera_offsets_xy[start_row:stop_row]
                        if source_camera_offsets_xy is not None
                        else None
                    ),
                )
            chunk_timing = dict(result["chunk_timing"])
            chunk_timings.append(chunk_timing)
            for component_name, count in dict(result.get("rows_with_component") or {}).items():
                rows_with_component[str(component_name)] = int(rows_with_component.get(str(component_name), 0)) + int(count)

    duration_seconds = float(time.perf_counter() - stage_start)
    rows_per_second = float(total_rows / duration_seconds) if duration_seconds > 0.0 else float("inf")
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["rows_per_second"] = rows_per_second
    run_group.attrs["rows_with_component"] = rows_with_component
    run_group.attrs["subject_shape_timing_summary"] = {
        "total_rows": total_rows,
        "duration_seconds": duration_seconds,
        "rows_per_second": rows_per_second,
        "execution_backend": backend,
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_chunk_size": max(1, int(chunk_size)),
        "worker_context_cache_policy": (
            "exact_store_metadata_identity_process_local_v1"
            if backend == DASK_WORKER_EXECUTION_BACKEND
            else "not_used_serial_driver"
        ),
        "centerline_crop_to_foreground": bool(centerline_crop_to_foreground),
        "native_threads_per_worker": (
            max(1, int(native_threads)) if native_threads is not None else None
        ),
        "dask_version": getattr(dask, "__version__", "unknown"),
        "chunk_timings": _summarize_subject_shape_chunk_timings(chunk_timings),
    }
    run_group.attrs["subject_shape_chunk_timing_count"] = len(chunk_timings)
    run_group.attrs["subject_shape_chunk_timing_storage"] = (
        "embedded_full_records" if include_chunk_timings else "summary_only"
    )
    if include_chunk_timings:
        run_group.attrs["subject_shape_chunk_timings"] = list(_json_safe(chunk_timings))
    if _unbound_coordinate_stage:
        run_group.attrs[SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR] = (
            SUBJECT_SHAPE_UNBOUND_STAGE_STATUS
        )
        # An unbound scratch artifact has no selector activation phase. Close
        # its reused source proofs while the child is still running so a
        # failed closing recheck cannot leave a newly completed artifact.
        finish_proof_verification()
        mark_run_complete(
            run_group,
            parent_group=None,
            run_name=target_run,
            run_provenance=build_run_provenance_from_stage_record(
                run_group.attrs.get("provenance", {}),
                fallback_command="subject_shape_runs_unbound_stage",
            ),
        )
        payload_scan = build_subject_shape_unbound_payload_scan_receipt(
            run_group,
            workers=payload_scan_workers,
            block_rows=(
                payload_scan_block_rows
                if payload_scan_block_rows is not None
                else worker_chunk_size
            ),
        )
        payload_hashes = dict(payload_scan["array_content_sha256"])
        _stamp_unbound_numeric_manifest(
            run_group,
            array_content_sha256=payload_hashes,
        )
        validation = _validate_unbound_subject_shape_payload(
            root,
            run_group,
            expected_refined_run=refined_run_name,
            expected_run_name=target_run,
            expected_binding_status=SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
            require_complete=True,
            expected_subject_mask_bundle_id=(
                bundle_source.bundle_id if bundle_source is not None else None
            ),
            array_content_sha256=payload_hashes,
        )
        decoded_payload = dict(payload_scan["decoded_payload"])
        summary.update(
            {
                "status": SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
                "coordinate_binding_status": SUBJECT_SHAPE_UNBOUND_STAGE_STATUS,
                "unbound_validation": validation,
                "duration_seconds": duration_seconds,
                "rows_per_second": rows_per_second,
                "rows_with_component": rows_with_component,
                "chunk_timing_count": len(chunk_timings),
                "producer_payload_scan_receipt": {
                    "schema_id": payload_scan["schema_id"],
                    "schema_version": payload_scan["schema_version"],
                    "run_ref": payload_scan["run_ref"],
                    "array_payload_canonicalization": payload_scan[
                        "array_payload_canonicalization"
                    ],
                    "closed_array_inventory": payload_scan[
                        "closed_array_inventory"
                    ],
                    "mutation_exclusion": payload_scan["mutation_exclusion"],
                    "requested_workers": payload_scan["requested_workers"],
                    "effective_workers": payload_scan["effective_workers"],
                    "block_rows": payload_scan["block_rows"],
                    "duration_seconds": payload_scan["duration_seconds"],
                    "array_count": decoded_payload["array_count"],
                    "decoded_bytes": decoded_payload["decoded_bytes"],
                    "decoded_payload_root_sha256": decoded_payload["root_sha256"],
                },
            }
        )
        if include_chunk_timings:
            summary["chunk_timings"] = list(_json_safe(chunk_timings))
        return dict(_json_safe(summary))

    run_group.attrs[SUBJECT_SHAPE_COORDINATE_BINDING_STATUS_ATTR] = (
        SUBJECT_SHAPE_PUBLISHING_BINDING_STATUS
    )
    _stamp_unbound_numeric_manifest(run_group)
    try:
        bind_staged_subject_shape_run(
            root,
            run_group,
            expected_refined_run=refined_run_name,
            expected_run_name=target_run,
            expected_subject_mask_bundle_id=(
                bundle_source.bundle_id if bundle_source is not None else None
            ),
        )
    except BaseException as exc:
        try:
            mark_run_failed(
                run_group,
                parent_group=None,
                run_name=target_run,
                error=f"coordinate publication failed: {exc}",
            )
        except BaseException:
            pass
        raise
    complete_and_activate_bound_subject_shape_run(
        root,
        run_group,
        expected_run_name=target_run,
        publication_owner=publication_owner,
    )
    summary.update(
        {
            "status": "updated",
            "duration_seconds": duration_seconds,
            "rows_per_second": rows_per_second,
            "rows_with_component": rows_with_component,
            "chunk_timing_count": len(chunk_timings),
        }
    )
    if include_chunk_timings:
        summary["chunk_timings"] = list(_json_safe(chunk_timings))
    return dict(_json_safe(summary))


def write_subject_shape_run(
    zarr_path: str | Path,
    *,
    refined_run: Optional[str] = None,
    run_name: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    execution_backend: str = SERIAL_EXECUTION_BACKEND,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    include_chunk_timings: bool = False,
    centerline_crop_to_foreground: bool = False,
    native_threads: Optional[int] = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run=refined_run,
        run_name=run_name,
        components=components,
        chunk_size=chunk_size,
        execution_backend=execution_backend,
        scheduler=scheduler,
        num_workers=num_workers,
        overwrite=overwrite,
        dry_run=dry_run,
        include_chunk_timings=include_chunk_timings,
        centerline_crop_to_foreground=centerline_crop_to_foreground,
        native_threads=native_threads,
    )


def audit_subject_shape_source_revisions(
    zarr_path: str | Path,
    *,
    shape_run: Optional[str] = None,
    refined_run: Optional[str] = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    return dict(_json_safe(audit_subject_shape_source_revisions_group(root, shape_run=shape_run, refined_run=refined_run)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write analysis/subject_shape_runs from refined subject-mask components."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--refined-run", help="refined_subject_masks_runs/<run> to consume; defaults to latest.")
    parser.add_argument("--run-name", help="Target analysis/subject_shape_runs/<run>; defaults to timestamped name.")
    parser.add_argument("--shape-run", help="Existing analysis/subject_shape_runs/<run> for audit commands.")
    parser.add_argument(
        "--audit-source-revisions",
        action="store_true",
        help="Read-only check for refined subject-mask row_revision drift in an existing subject-shape run.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENT_ORDER,
        help="Optional component subset. Defaults to all available known subject-shape components.",
    )
    parser.add_argument("--component", action="append", dest="component_values", choices=COMPONENT_ORDER)
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of refined rows per worker chunk.")
    parser.add_argument(
        "--execution-backend",
        choices=EXECUTION_BACKENDS,
        default=SERIAL_EXECUTION_BACKEND,
        help="Use dask_worker_chunks to let Dask workers write disjoint row chunks.",
    )
    parser.add_argument(
        "--scheduler",
        type=_normalize_scheduler,
        choices=SUPPORTED_SCHEDULERS,
        default="single-threaded",
        help="Dask scheduler used when --execution-backend=dask_worker_chunks.",
    )
    parser.add_argument("--num-workers", type=int, help="Dask worker count.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target subject-shape run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs without mutating the archive.")
    parser.add_argument(
        "--include-chunk-timings",
        action="store_true",
        help=(
            "Include full per-chunk timing records in stdout and Zarr attrs. "
            "Production runs otherwise persist only a compact timing summary."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def _parse_components(values: Optional[Sequence[str]], repeated: Optional[Sequence[str]]) -> Optional[list[str]]:
    result: list[str] = []
    if values:
        result.extend(str(value) for value in values)
    if repeated:
        result.extend(str(value) for value in repeated)
    return result or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.audit_source_revisions):
        summary = audit_subject_shape_source_revisions(
            args.zarr_path,
            shape_run=args.shape_run or args.run_name,
            refined_run=args.refined_run,
        )
        print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
        return 0
    summary = write_subject_shape_run(
        args.zarr_path,
        refined_run=args.refined_run,
        run_name=args.run_name,
        components=_parse_components(args.components, args.component_values),
        chunk_size=int(args.chunk_size),
        execution_backend=args.execution_backend,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        include_chunk_timings=bool(args.include_chunk_timings),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

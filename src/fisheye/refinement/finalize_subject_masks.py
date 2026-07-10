"""Create refined subject-mask candidates from raw subject-mask outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import cv2
import numpy as np
import zarr

from ..shared.detect_reason_codec import read_reason_labels, write_reason_columns
from ..shared.json_safety import json_attr_safe
from ..shared.mask_geometry import batch_mask_spatial_metrics
from ..shared.mask_probability_encoding import decode_probability_values
from ..shared.mask_bitpack import (
    MASK_BITPACKED_AXIS,
    MASK_BITPACKED_BITORDER,
    MASK_BITPACKED_ENCODING,
    MASK_BITPACKED_LAYOUT,
    MASK_BITPACKED_SCHEMA_ID,
    MASK_BITPACKED_VALUE_SEMANTICS,
    pack_binary_mask_stack,
    packed_width_bytes,
)
from ..shared.provenance_attrs import (
    ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
    build_assignment_keypoint_attrs,
    build_source_keypoints_attrs,
)
from ..shared.row_lineage import copy_row_lineage_arrays_from_sources
from ..shared.run_provenance import build_run_provenance_from_stage_record
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.mask_store import (
    DENSE_MASK_ENCODING_V1,
    MASK_RLE_VALIDATION_MODES,
    MaskStoreError,
    open_mask_store,
    update_mask_storage_attrs,
    validate_bitpacked_mask_store_invariants,
    write_bitpacked_mask_store_from_dense,
    write_component_rle_mask_store_from_dense,
)
from ..shared.subject_mask_chunks import (
    refined_subject_mask_bitpacked_chunks,
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_row_chunk,
    refined_subject_mask_storage_chunks,
)
from ..shared.subject_mask_registry_status import emit_refined_subject_mask_stage_completion
from ..shared.zarr_run_completion import mark_run_complete, require_runs_parent
from ..tune.refined_subject_mask_review import (
    DEFAULT_REVIEW_INTENDED_USE,
    DEFAULT_REVIEW_METHOD,
    DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE,
    DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID,
    DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY,
    REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID,
    REFINED_SUBJECT_STAGE_NAME,
    SourceSubjectMaskRun,
    _compute_component_curvature_var_metrics,
    _compute_component_shape_qc_metrics,
    _compute_component_sigma_noise_metrics,
    _compute_component_topology_metrics,
    _compute_mask_row_fingerprints,
    _default_refined_run_name,
    _ensure_refined_component_provenance_payload,
    _infer_refined_label_schema_id,
    _load_source_subject_mask_run,
    _normalize_component_name,
    _probability_encoding_for_group,
    _probability_thresholds_for_labels,
    _review_payload,
    _source_component_provenance_payload,
)
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root
from .assemble_refined_subject_masks import (
    CANONICAL_COMPONENT_ORDER,
    _has_available_component,
    _require_available_component,
    _resolve_eye_keypoint_indices,
    _resolve_keypoint_success_array,
    _resolve_subject_keypoint_group,
)
from ..shared.refined_subject_eye_geometry import write_refined_subject_eye_geometry
from ..shared.refined_subject_eye_geometry import EYE_GEOMETRY_SCHEMA_ID
from ..shared.refined_subject_eye_geometry import EYE_PAIR_RELATION_SCHEMA_ID
from ..shared.refined_subject_component_contours import (
    COMPONENT_CONTOUR_SCHEMA_ID,
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    DEFAULT_SAMPLED_CONTOUR_COUNTS,
    DEFAULT_SAMPLED_CONTOUR_ROW_CHUNK,
    extract_largest_external_contour,
    resample_closed_contour,
    write_refined_subject_component_contours,
    write_refined_subject_sampled_component_contours,
    write_sampled_component_contour_arrays,
)
from .subject_eye_assignment import (
    EYES_UNION_ASSIGNMENT_METHOD,
    EyesUnionAssignmentResult,
    assign_eyes_union_to_lr,
    reconcile_keypoint_mask_row_identity,
)
from .subject_mask_finalization import (
    ComponentFinalizationPolicy,
    _default_policy_for_component,
    finalize_component_mask,
)

SMART_FINALIZE_SUBJECT_MASKS_METHOD = "smart_finalize_subject_masks_v1"
SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA = "palette_subject_mask_shard_collection_finalizer_v1"
SUBJECT_MASK_CANONICAL_PARENT = "subject_mask_runs"
SUBJECT_MASK_SHARD_PARENT = "subject_mask_shard_runs"
_REFINED_SUBJECT_MASKS_STATUS_SOURCE = "runtime_smart_finalize_subject_masks"
_RAW_EYE_UNION_COMPONENT = "eyes_union"
_EYE_COMPONENTS = ("eye_left", "eye_right")
_COMPONENT_CONTOUR_COMPONENTS = ("subject_body", "swim_bladder")
_FINALIZABLE_RAW_COMPONENTS = ("subject_body", "swim_bladder", _RAW_EYE_UNION_COMPONENT)
_METRIC_LEVELS = ("cheap", "full")
_EXECUTION_BACKENDS = ("serial_driver", "process_shards")
_SERIAL_EXECUTION_BACKEND = "serial_driver"
_PROCESS_SHARD_EXECUTION_BACKEND = "process_shards"
_POSTCOMPUTE_BACKENDS = ("serial", "process_shards")
_SERIAL_POSTCOMPUTE_BACKEND = "serial"
_PROCESS_SHARD_POSTCOMPUTE_BACKEND = "process_shards"
_ASSIGNMENT_REUSE_POSTCOMPUTE_BACKEND = "assignment_reuse"
_ASSIGNMENT_REUSE_EYE_GEOMETRY_SOURCE = "eyes_union_assignment_measure_mask"
_POSTCOMPUTE_EYE_GEOMETRY_SOURCE = "refined_subject_component_mask_measure_mask"
_EYE_GEOMETRY_REUSE_STATUS_NOT_REQUESTED = "not_requested"
_EYE_GEOMETRY_REUSE_STATUS_ASSIGNMENT_REUSE = "assignment_reuse"
_EYE_GEOMETRY_REUSE_STATUS_FALLBACK_REFIT = "fallback_refit"
_EYE_GEOMETRY_FALLBACK_WARNING_CODE = "EYE_GEOMETRY_FALLBACK_REFIT"
_MASK_STORAGE_CHOICES = (
    "dense_uint8",
    "dense_and_bitpacked",
    "bitpacked_v1",
    "dense_and_rle",
    "rle_v1",
    "dense_bitpacked_and_rle",
)
_MASK_STORAGE_COMPACT_ONLY = {"bitpacked_v1", "rle_v1"}
_MASK_STORAGES_WITH_BITPACKED = {"dense_and_bitpacked", "dense_bitpacked_and_rle"}
_MASK_STORAGES_WITH_RLE = {"dense_and_rle", "dense_bitpacked_and_rle"}
_MASK_STORAGES_REMOVE_DENSE: set[str] = set()
_MASK_STORAGES_DIRECT_BITPACKED: set[str] = set()
_MASK_RLE_VALIDATION_MODES = MASK_RLE_VALIDATION_MODES
_COMPONENT_METRICS_SCHEMA_ID = "refined_subject_component_mask_metrics_v1"
_COMPONENT_METRIC_QC_SCHEMA_ID = "refined_subject_component_metric_qc_reasons_v1"
_SOURCE_SEED_MASKS_SCHEMA_ID = "refined_subject_component_source_seed_masks_v1"
_COMPONENT_QC_REASON_PREFIX = "needs_review_metric_"
_COMPONENT_METRIC_NAMES = (
    "component_count",
    "largest_component_fraction",
    "hole_count",
    "hole_area_fraction",
    "sigma_noise",
    "curvature_var",
    "ipr",
    "solidity",
)
_FINALIZATION_METRIC_NAMES = (
    "added_area_px",
    "area_px_after",
    "area_px_before",
    "changed_area_fraction",
    "changed_area_px",
    "component_count_after",
    "component_count_before",
    "hole_area_fraction_after",
    "hole_area_fraction_before",
    "hole_count_after",
    "hole_count_before",
    "largest_component_fraction_after",
    "largest_component_fraction_before",
    "removed_area_fraction",
    "removed_area_px",
    "removed_component_count",
    "removed_high_prob_area_px",
    "removed_prob_mass",
    "removed_prob_mass_fraction",
)
FINALIZATION_METRIC_ROW_CHUNK = 16384
FINALIZATION_METRIC_WRITE_POLICY = "driver_merged_sealed_v1"
COMMON_DERIVED_METRIC_ROW_CHUNK = 16384
COMMON_DERIVED_METRIC_WRITE_POLICY = "driver_merged_sealed_v1"
_COMPONENT_METRIC_FINALIZATION_SOURCES = {
    "component_count": ("component_count_after", np.int32),
    "largest_component_fraction": ("largest_component_fraction_after", np.float32),
    "hole_count": ("hole_count_after", np.int32),
    "hole_area_fraction": ("hole_area_fraction_after", np.float32),
}
_CROP_REBASE_IDENTITY_ARRAYS = (
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "frame_indices",
    "roi_coordinates_full",
)
_CROP_REBASE_COPY_ARRAYS = (
    "frame_indices",
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "detection_indices",
)
_SUBJECT_MASK_SHARD_COMPAT_ATTRS = (
    "mask_labels",
    "label_schema_id",
    "probabilities_encoding",
    "mask_probability_threshold",
    "thresholds_by_label",
    "threshold_by_component",
    "threshold_by_label",
)


@dataclass(frozen=True)
class _FinalizedComponentBatch:
    component_name: str
    masks: np.ndarray
    source_masks: np.ndarray
    reason_labels: np.ndarray
    quality_code: np.ndarray
    quality_score: np.ndarray
    review_recommendation: np.ndarray
    metrics: dict[str, np.ndarray]
    policy: ComponentFinalizationPolicy
    source_surface_path: str
    source_surface_kind: str
    source_probability_encoding: Optional[str]
    source_probability_threshold: float


@dataclass(frozen=True)
class _SubjectMaskShardSource:
    name: str
    source: SourceSubjectMaskRun
    source_crop_run: str
    row_count: int


@dataclass(frozen=True)
class _SubjectMaskShardCollection:
    shard_runs: tuple[str, ...]
    shard_run_paths: tuple[str, ...]
    shard_crop_runs: tuple[str, ...]
    source_crop_run: str
    source_crop_rebased_from_shards: bool
    row_source_indices: np.ndarray
    row_local_indices: np.ndarray
    source_crop_row_ids: np.ndarray


class _SubjectMaskParentAliasRoot:
    """Root proxy exposing shard runs as ``subject_mask_runs`` for the legacy loader."""

    def __init__(self, root: zarr.Group, source_parent: str) -> None:
        self._root = root
        self._source_parent = str(source_parent)

    def get(self, path: str, default: object | None = None) -> object | None:
        path = str(path)
        if path == SUBJECT_MASK_CANONICAL_PARENT:
            return self._root.get(self._source_parent, default)
        prefix = f"{SUBJECT_MASK_CANONICAL_PARENT}/"
        if path.startswith(prefix):
            return self._root.get(f"{self._source_parent}/{path[len(prefix):]}", default)
        return self._root.get(path, default)


class _IndexedCollectionArray:
    """Array-like row view over shard arrays in finalized collection order."""

    def __init__(
        self,
        arrays: Sequence[object],
        *,
        source_indices: np.ndarray,
        local_indices: np.ndarray,
    ) -> None:
        if not arrays:
            raise ValueError("Indexed collection array requires at least one source array.")
        self._arrays = tuple(arrays)
        self._source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        self._local_indices = np.asarray(local_indices, dtype=np.int64).reshape(-1)
        if self._source_indices.shape != self._local_indices.shape:
            raise ValueError("source_indices and local_indices must have the same shape.")
        first = arrays[0]
        first_shape = tuple(int(value) for value in getattr(first, "shape"))
        self.shape = (int(self._source_indices.shape[0]), *first_shape[1:])
        self.dtype = np.dtype(getattr(first, "dtype"))
        self.ndim = len(self.shape)
        chunks = getattr(first, "chunks", None)
        self.chunks = None
        if chunks:
            chunks_tuple = tuple(int(value) for value in chunks)
            self.chunks = (min(max(1, self.shape[0]), chunks_tuple[0]), *chunks_tuple[1:])

    def __getitem__(self, key: object) -> np.ndarray:
        if isinstance(key, tuple):
            row_key = key[0]
            rest = key[1:]
        else:
            row_key = key
            rest = ()
        scalar_row = isinstance(row_key, (int, np.integer))
        positions = np.arange(self.shape[0], dtype=np.int64)[row_key]
        positions = (
            np.asarray([positions], dtype=np.int64)
            if np.ndim(positions) == 0
            else np.asarray(positions, dtype=np.int64)
        )
        rows: list[np.ndarray] = []
        for position in positions.reshape(-1):
            source_idx = int(self._source_indices[int(position)])
            local_idx = int(self._local_indices[int(position)])
            rows.append(np.asarray(self._arrays[source_idx][(local_idx, *rest)]))
        if not rows:
            return np.empty((0, *self.shape[1 + len(rest):]), dtype=self.dtype)
        result = np.stack(rows, axis=0)
        return result[0] if scalar_row else result


class _SubjectMaskCollectionGroup:
    """Minimal group-like object satisfying the finalizer's source-run contract."""

    def __init__(self, attrs: Mapping[str, object], arrays: Mapping[str, object], *, path: str) -> None:
        self.attrs = dict(attrs)
        self._arrays = dict(arrays)
        self.path = str(path)

    def get(self, name: str, default: object | None = None) -> object | None:
        return self._arrays.get(str(name), default)

    def __getitem__(self, name: str) -> object:
        return self._arrays[str(name)]

    def __contains__(self, name: str) -> bool:
        return str(name) in self._arrays


@dataclass(frozen=True)
class _ComponentMetricQcPolicy:
    component_name: str
    min_area_px: float = 1.0
    max_component_count: int = 1
    max_hole_count: int = 0
    min_largest_component_fraction: float = 0.90
    min_solidity: Optional[float] = None


@dataclass(frozen=True)
class _ComponentMetricWriteResult:
    mask_present: np.ndarray
    reason_labels: np.ndarray


@dataclass(frozen=True)
class _ComponentMetricPayload:
    spatial_metrics: dict[str, np.ndarray]
    component_metrics: dict[str, np.ndarray]
    reason_labels: np.ndarray


@dataclass(frozen=True)
class _CanonicalComponentChunkResult:
    mask_present: np.ndarray
    reason_labels: np.ndarray
    spatial_metrics: dict[str, np.ndarray]
    component_metrics: dict[str, np.ndarray]
    source_row_fingerprint: np.ndarray


@dataclass(frozen=True)
class _EyeAssignmentContext:
    keypoints_roi: Any
    keypoint_success: np.ndarray
    eye_keypoint_indices: tuple[int, int]
    keypoint_run_name: str
    keypoint_group_name: str
    keypoint_success_dataset: str
    keypoint_source_kind: str
    row_identity_summary: Mapping[str, object]


@dataclass(frozen=True)
class _EyeAssignmentChunk:
    masks: dict[str, np.ndarray]
    reason_labels: dict[str, np.ndarray]
    component_metrics: dict[str, dict[str, np.ndarray]]
    summary: dict[str, object]
    phase_seconds: dict[str, float]
    eye_geometry: dict[str, object] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_execution_backend(execution_backend: object) -> str:
    backend = str(execution_backend or _SERIAL_EXECUTION_BACKEND).strip().lower()
    if backend not in _EXECUTION_BACKENDS:
        raise ValueError(
            f"Unsupported execution_backend {execution_backend!r}; expected one of "
            f"{', '.join(_EXECUTION_BACKENDS)}."
        )
    return backend


def _normalize_postcompute_backend(postcompute_backend: object) -> str:
    backend = str(postcompute_backend or _SERIAL_POSTCOMPUTE_BACKEND).strip().lower()
    if backend not in _POSTCOMPUTE_BACKENDS:
        raise ValueError(
            f"Unsupported postcompute_backend {postcompute_backend!r}; expected one of "
            f"{', '.join(_POSTCOMPUTE_BACKENDS)}."
        )
    return backend


def _stable_json_equal(left: object, right: object) -> bool:
    return json.dumps(json_attr_safe(left), sort_keys=True, separators=(",", ":")) == json.dumps(
        json_attr_safe(right),
        sort_keys=True,
        separators=(",", ":"),
    )


def _array_data(array: object) -> np.ndarray:
    return np.asarray(array[:])  # type: ignore[index]


def _load_shard_names_from_file(path: Path) -> list[str]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, Mapping):
        for key in ("shard_runs", "subject_mask_shard_runs", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                names: list[str] = []
                for item in value:
                    if isinstance(item, Mapping):
                        raw_name = item.get("run_name") or item.get("run") or item.get("name")
                        if raw_name is None:
                            raise ValueError(f"Shard-run entry in {path} is missing run_name/run/name: {item!r}")
                        names.append(str(raw_name))
                    else:
                        names.append(str(item))
                return names
    raise ValueError(f"Could not read shard run list from {path}; expected list or mapping with shard_runs.")


def _row_identity_from_group(group: zarr.Group, row_ids: np.ndarray, names: Sequence[str]) -> list[tuple[object, ...]]:
    arrays = {name: _array_data(group[name]) for name in names}
    identities: list[tuple[object, ...]] = []
    for row_id in np.asarray(row_ids, dtype=np.int64).reshape(-1):
        parts: list[object] = []
        for name in names:
            value = np.asarray(arrays[name][int(row_id)])
            parts.append(value.item() if value.ndim == 0 else tuple(value.reshape(-1).tolist()))
        identities.append(tuple(parts))
    return identities


def _load_subject_mask_shard_sources(
    root: zarr.Group,
    shard_runs: Sequence[str],
) -> list[_SubjectMaskShardSource]:
    names = tuple(str(name) for name in shard_runs)
    if not names:
        raise ValueError("At least one subject-mask shard run is required.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate subject-mask shard run names were supplied.")
    parent = root.get(SUBJECT_MASK_SHARD_PARENT)
    if parent is None:
        raise ValueError(f"Missing {SUBJECT_MASK_SHARD_PARENT} group.")
    alias_root = _SubjectMaskParentAliasRoot(root, SUBJECT_MASK_SHARD_PARENT)
    sources: list[_SubjectMaskShardSource] = []
    for name in names:
        if name not in parent:
            raise ValueError(f"{SUBJECT_MASK_SHARD_PARENT}/{name} not found.")
        group = parent[name]
        status = str(group.attrs.get("palette_run_completion_status") or "")
        if status and status != "complete":
            raise ValueError(f"{SUBJECT_MASK_SHARD_PARENT}/{name} is not complete (status={status!r}).")
        source = _load_source_subject_mask_run(alias_root, name)  # type: ignore[arg-type]
        if source.source_crop_row_ids is None:
            raise ValueError(f"{SUBJECT_MASK_SHARD_PARENT}/{name} missing source_crop_row_ids.")
        sources.append(
            _SubjectMaskShardSource(
                name=name,
                source=source,
                source_crop_run=str(source.crop_run),
                row_count=int(source.masks_roi.shape[0]),
            )
        )
    return sources


def _validate_subject_mask_shard_compatibility(shards: Sequence[_SubjectMaskShardSource]) -> None:
    first = shards[0].source
    for shard in shards[1:]:
        source = shard.source
        if tuple(source.mask_labels) != tuple(first.mask_labels):
            raise ValueError(f"Shard mask_labels differ between {shards[0].name} and {shard.name}.")
        if not np.array_equal(np.asarray(source.available_channels, dtype=bool), np.asarray(first.available_channels, dtype=bool)):
            raise ValueError(f"Shard available_channels differ between {shards[0].name} and {shard.name}.")
        if tuple(source.masks_roi.shape[1:]) != tuple(first.masks_roi.shape[1:]):
            raise ValueError(f"Shard ROI/probability shape differs between {shards[0].name} and {shard.name}.")
        if tuple(source.probability_thresholds) != tuple(first.probability_thresholds):
            raise ValueError(f"Shard probability thresholds differ between {shards[0].name} and {shard.name}.")
        if source.probability_encoding != first.probability_encoding:
            raise ValueError(f"Shard probability encoding differs between {shards[0].name} and {shard.name}.")
        for attr_name in _SUBJECT_MASK_SHARD_COMPAT_ATTRS:
            left = first.group.attrs.get(attr_name)
            right = source.group.attrs.get(attr_name)
            if left is not None or right is not None:
                if not _stable_json_equal(left, right):
                    raise ValueError(f"Shard attr {attr_name!r} differs between {shards[0].name} and {shard.name}.")


def _resolve_subject_mask_crop_rebase(
    root: zarr.Group,
    shards: Sequence[_SubjectMaskShardSource],
    *,
    target_crop_run: str | None,
) -> tuple[str, np.ndarray, tuple[str, ...], bool]:
    source_crop_runs = tuple(shard.source_crop_run for shard in shards)
    unique_crop_runs = tuple(sorted(set(source_crop_runs)))
    if not target_crop_run:
        if len(unique_crop_runs) != 1:
            raise ValueError(
                "Mixed source_crop_run subject-mask shard finalization requires --target-crop-run; "
                f"found {list(unique_crop_runs)!r}."
            )
        row_ids = np.concatenate(
            [np.asarray(shard.source.source_crop_row_ids[:], dtype=np.int64).reshape(-1) for shard in shards],
            axis=0,
        )
        return unique_crop_runs[0], row_ids, unique_crop_runs, False

    crop_parent = root.get("crop_runs")
    if crop_parent is None or target_crop_run not in crop_parent:
        raise ValueError(f"target crop run not found: crop_runs/{target_crop_run}")
    target_group = crop_parent[target_crop_run]
    missing_target = [name for name in _CROP_REBASE_IDENTITY_ARRAYS if name not in target_group]
    if missing_target:
        raise ValueError(f"target crop run crop_runs/{target_crop_run} missing identity arrays: {missing_target}")
    target_rows = np.arange(int(target_group["frame_indices"].shape[0]), dtype=np.int64)
    target_identities = _row_identity_from_group(target_group, target_rows, _CROP_REBASE_IDENTITY_ARRAYS)
    target_map: dict[tuple[object, ...], int] = {}
    for row_idx, identity in enumerate(target_identities):
        if identity in target_map:
            raise ValueError(f"target crop run crop_runs/{target_crop_run} has duplicate crop row identities.")
        target_map[identity] = int(row_idx)

    mapped_chunks: list[np.ndarray] = []
    for shard in shards:
        source_group = crop_parent.get(shard.source_crop_run)
        if source_group is None:
            raise ValueError(f"source crop run not found: crop_runs/{shard.source_crop_run}")
        missing_source = [name for name in _CROP_REBASE_IDENTITY_ARRAYS if name not in source_group]
        if missing_source:
            raise ValueError(f"source crop run crop_runs/{shard.source_crop_run} missing identity arrays: {missing_source}")
        source_rows = np.asarray(shard.source.source_crop_row_ids[:], dtype=np.int64).reshape(-1)
        source_identities = _row_identity_from_group(source_group, source_rows, _CROP_REBASE_IDENTITY_ARRAYS)
        mapped = np.full(source_rows.shape[0], -1, dtype=np.int64)
        for local_idx, identity in enumerate(source_identities):
            target_row = target_map.get(identity)
            if target_row is None:
                raise ValueError(
                    f"Could not map {SUBJECT_MASK_SHARD_PARENT}/{shard.name} row {local_idx} "
                    f"from crop_runs/{shard.source_crop_run} into crop_runs/{target_crop_run}."
                )
            mapped[local_idx] = int(target_row)
        mapped_chunks.append(mapped)
    return str(target_crop_run), np.concatenate(mapped_chunks, axis=0), unique_crop_runs, True


def _target_crop_lineage_array(
    root: zarr.Group,
    *,
    crop_run: str,
    target_rows: np.ndarray,
    name: str,
) -> np.ndarray | None:
    crop_group = root.get(f"crop_runs/{crop_run}")
    if crop_group is None or name not in crop_group:
        return None
    return _array_data(crop_group[name])[np.asarray(target_rows, dtype=np.int64).reshape(-1)]


def _load_subject_mask_source(
    root: zarr.Group,
    *,
    subject_run: Optional[str],
    subject_shard_runs: Sequence[str] | None = None,
    target_crop_run: str | None = None,
) -> tuple[SourceSubjectMaskRun, _SubjectMaskShardCollection | None]:
    if subject_shard_runs:
        if subject_run:
            raise ValueError("Pass either --subject-run or --subject-shard-run, not both.")
        shards = _load_subject_mask_shard_sources(root, subject_shard_runs)
        _validate_subject_mask_shard_compatibility(shards)
        source_crop_run, source_crop_row_ids, shard_crop_runs, rebased = _resolve_subject_mask_crop_rebase(
            root,
            shards,
            target_crop_run=target_crop_run,
        )
        source_indices = np.concatenate(
            [np.full(shard.row_count, shard_idx, dtype=np.int64) for shard_idx, shard in enumerate(shards)],
            axis=0,
        )
        local_indices = np.concatenate(
            [np.arange(shard.row_count, dtype=np.int64) for shard in shards],
            axis=0,
        )
        order = np.argsort(np.asarray(source_crop_row_ids, dtype=np.int64), kind="stable")
        sorted_crop_rows = np.asarray(source_crop_row_ids, dtype=np.int64)[order]
        if np.unique(sorted_crop_rows).shape[0] != sorted_crop_rows.shape[0]:
            raise ValueError("Duplicate source_crop_row_ids found across subject-mask shards for the target source_crop_run.")
        source_indices = source_indices[order]
        local_indices = local_indices[order]

        first_source = shards[0].source
        arrays: dict[str, object] = {}
        array_names = (
            "mask_probs_roi",
            "masks_roi",
            "detection_source",
            "frame_indices",
            "detection_indices",
            "source_frame_indices",
            "source_clip_indices",
            "source_clip_local_frame_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
            "instance_key",
        )
        for name in array_names:
            source_arrays = [shard.source.group.get(name) for shard in shards]
            if all(array is not None for array in source_arrays):
                arrays[name] = _IndexedCollectionArray(
                    [array for array in source_arrays if array is not None],
                    source_indices=source_indices,
                    local_indices=local_indices,
                )
        arrays["source_crop_row_ids"] = sorted_crop_rows
        if target_crop_run:
            for name in _CROP_REBASE_COPY_ARRAYS:
                target_values = _target_crop_lineage_array(root, crop_run=source_crop_run, target_rows=sorted_crop_rows, name=name)
                if target_values is not None:
                    arrays[name] = np.asarray(target_values)
        frame_counts = None
        if target_crop_run:
            crop_group = root.get(f"crop_runs/{source_crop_run}")
            if crop_group is not None and "frame_counts" in crop_group:
                frame_counts = np.asarray(crop_group["frame_counts"][:], dtype=np.int32)
        if frame_counts is None:
            frames = (
                np.asarray(arrays["frame_indices"][:], dtype=np.int64).reshape(-1)
                if "frame_indices" in arrays
                else np.arange(sorted_crop_rows.shape[0], dtype=np.int64)
            )
            frame_axis_len = int(frames.max()) + 1 if frames.size else 0
            frame_counts = np.bincount(frames, minlength=frame_axis_len).astype(np.int32)
        arrays["frame_counts"] = np.asarray(frame_counts, dtype=np.int32)

        attrs = dict(first_source.group.attrs)
        attrs.update(
            {
                "source_crop_run": source_crop_run,
                "source_subject_mask_shard_runs": [shard.name for shard in shards],
                "source_subject_mask_shard_run_paths": [f"{SUBJECT_MASK_SHARD_PARENT}/{shard.name}" for shard in shards],
                "source_subject_mask_shard_crop_runs": list(shard_crop_runs),
                "source_crop_rebased_from_shards": bool(rebased),
                "collection_finalizer_schema": SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA,
            }
        )
        source_crop_snapshot = dict(first_source.source_crop_snapshot)
        crop_group = root.get(f"crop_runs/{source_crop_run}")
        if crop_group is not None:
            for key, target_key in (
                ("crop_storage_mode", "source_crop_storage_mode"),
                ("crop_signature", "source_crop_signature"),
                ("crop_revision", "source_crop_revision"),
                ("detect_review_status_ref", "source_detect_review_status_ref"),
                ("source_roi_pixel_contract", "source_roi_pixel_contract"),
                ("source_roi_pixel_contract_name", "source_roi_pixel_contract_name"),
                ("source_roi_image_representation", "source_roi_image_representation"),
            ):
                value = crop_group.attrs.get(key)
                if value is not None:
                    source_crop_snapshot[target_key] = value

        virtual_group = _SubjectMaskCollectionGroup(
            attrs,
            arrays,
            path=f"{SUBJECT_MASK_SHARD_PARENT}/<collection>",
        )
        if first_source.mask_surface_path == "mask_probs_roi" and "mask_probs_roi" in arrays:
            masks_roi = first_source.masks_roi.__class__(  # type: ignore[misc]
                arrays["mask_probs_roi"],
                thresholds=first_source.probability_thresholds,
                encoding=first_source.probability_encoding,
                source_path=f"{SUBJECT_MASK_SHARD_PARENT}/<collection>/mask_probs_roi",
            )
        elif "masks_roi" in arrays:
            masks_roi = arrays["masks_roi"]
        elif "mask_probs_roi" in arrays:
            masks_roi = first_source.masks_roi.__class__(  # type: ignore[misc]
                arrays["mask_probs_roi"],
                thresholds=first_source.probability_thresholds,
                encoding=first_source.probability_encoding,
                source_path=f"{SUBJECT_MASK_SHARD_PARENT}/<collection>/mask_probs_roi",
            )
        else:
            raise ValueError("Subject-mask shards must provide mask_probs_roi or masks_roi.")

        collection = _SubjectMaskShardCollection(
            shard_runs=tuple(shard.name for shard in shards),
            shard_run_paths=tuple(f"{SUBJECT_MASK_SHARD_PARENT}/{shard.name}" for shard in shards),
            shard_crop_runs=tuple(shard_crop_runs),
            source_crop_run=str(source_crop_run),
            source_crop_rebased_from_shards=bool(rebased),
            row_source_indices=source_indices,
            row_local_indices=local_indices,
            source_crop_row_ids=sorted_crop_rows,
        )
        source = SourceSubjectMaskRun(
            run_name="subject_mask_shard_collection",
            group=virtual_group,  # type: ignore[arg-type]
            crop_run=str(source_crop_run),
            source_crop_snapshot=source_crop_snapshot,
            masks_roi=masks_roi,
            detection_source=arrays["detection_source"],
            mask_labels=tuple(first_source.mask_labels),
            available_channels=np.asarray(first_source.available_channels, dtype=bool),
            frame_indices=arrays.get("frame_indices"),
            frame_counts=arrays.get("frame_counts"),
            detection_indices=arrays.get("detection_indices"),
            source_method=first_source.source_method,
            source_keypoints_run=first_source.source_keypoints_run,
            source_keypoint_group=first_source.source_keypoint_group,
            assignment_keypoints_run=first_source.assignment_keypoints_run,
            assignment_keypoint_group=first_source.assignment_keypoint_group,
            source_crop_row_ids=arrays["source_crop_row_ids"],
            source_refined_row_ids=arrays.get("source_refined_row_ids"),
            source_detect_row_index=arrays.get("source_detect_row_index"),
            instance_key=arrays.get("instance_key"),
            mask_surface_kind=first_source.mask_surface_kind,
            mask_surface_path=first_source.mask_surface_path,
            probability_thresholds=tuple(first_source.probability_thresholds),
            probability_encoding=first_source.probability_encoding,
        )
        return source, collection
    if target_crop_run:
        raise ValueError("--target-crop-run is only valid with subject-mask shard finalization.")
    return _load_source_subject_mask_run(root, subject_run), None


def _component_contour_targets(component_names: Sequence[str]) -> list[str]:
    requested = {str(component) for component in component_names}
    return [component for component in _COMPONENT_CONTOUR_COMPONENTS if component in requested]


def _summaries_to_json_safe(summaries: Sequence[object]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for summary in summaries:
        if isinstance(summary, Mapping):
            payload.append(
                {
                    "component": str(summary.get("component", "")),
                    "status": str(summary.get("status", "")),
                    "reason": summary.get("reason"),
                    "roi_count": int(summary.get("roi_count", 0) or 0),
                    "contour_count": int(summary.get("contour_count", 0) or 0),
                    "point_count": int(summary.get("point_count", 0) or 0),
                    "existing": bool(summary.get("existing", False)),
                }
            )
            continue
        payload.append(
            {
                "component": str(getattr(summary, "component", "")),
                "status": str(getattr(summary, "status", "")),
                "reason": getattr(summary, "reason", None),
                "roi_count": int(getattr(summary, "roi_count", 0) or 0),
                "contour_count": int(getattr(summary, "contour_count", 0) or 0),
                "point_count": int(getattr(summary, "point_count", 0) or 0),
                "existing": bool(getattr(summary, "existing", False)),
            }
        )
    return payload


_json_safe = json_attr_safe


@dataclass
class _TimingRecorder:
    phase_seconds: dict[str, float] = field(default_factory=dict)
    phase_counts: dict[str, int] = field(default_factory=dict)
    chunk_timings: list[dict[str, object]] = field(default_factory=list)

    def add(self, phase: str, seconds: float) -> float:
        key = str(phase)
        elapsed = float(seconds)
        self.phase_seconds[key] = float(self.phase_seconds.get(key, 0.0) + elapsed)
        self.phase_counts[key] = int(self.phase_counts.get(key, 0) + 1)
        return elapsed

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, time.perf_counter() - start)

    def summary(self, *, total_rows: int, duration_seconds: float) -> dict[str, object]:
        total = float(duration_seconds)
        rows = int(total_rows)
        phase_seconds = {
            str(key): float(value)
            for key, value in sorted(self.phase_seconds.items())
        }
        return {
            "duration_seconds": total,
            "rows_total": rows,
            "rows_per_second": float(rows / total) if total > 0 else 0.0,
            "phase_seconds": phase_seconds,
            "phase_counts": {
                str(key): int(value)
                for key, value in sorted(self.phase_counts.items())
            },
            "chunk_count": int(len(self.chunk_timings)),
            "chunk_seconds_total": float(sum(float(item.get("total_seconds") or 0.0) for item in self.chunk_timings)),
            "slowest_chunks": sorted(
                (
                    {
                        "chunk_index": int(item.get("chunk_index") or 0),
                        "start_row": int(item.get("start_row") or 0),
                        "stop_row": int(item.get("stop_row") or 0),
                        "total_seconds": float(item.get("total_seconds") or 0.0),
                    }
                    for item in self.chunk_timings
                ),
                key=lambda item: float(item["total_seconds"]),
                reverse=True,
            )[:5],
        }


@contextmanager
def _timed_chunk_phase(
    timing: Optional[_TimingRecorder],
    chunk_timing: Optional[dict[str, object]],
    phase: str,
) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = float(time.perf_counter() - start)
        if timing is not None:
            timing.add(phase, elapsed)
        if chunk_timing is not None:
            key = f"{phase}_seconds"
            chunk_timing[key] = float(chunk_timing.get(key) or 0.0) + elapsed


@dataclass
class _ProgressJsonlReporter:
    path: Optional[Path] = None
    start_time: float = field(default_factory=time.perf_counter)

    def emit(self, event: str, **payload: object) -> None:
        if self.path is None:
            return
        record = {
            "event": str(event),
            "ts_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - self.start_time),
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":")) + "\n")

    @contextmanager
    def phase(self, phase: str, **payload: object) -> Iterator[None]:
        start = time.perf_counter()
        self.emit(f"{phase}_start", **payload)
        try:
            yield
        except Exception as exc:
            self.emit(
                f"{phase}_error",
                duration_seconds=float(time.perf_counter() - start),
                error=repr(exc),
                **payload,
            )
            raise
        else:
            self.emit(
                f"{phase}_end",
                duration_seconds=float(time.perf_counter() - start),
                **payload,
            )


def _decode_probabilities(values: np.ndarray, *, encoding: Optional[str], source_path: str) -> np.ndarray:
    return decode_probability_values(values, encoding=encoding, source_path=source_path)


def _join_reason_tags(tags: Sequence[object], *, probability_source: bool) -> str:
    merged: list[str] = []
    if probability_source:
        merged.append("cleanup_thresholded_probability")
    for raw_tag in tags:
        tag = str(raw_tag or "").strip()
        if not tag:
            continue
        if probability_source and tag == "clean":
            continue
        if tag not in merged:
            merged.append(tag)
    if not merged:
        merged.append("clean")
    return "|".join(merged)


def _combine_reason_labels(*labels: object) -> str:
    merged: list[str] = []
    for label in labels:
        for raw_tag in str(label or "").split("|"):
            tag = raw_tag.strip()
            if not tag or tag == "clean":
                continue
            if tag not in merged:
                merged.append(tag)
    return "|".join(merged) if merged else "clean"


def _policy_payload(policy: ComponentFinalizationPolicy) -> dict[str, object]:
    return {str(key): _json_safe(value) for key, value in asdict(policy).items()}


def _component_metric_qc_policy(component_name: str) -> _ComponentMetricQcPolicy:
    """Return conservative mask-local QC gates for one refined component."""

    if component_name == "subject_body":
        return _ComponentMetricQcPolicy(
            component_name=component_name,
            min_area_px=8.0,
            max_component_count=1,
            max_hole_count=0,
            min_largest_component_fraction=0.90,
            min_solidity=0.20,
        )
    if component_name == "swim_bladder":
        return _ComponentMetricQcPolicy(
            component_name=component_name,
            min_area_px=4.0,
            max_component_count=1,
            max_hole_count=0,
            min_largest_component_fraction=0.95,
            min_solidity=0.20,
        )
    if component_name in _EYE_COMPONENTS:
        return _ComponentMetricQcPolicy(
            component_name=component_name,
            min_area_px=4.0,
            max_component_count=1,
            max_hole_count=0,
            min_largest_component_fraction=0.90,
            min_solidity=0.20,
        )
    return _ComponentMetricQcPolicy(component_name=component_name)


def _component_metric_qc_policy_payload(component_name: str) -> dict[str, object]:
    return _json_safe(asdict(_component_metric_qc_policy(component_name)))  # type: ignore[return-value]


def _split_reason_tags(label: object) -> list[str]:
    tags: list[str] = []
    for raw_tag in str(label or "").split("|"):
        tag = raw_tag.strip()
        if not tag or tag == "clean":
            continue
        if tag not in tags:
            tags.append(tag)
    return tags


def _merge_reason_label_arrays(base_labels: np.ndarray, extra_labels: np.ndarray) -> np.ndarray:
    base = np.asarray(base_labels, dtype=object).reshape(-1)
    extra = np.asarray(extra_labels, dtype=object).reshape(-1)
    if base.shape[0] != extra.shape[0]:
        raise ValueError("base_labels and extra_labels must have the same row count.")
    merged = np.empty(base.shape, dtype=object)
    for row_idx, (base_label, extra_label) in enumerate(zip(base, extra)):
        merged[row_idx] = _combine_reason_labels(base_label, extra_label)
    return merged


def _replace_metric_qc_reason_labels(base_labels: np.ndarray, qc_labels: np.ndarray) -> np.ndarray:
    """Refresh generated metric-QC tags while preserving manual/operator tags."""

    base = np.asarray(base_labels, dtype=object).reshape(-1)
    qc = np.asarray(qc_labels, dtype=object).reshape(-1)
    if base.shape[0] != qc.shape[0]:
        raise ValueError("base_labels and qc_labels must have the same row count.")
    refreshed = np.empty(base.shape, dtype=object)
    for row_idx, (base_label, qc_label) in enumerate(zip(base, qc)):
        tags = [
            tag
            for tag in _split_reason_tags(base_label)
            if not str(tag).startswith(_COMPONENT_QC_REASON_PREFIX)
        ]
        for tag in _split_reason_tags(qc_label):
            if tag not in tags:
                tags.append(tag)
        refreshed[row_idx] = "|".join(tags) if tags else "clean"
    return refreshed


def _compute_component_metric_qc_reason_labels(
    component_name: str,
    *,
    mask_present: np.ndarray,
    area_px: np.ndarray,
    component_metrics: Mapping[str, np.ndarray],
) -> np.ndarray:
    policy = _component_metric_qc_policy(component_name)
    present = np.asarray(mask_present, dtype=bool).reshape(-1)
    area = np.asarray(area_px, dtype=np.float32).reshape(-1)
    component_count = np.asarray(
        component_metrics.get("component_count", np.zeros_like(area, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    largest_fraction = np.asarray(
        component_metrics.get("largest_component_fraction", np.ones_like(area, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    hole_count = np.asarray(
        component_metrics.get("hole_count", np.zeros_like(area, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    solidity_raw = component_metrics.get("solidity")
    solidity = np.asarray(solidity_raw, dtype=np.float32).reshape(-1) if solidity_raw is not None else None

    labels = np.full((int(area.shape[0]),), "clean", dtype=object)
    for row_idx in range(int(area.shape[0])):
        tags: list[str] = []
        if not bool(present[row_idx]) or float(area[row_idx]) <= 0.0:
            tags.append("needs_review_metric_empty_mask")
        elif float(area[row_idx]) < float(policy.min_area_px):
            tags.append("needs_review_metric_small_area")
        if int(component_count[row_idx]) > int(policy.max_component_count):
            tags.append("needs_review_metric_multiple_components")
        if bool(present[row_idx]) and float(largest_fraction[row_idx]) < float(policy.min_largest_component_fraction):
            tags.append("needs_review_metric_fragmented_component")
        if int(hole_count[row_idx]) > int(policy.max_hole_count):
            tags.append("needs_review_metric_holes")
        if solidity is not None and policy.min_solidity is not None:
            value = float(solidity[row_idx])
            if np.isfinite(value) and value < float(policy.min_solidity):
                tags.append("needs_review_metric_low_solidity")
        labels[row_idx] = "|".join(tags) if tags else "clean"
    return labels


def _component_threshold(source: SourceSubjectMaskRun, component_name: str, component_idx: int) -> float:
    if component_idx < len(source.probability_thresholds):
        return float(source.probability_thresholds[component_idx])
    labels = tuple(str(label) for label in source.mask_labels)
    thresholds = _probability_thresholds_for_labels(source.group, labels)
    if component_idx < len(thresholds):
        return float(thresholds[component_idx])
    return 0.5


def _component_surface_rows(
    source: SourceSubjectMaskRun,
    component_name: str,
    *,
    start_row: int,
    stop_row: int,
) -> tuple[np.ndarray, bool, str, Optional[str], float, int]:
    component_idx = _require_available_component(source, component_name, "subject_mask_runs")
    probabilities = source.group.get("mask_probs_roi")
    threshold = _component_threshold(source, component_name, component_idx)
    if probabilities is not None:
        encoding = source.probability_encoding or _probability_encoding_for_group(source.group)
        raw = np.asarray(probabilities[int(start_row) : int(stop_row), component_idx])
        source_path = f"subject_mask_runs/{source.run_name}/mask_probs_roi"
        return (
            _decode_probabilities(raw, encoding=encoding, source_path=source_path),
            True,
            "mask_probs_roi",
            encoding,
            threshold,
            component_idx,
        )

    masks = source.group.get("masks_roi")
    if masks is None:
        if source.mask_surface_path != "mask_rle":
            raise RuntimeError(f"subject_mask_runs/{source.run_name} missing masks_roi, mask_probs_roi, or mask_rle.")
        masks = source.masks_roi
    return (
        np.asarray(masks[int(start_row) : int(stop_row), component_idx], dtype=np.uint8),
        False,
        str(source.mask_surface_path or "masks_roi"),
        None,
        threshold,
        component_idx,
    )


def _component_surface_batch(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> tuple[np.ndarray, bool, str, Optional[str], float, int]:
    return _component_surface_rows(
        source,
        component_name,
        start_row=0,
        stop_row=int(source.masks_roi.shape[0]),
    )


def _finalize_source_component(
    source: SourceSubjectMaskRun,
    component_name: str,
) -> _FinalizedComponentBatch:
    return _finalize_source_component_rows(
        source,
        component_name,
        start_row=0,
        stop_row=int(source.masks_roi.shape[0]),
    )


def _finalize_source_component_rows(
    source: SourceSubjectMaskRun,
    component_name: str,
    *,
    start_row: int,
    stop_row: int,
) -> _FinalizedComponentBatch:
    surfaces, is_probability, surface_path, encoding, threshold, _component_idx = _component_surface_rows(
        source,
        component_name,
        start_row=start_row,
        stop_row=stop_row,
    )
    if surfaces.ndim != 3:
        raise ValueError(
            f"subject_mask_runs/{source.run_name}/{surface_path} component {component_name!r} "
            f"must have shape (N,H,W), got {tuple(surfaces.shape)}."
        )
    base_policy = _default_policy_for_component(component_name)
    policy = replace(base_policy, threshold=float(threshold))

    total_rows = int(surfaces.shape[0])
    masks = np.zeros(surfaces.shape, dtype=np.uint8)
    source_masks = np.zeros(surfaces.shape, dtype=np.uint8)
    reason_labels = np.full((total_rows,), "clean", dtype=object)
    quality_code = np.zeros((total_rows,), dtype=np.int16)
    quality_score = np.zeros((total_rows,), dtype=np.float32)
    review_recommendation = np.full((total_rows,), "pending", dtype=object)
    metric_values: dict[str, list[float]] = {}

    for row_idx in range(total_rows):
        result = finalize_component_mask(
            component_name,
            surfaces[row_idx],
            policy=policy,
            surface_is_probability=is_probability,
        )
        masks[row_idx] = np.asarray(result.mask, dtype=np.uint8)
        source_masks[row_idx] = np.asarray(result.source_mask, dtype=np.uint8)
        reason_labels[row_idx] = _join_reason_tags(
            result.reason_tags,
            probability_source=bool(is_probability),
        )
        quality_code[row_idx] = np.int16(result.quality_code)
        quality_score[row_idx] = np.float32(result.quality_score)
        review_recommendation[row_idx] = str(result.review_recommendation)
        for name, value in result.metrics.items():
            metric_values.setdefault(str(name), []).append(float(value))

    metrics = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in sorted(metric_values.items())
    }
    return _FinalizedComponentBatch(
        component_name=component_name,
        masks=masks,
        source_masks=source_masks,
        reason_labels=reason_labels,
        quality_code=quality_code,
        quality_score=quality_score,
        review_recommendation=review_recommendation,
        metrics=metrics,
        policy=policy,
        source_surface_path=surface_path,
        source_surface_kind="probability" if is_probability else "binary",
        source_probability_encoding=encoding,
        source_probability_threshold=float(threshold),
    )


def _source_payload_for_finalized_component(
    source: SourceSubjectMaskRun,
    batch: _FinalizedComponentBatch,
) -> dict[str, object]:
    payload = _source_component_provenance_payload(source, batch.component_name)
    source_parent = (
        SUBJECT_MASK_SHARD_PARENT
        if source.group.attrs.get("collection_finalizer_schema") == SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA
        else SUBJECT_MASK_CANONICAL_PARENT
    )
    payload["finalization_method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    payload["finalization_policy"] = _policy_payload(batch.policy)
    payload["source_stage"] = source_parent
    payload["source_surface_path"] = (
        f"{source_parent}/{source.run_name}/{batch.source_surface_path}"
    )
    payload["source_surface_kind"] = batch.source_surface_kind
    if batch.source_surface_kind == "probability":
        payload["source_probability_path"] = (
            f"{source_parent}/{source.run_name}/{batch.source_surface_path}"
        )
        payload["source_probability_encoding"] = str(batch.source_probability_encoding or "")
        payload["source_probability_threshold"] = float(batch.source_probability_threshold)
        payload["source_binary_derivation"] = "smart_finalize(mask_probs_roi)"
    else:
        payload["source_binary_derivation"] = f"smart_finalize({batch.source_surface_path})"
    return payload


def _source_payload_for_assigned_eye_component(
    source: SourceSubjectMaskRun,
    *,
    component_name: str,
    union_batch: _FinalizedComponentBatch,
    assignment_summary: Mapping[str, object],
    keypoint_run_name: str,
    keypoint_group_name: str,
    keypoint_success_dataset: str,
    keypoint_source_kind: str,
) -> dict[str, object]:
    source_parent = (
        SUBJECT_MASK_SHARD_PARENT
        if source.group.attrs.get("collection_finalizer_schema") == SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA
        else SUBJECT_MASK_CANONICAL_PARENT
    )
    payload: dict[str, object] = {
        "source_stage": source_parent,
        "source_run": source.run_name,
        "source_method": source.source_method or source.group.attrs.get("method") or "unknown",
        "source_channels": [_RAW_EYE_UNION_COMPONENT],
        "source_crop_run": source.crop_run,
        "derived_component": str(component_name),
        "derived_from_component": _RAW_EYE_UNION_COMPONENT,
        "assignment_method": EYES_UNION_ASSIGNMENT_METHOD,
        "assignment_summary": dict(_json_safe(dict(assignment_summary))),
        "assignment_keypoint_run": str(keypoint_run_name),
        "assignment_keypoint_group": str(keypoint_group_name),
        "assignment_keypoint_success_dataset": str(keypoint_success_dataset),
        "assignment_keypoint_source_kind": str(keypoint_source_kind),
        "finalization_method": SMART_FINALIZE_SUBJECT_MASKS_METHOD,
        "finalization_policy": _policy_payload(union_batch.policy),
        "source_surface_path": f"{source_parent}/{source.run_name}/{union_batch.source_surface_path}",
        "source_surface_kind": union_batch.source_surface_kind,
        **source.source_crop_snapshot,
    }
    label_schema_id = source.group.attrs.get("label_schema_id")
    if label_schema_id is not None:
        payload["source_label_schema_id"] = str(label_schema_id)
    created_at = source.group.attrs.get("created_at_utc") or source.group.attrs.get("created_utc")
    if created_at is not None:
        payload["source_created_at_utc"] = str(created_at)
    if union_batch.source_surface_kind == "probability":
        payload["source_probability_path"] = (
            f"{source_parent}/{source.run_name}/{union_batch.source_surface_path}"
        )
        payload["source_probability_encoding"] = str(union_batch.source_probability_encoding or "")
        payload["source_probability_threshold"] = float(union_batch.source_probability_threshold)
        payload["source_binary_derivation"] = "smart_finalize(mask_probs_roi)"
    else:
        payload["source_binary_derivation"] = f"smart_finalize({union_batch.source_surface_path})"
    return payload


def _read_optional_source_crop_row_ids(value: object | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        data = np.asarray(value[:], dtype=np.int64)  # type: ignore[index]
    except Exception:
        data = np.asarray(value, dtype=np.int64)
    return data.reshape(-1)


def _resolve_assignment_keypoint_rows(
    *,
    keypoints_roi: Any,
    keypoint_success: np.ndarray,
    keypoint_source_crop_row_ids: object | None,
    mask_source_crop_row_ids: object | None,
    expected_rows: int,
    keypoint_run_name: str,
    keypoint_group_name: str,
    mask_run_name: str,
) -> tuple[Any, np.ndarray, dict[str, object]]:
    """Return keypoints aligned to mask rows, subsetting collection runs when needed."""

    keypoint_ids = _read_optional_source_crop_row_ids(keypoint_source_crop_row_ids)
    mask_ids = _read_optional_source_crop_row_ids(mask_source_crop_row_ids)
    expected = int(expected_rows)

    if keypoint_ids is None or mask_ids is None:
        summary = reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=keypoint_source_crop_row_ids,
            mask_source_crop_row_ids=mask_source_crop_row_ids,
            expected_rows=expected,
            keypoint_run_name=keypoint_run_name,
            keypoint_group_name=keypoint_group_name,
            mask_run_name=mask_run_name,
            mask_group_name="subject_mask_runs",
        )
        return keypoints_roi, np.asarray(keypoint_success, dtype=bool), dict(summary)

    keypoint_path = f"{keypoint_group_name}/{keypoint_run_name}"
    mask_path = f"subject_mask_runs/{mask_run_name}"
    if int(mask_ids.shape[0]) != expected:
        raise ValueError(
            f"Cannot assign eyes_union from {mask_path}: mask source_crop_row_ids has "
            f"{int(mask_ids.shape[0])} rows, expected {expected}."
        )
    if int(keypoints_roi.shape[0]) != int(keypoint_ids.shape[0]):
        raise ValueError(
            f"Cannot assign eyes_union with {keypoint_path}: keypoints_roi has "
            f"{int(keypoints_roi.shape[0])} rows but source_crop_row_ids has "
            f"{int(keypoint_ids.shape[0])} rows."
        )
    success = np.asarray(keypoint_success, dtype=bool)
    if int(success.shape[0]) != int(keypoint_ids.shape[0]):
        raise ValueError(
            f"Cannot assign eyes_union with {keypoint_path}: {success.shape[0]} keypoint success rows "
            f"but source_crop_row_ids has {int(keypoint_ids.shape[0])} rows."
        )

    if int(keypoint_ids.shape[0]) == expected:
        summary = reconcile_keypoint_mask_row_identity(
            keypoint_source_crop_row_ids=keypoint_source_crop_row_ids,
            mask_source_crop_row_ids=mask_source_crop_row_ids,
            expected_rows=expected,
            keypoint_run_name=keypoint_run_name,
            keypoint_group_name=keypoint_group_name,
            mask_run_name=mask_run_name,
            mask_group_name="subject_mask_runs",
        )
        return keypoints_roi, success, dict(summary)

    unique_ids, counts = np.unique(keypoint_ids, return_counts=True)
    if unique_ids.shape[0] != keypoint_ids.shape[0]:
        duplicate = int(unique_ids[np.flatnonzero(counts > 1)[0]])
        raise ValueError(
            f"Cannot subset {keypoint_path} for eyes_union assignment: duplicate "
            f"source_crop_row_ids value {duplicate}."
        )
    keypoint_row_by_crop_row = {
        int(crop_row_id): int(row_idx)
        for row_idx, crop_row_id in enumerate(keypoint_ids.tolist())
    }
    missing = [
        int(crop_row_id)
        for crop_row_id in mask_ids.tolist()
        if int(crop_row_id) not in keypoint_row_by_crop_row
    ]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        raise ValueError(
            f"Cannot subset {keypoint_path} for eyes_union assignment: missing "
            f"{len(missing)} source_crop_row_ids required by {mask_path}: {preview}."
        )

    selected_rows = np.asarray(
        [keypoint_row_by_crop_row[int(crop_row_id)] for crop_row_id in mask_ids.tolist()],
        dtype=np.int64,
    )
    keypoints_subset = np.asarray(keypoints_roi[:], dtype=np.float32)[selected_rows]
    success_subset = success[selected_rows]
    return (
        keypoints_subset,
        success_subset,
        {
            "row_identity_check": "source_crop_row_ids_subset",
            "rows_checked": expected,
            "keypoint_has_source_crop_row_ids": True,
            "mask_has_source_crop_row_ids": True,
            "keypoint_rows_available": int(keypoint_ids.shape[0]),
            "keypoint_rows_selected": int(selected_rows.shape[0]),
            "keypoint_selection_min_row": int(selected_rows.min()) if selected_rows.size else None,
            "keypoint_selection_max_row": int(selected_rows.max()) if selected_rows.size else None,
        },
    )


def _resolve_eye_assignment_context(
    root: zarr.Group,
    source: SourceSubjectMaskRun,
    *,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
) -> _EyeAssignmentContext:
    kp_group, keypoint_run_name, keypoint_group_name, keypoint_source_kind = _resolve_subject_keypoint_group(
        root,
        source,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
    )
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Keypoint run {keypoint_run_name!r} missing keypoints_roi; cannot assign eyes_union.")
    keypoint_success, success_dataset = _resolve_keypoint_success_array(kp_group, keypoint_run_name)
    eye_keypoint_indices = _resolve_eye_keypoint_indices(kp_group, keypoint_run_name)
    keypoints_roi, keypoint_success, row_identity_summary = _resolve_assignment_keypoint_rows(
        keypoints_roi=keypoints_roi,
        keypoint_success=keypoint_success,
        keypoint_source_crop_row_ids=kp_group.get("source_crop_row_ids"),
        mask_source_crop_row_ids=source.group.get("source_crop_row_ids"),
        expected_rows=int(source.masks_roi.shape[0]),
        keypoint_run_name=keypoint_run_name,
        keypoint_group_name=keypoint_group_name,
        mask_run_name=source.run_name,
    )
    return _EyeAssignmentContext(
        keypoints_roi=keypoints_roi,
        keypoint_success=np.asarray(keypoint_success, dtype=bool),
        eye_keypoint_indices=eye_keypoint_indices,
        keypoint_run_name=str(keypoint_run_name),
        keypoint_group_name=str(keypoint_group_name),
        keypoint_success_dataset=str(success_dataset),
        keypoint_source_kind=str(keypoint_source_kind),
        row_identity_summary=dict(row_identity_summary),
    )


def _assign_finalized_eyes_union_rows(
    source: SourceSubjectMaskRun,
    union_batch: _FinalizedComponentBatch,
    context: _EyeAssignmentContext,
    *,
    start_row: int,
    stop_row: int,
) -> _EyeAssignmentChunk:
    assignment = assign_eyes_union_to_lr(
        np.asarray(union_batch.masks, dtype=np.uint8),
        keypoints_roi=np.asarray(context.keypoints_roi[int(start_row) : int(stop_row)], dtype=np.float32),
        keypoint_success=np.asarray(context.keypoint_success[int(start_row) : int(stop_row)], dtype=bool),
        eye_keypoint_indices=context.eye_keypoint_indices,
    )
    summary = dict(assignment.summary)
    summary["keypoint_run"] = context.keypoint_run_name
    summary["keypoint_group"] = context.keypoint_group_name
    summary["keypoint_success_dataset"] = context.keypoint_success_dataset
    summary["keypoint_source_kind"] = context.keypoint_source_kind
    summary["keypoint_mask_row_identity"] = dict(context.row_identity_summary)
    summary["keypoint_mask_row_identity_check"] = str(
        context.row_identity_summary.get("row_identity_check", "unknown")
    )
    summary["assignment_keypoint_contract"] = ASSIGNMENT_KEYPOINT_CONTRACT_VALUE

    reason_labels_by_component: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    component_metrics: dict[str, dict[str, np.ndarray]] = {}
    for component_name in _EYE_COMPONENTS:
        assignment_reasons = np.asarray(assignment.reason_labels[component_name], dtype=object)
        reason_labels = np.asarray(
            [
                _combine_reason_labels(union_batch.reason_labels[row_idx], assignment_reasons[row_idx])
                for row_idx in range(int(union_batch.reason_labels.shape[0]))
            ],
            dtype=object,
        )
        reason_labels_by_component[component_name] = reason_labels
        masks[component_name] = np.asarray(assignment.masks[component_name], dtype=np.uint8)
        component_metrics[component_name] = _component_metrics_from_assigned_eye_masks(masks[component_name])
    return _EyeAssignmentChunk(
        masks=masks,
        reason_labels=reason_labels_by_component,
        component_metrics=component_metrics,
        summary=summary,
        phase_seconds={str(key): float(value) for key, value in dict(assignment.phase_seconds).items()},
        eye_geometry=_eye_geometry_from_assignment_result(assignment),
    )


def _merge_assignment_summary(target: dict[str, object], chunk_summary: Mapping[str, object]) -> dict[str, object]:
    if not target:
        target.update(
            {
                "assignment_method": EYES_UNION_ASSIGNMENT_METHOD,
                "total_rows": 0,
                "assigned_rows": 0,
                "assigned_needs_review_rows": 0,
                "failed_rows": 0,
                "status_counts": {},
                "reason_counts": {},
                "keypoint_run": chunk_summary.get("keypoint_run"),
                "keypoint_group": chunk_summary.get("keypoint_group"),
                "keypoint_success_dataset": chunk_summary.get("keypoint_success_dataset"),
                "keypoint_source_kind": chunk_summary.get("keypoint_source_kind"),
                "keypoint_mask_row_identity": dict(chunk_summary.get("keypoint_mask_row_identity") or {}),
                "keypoint_mask_row_identity_check": chunk_summary.get("keypoint_mask_row_identity_check"),
                "assignment_keypoint_contract": ASSIGNMENT_KEYPOINT_CONTRACT_VALUE,
            }
        )
    for key in ("total_rows", "assigned_rows", "assigned_needs_review_rows", "failed_rows"):
        target[key] = int(target.get(key) or 0) + int(chunk_summary.get(key) or 0)
    for key in ("status_counts", "reason_counts"):
        merged = dict(target.get(key) or {})
        for item_key, item_value in dict(chunk_summary.get(key) or {}).items():
            merged[str(item_key)] = int(merged.get(str(item_key), 0)) + int(item_value)
        target[key] = merged
    return target


def _record_eye_assignment_phase_seconds(
    timing: _TimingRecorder,
    chunk_timing: Optional[dict[str, object]],
    phase_seconds: Mapping[str, float],
) -> None:
    for phase_name, seconds in dict(phase_seconds).items():
        key = f"eye_assignment_{phase_name}"
        elapsed = timing.add(key, float(seconds))
        if chunk_timing is not None:
            chunk_timing[f"{key}_seconds"] = elapsed


def _requested_output_components(
    source: SourceSubjectMaskRun,
    components: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if not components:
        output: list[str] = []
        if _has_available_component(source, "subject_body"):
            output.append("subject_body")
        if _has_available_component(source, _RAW_EYE_UNION_COMPONENT):
            output.extend(_EYE_COMPONENTS)
        if _has_available_component(source, "swim_bladder"):
            output.append("swim_bladder")
        return tuple(component for component in CANONICAL_COMPONENT_ORDER if component in output)

    normalized = []
    seen: set[str] = set()
    for raw_component in components:
        component = _normalize_component_name(raw_component)
        if component is None:
            continue
        if component == _RAW_EYE_UNION_COMPONENT:
            for eye_component in _EYE_COMPONENTS:
                if eye_component not in seen:
                    seen.add(eye_component)
                    normalized.append(eye_component)
            continue
        if component not in seen:
            seen.add(component)
            normalized.append(component)
    if "eye_left" in seen or "eye_right" in seen:
        if not _has_available_component(source, _RAW_EYE_UNION_COMPONENT):
            raise ValueError("Requested eye_left/eye_right finalization requires an available eyes_union source channel.")
        if not set(_EYE_COMPONENTS).issubset(seen):
            raise ValueError("Request both eye_left and eye_right, or request eyes_union, for smart eye finalization.")
    invalid = [component for component in normalized if component not in CANONICAL_COMPONENT_ORDER]
    if invalid:
        raise ValueError(f"Unsupported smart-finalizer component(s): {invalid}.")
    return tuple(component for component in CANONICAL_COMPONENT_ORDER if component in normalized)


def _review_counts_from_labels(labels: np.ndarray) -> dict[str, int]:
    needs_review = 0
    pending = 0
    for label in np.asarray(labels, dtype=object).reshape(-1):
        if "needs_review" in str(label):
            needs_review += 1
        else:
            pending += 1
    return {"pending": int(pending), "needs_review": int(needs_review)}


def _add_review_counts(target: dict[str, dict[str, int]], component_name: str, labels: np.ndarray) -> None:
    counts = _review_counts_from_labels(labels)
    existing = target.setdefault(str(component_name), {"pending": 0, "needs_review": 0})
    for key, value in counts.items():
        existing[str(key)] = int(existing.get(str(key), 0)) + int(value)


def _merge_review_counts(target: dict[str, dict[str, int]], source: Mapping[str, object]) -> None:
    for component_name, counts_raw in dict(source).items():
        existing = target.setdefault(str(component_name), {"pending": 0, "needs_review": 0})
        for key, value in dict(counts_raw or {}).items():
            existing[str(key)] = int(existing.get(str(key), 0)) + int(value)


def _row_chunks(total_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    total = max(0, int(total_rows))
    size = max(1, int(chunk_size))
    return [
        (start, min(total, start + size))
        for start in range(0, total, size)
    ]


def _parallel_worker_row_chunk_size(
    total_rows: int,
    requested_chunk_size: int,
    *,
    dense_mask_row_chunk: int | None = None,
) -> int:
    """Return a worker chunk size that avoids concurrent partial Zarr chunk writes."""

    requested = max(1, int(requested_chunk_size))
    dense_chunk = refined_subject_mask_storage_row_chunk(total_rows, dense_mask_row_chunk)
    return int(max(dense_chunk, ((requested + dense_chunk - 1) // dense_chunk) * dense_chunk))


def _worker_chunk_size_for_backend(
    total_rows: int,
    requested_chunk_size: int,
    execution_backend: str,
    *,
    dense_mask_row_chunk: int | None = None,
) -> int:
    requested = max(1, int(requested_chunk_size))
    if execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND:
        return _parallel_worker_row_chunk_size(
            total_rows,
            requested,
            dense_mask_row_chunk=dense_mask_row_chunk,
        )
    return requested


def _chunk_alignment_label(execution_backend: str, *, dense_mask_row_chunk: int | None = None) -> str:
    if execution_backend != _PROCESS_SHARD_EXECUTION_BACKEND:
        return "requested_chunk_size"
    return "dense_mask_row_chunk"


def _row_chunk_shards(
    chunk_ranges: Sequence[tuple[int, int]],
    *,
    num_workers: Optional[int],
) -> list[list[tuple[int, int, int]]]:
    """Split row chunks into contiguous worker shards.

    Each shard is processed by one worker process that opens the zarr once and
    loops over its assigned chunks. Chunks keep their original global index so
    timing/provenance remains comparable with serial execution.
    """

    indexed = [
        (int(chunk_index), int(start_row), int(stop_row))
        for chunk_index, (start_row, stop_row) in enumerate(chunk_ranges)
    ]
    if not indexed:
        return []
    requested = int(num_workers) if num_workers is not None else (os.cpu_count() or 1)
    worker_count = max(1, min(int(requested), len(indexed)))
    shards: list[list[tuple[int, int, int]]] = []
    for shard_index in range(worker_count):
        start = (len(indexed) * shard_index) // worker_count
        stop = (len(indexed) * (shard_index + 1)) // worker_count
        shard = indexed[start:stop]
        if shard:
            shards.append(shard)
    return shards


def _create_filled_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> Any:
    return group.create_array(
        name,
        shape=tuple(int(dim) for dim in shape),
        dtype=np.dtype(dtype),
        chunks=tuple(int(dim) for dim in chunks),
        fill_value=fill_value,
        overwrite=True,
    )


def _common_derived_metric_row_chunk(total_rows: int) -> int:
    return max(1, min(int(COMMON_DERIVED_METRIC_ROW_CHUNK), int(total_rows)))


def _derived_metric_chunks_2d(total_rows: int, component_count: int) -> tuple[int, int]:
    return (_common_derived_metric_row_chunk(total_rows), int(component_count))


def _derived_metric_chunks_lastdim(total_rows: int, component_count: int, width: int) -> tuple[int, int, int]:
    return (_common_derived_metric_row_chunk(total_rows), int(component_count), int(width))


def _live_metric_chunks_2d(total_rows: int) -> tuple[int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1)


def _source_lineage_map(source: SourceSubjectMaskRun) -> dict[str, object | None]:
    lineage = {
        "frame_indices": source.frame_indices,
        "frame_counts": source.frame_counts,
        "detection_indices": source.detection_indices,
        "source_crop_row_ids": source.source_crop_row_ids,
        "source_refined_row_ids": source.source_refined_row_ids,
        "source_detect_row_index": source.source_detect_row_index,
        "instance_key": source.instance_key,
    }
    for name in ("source_frame_indices", "source_clip_indices", "source_clip_local_frame_indices"):
        if name not in lineage:
            lineage[name] = source.group.get(name)
    return lineage


def _create_component_shell(
    run_group: zarr.Group,
    *,
    component_name: str,
    total_rows: int,
    height: int,
    width: int,
    retain_source_seeds: bool,
    dense_mask_row_chunk: int | None = None,
) -> zarr.Group:
    component_group = run_group.require_group("components").require_group(component_name)
    derived_metric_chunks = (_common_derived_metric_row_chunk(total_rows),)
    live_metric_chunks = (refined_subject_mask_metric_row_chunk(total_rows),)
    _create_filled_array(
        component_group,
        "mask_present",
        shape=(total_rows,),
        dtype=bool,
        chunks=derived_metric_chunks,
    )
    _create_filled_array(
        component_group,
        "area_px",
        shape=(total_rows,),
        dtype=np.float32,
        chunks=derived_metric_chunks,
    )
    _create_filled_array(
        component_group,
        "edit_applied",
        shape=(total_rows,),
        dtype=bool,
        chunks=live_metric_chunks,
    )
    component_group.attrs["source_sync_schema_id"] = REFINED_SUBJECT_SOURCE_SYNC_SCHEMA_ID
    _create_filled_array(
        component_group,
        "source_row_fingerprint",
        shape=(total_rows,),
        dtype=np.uint64,
        chunks=derived_metric_chunks,
    )
    _create_filled_array(
        component_group,
        "manual_override",
        shape=(total_rows,),
        dtype=bool,
        chunks=live_metric_chunks,
    )
    _create_filled_array(
        component_group,
        "source_row_stale",
        shape=(total_rows,),
        dtype=bool,
        chunks=live_metric_chunks,
    )
    if retain_source_seeds:
        _create_filled_array(
            component_group,
            "source_seed_masks_roi",
            shape=(total_rows, height, width),
            dtype=np.uint8,
            chunks=(
                refined_subject_mask_storage_row_chunk(total_rows, dense_mask_row_chunk),
                int(height),
                int(width),
            ),
        )
        component_group.attrs["source_seed_masks_schema_id"] = _SOURCE_SEED_MASKS_SCHEMA_ID
        component_group.attrs["source_seed_masks_status"] = "retained"
        component_group.attrs["source_seed_masks_reason"] = "retain_source_seeds=true"
    else:
        component_group.attrs["source_seed_masks_status"] = "omitted"
        component_group.attrs["source_seed_masks_reason"] = "production_default"
    if component_name == "subject_body":
        component_group.attrs["component_schema_id"] = DEFAULT_SUBJECT_BODY_COMPONENT_SCHEMA_ID
        component_group.attrs["anatomical_scope"] = DEFAULT_SUBJECT_BODY_ANATOMICAL_SCOPE
        component_group.attrs["pectoral_fin_policy"] = DEFAULT_SUBJECT_BODY_PECTORAL_FIN_POLICY

    metrics_group = component_group.require_group("metrics")
    component_group.attrs["derived_metric_row_chunk"] = int(derived_metric_chunks[0])
    component_group.attrs["derived_metric_write_policy"] = COMMON_DERIVED_METRIC_WRITE_POLICY
    metrics_group.attrs["surface_role"] = "sealed_derived_analysis"
    metrics_group.attrs["row_chunk"] = int(derived_metric_chunks[0])
    metrics_group.attrs["write_policy"] = COMMON_DERIVED_METRIC_WRITE_POLICY
    for metric_name, dtype in (
        ("component_count", np.int32),
        ("largest_component_fraction", np.float32),
        ("hole_count", np.int32),
        ("hole_area_fraction", np.float32),
    ):
        _create_filled_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=derived_metric_chunks,
        )
    for metric_name, dtype, fill_value in (
        ("sigma_noise", np.float32, np.nan),
        ("curvature_var", np.float32, np.nan),
        ("ipr", np.float32, np.nan),
        ("solidity", np.float32, np.nan),
    ):
        _create_filled_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=derived_metric_chunks,
            fill_value=fill_value,
        )
    return component_group


def _create_finalization_metric_shell(
    component_group: zarr.Group,
    *,
    batch: _FinalizedComponentBatch | None = None,
    metric_names: Sequence[str] = (),
    total_rows: int,
    write_policy: str = FINALIZATION_METRIC_WRITE_POLICY,
) -> None:
    metrics_group = component_group.require_group("finalization_metrics")
    metric_chunks = (max(1, min(int(FINALIZATION_METRIC_ROW_CHUNK), int(total_rows))),)
    names_source = metric_names or (batch.metrics if batch is not None else ())
    names = sorted(str(name) for name in names_source)
    for metric_name in names:
        if metric_name not in metrics_group:
            _create_filled_array(
                metrics_group,
                metric_name,
                shape=(total_rows,),
                dtype=np.float32,
                chunks=metric_chunks,
            )
    if "quality_code" not in metrics_group:
        _create_filled_array(
            metrics_group,
            "quality_code",
            shape=(total_rows,),
            dtype=np.int16,
            chunks=metric_chunks,
        )
    if "quality_score" not in metrics_group:
        _create_filled_array(
            metrics_group,
            "quality_score",
            shape=(total_rows,),
            dtype=np.float32,
            chunks=metric_chunks,
        )
    metrics_group.attrs["schema_id"] = "refined_subject_component_finalization_metrics_v1"
    metrics_group.attrs["method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    metrics_group.attrs["surface_role"] = "sealed_derived_analysis"
    metrics_group.attrs["write_policy"] = str(write_policy)
    metrics_group.attrs["row_chunk"] = int(metric_chunks[0])
    if batch is not None:
        metrics_group.attrs["source_component"] = str(batch.component_name)
        metrics_group.attrs["source_surface_path"] = str(batch.source_surface_path)
        metrics_group.attrs["source_surface_kind"] = str(batch.source_surface_kind)


def _create_bitpacked_mask_store_shell(
    run_group: zarr.Group,
    *,
    total_rows: int,
    component_names: Sequence[str],
    height: int,
    width: int,
    row_chunk: int,
    extra_attrs: Mapping[str, object] | None = None,
) -> zarr.Group:
    names = tuple(str(value) for value in component_names)
    n_rows = int(total_rows)
    n_channels = int(len(names))
    mask_height = int(height)
    mask_width = int(width)
    packed_width = packed_width_bytes(mask_width)
    if "mask_bitpacked" in run_group:
        del run_group["mask_bitpacked"]
    group = run_group.create_group("mask_bitpacked")
    group.attrs.update(
        {
            "schema_id": MASK_BITPACKED_SCHEMA_ID,
            "mask_encoding": MASK_BITPACKED_ENCODING,
            "mask_value_semantics": MASK_BITPACKED_VALUE_SEMANTICS,
            "layout": MASK_BITPACKED_LAYOUT,
            "logical_shape": [n_rows, n_channels, mask_height, mask_width],
            "encoded_shape": [n_rows, n_channels, mask_height, int(packed_width)],
            "component_names": list(names),
            "packed_axis": MASK_BITPACKED_AXIS,
            "packed_bitorder": MASK_BITPACKED_BITORDER,
            "packed_width_bytes": int(packed_width),
            "source_encoding": "chunk_finalizer_binary_masks",
        }
    )
    if extra_attrs:
        group.attrs.update(dict(extra_attrs))
    group.create_array(
        "masks_packed",
        shape=(n_rows, n_channels, mask_height, int(packed_width)),
        chunks=refined_subject_mask_bitpacked_chunks(
            n_rows,
            n_channels,
            mask_height,
            mask_width,
            row_chunk=max(1, int(row_chunk)),
        ),
        dtype="uint8",
        fill_value=0,
        overwrite=True,
    )
    run_group.attrs["mask_bitpacked_schema_id"] = MASK_BITPACKED_SCHEMA_ID
    run_group.attrs["mask_bitpacked_encoding"] = MASK_BITPACKED_ENCODING
    run_group.attrs["mask_bitpacked_layout"] = MASK_BITPACKED_LAYOUT
    return group


def _create_refined_run_shell(
    *,
    refined_parent: zarr.Group,
    target_run: str,
    source: SourceSubjectMaskRun,
    component_names: Sequence[str],
    extra_attrs: Mapping[str, object],
    provenance_inputs: Mapping[str, object],
    stage_command: Optional[str],
    retain_source_seeds: bool,
    dense_mask_row_chunk: int | None = None,
    create_dense_masks: bool = True,
    create_bitpacked_masks: bool = False,
) -> zarr.Group:
    total_rows = int(source.masks_roi.shape[0])
    height = int(source.masks_roi.shape[2])
    width = int(source.masks_roi.shape[3])
    dense_storage_chunks = refined_subject_mask_storage_chunks(
        total_rows,
        height,
        width,
        dense_mask_row_chunk,
    )
    component_names = tuple(str(name) for name in component_names)
    created = _utc_now()

    run_group = refined_parent.create_group(target_run)
    run_group.create_array(
        "detection_source",
        data=np.asarray(source.detection_source[:], dtype=np.int8),
        chunks=(refined_subject_mask_metric_row_chunk(total_rows),),
        overwrite=True,
    )
    if create_dense_masks:
        _create_filled_array(
            run_group,
            "masks_roi",
            shape=(total_rows, len(component_names), height, width),
            dtype=np.uint8,
            chunks=dense_storage_chunks,
        )
    if create_bitpacked_masks:
        _create_bitpacked_mask_store_shell(
            run_group,
            total_rows=total_rows,
            component_names=component_names,
            height=height,
            width=width,
            row_chunk=int(dense_storage_chunks[0]),
            extra_attrs={
                "source_run": str(target_run),
                "source_array": "chunk_finalizer_binary_masks",
            },
        )
    run_group.create_array(
        "available_channels",
        data=np.ones((len(component_names),), dtype=bool),
        overwrite=True,
    )
    _create_filled_array(
        run_group,
        "edit_applied",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_live_metric_chunks_2d(total_rows),
    )
    copy_row_lineage_arrays_from_sources(run_group, _source_lineage_map(source), total_rois=total_rows)

    metrics_group = run_group.require_group("metrics")
    _create_filled_array(
        metrics_group,
        "mask_present",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, len(component_names)),
    )
    _create_filled_array(
        metrics_group,
        "area_px",
        shape=(total_rows, len(component_names)),
        dtype=np.float32,
        chunks=_derived_metric_chunks_2d(total_rows, len(component_names)),
    )
    _create_filled_array(
        metrics_group,
        "centroid_xy",
        shape=(total_rows, len(component_names), 2),
        dtype=np.float32,
        chunks=_derived_metric_chunks_lastdim(total_rows, len(component_names), 2),
        fill_value=np.nan,
    )
    _create_filled_array(
        metrics_group,
        "centroid_valid",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, len(component_names)),
    )
    _create_filled_array(
        metrics_group,
        "bbox_xyxy",
        shape=(total_rows, len(component_names), 4),
        dtype=np.float32,
        chunks=_derived_metric_chunks_lastdim(total_rows, len(component_names), 4),
    )
    _create_filled_array(
        metrics_group,
        "bbox_valid",
        shape=(total_rows, len(component_names)),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, len(component_names)),
    )
    metrics_group.attrs["surface_role"] = "sealed_derived_analysis"
    metrics_group.attrs["row_chunk"] = int(_common_derived_metric_row_chunk(total_rows))
    metrics_group.attrs["component_chunk"] = int(len(component_names))
    metrics_group.attrs["write_policy"] = COMMON_DERIVED_METRIC_WRITE_POLICY

    for component_name in component_names:
        _create_component_shell(
            run_group,
            component_name=component_name,
            total_rows=total_rows,
            height=height,
            width=width,
            retain_source_seeds=retain_source_seeds,
            dense_mask_row_chunk=dense_mask_row_chunk,
        )

    if source.run_name:
        run_group.attrs["source_subject_mask_run"] = source.run_name
    if source.source_method:
        run_group.attrs["source_subject_mask_method"] = source.source_method
    if source.source_keypoints_run:
        run_group.attrs.update(build_source_keypoints_attrs(source.source_keypoints_run, include_legacy_alias=True))
    if source.source_keypoint_group:
        run_group.attrs["source_keypoint_group"] = source.source_keypoint_group
    run_group.attrs["source_crop_run"] = source.crop_run
    run_group.attrs.update(source.source_crop_snapshot)
    run_group.attrs["mask_labels"] = list(component_names)
    run_group.attrs["dense_mask_row_chunk"] = int(dense_storage_chunks[0])
    run_group.attrs["dense_mask_storage_chunks"] = [int(value) for value in dense_storage_chunks]
    run_group.attrs["label_schema_id"] = _infer_refined_label_schema_id(component_names)
    run_group.attrs["output_semantics"] = "multilabel"
    run_group.attrs["refinement_semantics"] = "canonical_component_masks"
    run_group.attrs["method"] = SMART_FINALIZE_SUBJECT_MASKS_METHOD
    run_group.attrs["created_at_utc"] = created
    run_group.attrs["created_utc"] = created
    run_group.attrs["duration_seconds"] = 0.0
    for key, value in extra_attrs.items():
        run_group.attrs[str(key)] = value

    component_reviews = {
        component_name: _review_payload(
            state="pending",
            method=DEFAULT_REVIEW_METHOD,
            intended_use=DEFAULT_REVIEW_INTENDED_USE,
        )
        for component_name in component_names
    }
    run_group.attrs["component_review_statuses"] = component_reviews
    run_group.attrs["refined_subject_mask_review_status"] = _review_payload(
        state="pending",
        method=DEFAULT_REVIEW_METHOD,
        intended_use=DEFAULT_REVIEW_INTENDED_USE,
        notes="auto_initialized_from_components",
    )

    git_info = get_git_info(repo_path=Path(__file__).resolve().parents[3])
    env_info = get_environment_info(
        include_all_packages=False,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    stage_inputs_payload = {
        "source_subject_mask_run": source.run_name,
        "source_subject_mask_method": source.source_method,
        "source_crop_run": source.crop_run,
        **source.source_crop_snapshot,
        "source_keypoints_run": source.source_keypoints_run,
        "source_keypoint_group": source.source_keypoint_group,
    }
    stage_inputs_payload.update({str(key): value for key, value in provenance_inputs.items()})
    provenance = build_stage_provenance(
        stage=REFINED_SUBJECT_STAGE_NAME,
        command=stage_command or (" ".join(sys.argv) if sys.argv else "unknown"),
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
        parameters={
            "method": SMART_FINALIZE_SUBJECT_MASKS_METHOD,
            "refinement_semantics": "canonical_component_masks",
            "components": list(component_names),
            "component_count": int(len(component_names)),
            "metric_level": provenance_inputs.get("component_metric_level"),
            "chunk_size": provenance_inputs.get("chunk_size"),
            "worker_chunk_size": provenance_inputs.get("worker_chunk_size"),
            "chunk_count": provenance_inputs.get("chunk_count"),
            "dense_mask_row_chunk": provenance_inputs.get("dense_mask_row_chunk"),
            "dense_mask_storage_chunks": provenance_inputs.get("dense_mask_storage_chunks"),
            "finalization_metric_row_chunk": provenance_inputs.get("finalization_metric_row_chunk"),
            "finalization_metric_write_policy": provenance_inputs.get(
                "finalization_metric_write_policy"
            ),
            "common_metric_row_chunk": provenance_inputs.get("common_metric_row_chunk"),
            "common_metric_component_chunk": provenance_inputs.get("common_metric_component_chunk"),
            "common_metric_write_policy": provenance_inputs.get("common_metric_write_policy"),
            "write_eye_geometry": provenance_inputs.get("eye_geometry_requested"),
            "write_component_contours": provenance_inputs.get("component_contours_requested"),
            "write_sampled_component_contours": provenance_inputs.get(
                "sampled_component_contours_requested"
            ),
            "sampled_contour_counts": provenance_inputs.get("sampled_contour_counts"),
            "sampled_contour_row_chunk": provenance_inputs.get("sampled_contour_row_chunk"),
            "retain_source_seeds": provenance_inputs.get("retain_source_seeds"),
            "source_seed_masks_status": provenance_inputs.get("source_seed_masks_status"),
            "execution_backend": provenance_inputs.get("execution_backend"),
            "process_shard_execution_enabled": provenance_inputs.get("process_shard_execution_enabled"),
            "worker_process_count": provenance_inputs.get("worker_process_count"),
            "requested_chunk_size": provenance_inputs.get("requested_chunk_size"),
            "chunk_alignment": provenance_inputs.get("chunk_alignment"),
        },
        inputs=stage_inputs_payload,
    )
    write_stage_provenance(run_group, provenance)
    return run_group


def _component_metrics_from_finalization_batch(
    batch: _FinalizedComponentBatch,
    *,
    row_count: int,
) -> dict[str, np.ndarray]:
    """Reuse topology metrics already computed during smart component finalization."""

    component_metrics: dict[str, np.ndarray] = {}
    for target_name, (source_name, dtype) in _COMPONENT_METRIC_FINALIZATION_SOURCES.items():
        values = batch.metrics.get(source_name)
        if values is None:
            continue
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        if int(arr.shape[0]) != int(row_count):
            continue
        component_metrics[str(target_name)] = arr
    return component_metrics


def _component_metrics_from_assigned_eye_masks(masks: np.ndarray) -> dict[str, np.ndarray]:
    """Seed topology metrics for assignment-derived eye masks.

    ``assign_eyes_union_to_lr`` writes either an empty mask or one selected
    connected component per eye. Hole metrics still require a background pass,
    but foreground component count and largest-component fraction are known.
    """

    binary = np.asarray(masks, dtype=np.uint8) > 0
    if binary.ndim != 3:
        raise ValueError(f"Expected assigned eye masks with shape (N,H,W), got {tuple(binary.shape)}.")
    present = np.any(binary, axis=(1, 2))
    return {
        "component_count": present.astype(np.int32, copy=False),
        "largest_component_fraction": present.astype(np.float32, copy=False),
    }


def _eye_geometry_from_assignment_result(assignment: EyesUnionAssignmentResult) -> dict[str, object]:
    row_count = int(next(iter(assignment.masks.values())).shape[0]) if assignment.masks else 0
    geometry = dict(assignment.eye_geometry)
    ellipse_params = np.asarray(geometry.get("ellipse_params"), dtype=np.float32).reshape(row_count, 2, 5)
    ellipse_success = np.asarray(geometry.get("ellipse_success"), dtype=bool).reshape(row_count, 2)
    centroids = np.asarray(geometry.get("centroids"), dtype=np.float32).reshape(row_count, 2, 2)

    separation_px = np.full((row_count,), np.nan, dtype=np.float32)
    separation_valid = np.zeros((row_count,), dtype=bool)
    if row_count > 0:
        pair_valid = np.all(ellipse_success, axis=1) & np.all(np.isfinite(centroids), axis=(1, 2))
        separation_valid[pair_valid] = True
        separation_px[pair_valid] = np.linalg.norm(
            centroids[pair_valid, 0] - centroids[pair_valid, 1],
            axis=1,
        ).astype(np.float32, copy=False)

    contours_raw = geometry.get("contours")
    contours_by_component = contours_raw if isinstance(contours_raw, Mapping) else {}
    eye_contours = {name: _empty_local_contour_pack(row_count) for name in _EYE_COMPONENTS}
    for component_name in _EYE_COMPONENTS:
        contour_values = contours_by_component.get(component_name)
        if contour_values is None:
            continue
        for local_idx, contour in enumerate(contour_values):
            if local_idx >= row_count:
                break
            _add_local_contour(eye_contours[component_name], local_idx, contour, min_points=1)

    return {
        "ellipse_params": ellipse_params,
        "ellipse_success": ellipse_success,
        "separation_px": separation_px,
        "separation_valid": separation_valid,
        "contours": {name: _finalize_local_contour_pack(pack) for name, pack in eye_contours.items()},
        "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
        "pair_success_count": int(np.count_nonzero(separation_valid)),
        "source": _ASSIGNMENT_REUSE_POSTCOMPUTE_BACKEND,
    }


def _compute_component_hole_metrics(masks_roi: np.ndarray) -> dict[str, np.ndarray]:
    """Compute only hole topology metrics for a component row chunk."""

    binary = np.asarray(masks_roi, dtype=np.uint8) > 0
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}.")
    total = int(binary.shape[0])
    hole_count = np.zeros((total,), dtype=np.int32)
    hole_area_fraction = np.zeros((total,), dtype=np.float32)
    for row_idx in range(total):
        mask = np.asarray(binary[row_idx], dtype=bool)
        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue
        background_input = (~mask).astype(np.uint8, copy=False)
        num_bg_labels, bg_labels = cv2.connectedComponents(background_input, connectivity=8)
        if num_bg_labels <= 1:
            continue
        border_labels = set()
        border_labels.update(int(value) for value in bg_labels[0, :].tolist())
        border_labels.update(int(value) for value in bg_labels[-1, :].tolist())
        border_labels.update(int(value) for value in bg_labels[:, 0].tolist())
        border_labels.update(int(value) for value in bg_labels[:, -1].tolist())
        hole_labels = [label for label in range(1, int(num_bg_labels)) if label not in border_labels]
        if not hole_labels:
            continue
        bg_counts = np.bincount(bg_labels.reshape(-1), minlength=int(num_bg_labels))
        hole_area = int(sum(int(bg_counts[int(label)]) for label in hole_labels))
        hole_count[row_idx] = np.int32(len(hole_labels))
        hole_area_fraction[row_idx] = np.float32(hole_area / float(area))
    return {
        "hole_count": hole_count,
        "hole_area_fraction": hole_area_fraction,
    }


def _compute_component_spatial_metrics(masks: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mask-local spatial metrics for one component over a row chunk."""

    binary = np.asarray(masks, dtype=np.uint8)
    if binary.ndim != 3:
        raise ValueError(f"Expected component masks with shape (N,H,W), got {tuple(binary.shape)}.")
    return batch_mask_spatial_metrics(binary)


def _write_component_metrics_chunk(
    component_group: zarr.Group,
    *,
    row_slice: slice,
    masks: np.ndarray,
    metric_level: str,
    precomputed_metrics: Optional[Mapping[str, np.ndarray]] = None,
    write_attrs: bool = True,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> dict[str, np.ndarray]:
    component_name = str(component_group.name).rstrip("/").split("/")[-1]
    component_metrics = _compute_component_metrics_payload(
        component_name=component_name,
        masks=masks,
        metric_level=metric_level,
        precomputed_metrics=precomputed_metrics,
        timing=timing,
        chunk_timing=chunk_timing,
    )
    metrics_group = component_group["metrics"]
    with _timed_chunk_phase(timing, chunk_timing, f"write_component_metric_arrays_{component_name}"):
        for metric_name, values in component_metrics.items():
            metrics_group[str(metric_name)][row_slice] = np.asarray(values)
    if write_attrs:
        _set_component_metric_attrs(component_group, metric_level=metric_level)
    return component_metrics


def _compute_component_metrics_payload(
    *,
    component_name: str,
    masks: np.ndarray,
    metric_level: str,
    precomputed_metrics: Optional[Mapping[str, np.ndarray]] = None,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> dict[str, np.ndarray]:
    """Compute component metrics without opening or mutating a destination."""

    if metric_level not in _METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {_METRIC_LEVELS}; got {metric_level!r}.")
    component_metrics: dict[str, np.ndarray] = {
        str(name): np.asarray(values)
        for name, values in dict(precomputed_metrics or {}).items()
    }
    missing_topology = [
        name
        for name in _COMPONENT_METRIC_FINALIZATION_SOURCES
        if name not in component_metrics
    ]
    if missing_topology:
        if set(missing_topology).issubset({"hole_count", "hole_area_fraction"}):
            with _timed_chunk_phase(timing, chunk_timing, f"compute_hole_metrics_{component_name}"):
                computed_topology = _compute_component_hole_metrics(masks)
        else:
            with _timed_chunk_phase(timing, chunk_timing, f"compute_topology_metrics_{component_name}"):
                computed_topology = _compute_component_topology_metrics(masks)
        for name in missing_topology:
            component_metrics[str(name)] = np.asarray(computed_topology[str(name)])
    if metric_level == "full":
        with _timed_chunk_phase(timing, chunk_timing, f"compute_sigma_noise_metrics_{component_name}"):
            component_metrics.update(_compute_component_sigma_noise_metrics(masks))
        with _timed_chunk_phase(timing, chunk_timing, f"compute_curvature_var_metrics_{component_name}"):
            component_metrics.update(_compute_component_curvature_var_metrics(masks))
        with _timed_chunk_phase(timing, chunk_timing, f"compute_shape_qc_metrics_{component_name}"):
            component_metrics.update(_compute_component_shape_qc_metrics(masks))
    return component_metrics


def _set_component_metric_attrs(component_group: zarr.Group, *, metric_level: str) -> None:
    metrics_group = component_group["metrics"]
    component_name = str(component_group.name).rstrip("/").split("/")[-1]
    metrics_group.attrs["schema_id"] = _COMPONENT_METRICS_SCHEMA_ID
    metrics_group.attrs["schema_version"] = 1
    metrics_group.attrs["component_name"] = component_name
    metrics_group.attrs["metric_level"] = metric_level
    metrics_group.attrs["metric_names"] = list(_COMPONENT_METRIC_NAMES)
    metrics_group.attrs["computed_metric_names"] = (
        list(_COMPONENT_METRIC_NAMES)
        if metric_level == "full"
        else ["component_count", "largest_component_fraction", "hole_count", "hole_area_fraction"]
    )
    metrics_group.attrs["deferred_metric_names"] = (
        []
        if metric_level == "full"
        else ["sigma_noise", "curvature_var", "ipr", "solidity"]
    )
    metrics_group.attrs["qc_schema_id"] = _COMPONENT_METRIC_QC_SCHEMA_ID
    metrics_group.attrs["qc_policy"] = _component_metric_qc_policy_payload(component_name)
    component_group.attrs["component_metric_level"] = metric_level
    component_group.attrs["component_metrics_schema_id"] = _COMPONENT_METRICS_SCHEMA_ID
    component_group.attrs["metric_qc_schema_id"] = _COMPONENT_METRIC_QC_SCHEMA_ID
    if metric_level == "full":
        component_group.attrs["shape_qc_metrics_status"] = "computed"
        component_group.attrs.pop("shape_qc_metrics_deferred_reason", None)
    else:
        component_group.attrs["shape_qc_metrics_status"] = "deferred"
        component_group.attrs["shape_qc_metrics_deferred_reason"] = "metric_level=cheap"


def _write_mask_local_metrics_chunk(
    run_group: zarr.Group,
    *,
    component_name: str,
    component_idx: int,
    row_slice: slice,
    masks: np.ndarray,
    metric_level: str,
    precomputed_component_metrics: Optional[Mapping[str, np.ndarray]] = None,
    write_metric_attrs: bool = True,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> _ComponentMetricWriteResult:
    payload = _compute_mask_local_metric_payload(
        component_name=component_name,
        masks=masks,
        metric_level=metric_level,
        precomputed_component_metrics=precomputed_component_metrics,
        timing=timing,
        chunk_timing=chunk_timing,
    )
    return _write_mask_local_metric_payload(
        run_group,
        component_name=component_name,
        component_idx=component_idx,
        row_slice=row_slice,
        payload=payload,
        metric_level=metric_level,
        write_metric_attrs=write_metric_attrs,
        timing=timing,
        chunk_timing=chunk_timing,
    )


def _write_mask_local_metric_payload(
    run_group: zarr.Group,
    *,
    component_name: str,
    component_idx: int,
    row_slice: slice,
    payload: _ComponentMetricPayload,
    metric_level: str,
    write_metric_attrs: bool = True,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> _ComponentMetricWriteResult:
    spatial_metrics = payload.spatial_metrics
    mask_present = np.asarray(spatial_metrics["mask_present"], dtype=bool)
    area_px = np.asarray(spatial_metrics["area_px"], dtype=np.float32)
    with _timed_chunk_phase(timing, chunk_timing, f"write_run_spatial_metrics_{component_name}"):
        run_group["metrics/mask_present"][row_slice, int(component_idx)] = mask_present
        run_group["metrics/area_px"][row_slice, int(component_idx)] = area_px
        run_group["metrics/centroid_xy"][row_slice, int(component_idx), :] = np.asarray(
            spatial_metrics["centroid_xy"], dtype=np.float32
        )
        run_group["metrics/centroid_valid"][row_slice, int(component_idx)] = np.asarray(
            spatial_metrics["centroid_valid"], dtype=bool
        )
        run_group["metrics/bbox_xyxy"][row_slice, int(component_idx), :] = np.asarray(
            spatial_metrics["bbox_xyxy"], dtype=np.float32
        )
        run_group["metrics/bbox_valid"][row_slice, int(component_idx)] = np.asarray(
            spatial_metrics["bbox_valid"], dtype=bool
        )

    component_group = run_group["components"][component_name]
    with _timed_chunk_phase(timing, chunk_timing, f"write_component_spatial_metrics_{component_name}"):
        component_group["mask_present"][row_slice] = mask_present
        component_group["area_px"][row_slice] = area_px
    metrics_group = component_group["metrics"]
    with _timed_chunk_phase(timing, chunk_timing, f"write_component_metric_arrays_{component_name}"):
        for metric_name, values in payload.component_metrics.items():
            metrics_group[str(metric_name)][row_slice] = np.asarray(values)
    if write_metric_attrs:
        _set_component_metric_attrs(component_group, metric_level=metric_level)
    return _ComponentMetricWriteResult(
        mask_present=np.asarray(mask_present, dtype=bool),
        reason_labels=np.asarray(payload.reason_labels, dtype=object),
    )


def _compute_mask_local_metric_payload(
    *,
    component_name: str,
    masks: np.ndarray,
    metric_level: str,
    precomputed_component_metrics: Optional[Mapping[str, np.ndarray]] = None,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> _ComponentMetricPayload:
    """Compute all fixed-shape mask-local metrics without destination I/O."""

    masks_u8 = np.asarray(masks, dtype=np.uint8)
    with _timed_chunk_phase(timing, chunk_timing, f"compute_spatial_metrics_{component_name}"):
        spatial_metrics = {
            str(name): np.asarray(values)
            for name, values in _compute_component_spatial_metrics(masks_u8).items()
        }
    component_metrics = _compute_component_metrics_payload(
        component_name=component_name,
        masks=masks_u8,
        metric_level=metric_level,
        precomputed_metrics=precomputed_component_metrics,
        timing=timing,
        chunk_timing=chunk_timing,
    )
    with _timed_chunk_phase(timing, chunk_timing, f"compute_metric_qc_reasons_{component_name}"):
        reason_labels = _compute_component_metric_qc_reason_labels(
            component_name,
            mask_present=np.asarray(spatial_metrics["mask_present"], dtype=bool),
            area_px=np.asarray(spatial_metrics["area_px"], dtype=np.float32),
            component_metrics=component_metrics,
        )
    return _ComponentMetricPayload(
        spatial_metrics=spatial_metrics,
        component_metrics=component_metrics,
        reason_labels=np.asarray(reason_labels, dtype=object),
    )


def _write_canonical_component_chunk(
    run_group: zarr.Group,
    *,
    component_name: str,
    component_idx: int,
    row_slice: slice,
    masks: np.ndarray,
    source_masks: np.ndarray,
    metric_level: str,
    precomputed_component_metrics: Optional[Mapping[str, np.ndarray]] = None,
    write_metric_attrs: bool = True,
    write_derived_metrics: bool = True,
    retain_source_seeds: bool = False,
    timing: Optional[_TimingRecorder] = None,
    chunk_timing: Optional[dict[str, object]] = None,
) -> _CanonicalComponentChunkResult:
    masks_u8 = np.asarray(masks, dtype=np.uint8)
    source_u8 = np.asarray(source_masks, dtype=np.uint8)
    if "masks_roi" in run_group:
        with _timed_chunk_phase(timing, chunk_timing, f"write_masks_roi_{component_name}"):
            run_group["masks_roi"][row_slice, int(component_idx)] = masks_u8
    elif "mask_bitpacked" in run_group:
        with _timed_chunk_phase(timing, chunk_timing, f"write_mask_bitpacked_{component_name}"):
            run_group["mask_bitpacked/masks_packed"][
                row_slice,
                int(component_idx) : int(component_idx) + 1,
            ] = pack_binary_mask_stack(masks_u8[:, None, :, :])
    else:
        raise RuntimeError("Refined subject-mask run has neither masks_roi nor mask_bitpacked storage.")
    component_group = run_group["components"][component_name]
    if retain_source_seeds:
        with _timed_chunk_phase(timing, chunk_timing, f"write_source_seed_masks_{component_name}"):
            component_group["source_seed_masks_roi"][row_slice] = source_u8
    with _timed_chunk_phase(timing, chunk_timing, f"compute_source_row_fingerprint_{component_name}"):
        source_row_fingerprint = _compute_mask_row_fingerprints(source_u8)
    metric_payload = _compute_mask_local_metric_payload(
        component_name=component_name,
        masks=masks_u8,
        metric_level=metric_level,
        precomputed_component_metrics=precomputed_component_metrics,
        timing=timing,
        chunk_timing=chunk_timing,
    )
    if write_derived_metrics:
        with _timed_chunk_phase(timing, chunk_timing, f"write_source_row_fingerprint_{component_name}"):
            component_group["source_row_fingerprint"][row_slice] = source_row_fingerprint
        write_result = _write_mask_local_metric_payload(
            run_group,
            component_name=component_name,
            component_idx=component_idx,
            row_slice=row_slice,
            payload=metric_payload,
            metric_level=metric_level,
            write_metric_attrs=write_metric_attrs,
            timing=timing,
            chunk_timing=chunk_timing,
        )
    else:
        write_result = _ComponentMetricWriteResult(
            mask_present=np.asarray(metric_payload.spatial_metrics["mask_present"], dtype=bool),
            reason_labels=np.asarray(metric_payload.reason_labels, dtype=object),
        )
    return _CanonicalComponentChunkResult(
        mask_present=np.asarray(write_result.mask_present, dtype=bool),
        reason_labels=np.asarray(write_result.reason_labels, dtype=object),
        spatial_metrics={str(name): np.asarray(values) for name, values in metric_payload.spatial_metrics.items()},
        component_metrics={str(name): np.asarray(values) for name, values in metric_payload.component_metrics.items()},
        source_row_fingerprint=np.asarray(source_row_fingerprint, dtype=np.uint64),
    )


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): int(idx) for idx, label in enumerate(labels_raw)}


def _available_channels_for_run(group: zarr.Group, channel_count: int) -> np.ndarray:
    available_arr = group.get("available_channels")
    if available_arr is None:
        return np.ones((int(channel_count),), dtype=bool)
    available = np.asarray(available_arr[:], dtype=bool).reshape(-1)
    if int(available.shape[0]) >= int(channel_count):
        return available[: int(channel_count)]
    padded = np.zeros((int(channel_count),), dtype=bool)
    padded[: int(available.shape[0])] = available
    return padded


def _empty_local_contour_pack(row_count: int) -> dict[str, object]:
    return {
        "ptr": np.full((int(row_count),), -1, dtype=np.int64),
        "len": np.zeros((int(row_count),), dtype=np.int32),
        "points": [],
        "point_count": 0,
        "contour_count": 0,
    }


def _add_local_contour(pack: dict[str, object], row_index: int, contour: np.ndarray | None, *, min_points: int) -> None:
    if contour is None:
        return
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if int(points.shape[0]) < int(min_points):
        return
    ptr = pack["ptr"]
    length = pack["len"]
    points_list = pack["points"]
    if not isinstance(ptr, np.ndarray) or not isinstance(length, np.ndarray) or not isinstance(points_list, list):
        raise TypeError("Invalid local contour pack.")
    local_offset = int(pack["point_count"])
    ptr[int(row_index)] = np.int64(local_offset)
    length[int(row_index)] = np.int32(points.shape[0])
    points_list.append(points)
    pack["point_count"] = local_offset + int(points.shape[0])
    pack["contour_count"] = int(pack["contour_count"]) + 1


def _finalize_local_contour_pack(pack: dict[str, object]) -> dict[str, object]:
    points_list = pack["points"]
    if not isinstance(points_list, list):
        raise TypeError("Invalid local contour pack points list.")
    points_xy = (
        np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
        if points_list
        else np.zeros((0, 2), dtype=np.float32)
    )
    return {
        "ptr": pack["ptr"],
        "len": pack["len"],
        "points_xy": points_xy,
        "point_count": int(pack["point_count"]),
        "contour_count": int(pack["contour_count"]),
    }


def _sample_packed_contours(pack: Mapping[str, object], sample_count: int) -> dict[str, np.ndarray]:
    """Convert one row-indexed ragged pack into a fixed-K numeric payload."""

    ptr = np.asarray(pack["ptr"], dtype=np.int64).reshape(-1)
    length = np.asarray(pack["len"], dtype=np.int32).reshape(-1)
    points = np.asarray(pack["points_xy"], dtype=np.float32).reshape(-1, 2)
    row_count = int(length.shape[0])
    k = int(sample_count)
    if k <= 0:
        raise ValueError("sample_count must be positive.")
    sampled = np.full((row_count, k, 2), np.nan, dtype=np.float32)
    valid = np.zeros((row_count,), dtype=bool)
    source_point_count = np.maximum(length, 0).astype(np.int32, copy=True)
    for row_index in np.flatnonzero(length >= 2):
        offset = int(ptr[int(row_index)])
        count = int(length[int(row_index)])
        if offset < 0 or offset + count > int(points.shape[0]):
            continue
        row_points = resample_closed_contour(points[offset : offset + count], k)
        if not bool(np.isfinite(row_points).all()):
            continue
        sampled[int(row_index)] = row_points
        valid[int(row_index)] = True
    return {
        "points_xy": sampled,
        "valid": valid,
        "source_point_count": source_point_count,
    }


def _compute_refined_subject_postcompute_shard(
    zarr_path: str,
    refined_run: str,
    start_row: int,
    stop_row: int,
    *,
    write_eye_geometry: bool,
    write_component_contours: bool,
    sampled_contour_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    from ..shared.mask_geometry import measure_mask_ellipse as _measure_mask

    started = time.perf_counter()
    root = open_zarr_root(zarr_path, mode="r")
    run_group = root["refined_subject_masks_runs"][refined_run]
    masks_roi = run_group["masks_roi"]
    label_map = _label_index_map(run_group)
    channel_count = int(masks_roi.shape[1])
    available = _available_channels_for_run(run_group, channel_count)
    start = int(start_row)
    stop = int(stop_row)
    row_count = max(0, stop - start)
    masks = np.asarray(masks_roi[start:stop], dtype=np.uint8)

    sampled_counts = {str(name): int(value) for name, value in dict(sampled_contour_counts or {}).items()}
    raw_contour_packs: dict[str, dict[str, object]] = {}
    eye_payload: dict[str, object] | None = None
    if write_eye_geometry and set(_EYE_COMPONENTS).issubset(label_map):
        ellipse_params = np.full((row_count, 2, 5), np.nan, dtype=np.float32)
        ellipse_success = np.zeros((row_count, 2), dtype=bool)
        separation_px = np.full((row_count,), np.nan, dtype=np.float32)
        separation_valid = np.zeros((row_count,), dtype=bool)
        centroids = np.full((row_count, 2, 2), np.nan, dtype=np.float32)
        eye_contours = {name: _empty_local_contour_pack(row_count) for name in _EYE_COMPONENTS}

        for local_idx in range(row_count):
            for eye_idx, component_name in enumerate(_EYE_COMPONENTS):
                comp_idx = int(label_map[component_name])
                if comp_idx >= int(available.shape[0]) or not bool(available[comp_idx]):
                    continue
                success, ellipse, centroid, contour, _failure = _measure_mask(masks[local_idx, comp_idx])
                ellipse_params[local_idx, eye_idx] = np.asarray(ellipse, dtype=np.float32)
                ellipse_success[local_idx, eye_idx] = bool(success)
                centroids[local_idx, eye_idx] = np.asarray(centroid, dtype=np.float32)
                _add_local_contour(eye_contours[component_name], local_idx, contour, min_points=1)
            if bool(np.all(ellipse_success[local_idx])) and bool(np.all(np.isfinite(centroids[local_idx]))):
                separation_px[local_idx] = np.float32(np.linalg.norm(centroids[local_idx, 0] - centroids[local_idx, 1]))
                separation_valid[local_idx] = True

        finalized_eye_contours = {
            name: _finalize_local_contour_pack(pack) for name, pack in eye_contours.items()
        }
        raw_contour_packs.update(finalized_eye_contours)
        eye_payload = {
            "ellipse_params": ellipse_params,
            "ellipse_success": ellipse_success,
            "separation_px": separation_px,
            "separation_valid": separation_valid,
            "contours": finalized_eye_contours,
            "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
            "pair_success_count": int(np.count_nonzero(separation_valid)),
        }

    raw_components = set(sampled_counts)
    if write_component_contours:
        raw_components.update(_COMPONENT_CONTOUR_COMPONENTS)
    for component_name in sorted(raw_components):
        if component_name in raw_contour_packs:
            continue
        if component_name not in label_map:
            continue
        comp_idx = int(label_map[component_name])
        if comp_idx >= int(available.shape[0]) or not bool(available[comp_idx]):
            continue
        pack = _empty_local_contour_pack(row_count)
        for local_idx in range(row_count):
            contour = extract_largest_external_contour(masks[local_idx, comp_idx], min_points=2)
            _add_local_contour(pack, local_idx, contour, min_points=2)
        raw_contour_packs[component_name] = _finalize_local_contour_pack(pack)

    contour_payload: dict[str, object] = {}
    if write_component_contours:
        for component_name in _COMPONENT_CONTOUR_COMPONENTS:
            if component_name not in label_map:
                continue
            if component_name in raw_contour_packs:
                contour_payload[component_name] = raw_contour_packs[component_name]

    sampled_payload = {
        component_name: _sample_packed_contours(raw_contour_packs[component_name], sample_count)
        for component_name, sample_count in sampled_counts.items()
        if component_name in raw_contour_packs
    }

    return {
        "start_row": start,
        "stop_row": stop,
        "row_count": int(row_count),
        "duration_seconds": float(time.perf_counter() - started),
        "eye_geometry": eye_payload,
        "component_contours": contour_payload,
        "sampled_component_contours": sampled_payload,
    }


def _merge_contour_packs(
    shards: Sequence[Mapping[str, object]],
    component: str,
    *,
    total_rois: int,
    source_key: str,
) -> dict[str, object]:
    ptr = np.full((int(total_rois),), -1, dtype=np.int64)
    length = np.zeros((int(total_rois),), dtype=np.int32)
    point_chunks: list[np.ndarray] = []
    global_offset = 0
    contour_count = 0

    for shard in shards:
        source_payload = shard.get(source_key)
        if not isinstance(source_payload, Mapping):
            continue
        if source_key == "eye_geometry":
            contours_by_component = source_payload.get("contours")
            if not isinstance(contours_by_component, Mapping):
                continue
            pack = contours_by_component.get(component)
        else:
            pack = source_payload.get(component)
        if not isinstance(pack, Mapping):
            continue
        start = int(shard["start_row"])
        local_ptr = np.asarray(pack["ptr"], dtype=np.int64)
        local_len = np.asarray(pack["len"], dtype=np.int32)
        local_points = np.asarray(pack["points_xy"], dtype=np.float32).reshape(-1, 2)
        valid = local_len > 0
        for local_idx in np.nonzero(valid)[0]:
            ptr[start + int(local_idx)] = np.int64(global_offset + int(local_ptr[int(local_idx)]))
            length[start + int(local_idx)] = np.int32(local_len[int(local_idx)])
        if int(local_points.shape[0]) > 0:
            point_chunks.append(local_points)
            global_offset += int(local_points.shape[0])
        contour_count += int(np.count_nonzero(valid))

    points_xy = (
        np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        if point_chunks
        else np.zeros((1, 2), dtype=np.float32)
    )
    return {
        "ptr": ptr,
        "len": length,
        "points_xy": points_xy,
        "point_count": int(points_xy.shape[0]) if point_chunks else 0,
        "contour_count": int(contour_count),
        "points_placeholder_when_empty": bool(not point_chunks),
    }


def _write_packed_component_contours(
    component_group: zarr.Group,
    pack: Mapping[str, object],
    *,
    chunk_rois: int,
    component: str,
    source_mask_run: str,
    source_mask_label_schema_id: str,
    min_points: int,
    postcompute_backend: str,
) -> dict[str, object]:
    ptr = np.asarray(pack["ptr"], dtype=np.int64)
    length = np.asarray(pack["len"], dtype=np.int32)
    points_xy = np.asarray(pack["points_xy"], dtype=np.float32).reshape(-1, 2)
    contours_group = component_group.require_group("contours")
    contours_group.attrs.update(
        {
            "schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "contour_schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
            "point_order": "xy",
            "source_component": str(component),
            "source_mask_run": str(source_mask_run),
            "source_mask_label_schema_id": str(source_mask_label_schema_id or ""),
            "method": DEFAULT_CONTOUR_METHOD,
            "method_version": DEFAULT_CONTOUR_METHOD_VERSION,
            "boundary_policy": DEFAULT_BOUNDARY_POLICY,
            "min_points": int(min_points),
            "generated_at_utc": _utc_now(),
            "points_placeholder_when_empty": bool(pack.get("points_placeholder_when_empty")),
            "cache_coverage": "full_indexed_rows",
            "postcompute_backend": str(postcompute_backend),
        }
    )
    contours_group.create_array("ptr", data=ptr, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array("len", data=length, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array(
        "points_xy",
        data=points_xy,
        chunks=(max(1, min(4096, int(points_xy.shape[0]))), 2),
        overwrite=True,
    )
    return {
        "component": str(component),
        "status": "written",
        "roi_count": int(ptr.shape[0]),
        "contour_count": int(pack.get("contour_count", 0)),
        "point_count": int(pack.get("point_count", 0)),
    }


def _write_sharded_eye_geometry(
    run_group: zarr.Group,
    shards: Sequence[Mapping[str, object]],
    *,
    chunk_rois: int,
    refined_run: str,
    postcompute_backend: str,
    write_component_contours: bool = True,
) -> dict[str, object]:
    label_map = _label_index_map(run_group)
    if not set(_EYE_COMPONENTS).issubset(label_map):
        return {"status": "skipped", "reason": "missing_eye_components"}
    try:
        mask_store = open_mask_store(run_group, source_path=f"refined_subject_masks_runs/{refined_run}", prefer="dense")
    except MaskStoreError as exc:
        return {"status": "skipped", "reason": "missing_mask_store", "error": str(exc)}

    total_rois = int(mask_store.n_rows)
    geometry_source = (
        _ASSIGNMENT_REUSE_EYE_GEOMETRY_SOURCE
        if str(postcompute_backend) == _ASSIGNMENT_REUSE_POSTCOMPUTE_BACKEND
        else _POSTCOMPUTE_EYE_GEOMETRY_SOURCE
    )
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    separation_px = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_valid = np.zeros((total_rois,), dtype=bool)

    for shard in shards:
        eye_payload = shard.get("eye_geometry")
        if not isinstance(eye_payload, Mapping):
            continue
        start = int(shard["start_row"])
        stop = int(shard["stop_row"])
        ellipse_params[start:stop] = np.asarray(eye_payload["ellipse_params"], dtype=np.float32)
        ellipse_success[start:stop] = np.asarray(eye_payload["ellipse_success"], dtype=bool)
        separation_px[start:stop] = np.asarray(eye_payload["separation_px"], dtype=np.float32)
        separation_valid[start:stop] = np.asarray(eye_payload["separation_valid"], dtype=bool)

    components_parent = run_group.require_group("components")
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    for eye_idx, component_name in enumerate(_EYE_COMPONENTS):
        component_group = components_parent.require_group(component_name)
        geometry_group = component_group.require_group("geometry")
        geometry_group.attrs["geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
        geometry_group.attrs["geometry_method"] = "fit_ellipse_from_refined_subject_component_mask"
        geometry_group.attrs["source_mask_component"] = component_name
        geometry_group.attrs["source_measurement"] = geometry_source
        geometry_group.attrs["updated_at_utc"] = _utc_now()
        geometry_group.attrs["postcompute_backend"] = str(postcompute_backend)
        geometry_group.create_array(
            "ellipse_params",
            data=ellipse_params[:, eye_idx, :],
            chunks=(max(1, int(chunk_rois)), 5),
            overwrite=True,
        )
        geometry_group.create_array(
            "ellipse_success",
            data=ellipse_success[:, eye_idx],
            chunks=(max(1, int(chunk_rois)),),
            overwrite=True,
        )
        if write_component_contours:
            pack = _merge_contour_packs(shards, component_name, total_rois=total_rois, source_key="eye_geometry")
            _write_packed_component_contours(
                component_group,
                pack,
                chunk_rois=chunk_rois,
                component=component_name,
                source_mask_run=refined_run,
                source_mask_label_schema_id=source_label_schema,
                min_points=1,
                postcompute_backend=postcompute_backend,
            )

    relation_metrics = run_group.require_group("relations").require_group("eye_pair").require_group("metrics")
    relation_metrics.attrs["relation_schema_id"] = EYE_PAIR_RELATION_SCHEMA_ID
    relation_metrics.attrs["relation_components"] = list(_EYE_COMPONENTS)
    relation_metrics.attrs["relation_method"] = "ellipse_centroid_distance"
    relation_metrics.attrs["source_measurement"] = geometry_source
    relation_metrics.attrs["updated_at_utc"] = _utc_now()
    relation_metrics.attrs["postcompute_backend"] = str(postcompute_backend)
    relation_metrics.create_array("separation_px", data=separation_px, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    relation_metrics.create_array(
        "separation_valid",
        data=separation_valid,
        chunks=(max(1, int(chunk_rois)),),
        overwrite=True,
    )

    run_group.attrs["eye_geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
    run_group.attrs["eye_geometry_source_measurement"] = geometry_source
    run_group.attrs["eye_geometry_updated_at_utc"] = _utc_now()
    run_group.attrs["eye_geometry_status"] = "computed"
    run_group.attrs["eye_geometry_postcompute_backend"] = str(postcompute_backend)
    run_group.attrs.pop("eye_geometry_deferred_reason", None)
    return {
        "status": "updated",
        "roi_count": total_rois,
        "components": list(_EYE_COMPONENTS),
        "source_measurement": geometry_source,
        "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
        "pair_success_count": int(np.count_nonzero(separation_valid)),
    }


def _write_sharded_component_contours(
    run_group: zarr.Group,
    shards: Sequence[Mapping[str, object]],
    *,
    chunk_rois: int,
    refined_run: str,
    postcompute_backend: str,
) -> list[dict[str, object]]:
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return []
    total_rois = int(masks_roi.shape[0])
    label_map = _label_index_map(run_group)
    components_parent = run_group.require_group("components")
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    summaries: list[dict[str, object]] = []
    for component_name in _COMPONENT_CONTOUR_COMPONENTS:
        if component_name not in label_map:
            continue
        pack = _merge_contour_packs(shards, component_name, total_rois=total_rois, source_key="component_contours")
        component_group = components_parent.require_group(component_name)
        summaries.append(
            _write_packed_component_contours(
                component_group,
                pack,
                chunk_rois=chunk_rois,
                component=component_name,
                source_mask_run=refined_run,
                source_mask_label_schema_id=source_label_schema,
                min_points=2,
                postcompute_backend=postcompute_backend,
            )
        )
    if summaries:
        run_group.attrs["component_contours_status"] = "computed"
        run_group.attrs["component_contours_components"] = [item["component"] for item in summaries]
        run_group.attrs["component_contours_updated_at_utc"] = _utc_now()
        run_group.attrs["component_contours_summary"] = list(_json_safe(summaries))
        run_group.attrs["component_contours_postcompute_backend"] = str(postcompute_backend)
    return summaries


def _write_sharded_sampled_component_contours(
    run_group: zarr.Group,
    shards: Sequence[Mapping[str, object]],
    *,
    sample_counts: Mapping[str, int],
    row_chunk: int,
    refined_run: str,
    postcompute_backend: str,
) -> list[dict[str, object]]:
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return []
    total_rois = int(masks_roi.shape[0])
    label_map = _label_index_map(run_group)
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    components_parent = run_group.require_group("components")
    summaries: list[dict[str, object]] = []
    for component_name, raw_sample_count in sample_counts.items():
        if component_name not in label_map:
            continue
        sample_count = int(raw_sample_count)
        points_xy = np.full((total_rois, sample_count, 2), np.nan, dtype=np.float32)
        valid = np.zeros((total_rois,), dtype=bool)
        source_point_count = np.zeros((total_rois,), dtype=np.int32)
        for shard in shards:
            sampled_by_component = shard.get("sampled_component_contours")
            if not isinstance(sampled_by_component, Mapping):
                continue
            payload = sampled_by_component.get(component_name)
            if not isinstance(payload, Mapping):
                continue
            start = int(shard["start_row"])
            stop = int(shard["stop_row"])
            points_xy[start:stop] = np.asarray(payload["points_xy"], dtype=np.float32)
            valid[start:stop] = np.asarray(payload["valid"], dtype=bool)
            source_point_count[start:stop] = np.asarray(payload["source_point_count"], dtype=np.int32)
        summary = write_sampled_component_contour_arrays(
            components_parent.require_group(component_name),
            points_xy=points_xy,
            valid=valid,
            source_point_count=source_point_count,
            row_chunk=int(row_chunk),
            component=component_name,
            source_mask_run=refined_run,
            source_mask_label_schema_id=source_label_schema,
            min_points=2,
        )
        payload = asdict(summary)
        payload["postcompute_backend"] = str(postcompute_backend)
        summaries.append(payload)
    _update_sampled_component_contour_attrs(
        run_group,
        summaries,
        postcompute_backend=postcompute_backend,
    )
    return summaries


def _update_sampled_component_contour_attrs(
    run_group: zarr.Group,
    summaries: Sequence[Mapping[str, object]],
    *,
    postcompute_backend: str,
) -> None:
    if not summaries:
        return
    combined = {
        str(item["component"]): dict(item)
        for item in list(run_group.attrs.get("sampled_component_contours_summary") or [])
        if isinstance(item, Mapping) and item.get("component")
    }
    for item in summaries:
        combined[str(item["component"])] = dict(item)
    ordered = [combined[name] for name in CANONICAL_COMPONENT_ORDER if name in combined]
    run_group.attrs["sampled_component_contours_status"] = "computed"
    run_group.attrs["sampled_component_contours_components"] = [item["component"] for item in ordered]
    run_group.attrs["sampled_component_contours_updated_at_utc"] = _utc_now()
    run_group.attrs["sampled_component_contours_summary"] = list(_json_safe(ordered))
    previous_backend = str(run_group.attrs.get("sampled_component_contours_postcompute_backend") or "")
    resolved_backend = (
        str(postcompute_backend)
        if not previous_backend or previous_backend == str(postcompute_backend)
        else "mixed"
    )
    run_group.attrs["sampled_component_contours_postcompute_backend"] = resolved_backend


def _write_sampled_component_contours_from_raw_shards(
    run_group: zarr.Group,
    shards: Sequence[Mapping[str, object]],
    *,
    sample_counts: Mapping[str, int],
    source_key: str,
    row_chunk: int,
    refined_run: str,
    postcompute_backend: str,
) -> list[dict[str, object]]:
    total_rois = int(run_group["masks_roi"].shape[0])
    source_label_schema = str(run_group.attrs.get("label_schema_id") or "")
    components_parent = run_group.require_group("components")
    summaries: list[dict[str, object]] = []
    for component_name, raw_sample_count in sample_counts.items():
        sample_count = int(raw_sample_count)
        points_xy = np.full((total_rois, sample_count, 2), np.nan, dtype=np.float32)
        valid = np.zeros((total_rois,), dtype=bool)
        source_point_count = np.zeros((total_rois,), dtype=np.int32)
        for shard in shards:
            source_payload = shard.get(source_key)
            if not isinstance(source_payload, Mapping):
                continue
            contours = source_payload.get("contours")
            if not isinstance(contours, Mapping):
                continue
            pack = contours.get(component_name)
            if not isinstance(pack, Mapping):
                continue
            sampled = _sample_packed_contours(pack, sample_count)
            start = int(shard["start_row"])
            stop = int(shard["stop_row"])
            points_xy[start:stop] = sampled["points_xy"]
            valid[start:stop] = sampled["valid"]
            source_point_count[start:stop] = sampled["source_point_count"]
        summary = write_sampled_component_contour_arrays(
            components_parent.require_group(component_name),
            points_xy=points_xy,
            valid=valid,
            source_point_count=source_point_count,
            row_chunk=int(row_chunk),
            component=component_name,
            source_mask_run=refined_run,
            source_mask_label_schema_id=source_label_schema,
            min_points=2,
        )
        payload = asdict(summary)
        payload["postcompute_backend"] = str(postcompute_backend)
        summaries.append(payload)
    _update_sampled_component_contour_attrs(
        run_group,
        summaries,
        postcompute_backend=postcompute_backend,
    )
    return summaries


def _run_sharded_refined_subject_postcompute(
    zarr_path: str | Path,
    *,
    refined_run: str,
    chunk_size: int,
    num_workers: Optional[int],
    write_eye_geometry: bool,
    write_component_contours: bool,
    sampled_contour_counts: Mapping[str, int] | None = None,
    sampled_contour_row_chunk: int = DEFAULT_SAMPLED_CONTOUR_ROW_CHUNK,
) -> dict[str, object]:
    sampled_counts = {str(name): int(value) for name, value in dict(sampled_contour_counts or {}).items()}
    if not write_eye_geometry and not write_component_contours and not sampled_counts:
        return {"status": "skipped", "reason": "no_postcompute_requested"}

    root = open_zarr_root(zarr_path, mode="r")
    run_group = root["refined_subject_masks_runs"][refined_run]
    masks_roi = run_group.get("masks_roi")
    if masks_roi is None:
        return {"status": "skipped", "reason": "missing_masks_roi"}
    total_rois = int(masks_roi.shape[0])
    shard_size = max(1, min(int(chunk_size), total_rois if total_rois > 0 else 1))
    ranges = [(start, min(total_rois, start + shard_size)) for start in range(0, total_rois, shard_size)]
    worker_count = max(1, int(num_workers or 1))
    started = time.perf_counter()
    if worker_count == 1 or len(ranges) <= 1:
        shards = [
            _compute_refined_subject_postcompute_shard(
                str(zarr_path),
                refined_run,
                start,
                stop,
                write_eye_geometry=write_eye_geometry,
                write_component_contours=write_component_contours,
                sampled_contour_counts=sampled_counts,
            )
            for start, stop in ranges
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    _compute_refined_subject_postcompute_shard,
                    str(zarr_path),
                    refined_run,
                    start,
                    stop,
                    write_eye_geometry=write_eye_geometry,
                    write_component_contours=write_component_contours,
                    sampled_contour_counts=sampled_counts,
                )
                for start, stop in ranges
            ]
            shards = [future.result() for future in futures]
    shards = sorted(shards, key=lambda item: int(item["start_row"]))

    root = open_zarr_root(zarr_path, mode="a")
    run_group = root["refined_subject_masks_runs"][refined_run]
    chunk_rois = max(1, min(256, total_rois if total_rois > 0 else 1))
    eye_summary = (
        _write_sharded_eye_geometry(
            run_group,
            shards,
            chunk_rois=chunk_rois,
            refined_run=refined_run,
            postcompute_backend=_PROCESS_SHARD_POSTCOMPUTE_BACKEND,
            write_component_contours=bool(write_component_contours),
        )
        if write_eye_geometry
        else {"status": "skipped", "reason": "write_eye_geometry=false"}
    )
    contour_summaries = (
        _write_sharded_component_contours(
            run_group,
            shards,
            chunk_rois=chunk_rois,
            refined_run=refined_run,
            postcompute_backend=_PROCESS_SHARD_POSTCOMPUTE_BACKEND,
        )
        if write_component_contours
        else []
    )
    sampled_contour_summaries = _write_sharded_sampled_component_contours(
        run_group,
        shards,
        sample_counts=sampled_counts,
        row_chunk=int(sampled_contour_row_chunk),
        refined_run=refined_run,
        postcompute_backend=_PROCESS_SHARD_POSTCOMPUTE_BACKEND,
    )
    duration_seconds = float(time.perf_counter() - started)
    return dict(
        _json_safe(
            {
                "status": "updated",
                "postcompute_backend": _PROCESS_SHARD_POSTCOMPUTE_BACKEND,
                "duration_seconds": duration_seconds,
                "rows_per_second": float(total_rois / duration_seconds) if duration_seconds > 0 else None,
                "roi_count": total_rois,
                "shard_count": len(shards),
                "shard_size": int(shard_size),
                "num_workers": int(worker_count),
                "write_eye_geometry": bool(write_eye_geometry),
                "write_component_contours": bool(write_component_contours),
                "write_sampled_component_contours": bool(sampled_counts),
                "eye_geometry": eye_summary,
                "component_contours": contour_summaries,
                "sampled_component_contours": sampled_contour_summaries,
                "worker_durations_seconds": [float(item.get("duration_seconds") or 0.0) for item in shards],
            }
        )
    )


def _array_shape(group: zarr.Group, name: str) -> tuple[int, ...] | None:
    arr = group.get(name)
    if arr is None:
        return None
    try:
        return tuple(int(dim) for dim in arr.shape)
    except Exception:
        return None


def _ensure_metric_array(
    group: zarr.Group,
    name: str,
    *,
    shape: Sequence[int],
    dtype: object,
    chunks: Sequence[int],
    fill_value: object = 0,
) -> None:
    expected_shape = tuple(int(dim) for dim in shape)
    if _array_shape(group, name) == expected_shape:
        return
    _create_filled_array(
        group,
        name,
        shape=expected_shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
    )


def _ensure_refined_run_metric_shell(run_group: zarr.Group, *, total_rows: int, component_count: int) -> None:
    metrics_group = run_group.require_group("metrics")
    _ensure_metric_array(
        metrics_group,
        "mask_present",
        shape=(total_rows, component_count),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, component_count),
    )
    _ensure_metric_array(
        metrics_group,
        "area_px",
        shape=(total_rows, component_count),
        dtype=np.float32,
        chunks=_derived_metric_chunks_2d(total_rows, component_count),
    )
    _ensure_metric_array(
        metrics_group,
        "centroid_xy",
        shape=(total_rows, component_count, 2),
        dtype=np.float32,
        chunks=_derived_metric_chunks_lastdim(total_rows, component_count, 2),
        fill_value=np.nan,
    )
    _ensure_metric_array(
        metrics_group,
        "centroid_valid",
        shape=(total_rows, component_count),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, component_count),
    )
    _ensure_metric_array(
        metrics_group,
        "bbox_xyxy",
        shape=(total_rows, component_count, 4),
        dtype=np.float32,
        chunks=_derived_metric_chunks_lastdim(total_rows, component_count, 4),
    )
    _ensure_metric_array(
        metrics_group,
        "bbox_valid",
        shape=(total_rows, component_count),
        dtype=bool,
        chunks=_derived_metric_chunks_2d(total_rows, component_count),
    )


def _ensure_refined_component_metric_shell(
    run_group: zarr.Group,
    *,
    component_name: str,
    total_rows: int,
) -> zarr.Group:
    component_group = run_group.require_group("components").require_group(component_name)
    derived_metric_chunks = (_common_derived_metric_row_chunk(total_rows),)
    live_metric_chunks = (refined_subject_mask_metric_row_chunk(total_rows),)
    _ensure_metric_array(
        component_group,
        "mask_present",
        shape=(total_rows,),
        dtype=bool,
        chunks=derived_metric_chunks,
    )
    _ensure_metric_array(
        component_group,
        "area_px",
        shape=(total_rows,),
        dtype=np.float32,
        chunks=derived_metric_chunks,
    )
    if "edit_applied" not in component_group:
        _create_filled_array(
            component_group,
            "edit_applied",
            shape=(total_rows,),
            dtype=bool,
            chunks=live_metric_chunks,
        )
    metrics_group = component_group.require_group("metrics")
    for metric_name, dtype in (
        ("component_count", np.int32),
        ("largest_component_fraction", np.float32),
        ("hole_count", np.int32),
        ("hole_area_fraction", np.float32),
    ):
        _ensure_metric_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=derived_metric_chunks,
        )
    for metric_name, dtype, fill_value in (
        ("sigma_noise", np.float32, np.nan),
        ("curvature_var", np.float32, np.nan),
        ("ipr", np.float32, np.nan),
        ("solidity", np.float32, np.nan),
    ):
        _ensure_metric_array(
            metrics_group,
            metric_name,
            shape=(total_rows,),
            dtype=dtype,
            chunks=derived_metric_chunks,
            fill_value=fill_value,
        )
    return component_group


def _resolve_refined_subject_run_group(
    root: zarr.Group,
    refined_run: Optional[str],
) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    if refined_run:
        run_name = str(refined_run)
    else:
        latest = parent.attrs.get("latest")
        if latest is None:
            keys = sorted(str(key) for key in parent.keys())
            if not keys:
                raise ValueError("refined_subject_masks_runs has no runs.")
            run_name = keys[-1]
        else:
            run_name = str(latest)
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _component_indices_for_refined_run(
    run_group: zarr.Group,
    components: Optional[Sequence[str]],
) -> tuple[tuple[str, int], ...]:
    labels_raw = run_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        raise ValueError("refined subject-mask run is missing mask_labels attrs.")
    label_map = {str(label): int(idx) for idx, label in enumerate(labels_raw)}
    requested = [str(value) for value in components] if components else list(label_map)
    resolved: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw_component in requested:
        component = _normalize_component_name(raw_component)
        if component is None or component in seen:
            continue
        if component not in label_map:
            raise ValueError(f"Component {component!r} not present in refined run mask_labels.")
        seen.add(component)
        resolved.append((component, int(label_map[component])))
    return tuple(resolved)


def _existing_component_reasons(component_group: zarr.Group, total_rows: int) -> np.ndarray:
    labels = read_reason_labels(component_group)
    if labels is None:
        return np.full((total_rows,), "clean", dtype=object)
    arr = np.asarray(labels, dtype=object).reshape(-1)
    if int(arr.shape[0]) != int(total_rows):
        return np.full((total_rows,), "clean", dtype=object)
    return arr.copy()


def refresh_refined_subject_mask_metrics_run(
    root: zarr.Group,
    *,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    metric_level: str = "cheap",
    chunk_size: int = 256,
    refresh_reason_tags: bool = True,
    write_eye_geometry: bool = False,
    write_component_contours: bool = False,
) -> dict[str, object]:
    """Refresh mask-local metrics/QC for an existing refined-subject run."""

    metric_level = str(metric_level)
    if metric_level not in _METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {_METRIC_LEVELS}; got {metric_level!r}.")
    stage_start = time.perf_counter()
    timing = _TimingRecorder()
    run_name, run_group = _resolve_refined_subject_run_group(root, refined_run)
    try:
        mask_store = open_mask_store(
            run_group,
            source_path=f"refined_subject_masks_runs/{run_name}",
            prefer="dense",
        )
    except MaskStoreError as exc:
        raise ValueError(
            f"refined_subject_masks_runs/{run_name} missing usable mask store (masks_roi or mask_rle)."
        ) from exc
    mask_shape = tuple(int(value) for value in mask_store.shape)
    if len(mask_shape) != 4:
        raise ValueError(
            f"refined_subject_masks_runs/{run_name} mask store must be 4D, got {mask_shape}."
        )

    total_rows = int(mask_shape[0])
    requested_chunk_size = max(1, int(chunk_size))
    worker_chunk_size = requested_chunk_size
    execution_metadata: dict[str, object] = {
        "execution_backend": _SERIAL_EXECUTION_BACKEND,
        "process_shard_execution_enabled": False,
        "mask_store_encoding": mask_store.encoding,
        "mask_storage_surface": mask_store.storage_surface,
        "mask_store_path": mask_store.storage_path,
        "requested_chunk_size": requested_chunk_size,
        "worker_chunk_size": worker_chunk_size,
        "chunk_alignment": "requested_chunk_size",
    }
    component_count = int(mask_shape[1])
    _ensure_refined_run_metric_shell(run_group, total_rows=total_rows, component_count=component_count)
    component_indices = _component_indices_for_refined_run(run_group, components)
    chunk_ranges = _row_chunks(total_rows, worker_chunk_size)
    review_counts: dict[str, dict[str, int]] = {}
    refreshed_components: list[str] = []
    reason_labels_by_component: dict[str, np.ndarray] = {}
    rows_with_component_by_component: dict[str, int] = {}
    contour_summaries: list[dict[str, object]] = []

    for component_name, component_idx in component_indices:
        component_group = _ensure_refined_component_metric_shell(
            run_group,
            component_name=component_name,
            total_rows=total_rows,
        )
        reason_labels_by_component[component_name] = _existing_component_reasons(component_group, total_rows)
        rows_with_component_by_component[component_name] = 0

    for start_row, stop_row in chunk_ranges:
        chunk_timing: dict[str, object] = {
            "chunk_index": int(len(timing.chunk_timings)),
            "start_row": int(start_row),
            "stop_row": int(stop_row),
            "row_count": int(stop_row) - int(start_row),
            "execution_backend": _SERIAL_EXECUTION_BACKEND,
        }
        chunk_start = time.perf_counter()
        row_slice = slice(int(start_row), int(stop_row))
        for component_name, component_idx in component_indices:
            phase_start = time.perf_counter()
            component_masks = np.asarray(
                mask_store.read_dense(rows=row_slice, channels=int(component_idx))[:, 0],
                dtype=np.uint8,
            )
            write_result = _write_mask_local_metrics_chunk(
                run_group,
                component_name=component_name,
                component_idx=int(component_idx),
                row_slice=row_slice,
                masks=component_masks,
                metric_level=metric_level,
                write_metric_attrs=False,
            )
            elapsed = timing.add(f"refresh_metrics_{component_name}", time.perf_counter() - phase_start)
            chunk_timing[f"refresh_metrics_{component_name}_seconds"] = elapsed
            rows_with_component_by_component[component_name] += int(np.count_nonzero(write_result.mask_present))
            if refresh_reason_tags:
                reason_labels_by_component[component_name][row_slice] = _replace_metric_qc_reason_labels(
                    reason_labels_by_component[component_name][row_slice],
                    write_result.reason_labels,
                )
        chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
        timing.chunk_timings.append(chunk_timing)

    for component_name, _component_idx in component_indices:
        component_group = run_group["components"][component_name]
        _set_component_metric_attrs(component_group, metric_level=metric_level)
        component_group.attrs["metric_qc_refreshed_at_utc"] = _utc_now()
        component_group.attrs["metric_qc_rows_with_component"] = int(
            rows_with_component_by_component.get(component_name, 0)
        )
        if refresh_reason_tags:
            write_reason_columns(
                component_group,
                reason_labels_by_component[component_name],
                chunk_size=max(1, min(256, total_rows)),
                include_reason_text=True,
                overwrite=True,
            )
        _add_review_counts(review_counts, component_name, reason_labels_by_component[component_name])
        refreshed_components.append(component_name)

    if write_eye_geometry and set(_EYE_COMPONENTS).issubset({name for name, _idx in component_indices}):
        write_refined_subject_eye_geometry(run_group)
        run_group.attrs["eye_geometry_status"] = "computed"

    if write_component_contours:
        contour_components = _component_contour_targets([name for name, _idx in component_indices])
        if contour_components:
            with timing.phase("write_component_contours"):
                contour_summaries = _summaries_to_json_safe(
                    write_refined_subject_component_contours(
                        run_group,
                        components=contour_components,
                        source_mask_run=run_name,
                        chunk_rois=max(1, min(256, total_rows)),
                        overwrite=True,
                    )
                )
            run_group.attrs["component_contours_status"] = "computed"
            run_group.attrs["component_contours_components"] = list(contour_components)
            run_group.attrs["component_contours_updated_at_utc"] = _utc_now()
            run_group.attrs["component_contours_summary"] = list(_json_safe(contour_summaries))
        else:
            run_group.attrs["component_contours_status"] = "deferred"
            run_group.attrs["component_contours_deferred_reason"] = "no_subject_body_or_swim_bladder_components_selected"

    refreshed_at = _utc_now()
    duration_seconds = float(time.perf_counter() - stage_start)
    timing_summary = timing.summary(total_rows=total_rows, duration_seconds=duration_seconds)
    timing_summary.update(execution_metadata)
    run_group.attrs["component_metric_level"] = metric_level
    run_group.attrs["component_metrics_schema_id"] = _COMPONENT_METRICS_SCHEMA_ID
    run_group.attrs["metric_qc_schema_id"] = _COMPONENT_METRIC_QC_SCHEMA_ID
    run_group.attrs["component_metric_qc_refreshed_at_utc"] = refreshed_at
    run_group.attrs["component_metric_qc_review_counts"] = review_counts
    run_group.attrs["component_metric_qc_execution_backend"] = _SERIAL_EXECUTION_BACKEND
    run_group.attrs["component_metric_qc_timing_summary"] = dict(_json_safe(timing_summary))
    run_group.attrs["component_metric_qc_chunk_timings"] = list(_json_safe(timing.chunk_timings))
    summary_stats = dict(run_group.attrs.get("summary_statistics") or {})
    summary_stats["rows_total"] = int(total_rows)
    summary_stats["component_metric_level"] = metric_level
    summary_stats["component_metric_qc_refreshed_at_utc"] = refreshed_at
    summary_stats["component_metric_qc_duration_seconds"] = duration_seconds
    summary_stats["component_metric_qc_execution_backend"] = _SERIAL_EXECUTION_BACKEND
    run_group.attrs["summary_statistics"] = summary_stats

    return {
        "status": "updated",
        "refined_run": run_name,
        "components": refreshed_components,
        "metric_level": metric_level,
        "chunk_size": requested_chunk_size,
        "worker_chunk_size": worker_chunk_size,
        "chunk_count": len(chunk_ranges),
        "refresh_reason_tags": bool(refresh_reason_tags),
        "write_eye_geometry": bool(write_eye_geometry),
        "write_component_contours": bool(write_component_contours),
        "component_contours": contour_summaries,
        "review_counts": review_counts,
        "duration_seconds": duration_seconds,
        "timing_summary": dict(_json_safe(timing_summary)),
        **execution_metadata,
    }


def refresh_refined_subject_mask_metrics(
    zarr_path: str | Path,
    *,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    metric_level: str = "cheap",
    chunk_size: int = 256,
    refresh_reason_tags: bool = True,
    write_eye_geometry: bool = False,
    write_component_contours: bool = False,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    return refresh_refined_subject_mask_metrics_run(
        root,
        refined_run=refined_run,
        components=components,
        metric_level=metric_level,
        chunk_size=chunk_size,
        refresh_reason_tags=refresh_reason_tags,
        write_eye_geometry=write_eye_geometry,
        write_component_contours=write_component_contours,
    )


def _common_metric_chunk_payload(result: _CanonicalComponentChunkResult) -> dict[str, object]:
    return {
        "spatial_metrics": {
            str(name): np.asarray(values)
            for name, values in result.spatial_metrics.items()
        },
        "component_metrics": {
            str(name): np.asarray(values)
            for name, values in result.component_metrics.items()
        },
        "source_row_fingerprint": np.asarray(result.source_row_fingerprint, dtype=np.uint64),
    }


def _merge_common_metric_chunk_payload(
    target: dict[str, dict[str, object]],
    *,
    component_name: str,
    row_slice: slice,
    payload: Mapping[str, object],
    total_rows: int,
) -> None:
    component = target.get(str(component_name))
    raw_spatial = dict(payload.get("spatial_metrics") or {})
    raw_component_metrics = dict(payload.get("component_metrics") or {})
    if component is None:
        component = {
            "spatial_metrics": {
                str(name): np.zeros(
                    (int(total_rows), *tuple(np.asarray(values).shape[1:])),
                    dtype=np.asarray(values).dtype,
                )
                for name, values in raw_spatial.items()
            },
            "component_metrics": {
                str(name): np.zeros(
                    (int(total_rows), *tuple(np.asarray(values).shape[1:])),
                    dtype=np.asarray(values).dtype,
                )
                for name, values in raw_component_metrics.items()
            },
            "source_row_fingerprint": np.zeros((int(total_rows),), dtype=np.uint64),
        }
        target[str(component_name)] = component

    spatial_arrays = dict(component["spatial_metrics"])
    for name, values in raw_spatial.items():
        np.asarray(spatial_arrays[str(name)])[row_slice] = np.asarray(values)
    component["spatial_metrics"] = spatial_arrays

    metric_arrays = dict(component["component_metrics"])
    for name, values in raw_component_metrics.items():
        metric_name = str(name)
        if metric_name not in metric_arrays:
            metric_arrays[metric_name] = np.zeros(
                (int(total_rows), *tuple(np.asarray(values).shape[1:])),
                dtype=np.asarray(values).dtype,
            )
        np.asarray(metric_arrays[metric_name])[row_slice] = np.asarray(values)
    component["component_metrics"] = metric_arrays
    np.asarray(component["source_row_fingerprint"], dtype=np.uint64)[row_slice] = np.asarray(
        payload["source_row_fingerprint"],
        dtype=np.uint64,
    )


def _write_common_run_metrics(
    run_group: zarr.Group,
    *,
    component_names: Sequence[str],
    payloads_by_component: Mapping[str, Mapping[str, object]],
) -> None:
    run_metrics = run_group["metrics"]
    for name in ("mask_present", "area_px", "centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid"):
        component_values = [
            np.asarray(dict(payloads_by_component[str(component_name)]["spatial_metrics"])[name])
            for component_name in component_names
        ]
        run_metrics[name][:] = np.stack(component_values, axis=1)


def _write_common_component_metrics(
    run_group: zarr.Group,
    *,
    component_name: str,
    payload: Mapping[str, object],
    metric_level: str,
) -> None:
    spatial_metrics = dict(payload.get("spatial_metrics") or {})
    component_metrics = dict(payload.get("component_metrics") or {})
    component_group = run_group["components"][component_name]
    component_group["mask_present"][:] = np.asarray(spatial_metrics["mask_present"], dtype=bool)
    component_group["area_px"][:] = np.asarray(spatial_metrics["area_px"], dtype=np.float32)
    component_group["source_row_fingerprint"][:] = np.asarray(
        payload["source_row_fingerprint"],
        dtype=np.uint64,
    )
    for name, values in component_metrics.items():
        component_group["metrics"][str(name)][:] = np.asarray(values)
    _set_component_metric_attrs(component_group, metric_level=metric_level)


def _finalization_metric_chunk_payload(batch: _FinalizedComponentBatch) -> dict[str, object]:
    return {
        "metrics": {
            str(metric_name): np.asarray(values, dtype=np.float32)
            for metric_name, values in batch.metrics.items()
        },
        "quality_code": np.asarray(batch.quality_code, dtype=np.int16),
        "quality_score": np.asarray(batch.quality_score, dtype=np.float32),
    }


def _merge_finalization_metric_chunk_payload(
    target: dict[str, dict[str, object]],
    *,
    component_name: str,
    row_slice: slice,
    payload: Mapping[str, object],
    total_rows: int,
) -> None:
    component = target.get(str(component_name))
    raw_metrics = dict(payload.get("metrics") or {})
    if component is None:
        component = {
            "metrics": {
                str(metric_name): np.zeros((int(total_rows),), dtype=np.float32)
                for metric_name in raw_metrics
            },
            "quality_code": np.zeros((int(total_rows),), dtype=np.int16),
            "quality_score": np.zeros((int(total_rows),), dtype=np.float32),
        }
        target[str(component_name)] = component

    metric_arrays = dict(component["metrics"])
    for metric_name, values in raw_metrics.items():
        name = str(metric_name)
        if name not in metric_arrays:
            metric_arrays[name] = np.zeros((int(total_rows),), dtype=np.float32)
        np.asarray(metric_arrays[name], dtype=np.float32)[row_slice] = np.asarray(values, dtype=np.float32)
    component["metrics"] = metric_arrays
    np.asarray(component["quality_code"], dtype=np.int16)[row_slice] = np.asarray(
        payload["quality_code"],
        dtype=np.int16,
    )
    np.asarray(component["quality_score"], dtype=np.float32)[row_slice] = np.asarray(
        payload["quality_score"],
        dtype=np.float32,
    )


def _write_finalization_metrics_component(
    run_group: zarr.Group,
    *,
    component_name: str,
    payload: Mapping[str, object],
    total_rows: int,
    source_batch: _FinalizedComponentBatch | None = None,
) -> None:
    component_group = run_group["components"][component_name]
    metric_arrays = dict(payload.get("metrics") or {})
    _create_finalization_metric_shell(
        component_group,
        batch=source_batch,
        metric_names=tuple(str(name) for name in metric_arrays),
        total_rows=total_rows,
    )
    metrics_group = component_group["finalization_metrics"]
    for metric_name, values in metric_arrays.items():
        metrics_group[str(metric_name)][:] = np.asarray(values, dtype=np.float32)
    metrics_group["quality_code"][:] = np.asarray(payload["quality_code"], dtype=np.int16)
    metrics_group["quality_score"][:] = np.asarray(payload["quality_score"], dtype=np.float32)


def _process_and_write_finalizer_chunk_open(
    *,
    source: SourceSubjectMaskRun,
    run_group: zarr.Group,
    component_names: Sequence[str],
    required_raw_components: Sequence[str],
    start_row: int,
    stop_row: int,
    chunk_index: int,
    metric_level: str,
    eye_assignment_context: _EyeAssignmentContext | None,
    retain_source_seeds: bool,
    execution_backend: str = _PROCESS_SHARD_EXECUTION_BACKEND,
) -> dict[str, object]:
    component_names = tuple(str(name) for name in component_names)
    component_to_index = {name: idx for idx, name in enumerate(component_names)}
    timing = _TimingRecorder()

    row_slice = slice(int(start_row), int(stop_row))
    chunk_timing: dict[str, object] = {
        "chunk_index": int(chunk_index),
        "start_row": int(start_row),
        "stop_row": int(stop_row),
        "row_count": int(stop_row) - int(start_row),
        "execution_backend": execution_backend,
    }
    chunk_start = time.perf_counter()
    chunk_any = np.zeros((int(stop_row) - int(start_row),), dtype=bool)
    chunk_batches: dict[str, _FinalizedComponentBatch] = {}
    review_counts: dict[str, dict[str, int]] = {}
    reason_labels_by_component: dict[str, list[str]] = {}
    common_metrics_by_component: dict[str, dict[str, object]] = {}
    finalization_metrics_by_component: dict[str, dict[str, object]] = {}
    eyes_union_assignment_summary: dict[str, object] = {}
    eye_geometry_payload: dict[str, object] | None = None

    for raw_component in required_raw_components:
        phase_start = time.perf_counter()
        batch = _finalize_source_component_rows(
            source,
            raw_component,
            start_row=start_row,
            stop_row=stop_row,
        )
        elapsed = timing.add(f"finalize_{raw_component}", time.perf_counter() - phase_start)
        chunk_timing[f"finalize_{raw_component}_seconds"] = elapsed
        chunk_batches[raw_component] = batch
        if raw_component == _RAW_EYE_UNION_COMPONENT:
            continue

        component_idx = int(component_to_index[raw_component])
        phase_start = time.perf_counter()
        write_result = _write_canonical_component_chunk(
            run_group,
            component_name=raw_component,
            component_idx=component_idx,
            row_slice=row_slice,
            masks=batch.masks,
            source_masks=batch.source_masks,
            metric_level=metric_level,
            precomputed_component_metrics=_component_metrics_from_finalization_batch(
                batch,
                row_count=int(stop_row) - int(start_row),
            ),
            write_metric_attrs=False,
            write_derived_metrics=False,
            retain_source_seeds=retain_source_seeds,
            timing=timing,
            chunk_timing=chunk_timing,
        )
        common_metrics_by_component[raw_component] = _common_metric_chunk_payload(write_result)
        finalization_metrics_by_component[raw_component] = _finalization_metric_chunk_payload(batch)
        elapsed = timing.add(f"write_{raw_component}", time.perf_counter() - phase_start)
        chunk_timing[f"write_{raw_component}_seconds"] = elapsed
        chunk_any |= write_result.mask_present
        labels = _merge_reason_label_arrays(batch.reason_labels, write_result.reason_labels)
        reason_labels_by_component[raw_component] = [str(value) for value in labels.tolist()]
        _add_review_counts(review_counts, raw_component, labels)

    if eye_assignment_context is not None:
        union_batch = chunk_batches[_RAW_EYE_UNION_COMPONENT]
        phase_start = time.perf_counter()
        assignment_chunk = _assign_finalized_eyes_union_rows(
            source,
            union_batch,
            eye_assignment_context,
            start_row=start_row,
            stop_row=stop_row,
        )
        elapsed = timing.add("eye_assignment", time.perf_counter() - phase_start)
        chunk_timing["eye_assignment_seconds"] = elapsed
        _record_eye_assignment_phase_seconds(timing, chunk_timing, assignment_chunk.phase_seconds)
        _merge_assignment_summary(eyes_union_assignment_summary, assignment_chunk.summary)
        eye_geometry_payload = assignment_chunk.eye_geometry
        for component_name in _EYE_COMPONENTS:
            component_idx = int(component_to_index[component_name])
            masks = np.asarray(assignment_chunk.masks[component_name], dtype=np.uint8)
            phase_start = time.perf_counter()
            write_result = _write_canonical_component_chunk(
                run_group,
                component_name=component_name,
                component_idx=component_idx,
                row_slice=row_slice,
                masks=masks,
                source_masks=masks,
                metric_level=metric_level,
                precomputed_component_metrics=assignment_chunk.component_metrics.get(component_name),
                write_metric_attrs=False,
                write_derived_metrics=False,
                retain_source_seeds=retain_source_seeds,
                timing=timing,
                chunk_timing=chunk_timing,
            )
            common_metrics_by_component[component_name] = _common_metric_chunk_payload(write_result)
            elapsed = timing.add(f"write_{component_name}", time.perf_counter() - phase_start)
            chunk_timing[f"write_{component_name}_seconds"] = elapsed
            chunk_any |= write_result.mask_present
            labels = _merge_reason_label_arrays(
                assignment_chunk.reason_labels[component_name],
                write_result.reason_labels,
            )
            reason_labels_by_component[component_name] = [str(value) for value in labels.tolist()]
            _add_review_counts(review_counts, component_name, labels)

    chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
    return {
        "chunk_timing": chunk_timing,
        "phase_seconds": dict(timing.phase_seconds),
        "phase_counts": dict(timing.phase_counts),
        "review_counts": review_counts,
        "reason_labels_by_component": reason_labels_by_component,
        "common_metrics_by_component": common_metrics_by_component,
        "finalization_metrics_by_component": finalization_metrics_by_component,
        "rows_with_nonempty_masks": int(np.count_nonzero(chunk_any)),
        "eyes_union_assignment_summary": dict(_json_safe(eyes_union_assignment_summary)),
        "eye_geometry": eye_geometry_payload,
    }


def _process_and_write_finalizer_shard(
    zarr_path: str,
    *,
    subject_run: str,
    subject_shard_runs: Sequence[str] | None = None,
    target_crop_run: Optional[str] = None,
    refined_run: str,
    component_names: Sequence[str],
    required_raw_components: Sequence[str],
    chunk_specs: Sequence[tuple[int, int, int]],
    metric_level: str,
    assignment_keypoint_group: Optional[str],
    assignment_keypoints_run: Optional[str],
    retain_source_seeds: bool,
    shard_index: int,
    total_shards: int,
    worker_count: int,
    progress_jsonl: Optional[str],
    progress_start_time: Optional[float],
) -> list[dict[str, object]]:
    root = open_zarr_root(zarr_path, mode="a")
    source, _collection = _load_subject_mask_source(
        root,
        subject_run=subject_run,
        subject_shard_runs=subject_shard_runs,
        target_crop_run=target_crop_run,
    )
    run_group = root["refined_subject_masks_runs"][refined_run]
    component_names = tuple(str(name) for name in component_names)
    progress = _ProgressJsonlReporter(
        Path(progress_jsonl).expanduser().resolve() if progress_jsonl else None,
        start_time=float(progress_start_time) if progress_start_time is not None else time.perf_counter(),
    )

    eye_assignment_context: _EyeAssignmentContext | None = None
    if set(_EYE_COMPONENTS).issubset(component_names):
        eye_assignment_context = _resolve_eye_assignment_context(
            root,
            source,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )

    results: list[dict[str, object]] = []
    rows_completed_in_shard = 0
    shard_start = int(chunk_specs[0][1]) if chunk_specs else 0
    shard_stop = int(chunk_specs[-1][2]) if chunk_specs else 0
    for chunk_ordinal, (chunk_index, start_row, stop_row) in enumerate(chunk_specs, start=1):
        chunk_start = int(start_row)
        chunk_stop = int(stop_row)
        try:
            chunk_result = _process_and_write_finalizer_chunk_open(
                source=source,
                run_group=run_group,
                component_names=component_names,
                required_raw_components=required_raw_components,
                start_row=chunk_start,
                stop_row=chunk_stop,
                chunk_index=int(chunk_index),
                metric_level=metric_level,
                eye_assignment_context=eye_assignment_context,
                retain_source_seeds=retain_source_seeds,
                execution_backend=_PROCESS_SHARD_EXECUTION_BACKEND,
            )
        except Exception as exc:
            progress.emit(
                "process_shard_chunk_failed",
                shard_index=int(shard_index),
                total_shards=int(total_shards),
                worker_count=int(worker_count),
                chunk_index=int(chunk_index),
                chunk_ordinal=int(chunk_ordinal),
                chunk_count=int(len(chunk_specs)),
                start_row=chunk_start,
                stop_row=chunk_stop,
                shard_start_row=shard_start,
                shard_stop_row=shard_stop,
                error=repr(exc),
            )
            raise
        results.append(chunk_result)
        rows_completed_in_shard += chunk_stop - chunk_start
        chunk_timing = dict(chunk_result.get("chunk_timing") or {})
        progress.emit(
            "process_shard_chunk_completed",
            shard_index=int(shard_index),
            total_shards=int(total_shards),
            worker_count=int(worker_count),
            chunk_index=int(chunk_index),
            chunk_ordinal=int(chunk_ordinal),
            chunk_count=int(len(chunk_specs)),
            start_row=chunk_start,
            stop_row=chunk_stop,
            shard_start_row=shard_start,
            shard_stop_row=shard_stop,
            rows_completed_in_shard=int(rows_completed_in_shard),
            shard_rows_total=int(shard_stop - shard_start),
            duration_seconds=float(chunk_timing.get("total_seconds") or 0.0),
        )
    return results


def _compute_finalizer_process_shards(
    zarr_path: str,
    *,
    subject_run: str,
    subject_shard_runs: Sequence[str] | None = None,
    target_crop_run: Optional[str] = None,
    refined_run: str,
    component_names: Sequence[str],
    required_raw_components: Sequence[str],
    chunk_ranges: Sequence[tuple[int, int]],
    metric_level: str,
    assignment_keypoint_group: Optional[str],
    assignment_keypoints_run: Optional[str],
    retain_source_seeds: bool,
    num_workers: Optional[int],
    progress: Optional[_ProgressJsonlReporter] = None,
) -> list[dict[str, object]]:
    shards = _row_chunk_shards(chunk_ranges, num_workers=num_workers)
    if not shards:
        return []
    worker_count = len(shards)
    progress = progress or _ProgressJsonlReporter()
    progress.emit(
        "process_shards_submitted",
        worker_count=int(worker_count),
        shard_count=int(len(shards)),
        chunk_count=int(len(chunk_ranges)),
    )
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_shard = {}
        for shard_index, shard in enumerate(shards):
            first_chunk = shard[0]
            last_chunk = shard[-1]
            shard_payload = {
                "shard_index": int(shard_index),
                "chunk_count": int(len(shard)),
                "start_row": int(first_chunk[1]),
                "stop_row": int(last_chunk[2]),
                "total_shards": int(len(shards)),
                "worker_count": int(worker_count),
            }
            progress.emit("process_shard_submitted", **shard_payload)
            future = executor.submit(
                _process_and_write_finalizer_shard,
                str(zarr_path),
                subject_run=subject_run,
                subject_shard_runs=tuple(subject_shard_runs or ()),
                target_crop_run=target_crop_run,
                refined_run=refined_run,
                component_names=tuple(component_names),
                required_raw_components=tuple(required_raw_components),
                chunk_specs=tuple(shard),
                metric_level=metric_level,
                assignment_keypoint_group=assignment_keypoint_group,
                assignment_keypoints_run=assignment_keypoints_run,
                retain_source_seeds=retain_source_seeds,
                shard_index=int(shard_index),
                total_shards=int(len(shards)),
                worker_count=int(worker_count),
                progress_jsonl=str(progress.path) if progress.path is not None else None,
                progress_start_time=float(progress.start_time),
            )
            future_to_shard[future] = shard_payload
        completed = 0
        rows_completed = 0
        total_rows = int(sum(int(stop) - int(start) for start, stop in chunk_ranges))
        for future in as_completed(future_to_shard):
            shard_payload = dict(future_to_shard[future])
            try:
                shard_results = list(future.result())
            except Exception as exc:
                progress.emit("process_shard_failed", error=repr(exc), **shard_payload)
                raise
            results.extend(shard_results)
            completed += 1
            rows_completed += int(shard_payload["stop_row"]) - int(shard_payload["start_row"])
            progress.emit(
                "process_shard_completed",
                completed_shards=int(completed),
                rows_completed=int(rows_completed),
                rows_total=int(total_rows),
                **shard_payload,
            )
    return results


def finalize_subject_mask_run(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    subject_run: Optional[str] = None,
    subject_shard_runs: Sequence[str] | None = None,
    target_crop_run: str | None = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    metric_level: str = "cheap",
    write_eye_geometry: bool = False,
    write_component_contours: bool = False,
    write_sampled_component_contours: bool = False,
    sampled_contour_counts: Mapping[str, int] | None = None,
    sampled_contour_row_chunk: int = DEFAULT_SAMPLED_CONTOUR_ROW_CHUNK,
    retain_source_seeds: bool = False,
    mask_storage: str = "dense_uint8",
    mask_rle_validation_mode: str = "full",
    dense_mask_row_chunk: int | None = None,
    postcompute_backend: str = _SERIAL_POSTCOMPUTE_BACKEND,
    postcompute_chunk_size: Optional[int] = None,
    postcompute_num_workers: Optional[int] = None,
    execution_backend: str = _SERIAL_EXECUTION_BACKEND,
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
    progress_jsonl: str | Path | None = None,
) -> dict[str, object]:
    """Finalize one subject-mask run into a canonical refined-subject run."""

    if bool(assignment_keypoint_group) != bool(assignment_keypoints_run):
        raise ValueError("Pass both assignment_keypoint_group and assignment_keypoints_run, or neither.")
    metric_level = str(metric_level)
    if metric_level not in _METRIC_LEVELS:
        raise ValueError(f"metric_level must be one of {_METRIC_LEVELS}; got {metric_level!r}.")
    mask_storage = str(mask_storage)
    if mask_storage not in _MASK_STORAGE_CHOICES:
        raise ValueError(f"mask_storage must be one of {_MASK_STORAGE_CHOICES}; got {mask_storage!r}.")
    if mask_storage in _MASK_STORAGE_COMPACT_ONLY:
        raise ValueError(
            f"mask_storage={mask_storage!r} is compact-only and is no longer valid for editable refined "
            "subject-mask outputs. Use dense_uint8, dense_and_bitpacked, dense_and_rle, or "
            "dense_bitpacked_and_rle so masks_roi remains the authoritative edit surface."
        )
    direct_bitpacked_output = mask_storage in _MASK_STORAGES_DIRECT_BITPACKED
    mask_rle_validation_mode = str(mask_rle_validation_mode).strip().lower()
    if mask_rle_validation_mode not in _MASK_RLE_VALIDATION_MODES:
        raise ValueError(
            f"mask_rle_validation_mode must be one of {_MASK_RLE_VALIDATION_MODES}; "
            f"got {mask_rle_validation_mode!r}."
        )
    normalized_num_workers = int(num_workers) if num_workers is not None else None
    execution_backend = _normalize_execution_backend(execution_backend)
    finalization_metric_write_policy = FINALIZATION_METRIC_WRITE_POLICY
    postcompute_backend = _normalize_postcompute_backend(postcompute_backend)
    normalized_postcompute_chunk_size = (
        max(1, int(postcompute_chunk_size))
        if postcompute_chunk_size is not None
        else max(1, int(chunk_size))
    )
    normalized_postcompute_num_workers = (
        max(1, int(postcompute_num_workers))
        if postcompute_num_workers is not None
        else normalized_num_workers
    )
    resolved_sampled_contour_counts = {
        str(name): int(value)
        for name, value in {**DEFAULT_SAMPLED_CONTOUR_COUNTS, **dict(sampled_contour_counts or {})}.items()
    }
    if any(value <= 0 for value in resolved_sampled_contour_counts.values()):
        raise ValueError("All sampled contour counts must be positive.")
    normalized_sampled_contour_row_chunk = max(1, int(sampled_contour_row_chunk))
    if execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND and zarr_path is None:
        raise ValueError(f"execution_backend={execution_backend!r} requires a filesystem zarr_path.")
    if (
        postcompute_backend == _PROCESS_SHARD_POSTCOMPUTE_BACKEND
        and (write_eye_geometry or write_component_contours or write_sampled_component_contours)
        and zarr_path is None
    ):
        raise ValueError("postcompute_backend='process_shards' requires a filesystem zarr_path.")
    source, shard_collection = _load_subject_mask_source(
        root,
        subject_run=subject_run,
        subject_shard_runs=subject_shard_runs,
        target_crop_run=target_crop_run,
    )
    total_rows = int(source.masks_roi.shape[0])
    height = int(source.masks_roi.shape[2])
    width = int(source.masks_roi.shape[3])
    if source.source_crop_row_ids is None:
        source_parent = SUBJECT_MASK_SHARD_PARENT if shard_collection is not None else SUBJECT_MASK_CANONICAL_PARENT
        raise ValueError(
            f"{source_parent}/{source.run_name} cannot be finalized without source_crop_row_ids; "
            "write or backfill explicit crop-row lineage first."
        )
    effective_dense_mask_row_chunk = refined_subject_mask_storage_row_chunk(total_rows, dense_mask_row_chunk)
    finalization_metric_row_chunk = max(
        1,
        min(int(FINALIZATION_METRIC_ROW_CHUNK), int(total_rows)),
    )
    common_metric_row_chunk = _common_derived_metric_row_chunk(total_rows)
    common_metric_write_policy = COMMON_DERIVED_METRIC_WRITE_POLICY
    dense_mask_storage_chunks = refined_subject_mask_storage_chunks(
        total_rows,
        height,
        width,
        effective_dense_mask_row_chunk,
    )
    requested_chunk_size = max(1, int(chunk_size))
    worker_chunk_size = _worker_chunk_size_for_backend(
        total_rows,
        requested_chunk_size,
        execution_backend,
        dense_mask_row_chunk=effective_dense_mask_row_chunk,
    )
    chunk_ranges = _row_chunks(total_rows, worker_chunk_size)
    execution_metadata: dict[str, object] = {
        "execution_backend": execution_backend,
        "process_shard_execution_enabled": execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND,
        "worker_process_count": (
            normalized_num_workers if execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND else None
        ),
        "requested_chunk_size": requested_chunk_size,
        "worker_chunk_size": worker_chunk_size,
        "chunk_alignment": _chunk_alignment_label(
            execution_backend,
            dense_mask_row_chunk=effective_dense_mask_row_chunk,
        ),
    }
    component_names = _requested_output_components(source, components)
    if not component_names:
        raise ValueError(f"subject_mask_runs/{source.run_name} has no finalizable subject-mask components.")
    required_raw_components = []
    if "subject_body" in component_names:
        required_raw_components.append("subject_body")
    if set(_EYE_COMPONENTS).issubset(component_names):
        required_raw_components.append(_RAW_EYE_UNION_COMPONENT)
    if "swim_bladder" in component_names:
        required_raw_components.append("swim_bladder")
    for raw_component in required_raw_components:
        if raw_component not in _FINALIZABLE_RAW_COMPONENTS:
            raise ValueError(f"Unsupported raw component {raw_component!r}.")
        _require_available_component(source, raw_component, "subject_mask_runs")

    target_run = str(refined_run or _default_refined_run_name())
    refined_parent = root.get("refined_subject_masks_runs")
    target_exists = refined_parent is not None and target_run in refined_parent
    summary: dict[str, object] = {
        "status": "planned" if dry_run else "updated",
        "source_subject_mask_run": source.run_name,
        "refined_run": target_run,
        "refined_run_exists": bool(target_exists),
        "would_create_refined_run": not bool(target_exists),
        "mutates_archive": not bool(dry_run),
        "component_names": list(component_names),
        "raw_component_names": list(required_raw_components),
        "source_crop_run": source.crop_run,
        "roi_count": total_rows,
        "chunk_size": requested_chunk_size,
        "worker_chunk_size": worker_chunk_size,
        "chunk_count": len(chunk_ranges),
        "dense_mask_row_chunk": int(effective_dense_mask_row_chunk),
        "dense_mask_storage_chunks": [int(value) for value in dense_mask_storage_chunks],
        "finalization_metric_row_chunk": int(finalization_metric_row_chunk),
        "finalization_metric_write_policy": finalization_metric_write_policy,
        "common_metric_row_chunk": int(common_metric_row_chunk),
        "common_metric_component_chunk": int(len(component_names)),
        "common_metric_write_policy": common_metric_write_policy,
        "metric_level": metric_level,
        "write_eye_geometry": bool(write_eye_geometry),
        "write_component_contours": bool(write_component_contours),
        "write_sampled_component_contours": bool(write_sampled_component_contours),
        "sampled_contour_counts": dict(resolved_sampled_contour_counts),
        "sampled_contour_row_chunk": int(normalized_sampled_contour_row_chunk),
        "retain_source_seeds": bool(retain_source_seeds),
        "source_seed_masks_status": "retained" if bool(retain_source_seeds) else "omitted",
        "mask_storage": mask_storage,
        "mask_rle_validation_mode": mask_rle_validation_mode,
        "would_write_mask_bitpacked": mask_storage in _MASK_STORAGES_WITH_BITPACKED,
        "would_write_mask_rle": mask_storage in _MASK_STORAGES_WITH_RLE,
        "postcompute_backend": postcompute_backend,
        "postcompute_chunk_size": normalized_postcompute_chunk_size,
        "postcompute_num_workers": normalized_postcompute_num_workers,
        **execution_metadata,
        "source_surface_kind": source.mask_surface_kind,
    }
    if shard_collection is not None:
        summary.update(
            {
                "collection_finalizer_schema": SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA,
                "finalized_from_subject_mask_shards": True,
                "source_subject_mask_shard_runs": list(shard_collection.shard_runs),
                "source_subject_mask_shard_run_paths": list(shard_collection.shard_run_paths),
                "source_subject_mask_shard_crop_runs": list(shard_collection.shard_crop_runs),
                "source_crop_rebased_from_shards": bool(shard_collection.source_crop_rebased_from_shards),
                "source_crop_rebase_target_run": str(target_crop_run or ""),
            }
        )
    if dry_run:
        return summary

    progress = _ProgressJsonlReporter(Path(progress_jsonl).expanduser().resolve() if progress_jsonl else None)
    progress.emit(
        "start",
        zarr_path=str(zarr_path or ""),
        subject_run=str(source.run_name),
        refined_run=str(target_run),
        total_rows=int(total_rows),
        requested_chunk_size=int(requested_chunk_size),
        worker_chunk_size=int(worker_chunk_size),
        dense_mask_row_chunk=int(effective_dense_mask_row_chunk),
        dense_mask_storage_chunks=[int(value) for value in dense_mask_storage_chunks],
        finalization_metric_row_chunk=int(finalization_metric_row_chunk),
        finalization_metric_write_policy=finalization_metric_write_policy,
        common_metric_row_chunk=int(common_metric_row_chunk),
        common_metric_component_chunk=int(len(component_names)),
        common_metric_write_policy=common_metric_write_policy,
        chunk_count=int(len(chunk_ranges)),
        execution_backend=str(execution_backend),
        num_workers=normalized_num_workers,
        write_eye_geometry=bool(write_eye_geometry),
        write_component_contours=bool(write_component_contours),
        write_sampled_component_contours=bool(write_sampled_component_contours),
        sampled_contour_counts=dict(resolved_sampled_contour_counts),
        sampled_contour_row_chunk=int(normalized_sampled_contour_row_chunk),
        retain_source_seeds=bool(retain_source_seeds),
        mask_storage=str(mask_storage),
        mask_rle_validation_mode=str(mask_rle_validation_mode),
        source_seed_masks_status="retained" if bool(retain_source_seeds) else "omitted",
        postcompute_backend=str(postcompute_backend),
        postcompute_chunk_size=int(normalized_postcompute_chunk_size),
        postcompute_num_workers=normalized_postcompute_num_workers,
        finalized_from_subject_mask_shards=bool(shard_collection is not None),
        source_subject_mask_shard_runs=list(shard_collection.shard_runs) if shard_collection is not None else [],
        source_crop_rebase_target_run=str(target_crop_run or ""),
    )

    refined_parent = require_runs_parent(root, "refined_subject_masks_runs")
    if target_run in refined_parent:
        if not overwrite:
            raise ValueError(
                f"refined_subject_masks_runs/{target_run} already exists. Pass overwrite=True to replace it."
            )
        del refined_parent[target_run]

    stage_start = time.perf_counter()
    timing = _TimingRecorder()
    component_names = tuple(component_names)
    component_to_index = {name: idx for idx, name in enumerate(component_names)}
    reason_labels_by_component = {
        component_name: np.full((total_rows,), "clean", dtype=object)
        for component_name in component_names
    }
    review_counts: dict[str, dict[str, int]] = {}
    source_payloads: dict[str, dict[str, object]] = {}
    first_batches: dict[str, _FinalizedComponentBatch] = {}
    common_metrics_by_component: dict[str, dict[str, object]] = {}
    finalization_metrics_by_component: dict[str, dict[str, object]] = {}
    rows_with_nonempty_masks = 0
    eyes_union_assignment_summary: dict[str, object] = {}
    assignment_eye_geometry_shards: list[dict[str, object]] = []
    assignment_keypoint_attrs: dict[str, object] = {}
    eye_assignment_context: _EyeAssignmentContext | None = None
    if set(_EYE_COMPONENTS).issubset(component_names):
        eye_assignment_context = _resolve_eye_assignment_context(
            root,
            source,
            assignment_keypoint_group=assignment_keypoint_group,
            assignment_keypoints_run=assignment_keypoints_run,
        )
        assignment_keypoint_attrs = build_assignment_keypoint_attrs(
            eye_assignment_context.keypoint_run_name,
            assignment_keypoint_group=eye_assignment_context.keypoint_group_name,
            selection=eye_assignment_context.keypoint_source_kind,
        )
        assignment_keypoint_attrs["assignment_keypoint_success_dataset"] = str(
            eye_assignment_context.keypoint_success_dataset
        )
        assignment_keypoint_attrs["assignment_keypoint_row_identity"] = dict(
            eye_assignment_context.row_identity_summary
        )
        assignment_keypoint_attrs["assignment_keypoint_row_identity_check"] = str(
            eye_assignment_context.row_identity_summary.get("row_identity_check", "unknown")
        )

    extra_attrs: dict[str, object] = {
        "finalization_semantics": "smart_probability_to_refined_candidate",
        "component_metric_level": metric_level,
        "eye_geometry_requested": bool(write_eye_geometry),
        "component_contours_requested": bool(write_component_contours),
        "sampled_component_contours_requested": bool(write_sampled_component_contours),
        "sampled_contour_counts": dict(resolved_sampled_contour_counts),
        "sampled_contour_row_chunk": int(normalized_sampled_contour_row_chunk),
        "retain_source_seeds": bool(retain_source_seeds),
        "source_seed_masks_status": "retained" if bool(retain_source_seeds) else "omitted",
        "source_seed_masks_reason": "retain_source_seeds=true" if bool(retain_source_seeds) else "production_default",
        "mask_storage_requested": mask_storage,
        "mask_rle_validation_mode": mask_rle_validation_mode,
        "dense_mask_row_chunk": int(effective_dense_mask_row_chunk),
        "dense_mask_storage_chunks": [int(value) for value in dense_mask_storage_chunks],
        "finalization_metric_row_chunk": int(finalization_metric_row_chunk),
        "finalization_metric_write_policy": finalization_metric_write_policy,
        "common_metric_row_chunk": int(common_metric_row_chunk),
        "common_metric_component_chunk": int(len(component_names)),
        "common_metric_write_policy": common_metric_write_policy,
        "postcompute_backend": postcompute_backend,
        "postcompute_chunk_size": int(normalized_postcompute_chunk_size),
        "postcompute_num_workers": normalized_postcompute_num_workers,
        **execution_metadata,
        "source_input_subject_mask_run": source.run_name,
        "source_component_runs": {
            component_name: source.run_name
            for component_name in component_names
        },
        "source_component_sources": {
            component_name: {
                "source_stage": SUBJECT_MASK_SHARD_PARENT if shard_collection is not None else SUBJECT_MASK_CANONICAL_PARENT,
                "source_run": source.run_name,
            }
            for component_name in component_names
        },
    }
    if shard_collection is not None:
        extra_attrs.update(
            {
                "collection_finalizer_schema": SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA,
                "finalized_from_subject_mask_shards": True,
                "source_subject_mask_shard_runs": list(shard_collection.shard_runs),
                "source_subject_mask_shard_run_paths": list(shard_collection.shard_run_paths),
                "source_subject_mask_shard_crop_runs": list(shard_collection.shard_crop_runs),
                "source_crop_rebased_from_shards": bool(shard_collection.source_crop_rebased_from_shards),
                "source_crop_rebase_target_run": str(target_crop_run or ""),
            }
        )
    if eye_assignment_context is not None:
        extra_attrs.update(assignment_keypoint_attrs)

    provenance_inputs: dict[str, object] = {
        "source_input_subject_mask_run": source.run_name,
        "finalization_semantics": "smart_probability_to_refined_candidate",
        "source_component_runs": dict(extra_attrs["source_component_runs"]),
        "source_component_sources": dict(extra_attrs["source_component_sources"]),
        "chunk_size": int(requested_chunk_size),
        "worker_chunk_size": int(worker_chunk_size),
        "chunk_count": int(len(chunk_ranges)),
        "dense_mask_row_chunk": int(effective_dense_mask_row_chunk),
        "dense_mask_storage_chunks": [int(value) for value in dense_mask_storage_chunks],
        "finalization_metric_row_chunk": int(finalization_metric_row_chunk),
        "finalization_metric_write_policy": finalization_metric_write_policy,
        "common_metric_row_chunk": int(common_metric_row_chunk),
        "common_metric_component_chunk": int(len(component_names)),
        "common_metric_write_policy": common_metric_write_policy,
        "component_metric_level": metric_level,
        "eye_geometry_requested": bool(write_eye_geometry),
        "component_contours_requested": bool(write_component_contours),
        "sampled_component_contours_requested": bool(write_sampled_component_contours),
        "sampled_contour_counts": dict(resolved_sampled_contour_counts),
        "sampled_contour_row_chunk": int(normalized_sampled_contour_row_chunk),
        "retain_source_seeds": bool(retain_source_seeds),
        "source_seed_masks_status": "retained" if bool(retain_source_seeds) else "omitted",
        "source_seed_masks_reason": "retain_source_seeds=true" if bool(retain_source_seeds) else "production_default",
        "mask_storage_requested": mask_storage,
        "mask_rle_validation_mode": mask_rle_validation_mode,
        "postcompute_backend": postcompute_backend,
        "postcompute_chunk_size": int(normalized_postcompute_chunk_size),
        "postcompute_num_workers": normalized_postcompute_num_workers,
        **execution_metadata,
    }
    if shard_collection is not None:
        provenance_inputs.update(
            {
                "collection_finalizer_schema": SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA,
                "finalized_from_subject_mask_shards": True,
                "source_subject_mask_shard_runs": list(shard_collection.shard_runs),
                "source_subject_mask_shard_run_paths": list(shard_collection.shard_run_paths),
                "source_subject_mask_shard_crop_runs": list(shard_collection.shard_crop_runs),
                "source_crop_rebased_from_shards": bool(shard_collection.source_crop_rebased_from_shards),
                "source_crop_rebase_target_run": str(target_crop_run or ""),
            }
        )
    if eye_assignment_context is not None:
        provenance_inputs.update(assignment_keypoint_attrs)

    with progress.phase("target_init"):
        with timing.phase("target_init"):
            run_group = _create_refined_run_shell(
                refined_parent=refined_parent,
                target_run=target_run,
                source=source,
                component_names=component_names,
                extra_attrs=extra_attrs,
                provenance_inputs=provenance_inputs,
                stage_command=" ".join(sys.argv) if sys.argv else "unknown",
                retain_source_seeds=bool(retain_source_seeds),
                dense_mask_row_chunk=int(effective_dense_mask_row_chunk),
                create_dense_masks=not direct_bitpacked_output,
                create_bitpacked_masks=direct_bitpacked_output,
            )

    if execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND:
        assert zarr_path is not None
        worker_subject_run = "" if shard_collection is not None else str(source.run_name)
        with progress.phase("parallel_prepare_shells"):
            with timing.phase("parallel_prepare_shells"):
                for raw_component in required_raw_components:
                    sample_batch = _finalize_source_component_rows(
                        source,
                        raw_component,
                        start_row=0,
                        stop_row=min(1, total_rows),
                    )
                    first_batches.setdefault(raw_component, sample_batch)
                    if raw_component == _RAW_EYE_UNION_COMPONENT:
                        continue
                    source_payloads.setdefault(
                        raw_component,
                        _source_payload_for_finalized_component(source, sample_batch),
                    )

        with progress.phase("process_shard_compute"):
            with timing.phase("process_shard_compute"):
                parallel_results = _compute_finalizer_process_shards(
                    str(zarr_path),
                    subject_run=worker_subject_run,
                    subject_shard_runs=tuple(subject_shard_runs or ()),
                    target_crop_run=target_crop_run,
                    refined_run=target_run,
                    component_names=component_names,
                    required_raw_components=tuple(required_raw_components),
                    chunk_ranges=chunk_ranges,
                    metric_level=metric_level,
                    assignment_keypoint_group=assignment_keypoint_group,
                    assignment_keypoints_run=assignment_keypoints_run,
                    retain_source_seeds=bool(retain_source_seeds),
                    num_workers=normalized_num_workers,
                    progress=progress,
                )

        for result in sorted(parallel_results, key=lambda item: int(dict(item["chunk_timing"]).get("chunk_index") or 0)):
            chunk_timing = dict(result["chunk_timing"])
            timing.chunk_timings.append(chunk_timing)
            for phase, seconds in dict(result.get("phase_seconds") or {}).items():
                timing.add(str(phase), float(seconds))
            _merge_review_counts(review_counts, dict(result.get("review_counts") or {}))
            rows_with_nonempty_masks += int(result.get("rows_with_nonempty_masks") or 0)
            row_slice = slice(int(chunk_timing["start_row"]), int(chunk_timing["stop_row"]))
            for component_name, labels in dict(result.get("reason_labels_by_component") or {}).items():
                reason_labels_by_component[str(component_name)][row_slice] = np.asarray(labels, dtype=object)
            for component_name, payload in dict(result.get("common_metrics_by_component") or {}).items():
                _merge_common_metric_chunk_payload(
                    common_metrics_by_component,
                    component_name=str(component_name),
                    row_slice=row_slice,
                    payload=dict(payload),
                    total_rows=total_rows,
                )
            for component_name, payload in dict(
                result.get("finalization_metrics_by_component") or {}
            ).items():
                _merge_finalization_metric_chunk_payload(
                    finalization_metrics_by_component,
                    component_name=str(component_name),
                    row_slice=row_slice,
                    payload=dict(payload),
                    total_rows=total_rows,
                )
            if result.get("eyes_union_assignment_summary"):
                _merge_assignment_summary(
                    eyes_union_assignment_summary,
                    dict(result["eyes_union_assignment_summary"]),
                )
            eye_geometry_payload = result.get("eye_geometry")
            if isinstance(eye_geometry_payload, Mapping):
                assignment_eye_geometry_shards.append(
                    {
                        "start_row": int(chunk_timing["start_row"]),
                        "stop_row": int(chunk_timing["stop_row"]),
                        "eye_geometry": eye_geometry_payload,
                    }
                )
        for component_name in component_names:
            _set_component_metric_attrs(
                run_group["components"][component_name],
                metric_level=metric_level,
            )
    else:
        for chunk_index, (start_row, stop_row) in enumerate(chunk_ranges):
            chunk_start = time.perf_counter()
            progress.emit(
                "serial_chunk_start",
                chunk_index=int(chunk_index),
                start_row=int(start_row),
                stop_row=int(stop_row),
            )
            chunk_timing: dict[str, object] = {
                "chunk_index": int(chunk_index),
                "start_row": int(start_row),
                "stop_row": int(stop_row),
                "row_count": int(stop_row) - int(start_row),
                "execution_backend": _SERIAL_EXECUTION_BACKEND,
            }
            row_slice = slice(int(start_row), int(stop_row))
            chunk_any = np.zeros((int(stop_row) - int(start_row),), dtype=bool)
            chunk_batches: dict[str, _FinalizedComponentBatch] = {}
            for raw_component in required_raw_components:
                phase_start = time.perf_counter()
                batch = _finalize_source_component_rows(
                    source,
                    raw_component,
                    start_row=start_row,
                    stop_row=stop_row,
                )
                elapsed = timing.add(f"finalize_{raw_component}", time.perf_counter() - phase_start)
                chunk_timing[f"finalize_{raw_component}_seconds"] = elapsed
                chunk_batches[raw_component] = batch
                first_batches.setdefault(raw_component, batch)
                if raw_component == _RAW_EYE_UNION_COMPONENT:
                    continue
                component_idx = int(component_to_index[raw_component])
                phase_start = time.perf_counter()
                write_result = _write_canonical_component_chunk(
                    run_group,
                    component_name=raw_component,
                    component_idx=component_idx,
                    row_slice=row_slice,
                    masks=batch.masks,
                    source_masks=batch.source_masks,
                    metric_level=metric_level,
                    precomputed_component_metrics=_component_metrics_from_finalization_batch(
                        batch,
                        row_count=int(stop_row) - int(start_row),
                    ),
                    write_derived_metrics=False,
                    retain_source_seeds=bool(retain_source_seeds),
                    timing=timing,
                    chunk_timing=chunk_timing,
                )
                _merge_common_metric_chunk_payload(
                    common_metrics_by_component,
                    component_name=raw_component,
                    row_slice=row_slice,
                    payload=_common_metric_chunk_payload(write_result),
                    total_rows=total_rows,
                )
                chunk_any |= write_result.mask_present
                _merge_finalization_metric_chunk_payload(
                    finalization_metrics_by_component,
                    component_name=raw_component,
                    row_slice=row_slice,
                    payload=_finalization_metric_chunk_payload(batch),
                    total_rows=total_rows,
                )
                elapsed = timing.add(f"write_{raw_component}", time.perf_counter() - phase_start)
                chunk_timing[f"write_{raw_component}_seconds"] = elapsed
                labels = _merge_reason_label_arrays(batch.reason_labels, write_result.reason_labels)
                reason_labels_by_component[raw_component][row_slice] = labels
                _add_review_counts(review_counts, raw_component, labels)
                source_payloads.setdefault(raw_component, _source_payload_for_finalized_component(source, batch))

            if eye_assignment_context is not None:
                union_batch = chunk_batches[_RAW_EYE_UNION_COMPONENT]
                phase_start = time.perf_counter()
                assignment_chunk = _assign_finalized_eyes_union_rows(
                    source,
                    union_batch,
                    eye_assignment_context,
                    start_row=start_row,
                    stop_row=stop_row,
                )
                elapsed = timing.add("eye_assignment", time.perf_counter() - phase_start)
                chunk_timing["eye_assignment_seconds"] = elapsed
                _record_eye_assignment_phase_seconds(timing, chunk_timing, assignment_chunk.phase_seconds)
                _merge_assignment_summary(eyes_union_assignment_summary, assignment_chunk.summary)
                if isinstance(assignment_chunk.eye_geometry, Mapping):
                    assignment_eye_geometry_shards.append(
                        {
                            "start_row": int(start_row),
                            "stop_row": int(stop_row),
                            "eye_geometry": assignment_chunk.eye_geometry,
                        }
                    )
                for component_name in _EYE_COMPONENTS:
                    component_idx = int(component_to_index[component_name])
                    masks = np.asarray(assignment_chunk.masks[component_name], dtype=np.uint8)
                    phase_start = time.perf_counter()
                    write_result = _write_canonical_component_chunk(
                        run_group,
                        component_name=component_name,
                        component_idx=component_idx,
                        row_slice=row_slice,
                        masks=masks,
                        source_masks=masks,
                        metric_level=metric_level,
                        precomputed_component_metrics=assignment_chunk.component_metrics.get(component_name),
                        write_derived_metrics=False,
                        retain_source_seeds=bool(retain_source_seeds),
                        timing=timing,
                        chunk_timing=chunk_timing,
                    )
                    _merge_common_metric_chunk_payload(
                        common_metrics_by_component,
                        component_name=component_name,
                        row_slice=row_slice,
                        payload=_common_metric_chunk_payload(write_result),
                        total_rows=total_rows,
                    )
                    elapsed = timing.add(f"write_{component_name}", time.perf_counter() - phase_start)
                    chunk_timing[f"write_{component_name}_seconds"] = elapsed
                    chunk_any |= write_result.mask_present
                    labels = _merge_reason_label_arrays(
                        assignment_chunk.reason_labels[component_name],
                        write_result.reason_labels,
                    )
                    reason_labels_by_component[component_name][row_slice] = labels
                    _add_review_counts(review_counts, component_name, labels)

            rows_with_nonempty_masks += int(np.count_nonzero(chunk_any))
            chunk_timing["total_seconds"] = float(time.perf_counter() - chunk_start)
            timing.chunk_timings.append(chunk_timing)
            progress.emit(
                "serial_chunk_completed",
                chunk_index=int(chunk_index),
                start_row=int(start_row),
                stop_row=int(stop_row),
                row_count=int(stop_row) - int(start_row),
                rows_completed=int(stop_row),
                rows_total=int(total_rows),
                total_seconds=float(chunk_timing["total_seconds"]),
            )

    missing_common_metric_components = sorted(set(component_names) - set(common_metrics_by_component))
    if missing_common_metric_components:
        raise RuntimeError(
            "Missing driver-merge common metrics for components: "
            f"{missing_common_metric_components}."
        )
    phase_name = "write_common_run_metrics"
    with progress.phase(
        phase_name,
        write_policy=COMMON_DERIVED_METRIC_WRITE_POLICY,
        row_chunk=int(_common_derived_metric_row_chunk(total_rows)),
        component_chunk=int(len(component_names)),
    ):
        with timing.phase(phase_name):
            _write_common_run_metrics(
                run_group,
                component_names=component_names,
                payloads_by_component=common_metrics_by_component,
            )
    for component_name in component_names:
        phase_name = f"write_common_metrics_{component_name}"
        with progress.phase(
            phase_name,
            component=component_name,
            write_policy=COMMON_DERIVED_METRIC_WRITE_POLICY,
            row_chunk=int(_common_derived_metric_row_chunk(total_rows)),
        ):
            with timing.phase(phase_name):
                _write_common_component_metrics(
                    run_group,
                    component_name=component_name,
                    payload=common_metrics_by_component[component_name],
                    metric_level=metric_level,
                )

    expected_finalization_metric_components = {
        str(component_name)
        for component_name in required_raw_components
        if str(component_name) != _RAW_EYE_UNION_COMPONENT
    }
    missing_finalization_metric_components = sorted(
        expected_finalization_metric_components - set(finalization_metrics_by_component)
    )
    if missing_finalization_metric_components:
        raise RuntimeError(
            "Missing driver-merge finalization metrics for components: "
            f"{missing_finalization_metric_components}."
        )
    for component_name in sorted(finalization_metrics_by_component):
        phase_name = f"write_finalization_metrics_{component_name}"
        with progress.phase(
            phase_name,
            component=component_name,
            write_policy=finalization_metric_write_policy,
            row_chunk=int(finalization_metric_row_chunk),
        ):
            with timing.phase(phase_name):
                _write_finalization_metrics_component(
                    run_group,
                    component_name=component_name,
                    payload=finalization_metrics_by_component[component_name],
                    total_rows=total_rows,
                    source_batch=first_batches.get(component_name),
                )

    if eye_assignment_context is not None:
        usable_rows = int(eyes_union_assignment_summary.get("assigned_rows") or 0) + int(
            eyes_union_assignment_summary.get("assigned_needs_review_rows") or 0
        )
        if usable_rows <= 0:
            raise ValueError(
                f"eyes_union assignment for subject_mask_runs/{source.run_name} produced no usable LR eye rows; "
                f"summary={eyes_union_assignment_summary!r}."
            )
        union_batch = first_batches[_RAW_EYE_UNION_COMPONENT]
        run_group.attrs["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance = dict(run_group.attrs.get("provenance") or {})
        provenance_inputs_payload = dict(provenance.get("inputs") or {})
        provenance_inputs_payload["eyes_union_assignment_summary"] = dict(_json_safe(eyes_union_assignment_summary))
        provenance["inputs"] = provenance_inputs_payload
        run_group.attrs["provenance"] = provenance
        for component_name in _EYE_COMPONENTS:
            source_payloads[component_name] = _source_payload_for_assigned_eye_component(
                source,
                component_name=component_name,
                union_batch=union_batch,
                assignment_summary=eyes_union_assignment_summary,
                keypoint_run_name=eye_assignment_context.keypoint_run_name,
                keypoint_group_name=eye_assignment_context.keypoint_group_name,
                keypoint_success_dataset=eye_assignment_context.keypoint_success_dataset,
                keypoint_source_kind=eye_assignment_context.keypoint_source_kind,
            )

    created = str(run_group.attrs.get("created_at_utc") or _utc_now())
    for component_name in component_names:
        component_group = run_group["components"][component_name]
        phase_name = f"write_reason_columns_{component_name}"
        with progress.phase(phase_name, component=component_name):
            with timing.phase(phase_name):
                write_reason_columns(
                    component_group,
                    reason_labels_by_component[component_name],
                    chunk_size=max(1, min(256, total_rows)),
                    include_reason_text=True,
                    overwrite=True,
                )
        phase_name = f"write_component_provenance_{component_name}"
        with progress.phase(phase_name, component=component_name):
            with timing.phase(phase_name):
                _ensure_refined_component_provenance_payload(
                    run_group,
                    component_name=component_name,
                    source_payload=source_payloads[component_name],
                    created_at_utc=created,
                )

    eye_components_present = set(_EYE_COMPONENTS).issubset(component_names)
    contour_components = _component_contour_targets(component_names)
    eye_geometry_requested = bool(write_eye_geometry and eye_components_present)
    component_contours_requested = bool(write_component_contours and contour_components)
    sampled_contour_counts_requested = (
        {
            name: int(resolved_sampled_contour_counts[name])
            for name in component_names
            if name in resolved_sampled_contour_counts
        }
        if write_sampled_component_contours
        else {}
    )
    sampled_component_contours_requested = bool(sampled_contour_counts_requested)
    assignment_eye_geometry_complete = bool(
        eye_geometry_requested
        and len(assignment_eye_geometry_shards) == len(chunk_ranges)
        and sum(int(item["stop_row"]) - int(item["start_row"]) for item in assignment_eye_geometry_shards)
        == int(total_rows)
    )
    assignment_eye_geometry_rows = int(
        sum(int(item["stop_row"]) - int(item["start_row"]) for item in assignment_eye_geometry_shards)
    )
    finalizer_warnings: list[dict[str, object]] = []
    postcompute_summary: dict[str, object] = {}
    contour_summaries: list[dict[str, object]] = []
    sampled_contour_summaries: list[dict[str, object]] = []
    remaining_sampled_contour_counts = dict(sampled_contour_counts_requested)

    if assignment_eye_geometry_complete:
        with progress.phase("write_eye_geometry_from_assignment"):
            with timing.phase("write_eye_geometry_from_assignment"):
                postcompute_summary["eye_geometry"] = _write_sharded_eye_geometry(
                    run_group,
                    sorted(assignment_eye_geometry_shards, key=lambda item: int(item["start_row"])),
                    chunk_rois=max(1, min(256, total_rows if total_rows > 0 else 1)),
                    refined_run=target_run,
                    postcompute_backend=_ASSIGNMENT_REUSE_POSTCOMPUTE_BACKEND,
                    write_component_contours=bool(component_contours_requested),
                )
        run_group.attrs["eye_geometry_status"] = "computed"
        sampled_eye_counts = {
            name: remaining_sampled_contour_counts.pop(name)
            for name in _EYE_COMPONENTS
            if name in remaining_sampled_contour_counts
        }
        if sampled_eye_counts:
            with progress.phase("write_sampled_eye_contours_from_assignment"):
                with timing.phase("write_sampled_eye_contours_from_assignment"):
                    assignment_sampled_summaries = _write_sampled_component_contours_from_raw_shards(
                        run_group,
                        sorted(assignment_eye_geometry_shards, key=lambda item: int(item["start_row"])),
                        sample_counts=sampled_eye_counts,
                        source_key="eye_geometry",
                        row_chunk=normalized_sampled_contour_row_chunk,
                        refined_run=target_run,
                        postcompute_backend=_ASSIGNMENT_REUSE_POSTCOMPUTE_BACKEND,
                    )
            sampled_contour_summaries.extend(_summaries_to_json_safe(assignment_sampled_summaries))
            postcompute_summary["sampled_component_contours"] = list(
                _json_safe(sampled_contour_summaries)
            )

    if (
        postcompute_backend == _PROCESS_SHARD_POSTCOMPUTE_BACKEND
        and (
            (eye_geometry_requested and not assignment_eye_geometry_complete)
            or component_contours_requested
            or bool(remaining_sampled_contour_counts)
        )
    ):
        assert zarr_path is not None
        with progress.phase(
            "postcompute_process_shards",
            write_eye_geometry=bool(eye_geometry_requested and not assignment_eye_geometry_complete),
            write_component_contours=bool(component_contours_requested),
            write_sampled_component_contours=bool(remaining_sampled_contour_counts),
            chunk_size=int(normalized_postcompute_chunk_size),
            num_workers=normalized_postcompute_num_workers,
        ):
            with timing.phase("postcompute_process_shards"):
                postcompute_result = _run_sharded_refined_subject_postcompute(
                    zarr_path,
                    refined_run=target_run,
                    chunk_size=int(normalized_postcompute_chunk_size),
                    num_workers=normalized_postcompute_num_workers,
                    write_eye_geometry=bool(eye_geometry_requested and not assignment_eye_geometry_complete),
                    write_component_contours=component_contours_requested,
                    sampled_contour_counts=remaining_sampled_contour_counts,
                    sampled_contour_row_chunk=normalized_sampled_contour_row_chunk,
                )
        for key, value in dict(postcompute_result).items():
            if key == "eye_geometry" and assignment_eye_geometry_complete:
                continue
            postcompute_summary[str(key)] = value
        contour_summaries = _summaries_to_json_safe(
            list(postcompute_summary.get("component_contours") or [])
        )
        process_sampled_summaries = _summaries_to_json_safe(
            list(postcompute_result.get("sampled_component_contours") or [])
        )
        _update_sampled_component_contour_attrs(
            run_group,
            process_sampled_summaries,
            postcompute_backend=_PROCESS_SHARD_POSTCOMPUTE_BACKEND,
        )
        sampled_contour_summaries.extend(process_sampled_summaries)
        postcompute_summary["sampled_component_contours"] = list(
            _json_safe(sampled_contour_summaries)
        )
        eye_summary = postcompute_summary.get("eye_geometry")
        if (
            not assignment_eye_geometry_complete
            and isinstance(eye_summary, Mapping)
            and str(eye_summary.get("status")) == "updated"
        ):
            run_group.attrs["eye_geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
            run_group.attrs["eye_geometry_updated_at_utc"] = _utc_now()
            run_group.attrs["eye_geometry_status"] = "computed"
            run_group.attrs["eye_geometry_postcompute_backend"] = _PROCESS_SHARD_POSTCOMPUTE_BACKEND
            run_group.attrs.pop("eye_geometry_deferred_reason", None)
        if contour_summaries:
            run_group.attrs["component_contours_status"] = "computed"
            run_group.attrs["component_contours_components"] = [
                str(item["component"]) for item in contour_summaries
            ]
            run_group.attrs["component_contours_updated_at_utc"] = _utc_now()
            run_group.attrs["component_contours_summary"] = list(_json_safe(contour_summaries))
            run_group.attrs["component_contours_postcompute_backend"] = _PROCESS_SHARD_POSTCOMPUTE_BACKEND
    elif eye_geometry_requested and not assignment_eye_geometry_complete:
        with progress.phase("write_eye_geometry"):
            with timing.phase("write_eye_geometry"):
                postcompute_summary["eye_geometry"] = write_refined_subject_eye_geometry(
                    run_group,
                    write_component_contours=bool(component_contours_requested),
                )
        run_group.attrs["eye_geometry_status"] = "computed"

    if eye_components_present and not write_eye_geometry:
        run_group.attrs["eye_geometry_status"] = "deferred"
        run_group.attrs["eye_geometry_deferred_reason"] = "write_eye_geometry=false"

    if not eye_geometry_requested:
        eye_geometry_reuse_status = _EYE_GEOMETRY_REUSE_STATUS_NOT_REQUESTED
    elif assignment_eye_geometry_complete:
        eye_geometry_reuse_status = _EYE_GEOMETRY_REUSE_STATUS_ASSIGNMENT_REUSE
    else:
        eye_geometry_reuse_status = _EYE_GEOMETRY_REUSE_STATUS_FALLBACK_REFIT
        warning = {
            "code": _EYE_GEOMETRY_FALLBACK_WARNING_CODE,
            "message": (
                "Eye geometry was requested but assignment-time geometry was incomplete; "
                "the finalizer used the fallback mask-refit path."
            ),
            "assignment_geometry_rows": int(assignment_eye_geometry_rows),
            "expected_rows": int(total_rows),
            "assignment_geometry_shards": int(len(assignment_eye_geometry_shards)),
            "expected_shards": int(len(chunk_ranges)),
        }
        finalizer_warnings.append(warning)
        progress.emit("warning", **warning)
    run_group.attrs["eye_geometry_reuse_status"] = str(eye_geometry_reuse_status)
    run_group.attrs["eye_geometry_assignment_geometry_rows"] = int(assignment_eye_geometry_rows)
    run_group.attrs["eye_geometry_assignment_geometry_expected_rows"] = int(total_rows if eye_geometry_requested else 0)
    run_group.attrs["eye_geometry_assignment_geometry_shards"] = int(len(assignment_eye_geometry_shards))
    run_group.attrs["eye_geometry_assignment_geometry_expected_shards"] = int(
        len(chunk_ranges) if eye_geometry_requested else 0
    )
    run_group.attrs["smart_finalizer_warnings"] = list(_json_safe(finalizer_warnings))

    if postcompute_backend != _PROCESS_SHARD_POSTCOMPUTE_BACKEND or not component_contours_requested:
        if component_contours_requested:
            with progress.phase("write_component_contours", components=list(contour_components)):
                with timing.phase("write_component_contours"):
                    contour_summaries = _summaries_to_json_safe(
                        write_refined_subject_component_contours(
                            run_group,
                            components=contour_components,
                            source_mask_run=target_run,
                            chunk_rois=max(1, min(256, total_rows)),
                            overwrite=True,
                        )
                    )
            run_group.attrs["component_contours_status"] = "computed"
            run_group.attrs["component_contours_components"] = list(contour_components)
            run_group.attrs["component_contours_updated_at_utc"] = _utc_now()
            run_group.attrs["component_contours_summary"] = list(_json_safe(contour_summaries))
            postcompute_summary["component_contours"] = list(_json_safe(contour_summaries))
    if postcompute_backend != _PROCESS_SHARD_POSTCOMPUTE_BACKEND and remaining_sampled_contour_counts:
        with progress.phase(
            "write_sampled_component_contours",
            components=list(remaining_sampled_contour_counts),
        ):
            with timing.phase("write_sampled_component_contours"):
                serial_sampled_summaries = _summaries_to_json_safe(
                    write_refined_subject_sampled_component_contours(
                        run_group,
                        components=tuple(remaining_sampled_contour_counts),
                        sample_counts=remaining_sampled_contour_counts,
                        source_mask_run=target_run,
                        row_chunk=normalized_sampled_contour_row_chunk,
                        read_chunk_size=max(1, min(256, total_rows)),
                    )
                )
                sampled_contour_summaries.extend(serial_sampled_summaries)
        _update_sampled_component_contour_attrs(
            run_group,
            serial_sampled_summaries,
            postcompute_backend=_SERIAL_POSTCOMPUTE_BACKEND,
        )
        postcompute_summary["sampled_component_contours"] = list(_json_safe(sampled_contour_summaries))
    if write_component_contours and not contour_components:
        run_group.attrs["component_contours_status"] = "deferred"
        run_group.attrs["component_contours_deferred_reason"] = "no_subject_body_or_swim_bladder_components_selected"
    if write_sampled_component_contours and not sampled_component_contours_requested:
        run_group.attrs["sampled_component_contours_status"] = "deferred"
        run_group.attrs["sampled_component_contours_deferred_reason"] = "no_supported_components_selected"
    sampled_order = {name: index for index, name in enumerate(CANONICAL_COMPONENT_ORDER)}
    sampled_contour_summaries = sorted(
        sampled_contour_summaries,
        key=lambda item: sampled_order.get(str(item.get("component")), len(sampled_order)),
    )
    if sampled_contour_summaries:
        postcompute_summary["sampled_component_contours"] = list(
            _json_safe(sampled_contour_summaries)
        )
    if postcompute_summary and "status" not in postcompute_summary:
        postcompute_summary = {
            "status": "updated",
            "postcompute_backend": postcompute_backend,
            **postcompute_summary,
        }
    elif not postcompute_summary:
        postcompute_summary = {
            "status": "skipped",
            "postcompute_backend": postcompute_backend,
            "reason": "no_expensive_postcompute_requested",
        }

    if direct_bitpacked_output:
        with progress.phase("validate_bitpacked_mask_store"):
            with timing.phase("validate_bitpacked_mask_store"):
                validation = validate_bitpacked_mask_store_invariants(
                    run_group,
                    expected_shape=(int(total_rows), int(len(component_names)), int(height), int(width)),
                    component_names=component_names,
                    source_path=f"refined_subject_masks_runs/{target_run}",
                )
        packed = run_group["mask_bitpacked/masks_packed"]
        logical_bytes = int(np.prod(tuple(int(value) for value in packed.shape), dtype=np.int64))
        mask_bitpacked_summary = {
            "status": "written_direct",
            "encoding": MASK_BITPACKED_ENCODING,
            "layout": MASK_BITPACKED_LAYOUT,
            "source_encoding": "chunk_finalizer_binary_masks",
            "rows": int(total_rows),
            "channels": int(len(component_names)),
            "shape": [int(total_rows), int(len(component_names)), int(height), int(width)],
            "encoded_shape": [int(value) for value in packed.shape],
            "component_names": list(component_names),
            "logical_bytes": int(logical_bytes),
            "validation_mode": "invariants",
            "requested_validation_mode": str(mask_rle_validation_mode),
            "mask_bitpacked_validation": validation,
            "roundtrip_validation": {
                "status": "skipped",
                "reason": "direct_bitpacked_output_has_no_dense_masks_roi",
            },
        }
    elif mask_storage in _MASK_STORAGES_WITH_BITPACKED:
        def _emit_bitpacked_progress(event: str, **payload: object) -> None:
            progress.emit(str(event), **payload)

        with progress.phase("write_bitpacked_mask_store"):
            with timing.phase("write_bitpacked_mask_store"):
                mask_bitpacked_summary = write_bitpacked_mask_store_from_dense(
                    run_group,
                    run_group["masks_roi"],
                    component_names=component_names,
                    overwrite=True,
                    encode_row_chunk_size=max(1, min(int(worker_chunk_size), 1024)),
                    validation_mode=mask_rle_validation_mode,
                    extra_attrs={
                        "source_array": "masks_roi",
                        "source_encoding": DENSE_MASK_ENCODING_V1,
                        "source_run": str(target_run),
                    },
                    progress_callback=_emit_bitpacked_progress,
                )
    else:
        mask_bitpacked_summary = {
            "status": "skipped",
            "reason": f"mask_storage={mask_storage}",
            "encoding": "bitpacked_binary_v1",
        }

    if mask_storage in _MASK_STORAGES_WITH_RLE:
        rle_encode_workers = max(1, int(normalized_num_workers or 1))
        rle_source_zarr_path = zarr_path if zarr_path is not None and rle_encode_workers > 1 else None
        rle_source_run_path = (
            f"refined_subject_masks_runs/{target_run}" if rle_source_zarr_path is not None else None
        )

        def _emit_rle_progress(event: str, **payload: object) -> None:
            progress.emit(str(event), **payload)

        with progress.phase("write_component_rle_mask_store"):
            with timing.phase("write_component_rle_mask_store"):
                mask_rle_summary = write_component_rle_mask_store_from_dense(
                    run_group,
                    run_group["masks_roi"],
                    component_names=component_names,
                    overwrite=True,
                    encode_row_chunk_size=max(1, min(int(worker_chunk_size), 1024)),
                    encode_workers=rle_encode_workers,
                    source_zarr_path=rle_source_zarr_path,
                    source_run_path=rle_source_run_path,
                    validation_mode=mask_rle_validation_mode,
                    extra_attrs={
                        "source_array": "masks_roi",
                        "source_encoding": DENSE_MASK_ENCODING_V1,
                        "source_run": str(target_run),
                    },
                    progress_callback=_emit_rle_progress,
                )
    else:
        mask_rle_summary = {
            "status": "skipped",
            "reason": f"mask_storage={mask_storage}",
            "encoding": "component_rle_v1",
        }

    if mask_storage in _MASK_STORAGES_REMOVE_DENSE:
        if "masks_roi" in run_group:
            del run_group["masks_roi"]
        mask_bitpacked_summary["dense_cache_removed"] = mask_storage in _MASK_STORAGES_WITH_BITPACKED
        mask_rle_summary["dense_cache_removed"] = mask_storage in _MASK_STORAGES_WITH_RLE

    update_mask_storage_attrs(
        run_group,
        has_dense="masks_roi" in run_group,
        has_bitpacked="mask_bitpacked" in run_group,
        has_rle="mask_rle" in run_group,
    )
    run_group.attrs["smart_finalizer_mask_storage"] = mask_storage
    run_group.attrs["smart_finalizer_mask_rle_validation_mode"] = mask_rle_validation_mode
    run_group.attrs["smart_finalizer_mask_bitpacked_summary"] = dict(_json_safe(mask_bitpacked_summary))
    run_group.attrs["smart_finalizer_mask_rle_summary"] = dict(_json_safe(mask_rle_summary))

    duration_seconds = float(time.perf_counter() - stage_start)
    timing_summary = timing.summary(total_rows=total_rows, duration_seconds=duration_seconds)
    timing_summary.update(execution_metadata)
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["summary_statistics"] = {
        "rows_total": int(total_rows),
        "rows_with_nonempty_masks": int(rows_with_nonempty_masks),
        "duration_seconds": float(duration_seconds),
        "rows_per_second": float(timing_summary["rows_per_second"]),
    }
    run_group.attrs["smart_finalizer_timing_summary"] = dict(_json_safe(timing_summary))
    run_group.attrs["smart_finalizer_chunk_timings"] = list(_json_safe(timing.chunk_timings))
    run_group.attrs["smart_finalizer_review_counts"] = review_counts
    run_group.attrs["smart_finalizer_chunk_size"] = int(max(1, int(chunk_size)))
    run_group.attrs["smart_finalizer_worker_chunk_size"] = int(worker_chunk_size)
    run_group.attrs["smart_finalizer_dense_mask_row_chunk"] = int(effective_dense_mask_row_chunk)
    run_group.attrs["smart_finalizer_dense_mask_storage_chunks"] = [
        int(value) for value in dense_mask_storage_chunks
    ]
    run_group.attrs["smart_finalizer_finalization_metric_row_chunk"] = int(
        finalization_metric_row_chunk
    )
    run_group.attrs["smart_finalizer_finalization_metric_write_policy"] = (
        finalization_metric_write_policy
    )
    run_group.attrs["smart_finalizer_common_metric_row_chunk"] = int(common_metric_row_chunk)
    run_group.attrs["smart_finalizer_common_metric_component_chunk"] = int(len(component_names))
    run_group.attrs["smart_finalizer_common_metric_write_policy"] = common_metric_write_policy
    run_group.attrs["smart_finalizer_chunk_count"] = int(len(chunk_ranges))
    run_group.attrs["smart_finalizer_metric_level"] = metric_level
    run_group.attrs["smart_finalizer_write_eye_geometry"] = bool(write_eye_geometry)
    run_group.attrs["smart_finalizer_write_component_contours"] = bool(write_component_contours)
    run_group.attrs["smart_finalizer_write_sampled_component_contours"] = bool(
        write_sampled_component_contours
    )
    run_group.attrs["smart_finalizer_sampled_contour_counts"] = dict(
        resolved_sampled_contour_counts
    )
    run_group.attrs["smart_finalizer_sampled_contour_row_chunk"] = int(
        normalized_sampled_contour_row_chunk
    )
    run_group.attrs["smart_finalizer_retain_source_seeds"] = bool(retain_source_seeds)
    run_group.attrs["source_seed_masks_status"] = "retained" if bool(retain_source_seeds) else "omitted"
    run_group.attrs["source_seed_masks_reason"] = (
        "retain_source_seeds=true" if bool(retain_source_seeds) else "production_default"
    )
    run_group.attrs["smart_finalizer_postcompute_backend"] = postcompute_backend
    run_group.attrs["smart_finalizer_postcompute_chunk_size"] = int(normalized_postcompute_chunk_size)
    run_group.attrs["smart_finalizer_postcompute_num_workers"] = normalized_postcompute_num_workers
    run_group.attrs["smart_finalizer_postcompute_summary"] = dict(_json_safe(postcompute_summary))
    run_group.attrs["smart_finalizer_mask_storage"] = mask_storage
    run_group.attrs["smart_finalizer_mask_rle_validation_mode"] = mask_rle_validation_mode
    run_group.attrs["smart_finalizer_mask_bitpacked_summary"] = dict(_json_safe(mask_bitpacked_summary))
    run_group.attrs["smart_finalizer_mask_rle_summary"] = dict(_json_safe(mask_rle_summary))
    run_group.attrs["smart_finalizer_execution_backend"] = execution_backend
    run_group.attrs["process_shard_execution_enabled"] = execution_backend == _PROCESS_SHARD_EXECUTION_BACKEND
    run_group.attrs["worker_process_count"] = execution_metadata["worker_process_count"]
    run_group.attrs["requested_chunk_size"] = int(requested_chunk_size)
    run_group.attrs["worker_chunk_size"] = int(worker_chunk_size)
    run_group.attrs["chunk_alignment"] = str(execution_metadata["chunk_alignment"])
    mark_run_complete(
        run_group,
        parent_group=refined_parent,
        run_name=target_run,
        run_provenance=build_run_provenance_from_stage_record(
            run_group.attrs.get("provenance", {}),
            fallback_command="finalize_subject_masks",
        ),
    )
    refined_parent.attrs["refined_subject_mask_review_status_latest"] = target_run

    summary.update(
        {
            "status": "updated",
            **execution_metadata,
            "refined_run_exists": False,
            "would_create_refined_run": True,
            "duration_seconds": duration_seconds,
            "timing_summary": dict(_json_safe(timing_summary)),
            "metric_level": metric_level,
            "write_eye_geometry": bool(write_eye_geometry),
            "write_component_contours": bool(write_component_contours),
            "write_sampled_component_contours": bool(write_sampled_component_contours),
            "sampled_contour_counts": dict(resolved_sampled_contour_counts),
            "sampled_contour_row_chunk": int(normalized_sampled_contour_row_chunk),
            "retain_source_seeds": bool(retain_source_seeds),
            "source_seed_masks_status": "retained" if bool(retain_source_seeds) else "omitted",
            "mask_storage": mask_storage,
            "mask_rle_validation_mode": mask_rle_validation_mode,
            "mask_bitpacked_summary": dict(_json_safe(mask_bitpacked_summary)),
            "mask_rle_summary": dict(_json_safe(mask_rle_summary)),
            "postcompute_backend": postcompute_backend,
            "postcompute_chunk_size": int(normalized_postcompute_chunk_size),
            "postcompute_num_workers": normalized_postcompute_num_workers,
            "postcompute_summary": dict(_json_safe(postcompute_summary)),
            "eye_geometry_reuse_status": str(eye_geometry_reuse_status),
            "eye_geometry_assignment_geometry_rows": int(assignment_eye_geometry_rows),
            "eye_geometry_assignment_geometry_expected_rows": int(total_rows if eye_geometry_requested else 0),
            "eye_geometry_assignment_geometry_shards": int(len(assignment_eye_geometry_shards)),
            "eye_geometry_assignment_geometry_expected_shards": int(
                len(chunk_ranges) if eye_geometry_requested else 0
            ),
            "warnings": list(_json_safe(finalizer_warnings)),
            "component_contours": contour_summaries,
            "sampled_component_contours": sampled_contour_summaries,
            "review_counts": review_counts,
            "eyes_union_assignment_summary": (
                dict(_json_safe(eyes_union_assignment_summary))
                if eyes_union_assignment_summary
                else None
            ),
        }
    )
    progress.emit(
        "complete",
        duration_seconds=float(duration_seconds),
        rows_total=int(total_rows),
        rows_per_second=float(timing_summary["rows_per_second"]),
        refined_run=str(target_run),
        mask_storage=str(mask_storage),
    )
    return summary


def finalize_subject_masks(
    zarr_path: str | Path,
    *,
    subject_run: Optional[str] = None,
    subject_shard_runs: Sequence[str] | None = None,
    target_crop_run: str | None = None,
    refined_run: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
    chunk_size: int = 256,
    metric_level: str = "cheap",
    write_eye_geometry: bool = False,
    write_component_contours: bool = False,
    write_sampled_component_contours: bool = False,
    sampled_contour_counts: Mapping[str, int] | None = None,
    sampled_contour_row_chunk: int = DEFAULT_SAMPLED_CONTOUR_ROW_CHUNK,
    retain_source_seeds: bool = False,
    mask_storage: str = "dense_uint8",
    mask_rle_validation_mode: str = "full",
    dense_mask_row_chunk: int | None = None,
    postcompute_backend: str = _SERIAL_POSTCOMPUTE_BACKEND,
    postcompute_chunk_size: Optional[int] = None,
    postcompute_num_workers: Optional[int] = None,
    execution_backend: str = _SERIAL_EXECUTION_BACKEND,
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    assignment_keypoint_group: Optional[str] = None,
    assignment_keypoints_run: Optional[str] = None,
    registry: str | Path | None = None,
    defer_registry_status: bool = False,
    progress_jsonl: str | Path | None = None,
    console: Any = None,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r" if dry_run else "a")
    summary = finalize_subject_mask_run(
        root,
        zarr_path=zarr_path,
        subject_run=subject_run,
        subject_shard_runs=subject_shard_runs,
        target_crop_run=target_crop_run,
        refined_run=refined_run,
        components=components,
        chunk_size=chunk_size,
        metric_level=metric_level,
        write_eye_geometry=write_eye_geometry,
        write_component_contours=write_component_contours,
        write_sampled_component_contours=write_sampled_component_contours,
        sampled_contour_counts=sampled_contour_counts,
        sampled_contour_row_chunk=sampled_contour_row_chunk,
        retain_source_seeds=retain_source_seeds,
        mask_storage=mask_storage,
        mask_rle_validation_mode=mask_rle_validation_mode,
        dense_mask_row_chunk=dense_mask_row_chunk,
        postcompute_backend=postcompute_backend,
        postcompute_chunk_size=postcompute_chunk_size,
        postcompute_num_workers=postcompute_num_workers,
        execution_backend=execution_backend,
        num_workers=num_workers,
        overwrite=overwrite,
        dry_run=dry_run,
        assignment_keypoint_group=assignment_keypoint_group,
        assignment_keypoints_run=assignment_keypoints_run,
        progress_jsonl=progress_jsonl,
    )
    if not dry_run and not defer_registry_status:
        run_name = str(summary["refined_run"])
        refined_group = root["refined_subject_masks_runs"][run_name]
        emit_refined_subject_mask_stage_completion(
            root,
            zarr_path,
            run_group=refined_group,
            run_name=run_name,
            source=_REFINED_SUBJECT_MASKS_STATUS_SOURCE,
            registry=registry,
            console=console,
            invalidate_on_ok=True,
        )
    return summary


def _parse_components(values: Optional[Sequence[Sequence[str]]], single_values: Optional[Sequence[str]]) -> Optional[list[str]]:
    merged: list[str] = []
    for group in values or ():
        merged.extend(str(value) for value in group)
    for value in single_values or ():
        merged.append(str(value))
    return merged or None


def _parse_sampled_contour_counts(values: Sequence[str] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values or ():
        if "=" not in str(value):
            raise ValueError(f"Expected COMPONENT=K for --sampled-contour-k; got {value!r}.")
        component, raw_count = str(value).split("=", 1)
        component = component.strip()
        count = int(raw_count)
        if not component or count <= 0:
            raise ValueError(f"Expected a component name and positive K; got {value!r}.")
        counts[component] = count
    return counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", help="Path to the Palette zarr archive.")
    parser.add_argument(
        "--subject-run",
        "--source-run",
        dest="subject_run",
        help="Source subject_mask_runs/<run> to finalize. Defaults to latest subject-mask run.",
    )
    parser.add_argument(
        "--subject-shard-run",
        action="append",
        default=[],
        help=(
            "Source subject_mask_shard_runs/<run> shard to include in a collection finalizer. "
            "Repeat for multiple clip/video shards. Mutually exclusive with --subject-run."
        ),
    )
    parser.add_argument(
        "--subject-shard-runs-file",
        type=Path,
        help=(
            "JSON file containing shard run names, either as a list or as an object with "
            "subject_shard_runs/shard_runs/runs. Mutually exclusive with --subject-run."
        ),
    )
    parser.add_argument(
        "--target-crop-run",
        help=(
            "Merged/proxy crop_runs/<run> used to rebase per-shard source_crop_row_ids into "
            "collection row order when finalizing subject_mask_shard_runs."
        ),
    )
    parser.add_argument(
        "--refined-run",
        "--run-name",
        dest="refined_run",
        help="Target refined_subject_masks_runs/<run>. Defaults to a timestamped refined run.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        action="append",
        help="Optional output components to create. Use eyes_union to request eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--component",
        action="append",
        dest="component_values",
        help="Optional single output component selector. Repeat to add more components.",
    )
    parser.add_argument("--chunk-size", type=int, default=256, help="Number of ROI rows to finalize per write chunk.")
    parser.add_argument(
        "--metric-level",
        choices=_METRIC_LEVELS,
        default="cheap",
        help="Metric depth to compute during finalization. cheap writes topology only; full also writes slower shape QC metrics.",
    )
    parser.add_argument(
        "--write-eye-geometry",
        action="store_true",
        help="Also compute refined eye geometry/ellipse relations during finalization.",
    )
    parser.add_argument(
        "--write-component-contours",
        action="store_true",
        help="Also compute full ragged component contours; intended for explicit analysis/archive builds.",
    )
    parser.add_argument(
        "--write-sampled-component-contours",
        action="store_true",
        help="Write fixed-K sampled display contours directly from finalized dense masks.",
    )
    parser.add_argument(
        "--sampled-contour-k",
        action="append",
        default=[],
        metavar="COMPONENT=K",
        help="Override a fixed-K sampled contour count. Repeat per component.",
    )
    parser.add_argument(
        "--sampled-contour-row-chunk",
        type=int,
        default=DEFAULT_SAMPLED_CONTOUR_ROW_CHUNK,
        help="Physical row chunk for fixed-K sampled contour arrays (default: 1024).",
    )
    parser.add_argument(
        "--retain-source-seeds",
        action="store_true",
        help=(
            "Retain per-component source_seed_masks_roi debug arrays. Default production runs omit "
            "these dense intermediate masks and keep masks_roi plus finalization metrics/QC instead."
        ),
    )
    parser.add_argument(
        "--mask-storage",
        choices=_MASK_STORAGE_CHOICES,
        default="dense_uint8",
        help=(
            "Physical mask storage to materialize. dense_uint8 preserves the historical masks_roi-only "
            "surface; dense_and_bitpacked additionally writes derived mask_bitpacked for compact "
            "publication/display; compact-only bitpacked_v1 is rejected for editable analysis; "
            "dense_and_rle additionally writes mask_rle/components for compact final products and "
            "selective reads; compact-only rle_v1 is rejected for editable analysis; "
            "dense_bitpacked_and_rle writes all three physical surfaces for validation/audit runs."
        ),
    )
    parser.add_argument(
        "--mask-rle-validation-mode",
        choices=_MASK_RLE_VALIDATION_MODES,
        default="full",
        help=(
            "Validation policy after writing component RLE. full decodes and compares every dense mask "
            "pixel against the source; invariants checks schema, row counts, RLE count sums, presence, "
            "area, and bbox consistency without reconstructing the dense logical surface; none skips "
            "compact-store validation."
        ),
    )
    parser.add_argument(
        "--dense-mask-row-chunk",
        type=int,
        help=(
            "Rows per physical Zarr chunk for dense refined masks_roi and retained "
            "source_seed_masks_roi arrays. Defaults to the current contract policy."
        ),
    )
    parser.add_argument(
        "--postcompute-backend",
        choices=_POSTCOMPUTE_BACKENDS,
        default=_SERIAL_POSTCOMPUTE_BACKEND,
        help=(
            "Backend for expensive derived artifacts requested by --write-eye-geometry or "
            "--write-component-contours. serial preserves the historical in-process path; "
            "process_shards computes row shards in worker processes and merges packed outputs."
        ),
    )
    parser.add_argument(
        "--postcompute-chunk-size",
        type=int,
        help=(
            "Rows per postcompute shard. Defaults to --chunk-size. Used only with "
            "--postcompute-backend=process_shards."
        ),
    )
    parser.add_argument(
        "--postcompute-num-workers",
        type=int,
        help=(
            "Worker process count for --postcompute-backend=process_shards. Defaults to "
            "--num-workers when set, otherwise one worker."
        ),
    )
    parser.add_argument(
        "--execution-backend",
        choices=_EXECUTION_BACKENDS,
        default=_SERIAL_EXECUTION_BACKEND,
        help=(
            "Execution backend. process_shards uses one local worker process per contiguous row shard; "
            "serial_driver is the deterministic single-process fallback."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Worker process count for process_shards.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target refined run.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the plan without mutating the archive.")
    parser.add_argument(
        "--assignment-keypoint-group",
        choices=("refined_keypoints_runs", "keypoints_runs"),
        help="Explicit keypoint group to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--assignment-keypoints-run",
        help="Explicit keypoint run to use for eyes_union -> eye_left/eye_right assignment.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Registry database to update when emitting refined_subject_masks completion status.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    parser.add_argument(
        "--progress-jsonl",
        help="Append live finalization progress events as JSONL. Useful for tailing long cluster jobs.",
    )
    parser.add_argument(
        "--defer-registry-status",
        action="store_true",
        help=(
            "Write the refined zarr run group without emitting registry step "
            "status. Use for local staged output that will be published to the "
            "canonical archive before status is emitted."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    components = _parse_components(args.components, args.component_values)
    try:
        sampled_contour_counts = _parse_sampled_contour_counts(args.sampled_contour_k)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    subject_shard_runs = list(args.subject_shard_run or [])
    if args.subject_shard_runs_file is not None:
        subject_shard_runs.extend(_load_shard_names_from_file(args.subject_shard_runs_file))
    summary = finalize_subject_masks(
        args.zarr_path,
        subject_run=args.subject_run,
        subject_shard_runs=subject_shard_runs or None,
        target_crop_run=args.target_crop_run,
        refined_run=args.refined_run,
        components=components,
        chunk_size=int(args.chunk_size),
        metric_level=args.metric_level,
        write_eye_geometry=bool(args.write_eye_geometry),
        write_component_contours=bool(args.write_component_contours),
        write_sampled_component_contours=bool(args.write_sampled_component_contours),
        sampled_contour_counts=sampled_contour_counts,
        sampled_contour_row_chunk=int(args.sampled_contour_row_chunk),
        retain_source_seeds=bool(args.retain_source_seeds),
        mask_storage=args.mask_storage,
        mask_rle_validation_mode=args.mask_rle_validation_mode,
        dense_mask_row_chunk=args.dense_mask_row_chunk,
        postcompute_backend=args.postcompute_backend,
        postcompute_chunk_size=args.postcompute_chunk_size,
        postcompute_num_workers=args.postcompute_num_workers,
        execution_backend=args.execution_backend,
        num_workers=args.num_workers,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        assignment_keypoint_group=args.assignment_keypoint_group,
        assignment_keypoints_run=args.assignment_keypoints_run,
        registry=args.registry,
        defer_registry_status=bool(args.defer_registry_status),
        progress_jsonl=args.progress_jsonl,
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

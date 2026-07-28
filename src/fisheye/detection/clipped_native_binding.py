"""Bind clipped detection artifacts to canonical recording-level arrays.

This module is deliberately independent of Zarr I/O and publication.  It
consumes the exact selector-ineligible artifact arrays emitted by
``detect_yolo`` and a proven clip-local-to-recording frame mapping, then emits
the frozen canonical-detection v1 logical array set.  Storage planning,
manifest construction, atomic publication, selectors, and registry mutation
remain separate boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ALGORITHM,
    mint_detection_instance_keys,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)


CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_ID = (
    "palette.clipped_detection.native_binding"
)
CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_VERSION = 1
ARTIFACT_ROW_ID_CONTRACT = "palette.detection_artifact_row_id.v1"
ARTIFACT_COORDINATE_CONTRACT = "unbound_detection_artifact_v1"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _require_safe_id(value: str, *, name: str) -> str:
    normalized = _require_text(value, name=name)
    if not _SAFE_ID_RE.fullmatch(normalized):
        raise ValueError(f"{name} is not a safe identifier: {normalized!r}.")
    return normalized


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return normalized


def _require_exact_array(
    value: Any,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ValueError(
            f"{name} dtype must be {dtype}, observed {array.dtype}."
        )
    if array.shape != shape:
        raise ValueError(
            f"{name} shape must be {shape}, observed {array.shape}."
        )
    return np.ascontiguousarray(array)


def _sha256_int64(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


@dataclass(frozen=True)
class ClippedDetectionArtifactMember:
    """One validated logical work-unit input to recording-level binding."""

    work_unit_id: str
    artifact_run_id: str
    clip_id: str
    clip_index: int
    camera_serial: str
    source_width: int
    source_height: int
    artifact_manifest_sha256: str
    run_group_tree_sha256: str
    parent_frame_indices: np.ndarray
    frame_indices: np.ndarray
    bbox_norm_coords: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    artifact_row_id: np.ndarray
    frame_counts: np.ndarray
    n_detections: np.ndarray

    def __post_init__(self) -> None:
        _require_safe_id(self.work_unit_id, name="work_unit_id")
        _require_safe_id(self.artifact_run_id, name="artifact_run_id")
        _require_safe_id(self.clip_id, name="clip_id")
        _require_safe_id(self.camera_serial, name="camera_serial")
        if type(self.clip_index) is not int or self.clip_index < 0:
            raise ValueError("clip_index must be a nonnegative exact integer.")
        for name in ("source_width", "source_height"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer.")
        _require_sha256(
            self.artifact_manifest_sha256,
            name="artifact_manifest_sha256",
        )
        _require_sha256(
            self.run_group_tree_sha256,
            name="run_group_tree_sha256",
        )


@dataclass(frozen=True)
class BoundClippedCanonicalDetection:
    """Canonical-v1 logical arrays plus immutable clipped binding evidence."""

    dimensions: CanonicalDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    binding_evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        if tuple(self.arrays) != CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
            raise ValueError(
                "Bound arrays must follow the exact canonical detection binding order."
            )
        copied: dict[str, np.ndarray] = {}
        for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
            values = np.array(self.arrays[path], copy=True, order="C")
            values.setflags(write=False)
            copied[path] = values
        CANONICAL_DETECTION_SCHEMA_V1.require(
            copied,
            dimensions=self.dimensions,
        )
        canonical_json_bytes(self.binding_evidence)
        object.__setattr__(self, "arrays", MappingProxyType(copied))
        object.__setattr__(
            self,
            "binding_evidence",
            MappingProxyType(dict(self.binding_evidence)),
        )


@dataclass(frozen=True)
class _ValidatedMember:
    member: ClippedDetectionArtifactMember
    parent_frame_indices: np.ndarray
    frame_indices: np.ndarray
    bbox_norm_coords: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    artifact_row_id: np.ndarray
    frame_counts: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.parent_frame_indices.shape[0])

    @property
    def parent_start(self) -> int:
        return int(self.parent_frame_indices[0])

    @property
    def parent_stop(self) -> int:
        return int(self.parent_frame_indices[-1]) + 1


def _validate_member(
    member: ClippedDetectionArtifactMember,
) -> _ValidatedMember:
    frame_counts = np.asarray(member.frame_counts)
    if frame_counts.ndim != 1:
        raise ValueError(
            f"{member.work_unit_id}: frame_counts must be rank one."
        )
    frame_count = int(frame_counts.shape[0])
    if frame_count <= 0:
        raise ValueError(
            f"{member.work_unit_id}: zero-frame artifacts cannot be bound."
        )
    parent_frames = _require_exact_array(
        member.parent_frame_indices,
        name=f"{member.work_unit_id}.parent_frame_indices",
        dtype=np.dtype(np.int64),
        shape=(frame_count,),
    )
    if np.any(parent_frames < 0) or np.any(np.diff(parent_frames) != 1):
        raise ValueError(
            f"{member.work_unit_id}: parent frames must be one nonnegative "
            "contiguous increasing interval."
        )

    frame_counts = _require_exact_array(
        member.frame_counts,
        name=f"{member.work_unit_id}.frame_counts",
        dtype=np.dtype(np.int32),
        shape=(frame_count,),
    )
    n_detections = _require_exact_array(
        member.n_detections,
        name=f"{member.work_unit_id}.n_detections",
        dtype=np.dtype(np.int32),
        shape=(frame_count,),
    )
    if not np.array_equal(frame_counts, n_detections):
        raise ValueError(
            f"{member.work_unit_id}: frame_counts and n_detections disagree."
        )
    if np.any(frame_counts < 0):
        raise ValueError(f"{member.work_unit_id}: frame counts cannot be negative.")
    row_count = int(np.sum(frame_counts, dtype=np.int64))
    frames = _require_exact_array(
        member.frame_indices,
        name=f"{member.work_unit_id}.frame_indices",
        dtype=np.dtype(np.int32),
        shape=(row_count,),
    )
    bbox = _require_exact_array(
        member.bbox_norm_coords,
        name=f"{member.work_unit_id}.bbox_norm_coords",
        dtype=np.dtype(np.float64),
        shape=(row_count, 4),
    )
    scores = _require_exact_array(
        member.scores,
        name=f"{member.work_unit_id}.scores",
        dtype=np.dtype(np.float32),
        shape=(row_count,),
    )
    classes = _require_exact_array(
        member.class_ids,
        name=f"{member.work_unit_id}.class_ids",
        dtype=np.dtype(np.int32),
        shape=(row_count,),
    )
    artifact_rows = _require_exact_array(
        member.artifact_row_id,
        name=f"{member.work_unit_id}.artifact_row_id",
        dtype=np.dtype(np.uint64),
        shape=(row_count,),
    )
    if not np.array_equal(
        artifact_rows,
        np.arange(row_count, dtype=np.uint64),
    ):
        raise ValueError(
            f"{member.work_unit_id}: artifact_row_id must be the dense run-local range."
        )
    if frames.size:
        if np.any(frames < 0) or np.any(frames >= frame_count):
            raise ValueError(
                f"{member.work_unit_id}: detection frame lies outside the clip domain."
            )
        if frames.size > 1 and np.any(np.diff(frames) < 0):
            raise ValueError(
                f"{member.work_unit_id}: detection rows are not frame sorted."
            )
    expected_counts = np.bincount(
        frames.astype(np.int64, copy=False),
        minlength=frame_count,
    ).astype(np.int32, copy=False)
    if not np.array_equal(frame_counts, expected_counts):
        raise ValueError(
            f"{member.work_unit_id}: counts do not describe frame_indices."
        )
    return _ValidatedMember(
        member=member,
        parent_frame_indices=parent_frames,
        frame_indices=frames,
        bbox_norm_coords=bbox,
        scores=scores,
        class_ids=classes,
        artifact_row_id=artifact_rows,
        frame_counts=frame_counts,
    )


def _require_unique(values: Sequence[object], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Clipped detection members contain duplicate {name} values.")


def bind_clipped_detection_artifacts(
    members: Sequence[ClippedDetectionArtifactMember],
    *,
    recording_identity: str,
    n_frames: int,
    source_width: int,
    source_height: int,
) -> BoundClippedCanonicalDetection:
    """Return exact canonical-v1 arrays for one complete clipped recording."""

    recording = _require_text(recording_identity, name="recording_identity")
    for name, value in (
        ("n_frames", n_frames),
        ("source_width", source_width),
        ("source_height", source_height),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact integer.")
    if n_frames > int(np.iinfo(np.int32).max):
        raise ValueError("n_frames exceeds the canonical int32 frame domain.")
    if not members:
        raise ValueError("At least one clipped detection artifact is required.")

    _require_unique([item.work_unit_id for item in members], name="work_unit_id")
    _require_unique([item.artifact_run_id for item in members], name="artifact_run_id")
    _require_unique([item.clip_id for item in members], name="clip_id")
    _require_unique([item.clip_index for item in members], name="clip_index")
    camera_serials = {str(item.camera_serial) for item in members}
    if len(camera_serials) != 1:
        raise ValueError(
            "Canonical clipped detection v1 currently requires exactly one camera."
        )
    for item in members:
        if item.source_width != source_width or item.source_height != source_height:
            raise ValueError(
                f"{item.work_unit_id}: source dimensions disagree with the recording."
            )

    validated = [_validate_member(item) for item in members]
    ordered = sorted(
        validated,
        key=lambda item: (
            item.parent_start,
            item.member.clip_index,
            item.member.work_unit_id,
        ),
    )
    combined_frame_map = np.concatenate(
        [item.parent_frame_indices for item in ordered]
    )
    expected_frame_map = np.arange(n_frames, dtype=np.int64)
    if not np.array_equal(combined_frame_map, expected_frame_map):
        raise ValueError(
            "Clipped parent-frame mappings must cover the canonical recording "
            "exactly once in increasing order."
        )

    canonical_frames_by_member = [
        item.parent_frame_indices[item.frame_indices.astype(np.int64, copy=False)]
        for item in ordered
    ]
    canonical_frames_i64 = np.concatenate(canonical_frames_by_member).astype(
        np.int64,
        copy=False,
    )
    if canonical_frames_i64.size > 1 and np.any(np.diff(canonical_frames_i64) < 0):
        raise RuntimeError("Internal error: bound detection rows are not frame sorted.")
    frame_indices = canonical_frames_i64.astype(np.int32, copy=False)
    bbox_norm = np.concatenate(
        [item.bbox_norm_coords for item in ordered],
        axis=0,
    ).astype(np.float32, copy=False)
    scores = np.concatenate([item.scores for item in ordered]).astype(
        np.float32,
        copy=False,
    )
    class_ids = np.concatenate([item.class_ids for item in ordered]).astype(
        np.int32,
        copy=False,
    )
    instance_keys = mint_detection_instance_keys(
        recording_identity=recording,
        frame_indices=frame_indices,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    bbox_img, centers_img = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=source_width,
        source_height=source_height,
    )
    frame_counts = np.bincount(
        canonical_frames_i64,
        minlength=n_frames,
    )
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    dimensions = CanonicalDetectionDimensions(
        n_frames=n_frames,
        n_instances=int(frame_indices.shape[0]),
        source_width=source_width,
        source_height=source_height,
    )
    arrays: dict[str, np.ndarray] = {
        "instances/frame_indices": np.ascontiguousarray(frame_indices),
        "instances/source_acquisition_frame_index": np.ascontiguousarray(
            canonical_frames_i64
        ),
        "instances/instance_key": np.ascontiguousarray(instance_keys),
        "instances/bbox_norm_coords": np.ascontiguousarray(bbox_norm),
        "instances/bbox_img_xyxy": np.ascontiguousarray(bbox_img),
        "instances/centers_img_xy": np.ascontiguousarray(centers_img),
        "instances/scores": np.ascontiguousarray(scores),
        "instances/class_ids": np.ascontiguousarray(class_ids),
        "instances/frame_row_offsets": offsets,
    }
    CANONICAL_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)

    row_cursor = 0
    member_evidence: list[dict[str, object]] = []
    for item in ordered:
        row_stop = row_cursor + item.row_count
        member_evidence.append(
            {
                "work_unit_id": item.member.work_unit_id,
                "artifact_run_id": item.member.artifact_run_id,
                "clip_id": item.member.clip_id,
                "clip_index": item.member.clip_index,
                "camera_serial": item.member.camera_serial,
                "artifact_manifest_sha256": _require_sha256(
                    item.member.artifact_manifest_sha256,
                    name="artifact_manifest_sha256",
                ),
                "run_group_tree_sha256": _require_sha256(
                    item.member.run_group_tree_sha256,
                    name="run_group_tree_sha256",
                ),
                "parent_frame_start": item.parent_start,
                "parent_frame_stop": item.parent_stop,
                "parent_frame_mapping_sha256": _sha256_int64(
                    item.parent_frame_indices
                ),
                "artifact_frame_count": item.frame_count,
                "artifact_detection_rows": item.row_count,
                "canonical_row_start": row_cursor,
                "canonical_row_stop": row_stop,
            }
        )
        row_cursor = row_stop

    payload: dict[str, object] = {
        "schema_id": CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_ID,
        "schema_version": CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_VERSION,
        "recording_identity": recording,
        "camera_serial": next(iter(camera_serials)),
        "dimensions": dimensions.as_manifest(),
        "artifact_contract": {
            "coordinate_contract": ARTIFACT_COORDINATE_CONTRACT,
            "row_identity": ARTIFACT_ROW_ID_CONTRACT,
            "selector_eligible": False,
        },
        "canonical_contract": {
            "schema_id": CANONICAL_DETECTION_SCHEMA_V1.schema_id,
            "schema_version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
            "instance_key_algorithm": INSTANCE_KEY_ALGORITHM,
            "row_order": "canonical_frame_then_persisted_artifact_order",
        },
        "frame_mapping": {
            "domain": "zero_based_recording_parent_frame_index",
            "coverage": "exactly_once_complete_recording",
            "sha256_int64_c_order": _sha256_int64(combined_frame_map),
        },
        "members": member_evidence,
        "canonical_arrays": {
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "sha256": {
                path: sha256_array(values) for path, values in arrays.items()
            },
        },
    }
    evidence: dict[str, object] = {
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(payload),
        "document": payload,
    }
    canonical_json_bytes(evidence)
    return BoundClippedCanonicalDetection(
        dimensions=dimensions,
        arrays=arrays,
        binding_evidence=evidence,
    )


__all__ = [
    "ARTIFACT_COORDINATE_CONTRACT",
    "ARTIFACT_ROW_ID_CONTRACT",
    "CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_ID",
    "CLIPPED_NATIVE_DETECTION_BINDING_SCHEMA_VERSION",
    "BoundClippedCanonicalDetection",
    "ClippedDetectionArtifactMember",
    "bind_clipped_detection_artifacts",
]

"""Detection-specific sparse delta v2 and deterministic in-memory resolution.

The module performs no Zarr writes.  It freezes the exact event-column schema,
validates immutable partition payloads, and resolves an immutable refined-v1
base plus ordered delta batches into another complete refined-v1 logical
snapshot.  Persistence, compaction, and selector changes are separate steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.detection_schema import (
    MAX_CANONICAL_CLASS_ID,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)


REFINED_DETECTION_DELTA_SCHEMA_ID = "palette.refined_detection.delta"
REFINED_DETECTION_DELTA_SCHEMA_VERSION = 2
REFINED_DETECTION_DELTA_LAYOUT = (
    "immutable_partitioned_events_with_full_add_replace_payload_v2"
)

REFINED_DETECTION_DELTA_OPERATION_CODE_MAP = {
    "add_instance": 1,
    "replace_instance": 2,
    "delete_instance": 3,
    "restore_instance": 4,
}
REFINED_DETECTION_DELTA_MERGE_ORDER = ("event_sequence",)

_PAYLOAD_OPERATIONS = frozenset(
    {
        REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["add_instance"],
        REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["replace_instance"],
    }
)
_NO_PAYLOAD_OPERATIONS = frozenset(
    {
        REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["delete_instance"],
        REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["restore_instance"],
    }
)
_REASON_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RefinedDetectionDeltaArray:
    """One exact persisted event-column declaration."""

    name: str
    dtype: str
    trailing_shape: tuple[int, ...]
    semantics: str

    def as_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": ["n_events", *self.trailing_shape],
            "semantics": self.semantics,
        }


REFINED_DETECTION_DELTA_ARRAYS = (
    RefinedDetectionDeltaArray(
        "event_sequence",
        "uint64",
        (),
        "positive globally unique total order within one delta lineage",
    ),
    RefinedDetectionDeltaArray(
        "expected_previous_event_sequence",
        "uint64",
        (),
        "optimistic-concurrency predecessor for the target instance; zero means base",
    ),
    RefinedDetectionDeltaArray(
        "operation_codes",
        "uint8",
        (),
        "operation from the exact v2 operation registry",
    ),
    RefinedDetectionDeltaArray(
        "instance_key",
        "uint64",
        (),
        "durable observation identity",
    ),
    RefinedDetectionDeltaArray(
        "refined_row_ids",
        "int64",
        (),
        "stable non-reused refined-lineage row identity",
    ),
    RefinedDetectionDeltaArray(
        "row_index_hint",
        "int64",
        (),
        "base physical-row hint or exact -1",
    ),
    RefinedDetectionDeltaArray(
        "timestamp_ns",
        "int64",
        (),
        "nonnegative provenance timestamp; never part of merge order",
    ),
    RefinedDetectionDeltaArray(
        "reason_codes",
        "uint16",
        (),
        "code in the partition-local canonical reason registry",
    ),
    RefinedDetectionDeltaArray(
        "payload_valid",
        "bool",
        (),
        "true exactly for add_instance and replace_instance",
    ),
    RefinedDetectionDeltaArray(
        "frame_indices",
        "int32",
        (),
        "complete post-operation frame for add/replace; -1 sentinel otherwise",
    ),
    RefinedDetectionDeltaArray(
        "source_acquisition_frame_index",
        "int64",
        (),
        "sealed acquisition frame for add/replace; -1 sentinel otherwise",
    ),
    RefinedDetectionDeltaArray(
        "bbox_norm_coords",
        "float32",
        (4,),
        "complete authoritative cx,cy,w,h for add/replace; zeros otherwise",
    ),
    RefinedDetectionDeltaArray(
        "scores",
        "float32",
        (),
        "complete post-operation score for add/replace; zero otherwise",
    ),
    RefinedDetectionDeltaArray(
        "score_valid",
        "bool",
        (),
        "complete post-operation score validity for add/replace; false otherwise",
    ),
    RefinedDetectionDeltaArray(
        "class_ids",
        "int32",
        (),
        "complete post-operation class for add/replace; -1 sentinel otherwise",
    ),
    RefinedDetectionDeltaArray(
        "source_kind_codes",
        "uint8",
        (),
        "complete post-operation source kind for add/replace; zero otherwise",
    ),
    RefinedDetectionDeltaArray(
        "manual_edit_flags",
        "bool",
        (),
        "true for every add/replace payload; false otherwise",
    ),
    RefinedDetectionDeltaArray(
        "source_detect_row_index",
        "int64",
        (),
        "raw audit-row lineage for add/replace; -1 for manual/no-payload rows",
    ),
)

_ARRAY_BY_NAME = {declaration.name: declaration for declaration in REFINED_DETECTION_DELTA_ARRAYS}
_PAYLOAD_FIELD_NAMES = (
    "frame_indices",
    "source_acquisition_frame_index",
    "bbox_norm_coords",
    "scores",
    "score_valid",
    "class_ids",
    "source_kind_codes",
    "manual_edit_flags",
    "source_detect_row_index",
)


class RefinedDetectionDeltaError(ValueError):
    """Raised when a delta batch or deterministic resolution is invalid."""


def _require_uuid(value: str, *, name: str) -> str:
    text = str(value).strip()
    try:
        parsed = uuid.UUID(text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RefinedDetectionDeltaError(f"{name} must be a canonical UUID.") from exc
    canonical = str(parsed)
    if text != canonical:
        raise RefinedDetectionDeltaError(f"{name} must use canonical lowercase UUID text.")
    return canonical


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not _SHA256_RE.fullmatch(text):
        raise RefinedDetectionDeltaError(
            f"{name} must be a lowercase 64-character SHA-256 digest."
        )
    return text


def _require_component(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text or not _SAFE_COMPONENT_RE.fullmatch(text):
        raise RefinedDetectionDeltaError(
            f"{name} must match {_SAFE_COMPONENT_RE.pattern!r}."
        )
    return text


def _normalize_reason_registry(value: Mapping[int, str]) -> Mapping[int, str]:
    normalized: dict[int, str] = {}
    for raw_code, raw_label in value.items():
        if type(raw_code) is not int or not 0 <= raw_code <= int(np.iinfo(np.uint16).max):
            raise RefinedDetectionDeltaError(
                "Reason registry codes must be exact uint16-domain integers."
            )
        label = str(raw_label).strip()
        if not _REASON_LABEL_RE.fullmatch(label):
            raise RefinedDetectionDeltaError(
                "Reason labels must be lowercase snake-case identifiers."
            )
        normalized[raw_code] = label
    if normalized.get(0) != "none":
        raise RefinedDetectionDeltaError("Reason registry code zero must be 'none'.")
    if len(set(normalized.values())) != len(normalized):
        raise RefinedDetectionDeltaError("Reason registry labels must be unique.")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class RefinedDetectionDeltaBatch:
    """One immutable, single-writer v2 event partition."""

    delta_lineage_id: str
    base_snapshot_id: str
    base_manifest_digest: str
    generation_ordinal: int
    partition_id: str
    actor_id: str
    reason_code_map: Mapping[int, str]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_lineage_id",
            _require_uuid(self.delta_lineage_id, name="delta_lineage_id"),
        )
        object.__setattr__(
            self,
            "base_snapshot_id",
            _require_uuid(self.base_snapshot_id, name="base_snapshot_id"),
        )
        object.__setattr__(
            self,
            "base_manifest_digest",
            _require_sha256(
                self.base_manifest_digest,
                name="base_manifest_digest",
            ),
        )
        if type(self.generation_ordinal) is not int or self.generation_ordinal < 0:
            raise RefinedDetectionDeltaError(
                "generation_ordinal must be a nonnegative exact integer."
            )
        object.__setattr__(
            self,
            "partition_id",
            _require_component(self.partition_id, name="partition_id"),
        )
        actor = str(self.actor_id).strip()
        if not actor:
            raise RefinedDetectionDeltaError("actor_id cannot be empty.")
        object.__setattr__(self, "actor_id", actor)
        registry = _normalize_reason_registry(self.reason_code_map)
        object.__setattr__(self, "reason_code_map", registry)

        if set(self.arrays) != set(_ARRAY_BY_NAME):
            missing = sorted(set(_ARRAY_BY_NAME) - set(self.arrays))
            unexpected = sorted(set(self.arrays) - set(_ARRAY_BY_NAME))
            raise RefinedDetectionDeltaError(
                f"Delta arrays must match v2 exactly; missing={missing}, "
                f"unexpected={unexpected}."
            )
        normalized_arrays: dict[str, np.ndarray] = {}
        row_count: int | None = None
        for declaration in REFINED_DETECTION_DELTA_ARRAYS:
            values = np.asarray(self.arrays[declaration.name])
            expected_dtype = np.dtype(declaration.dtype)
            if values.dtype != expected_dtype:
                raise RefinedDetectionDeltaError(
                    f"{declaration.name} dtype must be {expected_dtype}; "
                    f"got {values.dtype}."
                )
            expected_rank = 1 + len(declaration.trailing_shape)
            if values.ndim != expected_rank or values.shape[1:] != declaration.trailing_shape:
                raise RefinedDetectionDeltaError(
                    f"{declaration.name} shape must be "
                    f"(n_events, {declaration.trailing_shape}); got {values.shape}."
                )
            if row_count is None:
                row_count = int(values.shape[0])
            elif int(values.shape[0]) != row_count:
                raise RefinedDetectionDeltaError("Delta event-column lengths differ.")
            copied = np.array(values, copy=True, order="C")
            copied.setflags(write=False)
            normalized_arrays[declaration.name] = copied
        if not row_count:
            raise RefinedDetectionDeltaError("Delta partitions cannot be empty.")
        object.__setattr__(self, "arrays", MappingProxyType(normalized_arrays))
        self._validate_values()

    @property
    def row_count(self) -> int:
        return int(self.arrays["event_sequence"].shape[0])

    def _validate_values(self) -> None:
        arrays = self.arrays
        sequences = arrays["event_sequence"]
        if np.any(sequences == 0) or (
            sequences.size > 1 and np.any(sequences[1:] <= sequences[:-1])
        ):
            raise RefinedDetectionDeltaError(
                "event_sequence must be positive and strictly increasing per partition."
            )
        if np.any(arrays["timestamp_ns"] < 0):
            raise RefinedDetectionDeltaError("timestamp_ns cannot be negative.")
        operations = arrays["operation_codes"]
        unknown = sorted(
            set(int(value) for value in operations.tolist())
            - set(REFINED_DETECTION_DELTA_OPERATION_CODE_MAP.values())
        )
        if unknown:
            raise RefinedDetectionDeltaError(
                f"Unsupported refined-detection operation codes: {unknown}."
            )
        reason_codes = arrays["reason_codes"]
        missing_reason_codes = sorted(
            set(int(value) for value in reason_codes.tolist())
            - set(self.reason_code_map)
        )
        if missing_reason_codes:
            raise RefinedDetectionDeltaError(
                f"Delta reason codes are absent from the registry: {missing_reason_codes}."
            )
        expected_payload = np.isin(operations, tuple(_PAYLOAD_OPERATIONS))
        if not np.array_equal(arrays["payload_valid"], expected_payload):
            raise RefinedDetectionDeltaError(
                "payload_valid must be true exactly for add/replace operations."
            )
        no_payload = np.isin(operations, tuple(_NO_PAYLOAD_OPERATIONS))
        sentinel_checks = {
            "frame_indices": arrays["frame_indices"] == np.int32(-1),
            "source_acquisition_frame_index": (
                arrays["source_acquisition_frame_index"] == np.int64(-1)
            ),
            "scores": arrays["scores"] == np.float32(0.0),
            "score_valid": ~arrays["score_valid"],
            "class_ids": arrays["class_ids"] == np.int32(-1),
            "source_kind_codes": arrays["source_kind_codes"] == np.uint8(0),
            "manual_edit_flags": ~arrays["manual_edit_flags"],
            "source_detect_row_index": (
                arrays["source_detect_row_index"] == np.int64(-1)
            ),
        }
        for name, valid in sentinel_checks.items():
            if np.any(no_payload & ~valid):
                raise RefinedDetectionDeltaError(
                    f"No-payload operations require the exact {name} sentinel."
                )
        if np.any(no_payload & np.any(arrays["bbox_norm_coords"] != 0, axis=1)):
            raise RefinedDetectionDeltaError(
                "No-payload operations require an all-zero bbox sentinel."
            )
        payload = ~no_payload
        if np.any(payload & (arrays["frame_indices"] < 0)):
            raise RefinedDetectionDeltaError("Payload frame_indices must be nonnegative.")
        if np.any(
            payload
            & (
                arrays["source_acquisition_frame_index"]
                != arrays["frame_indices"].astype(np.int64)
            )
        ):
            raise RefinedDetectionDeltaError(
                "Payload acquisition frame identity must equal frame_indices."
            )
        if np.any(payload & ~arrays["manual_edit_flags"]):
            raise RefinedDetectionDeltaError(
                "Every add/replace payload must set manual_edit_flags=true."
            )
        if np.any(payload & ~np.isfinite(arrays["scores"])) or np.any(
            payload & ((arrays["scores"] < 0) | (arrays["scores"] > 1))
        ):
            raise RefinedDetectionDeltaError("Payload scores must be finite in [0,1].")
        if np.any(payload & ~arrays["score_valid"] & (arrays["scores"] != 0)):
            raise RefinedDetectionDeltaError(
                "Invalid payload scores must use exact physical zero."
            )
        if np.any(
            payload
            & (
                (arrays["class_ids"] < 0)
                | (arrays["class_ids"] > MAX_CANONICAL_CLASS_ID)
            )
        ):
            raise RefinedDetectionDeltaError("Payload class_ids are out of range.")
        kinds = arrays["source_kind_codes"]
        raw_mask = payload & (kinds == SOURCE_KIND_CODE_MAP["raw_detect"])
        manual_mask = payload & (kinds == SOURCE_KIND_CODE_MAP["manual"])
        if np.any(payload & ~(raw_mask | manual_mask)):
            raise RefinedDetectionDeltaError("Payload source kind is unsupported.")
        if np.any(raw_mask & (arrays["source_detect_row_index"] < 0)) or np.any(
            raw_mask & ~arrays["score_valid"]
        ):
            raise RefinedDetectionDeltaError(
                "Raw-backed payloads require source lineage and a valid model score."
            )
        if np.any(manual_mask & (arrays["source_detect_row_index"] != -1)) or np.any(
            manual_mask & arrays["score_valid"]
        ):
            raise RefinedDetectionDeltaError(
                "Manual payloads require source row -1 and score_valid=false."
            )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": REFINED_DETECTION_DELTA_SCHEMA_ID,
            "schema_version": REFINED_DETECTION_DELTA_SCHEMA_VERSION,
            "layout": REFINED_DETECTION_DELTA_LAYOUT,
            "delta_lineage_id": self.delta_lineage_id,
            "base_snapshot_id": self.base_snapshot_id,
            "base_manifest_digest": self.base_manifest_digest,
            "generation_ordinal": self.generation_ordinal,
            "partition_id": self.partition_id,
            "actor_id": self.actor_id,
            "row_count": self.row_count,
            "operation_code_map": dict(
                sorted(REFINED_DETECTION_DELTA_OPERATION_CODE_MAP.items())
            ),
            "reason_code_map": {
                str(code): label for code, label in self.reason_code_map.items()
            },
            "merge_order": list(REFINED_DETECTION_DELTA_MERGE_ORDER),
            "array_declarations": [
                declaration.as_manifest()
                for declaration in REFINED_DETECTION_DELTA_ARRAYS
            ],
            "storage": {
                "write_ownership": "one_immutable_partition_per_writer",
                "chunks": "one_complete_ordinary_chunk_per_array",
                "shards": None,
                "consolidated_metadata": False,
            },
        }


@dataclass(frozen=True)
class RefinedDetectionDeltaResolution:
    """Complete logical successor inputs produced by deterministic overlay."""

    dimensions: RefinedDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    instance_reason_codes: Mapping[int, str]
    source_reason_codes: Mapping[int, str]
    next_refined_row_id: int
    report: Mapping[str, object]


@dataclass
class _RowState:
    instance_key: int
    refined_row_id: int
    base_row_index: int | None
    active: bool
    last_event_sequence: int
    last_operation: str | None
    reason_label: str
    payload: dict[str, Any]


def refined_detection_delta_schema_manifest() -> dict[str, object]:
    """Return the exact writer-independent v2 logical declaration."""

    return {
        "schema_id": REFINED_DETECTION_DELTA_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_DELTA_SCHEMA_VERSION,
        "layout": REFINED_DETECTION_DELTA_LAYOUT,
        "operation_code_map": dict(
            sorted(REFINED_DETECTION_DELTA_OPERATION_CODE_MAP.items())
        ),
        "merge_order": list(REFINED_DETECTION_DELTA_MERGE_ORDER),
        "event_identity": "(delta_lineage_id,event_sequence)",
        "optimistic_concurrency": "expected_previous_event_sequence",
        "restore_boundary": (
            "only_an_uncompacted_tombstone_in_the_same_delta_lineage;_a_"
            "compacted_retirement_requires_a_new_add_identity"
        ),
        "lineage_profiles": [RefinedDetectionLineageProfile.FULL_ACQUISITION.value],
        "array_declarations": [
            declaration.as_manifest() for declaration in REFINED_DETECTION_DELTA_ARRAYS
        ],
        "storage": {
            "write_ownership": "one_immutable_partition_per_writer",
            "chunks": "one_complete_ordinary_chunk_per_array",
            "shards": None,
            "consolidated_metadata": False,
        },
    }


def _decode_reason_labels(
    codes: np.ndarray,
    registry: Mapping[int, str],
    *,
    name: str,
) -> list[str]:
    normalized = _normalize_reason_registry(registry)
    missing = sorted(set(int(code) for code in codes.tolist()) - set(normalized))
    if missing:
        raise RefinedDetectionDeltaError(
            f"{name} contains codes absent from its registry: {missing}."
        )
    return [normalized[int(code)] for code in codes.tolist()]


def _encode_reason_labels(labels: Sequence[str]) -> tuple[np.ndarray, Mapping[int, str]]:
    nonzero = sorted({label for label in labels if label != "none"})
    if len(nonzero) >= int(np.iinfo(np.uint16).max):
        raise RefinedDetectionDeltaError("Resolved reason registry exceeds uint16.")
    code_by_label = {label: index + 1 for index, label in enumerate(nonzero)}
    codes = np.asarray(
        [0 if label == "none" else code_by_label[label] for label in labels],
        dtype=np.uint16,
    )
    registry: Mapping[int, str] = MappingProxyType(
        {0: "none", **{code: label for label, code in code_by_label.items()}}
    )
    return codes, registry


def _frame_row_offsets(frame_indices: np.ndarray, *, n_frames: int) -> np.ndarray:
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    counts = np.bincount(frames, minlength=n_frames)
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def _payload_from_base(arrays: Mapping[str, Any], row: int) -> dict[str, Any]:
    prefix = "instances/"
    return {
        name: (
            np.array(arrays[f"{prefix}{name}"][row], copy=True)
            if name == "bbox_norm_coords"
            else np.asarray(arrays[f"{prefix}{name}"])[row].item()
        )
        for name in _PAYLOAD_FIELD_NAMES
    }


def _payload_from_event(batch: RefinedDetectionDeltaBatch, row: int) -> dict[str, Any]:
    return {
        name: (
            np.array(batch.arrays[name][row], copy=True)
            if name == "bbox_norm_coords"
            else batch.arrays[name][row].item()
        )
        for name in _PAYLOAD_FIELD_NAMES
    }


def _validate_payload_geometry(payload: Mapping[str, Any], *, n_frames: int) -> None:
    frame = int(payload["frame_indices"])
    bbox = np.asarray(payload["bbox_norm_coords"], dtype=np.float32).reshape(1, 4)
    if frame < 0 or frame >= n_frames:
        raise RefinedDetectionDeltaError("Delta payload frame is outside the base domain.")
    half = np.float32(0.5)
    valid = bool(
        np.isfinite(bbox).all()
        and bbox[0, 2] > 0
        and bbox[0, 3] > 0
        and bbox[0, 0] - bbox[0, 2] * half >= 0
        and bbox[0, 1] - bbox[0, 3] * half >= 0
        and bbox[0, 0] + bbox[0, 2] * half <= 1
        and bbox[0, 1] + bbox[0, 3] * half <= 1
    )
    if not valid:
        raise RefinedDetectionDeltaError(
            "Delta payload bbox must be finite, positive-area, and contained."
        )


def _require_same_lineage(
    batches: Sequence[RefinedDetectionDeltaBatch],
    *,
    base_snapshot_id: str,
    base_manifest_digest: str,
) -> str:
    expected_snapshot = _require_uuid(base_snapshot_id, name="base_snapshot_id")
    expected_digest = _require_sha256(
        base_manifest_digest,
        name="base_manifest_digest",
    )
    lineages = {batch.delta_lineage_id for batch in batches}
    if len(lineages) != 1:
        raise RefinedDetectionDeltaError(
            "All batches must belong to one delta_lineage_id."
        )
    for batch in batches:
        if batch.base_snapshot_id != expected_snapshot:
            raise RefinedDetectionDeltaError("Delta batch base_snapshot_id mismatch.")
        if batch.base_manifest_digest != expected_digest:
            raise RefinedDetectionDeltaError("Delta batch base manifest mismatch.")
    return next(iter(lineages))


def _event_rows(
    batches: Sequence[RefinedDetectionDeltaBatch],
) -> list[tuple[int, RefinedDetectionDeltaBatch, int]]:
    events = [
        (int(batch.arrays["event_sequence"][row]), batch, row)
        for batch in batches
        for row in range(batch.row_count)
    ]
    events.sort(key=lambda item: item[0])
    sequences = [event[0] for event in events]
    if len(sequences) != len(set(sequences)):
        raise RefinedDetectionDeltaError(
            "event_sequence must be globally unique across frozen partitions."
        )
    generation_order = [event[1].generation_ordinal for event in events]
    if any(left > right for left, right in zip(generation_order, generation_order[1:])):
        raise RefinedDetectionDeltaError(
            "event_sequence cannot move backward across generation ordinals."
        )
    return events


def resolve_refined_detection_deltas(
    *,
    base_dimensions: RefinedDetectionDimensions,
    base_arrays: Mapping[str, Any],
    base_instance_reason_codes: Mapping[int, str],
    base_source_reason_codes: Mapping[int, str],
    recording_identity: str,
    base_snapshot_id: str,
    base_manifest_digest: str,
    next_refined_row_id: int,
    batches: Sequence[RefinedDetectionDeltaBatch],
) -> RefinedDetectionDeltaResolution:
    """Resolve a full-acquisition refined-v1 base plus immutable v2 events."""

    if (
        base_dimensions.lineage_profile
        is not RefinedDetectionLineageProfile.FULL_ACQUISITION
    ):
        raise RefinedDetectionDeltaError(
            "Delta v2 initially targets full-acquisition/clip-local authorities; "
            "rebuild a clipped recording aggregate from its bound sources."
        )
    base_issues = REFINED_DETECTION_SCHEMA_V1.validate(
        base_arrays,
        dimensions=base_dimensions,
    )
    if base_issues:
        raise RefinedDetectionDeltaError(
            "Base refined snapshot is invalid: "
            + "; ".join(f"{issue.code}:{issue.path}" for issue in base_issues)
        )
    if type(next_refined_row_id) is not int or next_refined_row_id < 0:
        raise RefinedDetectionDeltaError(
            "next_refined_row_id must be a nonnegative exact integer."
        )
    base_row_ids = np.asarray(
        base_arrays["instances/refined_row_ids"], dtype=np.int64
    ).reshape(-1)
    if base_row_ids.size and next_refined_row_id <= int(np.max(base_row_ids)):
        raise RefinedDetectionDeltaError(
            "next_refined_row_id must exceed every base refined_row_id."
        )
    actor_recording = str(recording_identity).strip()
    if not actor_recording:
        raise RefinedDetectionDeltaError("recording_identity cannot be empty.")
    if not batches:
        raise RefinedDetectionDeltaError("At least one delta batch is required.")
    delta_lineage_id = _require_same_lineage(
        batches,
        base_snapshot_id=base_snapshot_id,
        base_manifest_digest=base_manifest_digest,
    )
    events = _event_rows(batches)

    instance_reason_labels = _decode_reason_labels(
        np.asarray(base_arrays["instances/reason_codes"], dtype=np.uint16),
        base_instance_reason_codes,
        name="instances/reason_codes",
    )
    source_reason_labels = _decode_reason_labels(
        np.asarray(base_arrays["source_detections/reason_codes"], dtype=np.uint16),
        base_source_reason_codes,
        name="source_detections/reason_codes",
    )
    base_keys = np.asarray(
        base_arrays["instances/instance_key"], dtype=np.uint64
    ).reshape(-1)
    states: dict[int, _RowState] = {}
    all_row_ids = set(int(value) for value in base_row_ids.tolist())
    for row, (key, row_id) in enumerate(
        zip(base_keys.tolist(), base_row_ids.tolist(), strict=True)
    ):
        states[int(key)] = _RowState(
            instance_key=int(key),
            refined_row_id=int(row_id),
            base_row_index=row,
            active=True,
            last_event_sequence=0,
            last_operation=None,
            reason_label=instance_reason_labels[row],
            payload=_payload_from_base(base_arrays, row),
        )

    source_keys = set(
        int(value)
        for value in np.asarray(
            base_arrays["source_detections/instance_key"], dtype=np.uint64
        ).tolist()
    )
    source_decisions = np.array(
        base_arrays["source_detections/decision_codes"],
        dtype=np.uint8,
        copy=True,
    )
    source_resolved = np.array(
        base_arrays["source_detections/resolved_refined_row_id"],
        dtype=np.int64,
        copy=True,
    )
    high_water = next_refined_row_id
    operation_counts = {
        name: 0 for name in REFINED_DETECTION_DELTA_OPERATION_CODE_MAP
    }
    touched_keys: set[int] = set()

    name_by_code = {
        code: name
        for name, code in REFINED_DETECTION_DELTA_OPERATION_CODE_MAP.items()
    }
    for sequence, batch, row in events:
        operation_code = int(batch.arrays["operation_codes"][row])
        operation = name_by_code[operation_code]
        key = int(batch.arrays["instance_key"][row])
        row_id = int(batch.arrays["refined_row_ids"][row])
        hint = int(batch.arrays["row_index_hint"][row])
        expected_previous = int(
            batch.arrays["expected_previous_event_sequence"][row]
        )
        reason_label = batch.reason_code_map[
            int(batch.arrays["reason_codes"][row])
        ]
        state = states.get(key)

        if operation == "add_instance":
            if state is not None or key in source_keys:
                raise RefinedDetectionDeltaError(
                    f"add_instance event {sequence} reuses instance_key {key}."
                )
            if hint != -1 or expected_previous != 0:
                raise RefinedDetectionDeltaError(
                    "add_instance requires row_index_hint=-1 and predecessor zero."
                )
            if row_id < high_water or row_id in all_row_ids:
                raise RefinedDetectionDeltaError(
                    "add_instance refined_row_id violates the monotonic allocator."
                )
            payload = _payload_from_event(batch, row)
            _validate_payload_geometry(payload, n_frames=base_dimensions.n_frames)
            if (
                int(payload["source_kind_codes"])
                != SOURCE_KIND_CODE_MAP["manual"]
                or int(payload["source_detect_row_index"]) != -1
                or bool(payload["score_valid"])
                or float(payload["scores"]) != 0.0
            ):
                raise RefinedDetectionDeltaError(
                    "add_instance must be a scoreless manual-origin row."
                )
            expected_key = int(
                mint_manual_curation_instance_keys(
                    recording_identity=actor_recording,
                    refined_row_ids=np.asarray([row_id], dtype=np.int64),
                    frame_indices=np.asarray(
                        [payload["frame_indices"]], dtype=np.int32
                    ),
                    bbox_norm_coords=np.asarray(
                        [payload["bbox_norm_coords"]], dtype=np.float32
                    ),
                    class_ids=np.asarray([payload["class_ids"]], dtype=np.int32),
                )[0]
            )
            if key != expected_key:
                raise RefinedDetectionDeltaError(
                    "add_instance instance_key does not match the frozen allocator."
                )
            states[key] = _RowState(
                instance_key=key,
                refined_row_id=row_id,
                base_row_index=None,
                active=True,
                last_event_sequence=sequence,
                last_operation=operation,
                reason_label=reason_label,
                payload=payload,
            )
            all_row_ids.add(row_id)
            high_water = max(high_water, row_id + 1)
        else:
            if state is None:
                raise RefinedDetectionDeltaError(
                    f"{operation} event {sequence} targets an unknown instance_key."
                )
            if row_id != state.refined_row_id:
                raise RefinedDetectionDeltaError(
                    f"{operation} event {sequence} refined_row_id mismatch."
                )
            if hint not in {-1, state.base_row_index}:
                raise RefinedDetectionDeltaError(
                    f"{operation} event {sequence} row_index_hint mismatch."
                )
            if expected_previous != state.last_event_sequence:
                raise RefinedDetectionDeltaError(
                    f"{operation} event {sequence} has a stale predecessor."
                )
            if operation == "replace_instance":
                if not state.active:
                    raise RefinedDetectionDeltaError(
                        "replace_instance cannot target a tombstoned row."
                    )
                payload = _payload_from_event(batch, row)
                _validate_payload_geometry(payload, n_frames=base_dimensions.n_frames)
                immutable_fields = (
                    "frame_indices",
                    "source_acquisition_frame_index",
                    "scores",
                    "score_valid",
                    "source_kind_codes",
                    "source_detect_row_index",
                )
                changed = [
                    name
                    for name in immutable_fields
                    if payload[name] != state.payload[name]
                ]
                if changed:
                    raise RefinedDetectionDeltaError(
                        "replace_instance changed sealed identity/provenance fields: "
                        + ", ".join(changed)
                    )
                state.payload = payload
                state.reason_label = reason_label
            elif operation == "delete_instance":
                if not state.active:
                    raise RefinedDetectionDeltaError(
                        "delete_instance cannot target an existing tombstone."
                    )
                state.active = False
                source_row = int(state.payload["source_detect_row_index"])
                if source_row >= 0:
                    source_decisions[source_row] = np.uint8(
                        SOURCE_DECISION_CODE_MAP["manual_clear"]
                    )
                    source_resolved[source_row] = np.int64(-1)
                    source_reason_labels[source_row] = reason_label
            elif operation == "restore_instance":
                if state.active or state.last_operation != "delete_instance":
                    raise RefinedDetectionDeltaError(
                        "restore_instance requires the target's latest event to be "
                        "an uncompacted delete in this delta lineage."
                    )
                state.active = True
                state.reason_label = reason_label
                source_row = int(state.payload["source_detect_row_index"])
                if source_row >= 0:
                    source_decisions[source_row] = np.uint8(
                        SOURCE_DECISION_CODE_MAP["accepted"]
                    )
                    source_resolved[source_row] = np.int64(state.refined_row_id)
                    source_reason_labels[source_row] = reason_label
            state.last_event_sequence = sequence
            state.last_operation = operation
        operation_counts[operation] += 1
        touched_keys.add(key)

    active_states = sorted(
        (state for state in states.values() if state.active),
        key=lambda state: (
            int(state.payload["frame_indices"]),
            state.refined_row_id,
        ),
    )
    n_instances = len(active_states)
    instance_frames = np.asarray(
        [state.payload["frame_indices"] for state in active_states],
        dtype=np.int32,
    )
    instance_bbox = (
        np.asarray(
            [state.payload["bbox_norm_coords"] for state in active_states],
            dtype=np.float32,
        ).reshape(n_instances, 4)
        if active_states
        else np.empty((0, 4), dtype=np.float32)
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        instance_bbox,
        source_width=base_dimensions.source_width,
        source_height=base_dimensions.source_height,
    )
    resolved_instance_labels = [state.reason_label for state in active_states]
    instance_reason_values, instance_registry = _encode_reason_labels(
        resolved_instance_labels
    )
    source_reason_values, source_registry = _encode_reason_labels(
        source_reason_labels
    )

    arrays: dict[str, np.ndarray] = {
        "instances/frame_indices": instance_frames,
        "instances/source_acquisition_frame_index": np.asarray(
            [state.payload["source_acquisition_frame_index"] for state in active_states],
            dtype=np.int64,
        ),
        "instances/instance_key": np.asarray(
            [state.instance_key for state in active_states], dtype=np.uint64
        ),
        "instances/refined_row_ids": np.asarray(
            [state.refined_row_id for state in active_states], dtype=np.int64
        ),
        "instances/bbox_norm_coords": instance_bbox,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": np.asarray(
            [state.payload["scores"] for state in active_states], dtype=np.float32
        ),
        "instances/score_valid": np.asarray(
            [state.payload["score_valid"] for state in active_states], dtype=np.bool_
        ),
        "instances/class_ids": np.asarray(
            [state.payload["class_ids"] for state in active_states], dtype=np.int32
        ),
        "instances/source_kind_codes": np.asarray(
            [state.payload["source_kind_codes"] for state in active_states],
            dtype=np.uint8,
        ),
        "instances/manual_edit_flags": np.asarray(
            [state.payload["manual_edit_flags"] for state in active_states],
            dtype=np.bool_,
        ),
        "instances/source_detect_row_index": np.asarray(
            [state.payload["source_detect_row_index"] for state in active_states],
            dtype=np.int64,
        ),
        "instances/reason_codes": instance_reason_values,
        "instances/frame_row_offsets": _frame_row_offsets(
            instance_frames,
            n_frames=base_dimensions.n_frames,
        ),
    }
    for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(base_dimensions):
        if not path.startswith("source_detections/"):
            continue
        arrays[path] = np.array(base_arrays[path], copy=True)
    arrays["source_detections/decision_codes"] = source_decisions
    arrays["source_detections/resolved_refined_row_id"] = source_resolved
    arrays["source_detections/reason_codes"] = source_reason_values

    dimensions = RefinedDetectionDimensions(
        n_frames=base_dimensions.n_frames,
        n_instances=n_instances,
        n_source_detections=base_dimensions.n_source_detections,
        source_width=base_dimensions.source_width,
        source_height=base_dimensions.source_height,
        lineage_profile=base_dimensions.lineage_profile,
    )
    issues = REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions)
    if issues:
        raise RefinedDetectionDeltaError(
            "Resolved refined snapshot is invalid: "
            + "; ".join(f"{issue.code}:{issue.path}" for issue in issues)
        )

    final_keys = set(int(state.instance_key) for state in active_states)
    initial_keys = set(int(value) for value in base_keys.tolist())
    report: Mapping[str, object] = MappingProxyType(
        {
            "schema_id": "palette.refined_detection.delta_resolution",
            "schema_version": 1,
            "delta_schema_id": REFINED_DETECTION_DELTA_SCHEMA_ID,
            "delta_schema_version": REFINED_DETECTION_DELTA_SCHEMA_VERSION,
            "delta_lineage_id": delta_lineage_id,
            "base_snapshot_id": str(base_snapshot_id),
            "base_manifest_digest": str(base_manifest_digest),
            "batch_count": len(batches),
            "event_count": len(events),
            "generation_ordinals": sorted(
                {batch.generation_ordinal for batch in batches}
            ),
            "operation_counts": operation_counts,
            "touched_instance_keys": sorted(touched_keys),
            "added_instance_keys": sorted(final_keys - initial_keys),
            "deleted_instance_keys": sorted(initial_keys - final_keys),
            "rowset_changed": final_keys != initial_keys,
            "next_refined_row_id": high_water,
            "compaction_required": True,
            "downstream_publication_policy": (
                "complete_replacement_required_before_selection"
            ),
            "production_state_changes": [],
        }
    )
    return RefinedDetectionDeltaResolution(
        dimensions=dimensions,
        arrays=MappingProxyType(arrays),
        instance_reason_codes=instance_registry,
        source_reason_codes=source_registry,
        next_refined_row_id=high_water,
        report=report,
    )


__all__ = [
    "REFINED_DETECTION_DELTA_ARRAYS",
    "REFINED_DETECTION_DELTA_LAYOUT",
    "REFINED_DETECTION_DELTA_MERGE_ORDER",
    "REFINED_DETECTION_DELTA_OPERATION_CODE_MAP",
    "REFINED_DETECTION_DELTA_SCHEMA_ID",
    "REFINED_DETECTION_DELTA_SCHEMA_VERSION",
    "RefinedDetectionDeltaArray",
    "RefinedDetectionDeltaBatch",
    "RefinedDetectionDeltaError",
    "RefinedDetectionDeltaResolution",
    "refined_detection_delta_schema_manifest",
    "resolve_refined_detection_deltas",
]

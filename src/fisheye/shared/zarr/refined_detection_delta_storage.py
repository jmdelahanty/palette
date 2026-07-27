"""Persisted refined-detection delta v2 partitions and frozen generations.

This module owns the Zarr-v3 persistence boundary for the logical contract in
``refined_detection_delta``.  Partitions are immutable single-writer objects;
generation manifests become authoritative only after every partition, array
digest, and physical declaration has been recomputed successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from fisheye.shared.zarr.array_contracts import (
    BOOL,
    FLOAT32,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT64,
    ArrayContract,
    DTypeContract,
)
from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.codec_profiles import get_codec_profile
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_LAYOUT,
    REFINED_DETECTION_DELTA_SCHEMA_ID,
    REFINED_DETECTION_DELTA_SCHEMA_VERSION,
    RefinedDetectionDeltaBatch,
    RefinedDetectionDeltaResolution,
    resolve_refined_detection_deltas,
)
from fisheye.shared.zarr.refined_detection_schema import RefinedDetectionDimensions
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import EDITABLE_LOCAL_V1


REFINED_DETECTION_DELTA_PARENT = "refined_detection_delta_runs"
REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_ID = (
    "palette.refined_detection.delta_lineage"
)
REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_VERSION = 1
REFINED_DETECTION_DELTA_GENERATION_SCHEMA_ID = (
    "palette.refined_detection.delta_generation"
)
REFINED_DETECTION_DELTA_GENERATION_SCHEMA_VERSION = 1
REFINED_DETECTION_DELTA_PARTITION_SCHEMA_ID = (
    "palette.refined_detection.delta_partition"
)
REFINED_DETECTION_DELTA_PARTITION_SCHEMA_VERSION = 1
REFINED_DETECTION_DELTA_CONTENT_DIGEST_ALGORITHM = (
    "sha256_canonical_array_little_endian_v1"
)
REFINED_DETECTION_DELTA_GENERATION_DIGEST_ALGORITHM = (
    "sha256_canonical_partition_receipts_v1"
)
REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION = 65_536

_LINEAGE_MANIFEST_ATTRIBUTE = "lineage_manifest"
_GENERATION_MANIFEST_ATTRIBUTE = "generation_manifest"
_PARTITION_MANIFEST_ATTRIBUTE = "partition_manifest"
_UINT64_MAX = int(np.iinfo(np.uint64).max)

_DTYPE_CONTRACTS: Mapping[str, DTypeContract] = MappingProxyType(
    {
        "bool": BOOL,
        "float32": FLOAT32,
        "int32": INT32,
        "int64": INT64,
        "uint8": UINT8,
        "uint16": UINT16,
        "uint64": UINT64,
    }
)


class RefinedDetectionDeltaStorageError(ValueError):
    """Raised when persisted delta state is incomplete or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_utc_timestamp(value: str, *, name: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be an ISO-8601 UTC timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RefinedDetectionDeltaStorageError(f"{name} must be timezone-aware UTC.")
    return text


def _require_text(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise RefinedDetectionDeltaStorageError(f"{name} cannot be empty.")
    return text


def _require_uuid(value: str, *, name: str) -> str:
    text = str(value).strip()
    try:
        canonical = str(uuid.UUID(text))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be a canonical UUID."
        ) from exc
    if text != canonical:
        raise RefinedDetectionDeltaStorageError(
            f"{name} must use canonical lowercase UUID text."
        )
    return canonical


def _require_sha256(value: str, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be a lowercase 64-character SHA-256 digest."
        )
    return text


def _normalize_group_path(value: str, *, name: str) -> str:
    text = str(value).strip().strip("/")
    parts = text.split("/") if text else []
    if not parts or any(
        not part
        or part in {".", ".."}
        or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for character in part)
        for part in parts
    ):
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be a nonempty path of safe Zarr group components."
        )
    return "/".join(parts)


def _require_ordinal(value: int, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be an exact uint64-domain integer."
        )
    return value


def _generation_name(ordinal: int) -> str:
    return f"generation_{_require_ordinal(ordinal, name='generation_ordinal'):020d}"


def _json_tree(value: Any) -> Any:
    """Normalize tuples and NumPy scalars into a strict JSON value tree."""

    return json.loads(canonical_json_bytes(value))


def _require_exact_fields(
    value: Any,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RefinedDetectionDeltaStorageError(f"{name} must be an object.")
    observed = {str(key) for key in value}
    if observed != expected:
        raise RefinedDetectionDeltaStorageError(
            f"{name} has an unexpected field set; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )
    return value


def _manifest(payload: Mapping[str, Any]) -> dict[str, object]:
    normalized = _json_tree(payload)
    return {
        "payload": normalized,
        "payload_digest": canonical_json_sha256(normalized),
    }


def _validated_manifest(value: Any, *, name: str) -> Mapping[str, Any]:
    manifest = _require_exact_fields(
        value,
        {"payload", "payload_digest"},
        name=name,
    )
    payload = manifest["payload"]
    if not isinstance(payload, Mapping):
        raise RefinedDetectionDeltaStorageError(f"{name}.payload must be an object.")
    observed = _require_sha256(
        str(manifest["payload_digest"]),
        name=f"{name}.payload_digest",
    )
    expected = canonical_json_sha256(payload)
    if observed != expected:
        raise RefinedDetectionDeltaStorageError(f"{name} payload digest mismatch.")
    return manifest


@dataclass(frozen=True)
class RefinedDetectionDeltaLineageBinding:
    """Exact base snapshot and allocator binding for one delta lineage."""

    delta_lineage_id: str
    base_run_path: str
    base_snapshot_id: str
    base_manifest_digest: str
    recording_identity: str
    base_next_refined_row_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_lineage_id",
            _require_uuid(self.delta_lineage_id, name="delta_lineage_id"),
        )
        object.__setattr__(
            self,
            "base_run_path",
            _normalize_group_path(self.base_run_path, name="base_run_path"),
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
        object.__setattr__(
            self,
            "recording_identity",
            _require_text(self.recording_identity, name="recording_identity"),
        )
        if (
            type(self.base_next_refined_row_id) is not int
            or self.base_next_refined_row_id < 0
        ):
            raise RefinedDetectionDeltaStorageError(
                "base_next_refined_row_id must be a nonnegative exact integer."
            )

    def as_manifest(self) -> dict[str, object]:
        return {
            "delta_lineage_id": self.delta_lineage_id,
            "base_run_path": self.base_run_path,
            "base_snapshot_id": self.base_snapshot_id,
            "base_manifest_digest": self.base_manifest_digest,
            "recording_identity": self.recording_identity,
            "base_next_refined_row_id": self.base_next_refined_row_id,
        }


@dataclass(frozen=True)
class FrozenRefinedDetectionDeltaGeneration:
    """Fully verified lineage prefix through one frozen generation."""

    binding: RefinedDetectionDeltaLineageBinding
    generation_ordinal: int
    batches: tuple[RefinedDetectionDeltaBatch, ...]
    generation_manifest: Mapping[str, Any]
    partition_manifests: Mapping[str, Mapping[str, Any]]

    def resolve(
        self,
        *,
        base_dimensions: RefinedDetectionDimensions,
        base_arrays: Mapping[str, Any],
        base_instance_reason_codes: Mapping[int, str],
        base_source_reason_codes: Mapping[int, str],
    ) -> RefinedDetectionDeltaResolution:
        """Resolve only after this object has passed every persisted digest gate."""

        return resolve_refined_detection_deltas(
            base_dimensions=base_dimensions,
            base_arrays=base_arrays,
            base_instance_reason_codes=base_instance_reason_codes,
            base_source_reason_codes=base_source_reason_codes,
            recording_identity=self.binding.recording_identity,
            base_snapshot_id=self.binding.base_snapshot_id,
            base_manifest_digest=self.binding.base_manifest_digest,
            next_refined_row_id=self.binding.base_next_refined_row_id,
            batches=self.batches,
        )


def _array_contract(name: str, dtype: str, trailing_shape: tuple[int, ...]) -> ArrayContract:
    try:
        dtype_contract = _DTYPE_CONTRACTS[dtype]
    except KeyError as exc:
        raise RefinedDetectionDeltaStorageError(
            f"No persisted dtype contract is registered for {dtype!r}."
        ) from exc
    axes = ("event",) if not trailing_shape else ("event", "bbox_component")
    return ArrayContract(
        schema_id=f"{REFINED_DETECTION_DELTA_SCHEMA_ID}.{name}",
        schema_version=REFINED_DETECTION_DELTA_SCHEMA_VERSION,
        dtype=dtype_contract,
        shape_template=("n_events", *trailing_shape),
        axis_names=axes,
        description=f"Refined-detection delta v2 event column {name}.",
    )


_ARRAY_CONTRACTS: Mapping[str, ArrayContract] = MappingProxyType(
    {
        declaration.name: _array_contract(
            declaration.name,
            declaration.dtype,
            declaration.trailing_shape,
        )
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
)


def _array_plan(name: str, values: np.ndarray) -> StoragePlan:
    contract = _ARRAY_CONTRACTS[name]
    shape = tuple(int(value) for value in values.shape)
    intent = contract.storage_intent(
        shape=shape,
        access=AccessPattern.EAGER,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1, *shape[1:]),
        growth_axis=0,
        shard_axes=(0,),
        name=name,
        dimensions={"n_events": int(shape[0])},
    )
    plan = plan_storage(intent, EDITABLE_LOCAL_V1)
    if plan.shard_shape is not None or plan.chunk_shape != shape:
        raise RefinedDetectionDeltaStorageError(
            f"Sparse delta array {name!r} must resolve to one ordinary chunk."
        )
    return plan


def refined_detection_delta_array_digest(name: str, values: np.ndarray) -> str:
    """Hash one exact logical array using canonical little-endian C bytes."""

    if name not in _ARRAY_CONTRACTS:
        raise RefinedDetectionDeltaStorageError(f"Unknown delta array {name!r}.")
    array = np.asarray(values)
    _ARRAY_CONTRACTS[name].require_observation(
        array,
        dimensions={"n_events": int(array.shape[0])},
    )
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    header = {
        "algorithm": REFINED_DETECTION_DELTA_CONTENT_DIGEST_ALGORITHM,
        "name": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "byte_order": "little",
        "array_order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(canonical.view(np.uint8).tobytes())
    return digest.hexdigest()


def _array_receipts(batch: RefinedDetectionDeltaBatch) -> dict[str, dict[str, object]]:
    return {
        declaration.name: {
            "dtype": declaration.dtype,
            "shape": list(batch.arrays[declaration.name].shape),
            "logical_digest_algorithm": (
                REFINED_DETECTION_DELTA_CONTENT_DIGEST_ALGORITHM
            ),
            "logical_sha256": refined_detection_delta_array_digest(
                declaration.name,
                batch.arrays[declaration.name],
            ),
        }
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }


def _partition_content_document(
    *,
    batch: RefinedDetectionDeltaBatch,
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "schema_id": "palette.refined_detection.delta_partition_content",
        "schema_version": 1,
        "delta_lineage_id": batch.delta_lineage_id,
        "generation_ordinal": batch.generation_ordinal,
        "partition_id": batch.partition_id,
        "arrays": _json_tree(receipts),
    }


def _partition_payload(
    batch: RefinedDetectionDeltaBatch,
    *,
    created_at_utc: str,
) -> dict[str, object]:
    if batch.row_count > REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION:
        raise RefinedDetectionDeltaStorageError(
            "Delta partition exceeds the frozen 65,536-event size bound."
        )
    plans = {
        declaration.name: _array_plan(
            declaration.name,
            batch.arrays[declaration.name],
        ).as_dict()
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
    receipts = _array_receipts(batch)
    content_document = _partition_content_document(
        batch=batch,
        receipts=receipts,
    )
    return {
        "schema_id": REFINED_DETECTION_DELTA_PARTITION_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_DELTA_PARTITION_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": _require_utc_timestamp(
            created_at_utc,
            name="partition created_at_utc",
        ),
        "batch_manifest": batch.as_manifest(),
        "storage_profile": EDITABLE_LOCAL_V1.as_manifest(),
        "codec_profile": get_codec_profile(
            EDITABLE_LOCAL_V1.codec_profile_id
        ).as_manifest(),
        "array_storage_plans": plans,
        "array_receipts": receipts,
        "partition_content_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "partition_content_digest": canonical_json_sha256(content_document),
    }


def _lineage_payload(
    binding: RefinedDetectionDeltaLineageBinding,
    *,
    created_at_utc: str,
    created_by: str,
) -> dict[str, object]:
    return {
        "schema_id": REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_VERSION,
        "status": "open",
        "created_at_utc": _require_utc_timestamp(
            created_at_utc,
            name="lineage created_at_utc",
        ),
        "created_by": _require_text(created_by, name="created_by"),
        "delta_schema": {
            "id": REFINED_DETECTION_DELTA_SCHEMA_ID,
            "version": REFINED_DETECTION_DELTA_SCHEMA_VERSION,
            "layout": REFINED_DETECTION_DELTA_LAYOUT,
        },
        "binding": binding.as_manifest(),
        "generation_path_contract": (
            "generations/generation_<20-digit-uint64>/partitions/<partition_id>"
        ),
        "partition_event_limit": REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION,
    }


def _generation_payload(
    binding: RefinedDetectionDeltaLineageBinding,
    *,
    generation_ordinal: int,
    created_at_utc: str,
    created_by: str,
    status: str,
    previous_generation_ordinal: int | None,
    previous_generation_manifest_digest: str | None,
    minimum_event_sequence_exclusive: int,
    frozen_at_utc: str | None = None,
    frozen_by: str | None = None,
    partition_receipts: Sequence[Mapping[str, Any]] = (),
    generation_content_digest: str | None = None,
) -> dict[str, object]:
    ordinal = _require_ordinal(generation_ordinal, name="generation_ordinal")
    if status not in {"open", "frozen"}:
        raise RefinedDetectionDeltaStorageError(
            "Generation status must be open or frozen."
        )
    if (previous_generation_ordinal is None) != (
        previous_generation_manifest_digest is None
    ):
        raise RefinedDetectionDeltaStorageError(
            "Generation predecessor ordinal and manifest digest must be paired."
        )
    if previous_generation_ordinal is not None:
        previous_generation_ordinal = _require_ordinal(
            previous_generation_ordinal,
            name="previous_generation_ordinal",
        )
        if previous_generation_ordinal >= ordinal:
            raise RefinedDetectionDeltaStorageError(
                "Generation predecessor ordinal must be smaller than its successor."
            )
        previous_generation_manifest_digest = _require_sha256(
            str(previous_generation_manifest_digest),
            name="previous_generation_manifest_digest",
        )
    sequence_floor = _require_ordinal(
        minimum_event_sequence_exclusive,
        name="minimum_event_sequence_exclusive",
    )
    if previous_generation_ordinal is None and sequence_floor != 0:
        raise RefinedDetectionDeltaStorageError(
            "The first generation must use an event-sequence floor of zero."
        )
    if status == "open":
        if (
            frozen_at_utc is not None
            or frozen_by is not None
            or partition_receipts
            or generation_content_digest is not None
        ):
            raise RefinedDetectionDeltaStorageError(
                "Open generations cannot declare frozen evidence."
            )
    else:
        if not partition_receipts:
            raise RefinedDetectionDeltaStorageError(
                "Frozen generations require at least one partition receipt."
            )
        frozen_at_utc = _require_utc_timestamp(
            str(frozen_at_utc),
            name="generation frozen_at_utc",
        )
        frozen_by = _require_text(str(frozen_by), name="frozen_by")
        generation_content_digest = _require_sha256(
            str(generation_content_digest),
            name="generation_content_digest",
        )
    return {
        "schema_id": REFINED_DETECTION_DELTA_GENERATION_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_DELTA_GENERATION_SCHEMA_VERSION,
        "status": status,
        "delta_lineage_id": binding.delta_lineage_id,
        "base_snapshot_id": binding.base_snapshot_id,
        "base_manifest_digest": binding.base_manifest_digest,
        "generation_ordinal": ordinal,
        "generation_name": _generation_name(ordinal),
        "previous_generation_ordinal": previous_generation_ordinal,
        "previous_generation_manifest_digest": (
            previous_generation_manifest_digest
        ),
        "minimum_event_sequence_exclusive": sequence_floor,
        "created_at_utc": _require_utc_timestamp(
            created_at_utc,
            name="generation created_at_utc",
        ),
        "created_by": _require_text(created_by, name="created_by"),
        "frozen_at_utc": frozen_at_utc,
        "frozen_by": frozen_by,
        "partition_count": len(partition_receipts),
        "partition_receipts": _json_tree(partition_receipts),
        "generation_content_digest_algorithm": (
            REFINED_DETECTION_DELTA_GENERATION_DIGEST_ALGORITHM
        ),
        "generation_content_digest": generation_content_digest,
    }


def _group_names(group: Any) -> list[str]:
    return sorted(str(name) for name in group.group_keys())


def _array_names(group: Any) -> list[str]:
    return sorted(str(name) for name in group.array_keys())


def _require_zarr_v3_group(group: Any, *, name: str) -> None:
    zarr_format = getattr(getattr(group, "metadata", None), "zarr_format", None)
    if zarr_format != 3:
        raise RefinedDetectionDeltaStorageError(
            f"{name} must be backed by a Zarr v3 group."
        )


def _generation_entries(lineage: Any) -> list[tuple[int, str]]:
    generations = lineage["generations"]
    if _array_names(generations):
        raise RefinedDetectionDeltaStorageError(
            "The generations container cannot contain arrays."
        )
    entries: list[tuple[int, str]] = []
    for name in _group_names(generations):
        prefix = "generation_"
        suffix = name.removeprefix(prefix)
        if (
            not name.startswith(prefix)
            or len(suffix) != 20
            or not suffix.isdigit()
            or _generation_name(int(suffix)) != name
        ):
            raise RefinedDetectionDeltaStorageError(
                f"Invalid refined-detection delta generation name: {name!r}."
            )
        entries.append((int(suffix), name))
    return sorted(entries)


def _require_only_attribute(group: Any, name: str, *, path: str) -> Mapping[str, Any]:
    attrs = dict(group.attrs)
    if set(attrs) != {name}:
        raise RefinedDetectionDeltaStorageError(
            f"{path} attributes must contain exactly {name!r}."
        )
    return _validated_manifest(attrs[name], name=f"{path}.{name}")


def _lineage_group(root: Any, delta_lineage_id: str) -> Any:
    _require_zarr_v3_group(root, name="delta archive root")
    lineage_id = _require_uuid(delta_lineage_id, name="delta_lineage_id")
    try:
        return root[f"{REFINED_DETECTION_DELTA_PARENT}/{lineage_id}"]
    except KeyError as exc:
        raise RefinedDetectionDeltaStorageError(
            f"Unknown refined-detection delta lineage {lineage_id}."
        ) from exc


def _read_lineage_binding(root: Any, delta_lineage_id: str) -> tuple[Any, RefinedDetectionDeltaLineageBinding, Mapping[str, Any]]:
    lineage = _lineage_group(root, delta_lineage_id)
    manifest = _require_only_attribute(
        lineage,
        _LINEAGE_MANIFEST_ATTRIBUTE,
        path=f"{REFINED_DETECTION_DELTA_PARENT}/{delta_lineage_id}",
    )
    payload = _require_exact_fields(
        manifest["payload"],
        {
            "schema_id",
            "schema_version",
            "status",
            "created_at_utc",
            "created_by",
            "delta_schema",
            "binding",
            "generation_path_contract",
            "partition_event_limit",
        },
        name="lineage manifest payload",
    )
    binding_value = _require_exact_fields(
        payload["binding"],
        {
            "delta_lineage_id",
            "base_run_path",
            "base_snapshot_id",
            "base_manifest_digest",
            "recording_identity",
            "base_next_refined_row_id",
        },
        name="lineage binding",
    )
    binding = RefinedDetectionDeltaLineageBinding(**binding_value)
    expected = _lineage_payload(
        binding,
        created_at_utc=str(payload["created_at_utc"]),
        created_by=str(payload["created_by"]),
    )
    if _json_tree(payload) != expected:
        raise RefinedDetectionDeltaStorageError(
            "Lineage manifest does not match the frozen v1 contract."
        )
    if binding.delta_lineage_id != delta_lineage_id:
        raise RefinedDetectionDeltaStorageError(
            "Lineage path identity does not match its manifest."
        )
    if _group_names(lineage) != ["generations"] or _array_names(lineage):
        raise RefinedDetectionDeltaStorageError(
            "Delta lineage must contain only the generations group."
        )
    return lineage, binding, manifest


def create_refined_detection_delta_lineage(
    root: Any,
    *,
    binding: RefinedDetectionDeltaLineageBinding,
    created_by: str,
    created_at_utc: str | None = None,
    initial_generation_ordinal: int = 0,
) -> Mapping[str, Any]:
    """Create one lineage and its first open generation, failing on reuse."""

    timestamp = _require_utc_timestamp(
        created_at_utc or _utc_now(),
        name="created_at_utc",
    )
    ordinal = _require_ordinal(
        initial_generation_ordinal,
        name="initial_generation_ordinal",
    )
    _require_zarr_v3_group(root, name="delta archive root")
    parent = root.require_group(REFINED_DETECTION_DELTA_PARENT)
    if binding.delta_lineage_id in parent:
        raise RefinedDetectionDeltaStorageError(
            f"Delta lineage already exists: {binding.delta_lineage_id}."
        )
    lineage = parent.create_group(binding.delta_lineage_id)
    lineage.attrs[_LINEAGE_MANIFEST_ATTRIBUTE] = _manifest(
        _lineage_payload(
            binding,
            created_at_utc=timestamp,
            created_by=created_by,
        )
    )
    lineage.create_group("generations")
    create_refined_detection_delta_generation(
        root,
        delta_lineage_id=binding.delta_lineage_id,
        generation_ordinal=ordinal,
        created_by=created_by,
        created_at_utc=timestamp,
    )
    return dict(lineage.attrs[_LINEAGE_MANIFEST_ATTRIBUTE])


def create_refined_detection_delta_generation(
    root: Any,
    *,
    delta_lineage_id: str,
    generation_ordinal: int,
    created_by: str,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Create one empty open generation under an existing lineage."""

    lineage, binding, _lineage_manifest = _read_lineage_binding(
        root,
        delta_lineage_id,
    )
    ordinal = _require_ordinal(generation_ordinal, name="generation_ordinal")
    name = _generation_name(ordinal)
    generations = lineage["generations"]
    if name in generations:
        raise RefinedDetectionDeltaStorageError(
            f"Delta generation already exists: {name}."
        )
    existing_ordinals = [value for value, _name in _generation_entries(lineage)]
    if existing_ordinals and ordinal <= max(existing_ordinals):
        raise RefinedDetectionDeltaStorageError(
            "New generation ordinal must exceed every existing generation."
        )
    previous_ordinal: int | None = None
    previous_digest: str | None = None
    sequence_floor = 0
    if existing_ordinals:
        previous_ordinal = max(existing_ordinals)
        previous = read_frozen_refined_detection_delta_generation(
            root,
            delta_lineage_id=binding.delta_lineage_id,
            generation_ordinal=previous_ordinal,
        )
        previous_digest = str(previous.generation_manifest["payload_digest"])
        sequence_floor = max(
            int(sequence)
            for batch in previous.batches
            for sequence in batch.arrays["event_sequence"]
        )
    timestamp = _require_utc_timestamp(
        created_at_utc or _utc_now(),
        name="created_at_utc",
    )
    generation = generations.create_group(name)
    generation.create_group("partitions")
    generation.attrs[_GENERATION_MANIFEST_ATTRIBUTE] = _manifest(
        _generation_payload(
            binding,
            generation_ordinal=ordinal,
            created_at_utc=timestamp,
            created_by=created_by,
            status="open",
            previous_generation_ordinal=previous_ordinal,
            previous_generation_manifest_digest=previous_digest,
            minimum_event_sequence_exclusive=sequence_floor,
        )
    )
    return dict(generation.attrs[_GENERATION_MANIFEST_ATTRIBUTE])


def _generation_group(root: Any, *, delta_lineage_id: str, generation_ordinal: int) -> tuple[Any, RefinedDetectionDeltaLineageBinding]:
    lineage, binding, _lineage_manifest = _read_lineage_binding(
        root,
        delta_lineage_id,
    )
    name = _generation_name(generation_ordinal)
    try:
        generation = lineage[f"generations/{name}"]
    except KeyError as exc:
        raise RefinedDetectionDeltaStorageError(
            f"Unknown refined-detection delta generation {name}."
        ) from exc
    if _group_names(generation) != ["partitions"] or _array_names(generation):
        raise RefinedDetectionDeltaStorageError(
            "Delta generation must contain only the partitions group."
        )
    return generation, binding


def _read_generation_manifest(
    generation: Any,
    *,
    binding: RefinedDetectionDeltaLineageBinding,
    generation_ordinal: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = _require_only_attribute(
        generation,
        _GENERATION_MANIFEST_ATTRIBUTE,
        path=_generation_name(generation_ordinal),
    )
    payload = _require_exact_fields(
        manifest["payload"],
        {
            "schema_id",
            "schema_version",
            "status",
            "delta_lineage_id",
            "base_snapshot_id",
            "base_manifest_digest",
            "generation_ordinal",
            "generation_name",
            "previous_generation_ordinal",
            "previous_generation_manifest_digest",
            "minimum_event_sequence_exclusive",
            "created_at_utc",
            "created_by",
            "frozen_at_utc",
            "frozen_by",
            "partition_count",
            "partition_receipts",
            "generation_content_digest_algorithm",
            "generation_content_digest",
        },
        name="generation manifest payload",
    )
    expected = _generation_payload(
        binding,
        generation_ordinal=generation_ordinal,
        created_at_utc=str(payload["created_at_utc"]),
        created_by=str(payload["created_by"]),
        status=str(payload["status"]),
        previous_generation_ordinal=payload["previous_generation_ordinal"],
        previous_generation_manifest_digest=payload[
            "previous_generation_manifest_digest"
        ],
        minimum_event_sequence_exclusive=payload[
            "minimum_event_sequence_exclusive"
        ],
        frozen_at_utc=payload["frozen_at_utc"],
        frozen_by=payload["frozen_by"],
        partition_receipts=payload["partition_receipts"],
        generation_content_digest=payload["generation_content_digest"],
    )
    if _json_tree(payload) != expected:
        raise RefinedDetectionDeltaStorageError(
            "Generation manifest does not match the frozen v1 contract."
        )
    return manifest, payload


def write_refined_detection_delta_partition(
    root: Any,
    *,
    batch: RefinedDetectionDeltaBatch,
    created_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Persist one complete immutable partition or accept an exact retry."""

    generation, binding = _generation_group(
        root,
        delta_lineage_id=batch.delta_lineage_id,
        generation_ordinal=batch.generation_ordinal,
    )
    _generation_manifest, generation_payload = _read_generation_manifest(
        generation,
        binding=binding,
        generation_ordinal=batch.generation_ordinal,
    )
    if (
        batch.base_snapshot_id != binding.base_snapshot_id
        or batch.base_manifest_digest != binding.base_manifest_digest
    ):
        raise RefinedDetectionDeltaStorageError(
            "Delta batch does not match the lineage base binding."
        )
    if batch.row_count > REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION:
        raise RefinedDetectionDeltaStorageError(
            "Delta partition exceeds the frozen 65,536-event size bound."
        )
    partitions = generation["partitions"]
    if batch.partition_id in partitions:
        existing_batch, existing_manifest = read_refined_detection_delta_partition(
            root,
            delta_lineage_id=batch.delta_lineage_id,
            generation_ordinal=batch.generation_ordinal,
            partition_id=batch.partition_id,
        )
        if existing_batch.as_manifest() != batch.as_manifest() or any(
            not np.array_equal(
                existing_batch.arrays[name],
                batch.arrays[name],
                equal_nan=True,
            )
            for name in existing_batch.arrays
        ):
            raise RefinedDetectionDeltaStorageError(
                "Partition retry conflicts with already persisted content."
            )
        return existing_manifest

    if generation_payload["status"] != "open":
        raise RefinedDetectionDeltaStorageError(
            "Cannot append to a generation that is not open."
        )
    sequence_floor = int(generation_payload["minimum_event_sequence_exclusive"])
    if np.any(batch.arrays["event_sequence"] <= np.uint64(sequence_floor)):
        raise RefinedDetectionDeltaStorageError(
            "Delta event_sequence does not advance beyond the prior generation."
        )

    proposed_sequences = set(int(value) for value in batch.arrays["event_sequence"])
    for partition_id in _group_names(partitions):
        existing_batch, _manifest_value = read_refined_detection_delta_partition(
            root,
            delta_lineage_id=batch.delta_lineage_id,
            generation_ordinal=batch.generation_ordinal,
            partition_id=partition_id,
        )
        existing_sequences = set(
            int(value) for value in existing_batch.arrays["event_sequence"]
        )
        if proposed_sequences.intersection(existing_sequences):
            raise RefinedDetectionDeltaStorageError(
                "event_sequence collides with an existing partition."
            )

    timestamp = _require_utc_timestamp(
        created_at_utc or _utc_now(),
        name="created_at_utc",
    )
    partition = partitions.create_group(batch.partition_id)
    for declaration in REFINED_DETECTION_DELTA_ARRAYS:
        values = batch.arrays[declaration.name]
        plan = _array_plan(declaration.name, values)
        destination = create_array_from_plan(
            partition,
            name=declaration.name,
            contract=_ARRAY_CONTRACTS[declaration.name],
            plan=plan,
            fill_value=False if values.dtype == np.dtype("bool") else 0,
        )
        destination[:] = values
    manifest = _manifest(_partition_payload(batch, created_at_utc=timestamp))
    partition.attrs[_PARTITION_MANIFEST_ATTRIBUTE] = manifest
    _loaded, verified = read_refined_detection_delta_partition(
        root,
        delta_lineage_id=batch.delta_lineage_id,
        generation_ordinal=batch.generation_ordinal,
        partition_id=batch.partition_id,
    )
    return verified


def _observed_array_metadata(array: Any) -> Mapping[str, Any]:
    metadata = getattr(array, "metadata", None)
    if metadata is None or not hasattr(metadata, "to_dict"):
        raise RefinedDetectionDeltaStorageError(
            "Persisted delta arrays must expose Zarr v3 metadata."
        )
    value = _json_tree(metadata.to_dict())
    if not isinstance(value, Mapping):
        raise RefinedDetectionDeltaStorageError("Array metadata must be an object.")
    return value


def read_refined_detection_delta_partition(
    root: Any,
    *,
    delta_lineage_id: str,
    generation_ordinal: int,
    partition_id: str,
) -> tuple[RefinedDetectionDeltaBatch, Mapping[str, Any]]:
    """Read one complete partition and recompute logical and physical evidence."""

    generation, binding = _generation_group(
        root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    partition_name = str(partition_id).strip()
    if not partition_name or "/" in partition_name:
        raise RefinedDetectionDeltaStorageError("partition_id must be one component.")
    try:
        partition = generation[f"partitions/{partition_name}"]
    except KeyError as exc:
        raise RefinedDetectionDeltaStorageError(
            f"Unknown refined-detection delta partition {partition_name}."
        ) from exc
    if _group_names(partition):
        raise RefinedDetectionDeltaStorageError(
            "Delta partitions cannot contain nested groups."
        )
    expected_arrays = sorted(
        declaration.name for declaration in REFINED_DETECTION_DELTA_ARRAYS
    )
    if _array_names(partition) != expected_arrays:
        raise RefinedDetectionDeltaStorageError(
            "Delta partition arrays do not match the exact v2 schema."
        )
    manifest = _require_only_attribute(
        partition,
        _PARTITION_MANIFEST_ATTRIBUTE,
        path=f"partition {partition_name}",
    )
    payload = _require_exact_fields(
        manifest["payload"],
        {
            "schema_id",
            "schema_version",
            "status",
            "created_at_utc",
            "batch_manifest",
            "storage_profile",
            "codec_profile",
            "array_storage_plans",
            "array_receipts",
            "partition_content_digest_algorithm",
            "partition_content_digest",
        },
        name="partition manifest payload",
    )
    batch_manifest = payload["batch_manifest"]
    if not isinstance(batch_manifest, Mapping):
        raise RefinedDetectionDeltaStorageError(
            "partition batch_manifest must be an object."
        )
    arrays = {
        declaration.name: np.asarray(partition[declaration.name][:])
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
    try:
        reason_code_map = {
            int(code): str(label)
            for code, label in dict(batch_manifest["reason_code_map"]).items()
        }
        batch = RefinedDetectionDeltaBatch(
            delta_lineage_id=str(batch_manifest["delta_lineage_id"]),
            base_snapshot_id=str(batch_manifest["base_snapshot_id"]),
            base_manifest_digest=str(batch_manifest["base_manifest_digest"]),
            generation_ordinal=int(batch_manifest["generation_ordinal"]),
            partition_id=str(batch_manifest["partition_id"]),
            actor_id=str(batch_manifest["actor_id"]),
            reason_code_map=reason_code_map,
            arrays=arrays,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RefinedDetectionDeltaStorageError(
            "Persisted partition batch manifest or arrays are invalid."
        ) from exc
    if batch.as_manifest() != _json_tree(batch_manifest):
        raise RefinedDetectionDeltaStorageError(
            "Persisted batch manifest differs from reconstructed v2 state."
        )
    if (
        batch.delta_lineage_id != binding.delta_lineage_id
        or batch.base_snapshot_id != binding.base_snapshot_id
        or batch.base_manifest_digest != binding.base_manifest_digest
        or batch.generation_ordinal != generation_ordinal
        or batch.partition_id != partition_name
    ):
        raise RefinedDetectionDeltaStorageError(
            "Persisted partition path or lineage binding mismatch."
        )
    expected_payload = _partition_payload(
        batch,
        created_at_utc=str(payload["created_at_utc"]),
    )
    if _json_tree(payload) != expected_payload:
        raise RefinedDetectionDeltaStorageError(
            "Partition manifest differs from reconstructed content or storage policy."
        )
    for declaration in REFINED_DETECTION_DELTA_ARRAYS:
        name = declaration.name
        plan = _array_plan(name, batch.arrays[name])
        expected_metadata = array_metadata_declaration_from_plan(
            contract=_ARRAY_CONTRACTS[name],
            plan=plan,
            fill_value=False if batch.arrays[name].dtype == np.dtype("bool") else 0,
        )
        observed_metadata = {
            key: value
            for key, value in _observed_array_metadata(partition[name]).items()
            if key not in {"zarr_format", "node_type", "consolidated_metadata"}
        }
        if observed_metadata != expected_metadata:
            raise RefinedDetectionDeltaStorageError(
                f"Persisted array metadata differs from the storage plan: {name}."
            )
    return batch, MappingProxyType(_json_tree(manifest))


def _partition_generation_receipt(
    batch: RefinedDetectionDeltaBatch,
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    payload = manifest["payload"]
    return {
        "partition_id": batch.partition_id,
        "row_count": batch.row_count,
        "event_sequence_first": int(batch.arrays["event_sequence"][0]),
        "event_sequence_last": int(batch.arrays["event_sequence"][-1]),
        "partition_payload_digest": str(manifest["payload_digest"]),
        "partition_content_digest": str(payload["partition_content_digest"]),
    }


def _generation_content_document(
    *,
    binding: RefinedDetectionDeltaLineageBinding,
    generation_ordinal: int,
    partition_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "schema_id": "palette.refined_detection.delta_generation_content",
        "schema_version": 1,
        "delta_lineage_id": binding.delta_lineage_id,
        "base_snapshot_id": binding.base_snapshot_id,
        "base_manifest_digest": binding.base_manifest_digest,
        "generation_ordinal": generation_ordinal,
        "partitions": _json_tree(partition_receipts),
    }


def _require_unique_generation_sequences(
    batches: Sequence[RefinedDetectionDeltaBatch],
) -> None:
    sequences = [
        int(sequence)
        for batch in batches
        for sequence in batch.arrays["event_sequence"]
    ]
    if len(sequences) != len(set(sequences)):
        raise RefinedDetectionDeltaStorageError(
            "Frozen generation contains duplicate global event_sequence values."
        )


def freeze_refined_detection_delta_generation(
    root: Any,
    *,
    delta_lineage_id: str,
    generation_ordinal: int,
    frozen_by: str,
    frozen_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Freeze an open generation after rereading every immutable partition."""

    generation, binding = _generation_group(
        root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    current_manifest, current_payload = _read_generation_manifest(
        generation,
        binding=binding,
        generation_ordinal=generation_ordinal,
    )
    if current_payload["status"] == "frozen":
        read_frozen_refined_detection_delta_generation(
            root,
            delta_lineage_id=delta_lineage_id,
            generation_ordinal=generation_ordinal,
        )
        return current_manifest
    partition_names = _group_names(generation["partitions"])
    if not partition_names:
        raise RefinedDetectionDeltaStorageError(
            "Cannot freeze an empty refined-detection delta generation."
        )
    loaded = [
        read_refined_detection_delta_partition(
            root,
            delta_lineage_id=delta_lineage_id,
            generation_ordinal=generation_ordinal,
            partition_id=name,
        )
        for name in partition_names
    ]
    batches = tuple(batch for batch, _manifest_value in loaded)
    _require_unique_generation_sequences(batches)
    receipts = [
        _partition_generation_receipt(batch, manifest)
        for batch, manifest in loaded
    ]
    content_document = _generation_content_document(
        binding=binding,
        generation_ordinal=generation_ordinal,
        partition_receipts=receipts,
    )
    payload = _generation_payload(
        binding,
        generation_ordinal=generation_ordinal,
        created_at_utc=str(current_payload["created_at_utc"]),
        created_by=str(current_payload["created_by"]),
        status="frozen",
        previous_generation_ordinal=current_payload[
            "previous_generation_ordinal"
        ],
        previous_generation_manifest_digest=current_payload[
            "previous_generation_manifest_digest"
        ],
        minimum_event_sequence_exclusive=current_payload[
            "minimum_event_sequence_exclusive"
        ],
        frozen_at_utc=frozen_at_utc or _utc_now(),
        frozen_by=frozen_by,
        partition_receipts=receipts,
        generation_content_digest=canonical_json_sha256(content_document),
    )
    generation.attrs[_GENERATION_MANIFEST_ATTRIBUTE] = _manifest(payload)
    frozen = read_frozen_refined_detection_delta_generation(
        root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    return frozen.generation_manifest


def read_frozen_refined_detection_delta_generation(
    root: Any,
    *,
    delta_lineage_id: str,
    generation_ordinal: int,
) -> FrozenRefinedDetectionDeltaGeneration:
    """Load a frozen generation after recomputing every declared digest."""

    generation, binding = _generation_group(
        root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    manifest, payload = _read_generation_manifest(
        generation,
        binding=binding,
        generation_ordinal=generation_ordinal,
    )
    if payload["status"] != "frozen":
        raise RefinedDetectionDeltaStorageError(
            "Only frozen delta generations may be loaded for resolution."
        )
    entries = _generation_entries(_lineage_group(root, delta_lineage_id))
    earlier_ordinals = [
        ordinal for ordinal, _name in entries if ordinal < generation_ordinal
    ]
    previous_ordinal = payload["previous_generation_ordinal"]
    previous_generation: FrozenRefinedDetectionDeltaGeneration | None = None
    if earlier_ordinals:
        expected_previous = max(earlier_ordinals)
        if previous_ordinal != expected_previous:
            raise RefinedDetectionDeltaStorageError(
                "Frozen generation does not bind its immediate predecessor."
            )
        previous_generation = read_frozen_refined_detection_delta_generation(
            root,
            delta_lineage_id=delta_lineage_id,
            generation_ordinal=expected_previous,
        )
        if payload["previous_generation_manifest_digest"] != str(
            previous_generation.generation_manifest["payload_digest"]
        ):
            raise RefinedDetectionDeltaStorageError(
                "Frozen generation predecessor manifest digest mismatch."
            )
        expected_floor = max(
            int(sequence)
            for batch in previous_generation.batches
            for sequence in batch.arrays["event_sequence"]
        )
        if payload["minimum_event_sequence_exclusive"] != expected_floor:
            raise RefinedDetectionDeltaStorageError(
                "Frozen generation event-sequence floor mismatch."
            )
    elif (
        previous_ordinal is not None
        or payload["previous_generation_manifest_digest"] is not None
        or payload["minimum_event_sequence_exclusive"] != 0
    ):
        raise RefinedDetectionDeltaStorageError(
            "First frozen generation declares unexpected predecessor state."
        )
    receipts = payload["partition_receipts"]
    if not isinstance(receipts, list):
        raise RefinedDetectionDeltaStorageError(
            "partition_receipts must be an ordered array."
        )
    receipt_names = [str(receipt.get("partition_id")) for receipt in receipts]
    actual_names = _group_names(generation["partitions"])
    if receipt_names != sorted(receipt_names) or receipt_names != actual_names:
        raise RefinedDetectionDeltaStorageError(
            "Frozen generation partition membership differs from its manifest."
        )
    loaded = [
        read_refined_detection_delta_partition(
            root,
            delta_lineage_id=delta_lineage_id,
            generation_ordinal=generation_ordinal,
            partition_id=name,
        )
        for name in receipt_names
    ]
    expected_receipts = [
        _partition_generation_receipt(batch, partition_manifest)
        for batch, partition_manifest in loaded
    ]
    if receipts != expected_receipts:
        raise RefinedDetectionDeltaStorageError(
            "Frozen generation partition receipts do not match persisted partitions."
        )
    generation_batches = tuple(batch for batch, _partition_manifest in loaded)
    _require_unique_generation_sequences(generation_batches)
    sequence_floor = int(payload["minimum_event_sequence_exclusive"])
    if any(
        int(sequence) <= sequence_floor
        for batch in generation_batches
        for sequence in batch.arrays["event_sequence"]
    ):
        raise RefinedDetectionDeltaStorageError(
            "Frozen generation event sequence does not advance its predecessor."
        )
    content_document = _generation_content_document(
        binding=binding,
        generation_ordinal=generation_ordinal,
        partition_receipts=expected_receipts,
    )
    expected_digest = canonical_json_sha256(content_document)
    if payload["generation_content_digest"] != expected_digest:
        raise RefinedDetectionDeltaStorageError(
            "Frozen generation content digest mismatch."
        )
    batches = (
        generation_batches
        if previous_generation is None
        else (*previous_generation.batches, *generation_batches)
    )
    _require_unique_generation_sequences(batches)
    partition_manifests = (
        {}
        if previous_generation is None
        else dict(previous_generation.partition_manifests)
    )
    generation_name = _generation_name(generation_ordinal)
    partition_manifests.update(
        {
            f"{generation_name}/{batch.partition_id}": partition_manifest
            for batch, partition_manifest in loaded
        }
    )
    return FrozenRefinedDetectionDeltaGeneration(
        binding=binding,
        generation_ordinal=generation_ordinal,
        batches=batches,
        generation_manifest=MappingProxyType(_json_tree(manifest)),
        partition_manifests=MappingProxyType(partition_manifests),
    )


__all__ = [
    "REFINED_DETECTION_DELTA_CONTENT_DIGEST_ALGORITHM",
    "REFINED_DETECTION_DELTA_GENERATION_DIGEST_ALGORITHM",
    "REFINED_DETECTION_DELTA_GENERATION_SCHEMA_ID",
    "REFINED_DETECTION_DELTA_GENERATION_SCHEMA_VERSION",
    "REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_ID",
    "REFINED_DETECTION_DELTA_LINEAGE_SCHEMA_VERSION",
    "REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION",
    "REFINED_DETECTION_DELTA_PARENT",
    "REFINED_DETECTION_DELTA_PARTITION_SCHEMA_ID",
    "REFINED_DETECTION_DELTA_PARTITION_SCHEMA_VERSION",
    "FrozenRefinedDetectionDeltaGeneration",
    "RefinedDetectionDeltaLineageBinding",
    "RefinedDetectionDeltaStorageError",
    "create_refined_detection_delta_generation",
    "create_refined_detection_delta_lineage",
    "freeze_refined_detection_delta_generation",
    "read_frozen_refined_detection_delta_generation",
    "read_refined_detection_delta_partition",
    "refined_detection_delta_array_digest",
    "write_refined_detection_delta_partition",
]

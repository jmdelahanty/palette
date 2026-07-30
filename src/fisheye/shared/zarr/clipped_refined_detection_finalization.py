"""Finalize strict clip-local refined detections into one recording snapshot.

Clip workers may compute independently, but the recording-level publisher owns
row ordering, both CSR indexes, physical storage planning, and immutable
publication.  This module is deliberately strict about identity: clip results
must already carry recording-stable ``instance_key`` values and globally
allocated ``refined_row_ids``.  The finalizer never invents or silently rebases
identities while joining clips.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_logical_content_digest,
)
from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
    RefinedDetectionClipSourceIdentity,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceCollectionIdentity,
    RefinedDetectionSourceIdentity,
    refined_detection_dimensions_from_manifest,
    validate_refined_detection_run_manifest,
    validate_refined_detection_snapshot_identity,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionClippedBinding,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    RefinedDetectionSnapshotPublication,
    publish_selector_ineligible_refined_detection_snapshot,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)


CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID = (
    "palette.clipped_refined_detection.finalization"
)
CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_VERSION = 1
CLIPPED_REFINED_DETECTION_FINALIZATION_RECEIPT_NAME = "finalization_receipt.json"


class ClippedRefinedDetectionFinalizationError(ValueError):
    """Raised when clip evidence cannot form one strict recording snapshot."""


@dataclass(frozen=True)
class PreparedClippedRefinedDetectionSnapshot:
    """Logical recording snapshot and every authority needed to publish it."""

    dimensions: RefinedDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    clipped_binding: RefinedDetectionClippedBinding
    source_collection: RefinedDetectionSourceCollectionIdentity
    clipped_source_evidence: tuple[RefinedDetectionBoundClipEvidence, ...]
    instance_reason_codes: Mapping[int, str]
    source_reason_codes: Mapping[int, str]
    canonical_source: RefinedDetectionSourceIdentity


@dataclass(frozen=True)
class ClippedRefinedDetectionPublication:
    """Published recording snapshot and canonical-pair provenance receipt."""

    snapshot: RefinedDetectionSnapshotPublication
    receipt_path: Path
    receipt: Mapping[str, object]


def _values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _reason_codes(manifest: Mapping[str, Any], name: str) -> dict[int, str]:
    payload = manifest.get("payload")
    registries = (
        payload.get("reason_registries") if isinstance(payload, Mapping) else None
    )
    registry = registries.get(name) if isinstance(registries, Mapping) else None
    codes = registry.get("codes") if isinstance(registry, Mapping) else None
    if not isinstance(codes, Mapping):
        raise ClippedRefinedDetectionFinalizationError(
            f"Clip manifest lacks the {name} reason registry."
        )
    try:
        return {int(code): str(label) for code, label in codes.items()}
    except (TypeError, ValueError) as exc:
        raise ClippedRefinedDetectionFinalizationError(
            f"Clip {name} reason registry is invalid."
        ) from exc


def _merged_reason_registry(
    manifests: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> dict[int, str]:
    """Build one deterministic recording registry from clip-local labels."""

    labels: set[str] = set()
    for manifest in manifests:
        registry = _reason_codes(manifest, name)
        if registry.get(0) != "none":
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {name} reason registry must reserve code zero for 'none'."
            )
        labels.update(label for code, label in registry.items() if code != 0)
    ordered = sorted(labels)
    if len(ordered) > int(np.iinfo(np.uint16).max):
        raise ClippedRefinedDetectionFinalizationError(
            f"Merged {name} reason registry exceeds the uint16 domain."
        )
    return {0: "none", **{index + 1: label for index, label in enumerate(ordered)}}


def _remap_reason_codes(
    values: Any,
    *,
    source_registry: Mapping[int, str],
    destination_registry: Mapping[int, str],
    label: str,
) -> np.ndarray:
    """Translate one clip's codes into the recording-level registry."""

    source = np.asarray(_values(values), dtype=np.uint16).reshape(-1)
    destination_by_label = {
        reason: code for code, reason in destination_registry.items()
    }
    unknown = sorted(
        int(code)
        for code in np.unique(source).tolist()
        if int(code) not in source_registry
    )
    if unknown:
        raise ClippedRefinedDetectionFinalizationError(
            f"{label} contains unregistered reason codes {unknown!r}."
        )
    return np.asarray(
        [destination_by_label[source_registry[int(code)]] for code in source],
        dtype=np.uint16,
    )


def _source_identity(manifest: Mapping[str, Any]) -> RefinedDetectionSourceIdentity:
    payload = manifest.get("payload")
    source = payload.get("source_detection") if isinstance(payload, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("authority_kind") != "canonical_run"
    ):
        raise ClippedRefinedDetectionFinalizationError(
            "Each clip must bind one strict canonical raw-detection run."
        )
    return RefinedDetectionSourceIdentity(
        run_id=source.get("run_id"),
        run_manifest_digest=source.get("run_manifest_digest"),
        logical_content_digest=source.get("logical_content_digest"),
    )


def _offsets(frames: np.ndarray, *, n_frames: int) -> np.ndarray:
    counts = np.bincount(
        np.asarray(frames, dtype=np.int64),
        minlength=n_frames,
    )
    offsets = np.zeros(n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def _concat(parts: list[np.ndarray], *, dtype: np.dtype[Any]) -> np.ndarray:
    if not parts:
        return np.empty((0,), dtype=dtype)
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=dtype)


def _require_canonical_pair(
    arrays: Mapping[str, np.ndarray],
    *,
    canonical_arrays: Mapping[str, Any],
    canonical_dimensions: CanonicalDetectionDimensions,
    canonical_source: RefinedDetectionSourceIdentity,
) -> None:
    canonical_digest = canonical_detection_logical_content_digest(
        canonical_arrays,
        dimensions=canonical_dimensions,
    )
    if canonical_digest != canonical_source.logical_content_digest:
        raise ClippedRefinedDetectionFinalizationError(
            "Canonical arrays do not match the bound canonical logical digest."
        )
    comparisons = (
        "frame_indices",
        "source_acquisition_frame_index",
        "instance_key",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
        "scores",
        "class_ids",
        "frame_row_offsets",
    )
    for name in comparisons:
        refined = arrays[f"source_detections/{name}"]
        canonical_path = f"instances/{name}"
        if canonical_path not in canonical_arrays:
            raise ClippedRefinedDetectionFinalizationError(
                f"Canonical pair lacks {canonical_path!r}."
            )
        if not np.array_equal(refined, _values(canonical_arrays[canonical_path])):
            raise ClippedRefinedDetectionFinalizationError(
                f"Recording source-audit rows differ from canonical {canonical_path}."
            )


def prepare_clipped_refined_detection_snapshot(
    evidence: Sequence[RefinedDetectionBoundClipEvidence],
    *,
    clipped_binding: RefinedDetectionClippedBinding,
    canonical_arrays: Mapping[str, Any],
    canonical_dimensions: CanonicalDetectionDimensions,
    canonical_source: RefinedDetectionSourceIdentity,
    recording_identity: str,
) -> PreparedClippedRefinedDetectionSnapshot:
    """Join all clip snapshots and prove the recording-level canonical pair."""

    ordered = tuple(evidence)
    if tuple(item.clip_index for item in ordered) != tuple(
        range(len(clipped_binding.clips))
    ):
        raise ClippedRefinedDetectionFinalizationError(
            "Clip evidence must contain every bound clip exactly once in order."
        )
    recording = str(recording_identity).strip()
    if not recording:
        raise ClippedRefinedDetectionFinalizationError(
            "recording_identity cannot be empty."
        )

    instance_parts: dict[str, list[np.ndarray]] = {}
    source_parts: dict[str, list[np.ndarray]] = {}
    members: list[RefinedDetectionClipSourceIdentity] = []
    instance_registry = _merged_reason_registry(
        [item.manifest for item in ordered],
        name="instances",
    )
    source_registry = _merged_reason_registry(
        [item.manifest for item in ordered],
        name="source_detections",
    )
    source_width: int | None = None
    source_height: int | None = None
    source_row_cursor = 0

    def append(target: dict[str, list[np.ndarray]], name: str, value: Any) -> None:
        target.setdefault(name, []).append(np.ascontiguousarray(_values(value)))

    for clip, item in zip(clipped_binding.clips, ordered, strict=True):
        manifest_errors = validate_refined_detection_run_manifest(item.manifest)
        if manifest_errors:
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} manifest is invalid: "
                + "; ".join(manifest_errors)
            )
        identity_errors = validate_refined_detection_snapshot_identity(
            manifest=item.manifest,
            arrays=item.arrays,
            parent_manifest=item.parent_manifest,
            parent_arrays=item.parent_arrays,
        )
        if identity_errors:
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} identity is invalid: "
                + "; ".join(identity_errors)
            )
        dimensions = refined_detection_dimensions_from_manifest(item.manifest)
        if (
            dimensions.lineage_profile
            is not RefinedDetectionLineageProfile.FULL_ACQUISITION
        ):
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} is not a full-acquisition clip snapshot."
            )
        if dimensions.n_frames != clip.frame_count:
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} frame count differs from its binding."
            )
        if source_width is None:
            source_width, source_height = (
                dimensions.source_width,
                dimensions.source_height,
            )
        elif (source_width, source_height) != (
            dimensions.source_width,
            dimensions.source_height,
        ):
            raise ClippedRefinedDetectionFinalizationError(
                "Clip source dimensions are not identical."
            )
        clip_payload = item.manifest["payload"]
        if clip_payload["run_id"] != clip.source_refined_run_id or (
            item.manifest["payload_digest"] != clip.source_refined_manifest_digest
        ):
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} manifest differs from its binding."
            )
        clip_recording = clip_payload["snapshot_lineage"][
            "manual_instance_key_allocator"
        ]["recording_identity"]
        if clip_recording != recording:
            raise ClippedRefinedDetectionFinalizationError(
                f"Clip {clip.clip_index} uses a different recording identity."
            )
        observed_instance_registry = _reason_codes(item.manifest, "instances")
        observed_source_registry = _reason_codes(item.manifest, "source_detections")

        raw_source = _source_identity(item.manifest)
        members.append(
            RefinedDetectionClipSourceIdentity(
                clip_index=clip.clip_index,
                source_refined_run_id=clip.source_refined_run_id,
                source_refined_manifest_digest=clip.source_refined_manifest_digest,
                source_detection=raw_source,
            )
        )

        local_instance_frames = _values(item.arrays["instances/frame_indices"])
        parent_instance_frames = (
            local_instance_frames.astype(np.int64, copy=False) + clip.parent_frame_start
        ).astype(np.int32, copy=False)
        local_source_frames = _values(item.arrays["source_detections/frame_indices"])
        parent_source_frames = (
            local_source_frames.astype(np.int64, copy=False) + clip.parent_frame_start
        ).astype(np.int32, copy=False)
        local_instance_source_rows = _values(
            item.arrays["instances/source_detect_row_index"]
        ).astype(np.int64, copy=False)
        global_instance_source_rows = np.where(
            local_instance_source_rows < 0,
            local_instance_source_rows,
            local_instance_source_rows + source_row_cursor,
        ).astype(np.int64, copy=False)

        for name in (
            "instance_key",
            "refined_row_ids",
            "bbox_norm_coords",
            "bbox_img_xyxy",
            "centers_img_xy",
            "scores",
            "score_valid",
            "class_ids",
            "source_kind_codes",
            "manual_edit_flags",
        ):
            append(instance_parts, name, item.arrays[f"instances/{name}"])
        append(
            instance_parts,
            "reason_codes",
            _remap_reason_codes(
                item.arrays["instances/reason_codes"],
                source_registry=observed_instance_registry,
                destination_registry=instance_registry,
                label=f"clip {clip.clip_index} instances/reason_codes",
            ),
        )
        append(instance_parts, "frame_indices", parent_instance_frames)
        append(
            instance_parts,
            "source_acquisition_frame_index",
            parent_instance_frames.astype(np.int64),
        )
        append(instance_parts, "source_detect_row_index", global_instance_source_rows)
        append(instance_parts, "source_recording_frame_ids", parent_instance_frames + 1)
        append(
            instance_parts,
            "source_clip_indices",
            np.full(local_instance_frames.shape, clip.clip_index, dtype=np.int32),
        )
        append(
            instance_parts,
            "source_clip_local_frame_indices",
            local_instance_frames,
        )
        append(
            instance_parts,
            "source_clip_detect_row_index",
            local_instance_source_rows,
        )
        append(
            instance_parts,
            "source_refined_row_ids",
            item.arrays["instances/refined_row_ids"],
        )

        for name in (
            "instance_key",
            "bbox_norm_coords",
            "bbox_img_xyxy",
            "centers_img_xy",
            "scores",
            "class_ids",
            "decision_codes",
            "resolved_refined_row_id",
        ):
            append(source_parts, name, item.arrays[f"source_detections/{name}"])
        append(
            source_parts,
            "reason_codes",
            _remap_reason_codes(
                item.arrays["source_detections/reason_codes"],
                source_registry=observed_source_registry,
                destination_registry=source_registry,
                label=f"clip {clip.clip_index} source_detections/reason_codes",
            ),
        )
        local_source_rows = _values(
            item.arrays["source_detections/source_detect_row_index"]
        ).astype(np.int64, copy=False)
        append(
            source_parts,
            "source_detect_row_index",
            local_source_rows + source_row_cursor,
        )
        append(source_parts, "frame_indices", parent_source_frames)
        append(
            source_parts,
            "source_acquisition_frame_index",
            parent_source_frames.astype(np.int64),
        )
        append(source_parts, "source_recording_frame_ids", parent_source_frames + 1)
        append(
            source_parts,
            "source_clip_indices",
            np.full(local_source_frames.shape, clip.clip_index, dtype=np.int32),
        )
        append(
            source_parts,
            "source_clip_local_frame_indices",
            local_source_frames,
        )
        append(source_parts, "source_clip_detect_row_index", local_source_rows)
        append(
            source_parts,
            "source_resolved_refined_row_id",
            item.arrays["source_detections/resolved_refined_row_id"],
        )
        source_row_cursor += dimensions.n_source_detections

    assert source_width is not None and source_height is not None
    instance_count = sum(part.shape[0] for part in instance_parts["frame_indices"])
    source_count = source_row_cursor
    dimensions = RefinedDetectionDimensions(
        n_frames=clipped_binding.n_frames,
        n_instances=instance_count,
        n_source_detections=source_count,
        source_width=source_width,
        source_height=source_height,
        lineage_profile=RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT,
    )
    binding_by_path = {
        binding.path: binding
        for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions)
    }
    arrays: dict[str, np.ndarray] = {}
    for path, binding in binding_by_path.items():
        group, name = path.split("/", 1)
        contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        dtype = contract.dtype.numpy_dtype
        if name == "frame_row_offsets":
            frame_name = f"{group}/frame_indices"
            frames = arrays[frame_name]
            arrays[path] = _offsets(frames, n_frames=dimensions.n_frames)
            continue
        parts = instance_parts if group == "instances" else source_parts
        arrays[path] = _concat(parts[name], dtype=dtype)

    row_ids = arrays["instances/refined_row_ids"]
    if np.unique(row_ids).shape[0] != row_ids.shape[0]:
        raise ClippedRefinedDetectionFinalizationError(
            "Clip refined_row_ids overlap; workers must use one recording-global "
            "allocator before finalization."
        )
    REFINED_DETECTION_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        clipped_binding=clipped_binding,
    )
    _require_canonical_pair(
        arrays,
        canonical_arrays=canonical_arrays,
        canonical_dimensions=canonical_dimensions,
        canonical_source=canonical_source,
    )
    frozen_arrays: dict[str, np.ndarray] = {}
    for path, values in arrays.items():
        copied = np.array(values, copy=True, order="C")
        copied.setflags(write=False)
        frozen_arrays[path] = copied
    return PreparedClippedRefinedDetectionSnapshot(
        dimensions=dimensions,
        arrays=MappingProxyType(frozen_arrays),
        clipped_binding=clipped_binding,
        source_collection=RefinedDetectionSourceCollectionIdentity(
            collection_id=clipped_binding.collection_id,
            collection_manifest_digest=clipped_binding.collection_manifest_digest,
            members=tuple(members),
        ),
        clipped_source_evidence=ordered,
        instance_reason_codes=MappingProxyType(dict(instance_registry)),
        source_reason_codes=MappingProxyType(dict(source_registry)),
        canonical_source=canonical_source,
    )


def validate_clipped_refined_detection_finalization_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the immutable recording-level pair receipt."""

    errors: list[str] = []
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("clipped refined finalization receipt has unexpected fields")
    if receipt.get("schema_id") != CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID:
        errors.append("clipped refined finalization receipt schema_id mismatch")
    if (
        receipt.get("schema_version")
        != CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_VERSION
    ):
        errors.append("clipped refined finalization receipt schema_version mismatch")
    if receipt.get("digest_algorithm") != "sha256_canonical_json_v1":
        errors.append("clipped refined finalization digest algorithm mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "clipped refined finalization payload must be an object")
    if receipt.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("clipped refined finalization payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("clipped refined finalization status is not complete")
    if payload.get("selector_eligible") is not False:
        errors.append("clipped refined finalization became selector eligible")
    if payload.get("registry_registered") is not False:
        errors.append("clipped refined finalization became registry registered")
    if payload.get("production_state_changes") != []:
        errors.append("clipped refined finalization reports production-state changes")
    if not isinstance(payload.get("canonical_detection"), Mapping):
        errors.append("clipped refined finalization lacks canonical detection binding")
    if not isinstance(payload.get("refined_detection"), Mapping):
        errors.append("clipped refined finalization lacks refined detection binding")
    clips = payload.get("clip_sources")
    if not isinstance(clips, list) or not clips:
        errors.append("clipped refined finalization lacks ordered clip sources")
    return tuple(dict.fromkeys(errors))


def publish_selector_ineligible_clipped_refined_detection_snapshot(
    prepared: PreparedClippedRefinedDetectionSnapshot,
    *,
    destination: Path,
    run_id: str,
    safe_root: Path,
    lineage_id: str,
    snapshot_id: str,
    recording_identity: str,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    created_by: str = "clipped_refined_detection_finalizer",
) -> ClippedRefinedDetectionPublication:
    """Publish a recording-level refined snapshot without selector mutation."""

    row_ids = np.asarray(prepared.arrays["instances/refined_row_ids"])
    next_row_id = 0 if row_ids.size == 0 else int(np.max(row_ids)) + 1
    snapshot = publish_selector_ineligible_refined_detection_snapshot(
        dimensions=prepared.dimensions,
        arrays=prepared.arrays,
        instance_reason_codes=prepared.instance_reason_codes,
        source_reason_codes=prepared.source_reason_codes,
        destination=destination,
        run_id=run_id,
        lineage=RefinedDetectionSnapshotLineage(
            lineage_id=lineage_id,
            snapshot_id=snapshot_id,
            recording_identity=recording_identity,
            next_refined_row_id=next_row_id,
        ),
        source=prepared.source_collection,
        created_by=created_by,
        publication_kind="clipped_recording_snapshot",
        safe_root=safe_root,
        profile=profile,
        clipped_binding=prepared.clipped_binding,
        clipped_source_evidence=prepared.clipped_source_evidence,
        run_attributes={
            "canonical_recording_source": prepared.canonical_source.as_manifest(),
            "finalization_partition": "complete_recording_snapshot",
        },
        selection_contract="none_selector_ineligible_direct_path_only",
        coordinate_catalog=True,
    )
    payload: dict[str, object] = {
        "status": "complete",
        "selector_eligible": False,
        "registry_registered": False,
        "canonical_detection": prepared.canonical_source.as_manifest(),
        "refined_detection": {
            "run_id": snapshot.run_id,
            "output_path": str(snapshot.output_path),
            "run_manifest_digest": snapshot.manifest["payload_digest"],
            "logical_content_digest": snapshot.receipt["logical_content_digest"],
            "storage_profile_id": profile.profile_id,
        },
        "clip_sources": [
            member.as_manifest() for member in prepared.source_collection.members
        ],
        "selector_activation": "none_direct_path_only",
        "production_state_changes": [],
    }
    receipt: dict[str, object] = {
        "schema_id": CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID,
        "schema_version": CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    errors = validate_clipped_refined_detection_finalization_receipt(receipt)
    if errors:
        raise ClippedRefinedDetectionFinalizationError(
            "Finalization receipt is invalid: " + "; ".join(errors)
        )
    receipt_path = (
        snapshot.output_path / CLIPPED_REFINED_DETECTION_FINALIZATION_RECEIPT_NAME
    )
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(
            receipt,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\n")
    return ClippedRefinedDetectionPublication(
        snapshot=snapshot,
        receipt_path=receipt_path,
        receipt=receipt,
    )


__all__ = [
    "CLIPPED_REFINED_DETECTION_FINALIZATION_RECEIPT_NAME",
    "CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_ID",
    "CLIPPED_REFINED_DETECTION_FINALIZATION_SCHEMA_VERSION",
    "ClippedRefinedDetectionFinalizationError",
    "ClippedRefinedDetectionPublication",
    "PreparedClippedRefinedDetectionSnapshot",
    "prepare_clipped_refined_detection_snapshot",
    "publish_selector_ineligible_clipped_refined_detection_snapshot",
    "validate_clipped_refined_detection_finalization_receipt",
]

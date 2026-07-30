"""Strict selector-ineligible evidence for one clipped detection work unit.

The maintained clipped detector/refiner still writes compatibility groups into
the recording archive.  This boundary converts those complete groups into a
canonical/refined v1 pair under a disposable or benchmark namespace, proves
the pair against the recording-level native canonical run, and never updates a
selector or registry.

The first adoption intentionally accepts automated raw-backed refinement only.
Manual additions require the recording-level delta allocator and compactor;
they are rejected here rather than receiving clip-local identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.canonical_detection_manifest import (
    canonical_detection_dimensions_from_manifest,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    canonical_detection_metadata_declaration_maps,
    publish_legacy_canonical_detection_shadow,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_KIND_CODE_MAP,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    RefinedDetectionShadowPublication,
    publish_refined_detection_shadow,
)
from fisheye.shared.zarr.refined_detection_transition import (
    RefinedDetectionTransitionResult,
    build_refined_detection_transition,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_io import open_zarr_root


STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_ID = "palette.clipped_detection.strict_evidence"
STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NativeCanonicalClipMember:
    """The exact row/frame interval assigned to one native clip artifact."""

    clip_id: str
    clip_index: int
    artifact_run_id: str
    parent_frame_start: int
    parent_frame_stop: int
    canonical_row_start: int
    canonical_row_stop: int

    @property
    def frame_count(self) -> int:
        return self.parent_frame_stop - self.parent_frame_start

    @property
    def row_count(self) -> int:
        return self.canonical_row_stop - self.canonical_row_start


@dataclass(frozen=True)
class StrictClipDetectionEvidencePublication:
    """A strict clip pair and the receipt binding it to recording authority."""

    canonical: CanonicalDetectionShadowPublication
    refined: RefinedDetectionShadowPublication
    member: NativeCanonicalClipMember
    receipt_path: Path
    receipt: Mapping[str, object]


def _canonical_run(
    archive: Path,
    *,
    run_id: str,
) -> tuple[Any, Mapping[str, Any], Any, Mapping[str, Any]]:
    root = open_zarr_root(archive.expanduser().resolve(), mode="r")
    run = root[f"detect_runs/{run_id}"]
    manifest = run.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Recording canonical run lacks its run_manifest.")
    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    arrays = {path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths}
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_canonical_detection_storage(dimensions, profile=profile)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        archive.expanduser().resolve(),
        run_id=run_id,
        plans=plans,
    )
    errors = validate_canonical_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
    )
    if errors:
        raise ValueError("Recording canonical run is invalid: " + "; ".join(errors))
    return dimensions, manifest, arrays, root


def _native_members(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = manifest.get("payload")
    source = payload.get("source_evidence") if isinstance(payload, Mapping) else None
    provenance = source.get("run_provenance") if isinstance(source, Mapping) else None
    document = provenance.get("document") if isinstance(provenance, Mapping) else None
    binding = (
        document.get("clipped_detection_binding")
        if isinstance(document, Mapping)
        else None
    )
    binding_document = binding.get("document") if isinstance(binding, Mapping) else None
    members = (
        binding_document.get("members")
        if isinstance(binding_document, Mapping)
        else None
    )
    if (
        not isinstance(members, list)
        or not members
        or not all(isinstance(item, Mapping) for item in members)
    ):
        raise ValueError(
            "Recording canonical run does not carry native clipped member evidence."
        )
    return tuple(members)


def _resolve_member(
    manifest: Mapping[str, Any],
    *,
    clip_id: str,
    clip_index: int,
    source_detect_run_id: str,
) -> NativeCanonicalClipMember:
    matches = [
        item
        for item in _native_members(manifest)
        if item.get("clip_id") == clip_id and item.get("clip_index") == clip_index
    ]
    if len(matches) != 1:
        raise ValueError(
            "Recording canonical evidence must contain exactly one matching clip."
        )
    raw = matches[0]
    if raw.get("artifact_run_id") != source_detect_run_id:
        raise ValueError(
            "Clip source detect run differs from the native canonical member."
        )
    member = NativeCanonicalClipMember(
        clip_id=str(raw["clip_id"]),
        clip_index=int(raw["clip_index"]),
        artifact_run_id=str(raw["artifact_run_id"]),
        parent_frame_start=int(raw["parent_frame_start"]),
        parent_frame_stop=int(raw["parent_frame_stop"]),
        canonical_row_start=int(raw["canonical_row_start"]),
        canonical_row_stop=int(raw["canonical_row_stop"]),
    )
    if (
        member.parent_frame_start < 0
        or member.parent_frame_stop <= member.parent_frame_start
        or member.canonical_row_start < 0
        or member.canonical_row_stop < member.canonical_row_start
    ):
        raise ValueError("Native canonical clip member has invalid intervals.")
    return member


def _values(value: Any) -> np.ndarray:
    try:
        return np.asarray(value[...])
    except (IndexError, TypeError):
        return np.asarray(value)


def _validate_clip_canonical_slice(
    clip: CanonicalDetectionShadowPublication,
    *,
    recording_dimensions: Any,
    recording_arrays: Mapping[str, Any],
    member: NativeCanonicalClipMember,
) -> None:
    if clip.dimensions.n_frames != member.frame_count:
        raise ValueError("Clip canonical frame count differs from native evidence.")
    if clip.dimensions.n_instances != member.row_count:
        raise ValueError("Clip canonical row count differs from native evidence.")
    if (
        clip.dimensions.source_width != recording_dimensions.source_width
        or clip.dimensions.source_height != recording_dimensions.source_height
    ):
        raise ValueError("Clip and recording canonical pixel extents differ.")
    row_slice = slice(member.canonical_row_start, member.canonical_row_stop)
    for name in (
        "frame_indices",
        "source_acquisition_frame_index",
        "instance_key",
        "bbox_norm_coords",
        "bbox_img_xyxy",
        "centers_img_xy",
        "scores",
        "class_ids",
    ):
        local = _values(clip.arrays[f"instances/{name}"])
        if name in {"frame_indices", "source_acquisition_frame_index"}:
            local = local.astype(np.int64, copy=False) + member.parent_frame_start
        recording = _values(recording_arrays[f"instances/{name}"])[row_slice]
        if not np.array_equal(local, recording):
            raise ValueError(
                f"Clip canonical values differ from recording canonical {name}."
            )
    local_offsets = _values(clip.arrays["instances/frame_row_offsets"]).astype(
        np.int64,
        copy=False,
    )
    recording_offsets = _values(recording_arrays["instances/frame_row_offsets"]).astype(
        np.int64, copy=False
    )
    expected = (
        recording_offsets[member.parent_frame_start : member.parent_frame_stop + 1]
        - member.canonical_row_start
    )
    if not np.array_equal(local_offsets, expected):
        raise ValueError(
            "Clip frame_row_offsets differ from the recording canonical interval."
        )


def allocate_automated_clip_refined_ids(
    transition: RefinedDetectionTransitionResult,
    *,
    canonical_row_start: int,
    recording_source_row_count: int,
) -> RefinedDetectionTransitionResult:
    """Allocate raw-backed refined IDs from recording canonical row positions.

    This is deliberately narrower than the future manual-edit allocator.  It
    refuses manual rows and uses the already frozen canonical source row order,
    making allocation deterministic across clip worker scheduling.
    """

    arrays = {
        path: np.ascontiguousarray(_values(value))
        for path, value in transition.arrays.items()
    }
    kinds = arrays["instances/source_kind_codes"].astype(np.uint8, copy=False)
    manual_flags = arrays["instances/manual_edit_flags"].astype(bool, copy=False)
    if np.any(kinds != SOURCE_KIND_CODE_MAP["raw_detect"]) or np.any(manual_flags):
        raise ValueError(
            "Strict clip adoption accepts automated raw-backed rows only; manual "
            "rows require the recording-level delta allocator and compactor."
        )
    source_rows = arrays["instances/source_detect_row_index"].astype(
        np.int64,
        copy=False,
    )
    if source_rows.size and (
        np.any(source_rows < 0)
        or np.any(source_rows >= transition.dimensions.n_source_detections)
        or np.unique(source_rows).shape[0] != source_rows.shape[0]
    ):
        raise ValueError(
            "Automated refined instances must reference unique local source rows."
        )
    start = int(canonical_row_start)
    stop = start + int(transition.dimensions.n_source_detections)
    total = int(recording_source_row_count)
    if start < 0 or stop > total:
        raise ValueError(
            "Clip source row allocation falls outside recording authority."
        )
    refined_ids = (start + source_rows).astype(np.int64, copy=False)
    resolved = np.full(
        transition.dimensions.n_source_detections,
        -1,
        dtype=np.int64,
    )
    resolved[source_rows] = refined_ids
    arrays["instances/refined_row_ids"] = np.ascontiguousarray(refined_ids)
    arrays["source_detections/resolved_refined_row_id"] = resolved
    REFINED_DETECTION_SCHEMA_V1.require(
        arrays,
        dimensions=transition.dimensions,
    )
    report = dict(transition.report)
    report["identity_allocation"] = {
        "policy": "recording_canonical_source_row_position_v1",
        "manual_rows_allowed": False,
        "canonical_source_row_start": start,
        "canonical_source_row_stop": stop,
        "recording_next_refined_row_id": total,
    }
    return RefinedDetectionTransitionResult(
        dimensions=transition.dimensions,
        arrays=MappingProxyType(arrays),
        instance_reason_codes=transition.instance_reason_codes,
        source_reason_codes=transition.source_reason_codes,
        report=MappingProxyType(report),
    )


def publish_strict_clip_detection_evidence(
    *,
    analysis_zarr: Path,
    source_detect_group_path: str,
    source_refined_group_path: str,
    recording_canonical_archive: Path,
    recording_canonical_run_id: str,
    recording_identity: str,
    clip_id: str,
    clip_index: int,
    output_root: Path,
    canonical_run_id: str,
    refined_run_id: str,
    coordinate_catalog: bool = True,
) -> StrictClipDetectionEvidencePublication:
    """Publish and validate one immutable strict clip evidence pair."""

    source_detect_path = (
        analysis_zarr.expanduser().resolve() / source_detect_group_path.strip("/")
    )
    source_refined_path = (
        analysis_zarr.expanduser().resolve() / source_refined_group_path.strip("/")
    )
    if not source_detect_path.is_dir() or not source_refined_path.is_dir():
        raise FileNotFoundError(
            "Clip detect/refined compatibility source groups are missing."
        )
    recording_dimensions, recording_manifest, recording_arrays, _ = _canonical_run(
        recording_canonical_archive,
        run_id=recording_canonical_run_id,
    )
    manifest_recording = recording_manifest["payload"]["source_evidence"].get(
        "recording_identity"
    )
    if manifest_recording != recording_identity:
        raise ValueError("Recording canonical identity differs from the clip request.")
    member = _resolve_member(
        recording_manifest,
        clip_id=clip_id,
        clip_index=int(clip_index),
        source_detect_run_id=Path(source_detect_group_path).name,
    )

    root = output_root.expanduser().resolve()
    clip_root = root / f"clip_{int(clip_index):06d}_{clip_id}"
    canonical = publish_legacy_canonical_detection_shadow(
        source_group_path=source_detect_path,
        recording_identity=recording_identity,
        source_run_id=Path(source_detect_group_path).name,
        destination=clip_root / "canonical.zarr",
        run_id=canonical_run_id,
        shadow_root=root,
        coordinate_catalog=coordinate_catalog,
    )
    _validate_clip_canonical_slice(
        canonical,
        recording_dimensions=recording_dimensions,
        recording_arrays=recording_arrays,
        member=member,
    )

    transition = build_refined_detection_transition(
        zarr.open_group(
            str(source_refined_path),
            mode="r",
            use_consolidated=False,
        ),
        n_frames=canonical.dimensions.n_frames,
        source_width=canonical.dimensions.source_width,
        source_height=canonical.dimensions.source_height,
        recording_identity=recording_identity,
        source_detect_group=zarr.open_group(
            str(source_detect_path),
            mode="r",
            use_consolidated=False,
        ),
    )
    allocated = allocate_automated_clip_refined_ids(
        transition,
        canonical_row_start=member.canonical_row_start,
        recording_source_row_count=recording_dimensions.n_instances,
    )
    lineage = RefinedDetectionSnapshotLineage(
        lineage_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"palette:clip-refined-lineage:{recording_identity}:{clip_index}",
            )
        ),
        snapshot_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "palette:clip-refined-snapshot:"
                f"{recording_identity}:{clip_index}:{refined_run_id}",
            )
        ),
        recording_identity=recording_identity,
        next_refined_row_id=recording_dimensions.n_instances,
    )
    refined = publish_refined_detection_shadow(
        allocated,
        destination=clip_root / "refined.zarr",
        run_id=refined_run_id,
        lineage=lineage,
        canonical_source=canonical,
        shadow_root=root,
        coordinate_catalog=coordinate_catalog,
    )
    receipt_path = clip_root / "strict_detection_evidence_receipt.json"
    receipt: dict[str, object] = {
        "schema_id": STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_ID,
        "schema_version": STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "selector_eligible": False,
        "registry_updated": False,
        "recording_identity": recording_identity,
        "recording_canonical": {
            "archive": str(recording_canonical_archive.expanduser().resolve()),
            "run_id": recording_canonical_run_id,
            "manifest_digest": recording_manifest["payload_digest"],
        },
        "clip": {
            "clip_id": member.clip_id,
            "clip_index": member.clip_index,
            "parent_frame_start": member.parent_frame_start,
            "parent_frame_stop": member.parent_frame_stop,
            "canonical_row_start": member.canonical_row_start,
            "canonical_row_stop": member.canonical_row_stop,
        },
        "sources": {
            "detect_group_path": source_detect_group_path.strip("/"),
            "refined_group_path": source_refined_group_path.strip("/"),
        },
        "canonical": {
            "archive": str(canonical.output_path),
            "run_id": canonical.run_id,
            "manifest_digest": canonical.manifest["payload_digest"],
            "storage_profile_id": canonical.plans.profile.profile_id,
        },
        "refined": {
            "archive": str(refined.output_path),
            "run_id": refined.run_id,
            "manifest_digest": refined.manifest["payload_digest"],
            "storage_profile_id": refined.manifest["payload"]["storage_plan"][
                "storage_profile"
            ]["profile_id"],
        },
        "identity_allocation": dict(allocated.report["identity_allocation"]),
        "manual_rows_allowed": False,
        "physical_layout_source": "shared_versioned_byte_planners",
    }
    write_json_atomic(receipt_path, receipt)
    return StrictClipDetectionEvidencePublication(
        canonical=canonical,
        refined=refined,
        member=member,
        receipt_path=receipt_path,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_ID",
    "STRICT_CLIP_DETECTION_EVIDENCE_SCHEMA_VERSION",
    "NativeCanonicalClipMember",
    "StrictClipDetectionEvidencePublication",
    "allocate_automated_clip_refined_ids",
    "publish_strict_clip_detection_evidence",
]

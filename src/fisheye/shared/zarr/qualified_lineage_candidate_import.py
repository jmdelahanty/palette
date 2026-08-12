"""Import one qualified refined-detection/crop lineage pair without selection.

The source snapshots are immutable, already-published Zarr-v3 artifacts.  This
boundary revalidates their complete logical contracts, stages exact physical
copies on node-local scratch, imports each run through the shared atomic run
publisher, and reconsolidates the destination archive only after both runs are
present.  It never updates selectors or the registry.

Clipped refined snapshots remain fail-closed: callers must provide the
directory containing every per-clip refined source artifact used by the
recording-level snapshot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    crop_refined_source_identity_from_manifest,
    validate_crop_publication,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1, CropDimensions
from fisheye.shared.zarr.crop_shadow import crop_metadata_declaration_maps
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
    RefinedDetectionBoundClipEvidence,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import REFINED_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root

QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID = (
    "palette.qualified_lineage.production_candidate_import"
)
QUALIFIED_LINEAGE_IMPORT_SCHEMA_VERSION = 1
QUALIFIED_LINEAGE_IMPORT_POLICY = (
    "validated_immutable_node_local_then_atomic_selector_ineligible_v1"
)
QUALIFIED_LINEAGE_IMPORT_ROLLBACK_POLICY = (
    "retain_complete_ineligible_prefix_or_failed_owner_bound_tombstone_v1"
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


@dataclass(frozen=True)
class QualifiedLineageSource:
    """Fully validated immutable sources and their external clip evidence."""

    refined: BoundRefinedDetectionCropSource
    refined_evidence: tuple[RefinedDetectionBoundClipEvidence, ...]
    refined_direct: Mapping[str, Mapping[str, Any]]
    refined_consolidated: Mapping[str, Mapping[str, Any]]
    crop_archive: Path
    crop_run_id: str
    crop_manifest: Mapping[str, Any]
    crop_arrays: Mapping[str, Any]
    crop_direct: Mapping[str, Mapping[str, Any]]
    crop_consolidated: Mapping[str, Mapping[str, Any]]
    recording_identity: str


def _required_run_id(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if normalized != value or not _RUN_ID.fullmatch(normalized):
        raise ValueError(f"{name} must be one exact safe child-group name.")
    return normalized


def _required_archive(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.suffix != ".zarr":
        raise FileNotFoundError(f"{name} is not a Zarr directory: {resolved}")
    return resolved


def _required_scratch_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Scratch root does not exist: {resolved}")
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    }:
        raise ValueError("Scratch root must be one bounded child directory.")
    if str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError("Qualified lineage import requires node-local scratch.")
    return resolved


def _selector_snapshot(root: Any, family: str) -> dict[str, dict[str, object]]:
    parent = root.get(family)
    attrs = {} if parent is None else parent.attrs
    return {
        name: {"present": name in attrs, "value": copy.deepcopy(attrs.get(name))}
        for name in _SELECTOR_ATTRS
    }


def _require_selector_snapshot(
    root: Any,
    *,
    family: str,
    expected: Mapping[str, object],
) -> None:
    observed = _selector_snapshot(root, family)
    if observed != dict(expected):
        raise RuntimeError(f"{family} selectors changed during lineage import.")


def _refined_arrays(run: Any, *, dimensions: Any) -> dict[str, Any]:
    return {
        path: run[path]
        for path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    }


def _crop_arrays(run: Any) -> dict[str, Any]:
    return {path: run[path] for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths}


def _clip_evidence(
    evidence_root: Path,
    *,
    recording_manifest: Mapping[str, Any],
) -> tuple[RefinedDetectionBoundClipEvidence, ...]:
    root = evidence_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Refined clip-evidence root not found: {root}")
    payload = recording_manifest["payload"]
    source = payload["source_detection"]
    members = source.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("Recording refined manifest lacks clipped source members.")
    evidence: list[RefinedDetectionBoundClipEvidence] = []
    for expected_index, member in enumerate(members):
        if (
            not isinstance(member, Mapping)
            or member.get("clip_index") != expected_index
        ):
            raise ValueError("Refined source members are not contiguous and ordered.")
        run_id = _required_run_id(
            str(member.get("source_refined_run_id") or ""),
            name=f"clip {expected_index} refined run",
        )
        matches = tuple(sorted(root.glob(f"clip_{expected_index:06d}_*")))
        if len(matches) != 1 or not matches[0].is_dir():
            raise ValueError(
                f"Expected one evidence directory for clip {expected_index}; "
                f"found {len(matches)}."
            )
        archive = _required_archive(
            matches[0] / "refined.zarr",
            name=f"clip {expected_index} refined archive",
        )
        bound = bind_refined_detection_crop_source(
            archive,
            run_id=run_id,
            allow_selector_ineligible_benchmark=True,
        )
        evidence.append(
            RefinedDetectionBoundClipEvidence(
                clip_index=expected_index,
                manifest=bound.manifest,
                arrays=bound.arrays,
                parent_manifest=bound.parent_manifest,
                parent_arrays=bound.parent_arrays,
            )
        )
    return tuple(evidence)


def _crop_dimensions(manifest: Mapping[str, Any]) -> CropDimensions:
    dimensions = manifest["payload"]["logical_schema"]["dimensions"]
    return CropDimensions(
        n_frames=int(dimensions["n_frames"]),
        n_instances=int(dimensions["n_instances"]),
        source_width=int(dimensions["source_width"]),
        source_height=int(dimensions["source_height"]),
    )


def inspect_qualified_lineage_source(
    *,
    refined_archive: Path,
    refined_run_id: str,
    refined_clip_evidence_root: Path,
    crop_archive: Path,
    crop_run_id: str,
) -> QualifiedLineageSource:
    """Deeply validate the external lineage pair without writing anything."""

    refined_path = _required_archive(refined_archive, name="refined_archive")
    refined_id = _required_run_id(refined_run_id, name="refined_run_id")
    provisional_root = open_zarr_root(refined_path, mode="r")
    provisional_run = provisional_root[f"refined_detect_runs/{refined_id}"]
    provisional_manifest = provisional_run.attrs.get(
        REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE
    )
    if not isinstance(provisional_manifest, Mapping):
        raise ValueError("Refined source lacks its exact run_manifest.")
    evidence = _clip_evidence(
        refined_clip_evidence_root,
        recording_manifest=provisional_manifest,
    )
    refined = bind_refined_detection_crop_source(
        refined_path,
        run_id=refined_id,
        allow_selector_ineligible_benchmark=True,
        clipped_source_evidence=evidence,
    )
    refined_profile = storage_profile_from_manifest(
        refined.manifest["payload"]["storage_plan"]["storage_profile"]
    )
    refined_plans = plan_refined_detection_storage(
        refined.dimensions,
        profile=refined_profile,
    )
    refined_direct, refined_consolidated = refined_detection_metadata_declaration_maps(
        refined_path,
        run_id=refined_id,
        plans=refined_plans,
    )

    crop_path = _required_archive(crop_archive, name="crop_archive")
    crop_id = _required_run_id(crop_run_id, name="crop_run_id")
    crop_root = open_zarr_root(crop_path, mode="r")
    crop_run = crop_root[f"crop_runs/{crop_id}"]
    crop_manifest = crop_run.attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(crop_manifest, Mapping):
        raise ValueError("Crop source lacks its exact run_manifest.")
    crop_dimensions = _crop_dimensions(crop_manifest)
    crop_profile = storage_profile_from_manifest(
        crop_manifest["payload"]["storage_plan"]["storage_profile"]
    )
    crop_plans = plan_crop_geometry_storage(crop_dimensions, profile=crop_profile)
    crop_direct, crop_consolidated = crop_metadata_declaration_maps(
        crop_path,
        run_id=crop_id,
        plans=crop_plans,
    )
    crop_arrays = _crop_arrays(crop_run)
    crop_errors = validate_crop_publication(
        crop_manifest,
        direct_metadata_declarations=crop_direct,
        consolidated_metadata_declarations=crop_consolidated,
        arrays=crop_arrays,
        source_manifest=refined.manifest,
        source_arrays=refined.arrays,
    )
    if crop_run.attrs.get("status") != "complete":
        crop_errors = (*crop_errors, "crop source is not complete")
    if crop_run.attrs.get("stage_selector_eligible") is not False:
        crop_errors = (*crop_errors, "crop source is not selector-ineligible")
    if crop_errors:
        raise ValueError(
            "Crop source publication is invalid: " + "; ".join(crop_errors)
        )

    crop_source = crop_refined_source_identity_from_manifest(
        crop_manifest["payload"]["source_refined_snapshot"]
    )
    lineage = refined.manifest["payload"]["snapshot_lineage"]
    recording_identity = str(
        lineage["manual_instance_key_allocator"]["recording_identity"]
    )
    expected = {
        "run_id": refined.run_id,
        "run_manifest_digest": refined.manifest["payload_digest"],
        "logical_content_digest": refined.logical_content_digest,
        "recording_identity": recording_identity,
        "lineage_id": lineage["lineage_id"],
        "snapshot_id": lineage["snapshot_id"],
    }
    observed = {
        "run_id": crop_source.run_id,
        "run_manifest_digest": crop_source.run_manifest_digest,
        "logical_content_digest": crop_source.logical_content_digest,
        "recording_identity": crop_source.recording_identity,
        "lineage_id": crop_source.lineage_id,
        "snapshot_id": crop_source.snapshot_id,
    }
    if observed != expected:
        raise ValueError("Crop source does not bind the exact refined snapshot.")
    return QualifiedLineageSource(
        refined=refined,
        refined_evidence=evidence,
        refined_direct=refined_direct,
        refined_consolidated=refined_consolidated,
        crop_archive=crop_path,
        crop_run_id=crop_id,
        crop_manifest=crop_manifest,
        crop_arrays=crop_arrays,
        crop_direct=crop_direct,
        crop_consolidated=crop_consolidated,
        recording_identity=recording_identity,
    )


def _mark_production_candidate(run: Any, *, nested_tables: Sequence[str]) -> None:
    attrs = dict(run.attrs)
    attrs.pop("shadow_only", None)
    attrs.pop("benchmark_only", None)
    attrs.update(
        {
            "immutable_snapshot": True,
            "production_candidate": True,
            "stage_selector_eligible": False,
            "production_selector_activation": "deferred",
        }
    )
    run.attrs.put(attrs)
    groups = [run]
    groups.extend(run[name] for name in nested_tables if name in run)
    for group in groups:
        for _, array in group.arrays():
            array_attrs = dict(array.attrs)
            array_attrs.pop("shadow_only", None)
            array_attrs.pop("benchmark_only", None)
            array_attrs["selector_eligible"] = False
            array.attrs.put(array_attrs)


def _candidate_state_errors(run: Any, *, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if run.attrs.get("run_manifest") != manifest:
        errors.append("run_manifest differs from the qualified source")
    if run.attrs.get("status") != "complete":
        errors.append("run is not complete")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("run is not selector-ineligible")
    if run.attrs.get("production_candidate") is not True:
        errors.append("run is not marked as a production candidate")
    if run.attrs.get("shadow_only") is not None:
        errors.append("run retains shadow-only state")
    if run.attrs.get("benchmark_only") is not None:
        errors.append("run retains benchmark-only state")
    return errors


def _refined_validator(
    path: Path,
    *,
    source: QualifiedLineageSource,
) -> dict[str, object]:
    errors: list[str] = []
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        errors.extend(_candidate_state_errors(run, manifest=source.refined.manifest))
        arrays = _refined_arrays(run, dimensions=source.refined.dimensions)
        errors.extend(
            validate_refined_detection_publication(
                source.refined.manifest,
                direct_metadata_declarations=source.refined_direct,
                consolidated_metadata_declarations=source.refined_consolidated,
                arrays=arrays,
                parent_manifest=source.refined.parent_manifest,
                parent_arrays=source.refined.parent_arrays,
                clipped_source_evidence=source.refined_evidence,
            )
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors))}


def _crop_validator(
    path: Path,
    *,
    source: QualifiedLineageSource,
    refined_arrays: Mapping[str, Any],
) -> dict[str, object]:
    errors: list[str] = []
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        errors.extend(_candidate_state_errors(run, manifest=source.crop_manifest))
        errors.extend(
            validate_crop_publication(
                source.crop_manifest,
                direct_metadata_declarations=source.crop_direct,
                consolidated_metadata_declarations=source.crop_consolidated,
                arrays=_crop_arrays(run),
                source_manifest=source.refined.manifest,
                source_arrays=refined_arrays,
            )
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors))}


def _prepare_family(family: str):
    def prepare(root: Any) -> tuple[Any, ...]:
        return (root.require_group(family),)

    return prepare


def _complete_ineligible(_root: Any, _parent: Any, run: Any) -> None:
    if (
        run.attrs.get("status") != "complete"
        or run.attrs.get("stage_selector_eligible") is not False
        or run.attrs.get("production_candidate") is not True
    ):
        raise RuntimeError("Imported lineage run is not complete/ineligible.")


def _verify_run_and_selectors(
    root: Any,
    *,
    family: str,
    run_id: str,
    selectors: Mapping[str, object],
) -> None:
    _require_selector_snapshot(root, family=family, expected=selectors)
    run = root[f"{family}/{run_id}"]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"{family}/{run_id} became selector-eligible.")


def _atomic_import(
    *,
    archive: Path,
    local_run: Path,
    family: str,
    run_id: str,
    role: str,
    selectors: Mapping[str, object],
    validator: Any,
    copy_backend: str,
) -> dict[str, Any]:
    target = archive / family / run_id
    if target.exists():
        validation = dict(validator(target))
        if not validation.get("valid"):
            raise FileExistsError(
                f"Existing immutable candidate is not reusable: {target}: {validation}"
            )
        root = open_zarr_root(archive, mode="r")
        _verify_run_and_selectors(
            root,
            family=family,
            run_id=run_id,
            selectors=selectors,
        )
        return {
            "status": "reused_exact_complete_candidate",
            "target_run_path": str(target),
            "validation": validation,
        }
    local_group = zarr.open_group(str(local_run), mode="r", use_consolidated=False)
    local_manifest = local_group.attrs.get("run_manifest")
    if not isinstance(local_manifest, Mapping):
        raise ValueError(f"Local lineage candidate lacks run_manifest: {local_run}")
    source_manifest_digest = str(local_manifest.get("payload_digest") or "")
    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=archive,
            local_run_path=local_run,
            target_run_path=target,
            run_name=run_id,
            lock_suffix="qualified_lineage_candidate_import",
            publish_schema_id=QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID,
            policy=QUALIFIED_LINEAGE_IMPORT_POLICY,
            rollback_policy=QUALIFIED_LINEAGE_IMPORT_ROLLBACK_POLICY,
            content_checksum=True,
        ),
        copy_backend=copy_backend,
        validate_run=validator,
        prepare_parents=_prepare_family(family),
        complete_run=_complete_ineligible,
        verify_pointers=lambda root: _verify_run_and_selectors(
            root,
            family=family,
            run_id=run_id,
            selectors=selectors,
        ),
        payload_metadata={
            "candidate_role": role,
            "source_manifest_digest": source_manifest_digest,
            "selector_activation": "deferred",
        },
        repair_failed_publication_visibility=lambda _path: (
            consolidate_metadata_capture_expected_warnings(archive)
        ),
    )


def publish_qualified_lineage_candidate(
    *,
    analysis_zarr: Path,
    refined_archive: Path,
    refined_run_id: str,
    refined_clip_evidence_root: Path,
    crop_archive: Path,
    crop_run_id: str,
    scratch_root: Path,
    copy_backend: str = "rsync",
    keep_scratch: bool = False,
) -> dict[str, object]:
    """Import the exact refined/crop pair and leave production selection unchanged."""

    started = time.perf_counter()
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    archive = _required_archive(analysis_zarr, name="analysis_zarr")
    scratch = _required_scratch_root(scratch_root)
    source = inspect_qualified_lineage_source(
        refined_archive=refined_archive,
        refined_run_id=refined_run_id,
        refined_clip_evidence_root=refined_clip_evidence_root,
        crop_archive=crop_archive,
        crop_run_id=crop_run_id,
    )
    target_root = open_zarr_root(archive, mode="r")
    if str(target_root.attrs.get("recording_id") or "") != source.recording_identity:
        raise ValueError("Destination recording_id differs from the lineage source.")
    selectors = {
        family: _selector_snapshot(target_root, family)
        for family in ("refined_detect_runs", "crop_runs")
    }

    session = scratch / f"palette_qualified_lineage_{uuid.uuid4().hex}"
    local_refined = session / "refined_detect_runs" / source.refined.run_id
    local_crop = session / "crop_runs" / source.crop_run_id
    success = False
    try:
        local_refined.parent.mkdir(parents=True)
        shutil.copytree(
            source.refined.archive_path / source.refined.run_path,
            local_refined,
        )
        local_crop.parent.mkdir(parents=True)
        shutil.copytree(
            source.crop_archive / "crop_runs" / source.crop_run_id,
            local_crop,
        )
        _mark_production_candidate(
            zarr.open_group(str(local_refined), mode="a", use_consolidated=False),
            nested_tables=("instances", "source_detections"),
        )
        _mark_production_candidate(
            zarr.open_group(str(local_crop), mode="a", use_consolidated=False),
            nested_tables=(),
        )

        def refined_validator(path: Path) -> dict[str, object]:
            return _refined_validator(path, source=source)

        refined_receipt = _atomic_import(
            archive=archive,
            local_run=local_refined,
            family="refined_detect_runs",
            run_id=source.refined.run_id,
            role="refined_detection_parent",
            selectors=selectors["refined_detect_runs"],
            validator=refined_validator,
            copy_backend=copy_backend,
        )
        persisted_refined = zarr.open_group(
            str(archive / "refined_detect_runs" / source.refined.run_id),
            mode="r",
            use_consolidated=False,
        )
        persisted_refined_arrays = _refined_arrays(
            persisted_refined,
            dimensions=source.refined.dimensions,
        )

        def crop_validator(path: Path) -> dict[str, object]:
            return _crop_validator(
                path,
                source=source,
                refined_arrays=persisted_refined_arrays,
            )

        crop_receipt = _atomic_import(
            archive=archive,
            local_run=local_crop,
            family="crop_runs",
            run_id=source.crop_run_id,
            role="geometry_only_crop",
            selectors=selectors["crop_runs"],
            validator=crop_validator,
            copy_backend=copy_backend,
        )

        consolidation = consolidate_metadata_capture_expected_warnings(archive)
        published_refined = bind_refined_detection_crop_source(
            archive,
            run_id=source.refined.run_id,
            allow_selector_ineligible_benchmark=True,
            clipped_source_evidence=source.refined_evidence,
        )
        crop_path = archive / "crop_runs" / source.crop_run_id
        final_crop_validation = _crop_validator(
            crop_path,
            source=source,
            refined_arrays=published_refined.arrays,
        )
        if not final_crop_validation["valid"]:
            raise RuntimeError(
                f"Published crop lineage validation failed: {final_crop_validation}"
            )
        final_root = open_zarr_root(archive, mode="r")
        for family, run_id in (
            ("refined_detect_runs", source.refined.run_id),
            ("crop_runs", source.crop_run_id),
        ):
            _verify_run_and_selectors(
                final_root,
                family=family,
                run_id=run_id,
                selectors=selectors[family],
            )
        consolidated_root = zarr.open_group(
            str(archive),
            mode="r",
            zarr_format=3,
            use_consolidated=True,
        )
        for path in (
            f"refined_detect_runs/{source.refined.run_id}",
            f"crop_runs/{source.crop_run_id}",
        ):
            if path not in consolidated_root:
                raise RuntimeError(f"Consolidated metadata does not expose {path}.")

        result = {
            "schema_id": QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID,
            "schema_version": QUALIFIED_LINEAGE_IMPORT_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "recording_identity": source.recording_identity,
            "refined": {
                "run_id": source.refined.run_id,
                "manifest_digest": source.refined.manifest["payload_digest"],
                "logical_content_digest": source.refined.logical_content_digest,
                "atomic_import": refined_receipt,
            },
            "crop": {
                "run_id": source.crop_run_id,
                "manifest_digest": source.crop_manifest["payload_digest"],
                "logical_content_digest": source.crop_manifest["payload"][
                    "logical_content"
                ]["digest"],
                "atomic_import": crop_receipt,
            },
            "clip_evidence_count": len(source.refined_evidence),
            "selector_eligible": False,
            "selector_activation": "none",
            "registry_updated": False,
            "selectors_unchanged": True,
            "direct_and_consolidated_validated": True,
            "consolidation": consolidation,
            "node_local_session": str(session),
            "node_local_session_retained": bool(keep_scratch),
            "total_seconds": float(time.perf_counter() - started),
        }
        success = True
        return json_attr_safe(result)
    finally:
        if session.exists() and success and not keep_scratch:
            shutil.rmtree(session)


__all__ = [
    "QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID",
    "QUALIFIED_LINEAGE_IMPORT_SCHEMA_VERSION",
    "QualifiedLineageSource",
    "inspect_qualified_lineage_source",
    "publish_qualified_lineage_candidate",
]

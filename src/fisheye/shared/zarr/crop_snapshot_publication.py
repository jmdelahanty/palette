"""Publish one selector-ineligible geometry-only crop production candidate.

The future-facing path binds the approved refined-detection authority and its
published source-pixel authority, materializes a complete immutable run on
node-local scratch, validates it, atomically imports the run into the recording
archive, reconsolidates archive metadata, and validates the imported run again.
It deliberately does not update a selector, registry, or production default.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import zarr

from fisheye.shared.hybrid_crop_provider import (
    validate_hybrid_crop_signed_identity,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    build_coordinate_crop_run_manifest,
    validate_crop_publication,
)
from fisheye.shared.zarr.crop_pixel_authority import (
    BoundCropPixelAuthority,
    bind_refined_crop_source_pixel_authority,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_EXPLICIT_ORIGIN_AUTHORITY_SCHEMA_ID,
    CROP_EXPLICIT_ORIGIN_AUTHORITY_SCHEMA_VERSION,
    CROP_GEOMETRY_SCHEMA_V1,
    CropGeometryPolicy,
    CropPlacementMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import (
    CropGeometryShadowPublication,
    PreparedCropGeometrySnapshot,
    crop_metadata_declaration_maps,
    prepare_crop_geometry_from_refined_source,
    publish_selector_ineligible_crop_geometry_snapshot,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_failed,
    require_runs_parent,
)


CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID = "palette.crop_geometry.production_publication"
CROP_SNAPSHOT_PUBLICATION_SCHEMA_VERSION = 2
CROP_SNAPSHOT_PUBLICATION_POLICY = (
    "node_local_v1_materialization_then_atomic_selector_ineligible_import_v1"
)
CROP_SNAPSHOT_ROLLBACK_POLICY = "retain_failed_owner_bound_selector_ineligible_child_v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


def publish_crop_geometry_from_explicit_refined_candidate(
    *,
    refined_archive: Path,
    refined_run_id: str,
    pixel_authority: BoundCropPixelAuthority,
    policy: CropGeometryPolicy,
    destination: Path,
    run_id: str,
    safe_root: Path,
    clipped_source_evidence: tuple[RefinedDetectionBoundClipEvidence, ...]
    | None = None,
    roi_sizes_full: np.ndarray | None = None,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "explicit_refined_crop_candidate",
) -> CropGeometryShadowPublication:
    """Publish geometry from one explicit selector-ineligible refined source.

    This is the candidate-set boundary used before selector activation.  It
    accepts both full-acquisition and clipped recording snapshots, but clipped
    sources remain fail-closed unless every bound per-clip artifact is supplied
    again for publication validation.
    """

    source = bind_refined_detection_crop_source(
        refined_archive,
        run_id=refined_run_id,
        allow_selector_ineligible_benchmark=True,
        clipped_source_evidence=clipped_source_evidence,
    )
    pixel_authority.assert_verified()
    prepared = prepare_crop_geometry_from_refined_source(
        source,
        policy=policy,
        pixel_authority=pixel_authority.pixel_authority,
        roi_sizes_full=roi_sizes_full,
    )
    return publish_selector_ineligible_crop_geometry_snapshot(
        prepared,
        destination=destination,
        run_id=run_id,
        shadow_root=safe_root,
        profile=profile,
        created_by=created_by,
        coordinate_catalog=True,
    )


def _require_run_id(value: str) -> str:
    normalized = str(value).strip()
    if not _RUN_ID_RE.fullmatch(normalized):
        raise ValueError("run_id must be one safe nonempty child-group name.")
    return normalized


def _require_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Crop publication scratch root not found: {resolved}")
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError("Crop publication scratch must be a bounded node-local path.")
    return resolved


def _bind_explicit_origin_provider(
    *,
    archive: Path,
    provider_run_id: str,
    source: BoundRefinedDetectionCropSource,
    base_policy: CropGeometryPolicy,
) -> tuple[np.ndarray, CropGeometryPolicy, dict[str, Any]]:
    """Bind exact verified per-row origins from one signed hybrid provider."""

    run_id = _require_run_id(provider_run_id)
    if base_policy.placement_mode is not CropPlacementMode.REFINED_DETECTION_CENTERED:
        raise ValueError(
            "The crop publisher owns explicit-origin policy construction; callers "
            "must supply the ordinary detection-centered base policy."
        )
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    try:
        provider = root[f"crop_runs/{run_id}"]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Explicit crop-origin provider not found: crop_runs/{run_id}"
        ) from exc
    if (
        provider.attrs.get("stage_selector_eligible") is not False
        or str(provider.attrs.get("status") or "") not in {"complete", "completed"}
    ):
        raise ValueError(
            "Explicit crop-origin provider must be complete and selector-ineligible."
        )
    provider_record_sha256 = str(
        provider.attrs.get("provider_record_sha256") or ""
    )
    signed = validate_hybrid_crop_signed_identity(
        provider,
        expected_provider_record_sha256=provider_record_sha256,
    )
    if (
        provider.attrs.get("source_refined_run_id") != source.run_id
        or provider.attrs.get("source_refined_manifest_digest")
        != source.manifest.get("payload_digest")
    ):
        raise ValueError(
            "Explicit crop-origin provider binds a different refined snapshot."
        )

    comparisons = (
        ("instance_key", "instances/instance_key"),
        ("source_refined_row_ids", "instances/refined_row_ids"),
        ("frame_indices", "instances/frame_indices"),
        (
            "source_acquisition_frame_index",
            "instances/source_acquisition_frame_index",
        ),
    )
    mismatched = [
        provider_path
        for provider_path, source_path in comparisons
        if provider_path not in provider
        or not np.array_equal(
            np.asarray(provider[provider_path][...]),
            np.asarray(source.arrays[source_path][...]),
        )
    ]
    if mismatched:
        raise ValueError(
            "Explicit crop-origin provider differs from the refined rowset at: "
            + ", ".join(mismatched)
        )
    if "roi_coordinates_full" not in provider or "roi_sizes_full" not in provider:
        raise ValueError("Explicit crop-origin provider lacks ROI geometry arrays.")
    origins = np.asarray(provider["roi_coordinates_full"][...])
    sizes = np.asarray(provider["roi_sizes_full"][...])
    expected_shape = (source.dimensions.n_instances, 2)
    if origins.dtype != np.dtype(np.int32) or origins.shape != expected_shape:
        raise ValueError(
            "Explicit crop-origin provider origins must have exact int32 [N,2] "
            "shape."
        )
    if sizes.dtype != np.dtype(np.int32) or sizes.shape != expected_shape:
        raise ValueError(
            "Explicit crop-origin provider sizes must have exact int32 [N,2] shape."
        )
    if base_policy.size_mode is CropSizeMode.FIXED_PER_RUN and not np.all(
        sizes
        == np.asarray(base_policy.fixed_size_wh, dtype=np.int32).reshape(1, 2)
    ):
        raise ValueError(
            "Explicit crop-origin provider sizes differ from fixed crop policy."
        )

    authority = {
        "schema_id": CROP_EXPLICIT_ORIGIN_AUTHORITY_SCHEMA_ID,
        "schema_version": CROP_EXPLICIT_ORIGIN_AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "signed_hybrid_crop_provider",
        "run_id": run_id,
        "provider_record_sha256": signed["provider_record_sha256"],
        "source_rowset_fingerprint": signed["source_rowset_fingerprint"],
        "source_pixel_fingerprint": signed["source_pixel_fingerprint"],
        "source_row_signature_spec_digest": signed[
            "source_row_signature_spec_digest"
        ],
    }
    policy = replace(
        base_policy,
        placement_mode=CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW,
        placement_authority=authority,
    )
    return np.array(origins, copy=True, order="C"), policy, {
        **authority,
        "row_count": int(signed["row_count"]),
        "ordered_refined_coverage_exact": True,
        "roi_sizes_match_policy": True,
    }


def _crop_arrays(run: Any) -> dict[str, Any]:
    return {path: run[path] for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths}


def _require_same_refined_source(
    expected: BoundRefinedDetectionCropSource,
    observed: BoundRefinedDetectionCropSource,
) -> None:
    if (
        observed.archive_path != expected.archive_path
        or observed.run_id != expected.run_id
        or observed.selection_mode != expected.selection_mode
        or observed.manifest != expected.manifest
        or observed.logical_content_digest != expected.logical_content_digest
        or observed.handoff_manifest != expected.handoff_manifest
    ):
        raise RuntimeError(
            "Approved refined-detection authority changed during publication."
        )


def _rebind_authorities(
    *,
    archive: Path,
    source: BoundRefinedDetectionCropSource,
    pixels: BoundCropPixelAuthority,
    expected_camera_identity: str,
    explicit_refined_source: bool,
) -> BoundRefinedDetectionCropSource:
    observed = (
        bind_refined_detection_crop_source(
            archive,
            run_id=source.run_id,
            allow_selector_ineligible_benchmark=True,
            allow_mutable_archive_direct_metadata=True,
        )
        if explicit_refined_source
        else bind_refined_detection_crop_source(archive)
    )
    _require_same_refined_source(source, observed)
    pixels.assert_verified()
    if pixels.pixel_authority.camera_identity != expected_camera_identity:
        raise RuntimeError(
            "Bound source-pixel camera identity changed during publication."
        )
    return observed


def _registered_gate_source_allows_selector_ineligible(
    *,
    archive: Path,
    source: BoundRefinedDetectionCropSource,
    gate_evidence: Mapping[str, Any],
) -> bool:
    """Select the strict raw-detection authority mode from persisted lineage."""

    source_identity = source.manifest["payload"].get("source_detection")
    if not isinstance(source_identity, Mapping) or (
        source_identity.get("authority_kind") != "canonical_run"
    ):
        raise RuntimeError(
            "Finalized refined source lacks one canonical raw-detection identity."
        )
    run_id = str(source_identity.get("run_id") or "").strip()
    expected_path = f"detect_runs/{run_id}"
    evidence_path = str(
        gate_evidence.get("source_detection_group_path")
        or gate_evidence.get("source_detection_path")
        or ""
    ).strip()
    if not run_id or evidence_path != expected_path:
        raise RuntimeError(
            "Finalized refined and registered-gate raw-detection identities differ."
        )

    root = open_zarr_root(archive, mode="r", use_consolidated=False)
    try:
        detection = root[expected_path]
    except KeyError as exc:
        raise RuntimeError(
            f"Finalized refined raw-detection source is missing: {expected_path}"
        ) from exc
    manifest = detection.attrs.get("run_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("payload_digest")
        != source_identity.get("run_manifest_digest")
    ):
        raise RuntimeError(
            "Finalized refined raw-detection manifest identity changed."
        )
    payload = manifest.get("payload")
    publication = payload.get("publication") if isinstance(payload, Mapping) else None
    manifest_eligible = (
        publication.get("stage_selector_eligible")
        if isinstance(publication, Mapping)
        else None
    )
    run_eligible = detection.attrs.get("stage_selector_eligible")
    if manifest_eligible is True and run_eligible is True:
        return False
    if (
        manifest_eligible is False
        and run_eligible is False
        and detection.attrs.get("production_candidate") is True
    ):
        return True
    raise RuntimeError(
        "Finalized refined raw-detection source is neither an active authority "
        "nor an exact selector-ineligible production candidate."
    )


def _mark_local_production_candidate(
    run: Any,
    *,
    source_run_id: str,
    source_manifest_digest: str,
    registered_gate_requirement: str,
    registered_gate_evidence: Mapping[str, Any] | None,
) -> None:
    attrs = dict(run.attrs)
    attrs.pop("shadow_only", None)
    attrs.pop("benchmark_only", None)
    attrs.update(
        {
            "immutable_snapshot": True,
            "production_candidate": True,
            "stage_selector_eligible": False,
            "production_selector_activation": "deferred",
            "source_refined_run_id": source_run_id,
            "source_refined_manifest_digest": source_manifest_digest,
            "source_registered_detection_gate_requirement": (
                registered_gate_requirement
            ),
            "source_registered_detection_gate": (
                None
                if registered_gate_evidence is None
                else json_attr_safe(dict(registered_gate_evidence))
            ),
        }
    )
    run.attrs.put(attrs)
    for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths:
        array = run[path]
        array_attrs = dict(array.attrs)
        array_attrs.pop("shadow_only", None)
        array_attrs.pop("benchmark_only", None)
        array_attrs["selector_eligible"] = False
        array.attrs.put(array_attrs)


def _build_and_persist_manifest(
    *,
    archive_path: Path,
    run_id: str,
    publication: CropGeometryShadowPublication,
    prepared: PreparedCropGeometrySnapshot,
) -> tuple[dict[str, object], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    run = zarr.open_group(
        str(archive_path / "crop_runs" / run_id),
        mode="a",
        use_consolidated=False,
    )
    arrays = _crop_arrays(run)
    consolidate_metadata_capture_expected_warnings(archive_path)
    direct, consolidated = crop_metadata_declaration_maps(
        archive_path,
        run_id=run_id,
        plans=publication.plans,
    )
    manifest = build_coordinate_crop_run_manifest(
        run_id=run_id,
        dimensions=publication.dimensions,
        policy=prepared.policy,
        storage_plan=publication.plans,
        arrays=arrays,
        source=prepared.source,
        pixel_authority=prepared.pixel_authority,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )
    run.attrs[CROP_RUN_MANIFEST_ATTRIBUTE] = manifest
    consolidate_metadata_capture_expected_warnings(archive_path)
    direct, consolidated = crop_metadata_declaration_maps(
        archive_path,
        run_id=run_id,
        plans=publication.plans,
    )
    return manifest, direct, consolidated


def _validate_candidate(
    *,
    run_path: Path,
    manifest: Mapping[str, Any],
    direct: Mapping[str, Mapping[str, Any]],
    consolidated: Mapping[str, Mapping[str, Any]],
    prepared: PreparedCropGeometrySnapshot,
) -> dict[str, object]:
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        errors = validate_crop_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=_crop_arrays(run),
            source_manifest=prepared.source_manifest,
            source_arrays=prepared.source_arrays,
        )
        if run.attrs.get("status") != "complete":
            errors = (*errors, "crop candidate status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors = (*errors, "crop candidate is not selector-ineligible")
        if run.attrs.get("production_candidate") is not True:
            errors = (*errors, "crop candidate lacks production-candidate state")
        if run.attrs.get("shadow_only") is not None:
            errors = (*errors, "crop candidate retains shadow-only state")
        return {"valid": not errors, "errors": list(errors)}
    except Exception as exc:
        return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}


def _prepare_parent(root: Any) -> tuple[Any, ...]:
    return (
        require_runs_parent(
            root, "crop_runs", completion_epoch=COMPLETION_EPOCH_STRICT
        ),
    )


def _require_unselected(root: Any, *, run_id: str) -> None:
    family = root["crop_runs"]
    selected = [
        name for name in _SELECTOR_ATTRIBUTES if family.attrs.get(name) == run_id
    ]
    if selected:
        raise RuntimeError(
            f"Selector-ineligible crop candidate {run_id!r} is selected by {selected!r}."
        )
    run = family[run_id]
    if run.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError(f"Crop candidate {run_id!r} became selector-eligible.")


def _mark_failed_import(path: Path, *, error: BaseException) -> None:
    if not path.is_dir():
        return
    try:
        run = zarr.open_group(str(path), mode="a", use_consolidated=False)
        attrs = dict(run.attrs)
        attrs.update(
            {
                "status": "failed",
                "stage_selector_eligible": False,
                "production_candidate": False,
                "production_selector_activation": "blocked_failed_publication",
                "publication_failure": f"{type(error).__name__}: {error}",
            }
        )
        run.attrs.put(attrs)
        mark_run_failed(
            run,
            run_name=path.name,
            error=f"{type(error).__name__}: {error}",
        )
    except Exception:
        return


def publish_crop_geometry_production_candidate(
    *,
    analysis_zarr: Path,
    run_id: str,
    policy: CropGeometryPolicy,
    expected_camera_identity: str,
    scratch_root: Path,
    roi_sizes_full: np.ndarray | None = None,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    copy_backend: str = "python",
    keep_scratch: bool = False,
    source_refined_run_id: str | None = None,
    registered_gate_requirement: str = "off",
    registered_gate_run: str | None = None,
    registered_gate_validator: Callable[..., dict[str, Any]] | None = None,
    geometry_origin_provider_run_id: str | None = None,
) -> dict[str, object]:
    """Publish a complete crop candidate without activating production state."""

    started = time.perf_counter()
    archive = analysis_zarr.expanduser().resolve()
    if not archive.is_dir() or archive.suffix != ".zarr":
        raise FileNotFoundError(f"Analysis Zarr not found: {archive}")
    candidate_id = _require_run_id(run_id)
    camera_identity = str(expected_camera_identity).strip()
    if not camera_identity or camera_identity != expected_camera_identity:
        raise ValueError("expected_camera_identity must be an exact nonempty string.")
    if copy_backend not in {"python", "rsync"}:
        raise ValueError("copy_backend must be 'python' or 'rsync'.")
    scratch = _require_node_local_scratch(scratch_root)
    target = archive / "crop_runs" / candidate_id
    if target.exists():
        raise FileExistsError(f"Immutable crop candidate already exists: {target}")

    explicit_refined_source = bool(str(source_refined_run_id or "").strip())
    source = (
        bind_refined_detection_crop_source(
            archive,
            run_id=str(source_refined_run_id).strip(),
            allow_selector_ineligible_benchmark=True,
            allow_mutable_archive_direct_metadata=True,
        )
        if explicit_refined_source
        else bind_refined_detection_crop_source(archive)
    )
    if not explicit_refined_source and source.selection_mode != "approved_authoritative_refined_v1":
        raise RuntimeError("Crop production requires the approved refined authority.")
    gate_requirement = str(registered_gate_requirement).strip()
    if gate_requirement not in {"off", "if_available", "required"}:
        raise ValueError(
            "registered_gate_requirement must be off, if_available, or required."
        )
    if gate_requirement != "off" and not explicit_refined_source:
        raise RuntimeError(
            "Configured registered geometry requires one explicit finalized refined run."
        )
    if gate_requirement == "required" and not str(registered_gate_run or "").strip():
        raise ValueError("Required-policy crop publication needs one exact gate run.")
    gate_evidence: Mapping[str, Any] | None = None
    if explicit_refined_source:
        if (
            source.run_group.attrs.get("finalized_recording_authority") is not True
            or source.run_group.attrs.get("immutable_snapshot") is not True
        ):
            raise RuntimeError(
                "Explicit crop source must be a finalized immutable recording authority."
            )
        observed_requirement = str(
            source.run_group.attrs.get("registered_detection_gate_requirement") or ""
        ).strip()
        if observed_requirement != gate_requirement:
            raise RuntimeError(
                "Finalized refined source gate requirement differs from crop policy."
            )
        raw_gate_evidence = source.run_group.attrs.get("registered_detection_gate")
        if not isinstance(raw_gate_evidence, Mapping):
            raise RuntimeError("Finalized refined source lacks gate-consumption evidence.")
        gate_evidence = raw_gate_evidence
        applied = (
            gate_evidence.get("applied") is True
            and gate_evidence.get("status") == "applied"
        )
        expected_gate = str(registered_gate_run or "").strip() or None
        observed_gate = str(gate_evidence.get("gate_run") or "").strip() or None
        if expected_gate is not None and observed_gate != expected_gate:
            raise RuntimeError(
                "Finalized refined source consumed a different registered gate."
            )
        if gate_requirement == "required" and not applied:
            raise RuntimeError(
                "Required-policy crop publication needs applied gate consumption."
            )
        if applied:
            if registered_gate_validator is None:
                raise RuntimeError(
                    "Applied registered geometry requires an explicit current-gate "
                    "validator at the crop publication boundary."
                )
            allow_selector_ineligible_gate_source = (
                _registered_gate_source_allows_selector_ineligible(
                    archive=archive,
                    source=source,
                    gate_evidence=gate_evidence,
                )
            )
            current_gate = registered_gate_validator(
                archive,
                source_group_path=str(
                    gate_evidence.get("source_detection_group_path")
                    or gate_evidence.get("source_detection_path")
                    or ""
                ),
                gate_run=str(observed_gate or ""),
                expected_instance_keys=np.asarray(
                    source.arrays["source_detections/instance_key"][...],
                    dtype=np.uint64,
                ),
                require_modern_operational_selection=True,
                allow_selector_ineligible_source=(
                    allow_selector_ineligible_gate_source
                ),
            )
            current_gate.pop("inside", None)
            mismatched = [
                key
                for key, value in current_gate.items()
                if key in gate_evidence and gate_evidence.get(key) != value
            ]
            if mismatched:
                raise RuntimeError(
                    "Finalized refined gate evidence is stale at crop publication: "
                    + ", ".join(sorted(mismatched))
                )
    pixels = bind_refined_crop_source_pixel_authority(
        source,
        expected_camera_identity=camera_identity,
    )
    pixels.assert_verified()
    origin_provider_run = str(geometry_origin_provider_run_id or "").strip()
    if origin_provider_run:
        explicit_origins, effective_policy, origin_binding = (
            _bind_explicit_origin_provider(
                archive=archive,
                provider_run_id=origin_provider_run,
                source=source,
                base_policy=policy,
            )
        )
    else:
        if policy.placement_mode is not CropPlacementMode.REFINED_DETECTION_CENTERED:
            raise ValueError(
                "Explicit crop placement requires geometry_origin_provider_run_id."
            )
        explicit_origins = None
        effective_policy = policy
        origin_binding = None
    root_before = open_zarr_root(archive, mode="r")
    root_attrs_before = dict(root_before.attrs)
    crop_parent_before = root_before.get("crop_runs")
    crop_parent_attrs_before = (
        {} if crop_parent_before is None else dict(crop_parent_before.attrs)
    )
    expected_crop_parent_attrs = dict(crop_parent_attrs_before)
    crop_parent_has_children = bool(
        crop_parent_before is not None
        and tuple(crop_parent_before.group_keys())
    )
    if not crop_parent_has_children:
        expected_crop_parent_attrs.setdefault(
            COMPLETION_EPOCH_ATTR, COMPLETION_EPOCH_STRICT
        )

    session = scratch / f"palette_crop_candidate_{uuid.uuid4().hex}"
    local_root = session / ".palette_benchmarks" / "production_candidate"
    local_root.mkdir(parents=True, exist_ok=False)
    success = False
    imported = False
    try:
        prepared = prepare_crop_geometry_from_refined_source(
            source,
            policy=effective_policy,
            pixel_authority=pixels.pixel_authority,
            roi_sizes_full=roi_sizes_full,
            roi_coordinates_full=explicit_origins,
        )
        local_archive = local_root / "crop.zarr"
        publication = publish_selector_ineligible_crop_geometry_snapshot(
            prepared,
            destination=local_archive,
            run_id=candidate_id,
            shadow_root=local_root,
            profile=profile,
            created_by="crop_geometry_production_candidate",
            coordinate_catalog=True,
        )
        local_run = local_archive / "crop_runs" / candidate_id
        local_group = zarr.open_group(str(local_run), mode="a", use_consolidated=False)
        _mark_local_production_candidate(
            local_group,
            source_run_id=source.run_id,
            source_manifest_digest=str(source.manifest["payload_digest"]),
            registered_gate_requirement=gate_requirement,
            registered_gate_evidence=gate_evidence,
        )
        local_manifest, local_direct, local_consolidated = _build_and_persist_manifest(
            archive_path=local_archive,
            run_id=candidate_id,
            publication=publication,
            prepared=prepared,
        )
        local_validation = _validate_candidate(
            run_path=local_run,
            manifest=local_manifest,
            direct=local_direct,
            consolidated=local_consolidated,
            prepared=prepared,
        )
        if not local_validation["valid"]:
            raise RuntimeError(f"Local crop candidate is invalid: {local_validation}")

        current_source = _rebind_authorities(
            archive=archive,
            source=source,
            pixels=pixels,
            expected_camera_identity=camera_identity,
            explicit_refined_source=explicit_refined_source,
        )
        if origin_provider_run:
            current_origins, current_policy, current_origin_binding = (
                _bind_explicit_origin_provider(
                    archive=archive,
                    provider_run_id=origin_provider_run,
                    source=current_source,
                    base_policy=policy,
                )
            )
            if (
                current_policy != effective_policy
                or current_origin_binding != origin_binding
                or not np.array_equal(current_origins, explicit_origins)
            ):
                raise RuntimeError(
                    "Explicit crop-origin provider changed before archive import."
                )
        current_prepared = PreparedCropGeometrySnapshot(
            dimensions=prepared.dimensions,
            policy=prepared.policy,
            source=prepared.source,
            pixel_authority=prepared.pixel_authority,
            arrays=prepared.arrays,
            source_manifest=current_source.manifest,
            source_arrays=current_source.arrays,
        )

        # The atomic publisher adds transaction attributes to the run group.
        # Validate decoded values against the already-proven local metadata while
        # it performs the copy, then rebuild the exact manifest after its final
        # transaction receipt has been persisted.
        def atomic_validator(path: Path) -> Mapping[str, Any]:
            return _validate_candidate(
                run_path=path,
                manifest=local_manifest,
                direct=local_direct,
                consolidated=local_consolidated,
                prepared=current_prepared,
            )

        def complete_run(_root: Any, _parent: Any, run: Any) -> None:
            if (
                run.attrs.get("status") != "complete"
                or run.attrs.get("stage_selector_eligible") is not False
                or run.attrs.get("production_candidate") is not True
            ):
                raise RuntimeError(
                    "Imported crop candidate is not complete and staged."
                )
            mark_run_complete(run, run_name=candidate_id)

        atomic_receipt = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=archive,
                local_run_path=local_run,
                target_run_path=target,
                run_name=candidate_id,
                lock_suffix="crop_snapshot_publication",
                publish_schema_id=CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
                policy=CROP_SNAPSHOT_PUBLICATION_POLICY,
                rollback_policy=CROP_SNAPSHOT_ROLLBACK_POLICY,
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=atomic_validator,
            prepare_parents=_prepare_parent,
            complete_run=complete_run,
            verify_pointers=lambda root: _require_unselected(root, run_id=candidate_id),
            payload_metadata={
                "snapshot_role": "geometry_only_crop_v2",
                "source_refined_run_id": source.run_id,
                "source_refined_manifest_digest": source.manifest["payload_digest"],
                "source_pixel_authority_digest": pixels.binding_document_digest,
                "selector_activation": "deferred",
                "registered_gate_requirement": gate_requirement,
                "registered_gate_run": (
                    None if gate_evidence is None else gate_evidence.get("gate_run")
                ),
                "geometry_origin_provider": origin_binding,
            },
        )
        imported = True

        current_source = _rebind_authorities(
            archive=archive,
            source=source,
            pixels=pixels,
            expected_camera_identity=camera_identity,
            explicit_refined_source=explicit_refined_source,
        )
        if origin_provider_run:
            final_origins, final_policy, final_origin_binding = (
                _bind_explicit_origin_provider(
                    archive=archive,
                    provider_run_id=origin_provider_run,
                    source=current_source,
                    base_policy=policy,
                )
            )
            if (
                final_policy != effective_policy
                or final_origin_binding != origin_binding
                or not np.array_equal(final_origins, explicit_origins)
            ):
                raise RuntimeError(
                    "Explicit crop-origin provider changed during publication."
                )
        final_prepared = PreparedCropGeometrySnapshot(
            dimensions=prepared.dimensions,
            policy=prepared.policy,
            source=prepared.source,
            pixel_authority=prepared.pixel_authority,
            arrays=prepared.arrays,
            source_manifest=current_source.manifest,
            source_arrays=current_source.arrays,
        )
        final_manifest, final_direct, final_consolidated = _build_and_persist_manifest(
            archive_path=archive,
            run_id=candidate_id,
            publication=publication,
            prepared=final_prepared,
        )
        final_validation = _validate_candidate(
            run_path=target,
            manifest=final_manifest,
            direct=final_direct,
            consolidated=final_consolidated,
            prepared=final_prepared,
        )
        if not final_validation["valid"]:
            raise RuntimeError(
                f"Published crop candidate is invalid: {final_validation}"
            )

        final_root = open_zarr_root(archive, mode="r")
        _require_unselected(final_root, run_id=candidate_id)
        if dict(final_root.attrs) != root_attrs_before:
            raise RuntimeError("Crop publication changed root archive attributes.")
        final_parent = final_root["crop_runs"]
        if dict(final_parent.attrs) != expected_crop_parent_attrs:
            raise RuntimeError("Crop publication changed crop selector attributes.")

        result = {
            "schema_id": CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
            "schema_version": CROP_SNAPSHOT_PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "published_at_utc": utc_now(),
            "analysis_zarr": str(archive),
            "run_id": candidate_id,
            "group_path": f"crop_runs/{candidate_id}",
            "run_manifest_digest": final_manifest["payload_digest"],
            "logical_content_digest": final_manifest["payload"]["logical_content"][
                "digest"
            ],
            "source_refined_run_id": source.run_id,
            "source_refined_manifest_digest": source.manifest["payload_digest"],
            "source_refined_selection_mode": source.selection_mode,
            "registered_gate_requirement": gate_requirement,
            "registered_gate_run": (
                None if gate_evidence is None else gate_evidence.get("gate_run")
            ),
            "registered_gate_applied": bool(
                gate_evidence is not None and gate_evidence.get("applied") is True
            ),
            "source_pixel_authority_digest": pixels.binding_document_digest,
            "source_video_path": (
                None
                if pixels.source_video_path is None
                else str(pixels.source_video_path)
            ),
            "source_video_paths": [str(item) for item in pixels.source_video_paths],
            "source_index_paths": [str(item) for item in pixels.source_index_paths],
            "source_pixel_provider_profile": pixels.binding_document[
                "provider_profile"
            ],
            "geometry_origin_binding": origin_binding,
            "storage_profile_id": profile.profile_id,
            "selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
            "registry_updated": False,
            "node_local_materialization": {
                "session_path": str(session),
                "retained_after_success": bool(keep_scratch),
                "writer_receipt": {
                    "publication_seconds": publication.receipt["publication_seconds"],
                    "phase_seconds": publication.receipt["phase_seconds"],
                    "per_array_write_seconds": publication.receipt[
                        "per_array_write_seconds"
                    ],
                    "writes": publication.receipt["writes"],
                    "logical_hashes": publication.receipt["logical_hashes"],
                    "consolidation": publication.receipt["consolidation"],
                },
            },
            "atomic_publication": atomic_receipt,
            "validation": {
                "local_errors": [],
                "published_errors": [],
                "direct_consolidated_metadata_equal": True,
                "source_and_pixel_authorities_reverified": True,
                "root_attributes_unchanged": True,
                "crop_selector_attributes_unchanged": True,
            },
            "total_seconds": float(time.perf_counter() - started),
        }
        success = True
        return json_attr_safe(result)
    except BaseException as exc:
        if imported:
            _mark_failed_import(target, error=exc)
            try:
                consolidate_metadata_capture_expected_warnings(archive)
            except Exception:
                pass
        raise
    finally:
        if session.exists() and success and not keep_scratch:
            shutil.rmtree(session)


__all__ = [
    "CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID",
    "CROP_SNAPSHOT_PUBLICATION_SCHEMA_VERSION",
    "publish_crop_geometry_from_explicit_refined_candidate",
    "publish_crop_geometry_production_candidate",
]

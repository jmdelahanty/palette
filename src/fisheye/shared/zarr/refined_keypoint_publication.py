"""Selector-ineligible immutable publication for refined keypoint v2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.keypoint_schema import REFINED_KEYPOINT_SCHEMA_V2
from fisheye.shared.zarr.keypoint_quality_manifest import (
    quality_profile_from_manifest,
)
from fisheye.shared.zarr.keypoint_quality_schema import KeypointQualityDimensions
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    RefinedKeypointSnapshotIdentity,
    RefinedKeypointSourceBindings,
    build_refined_keypoint_source_bindings,
    build_refined_keypoint_run_manifest,
    refined_keypoint_code_maps_from_manifest,
    refined_keypoint_logical_content_document,
    refined_keypoint_snapshot_identity_from_manifest,
    validate_refined_keypoint_publication,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    PreparedRefinedKeypointSnapshot,
)
from fisheye.shared.zarr.refined_keypoint_storage import (
    RefinedKeypointStoragePlanSet,
    plan_refined_keypoint_storage,
)
from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    require_runs_parent,
)

REFINED_KEYPOINT_SHADOW_SCHEMA_ID = "palette.refined_keypoint.shadow_publication"
REFINED_KEYPOINT_SHADOW_SCHEMA_VERSION = 1
DEFAULT_REFINED_KEYPOINT_SHADOW_ROOT = Path(
    "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
    "refined_keypoints"
)
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


@dataclass(frozen=True)
class RefinedKeypointShadowPublication:
    output_path: Path
    run_id: str
    prepared: PreparedRefinedKeypointSnapshot
    source: RefinedKeypointSourceBindings
    raw_manifest: Mapping[str, Any]
    quality_manifest: Mapping[str, Any]
    crop_manifest: Mapping[str, Any]
    raw_arrays: Mapping[str, Any]
    quality_arrays: Mapping[str, Any]
    source_crop_arrays: Mapping[str, Any]
    identity: RefinedKeypointSnapshotIdentity
    review_state_map: Mapping[int, str]
    reason_code_map: Mapping[int, str]
    retired_instance_keys: tuple[int, ...]
    plans: RefinedKeypointStoragePlanSet
    manifest: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float
    parent_manifest: Mapping[str, Any] | None = None
    parent_arrays: Mapping[str, Any] | None = None
    parent_retired_instance_keys: tuple[int, ...] | None = None


def require_safe_refined_keypoint_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_REFINED_KEYPOINT_SHADOW_ROOT,
) -> Path:
    root = shadow_root.expanduser().resolve()
    output = destination.expanduser().resolve()
    if output == root:
        raise ValueError("Refined-keypoint shadow destination cannot equal its root.")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Refined-keypoint shadow destination must be below shadow_root."
        ) from exc
    if output.exists():
        raise FileExistsError(
            f"Refined-keypoint shadow destination already exists: {output}"
        )
    return output


def _normalized_retired_keys(values: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 or value > np.iinfo(np.uint64).max for value in normalized):
        raise ValueError("Retired instance keys must be uint64-compatible.")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("Retired instance keys must be sorted and unique.")
    return normalized


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: StoragePlan,
) -> None:
    unit_shape = plan.shard_shape or plan.chunk_shape
    if unit_shape is None:
        raise ValueError("Refined-keypoint publication does not support scalar arrays.")
    unit_rows = max(1, int(unit_shape[0]))
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def refined_keypoint_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: RefinedKeypointStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"refined_keypoints_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path /= relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")

    archive_root = _read_strict_json(output_path / "zarr.json")
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Refined-keypoint shadow lacks root consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Refined-keypoint consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Refined-keypoint consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(
                f"Refined-keypoint consolidated metadata lacks {full_path!r}."
            )
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def validate_refined_keypoint_shadow_publication(
    publication: RefinedKeypointShadowPublication,
) -> tuple[str, ...]:
    try:
        direct, consolidated = refined_keypoint_metadata_declaration_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
        run = zarr.open_group(
            str(
                publication.output_path / "refined_keypoints_runs" / publication.run_id
            ),
            mode="r",
            use_consolidated=False,
        )
        arrays = {path: run[path] for path in REFINED_KEYPOINT_SCHEMA_V2.binding_paths}
    except (OSError, TypeError, ValueError) as exc:
        return (f"refined-keypoint shadow reopen failed: {exc}",)

    errors = list(
        validate_refined_keypoint_publication(
            publication.manifest,
            arrays=arrays,
            source_crop_arrays=publication.source_crop_arrays,
            raw_manifest=publication.raw_manifest,
            quality_manifest=publication.quality_manifest,
            crop_manifest=publication.crop_manifest,
            raw_arrays=publication.raw_arrays,
            quality_arrays=publication.quality_arrays,
            retired_instance_keys=publication.retired_instance_keys,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            parent_manifest=publication.parent_manifest,
            parent_arrays=publication.parent_arrays,
            parent_retired_instance_keys=publication.parent_retired_instance_keys,
        )
    )
    if run.attrs.get("status") != "complete":
        errors.append("refined-keypoint shadow status is not complete")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("refined-keypoint shadow is not selector-ineligible")
    family = zarr.open_group(
        str(publication.output_path / "refined_keypoints_runs"),
        mode="r",
        use_consolidated=False,
    )
    selected = [
        name
        for name in _SELECTOR_ATTRIBUTES
        if family.attrs.get(name) == publication.run_id
    ]
    if selected:
        errors.append(f"refined-keypoint shadow is selected by {selected!r}")
    return tuple(errors)


def publish_selector_ineligible_refined_keypoint_snapshot(
    prepared: PreparedRefinedKeypointSnapshot,
    *,
    source: RefinedKeypointSourceBindings,
    raw_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
    quality_arrays: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    identity: RefinedKeypointSnapshotIdentity,
    review_state_map: Mapping[int, str],
    reason_code_map: Mapping[int, str],
    retired_instance_keys: Sequence[int] = (),
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_REFINED_KEYPOINT_SHADOW_ROOT,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "refined_keypoint_v2_shadow",
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
    parent_retired_instance_keys: Sequence[int] | None = None,
) -> RefinedKeypointShadowPublication:
    """Write, consolidate, and reopen-gate one refined-keypoint-v2 snapshot."""

    output_path = require_safe_refined_keypoint_shadow_destination(
        destination,
        shadow_root=shadow_root,
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
    if "/" in str(run_id) or not str(run_id).strip():
        raise ValueError("run_id must be one nonempty archive group name.")
    retired = _normalized_retired_keys(retired_instance_keys)
    parent_retired = (
        None
        if parent_retired_instance_keys is None
        else _normalized_retired_keys(parent_retired_instance_keys)
    )
    if (parent_manifest is None) != (parent_arrays is None):
        raise ValueError("Parent manifest and arrays must be supplied together.")
    if parent_manifest is None and parent_retired is not None:
        raise ValueError("Parent retired keys require parent manifest and arrays.")

    REFINED_KEYPOINT_SCHEMA_V2.require(
        prepared.arrays,
        dimensions=prepared.dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=source.skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    plans = plan_refined_keypoint_storage(
        prepared.dimensions,
        profile=storage_profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": REFINED_KEYPOINT_SHADOW_SCHEMA_ID,
            "schema_version": REFINED_KEYPOINT_SHADOW_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    family = require_runs_parent(
        root,
        "refined_keypoints_runs",
        completion_epoch=COMPLETION_EPOCH_STRICT,
    )
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = family.create_group(str(run_id))
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "artifact_class": "reviewed_keypoint_authority_candidate",
            "keypoint_authority": False,
            "logical_schema": REFINED_KEYPOINT_SCHEMA_V2.as_manifest(
                dimensions=prepared.dimensions
            ),
            "storage_plan": plans.as_manifest(),
            "source_bindings": source.as_manifest(),
            "snapshot_identity": identity.as_manifest(),
            "review_state_map": {
                str(code): label for code, label in sorted(review_state_map.items())
            },
            "reason_code_map": {
                str(code): label for code, label in sorted(reason_code_map.items())
            },
        }
    )

    destination_arrays: dict[str, Any] = {}
    bindings = {
        binding.path: binding for binding in REFINED_KEYPOINT_SCHEMA_V2.bindings
    }
    phase_started = time.perf_counter()
    for entry in plans.entries:
        path = entry.rule.path
        values = np.asarray(prepared.arrays[path])
        binding = bindings[path]
        contract = REFINED_KEYPOINT_SCHEMA_V2.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        array = create_array_from_plan(
            run,
            name=path,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={
                "benchmark_only": True,
                "selector_eligible": False,
                "artifact_class": "reviewed_keypoint_authority_candidate",
            },
        )
        _write_by_physical_units(array, values, plan=entry.plan)
        destination_arrays[path] = array
    phase_seconds["create_and_write_arrays"] = time.perf_counter() - phase_started

    REFINED_KEYPOINT_SCHEMA_V2.require(
        destination_arrays,
        dimensions=prepared.dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=source.skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    run.attrs["status"] = "complete"

    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = refined_keypoint_metadata_declaration_maps(
        output_path,
        run_id=str(run_id),
        plans=plans,
    )
    phase_seconds["first_consolidation"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    manifest = build_refined_keypoint_run_manifest(
        run_id=str(run_id),
        dimensions=prepared.dimensions,
        source=source,
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
        storage_plan=plans,
        identity=identity,
        arrays=destination_arrays,
        source_crop_arrays=source_crop_arrays,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
        retired_instance_keys=retired,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
    )
    run.attrs[REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE] = manifest
    phase_seconds["build_manifest"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = refined_keypoint_metadata_declaration_maps(
        output_path,
        run_id=str(run_id),
        plans=plans,
    )
    errors = validate_refined_keypoint_publication(
        manifest,
        arrays=destination_arrays,
        source_crop_arrays=source_crop_arrays,
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
        raw_arrays=raw_arrays,
        quality_arrays=quality_arrays,
        retired_instance_keys=retired,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
        parent_retired_instance_keys=parent_retired,
    )
    phase_seconds["final_consolidation_and_gate"] = time.perf_counter() - phase_started
    if errors:
        run.attrs.update(
            {
                "status": "failed",
                "stage_selector_eligible": False,
                "publication_errors": list(errors),
            }
        )
        raise RuntimeError(
            "Refined-keypoint publication gate failed: " + "; ".join(errors)
        )

    publication = RefinedKeypointShadowPublication(
        output_path=output_path,
        run_id=str(run_id),
        prepared=prepared,
        source=source,
        raw_manifest=dict(raw_manifest),
        quality_manifest=dict(quality_manifest),
        crop_manifest=dict(crop_manifest),
        raw_arrays=raw_arrays,
        quality_arrays=quality_arrays,
        source_crop_arrays=dict(source_crop_arrays),
        identity=identity,
        review_state_map=dict(review_state_map),
        reason_code_map=dict(reason_code_map),
        retired_instance_keys=retired,
        plans=plans,
        manifest=manifest,
        phase_seconds=phase_seconds,
        elapsed_seconds=time.perf_counter() - started,
        parent_manifest=(None if parent_manifest is None else dict(parent_manifest)),
        parent_arrays=parent_arrays,
        parent_retired_instance_keys=parent_retired,
    )
    reopen_errors = validate_refined_keypoint_shadow_publication(publication)
    if reopen_errors:
        raise RuntimeError(
            "Reopened refined-keypoint publication gate failed: "
            + "; ".join(reopen_errors)
        )
    return publication


def republish_selector_ineligible_refined_keypoint_snapshot(
    *,
    source_refined_manifest: Mapping[str, Any],
    source_refined_arrays: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
    quality_arrays: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_REFINED_KEYPOINT_SHADOW_ROOT,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "refined_keypoint_v2_contract_republication",
) -> RefinedKeypointShadowPublication:
    """Republish unchanged logical arrays under the current exact v2 contract.

    The only accepted legacy transition is source-bindings v1 to v2, whose
    sole semantic addition is inline skeleton semantics.  All arrays,
    logical-content declarations, identity, and code registries remain exact.
    This is an immutable companion publication; it never edits the source.
    """

    legacy_error = "Refined-keypoint skeleton semantics are not exact."
    source_errors = validate_refined_keypoint_run_manifest(source_refined_manifest)
    if source_errors not in ((), (legacy_error,)):
        raise ValueError(
            "Source refined-keypoint manifest has unsupported defects: "
            + "; ".join(source_errors)
        )

    source = build_refined_keypoint_source_bindings(
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
    )
    payload = source_refined_manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise TypeError("Source refined-keypoint manifest payload is missing.")
    observed_bindings = payload.get("source_bindings")
    if not isinstance(observed_bindings, Mapping):
        raise TypeError("Source refined-keypoint bindings are missing.")
    expected_bindings = source.as_manifest()
    if source_errors:
        if (
            observed_bindings.get("schema_version") != 1
            or expected_bindings.get("schema_version") != 2
        ):
            raise ValueError(
                "Legacy skeleton-semantics omission must be source-bindings v1."
            )
        expected_bindings = dict(expected_bindings)
        expected_bindings["schema_version"] = 1
        skeleton = dict(expected_bindings["skeleton"])
        skeleton.pop("semantics")
        expected_bindings["skeleton"] = skeleton
    if dict(observed_bindings) != expected_bindings:
        raise ValueError(
            "Source refined-keypoint bindings differ beyond the accepted "
            "skeleton-semantics omission."
        )

    identity = refined_keypoint_snapshot_identity_from_manifest(source_refined_manifest)
    review_state_map, reason_code_map = refined_keypoint_code_maps_from_manifest(
        source_refined_manifest
    )
    quality_payload = quality_manifest.get("payload")
    if not isinstance(quality_payload, Mapping):
        raise TypeError("Quality manifest payload is missing.")
    quality_logical = quality_payload.get("logical_schema")
    if not isinstance(quality_logical, Mapping):
        raise TypeError("Quality logical schema is missing.")
    quality_profile_raw = quality_logical.get("profile")
    if not isinstance(quality_profile_raw, Mapping):
        raise TypeError("Quality profile is missing.")
    quality_profile = quality_profile_from_manifest(quality_profile_raw)
    dimensions = source.dimensions
    quality_dimensions = KeypointQualityDimensions(
        n_frames=dimensions.n_frames,
        n_instances=dimensions.n_instances,
        n_keypoints=dimensions.n_keypoints,
        n_keypoint_metrics=len(quality_profile.keypoint_metrics),
        n_pose_metrics=len(quality_profile.pose_metrics),
    )
    arrays = {
        path: source_refined_arrays[path]
        for path in REFINED_KEYPOINT_SCHEMA_V2.binding_paths
    }
    REFINED_KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=source.skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    observed_content = refined_keypoint_logical_content_document(
        arrays,
        dimensions=dimensions,
        source=source,
        identity=identity,
        source_crop_arrays=source_crop_arrays,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    source_content = payload.get("logical_content")
    if (
        not isinstance(source_content, Mapping)
        or source_content.get("document") != observed_content
    ):
        raise ValueError(
            "Source refined-keypoint logical-content declaration differs from arrays."
        )

    prepared = PreparedRefinedKeypointSnapshot(
        dimensions=dimensions,
        quality_dimensions=quality_dimensions,
        quality_profile=quality_profile,
        decisions=(),
        arrays=arrays,
    )
    publication = publish_selector_ineligible_refined_keypoint_snapshot(
        prepared,
        source=source,
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
        raw_arrays=raw_arrays,
        quality_arrays=quality_arrays,
        source_crop_arrays=source_crop_arrays,
        identity=identity,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
        destination=destination,
        run_id=run_id,
        shadow_root=shadow_root,
        storage_profile=storage_profile,
        created_by=created_by,
    )
    if publication.manifest["payload"]["logical_content"] != source_content:
        raise RuntimeError(
            "Republished refined-keypoint logical content differs from source."
        )
    return publication


__all__ = [
    "DEFAULT_REFINED_KEYPOINT_SHADOW_ROOT",
    "REFINED_KEYPOINT_SHADOW_SCHEMA_ID",
    "REFINED_KEYPOINT_SHADOW_SCHEMA_VERSION",
    "RefinedKeypointShadowPublication",
    "publish_selector_ineligible_refined_keypoint_snapshot",
    "republish_selector_ineligible_refined_keypoint_snapshot",
    "refined_keypoint_metadata_declaration_maps",
    "require_safe_refined_keypoint_shadow_destination",
    "validate_refined_keypoint_shadow_publication",
]

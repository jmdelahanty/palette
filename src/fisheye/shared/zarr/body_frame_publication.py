"""Selector-ineligible immutable publication for body-frame v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.body_frame_manifest import (
    BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
    build_body_frame_run_manifest,
    validate_body_frame_publication,
)
from fisheye.shared.zarr.body_frame_producer import PreparedBodyFrameSnapshot
from fisheye.shared.zarr.body_frame_schema import BODY_FRAME_SCHEMA_V1
from fisheye.shared.zarr.body_frame_storage import (
    BodyFrameStoragePlanSet,
    plan_body_frame_storage,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)


BODY_FRAME_SHADOW_SCHEMA_ID = "palette.body_frame.shadow_publication"
BODY_FRAME_SHADOW_SCHEMA_VERSION = 1
DEFAULT_BODY_FRAME_SHADOW_ROOT = Path(
    "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/body_frame"
)
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


@dataclass(frozen=True)
class BodyFrameShadowPublication:
    output_path: Path
    run_id: str
    prepared: PreparedBodyFrameSnapshot
    source_manifest: Mapping[str, Any]
    plans: BodyFrameStoragePlanSet
    manifest: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


def require_safe_body_frame_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_BODY_FRAME_SHADOW_ROOT,
) -> Path:
    root = shadow_root.expanduser().resolve()
    output = destination.expanduser().resolve()
    if output == root:
        raise ValueError("Body-frame shadow destination cannot equal its root.")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Body-frame shadow destination must be below shadow_root."
        ) from exc
    if output.exists():
        raise FileExistsError(f"Body-frame shadow destination already exists: {output}")
    return output


def _write_by_physical_units(
    destination: Any, values: np.ndarray, *, plan: StoragePlan
) -> None:
    unit_shape = plan.shard_shape or plan.chunk_shape
    if unit_shape is None:
        raise ValueError("Body-frame publication does not support scalar arrays.")
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


def body_frame_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: BodyFrameStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"analysis/body_frame_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path /= relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")
    archive_root = _read_strict_json(output_path / "zarr.json")
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Body-frame shadow lacks root consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Body-frame consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Body-frame consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Body-frame consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def validate_body_frame_shadow_publication(
    publication: BodyFrameShadowPublication,
) -> tuple[str, ...]:
    try:
        direct, consolidated = body_frame_metadata_declaration_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
        run = zarr.open_group(
            str(
                publication.output_path
                / "analysis"
                / "body_frame_runs"
                / publication.run_id
            ),
            mode="r",
            use_consolidated=False,
        )
        arrays = {path: run[path] for path in BODY_FRAME_SCHEMA_V1.binding_paths}
    except (OSError, TypeError, ValueError) as exc:
        return (f"body-frame shadow reopen failed: {exc}",)
    errors = list(
        validate_body_frame_publication(
            publication.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
            source_arrays=publication.prepared.source_arrays,
            source_manifest=publication.source_manifest,
        )
    )
    if run.attrs.get("status") != "complete":
        errors.append("body-frame shadow status is not complete")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("body-frame shadow is not selector-ineligible")
    family = zarr.open_group(
        str(publication.output_path / "analysis" / "body_frame_runs"),
        mode="r",
        use_consolidated=False,
    )
    selected = [
        name
        for name in _SELECTOR_ATTRIBUTES
        if family.attrs.get(name) == publication.run_id
    ]
    if selected:
        errors.append(f"body-frame shadow is selected by {selected!r}")
    return tuple(errors)


def publish_selector_ineligible_body_frame_snapshot(
    prepared: PreparedBodyFrameSnapshot,
    *,
    source_manifest: Mapping[str, Any],
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_BODY_FRAME_SHADOW_ROOT,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "body_frame_shadow",
    disposition: KeypointPublicationDisposition = KeypointPublicationDisposition(),
) -> BodyFrameShadowPublication:
    """Write and validate one immutable, selector-ineligible body-frame snapshot."""

    output_path = require_safe_body_frame_shadow_destination(
        destination, shadow_root=shadow_root
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
    if "/" in str(run_id) or not str(run_id).strip():
        raise ValueError("run_id must be one nonempty archive group name.")
    if canonical_json_sha256(source_manifest) != prepared.source.manifest_digest:
        raise ValueError(
            "Source manifest differs from the prepared body-frame binding."
        )
    BODY_FRAME_SCHEMA_V1.require(
        prepared.arrays,
        dimensions=prepared.dimensions,
        source_keypoint_arrays=prepared.source_arrays,
    )
    plans = plan_body_frame_storage(prepared.dimensions, profile=storage_profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            **disposition.root_attributes(),
            "schema_id": BODY_FRAME_SHADOW_SCHEMA_ID,
            "schema_version": BODY_FRAME_SHADOW_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    analysis = root.create_group("analysis")
    family = require_runs_parent(
        analysis, "body_frame_runs", completion_epoch=COMPLETION_EPOCH_STRICT
    )
    family.attrs.update(disposition.family_attributes())
    run = family.create_group(str(run_id))
    run.attrs.update(
        {
            "status": "running",
            **disposition.run_attributes(),
            "artifact_class": "derived_keypoint_body_frame_cache",
            "keypoint_authority": False,
            "logical_schema": BODY_FRAME_SCHEMA_V1.as_manifest(
                dimensions=prepared.dimensions
            ),
            "storage_plan": plans.as_manifest(),
            "source_keypoint_snapshot": prepared.source.as_manifest(),
            "heading_recipe": prepared.recipe.as_manifest(),
        }
    )
    if disposition.production_candidate:
        mark_run_started(run, run_name=str(run_id), stage="body_frame")

    destination_arrays: dict[str, Any] = {}
    bindings = {binding.path: binding for binding in BODY_FRAME_SCHEMA_V1.bindings}
    phase_started = time.perf_counter()
    for entry in plans.entries:
        path = entry.rule.path
        values = np.asarray(prepared.arrays[path])
        binding = bindings[path]
        contract = BODY_FRAME_SCHEMA_V1.contracts.resolve(
            binding.contract_id, binding.contract_version
        )
        array = create_array_from_plan(
            run,
            name=path,
            contract=contract,
            plan=entry.plan,
            fill_value=0,
            attributes={
                **disposition.array_attributes(),
                "artifact_class": "derived_keypoint_body_frame_cache",
                "keypoint_authority": False,
            },
        )
        _write_by_physical_units(array, values, plan=entry.plan)
        destination_arrays[path] = array
    phase_seconds["create_and_write_arrays"] = time.perf_counter() - phase_started

    BODY_FRAME_SCHEMA_V1.require(
        destination_arrays,
        dimensions=prepared.dimensions,
        source_keypoint_arrays=prepared.source_arrays,
    )
    run.attrs["status"] = "complete"
    if disposition.production_candidate:
        mark_run_complete(
            run,
            run_name=str(run_id),
            run_provenance=disposition.run_provenance,
        )

    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = body_frame_metadata_declaration_maps(
        output_path, run_id=str(run_id), plans=plans
    )
    phase_seconds["first_consolidation"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    manifest = build_body_frame_run_manifest(
        run_id=str(run_id),
        dimensions=prepared.dimensions,
        source=prepared.source,
        source_manifest=source_manifest,
        recipe=prepared.recipe,
        storage_plan=plans,
        arrays=destination_arrays,
        source_arrays=prepared.source_arrays,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
    )
    run.attrs[BODY_FRAME_RUN_MANIFEST_ATTRIBUTE] = manifest
    phase_seconds["build_manifest"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    consolidate_metadata_capture_expected_warnings(output_path)
    direct, consolidated = body_frame_metadata_declaration_maps(
        output_path, run_id=str(run_id), plans=plans
    )
    errors = validate_body_frame_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=destination_arrays,
        source_arrays=prepared.source_arrays,
        source_manifest=source_manifest,
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
        if disposition.production_candidate:
            mark_run_failed(
                run,
                run_name=str(run_id),
                error="; ".join(errors),
            )
        raise RuntimeError("Body-frame publication gate failed: " + "; ".join(errors))

    publication = BodyFrameShadowPublication(
        output_path=output_path,
        run_id=str(run_id),
        prepared=prepared,
        source_manifest=dict(source_manifest),
        plans=plans,
        manifest=manifest,
        phase_seconds=phase_seconds,
        elapsed_seconds=time.perf_counter() - started,
    )
    reopen_errors = validate_body_frame_shadow_publication(publication)
    if reopen_errors:
        raise RuntimeError(
            "Reopened body-frame publication gate failed: " + "; ".join(reopen_errors)
        )
    return publication


__all__ = [
    "BODY_FRAME_SHADOW_SCHEMA_ID",
    "BODY_FRAME_SHADOW_SCHEMA_VERSION",
    "DEFAULT_BODY_FRAME_SHADOW_ROOT",
    "BodyFrameShadowPublication",
    "body_frame_metadata_declaration_maps",
    "publish_selector_ineligible_body_frame_snapshot",
    "require_safe_body_frame_shadow_destination",
    "validate_body_frame_shadow_publication",
]

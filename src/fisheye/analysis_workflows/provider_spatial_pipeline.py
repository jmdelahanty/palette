"""Narrow, authority-bound adapter for the provider spatial analytics stages.

This module is intentionally a small cross-stage boundary.  Selection and
trajectory objects remain pure in-memory contracts; their materializers remain
the only code that publishes Zarr runs.  The adapter only resolves already
published, immutable, selector-ineligible runs and constructs the exact source
bindings required by the occupancy publisher.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyGrid,
    OccupancyTimingPolicy,
    ProviderOccupancySamples,
    ProviderOccupancyV2Result,
    calculate_provider_occupancy_v2,
    occupancy_samples_from_trajectory,
)
from fisheye.analysis.provider_spatial_trajectory import (
    ProviderSpatialTrajectory,
    SelectedFrameMembership,
    selected_frame_membership_from_compiled_selection,
)
from fisheye.analysis_workflows.composable_stimulus_selection import (
    CompiledSelection,
    canonical_json,
    canonical_sha256,
)
from fisheye.analysis_workflows.materializers.composable_stimulus_selection import (
    ARRAY_MANIFEST_DIGEST_ATTR as SELECTION_ARRAY_MANIFEST_DIGEST_ATTR,
    ARRAY_MANIFEST_JSON_ATTR as SELECTION_ARRAY_MANIFEST_JSON_ATTR,
    COMPILED_SELECTION_DIGEST_ATTR,
    PARENT_PATH as SELECTION_PARENT_PATH,
    REQUESTED_JSON_ATTR,
    RESOLVED_JSON_ATTR,
    RUN_SCHEMA_ID as SELECTION_RUN_SCHEMA_ID,
    RUN_SCHEMA_VERSION as SELECTION_RUN_SCHEMA_VERSION,
    SUPPORTED_RUN_SCHEMA_VERSIONS as SELECTION_SUPPORTED_RUN_SCHEMA_VERSIONS,
    REQUESTED_JSON_ARRAY,
    TIMELINE_AUTHORITY_JSON_ARRAY,
    TIMELINE_AUTHORITY_JSON_ATTR,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    ProviderOccupancyV2SourceBindings,
)
from fisheye.analysis_workflows.materializers.provider_spatial_trajectory import (
    ARRAY_MANIFEST_ATTR as TRAJECTORY_ARRAY_MANIFEST_ATTR,
    ARRAY_MANIFEST_SHA256_ATTR as TRAJECTORY_ARRAY_MANIFEST_SHA256_ATTR,
    PARENT_PATH as TRAJECTORY_PARENT_PATH,
    RUN_MANIFEST_ATTR as TRAJECTORY_RUN_MANIFEST_ATTR,
    RUN_MANIFEST_SHA256_ATTR as TRAJECTORY_RUN_MANIFEST_SHA256_ATTR,
    RUN_SCHEMA_ID as TRAJECTORY_RUN_SCHEMA_ID,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


SOURCE_BINDING_NAMES = (
    "trajectory",
    "compiled_selection",
    "provider",
    "timing",
    "geometry",
    "transform",
    "fixed_grid_policy",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SELECTOR_NAMES = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
        "selected_run",
        "active_run",
        "current_run",
        "default_run",
    }
)


class ProviderSpatialPipelineError(ValueError):
    """Raised when cross-stage identities cannot be proven exact."""


def _fail(message: str) -> ProviderSpatialPipelineError:
    return ProviderSpatialPipelineError(message)


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{field} must be one lowercase SHA-256 digest.")
    return value


def _strict_record(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise _fail(f"{field} must be one nonempty authority record.")
    try:
        encoded = canonical_json(dict(value))
    except (TypeError, ValueError) as exc:
        raise _fail(f"{field} is not strict canonical JSON.") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise _fail(f"{field} did not canonicalize to an object.")
    return decoded


def _bound_record(record: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    normalized = _strict_record(record, field=field)
    return {
        "record": normalized,
        "sha256": canonical_json_sha256(normalized),
    }


def _stable_subrecord(value: object, *, field: str) -> dict[str, Any]:
    """Normalize a scientific identity subrecord and reject display labels."""

    record = _strict_record(value, field=field)
    if any(key in record for key in ("label", "name", "display_name")):
        raise _fail(f"{field} contains a mutable display label.")
    return record


def _exact_run_path(value: object, *, parent: str, field: str) -> str:
    if type(value) is not str or not value.startswith(f"{parent}/"):
        raise _fail(f"{field} must be one exact path below {parent!r}.")
    child = value[len(parent) + 1 :]
    if "/" in child or not _RUN_NAME_RE.fullmatch(child or ""):
        raise _fail(f"{field} must name one exact non-selector child run.")
    if child.lower() in _SELECTOR_NAMES or child.lower().startswith(
        ("latest_", "active_", "current_", "default_", "authoritative_")
    ):
        raise _fail(f"{field} cannot be a selector alias.")
    return value


def _require_attrs(attrs: Mapping[str, Any], names: tuple[str, ...], *, field: str) -> None:
    missing = [name for name in names if name not in attrs]
    if missing:
        raise _fail(f"{field} is missing required attributes: {missing!r}.")


def _array_paths(group: Any, prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for name, _array in group.arrays():
        paths.append(f"{prefix}/{name}" if prefix else str(name))
    for name, child in group.groups():
        child_prefix = f"{prefix}/{name}" if prefix else str(name)
        paths.extend(_array_paths(child, child_prefix))
    return tuple(sorted(paths))


def _selection_array_digest(values: Any) -> str:
    import hashlib

    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S"}:
        raise _fail("Selection logical arrays cannot use string/object dtypes.")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_published_arrays(
    archive: Path,
    *,
    run_path: str,
    run: Any,
    array_manifest: Mapping[str, Any],
    digest_algorithm: str,
    field: str,
) -> None:
    declarations = array_manifest.get("arrays")
    if not isinstance(declarations, list) or not declarations:
        raise _fail(f"{field} array manifest has no declarations.")
    paths: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise _fail(f"{field} array declaration is not an object.")
        path = declaration.get("path")
        if type(path) is not str or not path or path in paths:
            raise _fail(f"{field} array declaration path is invalid.")
        paths.add(path)
        if type(declaration.get("dtype")) is not str or not isinstance(
            declaration.get("shape"), list
        ):
            raise _fail(f"{field} array declaration shape/dtype is invalid.")
        _require_sha256(declaration.get("content_sha256"), field=f"{field}.{path}.content_sha256")
    if set(_array_paths(run)) != paths:
        raise _fail(f"{field} arrays differ from the exact published manifest.")
    direct_root = open_zarr_root(archive, mode="r", use_consolidated=False)
    consolidated_root = open_zarr_root(archive, mode="r", use_consolidated=True)
    direct_run = direct_root[run_path]
    consolidated_run = consolidated_root[run_path]
    for declaration in declarations:
        path = str(declaration["path"])
        direct_values = np.asarray(direct_run[path][:])
        consolidated_values = np.asarray(consolidated_run[path][:])
        if direct_values.dtype.str != declaration["dtype"] or list(direct_values.shape) != declaration["shape"]:
            raise _fail(f"{field}.{path} physical declaration differs from its manifest.")
        if direct_values.dtype != consolidated_values.dtype or direct_values.shape != consolidated_values.shape:
            raise _fail(f"{field}.{path} direct/consolidated payload shapes differ.")
        if not np.array_equal(direct_values, consolidated_values, equal_nan=True):
            raise _fail(f"{field}.{path} direct/consolidated payload bytes differ.")
        digest = (
            _selection_array_digest(direct_values)
            if digest_algorithm == "selection"
            else array_values_sha256(direct_values)
        )
        if digest != declaration["content_sha256"]:
            raise _fail(f"{field}.{path} content digest is stale.")
    try:
        validate_direct_consolidated_subtree(archive, subtree_path=run_path)
    except Exception as exc:
        raise _fail(f"{field} direct/consolidated metadata is not equivalent.") from exc


def _manifest_array_digest(
    array_manifest: Mapping[str, Any],
    *,
    path: str,
    field: str,
) -> str:
    declarations = array_manifest.get("arrays")
    if not isinstance(declarations, list):
        raise _fail(f"{field} arrays are not declared.")
    for declaration in declarations:
        if isinstance(declaration, Mapping) and declaration.get("path") == path:
            return _require_sha256(
                declaration.get("content_sha256"),
                field=f"{field}.{path}.content_sha256",
            )
    raise _fail(f"{field} is missing array {path!r}.")


def _summary_equal(left: Any, right: Any) -> bool:
    scalar_fields = (
        "occurrence_id",
        "expected_selected_frames",
        "provider_present_count",
        "provider_valid_count",
        "transform_invalid_count",
        "nonfinite_count",
        "out_of_grid_count",
        "valid_in_grid_sample_count",
        "occupancy_time_s",
    )
    if any(getattr(left, field) != getattr(right, field) for field in scalar_fields):
        return False
    return all(
        np.array_equal(getattr(left, field), getattr(right, field), equal_nan=True)
        for field in ("counts", "occupancy_fraction", "occupancy_time_by_bin_s")
    )


def _occupancy_results_equal(
    supplied: ProviderOccupancyV2Result,
    recomputed: ProviderOccupancyV2Result,
) -> bool:
    if (
        supplied.schema_id != recomputed.schema_id
        or supplied.schema_version != recomputed.schema_version
        or supplied.config_digest != recomputed.config_digest
        or supplied.edge_policy_id != recomputed.edge_policy_id
        or supplied.timing_policy_id != recomputed.timing_policy_id
        or supplied.fps_hz != recomputed.fps_hz
        or not np.array_equal(supplied.x_edges, recomputed.x_edges)
        or not np.array_equal(supplied.y_edges, recomputed.y_edges)
        or len(supplied.per_occurrence) != len(recomputed.per_occurrence)
        or not _summary_equal(supplied.pooled, recomputed.pooled)
    ):
        return False
    return all(
        _summary_equal(left, right)
        for left, right in zip(
            supplied.per_occurrence,
            recomputed.per_occurrence,
            strict=True,
        )
    )


def _require_complete_candidate(
    run: Any,
    *,
    field: str,
    schema_id: str,
) -> dict[str, Any]:
    attrs = dict(run.attrs)
    if attrs.get("schema_id") != schema_id:
        raise _fail(f"{field} has an unexpected schema identity.")
    if attrs.get("stage_selector_eligible") is not False:
        raise _fail(f"{field} is selector eligible.")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise _fail(f"{field} is not complete.")
    if any(
        name in attrs
        for name in (
            "latest",
            "latest_complete",
            "latest_pending",
            "authoritative_run",
            "selected_run",
            "active_run",
            "current_run",
        )
    ):
        raise _fail(f"{field} contains selector attributes.")
    return attrs


def _selection_source_record(
    root: Any,
    *,
    archive: Path,
    run_path: str,
    compiled: CompiledSelection,
) -> dict[str, Any]:
    _exact_run_path(run_path, parent=SELECTION_PARENT_PATH, field="selection_run_path")
    try:
        run = root[run_path]
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"Published selection run is missing: {run_path!r}.") from exc
    attrs = _require_complete_candidate(
        run,
        field="selection run",
        schema_id=SELECTION_RUN_SCHEMA_ID,
    )
    if attrs.get("schema_version", 1) not in SELECTION_SUPPORTED_RUN_SCHEMA_VERSIONS:
        raise _fail("Published selection has an unsupported schema version.")
    child_name = run_path.rsplit("/", 1)[1]
    if attrs.get("palette_run_name") != child_name:
        raise _fail("Selection run name does not match its path.")
    required_attrs = [
            "selection_id",
            "request_digest",
            "resolved_digest",
            COMPILED_SELECTION_DIGEST_ATTR,
            "timeline_authority_sha256",
            SELECTION_ARRAY_MANIFEST_JSON_ATTR,
            SELECTION_ARRAY_MANIFEST_DIGEST_ATTR,
        ]
    compact = attrs.get("schema_version") == SELECTION_RUN_SCHEMA_VERSION
    if compact:
        required_attrs.extend(("selection_summary", "provenance_array_paths"))
    else:
        required_attrs.extend(
            (REQUESTED_JSON_ATTR, RESOLVED_JSON_ATTR, TIMELINE_AUTHORITY_JSON_ATTR)
        )
    _require_attrs(
        attrs,
        tuple(required_attrs),
        field="selection run",
    )
    if attrs["selection_id"] != compiled.selection_id:
        raise _fail("Published selection has a different selection_id.")
    if attrs["request_digest"] != compiled.request_digest:
        raise _fail("Published selection has a stale request digest.")
    if attrs["resolved_digest"] != compiled.resolved_digest:
        raise _fail("Published selection has a stale resolved digest.")
    if attrs[COMPILED_SELECTION_DIGEST_ATTR] != canonical_sha256(compiled.to_dict()):
        raise _fail("Published selection has a stale compiled-selection digest.")
    if attrs["timeline_authority_sha256"] != canonical_sha256(
        compiled.authority.to_dict()
    ):
        raise _fail("Published selection has a stale timeline-authority digest.")
    if compact:
        for path, expected in (
            (REQUESTED_JSON_ARRAY, canonical_json(compiled.requested)),
            (
                TIMELINE_AUTHORITY_JSON_ARRAY,
                canonical_json(compiled.authority.to_dict()),
            ),
        ):
            try:
                values = np.asarray(run[path][:])
                observed = values.tobytes().decode("utf-8")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
                raise _fail(f"Published selection provenance array {path!r} is unreadable.") from exc
            if values.dtype != np.dtype("uint8") or values.ndim != 1 or observed != expected:
                raise _fail(f"Published selection provenance array {path!r} is stale.")
        if any(
            name in attrs
            for name in (REQUESTED_JSON_ATTR, RESOLVED_JSON_ATTR, TIMELINE_AUTHORITY_JSON_ATTR)
        ):
            raise _fail("Compact selection contains legacy cardinality-scaled JSON attrs.")
    else:
        if attrs[REQUESTED_JSON_ATTR] != canonical_json(compiled.requested):
            raise _fail("Published selection requested JSON is stale.")
        if attrs[RESOLVED_JSON_ATTR] != canonical_json(compiled.resolved_payload()):
            raise _fail("Published selection resolved JSON is stale.")
        if attrs[TIMELINE_AUTHORITY_JSON_ATTR] != canonical_json(
            compiled.authority.to_dict()
        ):
            raise _fail("Published selection timeline-authority JSON is stale.")
    try:
        array_manifest = json.loads(attrs[SELECTION_ARRAY_MANIFEST_JSON_ATTR])
    except (TypeError, ValueError) as exc:
        raise _fail("Published selection array manifest is not valid JSON.") from exc
    if not isinstance(array_manifest, Mapping) or attrs[SELECTION_ARRAY_MANIFEST_DIGEST_ATTR] != canonical_json_sha256(array_manifest):
        raise _fail("Published selection array manifest digest is stale.")
    _validate_published_arrays(
        archive,
        run_path=run_path,
        run=run,
        array_manifest=array_manifest,
        digest_algorithm="selection",
        field="selection run",
    )
    return {
        "schema_id": attrs["schema_id"],
        "schema_version": attrs.get("schema_version"),
        "run_path": run_path,
        "run_name": child_name,
        "recording_id": compiled.authority.recording_id,
        "timeline_id": compiled.authority.timeline_id,
        "selection_id": compiled.selection_id,
        "request_digest": attrs["request_digest"],
        "resolved_digest": attrs["resolved_digest"],
        "compiled_selection_sha256": attrs[COMPILED_SELECTION_DIGEST_ATTR],
        "timeline_authority_sha256": attrs["timeline_authority_sha256"],
        "logical_array_manifest_sha256": attrs[SELECTION_ARRAY_MANIFEST_DIGEST_ATTR],
        "selection_summary": dict(attrs.get("selection_summary", {})),
        "provenance_array_paths": dict(attrs.get("provenance_array_paths", {})),
        "status": attrs[RUN_COMPLETION_STATUS_ATTR],
        "stage_selector_eligible": False,
    }


def _trajectory_source_record(
    root: Any,
    *,
    archive: Path,
    run_path: str,
    trajectory: ProviderSpatialTrajectory,
    selection_record: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_run_path(run_path, parent=TRAJECTORY_PARENT_PATH, field="trajectory_run_path")
    try:
        run = root[run_path]
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"Published trajectory run is missing: {run_path!r}.") from exc
    attrs = _require_complete_candidate(
        run,
        field="trajectory run",
        schema_id=TRAJECTORY_RUN_SCHEMA_ID,
    )
    child_name = run_path.rsplit("/", 1)[1]
    _require_attrs(
        attrs,
        (
            TRAJECTORY_RUN_MANIFEST_ATTR,
            TRAJECTORY_RUN_MANIFEST_SHA256_ATTR,
            TRAJECTORY_ARRAY_MANIFEST_ATTR,
            TRAJECTORY_ARRAY_MANIFEST_SHA256_ATTR,
        ),
        field="trajectory run",
    )
    run_manifest = attrs[TRAJECTORY_RUN_MANIFEST_ATTR]
    array_manifest = attrs[TRAJECTORY_ARRAY_MANIFEST_ATTR]
    if not isinstance(run_manifest, Mapping) or not isinstance(array_manifest, Mapping):
        raise _fail("Published trajectory manifests must be JSON objects.")
    if attrs[TRAJECTORY_RUN_MANIFEST_SHA256_ATTR] != canonical_json_sha256(run_manifest):
        raise _fail("Published trajectory run manifest digest is stale.")
    if attrs[TRAJECTORY_ARRAY_MANIFEST_SHA256_ATTR] != canonical_json_sha256(array_manifest):
        raise _fail("Published trajectory array manifest digest is stale.")
    _validate_published_arrays(
        archive,
        run_path=run_path,
        run=run,
        array_manifest=array_manifest,
        digest_algorithm="trajectory",
        field="trajectory run",
    )
    if run_manifest.get("run_path") != run_path or run_manifest.get("run_name") != child_name:
        raise _fail("Published trajectory manifest path is inconsistent.")
    if run_manifest.get("stage_selector_eligible") is not False:
        raise _fail("Published trajectory manifest is selector eligible.")
    if run_manifest.get("selection_sha256") != trajectory.selection.sha256:
        raise _fail("Published trajectory selection identity is stale.")
    if run_manifest.get("selection", {}).get("selection_authority_id") != selection_record["resolved_digest"]:
        raise _fail("Published trajectory is bound to a different selection authority.")
    if run_manifest.get("authorities") != trajectory.authorities.as_record():
        raise _fail("Published trajectory authorities differ from the supplied trajectory.")
    if run_manifest.get("transform_sha256") != trajectory.transform.sha256:
        raise _fail("Published trajectory transform identity is stale.")
    if run_manifest.get("source_rows_sha256") != trajectory.source_rows_sha256:
        raise _fail("Published trajectory source-row identity is stale.")
    if run_manifest.get("trajectory_sha256") != trajectory.trajectory_sha256:
        raise _fail("Published trajectory result identity is stale.")
    track_key_digest = _manifest_array_digest(
        array_manifest,
        path="track_sample_key",
        field="trajectory array manifest",
    )
    acquisition_frame_digest = _manifest_array_digest(
        array_manifest,
        path="acquisition_frame",
        field="trajectory array manifest",
    )
    if track_key_digest != array_values_sha256(trajectory.track_sample_key):
        raise _fail("Published trajectory track_sample_key identity is stale.")
    if acquisition_frame_digest != array_values_sha256(trajectory.acquisition_frame):
        raise _fail("Published trajectory acquisition-frame identity is stale.")
    denominator = run_manifest.get("selected_frame_denominator")
    if not isinstance(denominator, Mapping) or denominator.get(
        "array_path"
    ) != "selection/acquisition_frame":
        raise _fail("Published trajectory selected-frame denominator is malformed.")
    denominator_digest = _require_sha256(
        denominator.get("content_sha256"),
        field="trajectory selected-frame denominator content_sha256",
    )
    if denominator_digest != array_values_sha256(trajectory.selection.acquisition_frames):
        raise _fail("Published trajectory selected-frame denominator is stale.")
    return {
        "schema_id": attrs["schema_id"],
        "schema_version": attrs.get("schema_version"),
        "run_path": run_path,
        "run_name": child_name,
        "recording_id": trajectory.authorities.recording_id,
        "provider_id": trajectory.authorities.provider_id,
        "estimator_id": trajectory.authorities.estimator_id,
        "source_id": trajectory.authorities.source_id,
        "selection_authority_id": trajectory.authorities.selection_authority_id,
        "selection_sha256": run_manifest["selection_sha256"],
        "transform_sha256": run_manifest["transform_sha256"],
        "run_manifest_sha256": attrs[TRAJECTORY_RUN_MANIFEST_SHA256_ATTR],
        "array_manifest_sha256": attrs[TRAJECTORY_ARRAY_MANIFEST_SHA256_ATTR],
        "track_sample_key_sha256": track_key_digest,
        "acquisition_frame_sha256": acquisition_frame_digest,
        "selected_frame_denominator": dict(denominator),
        "row_axis": run_manifest["row_axis"],
        "source_rows_sha256": run_manifest["source_rows_sha256"],
        "trajectory_sha256": run_manifest["trajectory_sha256"],
        "policy_id": run_manifest["policy_id"],
        "status": attrs[RUN_COMPLETION_STATUS_ATTR],
        "stage_selector_eligible": False,
    }


def _compare(record: Mapping[str, Any], expected: Mapping[str, Any], *, field: str) -> None:
    for name, value in expected.items():
        if record.get(name) != value:
            raise _fail(f"{field}.{name} does not match the authoritative result.")


def _require_stable_text(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.lower() in _SELECTOR_NAMES
    ):
        raise _fail(f"{field} must be one exact immutable identity.")
    return value


def compiled_selection_membership(compiled: CompiledSelection) -> SelectedFrameMembership:
    """Convert one exact compiled selection without dropping overlap roles."""

    if type(compiled) is not CompiledSelection:
        raise TypeError("compiled must be one exact CompiledSelection.")
    expected_request = canonical_sha256(dict(compiled.requested))
    expected_resolved = canonical_sha256(compiled.resolved_payload())
    if compiled.request_digest != expected_request:
        raise _fail("Compiled selection request digest is stale.")
    if compiled.resolved_digest != expected_resolved:
        raise _fail("Compiled selection resolved digest is stale.")
    return selected_frame_membership_from_compiled_selection(compiled)


def occupancy_samples_from_provider_trajectory(
    trajectory: ProviderSpatialTrajectory,
    *,
    selection: SelectedFrameMembership | None = None,
) -> ProviderOccupancySamples:
    """Convert trajectory rows while preserving order, states, and denominator."""

    if type(trajectory) is not ProviderSpatialTrajectory:
        raise TypeError("trajectory must be one exact ProviderSpatialTrajectory.")
    if selection is not None:
        if type(selection) is not SelectedFrameMembership:
            raise TypeError("selection must be one exact SelectedFrameMembership.")
        if trajectory.selection.sha256 != selection.sha256:
            raise _fail("Trajectory and supplied complete selection identities differ.")
    samples = occupancy_samples_from_trajectory(trajectory)
    if samples.expected_occurrence_ids != trajectory.selection.occurrence_ids:
        raise _fail("Occupancy denominator was not copied from the complete selection.")
    return samples


def build_provider_occupancy_v2_source_bindings(
    analysis_zarr: str | Path,
    *,
    selection_run_path: str,
    trajectory_run_path: str,
    compiled_selection: CompiledSelection,
    trajectory: ProviderSpatialTrajectory,
    result: ProviderOccupancyV2Result,
    provider_authority: Mapping[str, Any],
    timing_authority: Mapping[str, Any],
    geometry_authority: Mapping[str, Any],
    transform_authority: Mapping[str, Any],
    fixed_grid_policy_authority: Mapping[str, Any],
) -> ProviderOccupancyV2SourceBindings:
    """Build the exact seven occupancy bindings from two published runs.

    Every run read here is consolidated metadata from a published immutable
    source.  This function never advances a selector or writes the archive.
    """

    if type(result) is not ProviderOccupancyV2Result:
        raise TypeError("result must be one exact ProviderOccupancyV2Result.")
    membership = compiled_selection_membership(compiled_selection)
    if trajectory.selection.sha256 != membership.sha256:
        raise _fail("Trajectory selection does not equal the compiled selection.")
    if trajectory.authorities.selection_authority_id != compiled_selection.resolved_digest:
        raise _fail("Trajectory selection authority does not equal the compiled selection.")
    try:
        recomputed_result = calculate_provider_occupancy_v2(
            occupancy_samples_from_trajectory(trajectory),
            OccupancyGrid(result.x_edges, result.y_edges),
            OccupancyTimingPolicy(result.fps_hz, result.timing_policy_id),
        )
    except (TypeError, ValueError) as exc:
        raise _fail("Supplied occupancy result cannot be recomputed from the trajectory.") from exc
    if not _occupancy_results_equal(result, recomputed_result):
        raise _fail(
            "Supplied occupancy result does not exactly match the validated "
            "trajectory and complete selection denominator."
        )

    archive = Path(analysis_zarr).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    root = open_zarr_root(archive, mode="r", use_consolidated=True)
    selection_record = _selection_source_record(
        root,
        archive=archive,
        run_path=selection_run_path,
        compiled=compiled_selection,
    )
    trajectory_record = _trajectory_source_record(
        root,
        archive=archive,
        run_path=trajectory_run_path,
        trajectory=trajectory,
        selection_record=selection_record,
    )

    provider = _strict_record(provider_authority, field="provider_authority")
    timing = _strict_record(timing_authority, field="timing_authority")
    geometry = _strict_record(geometry_authority, field="geometry_authority")
    transform = _strict_record(transform_authority, field="transform_authority")
    fixed_grid = _strict_record(
        fixed_grid_policy_authority,
        field="fixed_grid_policy_authority",
    )
    _compare(
        provider,
        {
            "recording_id": trajectory.authorities.recording_id,
            "provider_id": trajectory.authorities.provider_id,
            "estimator_id": trajectory.authorities.estimator_id,
            "source_id": trajectory.authorities.source_id,
        },
        field="provider_authority",
    )
    subject_id = _require_stable_text(
        provider.get("subject_id"),
        field="provider_authority.subject_id",
    )
    estimator = _stable_subrecord(
        provider.get("estimator", {"estimator_id": provider["estimator_id"]}),
        field="provider_authority.estimator",
    )
    if estimator.get("estimator_id") != trajectory.authorities.estimator_id:
        raise _fail("provider_authority.estimator.estimator_id is stale.")

    position_policy = _stable_subrecord(
        {
            "policy_id": trajectory.authorities.track_sample_policy_id,
            "row_axis": trajectory_record["row_axis"],
            "source_rows_sha256": trajectory_record["source_rows_sha256"],
            "provider_id": trajectory.authorities.provider_id,
            "recording_id": trajectory.authorities.recording_id,
        },
        field="trajectory.position_track_policy",
    )
    sample_unit = _stable_subrecord(
        {
            "policy_id": "one_tracked_subject_sample_per_acquisition_frame_v1",
            "row_axis": "track_samples",
            "recording_id": trajectory.authorities.recording_id,
            "subject_id": subject_id,
        },
        field="trajectory.sample_unit",
    )
    _compare(
        timing,
        {
            "recording_id": trajectory.authorities.recording_id,
            "timeline_authority_id": trajectory.authorities.timeline_authority_id,
            "timing_authority_id": trajectory.authorities.timing_authority_id,
            "fps_hz": result.fps_hz,
            "timing_policy_id": result.timing_policy_id,
        },
        field="timing_authority",
    )
    _compare(
        geometry,
        {
            "recording_id": trajectory.authorities.recording_id,
            "coordinate_authority_id": trajectory.authorities.coordinate_authority_id,
        },
        field="geometry_authority",
    )
    geometry_identity = _stable_subrecord(
        geometry.get("geometry", {"geometry_id": geometry.get("geometry_id")}),
        field="geometry_authority.geometry",
    )
    _require_stable_text(
        geometry_identity.get("geometry_id", geometry_identity.get("immutable_id")),
        field="geometry_authority.geometry_id",
    )
    derived_coordinate_frame = {
        "coordinate_frame_id": trajectory.transform.target_coordinate_authority_id,
        "coordinate_authority_id": trajectory.transform.target_coordinate_authority_id,
        "coordinate_space": "arena_mm",
    }
    coordinate_frame = _stable_subrecord(
        transform.get(
            "coordinate_frame",
            geometry.get("coordinate_frame", derived_coordinate_frame),
        ),
        field="transform_authority.coordinate_frame",
    )
    if coordinate_frame != derived_coordinate_frame:
        raise _fail(
            "transform_authority.coordinate_frame does not match the target "
            "coordinate authority."
        )
    transform_identity = _stable_subrecord(
        transform.get(
            "transform",
            {
                "transform_sha256": trajectory.transform.sha256,
                "source_coordinate_authority_id": trajectory.transform.source_coordinate_authority_id,
                "target_coordinate_authority_id": trajectory.transform.target_coordinate_authority_id,
            },
        ),
        field="transform_authority.transform",
    )
    _compare(
        transform_identity,
        {
            "transform_sha256": trajectory.transform.sha256,
            "source_coordinate_authority_id": trajectory.transform.source_coordinate_authority_id,
            "target_coordinate_authority_id": trajectory.transform.target_coordinate_authority_id,
        },
        field="transform_authority.transform",
    )
    _compare(
        transform,
        {
            "recording_id": trajectory.authorities.recording_id,
            "source_coordinate_authority_id": trajectory.transform.source_coordinate_authority_id,
            "target_coordinate_authority_id": trajectory.transform.target_coordinate_authority_id,
            "transform_sha256": trajectory.transform.sha256,
        },
        field="transform_authority",
    )
    _compare(
        fixed_grid,
        {
            "config_digest": result.config_digest,
            "edge_policy_id": result.edge_policy_id,
            "timing_policy_id": result.timing_policy_id,
            "fps_hz": result.fps_hz,
            "x_edges": result.x_edges.tolist(),
            "y_edges": result.y_edges.tolist(),
        },
        field="fixed_grid_policy_authority",
    )
    grid_id = _require_stable_text(
        fixed_grid.get("grid_id"),
        field="fixed_grid_policy_authority.grid_id",
    )

    provider_record = dict(provider)
    provider_record["estimator"] = estimator
    trajectory_record.update(
        {
            "position_track_policy": position_policy,
            "sample_unit": sample_unit,
            "coordinate_frame": coordinate_frame,
            "recording_id": trajectory.authorities.recording_id,
            "subject_id": subject_id,
        }
    )
    geometry_record = dict(geometry)
    geometry_record["geometry"] = geometry_identity
    geometry_record["coordinate_frame"] = coordinate_frame
    transform_record = dict(transform)
    transform_record["coordinate_frame"] = coordinate_frame
    transform_record["transform"] = transform_identity
    fixed_grid_record = dict(fixed_grid)
    fixed_grid_record.update(
        {
            "grid_id": grid_id,
            "x_edges": result.x_edges.tolist(),
            "y_edges": result.y_edges.tolist(),
            "config_digest": result.config_digest,
        }
    )
    records = {
        "trajectory": trajectory_record,
        "compiled_selection": selection_record,
        "provider": provider_record,
        "timing": timing,
        "geometry": geometry_record,
        "transform": transform_record,
        "fixed_grid_policy": fixed_grid_record,
    }
    return ProviderOccupancyV2SourceBindings.from_mapping(
        {name: _bound_record(records[name], field=name) for name in SOURCE_BINDING_NAMES}
    )


__all__ = [
    "ProviderSpatialPipelineError",
    "SOURCE_BINDING_NAMES",
    "build_provider_occupancy_v2_source_bindings",
    "compiled_selection_membership",
    "occupancy_samples_from_provider_trajectory",
]

"""Plan or publish one explicit selector-ineligible provider spatial canary.

The task document binds immutable source runs and all scientific choices.  This
utility never resolves a provider, geometry, epoch, tracking, or production
selector implicitly.  Publications are immutable named candidates and remain
permanently selector-ineligible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.provider_spatial_trajectory import (
    TrajectoryAuthorityIdentities,
    SourceCameraToArenaMMTransform,
    prepare_provider_spatial_trajectory,
)
from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyTimingPolicy,
    calculate_provider_occupancy_v2,
)
from fisheye.analysis.provider_occupancy_contrast import compute_occupancy_contrast
from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    EpochRoleBinding,
    TimelineAuthorityEvidence,
    compile_goodbatbadbat_selections,
)
from fisheye.analysis_workflows.composable_stimulus_selection import (
    CompiledSelection,
    canonical_json,
    canonical_sha256,
)
from fisheye.analysis_workflows.materializers.composable_stimulus_selection import (
    materialize_composable_stimulus_selection,
    reconstruct_compiled_selection,
    validate_composable_stimulus_selection_run,
)
from fisheye.analysis_workflows.materializers.arena_geometry_selection import (
    validate_arena_geometry_selection_record,
)
from fisheye.analysis_workflows.materializers.single_subject_tracking import (
    plan_single_subject_tracking_run,
    publish_single_subject_tracking_run,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_contrast import (
    SOURCE_SCOPE_POOLED,
    build_pooled_occupancy_contrast_summary,
    build_provider_occupancy_contrast_materialization_plan,
    publish_provider_occupancy_contrast_run,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR,
    materialize_provider_occupancy_v2,
)
from fisheye.analysis_workflows.materializers.provider_spatial_trajectory import (
    materialize_provider_spatial_trajectory_run,
)
from fisheye.analysis_workflows.provider_recording_timing_authority import (
    ProviderRecordingTimingAuthority,
    load_provider_recording_timing_authority,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.analysis_workflows.provider_spatial_grid_policy import (
    GEOMETRY_COORDINATE_SPACE_ID,
    PHYSICAL_RIM_BOUNDARY_ROLE,
    REVIEWED_TOP_RIM_BOUNDARY_ROLE,
    ArenaMMGridPolicy,
    CircularArenaGeometryAuthority,
    PhysicalScaleAuthority,
    SelectionAuthority,
    build_arena_mm_grid_policy,
)
from fisheye.analysis_workflows.provider_spatial_track_source import (
    ProviderTrackSourceAuthorities,
    build_provider_track_source,
)
from fisheye.analysis_workflows.provider_spatial_pipeline import (
    build_provider_occupancy_v2_source_bindings,
    compiled_selection_membership,
    occupancy_samples_from_provider_trajectory,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    load_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    TrackingSourceHandle,
    load_tracking_source_handle,
)
from fisheye.shared.source_camera_physical_authority import (
    load_source_camera_physical_authority,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


TASK_SCHEMA_ID = "palette.provider_spatial_canary_task"
TASK_SCHEMA_VERSION = 1
RESULT_SCHEMA_ID = "palette.provider_spatial_canary_result"
RESULT_SCHEMA_VERSION = 1
CANARY_DISPOSITION = "selector_ineligible_provider_comparison_only_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")
_PROVIDERS = ("detection", "keypoint")
_SELECTIONS = ("black_before", "chaser", "black_after")


class ProviderSpatialCanaryError(ValueError):
    """Raised when a canary task is incomplete, ambiguous, or stale."""


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderSpatialCanaryError(f"{field} must be one JSON object.")
    return dict(value)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderSpatialCanaryError(f"{field} must be one exact nonempty string.")
    return value


def _name(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _NAME_RE.fullmatch(result) is None:
        raise ProviderSpatialCanaryError(f"{field} must be one bare immutable name.")
    return result


def _run_path(value: object, *, field: str, parent: str) -> str:
    result = _text(value, field=field)
    if _PATH_RE.fullmatch(result) is None or not result.startswith(f"{parent}/"):
        raise ProviderSpatialCanaryError(
            f"{field} must be one exact child path below {parent!r}."
        )
    child = result[len(parent) + 1 :]
    if "/" in child or child.lower().startswith(("latest", "active", "current")):
        raise ProviderSpatialCanaryError(f"{field} cannot be a selector path.")
    return result


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProviderSpatialCanaryError(f"{field} must be one lowercase SHA-256.")
    return value


def _optional_digest(value: object, *, field: str) -> str | None:
    return None if value is None else _digest(value, field=field)


def _window_ids(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ProviderSpatialCanaryError(f"{field} must be one nonempty integer list.")
    if any(type(item) is not int or item < 0 for item in value):
        raise ProviderSpatialCanaryError(f"{field} contains an invalid window ID.")
    if len(set(value)) != len(value):
        raise ProviderSpatialCanaryError(f"{field} contains duplicate window IDs.")
    return list(value)


def _output_name(campaign_id: str, *parts: str) -> str:
    return _name("_".join((*parts, campaign_id)), field="derived output run name")


def _normalize_provider(value: object, *, provider: str) -> dict[str, Any]:
    record = _object(value, field=f"providers.{provider}")
    position_parent = "analysis/subject_position_runs/observation"
    position_path = _run_path(
        record.get("position_run_path"),
        field=f"providers.{provider}.position_run_path",
        parent=position_parent,
    )
    tracking_path_value = record.get("tracking_run_path")
    if tracking_path_value is not None:
        tracking_path = _run_path(
            tracking_path_value,
            field=f"providers.{provider}.tracking_run_path",
            parent="tracking_runs",
        )
        tracking_digest = _digest(
            record.get("tracking_manifest_sha256"),
            field=f"providers.{provider}.tracking_manifest_sha256",
        )
    else:
        tracking_path = None
        if record.get("tracking_manifest_sha256") is not None:
            raise ProviderSpatialCanaryError(
                f"providers.{provider}.tracking_manifest_sha256 requires a tracking path."
            )
        tracking_digest = None
    return {
        "position_run_path": position_path,
        "position_manifest_sha256": _digest(
            record.get("position_manifest_sha256"),
            field=f"providers.{provider}.position_manifest_sha256",
        ),
        "tracking_run_path": tracking_path,
        "tracking_manifest_sha256": tracking_digest,
    }


def load_task(path: str | Path) -> dict[str, Any]:
    """Load and normalize one immutable canary task document."""

    task_path = Path(path).expanduser().resolve()
    payload = _object(json.loads(task_path.read_text(encoding="utf-8")), field="task")
    if (
        payload.get("schema_id") != TASK_SCHEMA_ID
        or payload.get("schema_version") != TASK_SCHEMA_VERSION
    ):
        raise ProviderSpatialCanaryError("Task schema identity is unsupported.")
    if payload.get("disposition") != CANARY_DISPOSITION:
        raise ProviderSpatialCanaryError(
            "Task must explicitly remain a selector-ineligible provider comparison."
        )
    recording_id = _name(payload.get("recording_id"), field="recording_id")
    archive = Path(_text(payload.get("analysis_zarr"), field="analysis_zarr")).resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if archive.name != f"{recording_id}_analysis.zarr":
        raise ProviderSpatialCanaryError(
            "Analysis Zarr basename differs from the frozen recording identity."
        )
    campaign_id = _name(payload.get("campaign_id"), field="campaign_id")
    arena_id = payload.get("arena_id")
    if type(arena_id) is not int or arena_id < 0:
        raise ProviderSpatialCanaryError("arena_id must be one non-negative integer.")
    subject_id = _name(payload.get("subject_id"), field="subject_id")

    epoch = _object(payload.get("epoch_source"), field="epoch_source")
    epoch = {
        "run_name": _name(epoch.get("run_name"), field="epoch_source.run_name"),
        "manifest_sha256": _digest(
            epoch.get("manifest_sha256"), field="epoch_source.manifest_sha256"
        ),
    }
    geometry = _object(payload.get("geometry_source"), field="geometry_source")
    geometry = {
        "selection_run_name": _name(
            geometry.get("selection_run_name"),
            field="geometry_source.selection_run_name",
        ),
        "selection_record_sha256": _digest(
            geometry.get("selection_record_sha256"),
            field="geometry_source.selection_record_sha256",
        ),
        "physical_authority_sha256": _digest(
            geometry.get("physical_authority_sha256"),
            field="geometry_source.physical_authority_sha256",
        ),
    }
    timing_sha256 = _digest(
        payload.get("recording_timing_authority_sha256"),
        field="recording_timing_authority_sha256",
    )
    providers_value = _object(payload.get("providers"), field="providers")
    if set(providers_value) != set(_PROVIDERS):
        raise ProviderSpatialCanaryError(
            "providers must explicitly contain only detection and keypoint."
        )
    providers = {
        provider: _normalize_provider(providers_value[provider], provider=provider)
        for provider in _PROVIDERS
    }
    selections_value = _object(payload.get("selections"), field="selections")
    if set(selections_value) != set(_SELECTIONS):
        raise ProviderSpatialCanaryError(
            "selections must explicitly contain black_before, chaser, and black_after."
        )
    selections: dict[str, Any] = {}
    for selection_id in _SELECTIONS:
        record = _object(selections_value[selection_id], field=f"selections.{selection_id}")
        selections[selection_id] = {
            "window_ids": _window_ids(
                record.get("window_ids"), field=f"selections.{selection_id}.window_ids"
            ),
            "role": _name(record.get("role"), field=f"selections.{selection_id}.role"),
        }
        if len(selections[selection_id]["window_ids"]) != 1:
            raise ProviderSpatialCanaryError(
                f"selections.{selection_id} must bind exactly one explicit source window."
            )
        if selections[selection_id]["role"] != selection_id:
            raise ProviderSpatialCanaryError(
                f"selections.{selection_id}.role must preserve the canonical role name."
            )
    all_windows = [
        window_id
        for selection in selections.values()
        for window_id in selection["window_ids"]
    ]
    if len(set(all_windows)) != len(all_windows):
        raise ProviderSpatialCanaryError(
            "The three first-canary selections must use disjoint source windows."
        )
    grid = _object(payload.get("grid"), field="grid")
    bin_width = grid.get("bin_width_mm")
    if isinstance(bin_width, bool) or not isinstance(bin_width, (int, float)):
        raise ProviderSpatialCanaryError("grid.bin_width_mm must be numeric.")
    bin_width = float(bin_width)
    if not 0.0 < bin_width < 1000.0:
        raise ProviderSpatialCanaryError("grid.bin_width_mm must be finite and positive.")
    contrasts = payload.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ProviderSpatialCanaryError("contrasts must be one nonempty list.")
    normalized_contrasts: list[dict[str, str]] = []
    for index, value in enumerate(contrasts):
        record = _object(value, field=f"contrasts[{index}]")
        baseline = _name(record.get("baseline"), field=f"contrasts[{index}].baseline")
        treatment = _name(record.get("treatment"), field=f"contrasts[{index}].treatment")
        if baseline not in selections or treatment not in selections or baseline == treatment:
            raise ProviderSpatialCanaryError(
                f"contrasts[{index}] must bind two distinct declared selections."
            )
        normalized_contrasts.append({"baseline": baseline, "treatment": treatment})
    if len({(item["baseline"], item["treatment"]) for item in normalized_contrasts}) != len(
        normalized_contrasts
    ):
        raise ProviderSpatialCanaryError("contrasts contains duplicate directed pairs.")

    return {
        "schema_id": TASK_SCHEMA_ID,
        "schema_version": TASK_SCHEMA_VERSION,
        "disposition": CANARY_DISPOSITION,
        "campaign_id": campaign_id,
        "recording_id": recording_id,
        "analysis_zarr": str(archive),
        "arena_id": arena_id,
        "subject_id": subject_id,
        "epoch_source": epoch,
        "geometry_source": geometry,
        "recording_timing_authority_sha256": timing_sha256,
        "providers": providers,
        "selections": selections,
        "grid": {
            "policy_id": _name(grid.get("policy_id"), field="grid.policy_id"),
            "bin_width_mm": bin_width,
        },
        "contrasts": normalized_contrasts,
    }


def planned_run_names(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return every exact output name without opening or mutating the archive."""

    campaign_id = str(task["campaign_id"])
    selections = {
        selection_id: _output_name(campaign_id, "stimulus_selection", selection_id)
        for selection_id in _SELECTIONS
    }
    tracking = {
        provider: (
            str(task["providers"][provider]["tracking_run_path"]).split("/")[-1]
            if task["providers"][provider]["tracking_run_path"] is not None
            else _output_name(campaign_id, "tracking", provider)
        )
        for provider in _PROVIDERS
    }
    trajectories = {
        provider: {
            selection_id: _output_name(
                campaign_id, "provider_spatial_trajectory", provider, selection_id
            )
            for selection_id in _SELECTIONS
        }
        for provider in _PROVIDERS
    }
    occupancies = {
        provider: {
            selection_id: _output_name(
                campaign_id, "provider_occupancy", provider, selection_id
            )
            for selection_id in _SELECTIONS
        }
        for provider in _PROVIDERS
    }
    contrasts = {
        provider: [
            _output_name(
                campaign_id,
                "provider_occupancy_contrast",
                provider,
                item["treatment"],
                "minus",
                item["baseline"],
            )
            for item in task["contrasts"]
        ]
        for provider in _PROVIDERS
    }
    return {
        "tracking": tracking,
        "selections": selections,
        "trajectories": trajectories,
        "occupancies": occupancies,
        "contrasts": contrasts,
    }


def _strict_copy(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderSpatialCanaryError(f"{field} must be one JSON object.")

    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        if isinstance(item, list):
            return [thaw(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    try:
        encoded = json.dumps(
            thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        result = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialCanaryError(f"{field} is not strict JSON.") from exc
    if not isinstance(result, dict):  # pragma: no cover - defensive
        raise ProviderSpatialCanaryError(f"{field} must remain an object.")
    return result


def _selection_attrs(root: Any, *, run_name: str) -> dict[str, Any]:
    path = f"analysis/arena_geometry_selection/{run_name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise ProviderSpatialCanaryError(
            f"Exact arena-geometry selection is missing: {path}."
        ) from exc
    attrs = dict(group.attrs)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("operational_selection_status") != "selected"
        or attrs.get("stage_selector_eligible") is not True
    ):
        raise ProviderSpatialCanaryError(
            "Arena-geometry selection is not complete, selected, and operational."
        )
    record = attrs.get("selection_record")
    if not isinstance(record, Mapping):
        raise ProviderSpatialCanaryError("Arena-geometry selection record is absent.")
    validate_arena_geometry_selection_record(record)
    canonical = _strict_copy(record, field="arena geometry selection record")
    observed = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if attrs.get("selection_record_sha256") != observed:
        raise ProviderSpatialCanaryError("Arena-geometry selection digest is stale.")
    return {"path": path, "record": canonical, "sha256": observed}


def _circle_from_selected_candidate(
    selection: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any]]:
    selected = selection["selected_candidate"]
    physical = selected.get("physical_inner_rim")
    if isinstance(physical, Mapping):
        geometry = physical.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ProviderSpatialCanaryError(
                "Selected physical inner rim lacks circle geometry."
            )
        return PHYSICAL_RIM_BOUNDARY_ROLE, str(
            physical.get("observed_feature") or "dish_inner_rim_water_side_edge"
        ), geometry
    decision = selection.get("decision")
    if not isinstance(decision, Mapping) or decision.get("decision_source") != "manual_review":
        raise ProviderSpatialCanaryError(
            "A non-physical arena boundary requires an explicit manual review selection."
        )
    observed = selected.get("observed_boundary")
    if not isinstance(observed, Mapping):
        raise ProviderSpatialCanaryError(
            "Reviewed selected candidate lacks an observed boundary."
        )
    if observed.get("observed_feature") != REVIEWED_TOP_RIM_BOUNDARY_ROLE:
        raise ProviderSpatialCanaryError(
            "Only an explicit reviewed visible top-rim boundary may substitute for a "
            "physical inner-rim grid boundary in this canary."
        )
    geometry = observed.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ProviderSpatialCanaryError("Reviewed top-rim boundary lacks geometry.")
    return REVIEWED_TOP_RIM_BOUNDARY_ROLE, REVIEWED_TOP_RIM_BOUNDARY_ROLE, geometry


def load_grid_and_transform_authority(
    task: Mapping[str, Any],
) -> tuple[ArenaMMGridPolicy, SourceCameraToArenaMMTransform, dict[str, Any]]:
    """Load exact reviewed geometry/scale and build the fixed arena-mm authority."""

    archive = Path(str(task["analysis_zarr"]))
    geometry_task = task["geometry_source"]
    direct_root = open_zarr_root(archive, mode="r", use_consolidated=False)
    consolidated_root = open_zarr_root(archive, mode="r", use_consolidated=True)
    direct_selection = _selection_attrs(
        direct_root, run_name=geometry_task["selection_run_name"]
    )
    consolidated_selection = _selection_attrs(
        consolidated_root, run_name=geometry_task["selection_run_name"]
    )
    if direct_selection != consolidated_selection:
        raise ProviderSpatialCanaryError(
            "Direct and consolidated arena-geometry selection authorities differ."
        )
    if direct_selection["sha256"] != geometry_task["selection_record_sha256"]:
        raise ProviderSpatialCanaryError(
            "Arena-geometry selection differs from the frozen canary task."
        )
    direct_physical = load_source_camera_physical_authority(direct_root)
    consolidated_physical = load_source_camera_physical_authority(consolidated_root)
    if (
        direct_physical.manifest.record_sha256
        != consolidated_physical.manifest.record_sha256
        or direct_physical.physical_frame.record_sha256
        != consolidated_physical.physical_frame.record_sha256
        or direct_physical.mm_per_pixel != consolidated_physical.mm_per_pixel
    ):
        raise ProviderSpatialCanaryError(
            "Direct and consolidated source-camera physical authorities differ."
        )
    if direct_physical.manifest.record_sha256 != geometry_task["physical_authority_sha256"]:
        raise ProviderSpatialCanaryError(
            "Source-camera physical authority differs from the frozen canary task."
        )
    selection_record = direct_selection["record"]
    selected = selection_record["selected_candidate"]
    coordinate = selected["coordinate_binding"]
    coordinate_id = _digest(
        coordinate.get("pixel_frame_record_sha256"),
        field="selected geometry pixel-frame digest",
    )
    if coordinate_id != direct_physical.physical_frame.source_camera_pixels.record_sha256:
        raise ProviderSpatialCanaryError(
            "Selected geometry and physical scale use different source-camera frames."
        )
    boundary_role, observed_feature, circle = _circle_from_selected_candidate(
        selection_record
    )
    if circle.get("type") != "circle" or circle.get("coordinate_space") not in {
        None,
        "camera_native_pixels",
    }:
        raise ProviderSpatialCanaryError("Selected arena boundary is not a native circle.")
    center = circle.get("center_px")
    if not isinstance(center, Mapping):
        raise ProviderSpatialCanaryError("Selected circle center is absent.")
    values = (center.get("x"), center.get("y"), circle.get("radius_px"))
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ) or float(values[2]) <= 0:
        raise ProviderSpatialCanaryError("Selected arena circle is not finite and positive.")
    width = coordinate.get("native_width_px")
    height = coordinate.get("native_height_px")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise ProviderSpatialCanaryError("Selected geometry native extent is invalid.")
    if (
        coordinate.get("origin") != "top_left"
        or coordinate.get("positive_x") != "right"
        or coordinate.get("positive_y") != "down"
        or coordinate.get("pixel_convention") != "continuous"
    ):
        raise ProviderSpatialCanaryError(
            "Selected geometry does not use the native continuous camera convention."
        )

    geometry = CircularArenaGeometryAuthority(
        geometry_id=str(selected["candidate_id"]),
        coordinate_authority_id=coordinate_id,
        center_x_px=float(values[0]),
        center_y_px=float(values[1]),
        radius_px=float(values[2]),
        record_ref=f"/{direct_selection['path']}@selection_record:selected_candidate",
        boundary_role=boundary_role,
        observed_feature=observed_feature,
        coordinate_space=GEOMETRY_COORDINATE_SPACE_ID,
    )
    scale = PhysicalScaleAuthority(
        scale_id=f"source_camera_physical_{direct_physical.manifest.record_sha256[:24]}",
        coordinate_authority_id=coordinate_id,
        mm_per_pixel=direct_physical.mm_per_pixel,
        record_ref=direct_physical.physical_frame.record_ref,
    )
    selection = SelectionAuthority(
        selection_id=str(geometry_task["selection_run_name"]),
        recording_id=str(task["recording_id"]),
        record_sha256=str(direct_selection["sha256"]),
        record_ref=f"/{direct_selection['path']}@selection_record",
    )
    grid_policy = build_arena_mm_grid_policy(
        recording_id=str(task["recording_id"]),
        geometry=geometry,
        scale=scale,
        selection=selection,
        bin_width_mm=float(task["grid"]["bin_width_mm"]),
        policy_id=str(task["grid"]["policy_id"]),
    )
    target_coordinate_id = f"arena_centered_mm_{grid_policy.policy_digest[:24]}"
    mm_per_px = scale.mm_per_pixel
    matrix = np.asarray(
        [
            [mm_per_px, 0.0, -geometry.center_x_px * mm_per_px],
            [0.0, mm_per_px, -geometry.center_y_px * mm_per_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = SourceCameraToArenaMMTransform(
        source_coordinate_authority_id=coordinate_id,
        target_coordinate_authority_id=target_coordinate_id,
        matrix=matrix,
        grid_extent_mm=(
            float(grid_policy.x_edges[0]),
            float(grid_policy.x_edges[-1]),
            float(grid_policy.y_edges[0]),
            float(grid_policy.y_edges[-1]),
        ),
        source_camera_extent_px=(0.0, float(width), 0.0, float(height)),
    )
    evidence = {
        "selection": direct_selection,
        "source_camera_physical_authority": {
            "record": _strict_copy(
                direct_physical.manifest.record,
                field="source-camera physical authority",
            ),
            "sha256": direct_physical.manifest.record_sha256,
            "physical_frame_record_ref": direct_physical.physical_frame.record_ref,
            "physical_frame_record_sha256": direct_physical.physical_frame.record_sha256,
        },
        "grid_policy": grid_policy.as_record(),
        "transform": transform.as_record(),
        "transform_sha256": transform.sha256,
    }
    if canonical_json_sha256(grid_policy.payload()) != grid_policy.policy_digest:
        raise ProviderSpatialCanaryError("Derived grid policy digest is stale.")
    return grid_policy, transform, evidence


def load_recording_and_subject_authorities(
    task: Mapping[str, Any],
) -> tuple[ProviderRecordingTimingAuthority, dict[str, Any]]:
    """Load exact recording timing and one-subject identities in both metadata modes."""

    archive = Path(str(task["analysis_zarr"]))
    timing = load_provider_recording_timing_authority(
        archive,
        required=True,
        use_consolidated=True,
        expected_sha256=str(task["recording_timing_authority_sha256"]),
    )
    assert timing is not None
    if timing.recording_id != task["recording_id"]:
        raise ProviderSpatialCanaryError(
            "Recording timing authority and task recording identities differ."
        )
    direct_subject = resolve_subject_metadata(
        open_zarr_root(archive, mode="r", use_consolidated=False), allow_legacy=False
    )
    consolidated_subject = resolve_subject_metadata(
        open_zarr_root(archive, mode="r", use_consolidated=True), allow_legacy=False
    )
    if (
        direct_subject.record_sha256 != consolidated_subject.record_sha256
        or direct_subject.record != consolidated_subject.record
        or direct_subject.group_path != consolidated_subject.group_path
    ):
        raise ProviderSpatialCanaryError(
            "Direct and consolidated subject-metadata authorities differ."
        )
    if direct_subject.legacy or len(direct_subject.subject_ids) != 1:
        raise ProviderSpatialCanaryError(
            "The first spatial canary requires one canonical individual subject."
        )
    if direct_subject.subject_ids[0] != task["subject_id"]:
        raise ProviderSpatialCanaryError(
            "Canonical subject identity differs from the frozen canary task."
        )
    return timing, {
        "record": _strict_copy(direct_subject.record, field="subject metadata record"),
        "sha256": direct_subject.record_sha256,
        "group_path": direct_subject.group_path,
        "subject_id": direct_subject.subject_ids[0],
    }


def _load_position(task: Mapping[str, Any], *, provider: str) -> SubjectPositionSourceHandle:
    source = task["providers"][provider]
    return load_subject_position_source_handle(
        task["analysis_zarr"],
        source["position_run_path"],
        expected_selector_eligible=False,
        use_consolidated=True,
        expected_manifest_sha256=source["position_manifest_sha256"],
    )


def _tracking_for_provider(
    task: Mapping[str, Any],
    *,
    provider: str,
    position: SubjectPositionSourceHandle,
    output_name: str,
    scratch_root: Path,
) -> tuple[TrackingSourceHandle, dict[str, Any]]:
    source = task["providers"][provider]
    frozen_path = source["tracking_run_path"]
    run_path = frozen_path or f"tracking_runs/{output_name}"
    target = Path(str(task["analysis_zarr"])).joinpath(*run_path.split("/"))
    if frozen_path is not None:
        handle = load_tracking_source_handle(
            task["analysis_zarr"],
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
            expected_manifest_sha256=source["tracking_manifest_sha256"],
        )
        return handle, {
            "status": "reused_frozen_source",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
        }
    if target.exists():
        handle = load_tracking_source_handle(
            task["analysis_zarr"],
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        return handle, {
            "status": "reused_campaign_successor",
            "run_path": handle.run_path,
            "manifest_sha256": handle.manifest_sha256,
        }
    plan = plan_single_subject_tracking_run(
        position,
        arena_id=int(task["arena_id"]),
        run_name=output_name,
        scratch_root=scratch_root / f"tracking-{provider}",
    )
    published = publish_single_subject_tracking_run(plan, keep_scratch=False)
    handle = load_tracking_source_handle(
        task["analysis_zarr"],
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "publication_status": published["publication"]["status"],
    }


def _source_coordinate_authority_id(position: SubjectPositionSourceHandle) -> str:
    coordinate = position.coordinate_record
    frame = coordinate.get("frame_record")
    if not isinstance(frame, Mapping):
        extent = coordinate.get("reference_extent")
        authority = extent.get("authority") if isinstance(extent, Mapping) else None
        frame = authority if isinstance(authority, Mapping) else None
    if not isinstance(frame, Mapping):
        raise ProviderSpatialCanaryError(
            "Subject-position coordinate authority lacks a source-camera frame record."
        )
    return _digest(
        frame.get("record_sha256"),
        field="subject-position source-camera frame record",
    )


def load_provider_track_inputs(
    task: Mapping[str, Any],
    *,
    timeline_authority_id: str,
    output_names: Mapping[str, Any],
    scratch_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load or create each exact provider-specific tracking projection and keyed join."""

    providers: dict[str, Any] = {}
    stages: dict[str, Any] = {}
    for provider in _PROVIDERS:
        position = _load_position(task, provider=provider)
        tracking, tracking_stage = _tracking_for_provider(
            task,
            provider=provider,
            position=position,
            output_name=str(output_names["tracking"][provider]),
            scratch_root=scratch_root,
        )
        samples, evidence = build_provider_track_source(
            position,
            tracking,
            authorities=ProviderTrackSourceAuthorities(
                recording_id=str(task["recording_id"]),
                timeline_authority_id=timeline_authority_id,
                subject_identity=str(task["subject_id"]),
            ),
        )
        providers[provider] = {
            "position": position,
            "tracking": tracking,
            "samples": samples,
            "evidence": evidence,
            "source_coordinate_authority_id": _source_coordinate_authority_id(position),
        }
        stages[provider] = {
            "position": {
                "run_path": position.run_path,
                "manifest_sha256": position.manifest_sha256,
                "row_count": position.row_count,
            },
            "tracking": tracking_stage,
            "track_source": {
                "record_sha256": evidence.sha256,
                "track_sample_policy_id": evidence.track_sample_policy_id,
                "row_count": int(samples.acquisition_frame.size),
            },
        }
    return providers, stages


def _timeline_evidence(
    task: Mapping[str, Any],
    *,
    selection: ResolvedEpochSelection,
    timing: ProviderRecordingTimingAuthority,
) -> TimelineAuthorityEvidence:
    root = open_zarr_root(task["analysis_zarr"], mode="r", use_consolidated=True)
    source_video = _strict_copy(
        root.attrs.get("source_video_metadata"), field="source video metadata"
    )
    source_video_digest = canonical_sha256(source_video)
    timing_record = _strict_copy(timing.record, field="recording timing authority")
    clock = timing_record.get("acquisition_frame_clock")
    if not isinstance(clock, Mapping):
        raise ProviderSpatialCanaryError(
            "Recording timing authority lacks its acquisition frame-clock record."
        )
    clock_ref = _text(
        clock.get("run_path"), field="recording timing acquisition clock run path"
    )
    source_video_ref = "/@source_video_metadata"
    source_metadata = {
        "recording_id": str(task["recording_id"]),
        "source_timeline_digest": selection.source_timeline_digest,
        "source_epoch_run_path": selection.run_path,
        "source_epoch_run_manifest_sha256": selection.run_manifest_digest,
        "source_epoch_run_manifest_payload_sha256": (
            selection.run_manifest_payload_digest
        ),
        "source_epoch_logical_content_sha256": (
            selection.source_epoch_logical_content_digest
        ),
        "source_epoch_lineage_hash": selection.source_epoch_lineage_hash,
        "source_epoch_lineage_payload_sha256": (
            selection.source_epoch_lineage_payload_digest
        ),
        "timing_authority": _strict_copy(
            selection.timing_authority, field="stimulus epoch timing authority"
        ),
        "source_video_metadata_ref": source_video_ref,
        "source_video_metadata_sha256": source_video_digest,
        "acquisition_clock_authority_ref": clock_ref,
        "acquisition_clock_authority_sha256": timing.sha256,
        "acquisition_frame_domain": "camera_acquisition_frame",
        "frame_count": selection.native_frame_count,
        "fps": selection.fps,
    }
    source_metadata_digest = canonical_sha256(source_metadata)
    return TimelineAuthorityEvidence(
        recording_id=str(task["recording_id"]),
        timeline_id=selection.source_timeline_digest,
        stimulus_authority_id=selection.run_path,
        acquisition_frame_domain="camera_acquisition_frame",
        source_video_metadata_ref=source_video_ref,
        source_video_metadata_sha256=source_video_digest,
        source_video_metadata=source_video,
        acquisition_clock_authority_ref=clock_ref,
        acquisition_clock_authority_sha256=timing.sha256,
        acquisition_clock_authority=timing_record,
        source_metadata_sha256=source_metadata_digest,
        source_metadata=source_metadata,
    )


def compile_canary_selections(
    task: Mapping[str, Any],
    *,
    timing: ProviderRecordingTimingAuthority,
) -> tuple[dict[str, CompiledSelection], dict[str, Any]]:
    """Resolve one exact epoch-v2 run and apply only caller-declared role bindings."""

    epoch = task["epoch_source"]
    resolved = resolve_exact_stimulus_epoch_selection(
        task["analysis_zarr"],
        run_name=epoch["run_name"],
        expected_run_manifest_digest=epoch["manifest_sha256"],
    )
    if resolved.recording_timing_authority_sha256 != timing.sha256:
        raise ProviderSpatialCanaryError(
            "Stimulus epochs and canary timing task identify different authorities."
        )
    bindings = {
        role: EpochRoleBinding.by_window_id(task["selections"][role]["window_ids"][0])
        for role in _SELECTIONS
    }
    composed = compile_goodbatbadbat_selections(
        resolved,
        timeline_evidence=_timeline_evidence(task, selection=resolved, timing=timing),
        role_bindings=bindings,
        include_all_black=False,
    )
    compiled = {role: composed[role] for role in _SELECTIONS}
    return compiled, {
        "run_path": resolved.run_path,
        "run_manifest_sha256": resolved.run_manifest_digest,
        "run_manifest_payload_sha256": resolved.run_manifest_payload_digest,
        "selection_sha256": resolved.selection_digest,
        "source_timeline_digest": resolved.source_timeline_digest,
        "timeline_authority": composed.timeline_authority.to_dict(),
        "compiled": {
            role: {
                "request_digest": value.request_digest,
                "resolved_digest": value.resolved_digest,
            }
            for role, value in compiled.items()
        },
    }


def _same_compiled(left: CompiledSelection, right: CompiledSelection) -> bool:
    return (
        canonical_json(left.requested) == canonical_json(right.requested)
        and canonical_json(left.resolved_payload()) == canonical_json(right.resolved_payload())
        and left.request_digest == right.request_digest
        and left.resolved_digest == right.resolved_digest
        and left.authority.to_dict() == right.authority.to_dict()
    )


def publish_canary_selections(
    task: Mapping[str, Any],
    *,
    compiled: Mapping[str, CompiledSelection],
    output_names: Mapping[str, Any],
    scratch_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Publish or exactly reuse the three selector-ineligible selection runs."""

    archive = Path(str(task["analysis_zarr"]))
    paths: dict[str, str] = {}
    stages: dict[str, Any] = {}
    for role in _SELECTIONS:
        run_name = str(output_names["selections"][role])
        run_path = f"analysis/stimulus_selection_runs/{run_name}"
        target = archive.joinpath(*run_path.split("/"))
        if target.exists():
            observed = reconstruct_compiled_selection(target)
            if not _same_compiled(observed, compiled[role]):
                raise ProviderSpatialCanaryError(
                    f"Existing campaign selection differs for role {role!r}."
                )
            validation = validate_composable_stimulus_selection_run(
                target, expected_compiled_selection=compiled[role]
            )
            if validation.get("valid") is not True:
                raise ProviderSpatialCanaryError(
                    f"Existing selection failed validation: {validation.get('errors')!r}."
                )
            status = "reused_campaign_run"
            result: Mapping[str, Any] = validation
        else:
            result = materialize_composable_stimulus_selection(
                archive,
                compiled_selection=compiled[role],
                scratch_root=scratch_root / f"selection-{role}",
                run_name=run_name,
                apply=True,
                keep_scratch=False,
            )
            status = "published"
        paths[role] = run_path
        stages[role] = {
            "status": status,
            "run_path": run_path,
            "request_digest": compiled[role].request_digest,
            "resolved_digest": compiled[role].resolved_digest,
            "validation_status": (
                "valid"
                if status == "reused_campaign_run"
                else result.get("status", "complete")
            ),
        }
    return paths, stages


def _provider_id(provider: str, position: SubjectPositionSourceHandle) -> str:
    return f"{provider}_subject_position_{position.manifest_sha256[:24]}"


def _provider_authority_record(
    task: Mapping[str, Any],
    *,
    provider: str,
    position: SubjectPositionSourceHandle,
    evidence: Any,
) -> dict[str, Any]:
    estimator_id = _text(
        position.estimator_record.get("estimator_id"),
        field=f"{provider} estimator_id",
    )
    return {
        "schema_id": "palette.provider_spatial_provider_authority",
        "schema_version": 1,
        "recording_id": str(task["recording_id"]),
        "provider_id": _provider_id(provider, position),
        "estimator_id": estimator_id,
        "source_id": evidence.source_id,
        "subject_id": str(task["subject_id"]),
        "source_modality": provider,
        "estimator": _strict_copy(
            position.estimator_record, field=f"{provider} estimator record"
        ),
        "provider_track_source": {
            "record": _strict_copy(
                evidence.record, field=f"{provider} track source evidence"
            ),
            "sha256": evidence.sha256,
        },
    }


def _occupancy_authorities(
    task: Mapping[str, Any],
    *,
    provider: str,
    position: SubjectPositionSourceHandle,
    evidence: Any,
    timing: ProviderRecordingTimingAuthority,
    grid_policy: ArenaMMGridPolicy,
    transform: SourceCameraToArenaMMTransform,
    result: Any,
) -> dict[str, dict[str, Any]]:
    coordinate_frame = {
        "coordinate_frame_id": transform.target_coordinate_authority_id,
        "coordinate_authority_id": transform.target_coordinate_authority_id,
        "coordinate_space": "arena_mm",
        "origin": "selected_arena_circle_center",
        "positive_x": "camera_right",
        "positive_y": "camera_down",
        "units": "mm",
    }
    geometry_identity = {
        "geometry_id": grid_policy.geometry.geometry_id,
        "record_ref": grid_policy.geometry.record_ref,
        "record_sha256": grid_policy.geometry.record_sha256,
        "selection_record_sha256": grid_policy.selection.record_sha256,
        "boundary_role": grid_policy.geometry.boundary_role,
        "observed_feature": grid_policy.geometry.observed_feature,
    }
    transform_identity = {
        "transform_sha256": transform.sha256,
        "source_coordinate_authority_id": transform.source_coordinate_authority_id,
        "target_coordinate_authority_id": transform.target_coordinate_authority_id,
        "matrix": transform.matrix.tolist(),
        "source_camera_extent_px": list(transform.source_camera_extent_px or ()),
        "grid_extent_mm": list(transform.grid_extent_mm),
        "scale_record_sha256": grid_policy.scale.record_sha256,
        "selection_record_sha256": grid_policy.selection.record_sha256,
    }
    return {
        "provider": _provider_authority_record(
            task,
            provider=provider,
            position=position,
            evidence=evidence,
        ),
        "timing": {
            "schema_id": "palette.provider_spatial_timing_authority",
            "schema_version": 1,
            "recording_id": str(task["recording_id"]),
            "timeline_authority_id": evidence.record["authorities"][
                "timeline_authority_id"
            ],
            "timing_authority_id": timing.sha256,
            "fps_hz": result.fps_hz,
            "timing_policy_id": result.timing_policy_id,
            "record": _strict_copy(timing.record, field="recording timing authority"),
            "sha256": timing.sha256,
        },
        "geometry": {
            "schema_id": "palette.provider_spatial_geometry_authority",
            "schema_version": 1,
            "recording_id": str(task["recording_id"]),
            "coordinate_authority_id": transform.source_coordinate_authority_id,
            "geometry": geometry_identity,
            "coordinate_frame": coordinate_frame,
        },
        "transform": {
            "schema_id": "palette.provider_spatial_transform_authority",
            "schema_version": 1,
            "recording_id": str(task["recording_id"]),
            "source_coordinate_authority_id": transform.source_coordinate_authority_id,
            "target_coordinate_authority_id": transform.target_coordinate_authority_id,
            "transform_sha256": transform.sha256,
            "coordinate_frame": coordinate_frame,
            "transform": transform_identity,
        },
        "fixed_grid_policy": {
            "schema_id": "palette.provider_spatial_fixed_grid_policy_authority",
            "schema_version": 1,
            "grid_id": grid_policy.policy_id,
            "config_digest": result.config_digest,
            "edge_policy_id": result.edge_policy_id,
            "timing_policy_id": result.timing_policy_id,
            "fps_hz": result.fps_hz,
            "x_edges": result.x_edges.tolist(),
            "y_edges": result.y_edges.tolist(),
            "grid_policy": grid_policy.as_record(),
        },
    }


def publish_provider_spatial_products(
    task: Mapping[str, Any],
    *,
    compiled: Mapping[str, CompiledSelection],
    selection_paths: Mapping[str, str],
    providers: Mapping[str, Any],
    timing: ProviderRecordingTimingAuthority,
    grid_policy: ArenaMMGridPolicy,
    transform: SourceCameraToArenaMMTransform,
    output_names: Mapping[str, Any],
    scratch_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish exact trajectories and occupancies for both providers and all roles."""

    archive = Path(str(task["analysis_zarr"]))
    products: dict[str, Any] = {}
    stages: dict[str, Any] = {}
    occupancy_grid = grid_policy.to_occupancy_grid()
    timing_policy = OccupancyTimingPolicy(timing.nominal_fps)
    for provider in _PROVIDERS:
        bundle = providers[provider]
        if bundle["source_coordinate_authority_id"] != transform.source_coordinate_authority_id:
            raise ProviderSpatialCanaryError(
                f"{provider} position and selected geometry use different camera authorities."
            )
        position = bundle["position"]
        evidence = bundle["evidence"]
        provider_id = _provider_id(provider, position)
        provider_products: dict[str, Any] = {}
        provider_stages: dict[str, Any] = {}
        for role in _SELECTIONS:
            selection = compiled[role]
            trajectory = prepare_provider_spatial_trajectory(
                authorities=TrajectoryAuthorityIdentities(
                    recording_id=str(task["recording_id"]),
                    provider_id=provider_id,
                    track_sample_policy_id=evidence.track_sample_policy_id,
                    estimator_id=str(position.estimator_record["estimator_id"]),
                    source_id=evidence.source_id,
                    timing_authority_id=timing.sha256,
                    timeline_authority_id=selection.authority.timeline_id,
                    coordinate_authority_id=transform.source_coordinate_authority_id,
                    selection_authority_id=selection.resolved_digest,
                ),
                rows=bundle["samples"],
                selection=compiled_selection_membership(selection),
                transform=transform,
            )
            trajectory_name = str(output_names["trajectories"][provider][role])
            trajectory_path = (
                f"analysis/provider_spatial_trajectory_runs/{trajectory_name}"
            )
            if archive.joinpath(*trajectory_path.split("/")).exists():
                raise ProviderSpatialCanaryError(
                    f"Refusing existing immutable trajectory run: {trajectory_path}."
                )
            trajectory_result = materialize_provider_spatial_trajectory_run(
                archive,
                trajectory,
                run_name=trajectory_name,
                scratch_root=scratch_root / f"trajectory-{provider}-{role}",
                apply=True,
                keep_scratch=False,
            )
            samples = occupancy_samples_from_provider_trajectory(
                trajectory,
                selection=compiled_selection_membership(selection),
            )
            occupancy = calculate_provider_occupancy_v2(
                samples, occupancy_grid, timing_policy
            )
            authorities = _occupancy_authorities(
                task,
                provider=provider,
                position=position,
                evidence=evidence,
                timing=timing,
                grid_policy=grid_policy,
                transform=transform,
                result=occupancy,
            )
            bindings = build_provider_occupancy_v2_source_bindings(
                archive,
                selection_run_path=selection_paths[role],
                trajectory_run_path=trajectory_path,
                compiled_selection=selection,
                trajectory=trajectory,
                result=occupancy,
                provider_authority=authorities["provider"],
                timing_authority=authorities["timing"],
                geometry_authority=authorities["geometry"],
                transform_authority=authorities["transform"],
                fixed_grid_policy_authority=authorities["fixed_grid_policy"],
            )
            occupancy_name = str(output_names["occupancies"][provider][role])
            occupancy_path = f"analysis/provider_occupancy_runs/{occupancy_name}"
            if archive.joinpath(*occupancy_path.split("/")).exists():
                raise ProviderSpatialCanaryError(
                    f"Refusing existing immutable occupancy run: {occupancy_path}."
                )
            occupancy_result = materialize_provider_occupancy_v2(
                archive,
                occupancy,
                bindings,
                scratch_root=scratch_root / f"occupancy-{provider}-{role}",
                run_name=occupancy_name,
                apply=True,
                keep_scratch=False,
            )
            published_root = open_zarr_root(
                archive, mode="r", use_consolidated=True
            )
            occupancy_manifest_digest = _digest(
                published_root[occupancy_path].attrs.get(
                    PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR
                ),
                field=f"{provider} {role} occupancy manifest digest",
            )
            provider_products[role] = {
                "trajectory": trajectory,
                "trajectory_path": trajectory_path,
                "occupancy": occupancy,
                "occupancy_path": occupancy_path,
                "occupancy_manifest_sha256": occupancy_manifest_digest,
            }
            provider_stages[role] = {
                "trajectory": {
                    "status": trajectory_result["status"],
                    "run_path": trajectory_result["run_path"],
                    "run_manifest_sha256": trajectory_result[
                        "run_manifest_sha256"
                    ],
                    "array_manifest_sha256": trajectory_result[
                        "array_manifest_sha256"
                    ],
                },
                "occupancy": {
                    "status": occupancy_result["status"],
                    "run_path": occupancy_path,
                    "manifest_sha256": occupancy_manifest_digest,
                },
                "coverage": trajectory.counts.as_record(),
            }
        products[provider] = provider_products
        stages[provider] = provider_stages
    return products, stages


def publish_provider_spatial_contrasts(
    task: Mapping[str, Any],
    *,
    products: Mapping[str, Any],
    output_names: Mapping[str, Any],
    scratch_root: Path,
) -> dict[str, Any]:
    archive = Path(str(task["analysis_zarr"]))
    stages: dict[str, Any] = {}
    for provider in _PROVIDERS:
        provider_stages: list[dict[str, Any]] = []
        for index, comparison in enumerate(task["contrasts"]):
            baseline_role = comparison["baseline"]
            treatment_role = comparison["treatment"]
            baseline_product = products[provider][baseline_role]
            treatment_product = products[provider][treatment_role]
            baseline = build_pooled_occupancy_contrast_summary(
                archive,
                run_path=baseline_product["occupancy_path"],
                manifest_sha256=baseline_product["occupancy_manifest_sha256"],
                arm_role="baseline",
                source_scope=SOURCE_SCOPE_POOLED,
            )
            treatment = build_pooled_occupancy_contrast_summary(
                archive,
                run_path=treatment_product["occupancy_path"],
                manifest_sha256=treatment_product["occupancy_manifest_sha256"],
                arm_role="treatment",
                source_scope=SOURCE_SCOPE_POOLED,
            )
            contrast = compute_occupancy_contrast(
                baseline,
                treatment,
                config={
                    "selection_policy": "same_provider_named_goodbatbadbat_epoch_v1",
                    "baseline_selection": baseline_role,
                    "treatment_selection": treatment_role,
                },
            )
            run_name = str(output_names["contrasts"][provider][index])
            plan = build_provider_occupancy_contrast_materialization_plan(
                archive,
                baseline_run_path=baseline_product["occupancy_path"],
                treatment_run_path=treatment_product["occupancy_path"],
                baseline_manifest_digest=baseline_product[
                    "occupancy_manifest_sha256"
                ],
                treatment_manifest_digest=treatment_product[
                    "occupancy_manifest_sha256"
                ],
                contrast_result=contrast,
                run_name=run_name,
                scratch_root=scratch_root / f"contrast-{provider}-{index}",
                source_scope=SOURCE_SCOPE_POOLED,
            )
            result = publish_provider_occupancy_contrast_run(
                plan, keep_scratch=False
            )
            provider_stages.append(
                {
                    "baseline": baseline_role,
                    "treatment": treatment_role,
                    "run_path": result["run_path"],
                    "status": result["status"],
                    "manifest_sha256": result["manifest_sha256"],
                }
            )
        stages[provider] = provider_stages
    return stages


def materialize_canary(task: Mapping[str, Any], *, scratch_root: Path) -> dict[str, Any]:
    """Publish the exact canary chain without changing selectors or registry state."""

    archive = Path(str(task["analysis_zarr"]))
    scratch = scratch_root.expanduser().resolve()
    if scratch == archive or scratch.is_relative_to(archive):
        raise ProviderSpatialCanaryError(
            "scratch_root must be outside the authoritative analysis archive."
        )
    outputs = planned_run_names(task)
    # Immutable scientific products cannot be reused without a dedicated
    # loader that reconstitutes and compares the exact in-memory result.  A
    # partial canary therefore requires a new campaign_id rather than a
    # cardinality- or name-based resume.  Selection and tracking runs have
    # strict loaders and are the only safe reusable campaign products.
    collisions: list[str] = []
    for provider in _PROVIDERS:
        for role in _SELECTIONS:
            for parent, name in (
                (
                    "analysis/provider_spatial_trajectory_runs",
                    outputs["trajectories"][provider][role],
                ),
                (
                    "analysis/provider_occupancy_runs",
                    outputs["occupancies"][provider][role],
                ),
            ):
                path = f"{parent}/{name}"
                if archive.joinpath(*path.split("/")).exists():
                    collisions.append(path)
        for name in outputs["contrasts"][provider]:
            path = f"analysis/provider_occupancy_contrast_runs/{name}"
            if archive.joinpath(*path.split("/")).exists():
                collisions.append(path)
    if collisions:
        raise ProviderSpatialCanaryError(
            "Immutable canary outputs already exist; choose a new campaign_id: "
            f"{collisions!r}."
        )

    timing, subject = load_recording_and_subject_authorities(task)
    grid_policy, transform, geometry_evidence = load_grid_and_transform_authority(
        task
    )
    compiled, epoch_evidence = compile_canary_selections(task, timing=timing)
    timeline_ids = {value.authority.timeline_id for value in compiled.values()}
    if len(timeline_ids) != 1:
        raise ProviderSpatialCanaryError(
            "Compiled role selections do not share one exact timeline authority."
        )
    selection_paths, selection_stages = publish_canary_selections(
        task,
        compiled=compiled,
        output_names=outputs,
        scratch_root=scratch,
    )
    provider_inputs, provider_source_stages = load_provider_track_inputs(
        task,
        timeline_authority_id=next(iter(timeline_ids)),
        output_names=outputs,
        scratch_root=scratch,
    )
    products, product_stages = publish_provider_spatial_products(
        task,
        compiled=compiled,
        selection_paths=selection_paths,
        providers=provider_inputs,
        timing=timing,
        grid_policy=grid_policy,
        transform=transform,
        output_names=outputs,
        scratch_root=scratch,
    )
    contrast_stages = publish_provider_spatial_contrasts(
        task,
        products=products,
        output_names=outputs,
        scratch_root=scratch,
    )
    return json_attr_safe(
        {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": task["campaign_id"],
            "recording_id": task["recording_id"],
            "analysis_zarr": str(archive),
            "disposition": CANARY_DISPOSITION,
            "selector_updates": False,
            "registry_updates": False,
            "source_payloads_rewritten": False,
            "authorities": {
                "recording_timing": {
                    "sha256": timing.sha256,
                    "fps": timing.nominal_fps,
                    "frame_count": timing.frame_count,
                },
                "subject": subject,
                "epoch": epoch_evidence,
                "geometry": geometry_evidence,
            },
            "output_runs": outputs,
            "stages": {
                "selections": selection_stages,
                "provider_sources": provider_source_stages,
                "spatial_products": product_stages,
                "contrasts": contrast_stages,
            },
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task = load_task(args.task_json)
    if args.apply:
        result = materialize_canary(task, scratch_root=args.scratch_root)
        write_json_atomic(args.result_json, result)
    else:
        result = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "planned",
            "planned_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "output_runs": planned_run_names(task),
            "selector_updates": False,
            "registry_updates": False,
            "source_payloads_rewritten": False,
        }
    print(json.dumps(json_attr_safe(result), indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANARY_DISPOSITION",
    "ProviderSpatialCanaryError",
    "load_task",
    "materialize_canary",
    "planned_run_names",
]

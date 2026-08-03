"""Metadata-file-only availability checks for analysis workflow planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from fisheye.analysis_workflows.storage_contract_catalog import (
    DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
)
from fisheye.registry.stage_catalog import canonical_stage_id
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent_attrs,
    is_run_selector_eligible_attrs,
    resolve_latest_complete_run_name_from_attrs,
)


STAGE_RUN_PARENTS: Mapping[str, tuple[str, ...]] = {
    "refined_keypoints": ("refined_keypoints_runs",),
    "refined_subject_masks": ("refined_subject_masks_runs",),
    "tracks": ("tracking_runs",),
    "tail_posture_view": ("analysis/tail_posture_view_runs",),
    "bout_classification": ("analysis/bout_classification_runs",),
    **DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS,
}
TRACK_KINEMATICS_VISUALIZATION_STAGE = "track_kinematics_visualization"
TRACK_KINEMATICS_PARENT = "analysis/track_kinematics_runs/offline"
TRACK_KINEMATICS_VISUALIZATION_PARENT = (
    "analysis/track_kinematics_visualization_runs/offline"
)
TRACK_KINEMATICS_INTERACTIVE_ARTIFACT = (
    "visualizations/track_kinematics_summary_track_0_interactive"
)
TRACK_KINEMATICS_INTERACTIVE_RENDERER = "palette-track-kinematics-summary-v1"


@dataclass(frozen=True)
class StageAvailability:
    stage_id: str
    available: bool
    artifact_path: str | None = None
    run_name: str | None = None
    reason: str = ""
    completion_status: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "available": self.available,
            "artifact_path": self.artifact_path,
            "run_name": self.run_name,
            "reason": self.reason,
            "completion_status": self.completion_status,
        }


def _attrs(path: Path) -> dict[str, object]:
    metadata_path = path / "zarr.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    raw = payload.get("attributes") if isinstance(payload, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _safe_run_name(value: str) -> str:
    run_name = str(value).strip().strip("/")
    pure = PurePosixPath(run_name)
    if not run_name or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe workflow run selection: {value!r}")
    return run_name


def _completion_status(attrs: Mapping[str, object]) -> str | None:
    for key in (
        "palette_run_completion_status",
        "status",
        "run_status",
        "completion_status",
    ):
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return None


def stage_run_relative_path(stage_id: str, run_name: str) -> str:
    """Return the sole registered run path for an executable canonical stage."""

    canonical = canonical_stage_id(stage_id)
    if canonical == TRACK_KINEMATICS_VISUALIZATION_STAGE:
        return f"{TRACK_KINEMATICS_VISUALIZATION_PARENT}/{_safe_run_name(run_name)}"
    parents = STAGE_RUN_PARENTS.get(canonical)
    if not parents:
        raise KeyError(f"no run parent is registered for stage {canonical!r}")
    if len(parents) != 1:
        raise ValueError(
            f"stage {canonical!r} has ambiguous run parents; execution must select one"
        )
    return f"{parents[0]}/{_safe_run_name(run_name)}"


def _metadata_child_names(parent: Path) -> tuple[str, ...]:
    try:
        return tuple(
            child.name
            for child in parent.iterdir()
            if child.is_dir() and (child / "zarr.json").is_file()
        )
    except OSError:
        return ()


def _resolve_metadata_run_name(
    parent: Path,
    parent_attrs: Mapping[str, object],
    requested_run: str | None,
    *,
    parent_relative_path: str,
) -> tuple[str | None, str | None]:
    """Resolve one strict maintained run through the shared lifecycle contract."""

    requested = "" if requested_run is None else str(requested_run).strip().strip("/")
    parent_prefix = str(parent_relative_path).strip().strip("/") + "/"
    if requested.startswith(parent_prefix):
        requested = requested[len(parent_prefix) :]
    if requested and requested.lower() != "latest":
        run_name = _safe_run_name(requested)
        run_attrs = _attrs(parent / run_name)
        if not (parent / run_name / "zarr.json").is_file():
            return run_name, "selected run metadata is missing"
        if not is_run_complete_in_parent_attrs(
            parent_attrs,
            run_attrs,
            legacy_default=False,
        ):
            return run_name, "selected run is not complete"
        if not is_run_selector_eligible_attrs(run_attrs):
            return run_name, "selected run is not selector-eligible"
        return run_name, None

    run_name = resolve_latest_complete_run_name_from_attrs(
        parent_attrs=parent_attrs,
        child_names=_metadata_child_names(parent),
        child_attrs=lambda name: (
            _attrs(parent / name)
            if (parent / name / "zarr.json").is_file()
            else None
        ),
        legacy_default=False,
    )
    if run_name is None:
        return None, (
            "no stable complete selector-eligible run is selected; "
            "selector activation may be in progress"
        )
    return run_name, None


def _track_kinematics_visualization_availability(
    root: Path,
    *,
    requested_run: str | None,
    dependency_runs: Mapping[str, str] | None,
) -> StageAvailability:
    """Resolve the explorer contract embedded in one offline kinematics run."""

    parent = root / TRACK_KINEMATICS_PARENT
    if not (parent / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            reason="persisted track-kinematics run parent is missing",
        )
    parent_attrs = _attrs(parent)
    run_name, selection_error = _resolve_metadata_run_name(
        parent,
        parent_attrs,
        requested_run,
        parent_relative_path=TRACK_KINEMATICS_PARENT,
    )
    if run_name is None or selection_error is not None:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=(
                f"{TRACK_KINEMATICS_PARENT}/{run_name}"
                if run_name
                else TRACK_KINEMATICS_PARENT
            ),
            run_name=run_name,
            reason=selection_error or "track-kinematics run selection failed",
        )

    run_path = parent / run_name
    run_relative_path = f"{TRACK_KINEMATICS_PARENT}/{run_name}"
    if not (run_path / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=run_relative_path,
            run_name=run_name,
            reason="selected track-kinematics run metadata is missing",
        )
    run_attrs = _attrs(run_path)
    status = _completion_status(run_attrs)

    visualization_parent_relative = stage_run_relative_path(
        TRACK_KINEMATICS_VISUALIZATION_STAGE, run_name
    ) + "/tracks/id_0"
    visualization_parent = root / visualization_parent_relative
    if not (visualization_parent / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=visualization_parent_relative,
            run_name=run_name,
            reason="sibling track-kinematics visualization parent is missing",
            completion_status=status,
        )
    visualization_parent_attrs = _attrs(visualization_parent)
    render_name, render_selection_error = _resolve_metadata_run_name(
        visualization_parent,
        visualization_parent_attrs,
        None,
        parent_relative_path=visualization_parent_relative,
    )
    if render_name is None or render_selection_error is not None:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=visualization_parent_relative,
            run_name=run_name,
            reason=render_selection_error or "visualization render selection failed",
            completion_status=status,
        )
    render_relative_path = f"{visualization_parent_relative}/{render_name}"
    render_path = root / render_relative_path
    render_attrs = _attrs(render_path)
    render_status = _completion_status(render_attrs)
    artifact_relative_path = (
        f"{render_relative_path}/{TRACK_KINEMATICS_INTERACTIVE_ARTIFACT}"
    )
    artifact_path = root / artifact_relative_path
    if not (artifact_path / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive track-kinematics contract is missing",
            completion_status=status,
        )
    if not (artifact_path / "spec_json" / "zarr.json").is_file():
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive track-kinematics contract lacks spec_json",
            completion_status=status,
        )
    artifact_attrs = _attrs(artifact_path)
    renderer = str(artifact_attrs.get("renderer") or "").strip()
    if renderer != TRACK_KINEMATICS_INTERACTIVE_RENDERER:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason=(
                "interactive track-kinematics contract has unsupported renderer "
                f"{renderer!r}"
            ),
            completion_status=status,
        )
    motion_authority = artifact_attrs.get("track_motion_authority")
    motion_authority = (
        dict(motion_authority) if isinstance(motion_authority, Mapping) else {}
    )
    expected_run_ref = f"/{run_relative_path}"
    expected_track_ref = f"{expected_run_ref}/tracks/id_0"
    if (
        motion_authority.get("run_ref") != expected_run_ref
        or motion_authority.get("track_ref") != expected_track_ref
        or motion_authority.get("track_id") != 0
        or not str(motion_authority.get("motion_manifest_sha256") or "").strip()
        or not str(
            motion_authority.get("positions_px_coordinate_descriptor_sha256")
            or ""
        ).strip()
        or render_attrs.get("source_track_motion_authority") != motion_authority
        or render_attrs.get("track_id") != 0
    ):
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract lacks exact track-motion authority",
            completion_status=render_status,
        )
    expected_runs = dict(dependency_runs or {})
    source_runs = artifact_attrs.get("source_runs")
    source_runs = dict(source_runs) if isinstance(source_runs, Mapping) else {}
    expected_track_run = expected_runs.get("track_kinematics")
    persisted_track_run = str(source_runs.get("track_kinematics") or "").strip()
    if expected_track_run and persisted_track_run not in {
        expected_track_run,
        f"offline/{expected_track_run}",
    }:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract track-kinematics lineage does not match",
            completion_status=status,
        )
    parameters = artifact_attrs.get("parameters")
    parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
    expected_swim_bout_run = expected_runs.get("swim_bouts")
    persisted_swim_bout_run = str(parameters.get("swim_bout_run") or "").strip()
    if expected_swim_bout_run and persisted_swim_bout_run != expected_swim_bout_run:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=artifact_relative_path,
            run_name=run_name,
            reason="interactive contract swim-bout lineage does not match",
            completion_status=status,
        )
    return StageAvailability(
        stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
        available=True,
        artifact_path=artifact_relative_path,
        run_name=run_name,
        reason="persisted interactive track-kinematics contract is available",
        completion_status=render_status,
    )


def discover_stage_availability(
    zarr_path: str | Path,
    stage_id: str,
    *,
    requested_run: str | None = None,
    dependency_runs: Mapping[str, str] | None = None,
) -> StageAvailability:
    """Resolve one persisted run using direct ``zarr.json`` reads only."""

    canonical = canonical_stage_id(stage_id)
    root = Path(zarr_path)
    if canonical == TRACK_KINEMATICS_VISUALIZATION_STAGE:
        return _track_kinematics_visualization_availability(
            root,
            requested_run=requested_run,
            dependency_runs=dependency_runs,
        )
    parents = STAGE_RUN_PARENTS.get(canonical)
    if not parents:
        return StageAvailability(
            stage_id=canonical,
            available=False,
            reason="no metadata-only availability resolver is registered",
        )
    for relative_parent in parents:
        parent = root / relative_parent
        if not (parent / "zarr.json").is_file():
            continue
        parent_attrs = _attrs(parent)
        run_name, selection_error = _resolve_metadata_run_name(
            parent,
            parent_attrs,
            requested_run,
            parent_relative_path=relative_parent,
        )
        if run_name is None or selection_error is not None:
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=(
                    f"{relative_parent}/{run_name}"
                    if run_name
                    else relative_parent
                ),
                run_name=run_name,
                reason=selection_error or "run selection failed",
            )
        run_path = parent / run_name
        relative_run_path = f"{relative_parent}/{run_name}"
        if not (run_path / "zarr.json").is_file():
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=relative_run_path,
                run_name=run_name,
                reason="selected run metadata is missing",
            )
        run_attrs = _attrs(run_path)
        status = _completion_status(run_attrs)
        return StageAvailability(
            stage_id=canonical,
            available=True,
            artifact_path=relative_run_path,
            run_name=run_name,
            reason="persisted complete selector-eligible run is available",
            completion_status=status,
        )
    return StageAvailability(
        stage_id=canonical,
        available=False,
        reason="persisted run parent is missing",
    )


__all__ = [
    "STAGE_RUN_PARENTS",
    "StageAvailability",
    "discover_stage_availability",
    "stage_run_relative_path",
]

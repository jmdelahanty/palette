"""Metadata-file-only availability checks for analysis workflow planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from fisheye.registry.stage_catalog import canonical_stage_id


STAGE_RUN_PARENTS: Mapping[str, tuple[str, ...]] = {
    "refined_keypoints": ("refined_keypoints_runs",),
    "refined_subject_masks": ("refined_subject_masks_runs",),
    "tracks": ("tracking_runs",),
    "track_kinematics": ("analysis/track_kinematics_runs/offline",),
    "swim_bouts": ("analysis/swim_bout_runs",),
    "bout_kinematics": ("analysis/bout_kinematics_runs",),
    "eye_angles": ("analysis/eye_angle_runs",),
    "subject_shape": ("analysis/subject_shape_runs",),
    "tail_kinematics": ("analysis/tail_kinematics_runs",),
    "tail_posture_view": ("analysis/tail_posture_view_runs",),
    "bout_classification": ("analysis/bout_classification_runs",),
}
POINTER_KEYS = ("latest_complete", "latest_materialized", "latest")
COMPLETE_STATUSES = frozenset({"complete", "completed", "ok", "success"})
COMPLETION_EPOCH_ATTR = "palette_completion_epoch"
RUN_COMPLETION_CONTRACT_ATTR = "palette_run_completion_contract"
RUN_COMPLETION_CONTRACT = "palette.zarr_run_completion.v1"
TRACK_KINEMATICS_VISUALIZATION_STAGE = "track_kinematics_visualization"
TRACK_KINEMATICS_PARENT = "analysis/track_kinematics_runs/offline"
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
        return (
            f"{TRACK_KINEMATICS_PARENT}/{_safe_run_name(run_name)}/"
            f"{TRACK_KINEMATICS_INTERACTIVE_ARTIFACT}"
        )
    parents = STAGE_RUN_PARENTS.get(canonical)
    if not parents:
        raise KeyError(f"no run parent is registered for stage {canonical!r}")
    if len(parents) != 1:
        raise ValueError(
            f"stage {canonical!r} has ambiguous run parents; execution must select one"
        )
    return f"{parents[0]}/{_safe_run_name(run_name)}"


def _strict_completion_parent(attrs: Mapping[str, object]) -> bool:
    value = attrs.get(COMPLETION_EPOCH_ATTR)
    if value is None or isinstance(value, bool):
        return False
    try:
        return int(value) >= 1
    except (TypeError, ValueError):
        return False


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
    if requested_run and requested_run != "latest":
        run_name = _safe_run_name(requested_run)
    else:
        run_name = ""
        for key in POINTER_KEYS:
            value = parent_attrs.get(key)
            if isinstance(value, str) and value.strip():
                run_name = _safe_run_name(value)
                break
        if not run_name:
            return StageAvailability(
                stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
                available=False,
                artifact_path=TRACK_KINEMATICS_PARENT,
                reason="track-kinematics parent has no latest pointer",
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
    has_palette_contract = (
        run_attrs.get(RUN_COMPLETION_CONTRACT_ATTR) == RUN_COMPLETION_CONTRACT
    )
    if status is not None and status not in COMPLETE_STATUSES:
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=run_relative_path,
            run_name=run_name,
            reason=f"selected track-kinematics run is not complete ({status})",
            completion_status=status,
        )
    if status is None and (
        has_palette_contract or _strict_completion_parent(parent_attrs)
    ):
        return StageAvailability(
            stage_id=TRACK_KINEMATICS_VISUALIZATION_STAGE,
            available=False,
            artifact_path=run_relative_path,
            run_name=run_name,
            reason="selected track-kinematics run lacks a required complete marker",
        )

    artifact_relative_path = stage_run_relative_path(
        TRACK_KINEMATICS_VISUALIZATION_STAGE,
        run_name,
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
        completion_status=status,
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
        pointer_key: str | None = None
        if requested_run and requested_run != "latest":
            run_name = _safe_run_name(requested_run)
        else:
            run_name = ""
            for key in POINTER_KEYS:
                value = parent_attrs.get(key)
                if isinstance(value, str) and value.strip():
                    run_name = _safe_run_name(value)
                    pointer_key = key
                    break
            if not run_name:
                return StageAvailability(
                    stage_id=canonical,
                    available=False,
                    artifact_path=relative_parent,
                    reason="run parent has no latest pointer; select an explicit run",
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
        has_palette_contract = (
            run_attrs.get(RUN_COMPLETION_CONTRACT_ATTR) == RUN_COMPLETION_CONTRACT
        )
        if status is not None and status not in COMPLETE_STATUSES:
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=relative_run_path,
                run_name=run_name,
                reason=f"selected run is not complete ({status})",
                completion_status=status,
            )
        if status is None and (
            has_palette_contract or _strict_completion_parent(parent_attrs)
        ):
            return StageAvailability(
                stage_id=canonical,
                available=False,
                artifact_path=relative_run_path,
                run_name=run_name,
                reason="selected run lacks a required complete marker",
                completion_status=status,
            )
        if status in COMPLETE_STATUSES:
            reason = "persisted complete run is available"
        elif status is None:
            reason = "persisted legacy run is available (parent is not completion-strict)"
        else:
            reason = f"persisted run is available ({status})"
        return StageAvailability(
            stage_id=canonical,
            available=True,
            artifact_path=relative_run_path,
            run_name=run_name,
            reason=reason,
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

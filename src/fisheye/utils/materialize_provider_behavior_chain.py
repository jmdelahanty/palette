"""Materialize one explicit selector-ineligible provider behavior chain.

The task document freezes every source and destination run name.  This utility
never resolves ``latest`` and never changes a production selector.  It exists
to make commit-pinned cohort canaries retryable without combining scientific
selection with publication.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.stimulus_epoch_runs import (
    build_stimulus_epoch_result,
    write_stimulus_epoch_run,
)
from fisheye.analysis_workflows.body_frame_source_handle import (
    load_body_frame_source_handle,
)
from fisheye.analysis_workflows.materializers.provider_epoch_behavior_summary import (
    materialize_provider_epoch_behavior_summary,
)
from fisheye.analysis_workflows.materializers.provider_track_motion import (
    plan_provider_track_motion_run,
    prepare_provider_track_motion,
    publish_provider_track_motion_run,
)
from fisheye.analysis_workflows.materializers.single_subject_tracking import (
    plan_single_subject_tracking_run,
    publish_single_subject_tracking_run,
)
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    materialize_stimulus_epoch_candidate,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.materializers.swim_bouts import materialize_swim_bouts
from fisheye.analysis_workflows.position_body_frame_motion import (
    bind_position_body_frame_to_tracking,
    compose_position_body_frame_motion_authority,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    load_provider_track_motion_source_handle,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.anatomy_profile import load_anatomy_profile
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.subject_position_expression import (
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_keypoint_source import (
    KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
    KeypointPositionSourcePolicy,
    load_bound_keypoint_position_source,
)
from fisheye.shared.subject_position_preparation import prepare_subject_position_input


TASK_SCHEMA_ID = "palette.provider_behavior_chain_task"
LEGACY_TASK_SCHEMA_VERSION = 1
SEMANTIC_TASK_SCHEMA_VERSION = 2
# Preserve the original constant for legacy task producers.
TASK_SCHEMA_VERSION = LEGACY_TASK_SCHEMA_VERSION
SUPPORTED_TASK_SCHEMA_VERSIONS = (
    LEGACY_TASK_SCHEMA_VERSION,
    SEMANTIC_TASK_SCHEMA_VERSION,
)
RESULT_SCHEMA_ID = "palette.provider_behavior_chain_result"
RESULT_SCHEMA_VERSION = 1
LINEAR_ONLY_DISPOSITION = "linear_motion_and_swim_bouts_only_heading_excluded_v1"
KEYPOINT_BINDING_ID = "zebrafish_larva_keypoint_traditional_v2_v1"
SUMMARY_SPEED_LEVEL = "filtered"


class ProviderBehaviorChainError(ValueError):
    """Raised when an explicit cohort task is incomplete or inconsistent."""


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderBehaviorChainError(f"{label} must be one JSON object.")
    return dict(value)


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderBehaviorChainError(f"{label} must be one nonempty exact string.")
    return value


def _run_name(value: Any, *, label: str) -> str:
    name = _text(value, label=label)
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ProviderBehaviorChainError(f"{label} must be one bare run name.")
    return name


def load_task(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = _object(json.loads(source.read_text(encoding="utf-8")), label="task")
    schema_version = payload.get("schema_version")
    if (
        payload.get("schema_id") != TASK_SCHEMA_ID
        or type(schema_version) is not int
        or schema_version not in SUPPORTED_TASK_SCHEMA_VERSIONS
    ):
        raise ProviderBehaviorChainError("Task schema identity is unsupported.")
    payload["recording_id"] = _text(payload.get("recording_id"), label="recording_id")
    archive = Path(_text(payload.get("analysis_zarr"), label="analysis_zarr")).resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {archive}")
    if archive.name != f"{payload['recording_id']}_analysis.zarr":
        raise ProviderBehaviorChainError(
            "Analysis Zarr basename differs from the frozen recording identity."
        )
    payload["analysis_zarr"] = str(archive)
    sources = _object(payload.get("source_runs"), label="source_runs")
    outputs = _object(payload.get("output_runs"), label="output_runs")
    for key in ("keypoint", "body_frame", "stimulus"):
        sources[key] = _run_name(sources.get(key), label=f"source_runs.{key}")
    for key in (
        "position",
        "tracking",
        "motion",
        "swim_bouts",
        "stimulus_epochs_v1",
        "stimulus_epochs_v2",
        "epoch_summary",
    ):
        outputs[key] = _run_name(outputs.get(key), label=f"output_runs.{key}")
    payload["source_runs"] = sources
    payload["output_runs"] = outputs
    semantic_run = payload.get("protocol_semantic_selection_run")
    if schema_version == LEGACY_TASK_SCHEMA_VERSION and semantic_run is not None:
        raise ProviderBehaviorChainError(
            "protocol_semantic_selection_run requires provider behavior chain task v2."
        )
    if schema_version == SEMANTIC_TASK_SCHEMA_VERSION and semantic_run is None:
        raise ProviderBehaviorChainError(
            "Provider behavior chain task v2 requires protocol_semantic_selection_run."
        )
    if semantic_run is not None:
        payload["protocol_semantic_selection_run"] = _run_name(
            semantic_run,
            label="protocol_semantic_selection_run",
        )
    fps = payload.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0:
        raise ProviderBehaviorChainError("fps must be one positive number.")
    payload["fps"] = float(fps)
    if payload.get("metric_disposition") != LINEAR_ONLY_DISPOSITION:
        raise ProviderBehaviorChainError(
            "The task must explicitly bind the linear-only talk metric disposition."
        )
    return payload


def _path(archive: Path, parent: str, run_name: str) -> Path:
    return archive.joinpath(*parent.split("/"), run_name)


def _position(task: Mapping[str, Any], scratch: Path) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    sources = task["source_runs"]
    outputs = task["output_runs"]
    run_path = f"analysis/subject_position_runs/observation/{outputs['position']}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_subject_position_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        return handle, {"status": "reused", "run_path": handle.run_path}
    profile_path = Path(_text(task.get("anatomy_profile"), label="anatomy_profile"))
    profile = load_anatomy_profile(profile_path)
    keypoint_path = f"keypoints_runs/{sources['keypoint']}"
    keypoints = load_bound_keypoint_position_source(
        archive,
        run_path=keypoint_path,
        policy=KeypointPositionSourcePolicy(
            anatomy_profile=profile,
            binding_id=KEYPOINT_BINDING_ID,
            authority_mode=KEYPOINT_AUTHORITY_MODE_COORDINATE_SUCCESSOR_CANARY,
        ),
    )
    prepared = prepare_subject_position_input(
        keypoints,
        estimator_id=KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        software_record=_object(task.get("software"), label="software"),
    )
    plan = plan_subject_position_run(
        archive,
        prepared,
        run_name=outputs["position"],
        scratch_root=scratch / "position",
    )
    result = publish_subject_position_run(plan, keep_scratch=False)
    handle = load_subject_position_source_handle(
        archive,
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "publication": result["publication"],
    }


def _tracking(
    task: Mapping[str, Any], position: Any, scratch: Path
) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    run_name = task["output_runs"]["tracking"]
    run_path = f"tracking_runs/{run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_tracking_source_handle(
            archive,
            run_path,
            expected_selector_eligible=False,
            use_consolidated=True,
        )
        return handle, {"status": "reused", "run_path": handle.run_path}
    plan = plan_single_subject_tracking_run(
        position,
        arena_id=int(task.get("arena_id", 0)),
        run_name=run_name,
        scratch_root=scratch / "tracking",
    )
    result = publish_single_subject_tracking_run(plan, keep_scratch=False)
    handle = load_tracking_source_handle(
        archive,
        run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "publication": result["publication"],
    }


def _motion(
    task: Mapping[str, Any], position: Any, tracking: Any, scratch: Path
) -> tuple[Any, dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    run_name = task["output_runs"]["motion"]
    run_path = f"analysis/track_kinematics_runs/provider/{run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        handle = load_provider_track_motion_source_handle(
            archive,
            run_path,
            use_consolidated=True,
            require_authoritative_timing=False,
        )
        return handle, {"status": "reused", "run_path": handle.run_path}
    body_path = f"analysis/body_frame_runs/{task['source_runs']['body_frame']}"
    body_frame = load_body_frame_source_handle(
        archive,
        run_path=body_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )
    authority = compose_position_body_frame_motion_authority(position, body_frame)
    tracked = bind_position_body_frame_to_tracking(authority, tracking)
    prepared = prepare_provider_track_motion(
        tracked,
        fps=float(task["fps"]),
        smooth_seconds=0.05,
        hysteresis_high_px=4.0,
        hysteresis_low_px=2.0,
        hysteresis_min_frames=3,
    )
    plan = plan_provider_track_motion_run(
        archive,
        prepared,
        run_name=run_name,
        scratch_root=scratch / "motion",
    )
    result = publish_provider_track_motion_run(plan, keep_scratch=False)
    handle = load_provider_track_motion_source_handle(
        archive,
        run_path,
        use_consolidated=True,
        require_authoritative_timing=False,
    )
    return handle, {
        "status": "published",
        "run_path": handle.run_path,
        "manifest_sha256": handle.provider_manifest_sha256,
        "publication": result["publication"],
    }


def _swim_bouts(task: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    archive = Path(str(task["analysis_zarr"]))
    outputs = task["output_runs"]
    run_name = outputs["swim_bouts"]
    target = _path(archive, "analysis/swim_bout_runs", run_name)
    if target.exists():
        return {"status": "reused", "run_path": f"analysis/swim_bout_runs/{run_name}"}
    writer_arguments = (
        "--track-kinematics-scope",
        "provider",
        "--track-kinematics-run",
        outputs["motion"],
        "--track-id",
        "0",
        "--method",
        "peak_event",
        "--default-level",
        "exponential",
        "--exponential-tau-s",
        "0.25",
        "--min-peak-prominence-mm-s",
        "4.0",
        "--min-peak-distance-s",
        "0.10",
        "--peak-width-rel-height",
        "0.98",
        "--layout",
        "compact_v2",
    )
    result = materialize_swim_bouts(
        archive,
        scratch_root=scratch / "swim_bouts",
        run_name=run_name,
        writer_arguments=writer_arguments,
        promote=False,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
    )
    return {"status": "published", "run_path": f"analysis/swim_bout_runs/{run_name}", "result": result}


def _epochs(task: Mapping[str, Any], scratch: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    archive = Path(str(task["analysis_zarr"]))
    sources = task["source_runs"]
    outputs = task["output_runs"]
    v1_name = outputs["stimulus_epochs_v1"]
    v1_path = _path(archive, "analysis/stimulus_epoch_runs", v1_name)
    if v1_path.exists():
        v1 = {"status": "reused", "run_path": f"analysis/stimulus_epoch_runs/{v1_name}"}
    else:
        epoch_result = build_stimulus_epoch_result(
            archive,
            run_name=v1_name,
            stimulus_run=sources["stimulus"],
        )
        written = write_stimulus_epoch_run(
            archive,
            epoch_result,
            overwrite=False,
            selector_ineligible=True,
        )
        v1 = {"status": "published", "run_path": str(written)}
    v2_name = outputs["stimulus_epochs_v2"]
    v2_path = _path(archive, "analysis/stimulus_epoch_runs", v2_name)
    if v2_path.exists():
        v2 = {"status": "reused", "run_path": f"analysis/stimulus_epoch_runs/{v2_name}"}
    else:
        result = materialize_stimulus_epoch_candidate(
            archive,
            source_run=v1_name,
            run_name=v2_name,
            scratch_root=scratch / "stimulus_epochs_v2",
            allow_selector_ineligible_source=True,
            copy_backend="python",
            apply=True,
            keep_scratch=False,
        )
        v2 = {"status": "published", "run_path": f"analysis/stimulus_epoch_runs/{v2_name}", "result": result}
    return v1, v2


def _summary(task: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    archive = Path(str(task["analysis_zarr"]))
    outputs = task["output_runs"]
    run_name = outputs["epoch_summary"]
    run_path = f"analysis/stimulus_epoch_behavior_summary_runs/{run_name}"
    target = archive.joinpath(*run_path.split("/"))
    if target.exists():
        return {"status": "reused", "run_path": run_path}
    result = materialize_provider_epoch_behavior_summary(
        archive,
        scratch_root=scratch / "epoch_summary",
        run_name=run_name,
        epoch_run_name=outputs["stimulus_epochs_v2"],
        protocol_semantic_selection_run_name=task.get(
            "protocol_semantic_selection_run"
        ),
        motion_run=outputs["motion"],
        swim_bout_run_name=outputs["swim_bouts"],
        track_id=0,
        speed_level=SUMMARY_SPEED_LEVEL,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
    )
    return {"status": "published", "run_path": run_path, "result": result}


def materialize_chain(task: Mapping[str, Any], *, scratch_root: Path) -> dict[str, Any]:
    scratch = scratch_root.expanduser().resolve()
    archive = Path(str(task["analysis_zarr"]))
    if scratch == archive or scratch.is_relative_to(archive):
        raise ProviderBehaviorChainError("Scratch root must be outside the archive.")
    stages: dict[str, Any] = {}
    position, stages["position"] = _position(task, scratch)
    tracking, stages["tracking"] = _tracking(task, position, scratch)
    motion, stages["motion"] = _motion(task, position, tracking, scratch)
    if not np.isclose(float(motion.computation_record["parameters"]["fps"]), float(task["fps"]), rtol=0.0, atol=1e-12):
        raise ProviderBehaviorChainError("Published motion FPS differs from the task.")
    stages["swim_bouts"] = _swim_bouts(task, scratch)
    stages["stimulus_epochs_v1"], stages["stimulus_epochs_v2"] = _epochs(task, scratch)
    stages["epoch_summary"] = _summary(task, scratch)
    return json_attr_safe(
        {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "recording_id": task["recording_id"],
            "analysis_zarr": str(archive),
            "metric_disposition": LINEAR_ONLY_DISPOSITION,
            "selector_updates": False,
            "source_payloads_rewritten": False,
            "stages": stages,
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
    if not args.apply:
        result: Mapping[str, Any] = {
            "schema_id": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "planned",
            "task": task,
            "selector_updates": False,
        }
    else:
        result = materialize_chain(task, scratch_root=args.scratch_root)
        write_json_atomic(args.result_json, result)
    print(json.dumps(json_attr_safe(result), indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LINEAR_ONLY_DISPOSITION",
    "SUMMARY_SPEED_LEVEL",
    "ProviderBehaviorChainError",
    "load_task",
    "materialize_chain",
]

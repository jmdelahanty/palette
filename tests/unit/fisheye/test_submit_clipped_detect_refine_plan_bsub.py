from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import zarr

from fisheye.utils.submit_clipped_detect_refine_plan_bsub import (
    PLAN_SCHEMA,
    build_submission_bundle,
)


def _unit(index: int) -> dict[str, object]:
    clip_id = f"clip_{index:06d}"
    work_unit_id = f"recording_{clip_id}_cam2010093"
    detect_run = f"detect_wf_{clip_id}_cam2010093"
    return {
        "work_unit_id": work_unit_id,
        "clip_id": clip_id,
        "camera_serial": "2010093",
        "run_names": {
            "detect": detect_run,
            "detect_quality": f"detect_quality_wf_{clip_id}_cam2010093",
            "refined_detect": f"refined_detect_wf_{clip_id}_cam2010093",
        },
        "artifact_paths": {
            "expected_tarball": f"/shared/artifacts/{work_unit_id}/{work_unit_id}.<detect_jobid>.tar.gz",
        },
        "zarr_paths": {
            "detect_target_group_path": f"clips/{clip_id}/cameras/2010093/detect_runs/{detect_run}",
            "refined_group_path": f"clips/{clip_id}/cameras/2010093/refined_detect_runs/refined_detect_wf_{clip_id}_cam2010093",
        },
        "commands": {
            "detect_submit": f"scripts/submit_detect_artifact_bsub.sh --run-label {work_unit_id}",
            "validate_detect": f"scripts/py -m fisheye.utils.validate_imported_run_group analysis.zarr --target-group-path clips/{clip_id}/cameras/2010093/detect_runs/{detect_run}",
            "detect_quality": f"scripts/py -m fisheye.refinement.detect_quality analysis.zarr --run {detect_run}",
            "refine_detect": f"scripts/py -m fisheye.refinement.refine_detect analysis.zarr --detect-run {detect_run}",
            "validate_refined_detect": f"scripts/py -m fisheye.utils.validate_refined_detect_run analysis.zarr --target-group-path clips/{clip_id}/cameras/2010093/refined_detect_runs/refined_detect_wf_{clip_id}_cam2010093",
        },
    }


def _write_plan(path: Path, *, units: int = 1, analysis_zarr: Path | None = None) -> Path:
    payload = {
        "schema_version": PLAN_SCHEMA,
        "workflow_id": "wf",
        "work_units": [_unit(index) for index in range(units)],
    }
    if analysis_zarr is not None:
        payload["analysis_zarr"] = str(analysis_zarr)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_submission_bundle_dry_run_writes_dependency_scripts(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json")

    manifest = build_submission_bundle(
        plan_path,
        repo=tmp_path,
        run_dir=tmp_path / "submission",
    )

    assert manifest["status"] == "dry_run"
    assert manifest["work_unit_count"] == 1
    assert manifest["target_preflight"]["status"] == "unchecked"
    item = manifest["work_items"][0]
    assert item["detect_job_id"] == "<detect_jobid:recording_clip_000000_cam2010093>"
    assert (
        item["detect_artifact_tarball"]
        == "/shared/artifacts/recording_clip_000000_cam2010093/"
        "recording_clip_000000_cam2010093.<detect_jobid:recording_clip_000000_cam2010093>.tar.gz"
    )

    stages = {stage["stage"]: stage for stage in item["stages"]}
    for stage in stages.values():
        subprocess.run(["bash", "-n", stage["script"]], check=True)

    assert stages["import_detect"]["dependency"] == "done(<detect_jobid:recording_clip_000000_cam2010093>)"
    assert stages["validate_detect"]["dependency"] == "done(<import_detect_jobid:recording_clip_000000_cam2010093>)"
    assert stages["detect_quality"]["dependency"] == "done(<validate_detect_jobid:recording_clip_000000_cam2010093>)"
    assert stages["refine_detect"]["dependency"] == "done(<detect_quality_jobid:recording_clip_000000_cam2010093>)"
    assert stages["validate_refined_detect"]["dependency"] == "done(<refine_detect_jobid:recording_clip_000000_cam2010093>)"

    finalizer = manifest["finalizer"]
    assert finalizer["stage"] == "finalize_recording_collection"
    assert (
        finalizer["dependency"]
        == "done(<validate_refined_detect_jobid:recording_clip_000000_cam2010093>)"
    )
    assert "fisheye.utils.finalize_clipped_detect_refine_workflow" in finalizer["command"]
    assert "--submission-manifest" in finalizer["command"]
    assert "--apply" in finalizer["command"]

    import_script = Path(stages["import_detect"]["script"])
    script_text = import_script.read_text(encoding="utf-8")
    assert "STATUS_JSON_TEMPLATE=" in script_text
    assert "fisheye.utils.import_run_group_artifact" in script_text
    assert "--use-intended-target --apply" in script_text
    assert (
        "trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_stage_status failed \"$rc\" || true; fi' EXIT"
        in script_text
    )
    assert "write_stage_status ok 0" in script_text
    assert '"exit_code": int(os.environ.get("STAGE_EXIT_CODE", "0"))' in script_text

    finalizer_script = Path(finalizer["script"])
    subprocess.run(["bash", "-n", str(finalizer_script)], check=True)
    finalizer_text = finalizer_script.read_text(encoding="utf-8")
    assert "finalize_recording_collection" in finalizer_text
    assert "STAGE_SECONDS" in finalizer_text
    assert "write_stage_status failed" in finalizer_text
    assert "write_stage_status ok 0" in finalizer_text
    assert '"exit_code": int(os.environ.get("STAGE_EXIT_CODE", "0"))' in finalizer_text

    manifest_path = Path(manifest["submission_manifest"])
    assert manifest_path.exists()


def test_submit_mode_refuses_multiple_units_without_allow_multiple(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path / "plan.json", units=2)

    with pytest.raises(ValueError, match="Refusing to submit more than one"):
        build_submission_bundle(
            plan_path,
            repo=tmp_path,
            run_dir=tmp_path / "submission",
            submit=True,
        )


def test_submit_mode_refuses_existing_output_targets(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("clips/clip_000000/cameras/2010093/detect_runs/detect_wf_clip_000000_cam2010093")
    plan_path = _write_plan(tmp_path / "plan.json", analysis_zarr=zarr_path)

    dry_run = build_submission_bundle(
        plan_path,
        repo=tmp_path,
        run_dir=tmp_path / "dry_run_submission",
    )
    assert dry_run["target_preflight"]["status"] == "blocked"
    assert dry_run["target_preflight"]["collisions"][0]["kind"] == "detect_target"

    with pytest.raises(ValueError, match="planned output targets already exist"):
        build_submission_bundle(
            plan_path,
            repo=tmp_path,
            run_dir=tmp_path / "submission",
            submit=True,
        )

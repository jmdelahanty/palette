from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils.check_clipped_detect_refine_submission import (
    check_clipped_detect_refine_submission,
)
from fisheye.utils.submit_clipped_detect_refine_plan_bsub import SUBMISSION_SCHEMA


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, *, job_id: str = "123") -> Path:
    run_dir = tmp_path / "run"
    unit_dir = run_dir / "unit"
    tarball = tmp_path / "artifacts" / f"unit.{job_id}.tar.gz"
    manifest = {
        "schema_version": SUBMISSION_SCHEMA,
        "status": "submitted",
        "workflow_id": "wf",
        "run_dir": str(run_dir),
        "work_unit_count": 1,
        "work_items": [
            {
                "work_unit_id": "recording_clip_000000_cam2010093",
                "clip_id": "clip_000000",
                "camera_serial": "2010093",
                "detect_job_id": job_id,
                "detect_artifact_tarball": str(tarball),
                "stages": [
                    {
                        "stage": "import_detect",
                        "job_id": "124",
                        "dependency": f"done({job_id})",
                        "script": str(unit_dir / "scripts" / "run_import_detect.sh"),
                        "status_json": str(unit_dir / "status" / "import_detect.124.json"),
                        "stdout": str(unit_dir / "logs" / "import_detect_124.out"),
                        "stderr": str(unit_dir / "logs" / "import_detect_124.err"),
                    },
                    {
                        "stage": "validate_refined_detect",
                        "job_id": "125",
                        "dependency": "done(124)",
                        "script": str(unit_dir / "scripts" / "run_validate_refined_detect.sh"),
                        "status_json": str(unit_dir / "status" / "validate_refined_detect.125.json"),
                        "stdout": str(unit_dir / "logs" / "validate_refined_detect_125.out"),
                        "stderr": str(unit_dir / "logs" / "validate_refined_detect_125.err"),
                    },
                ],
            }
        ],
        "finalizer": {
            "stage": "finalize_recording_collection",
            "job_id": "126",
            "dependency": "done(125)",
            "script": str(run_dir / "finalizer" / "run.sh"),
            "status_json": str(run_dir / "finalizer" / "status" / "finalizer.126.json"),
            "stdout": str(run_dir / "finalizer" / "logs" / "finalizer_126.out"),
            "stderr": str(run_dir / "finalizer" / "logs" / "finalizer_126.err"),
        },
    }
    return _write_json(tmp_path / "submission_manifest.json", manifest)


def test_check_clipped_detect_refine_submission_reports_ok(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    tarball = tmp_path / "artifacts" / "unit.123.tar.gz"
    tarball.parent.mkdir(parents=True)
    tarball.write_bytes(b"tarball")
    _write_json(
        tmp_path / "artifacts" / "unit.123.summary.json",
        {
            "status": "ok",
            "run_name": "detect_wf_clip_000000_cam2010093",
            "target_group_path": "detect_runs/detect_wf_clip_000000_cam2010093",
            "intended_target_group_path": "clips/clip_000000/cameras/2010093/detect_runs/detect_wf_clip_000000_cam2010093",
        },
    )
    for path in [
        tmp_path / "run" / "unit" / "status" / "import_detect.124.json",
        tmp_path / "run" / "unit" / "status" / "validate_refined_detect.125.json",
        tmp_path / "run" / "finalizer" / "status" / "finalizer.126.json",
    ]:
        _write_json(path, {"status": "ok", "exit_code": 0, "stage_seconds": 1.25})

    result = check_clipped_detect_refine_submission(manifest_path)

    assert result["status"] == "ok"
    assert result["status_counts"] == {"ok": 4}


def test_check_clipped_detect_refine_submission_reports_incomplete(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)

    result = check_clipped_detect_refine_submission(manifest_path)

    assert result["status"] == "incomplete"
    assert result["status_counts"]["missing"] == 4


def test_check_clipped_detect_refine_submission_reports_failed_stage(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    tarball = tmp_path / "artifacts" / "unit.123.tar.gz"
    tarball.parent.mkdir(parents=True)
    tarball.write_bytes(b"tarball")
    _write_json(tmp_path / "artifacts" / "unit.123.summary.json", {"status": "ok"})
    _write_json(tmp_path / "run" / "unit" / "status" / "import_detect.124.json", {"status": "failed", "exit_code": 2})

    result = check_clipped_detect_refine_submission(manifest_path)

    assert result["status"] == "failed"
    assert result["status_counts"]["failed"] == 1
    assert result["status_counts"]["missing"] == 2


def test_check_clipped_detect_refine_submission_keeps_dry_run_planned(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path, job_id="<detect_jobid:unit>")

    result = check_clipped_detect_refine_submission(manifest_path)

    assert result["status"] == "incomplete"
    assert result["work_items"][0]["detect_artifact"]["status"] == "planned"

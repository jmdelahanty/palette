"""Run, publish, and validate one clipped YOLO work unit inside one LSF job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.utils.import_run_group_artifact import apply_import
from fisheye.utils.run_detection_artifact import build_detection_artifact
from fisheye.utils.validate_imported_run_group import validate_imported_run_group


REPORT_SCHEMA = "palette.clipped_detection_work_unit_report.v1"


def _scratch_root() -> Path:
    user = os.environ.get("USER") or "palette"
    job_id = os.environ.get("LSB_JOBID") or "local"
    preferred = Path("/scratch") / user / job_id / "palette_clipped_detection"
    if preferred.parent.parent.exists():
        return preferred
    return Path(tempfile.gettempdir()) / user / job_id / "palette_clipped_detection"


def run_work_unit(
    *,
    video_path: Path,
    target_zarr: Path,
    target_group_path: str,
    model_path: Path,
    model_sha256: str,
    model_registry_set_id: str,
    model_registry_run_id: str,
    config_path: Path,
    workflow_id: str,
    recording_id: str,
    clip_id: str,
    clip_index: int,
    camera_serial: str,
    run_name: str,
    report_path: Path,
    batch_size: int = 16,
    decode_backend: str = "pynvvc_luma_rgb",
) -> dict[str, Any]:
    """Build the artifact locally, atomically import it, and validate the import."""

    scratch = _scratch_root()
    if scratch.exists():
        shutil.rmtree(scratch)
    artifact_dir = scratch / "artifact"
    work_dir = scratch / "work"
    tarball = scratch / "detect_run.tar.gz"

    artifact = build_detection_artifact(
        video_path=video_path,
        target_zarr=target_zarr,
        artifact_dir=artifact_dir,
        work_dir=work_dir,
        tarball_output=tarball,
        model_path=model_path,
        model_sha256=model_sha256,
        model_registry_set_id=model_registry_set_id,
        model_registry_run_id=model_registry_run_id,
        config_path=config_path,
        batch_size=int(batch_size),
        decode_backend=decode_backend,
        latest_policy="do_not_set_latest",
        workflow_id=workflow_id,
        recording_id=recording_id,
        clip_id=clip_id,
        clip_index=int(clip_index),
        camera_serial=camera_serial,
        run_name=run_name,
        command=[sys.executable, "-m", "fisheye.utils.run_clipped_detection_work_unit"],
    )
    imported = apply_import(tarball_path=tarball, use_intended_target=True)
    if imported.get("status") != "ok" or not imported.get("applied"):
        raise RuntimeError(
            "Detection artifact import failed: "
            + json.dumps(imported, sort_keys=True, default=str)
        )
    validation = validate_imported_run_group(
        zarr_path=target_zarr,
        target_group_path=target_group_path,
    )
    if validation.get("status") != "pass":
        raise RuntimeError(
            "Imported detection validation failed: "
            + json.dumps(validation, sort_keys=True, default=str)
        )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "workflow_id": workflow_id,
        "recording_id": recording_id,
        "clip_id": clip_id,
        "clip_index": int(clip_index),
        "camera_serial": camera_serial,
        "target_zarr": str(target_zarr),
        "target_group_path": target_group_path,
        "run_name": run_name,
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "registry_set_id": model_registry_set_id,
            "registry_run_id": model_registry_run_id,
        },
        "artifact": artifact,
        "import": imported,
        "validation": validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--target-zarr", required=True, type=Path)
    parser.add_argument("--target-group-path", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-registry-set-id", required=True)
    parser.add_argument("--model-registry-run-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--clip-index", required=True, type=int)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-backend", default="pynvvc_luma_rgb")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_work_unit(
        video_path=args.video.expanduser().resolve(),
        target_zarr=args.target_zarr.expanduser().resolve(),
        target_group_path=args.target_group_path,
        model_path=args.model.expanduser().resolve(),
        model_sha256=args.model_sha256,
        model_registry_set_id=args.model_registry_set_id,
        model_registry_run_id=args.model_registry_run_id,
        config_path=args.config.expanduser().resolve(),
        workflow_id=args.workflow_id,
        recording_id=args.recording_id,
        clip_id=args.clip_id,
        clip_index=args.clip_index,
        camera_serial=args.camera_serial,
        run_name=args.run_name,
        report_path=args.report.expanduser().resolve(),
        batch_size=args.batch_size,
        decode_backend=args.decode_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

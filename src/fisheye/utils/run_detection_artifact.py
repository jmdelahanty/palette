"""Run detection into scratch and package the detect run group as an artifact."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fisheye.utils.system import get_environment_info, get_git_info

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at import time
    Console = None  # type: ignore


ARTIFACT_SCHEMA = "palette_run_group_artifact_v1"
RUN_FAMILY = "detect_runs"
DECODE_BACKEND_CHOICES = (
    "auto",
    "pynvvc_luma_rgb",
    "pynvvc_nv12_rgb",
    "decord_gpu",
    "decord_cpu",
    "opencv",
)
LATEST_POLICY_CHOICES = (
    "do_not_set_latest",
    "set_latest_if_newer",
    "set_latest_explicit",
)
REQUIRED_DETECT_ARRAYS = (
    "frame_indices",
    "bbox_norm_coords",
    "scores",
    "class_ids",
    "n_detections",
    "frame_counts",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "detect_artifact"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json_strict(path: Path) -> Any:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _find_non_finite(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        findings.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            findings.extend(_find_non_finite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            findings.extend(_find_non_finite(child, f"{path}[{idx}]"))
    return findings


def strict_json_report(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    bad_files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("zarr.json")):
        rel = path.relative_to(root).as_posix()
        try:
            payload = _read_json_strict(path)
            non_finite = _find_non_finite(payload)
            item = {
                "path": rel,
                "status": "ok" if not non_finite else "failed",
                "non_finite_paths": non_finite,
            }
        except Exception as exc:
            item = {"path": rel, "status": "failed", "error": str(exc)}
        files.append(item)
        if item["status"] != "ok":
            bad_files.append(item)
    return {
        "status": "pass" if not bad_files else "fail",
        "files_checked": len(files),
        "bad_json_files": len(bad_files),
        "bad_files": bad_files,
    }


def required_arrays_report(run_group_dir: Path) -> dict[str, Any]:
    arrays: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in REQUIRED_DETECT_ARRAYS:
        zarr_json = run_group_dir / name / "zarr.json"
        present = zarr_json.exists()
        arrays.append({"name": name, "zarr_json": str(zarr_json), "present": present})
        if not present:
            missing.append(name)
    attrs_present = (run_group_dir / "zarr.json").exists()
    return {
        "status": "pass" if attrs_present and not missing else "fail",
        "run_group_zarr_json_present": attrs_present,
        "arrays": arrays,
        "missing_arrays": missing,
    }


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _copy_run_group(source_run_group: Path, artifact_dir: Path) -> Path:
    run_group_dir = artifact_dir / "run_group"
    if run_group_dir.exists():
        shutil.rmtree(run_group_dir)
    shutil.copytree(source_run_group, run_group_dir)
    return run_group_dir


def _make_tarball(artifact_dir: Path, tarball_output: Path) -> Path:
    tarball_output.parent.mkdir(parents=True, exist_ok=True)
    if tarball_output.exists():
        tarball_output.unlink()
    with tarfile.open(tarball_output, "w:gz") as tar:
        tar.add(artifact_dir, arcname=artifact_dir.name)
    return tarball_output


def _cluster_provenance() -> dict[str, Optional[str]]:
    keys = (
        "LSB_JOBID",
        "LSB_JOBINDEX",
        "LSB_JOBNAME",
        "LSB_QUEUE",
        "LSB_DJOB_NUMPROC",
        "CUDA_VISIBLE_DEVICES",
        "PALETTE_JOB_CACHE",
    )
    return {key: os.environ.get(key) for key in keys}


def _clip_context(
    *,
    workflow_id: Optional[str],
    recording_id: Optional[str],
    clip_id: Optional[str],
    clip_index: Optional[int],
    camera_serial: Optional[str],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if workflow_id:
        context["workflow_id"] = workflow_id
    if recording_id:
        context["recording_id"] = recording_id
    if clip_id:
        context["clip_id"] = clip_id
    if clip_index is not None:
        context["clip_index"] = int(clip_index)
    if camera_serial:
        context["camera_serial"] = camera_serial
    if clip_id and camera_serial:
        context["scope"] = "clip_camera"
        context["clip_camera_key"] = f"{clip_id}/camera_{camera_serial}"
    elif context:
        context["scope"] = "partial_clip_context"
    return context


def _intended_target_group_path(*, run_name: str, clip_context: dict[str, Any]) -> str:
    clip_id = clip_context.get("clip_id")
    camera_serial = clip_context.get("camera_serial")
    if isinstance(clip_id, str) and clip_id and isinstance(camera_serial, str) and camera_serial:
        return f"clips/{clip_id}/cameras/{_safe_label(camera_serial)}/{RUN_FAMILY}/{run_name}"
    return f"{RUN_FAMILY}/{run_name}"


def _stderr_console() -> Any:
    if Console is None:
        return None
    return Console(stderr=True)


def _detect_yolo(**kwargs: Any) -> str:
    # Keep the Ultralytics import out of argparse/help paths because importing
    # detect_yolo initializes third-party settings/cache state.
    from fisheye.detection.detect_yolo import detect_yolo

    return detect_yolo(**kwargs)


def _extract_timing(source_run_group: Path) -> dict[str, Any]:
    zarr_json = source_run_group / "zarr.json"
    if not zarr_json.exists():
        return {}
    try:
        payload = _read_json_strict(zarr_json)
    except Exception:
        return {}
    attrs = payload.get("attributes") if isinstance(payload, dict) else None
    if not isinstance(attrs, dict):
        return {}
    timing = attrs.get("timing_summary")
    return timing if isinstance(timing, dict) else {}


def build_detection_artifact(
    *,
    video_path: Path,
    target_zarr: Path,
    artifact_dir: Path,
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_det: Optional[int] = None,
    batch_size: Optional[int] = None,
    resize_dims: Optional[Sequence[int]] = None,
    imgsz: Optional[Sequence[int]] = None,
    decode_backend: Optional[str] = None,
    use_gpu: Optional[bool] = None,
    latest_policy: str = "do_not_set_latest",
    work_dir: Optional[Path] = None,
    tarball_output: Optional[Path] = None,
    overwrite_artifact: bool = False,
    command: Optional[Sequence[str]] = None,
    workflow_id: Optional[str] = None,
    recording_id: Optional[str] = None,
    clip_id: Optional[str] = None,
    clip_index: Optional[int] = None,
    camera_serial: Optional[str] = None,
    run_name: Optional[str] = None,
) -> dict[str, Any]:
    """Run YOLO into scratch and package only the completed detect run group."""
    video_path = video_path.expanduser().resolve()
    target_zarr = target_zarr.expanduser().resolve()
    artifact_dir = artifact_dir.expanduser().resolve()
    work_dir = (
        work_dir.expanduser().resolve()
        if work_dir is not None
        else artifact_dir.parent / f".{artifact_dir.name}.work"
    )
    tarball_output = (
        tarball_output.expanduser().resolve()
        if tarball_output is not None
        else artifact_dir.parent / f"{artifact_dir.name}.tar.gz"
    )

    if latest_policy not in LATEST_POLICY_CHOICES:
        raise ValueError(f"latest_policy must be one of {LATEST_POLICY_CHOICES}")
    if not video_path.exists():
        raise FileNotFoundError(f"video path does not exist: {video_path}")
    if not target_zarr.exists():
        raise FileNotFoundError(f"target analysis zarr does not exist: {target_zarr}")
    if artifact_dir.exists():
        if not overwrite_artifact:
            raise FileExistsError(f"artifact directory already exists: {artifact_dir}")
        shutil.rmtree(artifact_dir)
    if work_dir.exists():
        if not overwrite_artifact:
            raise FileExistsError(f"work directory already exists: {work_dir}")
        shutil.rmtree(work_dir)

    artifact_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    artifact_start = time.perf_counter()
    artifact_timing: dict[str, Any] = {"schema_version": 1}
    scratch_zarr = work_dir / "detect_output.zarr"
    runtime_environment = get_environment_info(
        include_all_packages=False,
        disk_path=str(scratch_zarr),
        collect_ip=False,
        capture_env_vars=True,
    )
    command_list = list(command) if command is not None else sys.argv
    command_text = " ".join(command_list)
    clip_context = _clip_context(
        workflow_id=workflow_id,
        recording_id=recording_id,
        clip_id=clip_id,
        clip_index=clip_index,
        camera_serial=camera_serial,
    )
    _write_json(
        artifact_dir / "logs" / "job_context.json",
        {
            "created_at": _utc_now(),
            "command": command_text,
            "hostname": platform.node(),
            "cluster": _cluster_provenance(),
            "runtime": runtime_environment,
            "scratch_zarr": str(scratch_zarr),
            "clip_context": clip_context,
        },
    )
    (artifact_dir / "logs" / "command.log").write_text(command_text + "\n", encoding="utf-8")

    detect_start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        run_name = _detect_yolo(
            video_path=str(video_path),
            model_path=str(model_path) if model_path is not None else None,
            output_zarr=str(scratch_zarr),
            config_path=str(config_path) if config_path is not None else None,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            batch_size=batch_size,
            resize_dims=list(resize_dims) if resize_dims is not None else None,
            imgsz=list(imgsz) if imgsz is not None else None,
            decode_backend=decode_backend,
            console=_stderr_console(),
            use_gpu=use_gpu,
            write_raw_video_metadata=False,
            overwrite_raw_video_metadata=False,
            run_name=run_name,
        )
    artifact_timing["detect_yolo_seconds_total"] = time.perf_counter() - detect_start

    source_run_group = scratch_zarr / RUN_FAMILY / run_name
    if not source_run_group.exists():
        raise FileNotFoundError(f"detect run group was not written: {source_run_group}")
    copy_start = time.perf_counter()
    run_group_dir = _copy_run_group(source_run_group, artifact_dir)
    artifact_timing["copy_run_group_seconds_total"] = time.perf_counter() - copy_start

    strict_start = time.perf_counter()
    strict_report = strict_json_report(run_group_dir)
    artifact_timing["strict_json_validation_seconds_total"] = time.perf_counter() - strict_start
    arrays_start = time.perf_counter()
    arrays_report = required_arrays_report(run_group_dir)
    artifact_timing["required_array_validation_seconds_total"] = time.perf_counter() - arrays_start
    validation_write_start = time.perf_counter()
    _write_json(artifact_dir / "validation" / "strict_json_report.json", strict_report)
    _write_json(artifact_dir / "validation" / "array_presence_report.json", arrays_report)
    artifact_timing["validation_report_write_seconds_total"] = (
        time.perf_counter() - validation_write_start
    )

    hash_start = time.perf_counter()
    run_group_hash = tree_hash(run_group_dir)
    artifact_timing["run_group_tree_hash_seconds_total"] = time.perf_counter() - hash_start
    git_info = get_git_info()
    timing = _extract_timing(source_run_group)
    target_group_path = f"{RUN_FAMILY}/{run_name}"
    intended_target_group_path = _intended_target_group_path(
        run_name=run_name,
        clip_context=clip_context,
    )
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "created_at": _utc_now(),
        "target_archive_path": str(target_zarr),
        "target_group_path": target_group_path,
        "intended_target_group_path": intended_target_group_path,
        "run_family": RUN_FAMILY,
        "run_name": run_name,
        "layout": "detect_yolo_sparse_v1",
        "schema_version": 1,
        "latest_policy": latest_policy,
        "artifact_scope": clip_context.get("scope", "archive_top_level"),
        "clip_context": clip_context,
        "source_inputs": [
            {"path": str(video_path), "role": "source_video"},
            {"path": str(target_zarr), "role": "target_analysis_archive"},
        ],
        "provenance": {
            "palette_git_commit": git_info.get("commit_hash", "unknown"),
            "palette_git_short": git_info.get("short_hash", "unknown"),
            "palette_git_branch": git_info.get("branch", "unknown"),
            "palette_git_is_dirty": git_info.get("is_dirty", False),
            "command": command_text,
            "hostname": platform.node(),
            "cluster": _cluster_provenance(),
            "runtime": runtime_environment,
            "decoder_backend": decode_backend or "auto",
            "scratch_zarr": str(scratch_zarr),
            "clip_context": clip_context,
        },
        "timing": timing,
        "artifact_timing": artifact_timing,
        "checksums": {"run_group_tree_hash": run_group_hash},
        "validation": {
            "strict_json": strict_report["status"],
            "required_arrays": arrays_report["status"],
            "canonical_write": "not_performed",
        },
    }
    manifest_write_start = time.perf_counter()
    _write_json(artifact_dir / "artifact_manifest.json", manifest)
    artifact_timing["manifest_write_seconds_total"] = time.perf_counter() - manifest_write_start

    tarball_start = time.perf_counter()
    tarball_path = _make_tarball(artifact_dir, tarball_output)
    artifact_timing["tarball_seconds_total"] = time.perf_counter() - tarball_start
    artifact_timing["artifact_seconds_total"] = time.perf_counter() - artifact_start
    summary = {
        "status": (
            "ok"
            if strict_report["status"] == "pass" and arrays_report["status"] == "pass"
            else "failed"
        ),
        "artifact_dir": str(artifact_dir),
        "tarball_path": str(tarball_path),
        "scratch_zarr": str(scratch_zarr),
        "run_name": run_name,
        "target_group_path": target_group_path,
        "intended_target_group_path": intended_target_group_path,
        "artifact_scope": clip_context.get("scope", "archive_top_level"),
        "clip_context": clip_context,
        "manifest_path": str(artifact_dir / "artifact_manifest.json"),
        "artifact_timing": artifact_timing,
        "strict_json": strict_report,
        "required_arrays": arrays_report,
    }
    _write_json(artifact_dir / "artifact_summary.json", summary)
    return summary


def _parse_optional_ints(values: Optional[Sequence[str]]) -> Optional[list[int]]:
    if values is None:
        return None
    return [int(value) for value in values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO detection into a scratch Zarr and package the completed "
            "detect run group as a transferable artifact."
        )
    )
    parser.add_argument("video_path", type=Path, help="Input camera video path")
    parser.add_argument("--target-zarr", required=True, type=Path, help="Canonical analysis Zarr this package targets")
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Scratch package root to create")
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch work dir for temporary detect output")
    parser.add_argument("--tarball-output", type=Path, default=None, help="Output .tar.gz path; default is beside artifact dir")
    parser.add_argument("--model", "--model-path", dest="model_path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resize-dims", nargs="+", default=None)
    parser.add_argument("--imgsz", nargs="+", default=None)
    parser.add_argument("--decode-backend", choices=DECODE_BACKEND_CHOICES, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--latest-policy",
        choices=LATEST_POLICY_CHOICES,
        default="do_not_set_latest",
        help="Importer latest policy recorded in the package manifest",
    )
    parser.add_argument("--overwrite-artifact", action="store_true", help="Replace existing artifact/work dirs and tarball")
    parser.add_argument("--workflow-id", default=None, help="Optional workflow id for clipped/batch provenance")
    parser.add_argument("--recording-id", default=None, help="Optional recording id for clipped/batch provenance")
    parser.add_argument("--clip-id", default=None, help="Optional source clip id, e.g. clip_000000")
    parser.add_argument("--clip-index", type=int, default=None, help="Optional zero-based source clip index")
    parser.add_argument("--camera-serial", default=None, help="Optional camera serial for clip-camera provenance")
    parser.add_argument("--run-name", default=None, help="Optional explicit detect run group name")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = build_detection_artifact(
            video_path=args.video_path,
            target_zarr=args.target_zarr,
            artifact_dir=args.artifact_dir,
            model_path=args.model_path,
            config_path=args.config,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=_parse_optional_ints(args.resize_dims),
            imgsz=_parse_optional_ints(args.imgsz),
            decode_backend=args.decode_backend,
            use_gpu=False if args.cpu else None,
            latest_policy=args.latest_policy,
            work_dir=args.work_dir,
            tarball_output=args.tarball_output,
            overwrite_artifact=args.overwrite_artifact,
            command=[sys.executable, "-m", "fisheye.utils.run_detection_artifact", *(argv or sys.argv[1:])],
            workflow_id=args.workflow_id,
            recording_id=args.recording_id,
            clip_id=args.clip_id,
            clip_index=args.clip_index,
            camera_serial=args.camera_serial,
            run_name=args.run_name,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

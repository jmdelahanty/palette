"""Plan a clipped-recording detect -> quality -> refine workflow.

This module is intentionally dry-run only. It emits deterministic run names and
exact commands for each clip-camera work unit so cluster submission can be
reviewed before we fan out across many clips.
"""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from fisheye.shared.clipped_detection_plan_contract import (
    CLIPPED_DETECT_REFINE_WORKFLOW_PLAN_SCHEMA,
)

PLAN_SCHEMA = CLIPPED_DETECT_REFINE_WORKFLOW_PLAN_SCHEMA
DEFAULT_CONFIG = "configs/fisheye/yolo_detect_config.yaml"
DEFAULT_DECODE_BACKEND = "pynvvc_luma_rgb"


def _default_workflow_id() -> str:
    return "clipped_detect_refine_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _safe_component(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return label or fallback


def _resolve_recording_path(recording_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Expected path value, got None")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = recording_dir.resolve()
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Recording-relative path escapes recording root: {value}")
    return resolved


def _clip_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("clips")
    if raw_rows is None:
        raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("camera_artifacts")
    if not isinstance(raw_rows, list):
        raise ValueError("recording_clip_index JSON must include clips, rows, or camera_artifacts list")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("recording_clip_index row is not an object")
        if isinstance(raw.get("camera_artifacts"), list):
            base = {key: value for key, value in raw.items() if key != "camera_artifacts"}
            for artifact in raw["camera_artifacts"]:
                if not isinstance(artifact, Mapping):
                    raise ValueError("camera_artifacts item is not an object")
                rows.append({**base, **dict(artifact)})
        else:
            rows.append(dict(raw))
    return rows


def _default_analysis_zarr(recording_dir: Path) -> Path:
    return recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"


def _command(parts: Sequence[Any]) -> str:
    return shlex.join(str(part) for part in parts if part is not None and str(part) != "")


def _append_optional(parts: list[Any], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        parts.extend([flag, value])


def _normalize_selected(values: Optional[Sequence[str]]) -> set[str] | None:
    if not values:
        return None
    return {str(value) for value in values}


def build_plan(
    recording_dir: str | Path,
    *,
    analysis_zarr: str | Path | None = None,
    model: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    workflow_id: str | None = None,
    output_dir: str | Path | None = None,
    detect_queue: str = "gpu_l4",
    detect_ncores: int = 8,
    detect_mem_gb: int = 120,
    detect_gpus: int = 1,
    detect_walltime: str = "2:00",
    decode_backend: str = DEFAULT_DECODE_BACKEND,
    batch_size: int = 16,
    conf: float | None = None,
    iou: float | None = None,
    max_det: int | None = None,
    quality_threshold: float = 100.0,
    quality_threshold_mode: str = "scaled",
    quality_threshold_reference_width: float = 640.0,
    refine_config: str | Path = "configs/fisheye/default.yaml",
    per_frame_top_k: int | None = 1,
    clip_ids: Optional[Sequence[str]] = None,
    camera_serials: Optional[Sequence[str]] = None,
    limit: int | None = None,
) -> dict[str, Any]:
    recording_path = Path(recording_dir).expanduser().resolve()
    clip_index_path = recording_path / "recording_clip_index.json"
    if not clip_index_path.exists():
        raise FileNotFoundError(f"recording_clip_index.json not found: {clip_index_path}")

    clip_index = _read_json(clip_index_path)
    if not isinstance(clip_index, Mapping):
        raise ValueError(f"recording_clip_index JSON is not an object: {clip_index_path}")

    rows = _clip_rows(clip_index)
    selected_clip_ids = _normalize_selected(clip_ids)
    selected_cameras = _normalize_selected(camera_serials)
    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
        camera_serial = str(row.get("camera_serial") or "")
        if selected_clip_ids is not None and clip_id not in selected_clip_ids:
            continue
        if selected_cameras is not None and camera_serial not in selected_cameras:
            continue
        filtered_rows.append(row)
    if limit is not None:
        filtered_rows = filtered_rows[: int(limit)]
    if not filtered_rows:
        raise ValueError("No clip-camera rows matched the requested filters")

    workflow = workflow_id or _default_workflow_id()
    workflow_component = _safe_component(workflow, fallback="workflow")
    recording_id = str(clip_index.get("recording_id") or recording_path.name)
    zarr_path = Path(analysis_zarr).expanduser().resolve() if analysis_zarr else _default_analysis_zarr(recording_path).resolve()
    model_path = Path(model).expanduser().resolve()
    config_path = Path(config)
    if not config_path.is_absolute():
        config_display: str | Path = config_path
    else:
        config_display = config_path.resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else (recording_path / "derived" / "cluster_artifacts" / "detect").resolve()
    )

    work_units: list[dict[str, Any]] = []
    for row in filtered_rows:
        clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
        clip_index_value = int(row.get("clip_index") or int(clip_id.rsplit("_", 1)[-1]))
        camera_serial = str(row.get("camera_serial") or "")
        if not camera_serial:
            raise ValueError(f"Missing camera_serial for {clip_id}")
        camera_component = _safe_component(camera_serial, fallback="unknown_camera")
        clip_component = _safe_component(clip_id, fallback=f"clip_{clip_index_value:06d}")
        recording_component = _safe_component(recording_id, fallback=recording_path.name)
        label = _safe_component(
            f"{recording_component}_{clip_component}_cam{camera_component}",
            fallback=f"{clip_component}_cam{camera_component}",
        )
        work_unit_id = label
        detect_run_name = _safe_component(
            f"detect_{workflow_component}_{clip_component}_cam{camera_component}",
            fallback=f"detect_{clip_component}_cam{camera_component}",
        )
        quality_run_name = _safe_component(
            f"detect_quality_{workflow_component}_{clip_component}_cam{camera_component}",
            fallback=f"detect_quality_{clip_component}_cam{camera_component}",
        )
        refined_run_name = _safe_component(
            f"refined_detect_{workflow_component}_{clip_component}_cam{camera_component}",
            fallback=f"refined_detect_{clip_component}_cam{camera_component}",
        )
        video_path = _resolve_recording_path(recording_path, row.get("video_path"))
        artifact_family_path = (
            f"clips/{clip_id}/cameras/{camera_component}/detection_artifact_runs"
        )
        artifact_target_group_path = (
            f"{artifact_family_path}/{detect_run_name}"
        )
        detect_family_path = f"clips/{clip_id}/cameras/{camera_component}/detect_runs"
        refined_family_path = f"clips/{clip_id}/cameras/{camera_component}/refined_detect_runs"
        target_group_path = f"{detect_family_path}/{detect_run_name}"
        refined_group_path = f"{refined_family_path}/{refined_run_name}"
        run_dir = output_root / f"detect_artifact_{workflow_component}_{label}"
        expected_tarball = run_dir / f"{label}.<detect_jobid>.tar.gz"
        expected_summary = run_dir / f"{label}.<detect_jobid>.summary.json"

        detect_submit_parts: list[Any] = [
            "scripts/submit_detect_artifact_bsub.sh",
            "--zarr",
            zarr_path,
            "--video",
            video_path,
            "--model",
            model_path,
            "--output-dir",
            output_root,
            "--config",
            config_display,
            "--queue",
            detect_queue,
            "--ncores",
            detect_ncores,
            "--mem-gb",
            detect_mem_gb,
            "--gpus",
            detect_gpus,
            "--walltime",
            detect_walltime,
            "--decode-backend",
            decode_backend,
            "--batch-size",
            batch_size,
            "--run-id",
            workflow_component,
            "--run-label",
            label,
            "--workflow-id",
            workflow,
            "--recording-id",
            recording_id,
            "--clip-id",
            clip_id,
            "--clip-index",
            clip_index_value,
            "--camera-serial",
            camera_serial,
            "--detect-run-name",
            detect_run_name,
        ]
        _append_optional(detect_submit_parts, "--conf", conf)
        _append_optional(detect_submit_parts, "--iou", iou)
        _append_optional(detect_submit_parts, "--max-det", max_det)

        quality_parts: list[Any] = [
            "scripts/py",
            "-m",
            "fisheye.refinement.detect_quality",
            zarr_path,
            "--run",
            detect_run_name,
            "--detect-family-path",
            detect_family_path,
            "--threshold",
            quality_threshold,
            "--threshold-mode",
            quality_threshold_mode,
            "--threshold-reference-width",
            quality_threshold_reference_width,
            "--quality-run-name",
            quality_run_name,
        ]
        refine_parts: list[Any] = [
            "scripts/py",
            "-m",
            "fisheye.refinement.refine_detect",
            zarr_path,
            "--detect-run",
            detect_run_name,
            "--detect-family-path",
            detect_family_path,
            "--refined-family-path",
            refined_family_path,
            "--quality-run",
            quality_run_name,
            "--config",
            refine_config,
            "--run-name",
            refined_run_name,
        ]
        _append_optional(refine_parts, "--per-frame-top-k", per_frame_top_k)

        work_units.append(
            {
                "work_unit_id": work_unit_id,
                "recording_id": recording_id,
                "clip_id": clip_id,
                "clip_index": clip_index_value,
                "camera_serial": camera_serial,
                "frame_count": int(row["frame_count"]) if row.get("frame_count") is not None else None,
                "source": {
                    "video_path": str(video_path),
                    "metadata_path": str(_resolve_recording_path(recording_path, row["metadata_path"]))
                    if row.get("metadata_path")
                    else None,
                    "keyframe_path": str(_resolve_recording_path(recording_path, row["keyframe_path"]))
                    if row.get("keyframe_path")
                    else None,
                },
                "run_names": {
                    "detect": detect_run_name,
                    "detect_quality": quality_run_name,
                    "refined_detect": refined_run_name,
                },
                "zarr_paths": {
                    "detection_artifact_family_path": artifact_family_path,
                    "detection_artifact_target_group_path": artifact_target_group_path,
                    "detect_family_path": detect_family_path,
                    "detect_target_group_path": target_group_path,
                    "refined_family_path": refined_family_path,
                    "refined_group_path": refined_group_path,
                },
                "artifact_paths": {
                    "run_dir": str(run_dir),
                    "expected_tarball": str(expected_tarball),
                    "expected_summary": str(expected_summary),
                },
                "commands": {
                    "detect_submit": _command(detect_submit_parts),
                    "import_detect": _command(
                        [
                            "scripts/py",
                            "-m",
                            "fisheye.utils.import_run_group_artifact",
                            expected_tarball,
                            "--use-intended-target",
                            "--apply",
                        ]
                    ),
                    "validate_detect": _command(
                        [
                            "scripts/py",
                            "-m",
                            "fisheye.utils.validate_imported_run_group",
                            zarr_path,
                            "--target-group-path",
                            target_group_path,
                        ]
                    ),
                    "detect_quality": _command(quality_parts),
                    "refine_detect": _command(refine_parts),
                    "validate_refined_detect": _command(
                        [
                            "scripts/py",
                            "-m",
                            "fisheye.utils.validate_refined_detect_run",
                            zarr_path,
                            "--target-group-path",
                            refined_group_path,
                        ]
                    ),
                },
                "stage_plan": [
                    {"stage": "detect_artifact", "resource": "gpu", "queue": detect_queue},
                    {"stage": "import_detect", "resource": "cpu", "depends_on": "detect_artifact"},
                    {"stage": "validate_detect", "resource": "cpu", "depends_on": "import_detect"},
                    {"stage": "detect_quality", "resource": "cpu", "depends_on": "validate_detect"},
                    {"stage": "refine_detect", "resource": "cpu", "depends_on": "detect_quality"},
                    {"stage": "validate_refined_detect", "resource": "cpu", "depends_on": "refine_detect"},
                ],
            }
        )

    return {
        "status": "ok",
        "schema_version": PLAN_SCHEMA,
        "generated_at_utc": _utc_now(),
        "dry_run_only": True,
        "recording_dir": str(recording_path),
        "recording_id": recording_id,
        "analysis_zarr": str(zarr_path),
        "recording_clip_index": str(clip_index_path),
        "workflow_id": workflow,
        "model": str(model_path),
        "config": str(config_display),
        "output_dir": str(output_root),
        "decode_backend": decode_backend,
        "work_unit_count": len(work_units),
        "work_units": work_units,
        "finalizer": {
            "stage": "finalize_recording_collection",
            "resource": "cpu",
            "status": "planned_placeholder",
            "depends_on": [unit["work_unit_id"] for unit in work_units],
            "responsibilities": [
                "verify every planned clip-camera refined run exists",
                "write parent recording run-collection manifest",
                "defer any parent-level array merge until consumer contract is implemented",
            ],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run plan for clipped recording detect-quality-refine workflow."
    )
    parser.add_argument("recording_dir", type=Path, help="Recording folder containing recording_clip_index.json")
    parser.add_argument("--analysis-zarr", type=Path, default=None, help="Analysis zarr path; default is recording_dir/zarr/<recording>_analysis.zarr")
    parser.add_argument("--model", required=True, type=Path, help="Detection model path")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Detection config path (default: {DEFAULT_CONFIG})")
    parser.add_argument("--workflow-id", default=None, help="Stable workflow id; default is UTC timestamp")
    parser.add_argument("--output-dir", type=Path, default=None, help="Shared detection artifact output dir")
    parser.add_argument("--detect-queue", default="gpu_l4")
    parser.add_argument("--detect-ncores", type=int, default=8)
    parser.add_argument("--detect-mem-gb", type=int, default=120)
    parser.add_argument("--detect-gpus", type=int, default=1)
    parser.add_argument("--detect-walltime", default="2:00")
    parser.add_argument("--decode-backend", default=DEFAULT_DECODE_BACKEND)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--quality-threshold", type=float, default=100.0)
    parser.add_argument("--quality-threshold-mode", choices=["scaled", "pixels", "normalized"], default="scaled")
    parser.add_argument("--quality-threshold-reference-width", type=float, default=640.0)
    parser.add_argument("--refine-config", default="configs/fisheye/default.yaml")
    parser.add_argument("--per-frame-top-k", type=int, default=1)
    parser.add_argument("--no-per-frame-top-k", action="store_true", help="Do not add --per-frame-top-k to refine commands")
    parser.add_argument("--clip-id", action="append", default=None, help="Limit to one or more clip ids")
    parser.add_argument("--camera-serial", action="append", default=None, help="Limit to one or more camera serials")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of planned clip-camera rows")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the plan JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    plan = build_plan(
        args.recording_dir,
        analysis_zarr=args.analysis_zarr,
        model=args.model,
        config=args.config,
        workflow_id=args.workflow_id,
        output_dir=args.output_dir,
        detect_queue=args.detect_queue,
        detect_ncores=args.detect_ncores,
        detect_mem_gb=args.detect_mem_gb,
        detect_gpus=args.detect_gpus,
        detect_walltime=args.detect_walltime,
        decode_backend=args.decode_backend,
        batch_size=args.batch_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        quality_threshold=args.quality_threshold,
        quality_threshold_mode=args.quality_threshold_mode,
        quality_threshold_reference_width=args.quality_threshold_reference_width,
        refine_config=args.refine_config,
        per_frame_top_k=None if args.no_per_frame_top_k else args.per_frame_top_k,
        clip_ids=args.clip_id,
        camera_serials=args.camera_serial,
        limit=args.limit,
    )
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), plan)
    print(json.dumps(plan, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

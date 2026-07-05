#!/usr/bin/env python3
"""Resolve a detect model from registry, run detect_yolo, and persist resolution provenance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.registry.stage_complete import emit_stage_completion
from fisheye.shared.type_conversions import normalize_attr
from fisheye.utils.model_resolution_provenance import build_model_resolution_payload
from fisheye.registry.model_resolution import Candidate, TargetProfile, load_candidates, load_target_profile, resolve_recording_id

_DETECT_STATUS_SOURCE = "runtime_detect_with_registry_model"

DECODE_BACKEND_CHOICES = (
    "auto",
    "pynvvc_nv12_rgb",
    "pynvvc_luma_rgb",
    "decord_gpu",
    "decord_cpu",
    "opencv",
)


def _emit_detect_step_status(
    *,
    zarr_path: Path,
    status: str,
    run_name: Optional[str],
    reason: Optional[str],
    selected_model_path: Optional[str],
    selected_run_id: Optional[str],
    selected_set_id: Optional[str],
) -> None:
    """Write a detect step status row to the registry (non-fatal)."""
    try:
        resolved_zarr_path = zarr_path.expanduser().resolve()
        root = zarr.open(str(resolved_zarr_path), mode="r")

        # Extract method and coverage from the detect run if available
        method = None
        coverage_pct = None
        if run_name and status == "ok":
            detect_runs = root.get("detect_runs")
            if detect_runs is not None and run_name in detect_runs:
                detect_group = detect_runs[run_name]
                method = normalize_attr(detect_group.attrs.get("detection_method"))
                summary = detect_group.attrs.get("summary_statistics")
                if isinstance(summary, dict):
                    cov = summary.get("coverage_pct") or summary.get("coverage_percent")
                    if cov is not None:
                        try:
                            coverage_pct = float(cov)
                        except (TypeError, ValueError):
                            pass

        emit_stage_completion(
            root,
            resolved_zarr_path,
            step_name="detect",
            status=status,
            source=_DETECT_STATUS_SOURCE,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=None,
            details_json={
                "reason": reason,
                "selected_model_path": selected_model_path,
                "run_id": selected_run_id,
                "set_id": selected_set_id,
            },
            console=None,
            auto_registry_from_env=True,
            require_env_registry_exists=False,
            invalidate_on_ok=True,
            trigger_run_name=run_name,
        )
    except Exception:
        pass  # Non-fatal; batch runners log results via JSONL already


@dataclass(frozen=True)
class DetectRegistryResult:
    ok: bool
    status: str
    recording_dir: str
    output_zarr: str
    registry_path: str
    video_path: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    remediation: Optional[str] = None
    selected_model_path: Optional[str] = None
    selected_run_id: Optional[str] = None
    selected_set_id: Optional[str] = None
    detect_run: Optional[str] = None
    resolved_at_utc: Optional[str] = None
    resolution_payload: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "recording_dir": self.recording_dir,
            "output_zarr": self.output_zarr,
            "registry_path": self.registry_path,
            "video_path": self.video_path,
            "reason": self.reason,
            "error": self.error,
            "remediation": self.remediation,
            "selected_model_path": self.selected_model_path,
            "selected_run_id": self.selected_run_id,
            "selected_set_id": self.selected_set_id,
            "detect_run": self.detect_run,
            "resolved_at_utc": self.resolved_at_utc,
        }
        if self.resolution_payload is not None:
            payload["resolution_payload"] = self.resolution_payload
        return payload


def _resolve_video(recording_dir: Path, explicit_video: Optional[Path]) -> Path:
    if explicit_video is not None:
        path = explicit_video.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise SystemExit(f"Video does not exist: {path}")
        return path
    cams = recording_dir / "cams"
    mp4s = sorted(cams.glob("*.mp4"))
    if len(mp4s) == 1:
        return mp4s[0].resolve()
    if not mp4s:
        raise SystemExit(f"No camera video found in {cams}")
    raise SystemExit(f"Multiple camera videos found in {cams}; pass --video explicitly.")


def _resolve_output(recording_dir: Path, explicit_output: Optional[Path]) -> Path:
    if explicit_output is not None:
        return explicit_output.expanduser().resolve()
    return (recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr").resolve()


def _pick_best_candidate(candidates: list[Candidate], *, require_unique: bool) -> Candidate:
    if not candidates:
        raise SystemExit("No detect model candidates found.")
    best = candidates[0]
    if require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise SystemExit("Top candidate score tied; rerun with --set-id to choose deterministically.")
    return best


def _resolution_payload(
    *,
    args: argparse.Namespace,
    argv: Optional[list[str]],
    recording_dir: Path,
    video_path: Path,
    output_path: Path,
    registry_path: Path,
    recording_id: str,
    target: TargetProfile,
    selected: Candidate,
    candidates: list[Candidate],
    top_k: int,
) -> dict[str, Any]:
    target_payload = asdict(target)

    def _candidate_payload(item: Candidate) -> dict[str, Any]:
        return {
            "run_id": item.run_id,
            "set_id": item.set_id,
            "model_path": item.model_path,
            "score": item.weighted_score,
            "created_utc": item.created_utc,
            "status": item.status,
            "dataset_count": item.dataset_count,
            "feature_match_counts": item.feature_match_counts,
            "feature_weights_used": item.feature_weights_used,
        }

    selected_payload = _candidate_payload(selected)
    candidate_payloads = [_candidate_payload(item) for item in candidates[: max(0, int(top_k))]]

    return build_model_resolution_payload(
        tool="fisheye.utils.run_detect_with_registry_model",
        args=args,
        argv=argv,
        task="detect",
        registry_path=registry_path,
        recording_id=recording_id,
        target=target_payload,
        selected=selected_payload,
        candidates=candidate_payloads,
        parameters={
            "set_id_filter": args.set_id,
            "require_unique": bool(args.require_unique),
            "top_k": int(args.top_k),
            "include_non_success": bool(args.include_non_success),
            "dry_run": bool(args.dry_run),
            "cpu": bool(args.cpu),
            "config": args.config,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "batch_size": args.batch_size,
            "resize_dims": args.resize_dims,
            "imgsz": args.imgsz,
        },
        inputs={
            "recording_dir": str(recording_dir),
            "video_path": str(video_path),
            "output_zarr": str(output_path),
            "recording_id": recording_id,
            "target": target_payload,
        },
        artifacts={
            "selected_model": selected_payload,
            "candidate_models": candidate_payloads,
            "output_zarr": str(output_path),
            "input_video": str(video_path),
        },
    )


def _write_model_resolution_provenance(
    *,
    zarr_path: Path,
    run_name: str,
    payload: dict[str, Any],
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r+")
    detect_parent = root.get("detect_runs")
    if detect_parent is None or run_name not in detect_parent:
        raise RuntimeError(f"detect run not found for provenance annotation: detect_runs/{run_name}")

    detect_group = detect_parent[run_name]
    selected = payload.get("selected", {}) if isinstance(payload.get("selected"), dict) else {}
    attrs = dict(detect_group.attrs)
    attrs["model_resolution_mode"] = "registry"
    attrs["model_resolution_task"] = "detect"
    attrs["model_resolution_registry_path"] = payload.get("registry_path")
    attrs["model_resolution_recording_id"] = payload.get("recording_id")
    attrs["model_resolution_selected_run_id"] = selected.get("run_id")
    attrs["model_resolution_selected_set_id"] = selected.get("set_id")
    attrs["model_resolution_selected_model_path"] = selected.get("model_path")
    attrs["model_resolution_selected_score"] = selected.get("score")
    attrs["model_resolution_selected_created_utc"] = selected.get("created_utc")
    attrs["model_resolution_resolved_at_utc"] = payload.get("resolved_at_utc")
    attrs["model_resolution_candidates_json"] = json.dumps(payload.get("candidates", []), sort_keys=True)

    provenance = attrs.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["model_resolution"] = payload
    attrs["provenance"] = provenance
    detect_group.attrs.put(attrs)


def _failure_result(
    *,
    reason: str,
    error: str,
    remediation: str,
    recording_dir: Path,
    output_path: Path,
    registry_path: Path,
    video_path: Optional[Path] = None,
    selected_model_path: Optional[str] = None,
    selected_run_id: Optional[str] = None,
    selected_set_id: Optional[str] = None,
    detect_run: Optional[str] = None,
    resolved_at_utc: Optional[str] = None,
    resolution_payload: Optional[dict[str, Any]] = None,
) -> DetectRegistryResult:
    return DetectRegistryResult(
        ok=False,
        status="failed",
        reason=reason,
        error=error,
        remediation=remediation,
        recording_dir=str(recording_dir),
        output_zarr=str(output_path),
        registry_path=str(registry_path),
        video_path=str(video_path) if video_path is not None else None,
        selected_model_path=selected_model_path,
        selected_run_id=selected_run_id,
        selected_set_id=selected_set_id,
        detect_run=detect_run,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=resolution_payload,
    )


def _build_payload_args(
    *,
    set_id: Optional[str],
    require_unique: bool,
    top_k: int,
    include_non_success: bool,
    dry_run: bool,
    cpu: bool,
    config: Optional[str],
    conf: Optional[float],
    iou: Optional[float],
    max_det: Optional[int],
    batch_size: Optional[int],
    resize_dims: Optional[list[int]],
    imgsz: Optional[list[int]],
    decode_backend: Optional[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        set_id=set_id,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=bool(dry_run),
        cpu=bool(cpu),
        config=config,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch_size=batch_size,
        resize_dims=resize_dims,
        imgsz=imgsz,
        decode_backend=decode_backend,
    )


def run_detect_with_registry_model(
    *,
    recording_dir: Path,
    video: Optional[Path] = None,
    output: Optional[Path] = None,
    registry: Optional[Path] = None,
    set_id: Optional[str] = None,
    require_unique: bool = False,
    top_k: int = 5,
    include_non_success: bool = False,
    dry_run: bool = False,
    config: Optional[str] = None,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_det: Optional[int] = None,
    batch_size: Optional[int] = None,
    resize_dims: Optional[list[int]] = None,
    imgsz: Optional[list[int]] = None,
    decode_backend: Optional[str] = None,
    cpu: bool = False,
    write_raw_video_metadata: bool = False,
    overwrite_raw_video_metadata: bool = False,
    argv: Optional[list[str]] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> DetectRegistryResult:
    resolved_recording_dir = recording_dir.expanduser().resolve()
    resolved_registry_path = (registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    resolved_output_path = _resolve_output(resolved_recording_dir, output)

    try:
        resolved_video_path = _resolve_video(resolved_recording_dir, video)
    except SystemExit as exc:
        return _failure_result(
            reason="video_resolution_failed",
            error=str(exc),
            remediation="Ensure exactly one cams/*.mp4 exists, or pass --video explicitly.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
        )

    try:
        registry_db = Registry(resolved_registry_path)
    except Exception as exc:
        return _failure_result(
            reason="registry_open_failed",
            error=str(exc),
            remediation="Verify --registry points to a readable palette registry SQLite file.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
            video_path=resolved_video_path,
        )

    try:
        recording_id = resolve_recording_id(
            registry_db,
            recording_id=None,
            recording_dir=resolved_recording_dir,
        )
        target = load_target_profile(registry_db, recording_id)
        candidates = load_candidates(
            registry_db,
            target=target,
            task="detect",
            set_id_filter=set_id,
            include_non_success=bool(include_non_success),
        )
    except Exception as exc:
        return _failure_result(
            reason="model_resolution_failed",
            error=str(exc),
            remediation="Verify registry metadata for this recording and rerun with --include-non-success or --set-id as needed.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
            video_path=resolved_video_path,
        )
    finally:
        registry_db.close()

    try:
        best = _pick_best_candidate(candidates, require_unique=bool(require_unique))
    except SystemExit as exc:
        return _failure_result(
            reason="candidate_selection_failed",
            error=str(exc),
            remediation="Pass --set-id to pin a model set or remove --require-unique.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
            video_path=resolved_video_path,
        )

    payload_args = _build_payload_args(
        set_id=set_id,
        require_unique=bool(require_unique),
        top_k=int(top_k),
        include_non_success=bool(include_non_success),
        dry_run=bool(dry_run),
        cpu=bool(cpu),
        config=config,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch_size=batch_size,
        resize_dims=resize_dims,
        imgsz=imgsz,
        decode_backend=decode_backend,
    )

    payload = _resolution_payload(
        args=payload_args,
        argv=argv,
        recording_dir=resolved_recording_dir,
        video_path=resolved_video_path,
        output_path=resolved_output_path,
        registry_path=resolved_registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(top_k),
    )

    selected_payload = payload.get("selected") if isinstance(payload.get("selected"), dict) else {}
    selected_model_path = selected_payload.get("model_path") if isinstance(selected_payload.get("model_path"), str) else None
    selected_run_id = selected_payload.get("run_id") if isinstance(selected_payload.get("run_id"), str) else None
    selected_set_id = selected_payload.get("set_id") if isinstance(selected_payload.get("set_id"), str) else None
    resolved_at_utc = payload.get("resolved_at_utc") if isinstance(payload.get("resolved_at_utc"), str) else None

    if dry_run:
        return DetectRegistryResult(
            ok=True,
            status="dry_run",
            recording_dir=str(resolved_recording_dir),
            output_zarr=str(resolved_output_path),
            registry_path=str(resolved_registry_path),
            video_path=str(resolved_video_path),
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
            resolved_at_utc=resolved_at_utc,
            resolution_payload=payload,
        )

    try:
        from fisheye.detection.detect_yolo import detect_yolo

        effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
        if effective_run_provenance is None:
            effective_run_provenance = build_run_provenance(
                command="fisheye.utils.run_detect_with_registry_model",
                params={
                    **vars(payload_args),
                    "recording_dir": resolved_recording_dir,
                    "video": resolved_video_path,
                    "output": resolved_output_path,
                    "registry": resolved_registry_path,
                    "selected_model_path": selected_model_path,
                    "selected_run_id": selected_run_id,
                    "selected_set_id": selected_set_id,
                },
                input_run_ids={
                    "model_run": selected_run_id,
                    "model_set": selected_set_id,
                },
                cwd=Path.cwd(),
            )
        run_name = detect_yolo(
            video_path=str(resolved_video_path),
            model_path=best.model_path,
            output_zarr=str(resolved_output_path),
            config_path=config,
            conf_threshold=conf,
            iou_threshold=iou,
            max_det=max_det,
            batch_size=batch_size,
            resize_dims=resize_dims,
            imgsz=imgsz,
            decode_backend=decode_backend,
            use_gpu=(False if cpu else None),
            write_raw_video_metadata=bool(write_raw_video_metadata),
            overwrite_raw_video_metadata=bool(overwrite_raw_video_metadata),
            cli_provenance=effective_run_provenance,
            run_provenance=effective_run_provenance,
        )
    except Exception as exc:
        _emit_detect_step_status(
            zarr_path=resolved_output_path,
            status="error",
            run_name=None,
            reason="detect_inference_failed",
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
        )
        return _failure_result(
            reason="detect_inference_failed",
            error=str(exc),
            remediation="Inspect model/config inputs and rerun with --dry-run --json to verify resolved model selection.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
            video_path=resolved_video_path,
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
            resolved_at_utc=resolved_at_utc,
            resolution_payload=payload,
        )

    try:
        _write_model_resolution_provenance(
            zarr_path=resolved_output_path,
            run_name=run_name,
            payload=payload,
        )
    except Exception as exc:
        return _failure_result(
            reason="provenance_write_failed",
            error=str(exc),
            remediation="Ensure output zarr is writable and detect_runs/<run> exists before retrying.",
            recording_dir=resolved_recording_dir,
            output_path=resolved_output_path,
            registry_path=resolved_registry_path,
            video_path=resolved_video_path,
            selected_model_path=selected_model_path,
            selected_run_id=selected_run_id,
            selected_set_id=selected_set_id,
            detect_run=run_name,
            resolved_at_utc=resolved_at_utc,
            resolution_payload=payload,
        )

    _emit_detect_step_status(
        zarr_path=resolved_output_path,
        status="ok",
        run_name=run_name,
        reason="detect_inference_ok",
        selected_model_path=selected_model_path,
        selected_run_id=selected_run_id,
        selected_set_id=selected_set_id,
    )

    return DetectRegistryResult(
        ok=True,
        status="ok",
        recording_dir=str(resolved_recording_dir),
        output_zarr=str(resolved_output_path),
        registry_path=str(resolved_registry_path),
        video_path=str(resolved_video_path),
        selected_model_path=selected_model_path,
        selected_run_id=selected_run_id,
        selected_set_id=selected_set_id,
        detect_run=run_name,
        resolved_at_utc=resolved_at_utc,
        resolution_payload=payload,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--video", type=Path, help="Optional explicit video path.")
    parser.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    parser.add_argument("--registry", type=Path, help="Optional registry sqlite path.")
    parser.add_argument("--set-id", type=str, help="Optional set filter during model resolution.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to store in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve model only; do not run detect.")

    parser.add_argument("--config", type=str, default=None, help="Optional detect_yolo config path.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="Optional IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=None, help="Optional max detections override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument(
        "--resize-dims",
        nargs="+",
        type=int,
        default=None,
        help="Canonical inference size override [h w] (or one value for square); mapped to YOLO imgsz.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=None,
        help="Legacy alias for YOLO inference size; normalized into --resize-dims.",
    )
    parser.add_argument(
        "--decode-backend",
        choices=DECODE_BACKEND_CHOICES,
        default=None,
        help="Video decode backend passed to detect_yolo.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--write-raw-video-metadata",
        action="store_true",
        help="Write metadata-only raw_video attrs during detect.",
    )
    parser.add_argument(
        "--overwrite-raw-video-metadata",
        action="store_true",
        help="Overwrite existing metadata-only raw_video attrs during detect.",
    )
    parser.add_argument("--json", action="store_true", help="Print resolved payload JSON.")
    args = parser.parse_args(argv)

    result = run_detect_with_registry_model(
        recording_dir=args.recording_dir,
        video=args.video,
        output=args.output,
        registry=args.registry,
        set_id=args.set_id,
        require_unique=bool(args.require_unique),
        top_k=int(args.top_k),
        include_non_success=bool(args.include_non_success),
        dry_run=bool(args.dry_run),
        config=args.config,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        resize_dims=args.resize_dims,
        imgsz=args.imgsz,
        decode_backend=args.decode_backend,
        cpu=bool(args.cpu),
        write_raw_video_metadata=bool(args.write_raw_video_metadata),
        overwrite_raw_video_metadata=bool(args.overwrite_raw_video_metadata),
        argv=argv,
    )

    if args.json or args.dry_run:
        if result.resolution_payload is not None:
            print(json.dumps(result.resolution_payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    if result.status == "dry_run":
        return 0

    if not result.ok:
        print("Detect run failed")
        print(f"  recording_dir: {result.recording_dir}")
        print(f"  output_zarr: {result.output_zarr}")
        print(f"  reason: {result.reason or 'unknown'}")
        if result.error:
            print(f"  error: {result.error}")
        if result.remediation:
            print(f"  remediation: {result.remediation}")
        return 1

    print("Model resolution provenance written")
    print(f"  output_zarr: {result.output_zarr}")
    print(f"  detect_run: {result.detect_run}")
    print(f"  selected_model: {result.selected_model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

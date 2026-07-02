#!/usr/bin/env python3
"""Benchmark ROI-model inference across materialized and geometry-only crop modes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_image_source import resolve_materialized_crop_run
from fisheye.utils.resolve_detect_model import Candidate, _load_candidates, _load_target_profile, _resolve_recording_id
from fisheye.utils.zarr_recording_context import infer_recording_context


SCENARIO_MATERIALIZED = "materialized"
SCENARIO_GEOMETRY_LIVE = "geometry_live"
SCENARIO_GEOMETRY_CACHE_BUILD = "geometry_cache_build"
SCENARIO_GEOMETRY_CACHE_REUSE = "geometry_cache_reuse"
SCENARIO_CHOICES = (
    SCENARIO_MATERIALIZED,
    SCENARIO_GEOMETRY_LIVE,
    SCENARIO_GEOMETRY_CACHE_BUILD,
    SCENARIO_GEOMETRY_CACHE_REUSE,
)


@dataclass(frozen=True)
class CropRunSelection:
    materialized_run: Optional[str]
    geometry_run: Optional[str]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    crop_run: str
    roi_cache_policy: str
    roi_cache_dir: Optional[Path]
    description: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_registry_path(path: Optional[Path]) -> Path:
    return (path or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()


def _resolve_recording_dir(recording_dir: Optional[Path], zarr_path: Path) -> Path:
    if recording_dir is not None:
        return recording_dir.expanduser().resolve()
    return infer_recording_context(zarr_path).recording_dir


def _pick_best_candidate(candidates: list[Candidate], *, task: str, require_unique: bool) -> Candidate:
    if not candidates:
        raise RuntimeError(f"No {task} model candidates found.")
    best = candidates[0]
    if require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise RuntimeError(f"Top {task} candidate score tied; rerun with --require-unique disabled or set filters.")
    return best


def _resolve_model_from_registry(
    *,
    task: str,
    recording_dir: Path,
    registry_path: Path,
    set_id_filter: Optional[str],
    include_non_success: bool,
    require_unique: bool,
) -> Candidate:
    registry = Registry(registry_path)
    try:
        recording_id = _resolve_recording_id(registry, recording_id=None, recording_dir=recording_dir)
        target = _load_target_profile(registry, recording_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task=task,
            set_id_filter=set_id_filter,
            include_non_success=include_non_success,
        )
    finally:
        registry.close()
    return _pick_best_candidate(candidates, task=task, require_unique=require_unique)


def _infer_storage_mode(group: zarr.Group) -> str:
    explicit = _normalize_text(group.attrs.get("crop_storage_mode"))
    if explicit in {"materialized", "geometry_only"}:
        return explicit
    return "materialized" if "roi_images" in group else "geometry_only"


def _resolve_crop_runs(
    *,
    zarr_path: Path,
    materialized_crop_run: Optional[str],
    geometry_crop_run: Optional[str],
) -> CropRunSelection:
    root = zarr.open_group(str(zarr_path), mode="r")
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise RuntimeError(f"Zarr archive is missing crop_runs: {zarr_path}")

    selected_materialized: Optional[str] = None
    if materialized_crop_run is not None:
        _parent, _group, selected_materialized = resolve_materialized_crop_run(
            root,
            crop_run=materialized_crop_run,
            zarr_path=zarr_path,
        )
    else:
        try:
            _parent, _group, selected_materialized = resolve_materialized_crop_run(
                root,
                zarr_path=zarr_path,
            )
        except Exception:
            selected_materialized = None

    selected_geometry: Optional[str] = None
    if geometry_crop_run is not None:
        candidate = crop_parent.get(geometry_crop_run)
        if candidate is None:
            raise RuntimeError(f"Geometry crop run not found: {geometry_crop_run}")
        if _infer_storage_mode(candidate) != "geometry_only":
            raise RuntimeError(f"Requested geometry crop run is not geometry_only: {geometry_crop_run}")
        selected_geometry = geometry_crop_run
    else:
        latest_any = _normalize_text(crop_parent.attrs.get("latest_any"))
        if latest_any and latest_any in crop_parent and _infer_storage_mode(crop_parent[latest_any]) == "geometry_only":
            selected_geometry = latest_any
        else:
            for run_name in sorted(str(name) for name in crop_parent.keys())[::-1]:
                group = crop_parent.get(run_name)
                if isinstance(group, zarr.Group) and _infer_storage_mode(group) == "geometry_only":
                    selected_geometry = run_name
                    break

    return CropRunSelection(
        materialized_run=selected_materialized,
        geometry_run=selected_geometry,
    )


def _scenario_specs(
    *,
    crop_runs: CropRunSelection,
    scenario_names: Sequence[str],
    cache_root: Path,
) -> list[ScenarioSpec]:
    specs: list[ScenarioSpec] = []
    shared_geometry_cache_dir = cache_root / "geometry_shared_cache"

    for scenario_name in scenario_names:
        if scenario_name == SCENARIO_MATERIALIZED:
            if crop_runs.materialized_run is None:
                continue
            specs.append(
                ScenarioSpec(
                    name=scenario_name,
                    crop_run=crop_runs.materialized_run,
                    roi_cache_policy="never",
                    roi_cache_dir=None,
                    description="Materialized crop run baseline (no temporary ROI cache).",
                )
            )
            continue

        if crop_runs.geometry_run is None:
            continue

        if scenario_name == SCENARIO_GEOMETRY_LIVE:
            specs.append(
                ScenarioSpec(
                    name=scenario_name,
                    crop_run=crop_runs.geometry_run,
                    roi_cache_policy="never",
                    roi_cache_dir=None,
                    description="Geometry-only crop run with live ROI reads (no cache).",
                )
            )
        elif scenario_name == SCENARIO_GEOMETRY_CACHE_BUILD:
            specs.append(
                ScenarioSpec(
                    name=scenario_name,
                    crop_run=crop_runs.geometry_run,
                    roi_cache_policy="always",
                    roi_cache_dir=shared_geometry_cache_dir,
                    description="Geometry-only crop run with forced temporary ROI cache build.",
                )
            )
        elif scenario_name == SCENARIO_GEOMETRY_CACHE_REUSE:
            specs.append(
                ScenarioSpec(
                    name=scenario_name,
                    crop_run=crop_runs.geometry_run,
                    roi_cache_policy="always",
                    roi_cache_dir=shared_geometry_cache_dir,
                    description="Geometry-only crop run with warm temporary ROI cache reuse.",
                )
            )
        else:  # pragma: no cover - argparse guards choices
            raise ValueError(f"Unsupported scenario: {scenario_name}")
    return specs


def _build_keypoint_command(
    *,
    zarr_path: Path,
    model_path: Path,
    crop_run: str,
    run_name: str,
    roi_cache_policy: str,
    roi_cache_dir: Optional[Path],
    roi_live_acceleration: str,
    roi_live_gpu_chunk_frames: int,
    batch_size: int,
    device: Optional[str],
    imgsz: Optional[int],
    profile_timings: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.inference.predict_pose",
        "--model",
        str(model_path),
        "--zarr",
        str(zarr_path),
        "--crop-run",
        crop_run,
        "--run-name",
        run_name,
        "--batch-size",
        str(batch_size),
        "--roi-cache-policy",
        roi_cache_policy,
        "--roi-live-acceleration",
        roi_live_acceleration,
        "--roi-live-gpu-chunk-frames",
        str(roi_live_gpu_chunk_frames),
    ]
    if roi_cache_dir is not None:
        cmd.extend(["--roi-cache-dir", str(roi_cache_dir)])
    if device:
        cmd.extend(["--device", device])
    if imgsz is not None:
        cmd.extend(["--imgsz", str(imgsz)])
    if profile_timings:
        cmd.append("--profile-timings")
    return cmd


def _stage_attrs(zarr_path: Path, parent_name: str, run_name: str) -> dict[str, Any]:
    root = zarr.open_group(str(zarr_path), mode="r")
    parent = root.get(parent_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"Run not found after benchmark stage: {parent_name}/{run_name}")
    return dict(parent[run_name].attrs)


def _stage_metrics(
    *,
    stage: str,
    attrs: dict[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    total_rois = attrs.get("total_rois")
    duration_seconds = attrs.get("inference_duration_seconds") or attrs.get("duration_seconds")
    throughput = attrs.get("inference_poses_per_second")
    if throughput is None and total_rois is not None and duration_seconds:
        try:
            throughput = float(total_rois) / float(duration_seconds)
        except Exception:
            throughput = None
    return {
        "stage": stage,
        "wall_seconds": float(wall_seconds),
        "recorded_duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "throughput_per_second": float(throughput) if throughput is not None else None,
        "total_rois": int(total_rois) if total_rois is not None else None,
        "source_crop_run": _normalize_text(attrs.get("source_crop_run")),
        "source_crop_storage_mode": _normalize_text(attrs.get("source_crop_storage_mode")),
        "source_crop_signature": _normalize_text(attrs.get("source_crop_signature")),
        "source_roi_read_mode": _normalize_text(attrs.get("source_roi_read_mode")),
        "source_roi_cache_used": bool(attrs.get("source_roi_cache_used")),
        "source_roi_cache_path": _normalize_text(attrs.get("source_roi_cache_path")),
        "source_roi_cache_key": _normalize_text(attrs.get("source_roi_cache_key")),
        "roi_cache_policy": _normalize_text(attrs.get("roi_cache_policy")),
        "source_roi_live_acceleration_requested": _normalize_text(
            attrs.get("source_roi_live_acceleration_requested")
        ),
        "source_roi_live_acceleration_effective": _normalize_text(
            attrs.get("source_roi_live_acceleration_effective")
        ),
        "source_roi_live_acceleration_fallback_reason": _normalize_text(
            attrs.get("source_roi_live_acceleration_fallback_reason")
        ),
        "source_roi_live_gpu_chunk_frames": (
            int(attrs.get("source_roi_live_gpu_chunk_frames"))
            if attrs.get("source_roi_live_gpu_chunk_frames") is not None
            else None
        ),
        "timing_profile": attrs.get("timing_profile"),
    }


def _top_timing_stage(metrics: dict[str, Any]) -> Optional[str]:
    profile = metrics.get("timing_profile")
    if not isinstance(profile, dict):
        return None
    stages = profile.get("stages")
    if not isinstance(stages, dict) or not stages:
        return None
    try:
        stage_name, payload = max(
            stages.items(),
            key=lambda item: float(item[1].get("total_seconds", 0.0)) if isinstance(item[1], dict) else 0.0,
        )
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return f"{stage_name}={float(payload.get('total_seconds', 0.0)):.2f}s"


def _run_command(cmd: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(list(cmd), check=False)
    elapsed = time.perf_counter() - started
    return {
        "command": list(cmd),
        "returncode": int(completed.returncode),
        "wall_seconds": float(elapsed),
        "status": "ok" if completed.returncode == 0 else "failed",
    }


def _prepare_scenario_cache(
    *,
    scenario: ScenarioSpec,
    zarr_path: Path,
) -> dict[str, Any]:
    if scenario.roi_cache_dir is None:
        return {"prepared": False}

    if scenario.name == SCENARIO_GEOMETRY_CACHE_BUILD and scenario.roi_cache_dir.exists():
        shutil.rmtree(scenario.roi_cache_dir, ignore_errors=True)

    if scenario.name != SCENARIO_GEOMETRY_CACHE_REUSE:
        return {"prepared": False}

    root = zarr.open_group(str(zarr_path), mode="r")
    crop_source = CropImageSource.open(
        root,
        crop_run=scenario.crop_run,
        zarr_path=zarr_path,
        roi_cache_policy="always",
        roi_cache_dir=scenario.roi_cache_dir,
        console=None,
    )
    try:
        return {
            "prepared": True,
            "cache_created": bool(crop_source.roi_cache_created),
            "cache_used": bool(crop_source.roi_cache_used),
            "cache_path": crop_source.roi_cache_path,
            "cache_key": crop_source.roi_cache_key,
        }
    finally:
        crop_source.close()


def _default_output_json() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("runs/benchmarks") / f"roi_inference_cache_{timestamp}.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette Zarr archive to benchmark against.")
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Original recording directory for registry model resolution. Defaults to zarr-derived recording context.",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Optional registry sqlite path.")
    parser.add_argument("--pose-model", type=Path, default=None, help="Explicit pose model path (.pt/.onnx/.engine).")
    parser.add_argument("--pose-set-id", type=str, default=None, help="Optional pose model set filter.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success training runs in registry resolution.")
    parser.add_argument("--require-unique", action="store_true", help="Fail if top registry candidate scores tie.")
    parser.add_argument("--materialized-crop-run", type=str, default=None, help="Explicit materialized crop run name.")
    parser.add_argument("--geometry-crop-run", type=str, default=None, help="Explicit geometry-only crop run name.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIO_CHOICES,
        default=list(SCENARIO_CHOICES),
        help="Scenario order to benchmark.",
    )
    parser.add_argument("--keypoint-batch-size", type=int, default=256, help="Batch size for pose inference.")
    parser.add_argument("--device", type=str, default=None, help="Optional shared device override.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional pose imgsz override.")
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop scenarios (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Enable per-stage timing diagnostics inside the benchmarked inference stages.",
    )
    parser.add_argument("--benchmark-id", type=str, default=None, help="Optional stable prefix for run names and cache root.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    zarr_path = args.zarr_path.expanduser().resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    recording_dir = _resolve_recording_dir(args.recording_dir, zarr_path)
    registry_path = _resolve_registry_path(args.registry) if (args.registry or args.pose_model is None) else None

    pose_candidate = None
    pose_model_path = args.pose_model.expanduser().resolve() if args.pose_model is not None else None

    if pose_model_path is None:
        if registry_path is None:
            raise RuntimeError("Pose model resolution requires --registry or --pose-model.")
        pose_candidate = _resolve_model_from_registry(
            task="pose",
            recording_dir=recording_dir,
            registry_path=registry_path,
            set_id_filter=args.pose_set_id,
            include_non_success=bool(args.include_non_success),
            require_unique=bool(args.require_unique),
        )
        pose_model_path = Path(pose_candidate.model_path).expanduser().resolve()

    crop_runs = _resolve_crop_runs(
        zarr_path=zarr_path,
        materialized_crop_run=args.materialized_crop_run,
        geometry_crop_run=args.geometry_crop_run,
    )

    benchmark_id = args.benchmark_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cache_root = Path(tempfile.gettempdir()).resolve() / "palette_roi_cache_benchmarks" / benchmark_id
    specs = _scenario_specs(
        crop_runs=crop_runs,
        scenario_names=list(args.scenarios),
        cache_root=cache_root,
    )
    if not specs:
        raise RuntimeError("No runnable benchmark scenarios resolved from requested crop runs.")

    print("ROI inference cache benchmark fixture:")
    print(f"- zarr={zarr_path}")
    print(f"- recording_dir={recording_dir}")
    print(f"- materialized_crop_run={crop_runs.materialized_run or 'none'}")
    print(f"- geometry_crop_run={crop_runs.geometry_run or 'none'}")
    print(f"- pose_model={pose_model_path}")
    print(f"- scenarios={', '.join(spec.name for spec in specs)}")

    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"\nScenario: {spec.name}")
        print(f"- crop_run={spec.crop_run}")
        print(f"- roi_cache_policy={spec.roi_cache_policy}")
        if spec.roi_cache_dir is not None:
            print(f"- roi_cache_dir={spec.roi_cache_dir}")
        print(f"- description={spec.description}")

        scenario_result: dict[str, Any] = {
            "name": spec.name,
            "crop_run": spec.crop_run,
            "roi_cache_policy": spec.roi_cache_policy,
            "roi_cache_dir": str(spec.roi_cache_dir) if spec.roi_cache_dir is not None else None,
            "description": spec.description,
        }
        cache_prep = _prepare_scenario_cache(
            scenario=spec,
            zarr_path=zarr_path,
        )
        scenario_result["cache_prepare"] = cache_prep
        if cache_prep.get("prepared"):
            print(
                "  cache_prepare: "
                f"created={cache_prep.get('cache_created')} "
                f"cache_used={cache_prep.get('cache_used')}"
            )

        key_run_name = f"bench_{benchmark_id}_{spec.name}_keypoints"
        key_cmd = _build_keypoint_command(
            zarr_path=zarr_path,
            model_path=pose_model_path,
            crop_run=spec.crop_run,
            run_name=key_run_name,
            roi_cache_policy=spec.roi_cache_policy,
            roi_cache_dir=spec.roi_cache_dir,
            roi_live_acceleration=args.roi_live_acceleration,
            roi_live_gpu_chunk_frames=int(args.roi_live_gpu_chunk_frames),
            batch_size=int(args.keypoint_batch_size),
            device=args.device,
            imgsz=args.imgsz,
            profile_timings=bool(args.profile_timings),
        )
        key_exec = _run_command(key_cmd)
        scenario_result["keypoints"] = key_exec
        if key_exec["status"] == "ok":
            key_attrs = _stage_attrs(zarr_path, "keypoints_runs", key_run_name)
            scenario_result["keypoints"]["run_name"] = key_run_name
            scenario_result["keypoints"]["metrics"] = _stage_metrics(
                stage="keypoints",
                attrs=key_attrs,
                wall_seconds=float(key_exec["wall_seconds"]),
            )
            print(
                f"  keypoints: wall={key_exec['wall_seconds']:.1f}s "
                f"read_mode={scenario_result['keypoints']['metrics']['source_roi_read_mode']} "
                f"cache_used={scenario_result['keypoints']['metrics']['source_roi_cache_used']}"
            )
            live_accel = scenario_result["keypoints"]["metrics"].get("source_roi_live_acceleration_effective")
            if live_accel:
                print(f"    live_acceleration: {live_accel}")
            top_stage = _top_timing_stage(scenario_result["keypoints"]["metrics"])
            if top_stage:
                print(f"    timing_top: {top_stage}")
        else:
            print(f"  keypoints: failed (returncode={key_exec['returncode']})")
            results.append(scenario_result)
            continue

        results.append(scenario_result)

    payload = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "fixture": {
            "benchmark_id": benchmark_id,
            "zarr_path": str(zarr_path),
            "recording_dir": str(recording_dir),
            "registry_path": str(registry_path) if registry_path is not None else None,
            "pose_model_path": str(pose_model_path),
            "materialized_crop_run": crop_runs.materialized_run,
            "geometry_crop_run": crop_runs.geometry_run,
            "keypoint_batch_size": int(args.keypoint_batch_size),
            "device": args.device,
            "imgsz": args.imgsz,
            "roi_live_acceleration": str(args.roi_live_acceleration),
            "roi_live_gpu_chunk_frames": int(args.roi_live_gpu_chunk_frames),
            "profile_timings": bool(args.profile_timings),
        },
        "resolved_models": {
            "pose": None if pose_candidate is None else {
                "run_id": pose_candidate.run_id,
                "set_id": pose_candidate.set_id,
                "model_path": pose_candidate.model_path,
                "score": pose_candidate.weighted_score,
            },
        },
        "scenarios": results,
    }

    output_json = (args.output_json or _default_output_json()).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark JSON: {output_json}")


if __name__ == "__main__":  # pragma: no cover
    main()

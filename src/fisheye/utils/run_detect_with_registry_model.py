#!/usr/bin/env python3
"""Resolve a detect model from registry, run detect_yolo, and persist resolution provenance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import zarr

from fisheye.detection.detect_yolo import detect_yolo
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils.resolve_detect_model import Candidate, TargetProfile
from fisheye.utils.resolve_detect_model import _load_candidates, _load_target_profile, _resolve_recording_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    registry_path: Path,
    recording_id: str,
    target: TargetProfile,
    selected: Candidate,
    candidates: list[Candidate],
    top_k: int,
) -> dict[str, Any]:
    return {
        "mode": "registry",
        "task": "detect",
        "registry_path": str(registry_path),
        "recording_id": recording_id,
        "resolved_at_utc": _utc_now(),
        "target": asdict(target),
        "selected": {
            "run_id": selected.run_id,
            "set_id": selected.set_id,
            "model_path": selected.model_path,
            "score": selected.weighted_score,
            "created_utc": selected.created_utc,
            "status": selected.status,
            "dataset_count": selected.dataset_count,
            "feature_match_counts": selected.feature_match_counts,
            "feature_weights_used": selected.feature_weights_used,
        },
        "candidates": [
            {
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
            for item in candidates[: max(0, int(top_k))]
        ],
    }


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

    recording_dir = args.recording_dir.expanduser().resolve()
    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    video_path = _resolve_video(recording_dir, args.video)
    output_path = _resolve_output(recording_dir, args.output)

    registry = Registry(registry_path)
    try:
        recording_id = _resolve_recording_id(
            registry,
            recording_id=None,
            recording_dir=recording_dir,
        )
        target = _load_target_profile(registry, recording_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task="detect",
            set_id_filter=args.set_id,
            include_non_success=bool(args.include_non_success),
        )
    finally:
        registry.close()

    best = _pick_best_candidate(candidates, require_unique=bool(args.require_unique))
    payload = _resolution_payload(
        registry_path=registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(args.top_k),
    )

    if args.json or args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    run_name = detect_yolo(
        video_path=str(video_path),
        model_path=best.model_path,
        output_zarr=str(output_path),
        config_path=args.config,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        use_gpu=(False if args.cpu else None),
        write_raw_video_metadata=bool(args.write_raw_video_metadata),
        overwrite_raw_video_metadata=bool(args.overwrite_raw_video_metadata),
    )

    _write_model_resolution_provenance(
        zarr_path=output_path,
        run_name=run_name,
        payload=payload,
    )
    print("Model resolution provenance written")
    print(f"  output_zarr: {output_path}")
    print(f"  detect_run: {run_name}")
    print(f"  selected_model: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

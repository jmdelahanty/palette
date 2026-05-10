#!/usr/bin/env python3
"""Create or patch a video-only recording Zarr and stamp manual metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional

import zarr

from fisheye.registry.db import Registry, RegistryPaths


DEFAULT_RECORDING_TYPE = "behavior"
DEFAULT_RECORDING_SUBTYPE = "free"
DEFAULT_BEHAVIOR_MODE = "free"
DEFAULT_ARTIFACT_SCHEMA_ID = "video_only_v1"
EXPERIMENT_CONTEXT_STATUS = "absent"
EXPERIMENT_CONTEXT_SOURCE = "none"
EXPERIMENT_CONTEXT_STATUS_DETAIL = (
    "Video-only intake has no H5/protocol source; stimulus-dependent analyses are unavailable."
)


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "video_only_recording"


def _to_posix_relpath(path: Path, *, start: Path) -> str:
    relpath = os.path.relpath(str(path), str(start))
    return relpath.replace(os.sep, "/")


def _load_json_mapping(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _merge_fields(
    existing: dict[str, Any],
    updates: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            continue
        current = merged.get(key)
        if overwrite or current in (None, "", [], {}):
            merged[key] = value
    return merged


def _default_session_uuid(*, recording_dir: Path, video_path: Path) -> str:
    return _slugify(recording_dir.name or video_path.stem)


def _default_recording_name(*, recording_dir: Path, video_path: Path) -> str:
    return recording_dir.name or video_path.stem


def _default_zarr_path(
    *,
    recording_dir: Path,
    session_uuid: str,
    video_path: Path,
) -> Path:
    base = _slugify(session_uuid or recording_dir.name or video_path.stem)
    return recording_dir / "zarr" / f"{base}_training.zarr"


@dataclass(frozen=True)
class VideoOnlyRecordingMetadata:
    session_uuid: str
    recording_id: str
    recording_name: str
    session_start_iso8601_utc: Optional[str]
    recording_type: str
    recording_subtype: str
    behavior_mode: str
    artifact_schema_id: str
    dish_design: Optional[str]
    rig_id: Optional[str]
    arena_id: Optional[str]
    camera_id: Optional[str]
    canvas_name: Optional[str]
    protocol_name: Optional[str]
    genotype: Optional[str]
    dpf_at_acquisition: Optional[int]
    num_dishes: Optional[int]
    fish_per_dish: Optional[int]


def build_session_context(metadata: VideoOnlyRecordingMetadata) -> dict[str, Any]:
    context: dict[str, Any] = {
        "session_uuid": metadata.session_uuid,
        "recording_id": metadata.recording_id,
        "recording_name": metadata.recording_name,
        "recording_type": metadata.recording_type,
        "recording_subtype": metadata.recording_subtype,
        "behavior_mode": metadata.behavior_mode,
        "experiment_context_status": EXPERIMENT_CONTEXT_STATUS,
        "experiment_context_source": EXPERIMENT_CONTEXT_SOURCE,
        "stimulus_runs_available": False,
        "experiment_context_status_detail": EXPERIMENT_CONTEXT_STATUS_DETAIL,
    }
    if metadata.session_start_iso8601_utc:
        context["session_start_iso8601_utc"] = metadata.session_start_iso8601_utc
    if metadata.rig_id:
        context["rig_id"] = metadata.rig_id
    if metadata.arena_id:
        context["arena_id"] = metadata.arena_id
    if metadata.camera_id:
        context["camera_id"] = metadata.camera_id
    if metadata.canvas_name:
        context["canvas_name"] = metadata.canvas_name
    if metadata.protocol_name:
        context["protocol_name"] = metadata.protocol_name
        context["protocol_name_from_definition"] = metadata.protocol_name
    if metadata.genotype:
        context["genotype"] = metadata.genotype
    if metadata.dpf_at_acquisition is not None:
        context["dpf_at_acquisition"] = metadata.dpf_at_acquisition
    return context


def build_experiment_setup(metadata: VideoOnlyRecordingMetadata) -> Optional[dict[str, Any]]:
    num_dishes = metadata.num_dishes
    fish_per_dish = metadata.fish_per_dish
    if num_dishes is None and fish_per_dish is None:
        return None
    if num_dishes is None or fish_per_dish is None:
        raise ValueError("num_dishes and fish_per_dish must be provided together.")
    if num_dishes < 1:
        raise ValueError("num_dishes must be at least 1.")
    if fish_per_dish < 1:
        raise ValueError("fish_per_dish must be at least 1.")
    return {
        "num_dishes": int(num_dishes),
        "fish_per_dish": int(fish_per_dish),
        "total_expected_fish": int(num_dishes) * int(fish_per_dish),
        "setup_type": "single_dish" if int(num_dishes) == 1 else "multi_dish",
        "source": "video_only_intake_cli",
        "configured_at": datetime.now(timezone.utc).isoformat(),
    }


def build_manifest_payload(
    *,
    recording_dir: Path,
    video_path: Path,
    metadata: VideoOnlyRecordingMetadata,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_uuid": metadata.session_uuid,
        "recording_name": metadata.recording_name,
        "recording_type": metadata.recording_type,
        "recording_subtype": metadata.recording_subtype,
        "behavior_mode": metadata.behavior_mode,
        "artifact_schema_id": metadata.artifact_schema_id,
        "experiment_context_status": EXPERIMENT_CONTEXT_STATUS,
        "experiment_context_source": EXPERIMENT_CONTEXT_SOURCE,
        "stimulus_runs_available": False,
        "experiment_context_status_detail": EXPERIMENT_CONTEXT_STATUS_DETAIL,
        "files": {
            "cams": [_to_posix_relpath(video_path, start=recording_dir)],
        },
    }
    optional_fields = {
        "session_start_iso8601_utc": metadata.session_start_iso8601_utc,
        "dish_design": metadata.dish_design,
        "rig_id": metadata.rig_id,
        "arena_id": metadata.arena_id,
        "camera_id": metadata.camera_id,
        "canvas_name": metadata.canvas_name,
        "protocol_name_from_definition": metadata.protocol_name,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def write_manifest(
    *,
    recording_dir: Path,
    payload: dict[str, Any],
    overwrite: bool,
) -> Path:
    manifest_path = recording_dir / "recording_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Use --overwrite-manifest to replace it."
        )
    recording_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def apply_manual_metadata(
    *,
    zarr_path: Path,
    metadata: VideoOnlyRecordingMetadata,
    overwrite: bool,
) -> None:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="r+")
    analysis_meta = root.require_group("analysis_metadata")

    root_updates = {
        "session_uuid": metadata.session_uuid,
        "recording_id": metadata.recording_id,
        "recording_name": metadata.recording_name,
        "session_start_iso8601_utc": metadata.session_start_iso8601_utc,
        "recording_type": metadata.recording_type,
        "recording_subtype": metadata.recording_subtype,
        "behavior_mode": metadata.behavior_mode,
        "artifact_schema_id": metadata.artifact_schema_id,
        "dish_design": metadata.dish_design,
        "rig_id": metadata.rig_id,
        "arena_id": metadata.arena_id,
        "camera_id": metadata.camera_id,
        "canvas_name": metadata.canvas_name,
        "protocol_name": metadata.protocol_name,
        "protocol_name_from_definition": metadata.protocol_name,
        "experiment_context_status": EXPERIMENT_CONTEXT_STATUS,
        "experiment_context_source": EXPERIMENT_CONTEXT_SOURCE,
        "stimulus_runs_available": False,
        "experiment_context_status_detail": EXPERIMENT_CONTEXT_STATUS_DETAIL,
        "genotype": metadata.genotype,
        "dpf_at_acquisition": metadata.dpf_at_acquisition,
    }
    merged_root = _merge_fields(dict(root.attrs), root_updates, overwrite=overwrite)
    root.attrs.put(merged_root)

    existing_session_context = _load_json_mapping(analysis_meta.attrs.get("session_context"))
    merged_session_context = _merge_fields(
        existing_session_context,
        build_session_context(metadata),
        overwrite=overwrite,
    )
    analysis_meta.attrs["session_context"] = json.dumps(merged_session_context, sort_keys=True)
    analysis_meta.attrs["session_uuid"] = metadata.session_uuid

    experiment_setup = build_experiment_setup(metadata)
    if experiment_setup is not None:
        existing_setup = _load_json_mapping(root.attrs.get("experiment_setup"))
        merged_setup = _merge_fields(existing_setup, experiment_setup, overwrite=overwrite)
        root.attrs["experiment_setup"] = merged_setup


def _run_import_command(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"import_video failed with exit code {result.returncode}")


def run_video_import(
    *,
    video_path: Path,
    zarr_path: Path,
    config_path: Optional[Path],
    frame_step: int,
    overwrite_zarr: bool,
    cpu_only: bool,
    skip_tail_frames: int,
) -> None:
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "fisheye.capture.import_video",
        str(video_path),
        "--training-data",
        "--frame-step",
        str(frame_step),
        "--zarr-path",
        str(zarr_path),
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if overwrite_zarr:
        command.append("--overwrite")
    if cpu_only:
        command.append("--cpu-only")
    if skip_tail_frames:
        command.extend(["--skip-tail-frames", str(skip_tail_frames)])
    _run_import_command(command)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path, help="Source MP4 path.")
    parser.add_argument(
        "--recording-dir",
        type=Path,
        help="Logical recording directory. Defaults to the Zarr parent layout when possible.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        help="Output Zarr path. Defaults to <recording_dir>/zarr/<session_uuid>_training.zarr.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional import_video config path.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        help="Sample every Nth frame when importing video into a training Zarr.",
    )
    parser.add_argument(
        "--skip-tail-frames",
        type=int,
        default=0,
        help="Skip the last N frames during sampled import.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip MP4 import and only patch metadata on an existing Zarr.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU import.")
    parser.add_argument(
        "--overwrite-zarr",
        action="store_true",
        help="Overwrite an existing Zarr during MP4 import.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Overwrite existing manual metadata fields in the Zarr.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write recording_manifest.json into the recording directory.",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Overwrite an existing recording_manifest.json.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Scan the resulting Zarr into the registry after metadata stamping.",
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument(
        "--session-uuid",
        type=str,
        help="Stable session UUID / dataset identity. Defaults to a slug from recording_dir or video stem.",
    )
    parser.add_argument("--recording-id", type=str, help="Optional explicit recording_id.")
    parser.add_argument("--recording-name", type=str, help="Human-readable recording name.")
    parser.add_argument("--session-start-utc", type=str, help="Optional session start time in ISO 8601 UTC.")
    parser.add_argument("--recording-type", type=str, default=DEFAULT_RECORDING_TYPE)
    parser.add_argument("--recording-subtype", type=str, default=DEFAULT_RECORDING_SUBTYPE)
    parser.add_argument("--behavior-mode", type=str, default=DEFAULT_BEHAVIOR_MODE)
    parser.add_argument("--artifact-schema-id", type=str, default=DEFAULT_ARTIFACT_SCHEMA_ID)
    parser.add_argument("--dish-design", type=str)
    parser.add_argument("--rig-id", type=str)
    parser.add_argument("--arena-id", type=str)
    parser.add_argument("--camera-id", type=str)
    parser.add_argument("--canvas-name", type=str)
    parser.add_argument("--protocol-name", type=str)
    parser.add_argument("--genotype", type=str)
    parser.add_argument("--dpf-at-acquisition", type=int)
    parser.add_argument("--num-dishes", type=int)
    parser.add_argument("--fish-per-dish", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without writing.")
    return parser.parse_args(argv)


def _resolve_recording_dir(args: argparse.Namespace, *, zarr_path: Path, video_path: Path) -> Path:
    if args.recording_dir is not None:
        return args.recording_dir.expanduser().resolve()
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent.resolve()
    return video_path.parent.resolve()


def _build_metadata(
    args: argparse.Namespace,
    *,
    recording_dir: Path,
    video_path: Path,
) -> VideoOnlyRecordingMetadata:
    session_uuid = _normalize_text(args.session_uuid) or _default_session_uuid(
        recording_dir=recording_dir,
        video_path=video_path,
    )
    recording_id = _normalize_text(args.recording_id) or session_uuid
    recording_name = _normalize_text(args.recording_name) or _default_recording_name(
        recording_dir=recording_dir,
        video_path=video_path,
    )
    return VideoOnlyRecordingMetadata(
        session_uuid=session_uuid,
        recording_id=recording_id,
        recording_name=recording_name,
        session_start_iso8601_utc=_normalize_text(args.session_start_utc),
        recording_type=_normalize_text(args.recording_type) or DEFAULT_RECORDING_TYPE,
        recording_subtype=_normalize_text(args.recording_subtype) or DEFAULT_RECORDING_SUBTYPE,
        behavior_mode=_normalize_text(args.behavior_mode) or DEFAULT_BEHAVIOR_MODE,
        artifact_schema_id=_normalize_text(args.artifact_schema_id) or DEFAULT_ARTIFACT_SCHEMA_ID,
        dish_design=_normalize_text(args.dish_design),
        rig_id=_normalize_text(args.rig_id),
        arena_id=_normalize_text(args.arena_id),
        camera_id=_normalize_text(args.camera_id),
        canvas_name=_normalize_text(args.canvas_name),
        protocol_name=_normalize_text(args.protocol_name),
        genotype=_normalize_text(args.genotype),
        dpf_at_acquisition=_normalize_int(args.dpf_at_acquisition),
        num_dishes=_normalize_int(args.num_dishes),
        fish_per_dish=_normalize_int(args.fish_per_dish),
    )


def _print_plan(
    *,
    video_path: Path,
    recording_dir: Path,
    zarr_path: Path,
    metadata: VideoOnlyRecordingMetadata,
    write_manifest_flag: bool,
    register_flag: bool,
    metadata_only: bool,
) -> None:
    print("Video-only intake plan")
    print(f"  video: {video_path}")
    print(f"  recording_dir: {recording_dir}")
    print(f"  zarr_path: {zarr_path}")
    print(f"  metadata_only: {metadata_only}")
    print(f"  session_uuid: {metadata.session_uuid}")
    print(f"  recording_id: {metadata.recording_id}")
    print(f"  recording_name: {metadata.recording_name}")
    print(f"  artifact_schema_id: {metadata.artifact_schema_id}")
    print(f"  dish_design: {metadata.dish_design or '-'}")
    print(f"  rig_id: {metadata.rig_id or '-'}")
    print(f"  arena_id: {metadata.arena_id or '-'}")
    print(f"  camera_id: {metadata.camera_id or '-'}")
    print(f"  protocol_name: {metadata.protocol_name or '-'}")
    print(f"  manifest: {'write' if write_manifest_flag else 'skip'}")
    print(f"  register: {'yes' if register_flag else 'no'}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1
    if not args.metadata_only and args.frame_step is None:
        print("--frame-step is required unless --metadata-only is used.")
        return 1

    tentative_recording_dir = (
        args.recording_dir.expanduser().resolve() if args.recording_dir is not None else video_path.parent.resolve()
    )
    tentative_session_uuid = _normalize_text(args.session_uuid) or _default_session_uuid(
        recording_dir=tentative_recording_dir,
        video_path=video_path,
    )
    zarr_path = (
        args.zarr_path.expanduser().resolve()
        if args.zarr_path is not None
        else _default_zarr_path(
            recording_dir=tentative_recording_dir,
            session_uuid=tentative_session_uuid,
            video_path=video_path,
        ).resolve()
    )
    recording_dir = _resolve_recording_dir(args, zarr_path=zarr_path, video_path=video_path)
    metadata = _build_metadata(args, recording_dir=recording_dir, video_path=video_path)

    _print_plan(
        video_path=video_path,
        recording_dir=recording_dir,
        zarr_path=zarr_path,
        metadata=metadata,
        write_manifest_flag=bool(args.write_manifest),
        register_flag=bool(args.register),
        metadata_only=bool(args.metadata_only),
    )
    if args.dry_run:
        print("Dry run: no changes were made.")
        return 0

    if not args.metadata_only:
        run_video_import(
            video_path=video_path,
            zarr_path=zarr_path,
            config_path=args.config.expanduser().resolve() if args.config is not None else None,
            frame_step=int(args.frame_step),
            overwrite_zarr=bool(args.overwrite_zarr),
            cpu_only=bool(args.cpu_only),
            skip_tail_frames=int(args.skip_tail_frames or 0),
        )

    apply_manual_metadata(
        zarr_path=zarr_path,
        metadata=metadata,
        overwrite=bool(args.overwrite_metadata),
    )

    if args.write_manifest:
        payload = build_manifest_payload(
            recording_dir=recording_dir,
            video_path=video_path,
            metadata=metadata,
        )
        manifest_path = write_manifest(
            recording_dir=recording_dir,
            payload=payload,
            overwrite=bool(args.overwrite_manifest),
        )
        print(f"Wrote manifest: {manifest_path}")

    if args.register:
        registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        try:
            dataset_id = registry.scan_zarr(zarr_path)
        finally:
            registry.close()
        print(f"Registered dataset_id: {dataset_id}")

    print(f"Updated Zarr: {zarr_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

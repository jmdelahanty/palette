#!/usr/bin/env python3
"""
Plan a recording re-organization by mapping H5 files to camera artifacts.

This script inspects H5 root attributes to derive camera IDs and pairs each
recording with its Cam<id>.mp4 and Cam<id>_meta.csv. Use --dry-run to print
an ASCII tree of the proposed folder layout.
"""

import argparse
import json
import os
import re
import sys
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import h5py

try:
    from fisheye.utils.hevc_keyframe_flags import check_hevc_keyframe_flags
except ModuleNotFoundError:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from hevc_keyframe_flags import check_hevc_keyframe_flags


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: str, **fields: object) -> None:
        payload = {"event": event, "ts_utc": _utc_now(), "run_id": self.run_id}
        payload.update(fields)
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    dest_name: str


@dataclass
class RecordingPlan:
    name: str
    source_dir: Path
    dest_dir: Path
    raw_files: List[PlannedFile]
    cam_files: List[PlannedFile]
    derived_files: List[PlannedFile]
    camera_id: Optional[str]
    meta: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    keyframe_checks: Dict[str, Dict[str, object]] = field(default_factory=dict)


def _normalize_attr(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    match = re.search(r"cam_(\d+)", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", text)
    return digits[-1] if digits else None


def _sanitize_for_filename(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _read_camera_context(h5_path: Path) -> Tuple[Optional[str], Dict[str, str]]:
    meta: Dict[str, str] = {}
    try:
        with h5py.File(h5_path, "r") as h5:
            root = h5.attrs
            keys = (
                "session_uuid",
                "session_start_iso8601_utc",
                "rig_id",
                "arena_id",
                "camera_id",
                "canvas_name",
                "protocol_name_from_definition",
                "loaded_protocol_filepath",
                "stimulus_output_width",
                "stimulus_output_height",
                "ipc_source_name",
                "active_ipc_source",
                "hostname",
                "software_version",
            )
            for key in keys:
                if key in root:
                    value = _normalize_attr(root.get(key))
                    if value:
                        meta[key] = value
            camera_id = meta.get("camera_id")
            if not camera_id:
                derived = _derive_camera_id(meta.get("ipc_source_name"))
                if derived:
                    meta["camera_id"] = derived
                    meta["camera_id_source"] = "ipc_source_name"
                camera_id = derived
            return camera_id, meta
    except Exception as exc:
        meta["error"] = f"failed to read H5: {exc}"
        return None, meta


def _find_h5_files(source: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(source.rglob("*.h5"))
    return sorted(source.glob("*.h5"))


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _cam_file_candidates(camera_id: str, root: Path) -> Tuple[Path, Path]:
    return root / f"Cam{camera_id}.mp4", root / f"Cam{camera_id}_meta.csv"


def _extend_cam_roots(
    roots: List[Path],
    h5_path: Path,
    camera_id: str,
) -> List[Path]:
    cam_roots = _unique_paths(roots)
    has_cam = False
    for root in cam_roots:
        cam_mp4, cam_meta = _cam_file_candidates(camera_id, root)
        if cam_mp4.exists() or cam_meta.exists():
            has_cam = True
            break

    if has_cam:
        return cam_roots

    for ancestor in h5_path.parents:
        if ancestor in cam_roots:
            continue
        cam_mp4, cam_meta = _cam_file_candidates(camera_id, ancestor)
        if cam_mp4.exists() or cam_meta.exists():
            cam_roots.append(ancestor)
            break

    return _unique_paths(cam_roots)


def _unique_planned(files: List[PlannedFile]) -> List[PlannedFile]:
    seen: Set[Path] = set()
    unique: List[PlannedFile] = []
    for planned in files:
        if planned.source in seen:
            continue
        seen.add(planned.source)
        unique.append(planned)
    return unique


def _choose_session_tag(meta: Dict[str, str], h5_path: Path) -> str:
    return meta.get("session_uuid") or meta.get("session_start_iso8601_utc") or h5_path.stem


def _build_plan(
    h5_path: Path,
    dest_root: Path,
    cam_root: Optional[Path],
    rename_cams: bool,
) -> RecordingPlan:
    camera_id, meta = _read_camera_context(h5_path)
    name = h5_path.stem
    dest_dir = dest_root / name
    raw_files: List[PlannedFile] = [PlannedFile(h5_path, h5_path.name)]
    cam_files: List[PlannedFile] = []
    derived_files: List[PlannedFile] = []
    missing: List[str] = []

    rendered = h5_path.with_suffix(".mp4")
    if rendered.exists():
        raw_files.append(PlannedFile(rendered, rendered.name))
    else:
        missing.append(rendered.name)

    update_timing = h5_path.with_name(f"{h5_path.stem}_update_timing.csv")
    if update_timing.exists():
        raw_files.append(PlannedFile(update_timing, update_timing.name))

    if camera_id:
        search_roots = [h5_path.parent]
        if cam_root is not None:
            search_roots.append(cam_root)
        search_roots = _extend_cam_roots(search_roots, h5_path, str(camera_id))

        cam_mp4 = _first_existing(
            [root / f"Cam{camera_id}.mp4" for root in search_roots]
        )
        cam_meta = _first_existing(
            [root / f"Cam{camera_id}_meta.csv" for root in search_roots]
        )
        session_tag = _sanitize_for_filename(_choose_session_tag(meta, h5_path))
        cam_base = f"Cam{camera_id}_{session_tag}" if rename_cams else f"Cam{camera_id}"

        if cam_mp4:
            dest_name = f"{cam_base}.mp4"
            cam_files.append(PlannedFile(cam_mp4, dest_name))
        else:
            missing.append(f"Cam{camera_id}.mp4")
        if cam_meta:
            dest_name = f"{cam_base}_meta.csv"
            cam_files.append(PlannedFile(cam_meta, dest_name))
        else:
            missing.append(f"Cam{camera_id}_meta.csv")

        derived_patterns = [
            h5_path.parent / f"extracted_{camera_id}_*_image.png",
            h5_path.parent / f"extracted_{camera_id}_*.png",
        ]
        for pattern in derived_patterns:
            derived_files.extend(
                PlannedFile(path, path.name) for path in sorted(pattern.parent.glob(pattern.name))
            )
    else:
        missing.append("camera_id (missing in H5 attrs)")

    return RecordingPlan(
        name=name,
        source_dir=h5_path.parent,
        dest_dir=dest_dir,
        raw_files=raw_files,
        cam_files=cam_files,
        derived_files=_unique_planned(derived_files),
        camera_id=camera_id,
        meta=meta,
        missing=missing,
    )


def _format_recording_summary(plan: RecordingPlan) -> List[str]:
    lines = [f"Recording: {plan.name}"]
    camera_label = plan.camera_id or "unknown"
    lines.append(f"  camera_id: {camera_label}")
    if plan.cam_files:
        cam_names = ", ".join(
            f"{file.source.name} -> {file.dest_name}" if file.source.name != file.dest_name else file.source.name
            for file in plan.cam_files
        )
        lines.append(f"  cam files: {cam_names}")
    if plan.raw_files:
        raw_names = ", ".join(file.dest_name for file in plan.raw_files)
        lines.append(f"  raw files: {raw_names}")
    if plan.derived_files:
        derived_names = ", ".join(file.dest_name for file in plan.derived_files)
        lines.append(f"  derived files: {derived_names}")
    if plan.missing:
        missing = ", ".join(plan.missing)
        lines.append(f"  missing: {missing}")
    if "error" in plan.meta:
        lines.append(f"  error: {plan.meta['error']}")
    return lines


def _append_folder(lines: List[str], prefix: str, name: str, is_last: bool) -> str:
    connector = "`-- " if is_last else "|-- "
    lines.append(f"{prefix}{connector}{name}/")
    return prefix + ("    " if is_last else "|   ")


def _append_files(lines: List[str], prefix: str, files: List[PlannedFile]) -> None:
    for idx, planned in enumerate(files):
        is_last = idx == len(files) - 1
        connector = "`-- " if is_last else "|-- "
        lines.append(f"{prefix}{connector}{planned.dest_name}")


def _render_tree(root_label: str, plans: List[RecordingPlan]) -> List[str]:
    lines = [root_label]
    for idx, plan in enumerate(plans):
        is_last_plan = idx == len(plans) - 1
        child_prefix = _append_folder(lines, "", plan.name, is_last_plan)

        subfolders = [
            ("raw", plan.raw_files),
            ("cams", plan.cam_files),
            ("zarr", []),
            ("derived", plan.derived_files),
        ]
        for sub_idx, (folder_name, files) in enumerate(subfolders):
            is_last_sub = sub_idx == len(subfolders) - 1
            sub_prefix = _append_folder(lines, child_prefix, folder_name, is_last_sub)
            if files:
                _append_files(lines, sub_prefix, files)
    return lines


def _write_manifest(
    plan: RecordingPlan,
    organized_utc: str,
    run_id: str,
    log_path: Optional[Path],
) -> Optional[str]:
    manifest_path = plan.dest_dir / "recording_manifest.json"
    if manifest_path.exists():
        return f"Manifest exists, skipping: {manifest_path}"

    files = {
        "raw": [f"raw/{file.dest_name}" for file in plan.raw_files],
        "cams": [f"cams/{file.dest_name}" for file in plan.cam_files],
        "derived": [f"derived/{file.dest_name}" for file in plan.derived_files],
    }
    snapshot_path = plan.dest_dir / "derived" / "recording_snapshot.json"
    payload = {
        "recording_name": plan.name,
        "organized_utc": organized_utc,
        "organize_run_id": run_id,
        "organize_log": str(log_path) if log_path else None,
        "session_uuid": plan.meta.get("session_uuid"),
        "session_start_iso8601_utc": plan.meta.get("session_start_iso8601_utc"),
        "rig_id": plan.meta.get("rig_id"),
        "arena_id": plan.meta.get("arena_id"),
        "camera_id": plan.camera_id,
        "canvas_name": plan.meta.get("canvas_name"),
        "protocol_name_from_definition": plan.meta.get("protocol_name_from_definition"),
        "loaded_protocol_filepath": plan.meta.get("loaded_protocol_filepath"),
        "ipc_source_name": plan.meta.get("ipc_source_name"),
        "active_ipc_source": plan.meta.get("active_ipc_source"),
        "hostname": plan.meta.get("hostname"),
        "software_version": plan.meta.get("software_version"),
        "stimulus_output_width": plan.meta.get("stimulus_output_width"),
        "stimulus_output_height": plan.meta.get("stimulus_output_height"),
        "camera_id_source": plan.meta.get("camera_id_source"),
        "source_dir": str(plan.source_dir),
        "files": files,
        "hevc_keyframe_flags": plan.keyframe_checks if plan.keyframe_checks else None,
        "recording_snapshot": f"derived/{snapshot_path.name}" if snapshot_path.exists() else None,
    }
    try:
        plan.dest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Failed to write manifest for {plan.name}: {exc}"
    return None


def _build_snapshot_payload(
    snapshot: Dict[str, object],
    camera_id: Optional[str],
    mode: str,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if mode == "copy":
        return snapshot, None
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, dict):
        return None, "Snapshot missing cameras map; use --snapshot-mode copy."
    if not camera_id:
        return None, "Missing camera_id; cannot split snapshot."
    camera_payload = cameras.get(str(camera_id))
    if camera_payload is None:
        return None, f"Snapshot has no entry for camera_id {camera_id}."
    filtered = {k: v for k, v in snapshot.items() if k != "cameras"}
    filtered["cameras"] = {str(camera_id): camera_payload}
    return filtered, None


def _write_snapshot(
    plan: RecordingPlan,
    snapshot: Dict[str, object],
    mode: str,
) -> Optional[str]:
    payload, warning = _build_snapshot_payload(snapshot, plan.camera_id, mode)
    if warning:
        return warning
    if payload is None:
        return "Snapshot payload is empty."
    dest_dir = plan.dest_dir / "derived"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "recording_snapshot.json"
    if dest_path.exists():
        return f"Snapshot exists, skipping: {dest_path}"
    try:
        dest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return f"Failed to write snapshot for {plan.name}: {exc}"
    return None


def _apply_plan(
    plans: List[RecordingPlan],
    create_empty: bool,
    write_manifest: bool,
    snapshot: Optional[Dict[str, object]],
    snapshot_mode: str,
    logger: Optional[JsonLogger],
    run_id: str,
    log_path: Optional[Path],
) -> List[str]:
    warnings: List[str] = []
    moved: Set[Path] = set()
    planned_destinations: Set[Path] = set()

    def record_video_keyframe_status(
        *,
        plan: RecordingPlan,
        folder_name: str,
        planned: PlannedFile,
        dest: Path,
        session_uuid: Optional[str],
        logger: Optional[JsonLogger],
    ) -> None:
        rel_path = f"{folder_name}/{planned.dest_name}"
        result: Dict[str, object] = dict(check_hevc_keyframe_flags(dest))
        result["checked_at_utc"] = _utc_now()
        try:
            stat = dest.stat()
            result["file_size_bytes"] = int(stat.st_size)
            result["file_mtime_ns"] = int(stat.st_mtime_ns)
        except OSError as exc:
            result["fingerprint_error"] = str(exc)
        plan.keyframe_checks[rel_path] = result

        if logger:
            logger.log(
                "hevc_keyframe_check",
                recording_name=plan.name,
                session_uuid=session_uuid,
                path=str(dest),
                path_rel=rel_path,
                **result,
            )

    for plan in plans:
        moved_count = 0
        session_uuid = plan.meta.get("session_uuid")
        plan.keyframe_checks = {}

        def record_warning(message: str) -> None:
            warnings.append(message)
            if logger:
                logger.log(
                    "warning",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    message=message,
                )

        targets = [
            ("raw", plan.raw_files),
            ("cams", plan.cam_files),
            ("derived", plan.derived_files),
        ]
        for folder_name, files in targets:
            if not files and not create_empty:
                continue
            dest_dir = plan.dest_dir / folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for planned in files:
                src = planned.source
                if src in moved:
                    record_warning(f"Skipping duplicate source: {src}")
                    continue
                if not src.exists():
                    record_warning(f"Missing source: {src}")
                    continue
                dest = dest_dir / planned.dest_name
                if dest in planned_destinations:
                    record_warning(f"Duplicate destination planned, skipping: {dest}")
                    continue
                if dest.exists():
                    record_warning(f"Destination exists, skipping: {dest}")
                    continue
                print(f"Move: {src} -> {dest}")
                try:
                    shutil.move(str(src), str(dest))
                except Exception as exc:
                    record_warning(f"Failed to move {src} -> {dest}: {exc}")
                    continue
                moved.add(src)
                planned_destinations.add(dest)
                moved_count += 1
                if logger:
                    logger.log(
                        "file_moved",
                        recording_name=plan.name,
                        session_uuid=session_uuid,
                        source=str(src),
                        dest=str(dest),
                    )
                if dest.suffix.lower() == ".mp4":
                    record_video_keyframe_status(
                        plan=plan,
                        folder_name=folder_name,
                        planned=planned,
                        dest=dest,
                        session_uuid=session_uuid,
                        logger=logger,
                    )
                    check = plan.keyframe_checks.get(f"{folder_name}/{planned.dest_name}", {})
                    if bool(check.get("needs_fix", False)):
                        check_message = str(check.get("message", "")).strip()
                        record_warning(
                            f"HEVC keyframe flags issue for {dest}: "
                            f"{check_message or 'missing stss sync sample table'}"
                        )

        if snapshot is not None:
            warning = _write_snapshot(plan, snapshot, snapshot_mode)
            if warning:
                record_warning(warning)
            elif logger:
                logger.log(
                    "snapshot_written",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    dest=str(plan.dest_dir / "derived" / "recording_snapshot.json"),
                )

        if write_manifest:
            organized_utc = _utc_now()
            warning = _write_manifest(plan, organized_utc, run_id, log_path)
            if warning:
                record_warning(warning)
            elif logger:
                logger.log(
                    "manifest_written",
                    recording_name=plan.name,
                    session_uuid=session_uuid,
                    organized_utc=organized_utc,
                    dest=str(plan.dest_dir / "recording_manifest.json"),
                )

        if logger:
            logger.log(
                "recording_applied",
                recording_name=plan.name,
                session_uuid=session_uuid,
                camera_id=plan.camera_id,
                dest_dir=str(plan.dest_dir),
                moved_files=moved_count,
            )

    return warnings


def _cleanup_empty_dirs(root: Path) -> List[str]:
    warnings: List[str] = []
    if not root.exists() or not root.is_dir():
        return warnings

    candidates = [path for path in root.rglob("*") if path.is_dir()]
    candidates.append(root)
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    for path in candidates:
        try:
            if any(path.iterdir()):
                continue
            path.rmdir()
            print(f"Removed empty directory: {path}")
        except Exception as exc:
            warnings.append(f"Failed to remove {path}: {exc}")
    return warnings


def _cleanup_staging_dirs(
    root: Path,
    ignore_names: Set[str],
    logger: Optional[JsonLogger],
    batch_source: Optional[str],
) -> List[str]:
    warnings: List[str] = []
    if not root.exists() or not root.is_dir():
        return warnings

    candidates = [path for path in root.rglob("*") if path.is_dir()]
    candidates.append(root)
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    for path in candidates:
        try:
            entries = list(path.iterdir())
        except Exception as exc:
            warnings.append(f"Failed to inspect {path}: {exc}")
            continue

        removable = True
        for entry in entries:
            if entry.is_dir():
                if entry.exists():
                    removable = False
                continue
            if entry.name in ignore_names:
                try:
                    entry.unlink()
                    if logger:
                        logger.log(
                            "cleanup_removed",
                            batch_source=batch_source,
                            path=str(entry),
                        )
                except Exception as exc:
                    warnings.append(f"Failed to remove {entry}: {exc}")
                    removable = False
            else:
                removable = False

        if removable:
            try:
                path.rmdir()
                print(f"Removed empty directory: {path}")
                if logger:
                    logger.log(
                        "cleanup_removed",
                        batch_source=batch_source,
                        path=str(path),
                    )
            except Exception as exc:
                warnings.append(f"Failed to remove {path}: {exc}")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan recording re-organization by mapping H5 files to camera artifacts.",
        epilog=(
            "Environment variables:\n"
            "  PALETTE_STAGING_ROOT   default source root when no positional source is given\n"
            "  PALETTE_RECORDINGS_ROOT used by scripts/organize_staging.sh for --dest-root\n"
            "  PALETTE_LOG_ROOT       default log directory for JSONL logs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Directory containing H5 recordings "
            "(defaults to PALETTE_STAGING_ROOT or /nvme1/staging)."
        ),
    )
    default_dest_root = Path(os.environ.get("PALETTE_RECORDINGS_ROOT", "/nvme1/recordings"))
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=default_dest_root,
        help="Planned root directory for reorganized recordings (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to write JSONL logs (defaults to PALETTE_LOG_ROOT or <dest-root>/logs/organize_recordings).",
    )
    parser.add_argument(
        "--cam-root",
        type=Path,
        default=None,
        help="Optional root to search for Cam<id>.* files (defaults to source when --recursive).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for H5 files recursively under the source directory.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process every immediate subdirectory of the source (staging root).",
    )
    parser.add_argument(
        "--require-done",
        action="store_true",
        help="Only process batches that include the done marker file.",
    )
    parser.add_argument(
        "--done-name",
        default="TRANSFER_DONE",
        help="Marker filename that indicates transfer completion (default: TRANSFER_DONE).",
    )
    parser.add_argument(
        "--rename-cams",
        dest="rename_cams",
        action="store_true",
        default=True,
        help="Rename Cam files to include the session identifier (Cam<id>_<session>.*).",
    )
    parser.add_argument(
        "--no-rename-cams",
        dest="rename_cams",
        action="store_false",
        help="Keep original Cam file names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print an ASCII tree of the planned layout (no changes are made).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move files into the planned layout.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Optional recording_snapshot.json to attach to each recording.",
    )
    parser.add_argument(
        "--snapshot-mode",
        choices=("split", "copy"),
        default="split",
        help="How to attach the snapshot: split per camera or copy whole file.",
    )
    parser.add_argument(
        "--cleanup-empty",
        action="store_true",
        help="Remove empty directories under the source path after --apply.",
    )
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Remove staging batches that only contain marker/snapshot files after --apply.",
    )
    parser.add_argument(
        "--cleanup-ignore",
        action="append",
        default=[],
        help="Additional filenames to ignore during --cleanup-staging (can be repeated).",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write recording_manifest.json into each recording folder during --apply.",
    )

    args = parser.parse_args()

    source = args.source
    if source is None:
        source = Path(os.environ.get("PALETTE_STAGING_ROOT", "/nvme1/staging"))

    if not source.exists():
        print(f"Source path does not exist: {source}", file=sys.stderr)
        return 1

    if args.process_all:
        sources = sorted([path for path in source.iterdir() if path.is_dir()])
        if not sources:
            print(f"No subdirectories found under staging root: {source}")
            return 0
    else:
        sources = [source]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_id}_{os.getpid()}"
    log_dir: Optional[Path]
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        env_log = os.environ.get("PALETTE_LOG_ROOT")
        if env_log:
            log_dir = Path(env_log)
        else:
            log_dir = args.dest_root / "logs" / "organize_recordings"

    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"organize_recordings_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            source_root=str(source),
            process_all=args.process_all,
            recursive=args.recursive,
            dest_root=str(args.dest_root),
            rename_cams=args.rename_cams,
            write_manifest=args.write_manifest,
            snapshot_mode=args.snapshot_mode,
        )
    except Exception as exc:
        print(f"Warning: could not create log file: {exc}", file=sys.stderr)
        logger = None
        log_path = None

    for source_path in sources:
        if args.process_all:
            print(f"\n=== Processing {source_path} ===")
        if logger:
            logger.log("batch_start", batch_source=str(source_path))

        if args.require_done:
            marker = source_path / args.done_name
            if not marker.exists():
                print(f"Skipping {source_path}: missing marker {args.done_name}")
                if logger:
                    logger.log(
                        "batch_skipped",
                        batch_source=str(source_path),
                        reason=f"missing {args.done_name}",
                    )
                continue

        h5_files = _find_h5_files(source_path, args.recursive)
        if not h5_files:
            print(f"No .h5 files found in {source_path}.")
            if not args.recursive:
                print("Hint: use --recursive if H5 files live in subfolders.")
            continue

        cam_root = args.cam_root
        if cam_root is None and args.recursive:
            cam_root = source_path

        snapshot_payload: Optional[Dict[str, object]] = None
        snapshot_path: Optional[Path] = args.snapshot
        if snapshot_path is None:
            default_json = source_path / "recording_snapshot.json"
            default_plain = source_path / "recording_snapshot"
            if default_json.exists():
                snapshot_path = default_json
            elif default_plain.exists():
                snapshot_path = default_plain

        if snapshot_path is not None:
            if not snapshot_path.exists():
                print(f"Snapshot path does not exist: {snapshot_path}", file=sys.stderr)
            else:
                try:
                    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"Failed to read snapshot JSON: {exc}", file=sys.stderr)
                    snapshot_payload = None
                else:
                    if not isinstance(snapshot_payload, dict):
                        print("Snapshot JSON must be an object at the top level.", file=sys.stderr)
                        snapshot_payload = None

        plans = [
            _build_plan(h5_path, args.dest_root, cam_root, args.rename_cams)
            for h5_path in h5_files
        ]

        print(f"Found {len(plans)} H5 recording(s).")
        for plan in plans:
            for line in _format_recording_summary(plan):
                print(line)
            if logger:
                logger.log(
                    "recording_plan",
                    recording_name=plan.name,
                    session_uuid=plan.meta.get("session_uuid"),
                    camera_id=plan.camera_id,
                    missing=plan.missing,
                    raw_files=[file.dest_name for file in plan.raw_files],
                    cam_files=[file.dest_name for file in plan.cam_files],
                    derived_files=[file.dest_name for file in plan.derived_files],
                )

        if args.dry_run:
            print("\nPlanned layout (dry-run):")
            tree_lines = _render_tree(str(args.dest_root), plans)
            for line in tree_lines:
                print(line)

        if args.apply:
            print("\nApplying moves:")
            warnings = _apply_plan(
                plans,
                create_empty=False,
                write_manifest=args.write_manifest,
                snapshot=snapshot_payload,
                snapshot_mode=args.snapshot_mode,
                logger=logger,
                run_id=run_id,
                log_path=log_path,
            )
            if args.cleanup_empty:
                cleanup_warnings = _cleanup_empty_dirs(source_path)
                warnings.extend(cleanup_warnings)
                if logger:
                    for warning in cleanup_warnings:
                        logger.log("warning", batch_source=str(source_path), message=warning)
            if args.cleanup_staging:
                ignore_names = {"TRANSFER_DONE", "recording_snapshot.json", "recording_snapshot"}
                ignore_names.update(args.cleanup_ignore)
                cleanup_warnings = _cleanup_staging_dirs(
                    source_path,
                    ignore_names=ignore_names,
                    logger=logger,
                    batch_source=str(source_path),
                )
                warnings.extend(cleanup_warnings)
                if logger:
                    for warning in cleanup_warnings:
                        logger.log("warning", batch_source=str(source_path), message=warning)
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  - {warning}")

    if logger:
        logger.log("run_end")
        logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

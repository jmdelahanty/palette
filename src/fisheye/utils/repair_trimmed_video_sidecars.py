#!/usr/bin/env python3
"""Repair camera sidecars after an MP4 has been losslessly trimmed.

This is intended for video-only recordings where the MP4 was shortened with a
container copy operation, but the Orange camera metadata CSV and keyframe JSON
still describe the original longer acquisition.
"""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fisheye.diagnostics.video.batch import build_batch_report
from fisheye.diagnostics.video.camera_csv import expected_camera_csv_path
from fisheye.diagnostics.video.probe import inspect_stream
from fisheye.shared.batch_logging import make_run_id, utc_now
from fisheye.utils.recording_preflight import (
    PRECHECK_FAIL,
    PRECHECK_NOT_RUN,
    PRECHECK_PASS,
    PRECHECK_WARN,
    build_manifest_preflight_payload,
    build_video_preflight_payload,
)


@dataclass(frozen=True)
class RepairTarget:
    recording_dir: Path
    video_path: Path
    csv_path: Path
    keyframe_path: Path
    target_frames: int


@dataclass(frozen=True)
class CsvRepairResult:
    status: str
    original_rows: int
    repaired_rows: int
    backup_path: Optional[Path] = None


@dataclass(frozen=True)
class KeyframeRepairResult:
    status: str
    original_total_frames: Optional[int]
    repaired_total_frames: int
    original_keyframe_count: int
    repaired_keyframe_count: int
    backup_path: Optional[Path] = None


def _rel(recording_dir: Path, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return path.relative_to(recording_dir).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _discover_recording_dirs(paths: list[Path], *, name_prefix: Optional[str]) -> list[Path]:
    recordings: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        candidates: list[Path]
        if (path / "recording_manifest.json").exists():
            candidates = [path]
        elif path.is_dir():
            candidates = [child for child in sorted(path.iterdir()) if (child / "recording_manifest.json").exists()]
        else:
            candidates = []
        for candidate in candidates:
            if name_prefix and not candidate.name.startswith(name_prefix):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            recordings.append(candidate)
    return recordings


def _find_single_video(recording_dir: Path) -> Path:
    videos = sorted((recording_dir / "cams").glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"expected exactly one cams/*.mp4 under {recording_dir}, found {len(videos)}")
    return videos[0]


def _find_keyframe_json(recording_dir: Path, video_path: Path) -> Path:
    expected = recording_dir / "cams" / f"{video_path.stem}_keyframe.json"
    if expected.exists():
        return expected
    legacy_expected = recording_dir / "derived" / f"{video_path.stem}_keyframe.json"
    if legacy_expected.exists():
        return legacy_expected
    candidates = sorted((recording_dir / "cams").glob("*_keyframe.json"))
    if len(candidates) == 1:
        return candidates[0]
    candidates = sorted((recording_dir / "derived").glob("*_keyframe.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"expected one keyframe JSON under {recording_dir / 'cams'} or derived, found {len(candidates)}")


def _target_frames_from_video(video_path: Path, explicit_target: Optional[int]) -> int:
    if explicit_target is not None:
        if explicit_target <= 0:
            raise ValueError("--target-frames must be positive")
        return int(explicit_target)
    info, findings = inspect_stream(video_path)
    if info.nb_frames is None:
        codes = ", ".join(finding.code for finding in findings) if findings else "none"
        raise ValueError(f"ffprobe did not report nb_frames for {video_path}; findings={codes}")
    return int(info.nb_frames)


def _build_targets(args: argparse.Namespace) -> list[RepairTarget]:
    recordings = _discover_recording_dirs(args.recording_dirs, name_prefix=args.name_prefix)
    targets: list[RepairTarget] = []
    for recording_dir in recordings:
        video_path = _find_single_video(recording_dir)
        csv_path = expected_camera_csv_path(video_path)
        keyframe_path = _find_keyframe_json(recording_dir, video_path)
        target_frames = _target_frames_from_video(video_path, args.target_frames)
        targets.append(
            RepairTarget(
                recording_dir=recording_dir,
                video_path=video_path,
                csv_path=csv_path,
                keyframe_path=keyframe_path,
                target_frames=target_frames,
            )
        )
    return targets


def _backup_path(recording_dir: Path, source_path: Path, *, repair_id: str) -> Path:
    backup_dir = recording_dir / "derived" / "original_sidecars"
    suffix = source_path.suffix
    stem = source_path.name[: -len(suffix)] if suffix else source_path.name
    return backup_dir / f"{stem}.pre_trim_{repair_id}{suffix}"


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _repair_csv(
    target: RepairTarget,
    *,
    dry_run: bool,
    repair_id: str,
    overwrite_backup: bool,
) -> CsvRepairResult:
    if not target.csv_path.exists():
        raise FileNotFoundError(f"camera metadata CSV is missing: {target.csv_path}")
    original_rows = _count_csv_rows(target.csv_path)
    if original_rows < target.target_frames:
        raise ValueError(
            f"camera metadata CSV has fewer rows than video frames: "
            f"{target.csv_path} rows={original_rows} video_frames={target.target_frames}"
        )
    if original_rows == target.target_frames:
        return CsvRepairResult(status="unchanged", original_rows=original_rows, repaired_rows=original_rows)

    backup = _backup_path(target.recording_dir, target.csv_path, repair_id=repair_id)
    if backup.exists() and not overwrite_backup:
        raise FileExistsError(f"backup already exists: {backup}")
    if dry_run:
        return CsvRepairResult(
            status="trim_planned",
            original_rows=original_rows,
            repaired_rows=target.target_frames,
            backup_path=backup,
        )

    tmp_path = target.csv_path.with_name(f".{target.csv_path.name}.trim_tmp")
    try:
        with target.csv_path.open("r", encoding="utf-8", newline="") as src, tmp_path.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            header = src.readline()
            if not header:
                raise ValueError(f"camera metadata CSV is empty: {target.csv_path}")
            dst.write(header)
            copied = 0
            for line in src:
                if copied >= target.target_frames:
                    break
                dst.write(line)
                copied += 1
            if copied != target.target_frames:
                raise ValueError(
                    f"failed to copy requested metadata rows: copied={copied}, target={target.target_frames}"
                )
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and overwrite_backup:
            backup.unlink()
        shutil.move(str(target.csv_path), str(backup))
        shutil.move(str(tmp_path), str(target.csv_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return CsvRepairResult(
        status="trimmed",
        original_rows=original_rows,
        repaired_rows=target.target_frames,
        backup_path=backup,
    )


def _repair_keyframe_json(
    target: RepairTarget,
    *,
    dry_run: bool,
    repair_id: str,
    overwrite_backup: bool,
) -> KeyframeRepairResult:
    if not target.keyframe_path.exists():
        raise FileNotFoundError(f"keyframe JSON is missing: {target.keyframe_path}")
    payload = _load_json(target.keyframe_path)
    raw_keyframes = payload.get("keyframe_frames")
    if not isinstance(raw_keyframes, list):
        raise ValueError(f"keyframe JSON lacks keyframe_frames list: {target.keyframe_path}")
    keyframes = [int(frame) for frame in raw_keyframes]
    original_total = payload.get("total_frames")
    original_total_int = int(original_total) if original_total is not None else None
    repaired_keyframes = [frame for frame in keyframes if 0 <= frame < target.target_frames]
    unchanged = (
        original_total_int == target.target_frames
        and len(repaired_keyframes) == len(keyframes)
        and all(frame < target.target_frames for frame in keyframes)
    )
    if unchanged:
        return KeyframeRepairResult(
            status="unchanged",
            original_total_frames=original_total_int,
            repaired_total_frames=target.target_frames,
            original_keyframe_count=len(keyframes),
            repaired_keyframe_count=len(keyframes),
        )

    backup = _backup_path(target.recording_dir, target.keyframe_path, repair_id=repair_id)
    if backup.exists() and not overwrite_backup:
        raise FileExistsError(f"backup already exists: {backup}")
    if dry_run:
        return KeyframeRepairResult(
            status="trim_planned",
            original_total_frames=original_total_int,
            repaired_total_frames=target.target_frames,
            original_keyframe_count=len(keyframes),
            repaired_keyframe_count=len(repaired_keyframes),
            backup_path=backup,
        )

    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() and overwrite_backup:
        backup.unlink()
    shutil.copy2(target.keyframe_path, backup)
    payload["total_frames"] = int(target.target_frames)
    payload["keyframe_frames"] = repaired_keyframes
    payload["palette_trim_repair"] = {
        "tool": "fisheye.utils.repair_trimmed_video_sidecars",
        "repair_id": repair_id,
        "created_at_utc": utc_now(),
        "reason": "camera video was container-trimmed after acquisition",
        "original_total_frames": original_total_int,
        "repaired_total_frames": int(target.target_frames),
        "original_keyframe_count": len(keyframes),
        "repaired_keyframe_count": len(repaired_keyframes),
    }
    _write_json(target.keyframe_path, payload)
    return KeyframeRepairResult(
        status="trimmed",
        original_total_frames=original_total_int,
        repaired_total_frames=target.target_frames,
        original_keyframe_count=len(keyframes),
        repaired_keyframe_count=len(repaired_keyframes),
        backup_path=backup,
    )


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _patch_manifest(
    target: RepairTarget,
    csv_result: CsvRepairResult,
    keyframe_result: KeyframeRepairResult,
    *,
    dry_run: bool,
    repair_id: str,
    reason: str,
) -> str:
    manifest_path = target.recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return "manifest_missing"
    payload = _load_json(manifest_path)
    files = payload.get("files")
    if not isinstance(files, dict):
        files = {}
        payload["files"] = files
    derived_entries = files.get("derived")
    if not isinstance(derived_entries, list):
        derived_entries = []
        files["derived"] = derived_entries
    for backup in (csv_result.backup_path, keyframe_result.backup_path):
        rel_backup = _rel(target.recording_dir, backup)
        if rel_backup:
            _append_unique(derived_entries, rel_backup)

    repairs = payload.get("metadata_repairs")
    if not isinstance(repairs, list):
        repairs = []
        payload["metadata_repairs"] = repairs
    repair_payload = {
        "repair_type": "trimmed_video_sidecars_v1",
        "repair_id": repair_id,
        "created_at_utc": utc_now(),
        "tool": "fisheye.utils.repair_trimmed_video_sidecars",
        "reason": reason,
        "video": _rel(target.recording_dir, target.video_path),
        "video_frame_count": int(target.target_frames),
        "camera_metadata_csv": {
            "path": _rel(target.recording_dir, target.csv_path),
            "status": csv_result.status,
            "original_rows": int(csv_result.original_rows),
            "repaired_rows": int(csv_result.repaired_rows),
            "backup": _rel(target.recording_dir, csv_result.backup_path),
        },
        "keyframe_json": {
            "path": _rel(target.recording_dir, target.keyframe_path),
            "status": keyframe_result.status,
            "original_total_frames": keyframe_result.original_total_frames,
            "repaired_total_frames": int(keyframe_result.repaired_total_frames),
            "original_keyframe_count": int(keyframe_result.original_keyframe_count),
            "repaired_keyframe_count": int(keyframe_result.repaired_keyframe_count),
            "backup": _rel(target.recording_dir, keyframe_result.backup_path),
        },
    }
    _append_unique(repairs, repair_payload)

    if dry_run:
        return "manifest_patch_planned"
    _write_json(manifest_path, payload)
    return "manifest_patched"


def _diagnostic_finding_codes(findings: list[object], limit: int = 3) -> list[str]:
    codes: list[str] = []
    for finding in findings:
        code = getattr(finding, "code", None)
        if not code:
            continue
        text = str(code)
        if text not in codes:
            codes.append(text)
        if len(codes) >= limit:
            break
    return codes


def _persist_video_preflight(recording_dir: Path) -> str:
    return _persist_video_preflight_with_options(recording_dir, decode_backend="opencv")


def _persist_video_preflight_with_options(recording_dir: Path, *, decode_backend: str) -> str:
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return "preflight_manifest_missing"
    include_decode = decode_backend != "none"
    report = build_batch_report(
        [recording_dir],
        recursive=True,
        source="all",
        full_scan=False,
        sample_frames=120,
        decode_backend="all" if decode_backend == "none" else decode_backend,
        decode_frames=30,
        seek_samples=10,
        include_probe=True,
        include_timing=True,
        include_gop=True,
        include_decode=include_decode,
    )
    recording = next((item for item in report.recordings if item.recording_root == str(recording_dir)), None)
    media_status = str(recording.media_status if recording is not None else report.overall_status)
    tooling_status = str(recording.tooling_status if recording is not None else "skip")
    scanned = int(recording.item_count if recording is not None else report.summary.scanned)
    status = PRECHECK_PASS
    if scanned == 0:
        status = PRECHECK_WARN
        media_payload_status = PRECHECK_NOT_RUN
    else:
        media_payload_status = media_status
        if media_status == PRECHECK_FAIL:
            status = PRECHECK_FAIL
        elif media_status in {PRECHECK_WARN, "error"} or tooling_status in {PRECHECK_WARN, PRECHECK_FAIL, "error"}:
            status = PRECHECK_WARN
    finding_codes = _diagnostic_finding_codes([finding for item in report.items for finding in item.findings])
    video_payload = build_video_preflight_payload(
        status=status,
        media_status=media_payload_status,
        tooling_status=tooling_status,
        videos_scanned=scanned,
        finding_codes=finding_codes,
    )

    payload = _load_json(manifest_path)
    existing_preflight = payload.get("preflight")
    existing_h5 = None
    if isinstance(existing_preflight, dict) and isinstance(existing_preflight.get("h5"), dict):
        existing_h5 = existing_preflight.get("h5")
    payload["preflight"] = build_manifest_preflight_payload(
        checked_at_utc=utc_now(),
        video=video_payload,
        h5=existing_h5,
    )
    _write_json(manifest_path, payload)
    return f"preflight_{status}"


def _repair_target(
    target: RepairTarget,
    *,
    dry_run: bool,
    repair_id: str,
    reason: str,
    overwrite_backup: bool,
    run_video_preflight: bool,
    video_preflight_decode_backend: str,
) -> dict[str, object]:
    csv_result = _repair_csv(target, dry_run=dry_run, repair_id=repair_id, overwrite_backup=overwrite_backup)
    keyframe_result = _repair_keyframe_json(
        target,
        dry_run=dry_run,
        repair_id=repair_id,
        overwrite_backup=overwrite_backup,
    )
    manifest_status = _patch_manifest(
        target,
        csv_result,
        keyframe_result,
        dry_run=dry_run,
        repair_id=repair_id,
        reason=reason,
    )
    preflight_status = "preflight_not_requested"
    if run_video_preflight and not dry_run:
        preflight_status = _persist_video_preflight_with_options(
            target.recording_dir,
            decode_backend=video_preflight_decode_backend,
        )
    elif run_video_preflight:
        preflight_status = "preflight_planned"

    return {
        "recording": target.recording_dir.name,
        "video": target.video_path.name,
        "target_frames": target.target_frames,
        "csv_status": csv_result.status,
        "csv_original_rows": csv_result.original_rows,
        "csv_repaired_rows": csv_result.repaired_rows,
        "keyframe_status": keyframe_result.status,
        "keyframe_original_total_frames": keyframe_result.original_total_frames,
        "keyframe_repaired_total_frames": keyframe_result.repaired_total_frames,
        "keyframe_original_count": keyframe_result.original_keyframe_count,
        "keyframe_repaired_count": keyframe_result.repaired_keyframe_count,
        "manifest_status": manifest_status,
        "preflight_status": preflight_status,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording_dirs",
        nargs="+",
        type=Path,
        help="Recording directories, or a root containing recording directories.",
    )
    parser.add_argument("--apply", action="store_true", help="Write repaired sidecars and patch manifests.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned repair only.")
    parser.add_argument("--name-prefix", help="Only process recording directories with this name prefix.")
    parser.add_argument(
        "--target-frames",
        type=int,
        help="Use this frame count instead of probing each MP4. Normally omit this.",
    )
    parser.add_argument(
        "--repair-id",
        default=f"trimmed_video_sidecars_{make_run_id()}",
        help="Stable repair id used in backup filenames and manifest metadata.",
    )
    parser.add_argument(
        "--reason",
        default="camera video was trimmed with ffmpeg -c copy after acquisition",
        help="Human-readable reason recorded in the manifest.",
    )
    parser.add_argument("--overwrite-backup", action="store_true", help="Overwrite existing repair backup files.")
    parser.add_argument(
        "--run-video-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerun video diagnostics and refresh manifest preflight after apply.",
    )
    parser.add_argument(
        "--video-preflight-decode-backend",
        choices=["all", "opencv", "decord", "none"],
        default="opencv",
        help=(
            "Decode backend for the post-repair video preflight. Use 'none' for huge HEVC files "
            "when probe/timing/GOP/camera-CSV checks are enough."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.apply) == bool(args.dry_run):
        print("Specify exactly one of --dry-run or --apply.", file=sys.stderr)
        return 1
    try:
        targets = _build_targets(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not targets:
        print("No recording directories matched.")
        return 0

    summaries: list[dict[str, object]] = []
    exit_code = 0
    for target in targets:
        try:
            summary = _repair_target(
                target,
                dry_run=bool(args.dry_run),
                repair_id=str(args.repair_id),
                reason=str(args.reason),
                overwrite_backup=bool(args.overwrite_backup),
                run_video_preflight=bool(args.run_video_preflight),
                video_preflight_decode_backend=str(args.video_preflight_decode_backend),
            )
        except Exception as exc:
            summary = {"recording": target.recording_dir.name, "status": "error", "error": str(exc)}
            exit_code = 1
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))

    print("summary:")
    print(json.dumps({"exit_code": exit_code, "recordings": len(summaries)}, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

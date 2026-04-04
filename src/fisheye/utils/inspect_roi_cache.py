#!/usr/bin/env python3
"""Inspect and optionally prune temporary ROI cache stores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def _resolve_roi_cache_root(cache_root: Path | None) -> Path:
    if cache_root is not None:
        return cache_root.expanduser().resolve()

    env_root = os.environ.get("PALETTE_ROI_CACHE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(tempfile.gettempdir()).resolve() / "palette_roi_cache"


def _is_cache_store(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def _iter_cache_stores(cache_root: Path) -> Iterable[Path]:
    if not cache_root.is_dir():
        return []
    return sorted(path for path in cache_root.iterdir() if _is_cache_store(path))


def _load_root_attrs(path: Path) -> tuple[dict[str, Any], Optional[str]]:
    zarr_json = path / "zarr.json"
    if zarr_json.is_file():
        try:
            payload = json.loads(zarr_json.read_text())
        except Exception as exc:
            return {}, f"invalid zarr.json: {exc}"
        attrs = payload.get("attributes")
        if isinstance(attrs, dict):
            return attrs, None
        return {}, "zarr.json missing attributes object"

    zattrs = path / ".zattrs"
    if zattrs.is_file():
        try:
            payload = json.loads(zattrs.read_text())
        except Exception as exc:
            return {}, f"invalid .zattrs: {exc}"
        if isinstance(payload, dict):
            return payload, None
        return {}, ".zattrs did not decode to an object"

    return {}, "missing root metadata"


def _dir_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            file_count += 1
            total_bytes += int(stat.st_size)
    return file_count, total_bytes


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_roi_shape(value: object) -> Optional[tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    size = float(value)
    for unit in ("KB", "MB", "GB", "TB", "PB"):
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} EB"


def _format_age(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.0f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    if seconds < 86400.0:
        return f"{seconds / 3600.0:.1f}h"
    return f"{seconds / 86400.0:.1f}d"


@dataclass(frozen=True)
class RoiCacheEntry:
    path: str
    status: str
    cache_complete: bool
    cache_key: Optional[str]
    archive_path: Optional[str]
    crop_run_name: Optional[str]
    source_crop_storage_mode: Optional[str]
    frame_source_kind: Optional[str]
    frame_source_path: Optional[str]
    total_rois: Optional[int]
    roi_shape: Optional[tuple[int, int]]
    cache_layout_profile: Optional[str]
    cache_write_backend_effective: Optional[str]
    cache_acceleration: Optional[str]
    cache_fallback_reason: Optional[str]
    cache_roi_chunk_len: Optional[int]
    cache_roi_storage: Optional[str]
    cache_roi_use_sharding: Optional[bool]
    cache_gpu_chunk_frames: Optional[int]
    cache_duration_seconds: Optional[float]
    file_count: int
    total_bytes: int
    modified_at_utc: str
    age_seconds: float
    metadata_error: Optional[str] = None


def _build_entry(path: Path, *, now_ts: float) -> RoiCacheEntry:
    attrs, metadata_error = _load_root_attrs(path)
    try:
        modified_ts = float(path.stat().st_mtime)
    except OSError:
        modified_ts = now_ts
    file_count, total_bytes = _dir_stats(path)
    cache_complete = bool(attrs.get("cache_complete")) if metadata_error is None else False
    status = "complete" if cache_complete and metadata_error is None else "incomplete"
    if metadata_error is not None:
        status = "invalid"
    return RoiCacheEntry(
        path=str(path),
        status=status,
        cache_complete=cache_complete,
        cache_key=_normalize_text(attrs.get("cache_key")),
        archive_path=_normalize_text(attrs.get("archive_path")),
        crop_run_name=_normalize_text(attrs.get("crop_run_name")),
        source_crop_storage_mode=_normalize_text(attrs.get("source_crop_storage_mode")),
        frame_source_kind=_normalize_text(attrs.get("frame_source_kind")),
        frame_source_path=_normalize_text(attrs.get("frame_source_path")),
        total_rois=_normalize_int(attrs.get("total_rois")),
        roi_shape=_normalize_roi_shape(attrs.get("roi_shape")),
        cache_layout_profile=_normalize_text(attrs.get("cache_layout_profile")),
        cache_write_backend_effective=_normalize_text(attrs.get("cache_write_backend_effective")),
        cache_acceleration=_normalize_text(attrs.get("cache_acceleration")),
        cache_fallback_reason=_normalize_text(attrs.get("cache_fallback_reason")),
        cache_roi_chunk_len=_normalize_int(attrs.get("cache_roi_chunk_len")),
        cache_roi_storage=_normalize_text(attrs.get("cache_roi_storage")),
        cache_roi_use_sharding=(
            bool(attrs.get("cache_roi_use_sharding"))
            if attrs.get("cache_roi_use_sharding") is not None
            else None
        ),
        cache_gpu_chunk_frames=_normalize_int(attrs.get("cache_gpu_chunk_frames")),
        cache_duration_seconds=_normalize_float(attrs.get("cache_duration_seconds")),
        file_count=file_count,
        total_bytes=total_bytes,
        modified_at_utc=datetime.fromtimestamp(modified_ts, tz=timezone.utc).isoformat(),
        age_seconds=max(0.0, now_ts - modified_ts),
        metadata_error=metadata_error,
    )


@dataclass(frozen=True)
class RoiCachePlan:
    cache_root: str
    entries: list[RoiCacheEntry]
    delete_paths: list[str]


def _build_plan(
    cache_root: Path,
    *,
    delete_incomplete: bool,
    older_than_days: float | None,
) -> RoiCachePlan:
    now_ts = datetime.now(timezone.utc).timestamp()
    entries = [_build_entry(path, now_ts=now_ts) for path in _iter_cache_stores(cache_root)]
    delete_candidates: list[str] = []
    older_than_seconds = None if older_than_days is None else float(older_than_days) * 86400.0
    for entry in entries:
        should_delete = False
        if delete_incomplete and entry.status != "complete":
            should_delete = True
        if older_than_seconds is not None and entry.age_seconds >= older_than_seconds:
            should_delete = True
        if should_delete:
            delete_candidates.append(entry.path)
    return RoiCachePlan(
        cache_root=str(cache_root),
        entries=entries,
        delete_paths=sorted(delete_candidates),
    )


def _print_plan(plan: RoiCachePlan, *, apply: bool) -> None:
    total_bytes = sum(entry.total_bytes for entry in plan.entries)
    complete = sum(1 for entry in plan.entries if entry.status == "complete")
    incomplete = sum(1 for entry in plan.entries if entry.status == "incomplete")
    invalid = sum(1 for entry in plan.entries if entry.status == "invalid")
    print(f"ROI cache root: {plan.cache_root}")
    print(
        "Entries: total={total} complete={complete} incomplete={incomplete} invalid={invalid} bytes={bytes}".format(
            total=len(plan.entries),
            complete=complete,
            incomplete=incomplete,
            invalid=invalid,
            bytes=_format_bytes(total_bytes),
        )
    )
    for entry in sorted(plan.entries, key=lambda item: item.age_seconds):
        print(entry.path)
        print(
            "  status={status} age={age} size={size} files={files}".format(
                status=entry.status,
                age=_format_age(entry.age_seconds),
                size=_format_bytes(entry.total_bytes),
                files=entry.file_count,
            )
        )
        print(
            "  crop_run={crop_run} rois={rois} roi_shape={roi_shape} storage_mode={storage_mode}".format(
                crop_run=entry.crop_run_name or "-",
                rois=entry.total_rois if entry.total_rois is not None else "-",
                roi_shape=f"{entry.roi_shape[0]}x{entry.roi_shape[1]}" if entry.roi_shape else "-",
                storage_mode=entry.source_crop_storage_mode or "-",
            )
        )
        print(
            "  source={source} backend={backend} acceleration={accel} layout={layout} chunk_len={chunk_len}".format(
                source=entry.frame_source_kind or "-",
                backend=entry.cache_write_backend_effective or "-",
                accel=entry.cache_acceleration or "-",
                layout=entry.cache_layout_profile or "-",
                chunk_len=entry.cache_roi_chunk_len if entry.cache_roi_chunk_len is not None else "-",
            )
        )
        if entry.archive_path:
            print(f"  archive={entry.archive_path}")
        if entry.cache_key:
            print(f"  cache_key={entry.cache_key}")
        if entry.metadata_error:
            print(f"  metadata_error={entry.metadata_error}")
        if entry.path in plan.delete_paths:
            print("  delete=yes")
    if plan.delete_paths:
        mode = "Deleting" if apply else "Planned deletions"
        print(f"{mode}: {len(plan.delete_paths)}")
        for path in plan.delete_paths:
            print(f"  - {path}")
        if not apply:
            print("Use --apply to remove these cache stores.")


def _apply_plan(plan: RoiCachePlan) -> int:
    deleted = 0
    for path_text in plan.delete_paths:
        path = Path(path_text)
        if not path.exists():
            continue
        shutil.rmtree(path)
        deleted += 1
    return deleted


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally prune temporary Palette ROI caches.",
    )
    parser.add_argument(
        "cache_root",
        nargs="?",
        type=Path,
        help="Optional cache root to inspect. Defaults to --roi-cache-dir semantics: explicit path, then PALETTE_ROI_CACHE_ROOT, then /tmp/palette_roi_cache.",
    )
    parser.add_argument(
        "--delete-incomplete",
        action="store_true",
        help="Select incomplete/invalid cache stores for deletion.",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        help="Select cache stores older than the given age in days.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply selected deletions. Without this flag the tool only reports a dry-run plan.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary.",
    )
    args = parser.parse_args(argv)

    cache_root = _resolve_roi_cache_root(args.cache_root)
    plan = _build_plan(
        cache_root,
        delete_incomplete=bool(args.delete_incomplete),
        older_than_days=args.older_than_days,
    )

    if args.json:
        payload = {
            "cache_root": plan.cache_root,
            "entries": [asdict(entry) for entry in plan.entries],
            "delete_paths": plan.delete_paths,
            "apply": bool(args.apply),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(plan, apply=bool(args.apply))

    if not args.apply:
        return 0

    if not plan.delete_paths:
        return 0

    deleted = _apply_plan(plan)
    if not args.json:
        print("Summary:")
        print(f"  deleted={deleted}")
        print(f"  remaining={max(0, len(plan.entries) - deleted)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

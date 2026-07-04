#!/usr/bin/env python3
"""Batch write metadata-only raw_video attributes for recordings."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import zarr

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from .import_video_metadata import _probe_video, _write_metadata


@dataclass
class MetadataPlan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    video_path: Optional[Path] = None
    import_purpose: Optional[str] = None
    current_codec: Optional[str] = None
    current_pix_fmt: Optional[str] = None
    new_codec: Optional[str] = None
    new_pix_fmt: Optional[str] = None
    new_encoder: Optional[str] = None
    new_encoder_codec: Optional[str] = None
    new_format_title: Optional[str] = None
    new_format_encoder: Optional[str] = None
    new_format_comment: Optional[str] = None
    new_encoder_summary: Optional[str] = None


_utc_now = utc_now
JsonLogger = SharedJsonLogger


def _resolve_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "import_video_metadata_batch"
    if roots:
        return roots[0] / "logs" / "organize_recordings" / "import_video_metadata_batch"
    return Path.cwd() / "logs" / "organize_recordings" / "import_video_metadata_batch"


_run_id = make_run_id


def _load_file_list(path: Path) -> List[Path]:
    items: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            return None
    text = str(value).strip()
    return text or None


def _get_camera_id(root: zarr.Group) -> Optional[str]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        raw = analysis.attrs.get("camera_metadata")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", "ignore")
        if isinstance(raw, str):
            try:
                import json
                payload = json.loads(raw)
            except Exception:
                payload = None
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = None
        if isinstance(payload, dict):
            for key in ("device_serial_number", "camera_id", "device_snmp_comm_read"):
                if payload.get(key):
                    return str(payload.get(key))
    value = root.attrs.get("camera_id")
    return _normalize_text(value)


def _needs_update(raw: Optional[zarr.Group], *, overwrite: bool) -> bool:
    if overwrite:
        return True
    if raw is None:
        return True
    codec = _normalize_text(raw.attrs.get("video_codec"))
    pix_fmt = _normalize_text(raw.attrs.get("video_pix_fmt"))
    if not codec or codec.lower() == "unknown":
        return True
    if not pix_fmt or pix_fmt.lower() == "unknown":
        return True
    if not _normalize_text(raw.attrs.get("format_comment")):
        return True
    if not _normalize_text(raw.attrs.get("format_encoder")):
        return True
    if not _normalize_text(raw.attrs.get("encoder_name")):
        return True
    return False


def _current_codec_pixfmt(raw: Optional[zarr.Group]) -> Tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, None
    codec = _normalize_text(raw.attrs.get("video_codec"))
    pix_fmt = _normalize_text(raw.attrs.get("video_pix_fmt"))
    return codec, pix_fmt


def _encoder_summary(encoder_fields: object) -> Optional[str]:
    if not isinstance(encoder_fields, dict) or not encoder_fields:
        return None
    keys = [
        "encoder_name",
        "encoder_codec",
        "encoder_preset",
        "encoder_tuning",
        "encoder_rc",
        "encoder_bpp",
        "encoder_target_bps",
        "encoder_res",
        "encoder_fps",
    ]
    parts = []
    for key in keys:
        value = encoder_fields.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else None


def _infer_import_purpose(root: zarr.Group, arg_purpose: Optional[str]) -> str:
    if arg_purpose:
        return arg_purpose
    purpose = _normalize_text(root.attrs.get("zarr_purpose"))
    if purpose == "training":
        return "training_data"
    if purpose == "production":
        return "production"
    return "analysis"


def _find_video_path(
    root: zarr.Group,
    zarr_path: Path,
    *,
    video_dir_name: Optional[str],
) -> Tuple[Optional[Path], Optional[str]]:
    raw = root.get("raw_video")
    source_path = None
    for candidate in (
        raw.attrs.get("source_path") if raw is not None else None,
        root.attrs.get("source_video_path"),
        root.attrs.get("source_path"),
    ):
        text = _normalize_text(candidate)
        if text:
            source_path = Path(text)
            if source_path.exists():
                return source_path, None
    source_video = None
    for candidate in (
        raw.attrs.get("source_video") if raw is not None else None,
        root.attrs.get("source_video"),
    ):
        text = _normalize_text(candidate)
        if text:
            source_video = text
            break

    recording_dir = zarr_path.parent.parent if zarr_path.parent.name == "zarr" else zarr_path.parent
    video_dir = recording_dir / video_dir_name if video_dir_name else recording_dir
    if not video_dir.exists():
        return None, f"missing video dir: {video_dir}"

    mp4s = sorted(video_dir.glob("*.mp4"))
    if not mp4s:
        return None, "no .mp4 files found"
    if source_video:
        candidate = video_dir / source_video
        if candidate.exists():
            return candidate, None

    if len(mp4s) == 1:
        return mp4s[0], None

    camera_id = _get_camera_id(root)
    if camera_id:
        matches = [path for path in mp4s if f"Cam{camera_id}" in path.stem]
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"multiple mp4 matched camera_id {camera_id}"

    return None, "multiple .mp4 files and no unique match"


def _process_zarr(
    zarr_path: Path,
    *,
    apply: bool,
    overwrite: bool,
    video_dir_name: Optional[str],
    import_purpose: Optional[str],
) -> MetadataPlan:
    try:
        root = zarr.open_group(str(zarr_path), mode="r+")
    except Exception as exc:
        return MetadataPlan(zarr_path=zarr_path, status="failed", reason=f"open failed: {exc}")

    raw = root.get("raw_video")
    current_codec, current_pix_fmt = _current_codec_pixfmt(raw)
    if not _needs_update(raw, overwrite=overwrite):
        return MetadataPlan(zarr_path=zarr_path, status="skipped", reason="codec/pix_fmt already set")

    purpose = _infer_import_purpose(root, import_purpose)
    video_path, reason = _find_video_path(root, zarr_path, video_dir_name=video_dir_name)
    if video_path is None:
        return MetadataPlan(zarr_path=zarr_path, status="missing", reason=reason, import_purpose=purpose)

    if not apply:
        try:
            meta = _probe_video(video_path)
        except Exception as exc:
            return MetadataPlan(
                zarr_path=zarr_path,
                status="failed",
                reason=f"probe failed: {exc}",
                video_path=video_path,
                import_purpose=purpose,
                current_codec=current_codec,
                current_pix_fmt=current_pix_fmt,
            )
        encoder_fields = meta.get("encoder_fields") or {}
        encoder_name = encoder_fields.get("encoder_name") if isinstance(encoder_fields, dict) else None
        encoder_codec = encoder_fields.get("encoder_codec") if isinstance(encoder_fields, dict) else None
        summary = _encoder_summary(encoder_fields)
        format_tags = meta.get("format_tags") or {}
        return MetadataPlan(
            zarr_path=zarr_path,
            status="dry-run",
            reason="would update",
            video_path=video_path,
            import_purpose=purpose,
            current_codec=current_codec,
            current_pix_fmt=current_pix_fmt,
            new_codec=meta.get("codec"),
            new_pix_fmt=meta.get("pix_fmt"),
            new_encoder=encoder_name,
            new_encoder_codec=encoder_codec,
            new_format_title=format_tags.get("title") if isinstance(format_tags, dict) else None,
            new_format_encoder=format_tags.get("encoder") if isinstance(format_tags, dict) else None,
            new_format_comment=format_tags.get("comment") if isinstance(format_tags, dict) else None,
            new_encoder_summary=summary,
        )

    try:
        meta = _probe_video(video_path)
        encoder_fields = meta.get("encoder_fields") or {}
        encoder_name = encoder_fields.get("encoder_name") if isinstance(encoder_fields, dict) else None
        encoder_codec = encoder_fields.get("encoder_codec") if isinstance(encoder_fields, dict) else None
        summary = _encoder_summary(encoder_fields)
        format_tags = meta.get("format_tags") or {}
        _write_metadata(root, meta, overwrite=overwrite, import_purpose=purpose)
    except Exception as exc:
        return MetadataPlan(
            zarr_path=zarr_path,
            status="failed",
            reason=f"write failed: {exc}",
            video_path=video_path,
            import_purpose=purpose,
            current_codec=current_codec,
            current_pix_fmt=current_pix_fmt,
        )

    return MetadataPlan(
        zarr_path=zarr_path,
        status="ok",
        video_path=video_path,
        import_purpose=purpose,
        current_codec=current_codec,
        current_pix_fmt=current_pix_fmt,
        new_codec=meta.get("codec"),
        new_pix_fmt=meta.get("pix_fmt"),
        new_encoder=encoder_name,
        new_encoder_codec=encoder_codec,
        new_format_title=format_tags.get("title") if isinstance(format_tags, dict) else None,
        new_format_encoder=format_tags.get("encoder") if isinstance(format_tags, dict) else None,
        new_format_comment=format_tags.get("comment") if isinstance(format_tags, dict) else None,
        new_encoder_summary=summary,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Recordings root(s) or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Search recursively for zarrs.")
    parser.add_argument("--file-list", type=Path, help="Text file with zarr paths to process.")
    parser.add_argument("--video-dir", default="cams", help="Directory containing mp4s (default: cams).")
    parser.add_argument("--apply", action="store_true", help="Write metadata to zarrs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing attrs.")
    parser.add_argument("--import-purpose", default=None, help="Override import_purpose value.")
    parser.add_argument("--log-dir", type=Path, help="Optional log directory.")

    args = parser.parse_args(argv)

    roots = _resolve_roots(args.paths)
    if args.file_list:
        paths = _load_file_list(args.file_list)
    else:
        paths = list(_iter_zarr(roots, args.recursive))

    if not paths:
        print("No zarrs found.")
        return 1

    log_dir = _resolve_log_dir(args.log_dir, roots)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    log_path = log_dir / f"import_video_metadata_batch_{run_id}.jsonl"
    logger = JsonLogger(log_path, run_id)

    summary = {"ok": 0, "skipped": 0, "missing": 0, "failed": 0, "dry-run": 0}
    print(f"Log file: {log_path}")
    for zarr_path in paths:
        plan = _process_zarr(
            zarr_path,
            apply=args.apply,
            overwrite=args.overwrite,
            video_dir_name=args.video_dir,
            import_purpose=args.import_purpose,
        )
        summary[plan.status] = summary.get(plan.status, 0) + 1
        logger.log(
            "zarr",
            zarr=str(plan.zarr_path),
            status=plan.status,
            reason=plan.reason,
            video=str(plan.video_path) if plan.video_path else None,
            import_purpose=plan.import_purpose,
            current_codec=plan.current_codec,
            current_pix_fmt=plan.current_pix_fmt,
            new_codec=plan.new_codec,
            new_pix_fmt=plan.new_pix_fmt,
            new_encoder=plan.new_encoder,
            new_encoder_codec=plan.new_encoder_codec,
            new_format_title=plan.new_format_title,
            new_format_encoder=plan.new_format_encoder,
            new_format_comment=plan.new_format_comment,
            new_encoder_summary=plan.new_encoder_summary,
        )
        reason = f" ({plan.reason})" if plan.reason else ""
        print(f"{plan.status:8s} {plan.zarr_path}{reason}")
        if plan.status in {"dry-run", "ok"}:
            if plan.current_codec or plan.new_codec or plan.current_pix_fmt or plan.new_pix_fmt:
                print(
                    f"  codec: {plan.current_codec or '-'} -> {plan.new_codec or '-'}"
                    f" | pixfmt: {plan.current_pix_fmt or '-'} -> {plan.new_pix_fmt or '-'}"
                )
            if plan.new_encoder or plan.new_encoder_codec:
                print(
                    f"  encoder: {plan.new_encoder or '-'} | encoder_codec: {plan.new_encoder_codec or '-'}"
                )
            if plan.new_encoder_summary:
                print(f"  encoder_fields: {plan.new_encoder_summary}")
            if plan.new_format_title or plan.new_format_encoder or plan.new_format_comment:
                print(
                    f"  format_tags: title={plan.new_format_title or '-'}"
                    f" | encoder={plan.new_format_encoder or '-'}"
                    f" | comment={plan.new_format_comment or '-'}"
                )

    logger.close()
    print("\nSummary:")
    for key in ("ok", "dry-run", "skipped", "missing", "failed"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

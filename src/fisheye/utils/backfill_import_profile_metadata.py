"""Backfill singleton import-profile source metadata on existing Zarr stores.

This utility is intentionally narrow: it stamps missing root/raw_video attrs for
source-video stat fingerprints, source-H5 stat fingerprints, and encoded
source-video stream colorimetry. It does not repair historical unknown import
profiles or reinterpret pixel contracts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.import_profile_contract import SOURCE_H5_FINGERPRINT_ATTRS
from fisheye.shared.import_profile_contract import SOURCE_H5_PATH_ATTRS
from fisheye.shared.import_profile_contract import SOURCE_VIDEO_FINGERPRINT_ATTRS
from fisheye.shared.import_profile_contract import SOURCE_VIDEO_PATH_ATTRS
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import probe_video_colorimetry_attrs
from fisheye.shared.json_safety import write_json_atomic, write_jsonl_atomic
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs

VIDEO_COLORIMETRY_ATTRS = (
    "video_color_range",
    "video_color_space",
    "video_color_transfer",
    "video_color_primaries",
    "source_video_colorimetry_source",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _node_attrs_path(node_dir: Path) -> Path | None:
    """Return the Zarr v3 metadata path for a node.

    This backfill is intentionally scoped to current Palette Zarr v3 archives.
    Legacy Zarr v2 metadata (``.zattrs``/``.zgroup``) is skipped rather than
    silently mutated by a production metadata repair tool.
    """

    attrs_path = node_dir / "zarr.json"
    return attrs_path if attrs_path.is_file() else None


def _read_node_attrs(node_dir: Path) -> dict[str, Any]:
    attrs_path = _node_attrs_path(node_dir)
    if attrs_path is None:
        return {}
    payload = _read_json(attrs_path) or {}
    attrs = payload.get("attributes")
    return dict(attrs) if isinstance(attrs, Mapping) else {}


def _write_missing_node_attrs(node_dir: Path, values: Mapping[str, Any], *, apply: bool) -> dict[str, Any]:
    attrs_path = _node_attrs_path(node_dir)
    if attrs_path is None:
        return {"status": "skipped", "reason": "missing_zarr_json", "fields": []}

    payload = _read_json(attrs_path) or {}
    attrs = payload.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
        payload["attributes"] = attrs

    missing_values = {
        str(key): value
        for key, value in values.items()
        if value not in (None, "") and attrs.get(str(key)) in (None, "")
    }
    if not missing_values:
        return {"status": "skipped", "reason": "already_present", "fields": []}
    if not apply:
        return {"status": "planned", "reason": "dry_run", "fields": sorted(missing_values)}

    attrs.update(missing_values)
    write_json_atomic(attrs_path, payload)
    return {"status": "updated", "reason": "updated", "fields": sorted(missing_values)}


def _first_attr(attrs_list: Sequence[Mapping[str, Any]], names: Iterable[str]) -> Any | None:
    for attrs in attrs_list:
        for name in names:
            value = attrs.get(name)
            if value not in (None, ""):
                return value
    return None


def _has_any_attr(attrs_list: Sequence[Mapping[str, Any]], names: Iterable[str]) -> bool:
    return _first_attr(attrs_list, names) is not None


def _recording_dir_for_zarr(zarr_path: Path) -> Path:
    zarr_path = zarr_path.expanduser()
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent
    return zarr_path.parent


def _resolve_source_path(zarr_path: Path, value: Any, *, preferred_subdir: str | None = None) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path if path.exists() else path

    recording_dir = _recording_dir_for_zarr(zarr_path)
    candidates = [recording_dir / path]
    if preferred_subdir:
        candidates.append(recording_dir / preferred_subdir / path.name)
    candidates.extend(
        [
            recording_dir / "cams" / path.name,
            recording_dir / "raw" / path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_h5_path(zarr_path: Path, attrs_list: Sequence[Mapping[str, Any]]) -> tuple[Path | None, str]:
    value = _first_attr(attrs_list, SOURCE_H5_PATH_ATTRS)
    if value not in (None, ""):
        return _resolve_source_path(zarr_path, value, preferred_subdir="raw"), "attrs"

    raw_dir = _recording_dir_for_zarr(zarr_path) / "raw"
    try:
        h5_paths = sorted(path for path in raw_dir.glob("*.h5") if path.is_file())
    except OSError:
        h5_paths = []
    if len(h5_paths) == 1:
        return h5_paths[0], "single_raw_h5"
    if len(h5_paths) > 1:
        return None, "multiple_raw_h5"
    return None, "missing_source_h5_path"


def _source_video_extra(attrs_list: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for output_name, candidates in {
        "codec": ("video_codec", "codec", "codec_name"),
        "pix_fmt": ("video_pix_fmt", "pix_fmt", "pixel_format"),
        "width": ("source_video_width", "video_width", "width"),
        "height": ("source_video_height", "video_height", "height"),
        "fps": ("fps", "source_video_fps", "video_fps"),
        "frame_count": ("source_video_total_frames", "source_frame_count", "total_frames", "n_frames"),
    }.items():
        value = _first_attr(attrs_list, candidates)
        if value not in (None, ""):
            extra[output_name] = value
    return extra


def _action_row(
    *,
    zarr_path: Path,
    action: str,
    status: str,
    reason: str,
    root_result: Mapping[str, Any] | None = None,
    raw_result: Mapping[str, Any] | None = None,
    source_path: Path | None = None,
    source_resolution: str | None = None,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "import_profile_metadata_backfill_action",
        "zarr_path": str(zarr_path),
        "action": action,
        "status": status,
        "reason": reason,
        "source_path": str(source_path) if source_path is not None else None,
        "source_resolution": source_resolution,
        "root": dict(root_result or {}),
        "raw_video": dict(raw_result or {}),
        "values": dict(values or {}),
    }


def _combined_status(root_result: Mapping[str, Any], raw_result: Mapping[str, Any]) -> tuple[str, str]:
    statuses = {str(root_result.get("status")), str(raw_result.get("status"))}
    if "updated" in statuses:
        return "updated", "updated"
    if "planned" in statuses:
        return "planned", "dry_run"
    return "skipped", f"root={root_result.get('reason')};raw_video={raw_result.get('reason')}"


def backfill_zarr_import_profile_metadata(
    zarr_path: Path,
    *,
    apply: bool,
    include_existing_colorimetry: bool,
    skip_if_path_contains: Sequence[str] = (),
) -> list[dict[str, Any]]:
    zarr_path = zarr_path.expanduser()
    if any(token and token in str(zarr_path) for token in skip_if_path_contains):
        return [
            _action_row(
                zarr_path=zarr_path,
                action="skip_zarr",
                status="skipped",
                reason="path_excluded",
            )
        ]

    root_attrs = _read_node_attrs(zarr_path)
    raw_dir = zarr_path / "raw_video"
    raw_attrs = _read_node_attrs(raw_dir)
    attrs_list = (raw_attrs, root_attrs)
    actions: list[dict[str, Any]] = []

    source_video_value = _first_attr(attrs_list, SOURCE_VIDEO_PATH_ATTRS)
    source_video_path = _resolve_source_path(zarr_path, source_video_value, preferred_subdir="cams")
    source_video_exists = bool(source_video_path and source_video_path.exists())

    if not _has_any_attr(attrs_list, SOURCE_VIDEO_FINGERPRINT_ATTRS):
        if source_video_exists and source_video_path is not None:
            values = source_stat_fingerprint_attrs(
                source_video_path,
                attr_prefix="source_video",
                extra=_source_video_extra(attrs_list),
            )
            root_result = _write_missing_node_attrs(zarr_path, values, apply=apply)
            raw_result = _write_missing_node_attrs(raw_dir, values, apply=apply)
            status, reason = _combined_status(root_result, raw_result)
            actions.append(
                _action_row(
                    zarr_path=zarr_path,
                    action="source_video_stat_fingerprint",
                    status=status,
                    reason=reason,
                    root_result=root_result,
                    raw_result=raw_result,
                    source_path=source_video_path,
                    source_resolution="attrs",
                    values={"source_video_fingerprint": values.get("source_video_fingerprint")},
                )
            )
        else:
            actions.append(
                _action_row(
                    zarr_path=zarr_path,
                    action="source_video_stat_fingerprint",
                    status="skipped",
                    reason="source_video_missing",
                    source_path=source_video_path,
                )
            )

    has_video_colorimetry = _has_any_attr(attrs_list, VIDEO_COLORIMETRY_ATTRS)
    if include_existing_colorimetry or not has_video_colorimetry:
        if source_video_exists and source_video_path is not None:
            values = probe_video_colorimetry_attrs(source_video_path)
            if values:
                root_result = _write_missing_node_attrs(zarr_path, values, apply=apply)
                raw_result = _write_missing_node_attrs(raw_dir, values, apply=apply)
                status, reason = _combined_status(root_result, raw_result)
                actions.append(
                    _action_row(
                        zarr_path=zarr_path,
                        action="source_video_ffprobe_colorimetry",
                        status=status,
                        reason=reason,
                        root_result=root_result,
                        raw_result=raw_result,
                        source_path=source_video_path,
                        source_resolution="attrs",
                        values=values,
                    )
                )
            else:
                actions.append(
                    _action_row(
                        zarr_path=zarr_path,
                        action="source_video_ffprobe_colorimetry",
                        status="skipped",
                        reason="ffprobe_no_colorimetry",
                        source_path=source_video_path,
                    )
                )
        else:
            actions.append(
                _action_row(
                    zarr_path=zarr_path,
                    action="source_video_ffprobe_colorimetry",
                    status="skipped",
                    reason="source_video_missing",
                    source_path=source_video_path,
                )
            )

    if not _has_any_attr(attrs_list, SOURCE_H5_FINGERPRINT_ATTRS):
        h5_path, source_resolution = _resolve_h5_path(zarr_path, attrs_list)
        if h5_path is not None and h5_path.exists():
            values = source_stat_fingerprint_attrs(h5_path, attr_prefix="source_h5")
            root_values = {**values, "source_h5_path": str(h5_path), "source_h5": h5_path.name}
            raw_values = {**values, "source_h5_path": str(h5_path), "source_h5": h5_path.name}
            root_result = _write_missing_node_attrs(zarr_path, root_values, apply=apply)
            raw_result = _write_missing_node_attrs(raw_dir, raw_values, apply=apply)
            status, reason = _combined_status(root_result, raw_result)
            actions.append(
                _action_row(
                    zarr_path=zarr_path,
                    action="source_h5_stat_fingerprint",
                    status=status,
                    reason=reason,
                    root_result=root_result,
                    raw_result=raw_result,
                    source_path=h5_path,
                    source_resolution=source_resolution,
                    values={"source_h5_fingerprint": values.get("source_h5_fingerprint")},
                )
            )
        else:
            actions.append(
                _action_row(
                    zarr_path=zarr_path,
                    action="source_h5_stat_fingerprint",
                    status="skipped",
                    reason=source_resolution,
                    source_path=h5_path,
                    source_resolution=source_resolution,
                )
            )
    return actions


def _discover_paths(zarr_paths: Sequence[Path], recordings_roots: Sequence[Path]) -> list[Path]:
    paths = [path.expanduser() for path in zarr_paths]
    paths.extend(
        iter_filesystem_zarrs(
            (root.expanduser() for root in recordings_roots),
            recursive=False,
            pattern_policy="recording",
            include_zarr_files=False,
            require_zarr_root=True,
        )
    )
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_action = Counter(str(row.get("action")) for row in rows)
    by_status = Counter(str(row.get("status")) for row in rows)
    by_action_status = Counter(f"{row.get('action')}:{row.get('status')}" for row in rows)
    return {
        "schema_id": "palette.import_profile_metadata_backfill_summary.v1",
        "total_actions": len(rows),
        "action_counts": dict(sorted(by_action.items())),
        "status_counts": dict(sorted(by_status.items())),
        "action_status_counts": dict(sorted(by_action_status.items())),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="*", type=Path)
    parser.add_argument(
        "--recordings-root",
        action="append",
        type=Path,
        default=[],
        help="Bounded Palette recording-layout discovery; does not recurse through full recording trees.",
    )
    parser.add_argument(
        "--exclude-path-contains",
        action="append",
        default=[],
        help="Skip any Zarr path containing this text. Repeatable.",
    )
    parser.add_argument(
        "--include-existing-colorimetry",
        action="store_true",
        help="Probe/write missing video_color_* subfields even when some video_color_* attrs already exist.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write missing attrs. Omit for dry-run.")
    parser.add_argument("--output-jsonl", type=Path, help="Write action rows to this JSONL path. Defaults to stdout.")
    parser.add_argument("--summary-json", type=Path, help="Write aggregate action summary JSON.")
    args = parser.parse_args(argv)
    if not args.zarr_path and not args.recordings_root:
        parser.error("provide at least one zarr_path or --recordings-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    for zarr_path in _discover_paths(args.zarr_path, args.recordings_root):
        rows.extend(
            backfill_zarr_import_profile_metadata(
                zarr_path,
                apply=bool(args.apply),
                include_existing_colorimetry=bool(args.include_existing_colorimetry),
                skip_if_path_contains=tuple(str(item) for item in args.exclude_path_contains),
            )
        )

    if args.output_jsonl is None:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        write_jsonl_atomic(args.output_jsonl, rows)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.summary_json, _summary(rows))
    if args.output_jsonl is not None:
        print(json.dumps(_summary(rows), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

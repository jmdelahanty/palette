#!/usr/bin/env python3
"""Audit swim-bladder tuning metadata across training/analysis zarr archives."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from fisheye.utils.apply_tuning_by_camera import (
    SOURCE_SPECIFIC_SUBJECT_TUNING_CONTEXT_KEYS,
    _normalize_subject_mask_tuning_payload,
)


DEFAULT_RECORDINGS_ROOT = Path("/nvme1/recordings")
SWIM_COMPONENT = "swim_bladder"


@dataclass(frozen=True)
class AuditRow:
    zarr_path: Path
    camera_id: Optional[str]
    zarr_use: Optional[str]
    status: str
    method: Optional[str] = None
    tuned_timestamp: Optional[str] = None
    propagated_by_camera: bool = False
    propagated_component: Optional[str] = None
    context_keys: tuple[str, ...] = ()
    context_crop_run: Optional[str] = None
    context_keypoint_run: Optional[str] = None
    context_keypoint_source: Optional[str] = None
    local_crop_exists: bool = False
    local_keypoint_exists: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class CameraSummary:
    camera_id: str
    total_rows: int
    source_like: int
    clean_propagated: int
    stale_source_context: int
    missing_swim_tuning: int
    methods: tuple[str, ...]
    tuned_timestamps: tuple[str, ...]


def _resolve_roots(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [DEFAULT_RECORDINGS_ROOT]


def _infer_zarr_use(zarr_path: Path) -> Optional[str]:
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{path} did not contain a JSON object.")
    return payload


def _analysis_attrs(zarr_path: Path) -> Mapping[str, Any]:
    analysis_json = zarr_path / "analysis_metadata" / "zarr.json"
    payload = _read_json(analysis_json)
    attrs = payload.get("attributes")
    if not isinstance(attrs, Mapping):
        raise RuntimeError(f"{analysis_json} is missing attributes.")
    return attrs


def _parse_camera_metadata(raw: object) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _camera_id_from_analysis_attrs(attrs: Mapping[str, Any]) -> Optional[str]:
    camera_metadata = _parse_camera_metadata(attrs.get("camera_metadata"))
    for key in ("device_serial_number", "camera_id", "device_snmp_comm_read"):
        value = camera_metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalize_context(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}


def _local_keypoint_exists(zarr_path: Path, run_name: Optional[str], group_name: Optional[str]) -> bool:
    if not run_name:
        return False
    candidate_groups = [group_name] if group_name else []
    for default_group in ("refined_keypoints_runs", "keypoints_runs"):
        if default_group not in candidate_groups:
            candidate_groups.append(default_group)
    for candidate_group in candidate_groups:
        if not candidate_group:
            continue
        if (zarr_path / candidate_group / run_name).exists():
            return True
    return False


def _scan_zarr_path(
    zarr_path: Path,
    *,
    zarr_use_filter: str,
    camera_ids: set[str],
) -> AuditRow:
    zarr_use = _infer_zarr_use(zarr_path)
    if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=None,
            zarr_use=zarr_use,
            status="filtered_zarr_use",
            reason=f"wanted={zarr_use_filter} found={zarr_use or 'unknown'}",
        )

    try:
        attrs = _analysis_attrs(zarr_path)
    except FileNotFoundError:
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=None,
            zarr_use=zarr_use,
            status="missing_analysis_metadata",
        )
    except Exception as exc:
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=None,
            zarr_use=zarr_use,
            status="error",
            reason=str(exc),
        )

    camera_id = _camera_id_from_analysis_attrs(attrs)
    if camera_ids and camera_id not in camera_ids:
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=camera_id,
            zarr_use=zarr_use,
            status="filtered_camera",
        )

    subject_tuning = attrs.get("subject_mask_tuning")
    if not isinstance(subject_tuning, Mapping):
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=camera_id,
            zarr_use=zarr_use,
            status="missing_swim_tuning",
        )

    payload = _normalize_subject_mask_tuning_payload(subject_tuning)
    components = payload.get("components", {})
    if not isinstance(components, Mapping):
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=camera_id,
            zarr_use=zarr_use,
            status="missing_swim_tuning",
        )
    entry = components.get(SWIM_COMPONENT)
    if not isinstance(entry, Mapping):
        return AuditRow(
            zarr_path=zarr_path,
            camera_id=camera_id,
            zarr_use=zarr_use,
            status="missing_swim_tuning",
        )

    context = _normalize_context(entry.get("context"))
    context_keys = tuple(sorted(key for key in context if key in SOURCE_SPECIFIC_SUBJECT_TUNING_CONTEXT_KEYS))
    crop_run = str(context.get("crop_run")).strip() if context.get("crop_run") is not None else None
    keypoint_run = str(context.get("keypoint_run")).strip() if context.get("keypoint_run") is not None else None
    keypoint_source = (
        str(context.get("keypoint_source")).strip() if context.get("keypoint_source") is not None else None
    )
    crop_exists = bool(crop_run) and (zarr_path / "crop_runs" / str(crop_run)).exists()
    keypoint_exists = _local_keypoint_exists(zarr_path, keypoint_run, keypoint_source)
    propagated_by_camera = bool(entry.get("propagated_by_camera"))
    propagated_component = (
        str(entry.get("propagated_component")).strip()
        if entry.get("propagated_component") is not None
        else None
    )

    if propagated_by_camera and not context_keys:
        status = "clean_propagated"
    elif context_keys and crop_exists and keypoint_exists:
        status = "source_like"
    elif context_keys:
        status = "stale_source_context"
    else:
        status = "clean_unmarked"

    return AuditRow(
        zarr_path=zarr_path,
        camera_id=camera_id,
        zarr_use=zarr_use,
        status=status,
        method=str(entry.get("method")).strip() if entry.get("method") is not None else None,
        tuned_timestamp=str(entry.get("tuned_timestamp")).strip() if entry.get("tuned_timestamp") is not None else None,
        propagated_by_camera=propagated_by_camera,
        propagated_component=propagated_component,
        context_keys=context_keys,
        context_crop_run=crop_run,
        context_keypoint_run=keypoint_run,
        context_keypoint_source=keypoint_source,
        local_crop_exists=bool(crop_exists),
        local_keypoint_exists=bool(keypoint_exists),
    )


def _summarize_by_camera(rows: Sequence[AuditRow]) -> list[CameraSummary]:
    by_camera: dict[str, list[AuditRow]] = {}
    for row in rows:
        if row.camera_id is None:
            continue
        if row.status in {"filtered_camera", "filtered_zarr_use"}:
            continue
        by_camera.setdefault(row.camera_id, []).append(row)

    summaries: list[CameraSummary] = []
    for camera_id in sorted(by_camera):
        camera_rows = by_camera[camera_id]
        methods = tuple(sorted({row.method for row in camera_rows if row.method}))
        tuned_timestamps = tuple(sorted({row.tuned_timestamp for row in camera_rows if row.tuned_timestamp}))
        summaries.append(
            CameraSummary(
                camera_id=camera_id,
                total_rows=len(camera_rows),
                source_like=sum(1 for row in camera_rows if row.status == "source_like"),
                clean_propagated=sum(1 for row in camera_rows if row.status == "clean_propagated"),
                stale_source_context=sum(1 for row in camera_rows if row.status == "stale_source_context"),
                missing_swim_tuning=sum(1 for row in camera_rows if row.status == "missing_swim_tuning"),
                methods=methods,
                tuned_timestamps=tuned_timestamps,
            )
        )
    return summaries


def _print_row(row: AuditRow) -> None:
    print(f"{row.status.upper()} {row.zarr_path}")
    print(f"  camera_id: {row.camera_id or 'unknown'}")
    print(f"  use: {row.zarr_use or 'unknown'}")
    print(f"  method: {row.method or '—'}")
    print(f"  tuned_timestamp: {row.tuned_timestamp or '—'}")
    print(f"  propagated_by_camera: {row.propagated_by_camera}")
    if row.propagated_component:
        print(f"  propagated_component: {row.propagated_component}")
    if row.context_keys:
        print(f"  context_keys: {', '.join(row.context_keys)}")
    if row.context_crop_run:
        print(f"  context_crop_run: {row.context_crop_run} (local={row.local_crop_exists})")
    if row.context_keypoint_run:
        print(
            "  context_keypoint_run: "
            f"{row.context_keypoint_source or '?'}:{row.context_keypoint_run} "
            f"(local={row.local_keypoint_exists})"
        )
    if row.reason:
        print(f"  reason: {row.reason}")


def _print_summary(summaries: Sequence[CameraSummary]) -> None:
    print("Summary")
    for summary in summaries:
        print(
            f"  camera_id={summary.camera_id} total={summary.total_rows} "
            f"source_like={summary.source_like} clean_propagated={summary.clean_propagated} "
            f"stale_source_context={summary.stale_source_context} "
            f"missing_swim_tuning={summary.missing_swim_tuning}"
        )
        print(
            f"    methods={list(summary.methods) if summary.methods else []} "
            f"tuned_timestamps={list(summary.tuned_timestamps) if summary.tuned_timestamps else []}"
        )


def _has_strict_failures(summaries: Sequence[CameraSummary], rows: Sequence[AuditRow]) -> bool:
    for summary in summaries:
        if summary.stale_source_context > 0:
            return True
        if summary.missing_swim_tuning > 0:
            return True
        if summary.source_like != 1:
            return True
        if summary.methods and len(summary.methods) != 1:
            return True
        if summary.tuned_timestamps and len(summary.tuned_timestamps) != 1:
            return True
    return any(row.status == "error" for row in rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit swim-bladder subject_mask_tuning metadata and distinguish likely hand-tuned "
            "source zarrs from cleaned propagated targets."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan roots for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=("analysis", "training", "any"),
        default="training",
        help="Filter zarr archives by purpose (default: training).",
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        help="Optional camera id filter. May be passed multiple times or comma-separated.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print all scanned rows, not just source-like / stale / error rows.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if a camera does not have exactly one source-like row or has stale/missing entries.",
    )
    args = parser.parse_args(argv)

    camera_ids: set[str] = set()
    for raw in args.camera_id or ():
        for token in str(raw).split(","):
            token = token.strip()
            if token:
                camera_ids.add(token)

    roots = _resolve_roots(list(args.paths))
    rows = [
        _scan_zarr_path(
            zarr_path,
            zarr_use_filter=str(args.zarr_use),
            camera_ids=camera_ids,
        )
        for zarr_path in _iter_zarr(roots, bool(args.recursive))
    ]

    interesting_statuses = {"source_like", "stale_source_context", "missing_swim_tuning", "error", "clean_unmarked"}
    for row in rows:
        if row.status in {"filtered_camera", "filtered_zarr_use"}:
            continue
        if args.show_all or row.status in interesting_statuses:
            _print_row(row)

    summaries = _summarize_by_camera(rows)
    _print_summary(summaries)

    if args.strict and _has_strict_failures(summaries, rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

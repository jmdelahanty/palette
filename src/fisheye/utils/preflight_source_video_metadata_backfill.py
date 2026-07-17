"""Preflight a v2 source-video metadata backfill without mutating archives.

The command opens the Palette registry in SQLite read-only/query-only mode and
reads Zarr metadata files directly. It has no apply mode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from fisheye.shared.source_video_metadata import (
    SOURCE_VIDEO_LAYOUT_SINGLE,
    SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
    SOURCE_VIDEO_METADATA_SCHEMA_ID,
    SourceVideoMetadataError,
    build_source_video_metadata_v2,
    resolve_source_video_from_attrs,
)


REPORT_SCHEMA_ID = "palette.source_video_metadata_backfill_preflight.v1"
DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)
DEFAULT_PATH_CONTAINS = "GoodCopBadCop"
DEFAULT_EXPECTED_COUNT = 40
DEFAULT_REQUIRED_STORAGE_ROOT = Path("/groups")


@dataclass(frozen=True)
class RegistryDataset:
    dataset_id: str
    recording_id: str | None
    zarr_path: Path
    dataset_status: str | None
    zarr_use: str | None
    artifact_kind: str | None
    source_layout: str | None
    registry_recording_path: Path | None
    recording_name: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_path(value: Any) -> Path | None:
    text = _norm_text(value)
    return Path(text).expanduser().resolve() if text else None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _connect_registry_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def select_registry_datasets(
    registry_path: Path,
    *,
    path_contains: str,
) -> list[RegistryDataset]:
    conn = _connect_registry_read_only(registry_path)
    try:
        rows = conn.execute(
            """
            SELECT
                d.dataset_id,
                d.recording_id,
                d.zarr_path,
                d.status AS dataset_status,
                d.zarr_use,
                d.artifact_kind,
                d.source_layout,
                r.recording_path AS registry_recording_path,
                r.recording_name
            FROM datasets AS d
            LEFT JOIN recordings AS r ON r.recording_id = d.recording_id
            WHERE d.status = 'active'
              AND d.zarr_use = 'analysis'
              AND d.zarr_path LIKE ?
            ORDER BY d.zarr_path, d.dataset_id
            """,
            (f"%{path_contains}%",),
        ).fetchall()
    finally:
        conn.close()
    return [
        RegistryDataset(
            dataset_id=str(row["dataset_id"]),
            recording_id=_norm_text(row["recording_id"]),
            zarr_path=Path(str(row["zarr_path"])).expanduser(),
            dataset_status=_norm_text(row["dataset_status"]),
            zarr_use=_norm_text(row["zarr_use"]),
            artifact_kind=_norm_text(row["artifact_kind"]),
            source_layout=_norm_text(row["source_layout"]),
            registry_recording_path=(
                Path(str(row["registry_recording_path"])).expanduser()
                if _norm_text(row["registry_recording_path"])
                else None
            ),
            recording_name=_norm_text(row["recording_name"]),
        )
        for row in rows
    ]


def read_attrs_strict(
    group_path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    zarr_json = group_path / "zarr.json"
    if zarr_json.is_file():
        raw = zarr_json.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        attrs = payload.get("attributes")
        if not isinstance(attrs, dict):
            raise ValueError(f"{zarr_json}: attributes is not an object")
        stat = zarr_json.stat()
        return attrs, "zarr_v3", {
            "path": str(zarr_json),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    zattrs = group_path / ".zattrs"
    if zattrs.is_file():
        raw = zattrs.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{zattrs}: root value is not an object")
        stat = zattrs.stat()
        return payload, "zarr_v2", {
            "path": str(zattrs),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    raise FileNotFoundError(f"No zarr.json or .zattrs at {group_path}")


def _existing_source_metadata(root_attrs: Mapping[str, Any]) -> dict[str, Any]:
    value = root_attrs.get("source_video_metadata")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("root source_video_metadata is not an object")
    return dict(value)


def _proposed_source_metadata(
    root_attrs: Mapping[str, Any],
    *,
    source_video_path: Path,
    recording_path: Path,
) -> dict[str, Any]:
    metadata = _existing_source_metadata(root_attrs)
    metadata["source_path"] = str(source_video_path)
    field_sources = {
        "source_video": "source_video",
        "width": "width",
        "height": "height",
        "total_frames": "total_frames",
        "fps": "fps",
        "duration_seconds": "duration_seconds",
        "codec": "video_codec",
        "pix_fmt": "video_pix_fmt",
    }
    for metadata_key, root_key in field_sources.items():
        value = root_attrs.get(root_key)
        if metadata.get(metadata_key) in (None, "") and value not in (None, ""):
            metadata[metadata_key] = value
    return build_source_video_metadata_v2(
        metadata,
        recording_path=recording_path,
    )


def _operational_paths(
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
) -> dict[str, str]:
    candidates = {
        "root.source_video_path": root_attrs.get("source_video_path"),
        "root.source_path": root_attrs.get("source_path"),
        "root.video_source_path": root_attrs.get("video_source_path"),
        "source_video_metadata.source_path": source_metadata.get("source_path"),
        "raw_video.source_path": raw_attrs.get("source_path"),
        "raw_video.source_video_path": raw_attrs.get("source_video_path"),
    }
    return {
        key: text
        for key, value in candidates.items()
        if (text := _norm_text(value)) is not None
    }


def preflight_dataset(
    dataset: RegistryDataset,
    *,
    required_storage_root: Path = DEFAULT_REQUIRED_STORAGE_ROOT,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    zarr_path = dataset.zarr_path.expanduser().resolve()
    required_root = required_storage_root.expanduser().resolve()
    physical_recording_path = (
        zarr_path.parent.parent if zarr_path.parent.name == "zarr" else None
    )
    row: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "recording_id": dataset.recording_id,
        "zarr_path": str(zarr_path),
        "dataset_status": dataset.dataset_status,
        "zarr_use": dataset.zarr_use,
        "artifact_kind": dataset.artifact_kind,
        "source_layout": dataset.source_layout,
        "registry_recording_path": (
            str(dataset.registry_recording_path.expanduser().resolve())
            if dataset.registry_recording_path is not None
            else None
        ),
        "physical_recording_path": (
            str(physical_recording_path) if physical_recording_path is not None else None
        ),
        "errors": errors,
        "warnings": warnings,
    }

    if not zarr_path.is_dir():
        errors.append("zarr_path_missing")
        row["disposition"] = "blocked"
        return row
    if physical_recording_path is None:
        errors.append("zarr_not_under_recording_zarr_directory")

    try:
        root_attrs, metadata_format, root_metadata_precondition = read_attrs_strict(
            zarr_path
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"root_metadata_unreadable:{exc}")
        row["disposition"] = "blocked"
        return row
    row["metadata_format"] = metadata_format
    row["root_metadata_precondition"] = root_metadata_precondition
    try:
        raw_attrs, _, raw_metadata_precondition = read_attrs_strict(
            zarr_path / "raw_video"
        )
        row["raw_video_metadata_precondition"] = raw_metadata_precondition
    except FileNotFoundError:
        raw_attrs = {}
        warnings.append("raw_video_metadata_absent")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"raw_video_metadata_unreadable:{exc}")
        raw_attrs = {}

    declared_recording_path = _normalized_path(root_attrs.get("recording_path"))
    registry_recording_path = (
        dataset.registry_recording_path.expanduser().resolve()
        if dataset.registry_recording_path is not None
        else None
    )
    row["declared_recording_path"] = (
        str(declared_recording_path) if declared_recording_path is not None else None
    )
    if registry_recording_path is None:
        errors.append("registry_recording_path_missing")
    if physical_recording_path is not None and registry_recording_path is not None:
        if registry_recording_path != physical_recording_path:
            errors.append("registry_recording_path_mismatch")
    if declared_recording_path is not None and physical_recording_path is not None:
        if declared_recording_path != physical_recording_path:
            errors.append("declared_recording_path_mismatch")
    if not dataset.recording_id:
        errors.append("registry_recording_id_missing")

    recording_path = registry_recording_path or physical_recording_path
    if recording_path is None:
        errors.append("recording_root_unresolved")
        row["disposition"] = "blocked"
        return row

    try:
        existing_metadata = _existing_source_metadata(root_attrs)
    except ValueError as exc:
        errors.append(str(exc))
        existing_metadata = {}
    row["schema_before"] = _norm_text(existing_metadata.get("schema_id"))
    operational_paths = _operational_paths(root_attrs, raw_attrs, existing_metadata)
    row["operational_paths_before"] = operational_paths
    nvme1_fields = sorted(
        key for key, value in operational_paths.items() if "/nvme1" in value
    )
    row["nvme1_fields"] = nvme1_fields
    if nvme1_fields:
        errors.append("nvme1_operational_locator_present")

    if dataset.source_layout and dataset.source_layout not in {
        "recording",
        "source_recording",
        "single_video",
    }:
        errors.append(f"collection_or_unsupported_source_layout:{dataset.source_layout}")

    try:
        resolved = resolve_source_video_from_attrs(
            root_attrs,
            raw_video_attrs=raw_attrs,
            zarr_path=zarr_path,
            require_exists=True,
        )
    except SourceVideoMetadataError as exc:
        errors.append(f"source_video_resolution_failed:{exc}")
        row["disposition"] = "blocked"
        return row

    row["source_video_path"] = str(resolved.path)
    source_stat = resolved.path.stat()
    row["source_video_precondition"] = {
        "path": str(resolved.path),
        "size_bytes": int(source_stat.st_size),
        "mtime_ns": int(source_stat.st_mtime_ns),
    }
    row["resolver_source"] = resolved.source
    row["compatibility_sources"] = list(resolved.compatibility_sources)
    try:
        relative_video_path = resolved.path.relative_to(recording_path)
    except ValueError:
        errors.append("source_video_outside_recording_root")
        relative_video_path = None
    if relative_video_path is not None:
        row["source_video_relative_path"] = relative_video_path.as_posix()
        if not relative_video_path.parts or relative_video_path.parts[0] != "cams":
            errors.append("source_video_not_under_recording_cams")
    if not _is_under(resolved.path, required_root):
        errors.append("source_video_outside_required_storage_root")
    if not _is_under(zarr_path, required_root):
        errors.append("zarr_outside_required_storage_root")

    try:
        proposed_metadata = _proposed_source_metadata(
            root_attrs,
            source_video_path=resolved.path,
            recording_path=recording_path,
        )
        json.dumps(proposed_metadata, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError, SourceVideoMetadataError) as exc:
        errors.append(f"proposed_metadata_invalid:{exc}")
        row["disposition"] = "blocked"
        return row

    locator = proposed_metadata.get("locator")
    if not isinstance(locator, Mapping):
        errors.append("proposed_locator_missing")
    elif locator.get("kind") != SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE:
        errors.append("proposed_locator_not_recording_relative")

    row["planned_root_updates"] = {
        "recording_path": str(recording_path),
        "source_video_path": str(resolved.path),
        "source_path": str(resolved.path),
        "source_video_metadata": proposed_metadata,
    }
    row["planned_raw_video_updates"] = {
        "source_path": str(resolved.path),
    }
    already_v2 = (
        row["schema_before"] == SOURCE_VIDEO_METADATA_SCHEMA_ID
        and resolved.layout == SOURCE_VIDEO_LAYOUT_SINGLE
        and resolved.locator_kind == SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE
    )
    row["disposition"] = (
        "blocked" if errors else "already_v2" if already_v2 else "eligible"
    )
    return row


def build_preflight_report(
    datasets: Sequence[RegistryDataset],
    *,
    registry_path: Path,
    path_contains: str,
    expected_count: int,
    required_storage_root: Path = DEFAULT_REQUIRED_STORAGE_ROOT,
) -> dict[str, Any]:
    rows = [
        preflight_dataset(
            dataset,
            required_storage_root=required_storage_root,
        )
        for dataset in datasets
    ]
    path_counts = Counter(str(dataset.zarr_path.expanduser().resolve()) for dataset in datasets)
    recording_counts = Counter(dataset.recording_id for dataset in datasets if dataset.recording_id)
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    duplicate_recording_ids = sorted(
        recording_id for recording_id, count in recording_counts.items() if count > 1
    )
    dispositions = Counter(str(row.get("disposition") or "unknown") for row in rows)
    error_counts = Counter(
        error.split(":", 1)[0]
        for row in rows
        for error in row.get("errors", [])
    )
    cohort_errors: list[str] = []
    if len(datasets) != expected_count:
        cohort_errors.append(
            f"expected_count_mismatch:expected={expected_count}:actual={len(datasets)}"
        )
    if duplicate_paths:
        cohort_errors.append("duplicate_zarr_paths")
    if duplicate_recording_ids:
        cohort_errors.append("duplicate_recording_ids")
    ready = not cohort_errors and dispositions.get("blocked", 0) == 0
    return {
        "schema_id": REPORT_SCHEMA_ID,
        "created_at_utc": _utc_now(),
        "mode": "read_only_preflight",
        "registry_path": str(registry_path.expanduser().resolve()),
        "selection": {
            "dataset_status": "active",
            "zarr_use": "analysis",
            "path_contains": path_contains,
            "expected_count": expected_count,
            "required_storage_root": str(required_storage_root.expanduser().resolve()),
        },
        "summary": {
            "selected_rows": len(datasets),
            "distinct_zarr_paths": len(path_counts),
            "distinct_recording_ids": len(recording_counts),
            "dispositions": dict(sorted(dispositions.items())),
            "error_counts": dict(sorted(error_counts.items())),
            "duplicate_zarr_paths": duplicate_paths,
            "duplicate_recording_ids": duplicate_recording_ids,
            "cohort_errors": cohort_errors,
            "ready_to_apply": ready,
        },
        "rows": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--path-contains", default=DEFAULT_PATH_CONTAINS)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    parser.add_argument(
        "--required-storage-root",
        type=Path,
        default=DEFAULT_REQUIRED_STORAGE_ROOT,
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    datasets = select_registry_datasets(
        args.registry,
        path_contains=str(args.path_contains),
    )
    report = build_preflight_report(
        datasets,
        registry_path=args.registry,
        path_contains=str(args.path_contains),
        expected_count=int(args.expected_count),
        required_storage_root=args.required_storage_root,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json is not None:
        print(f"report: {args.output_json.expanduser().resolve()}")
    return 0 if summary["ready_to_apply"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

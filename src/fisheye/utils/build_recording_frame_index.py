"""Build a recording-level frame index sidecar for Orange/Palette recordings.

The frame index is a derived table. It maps a parent recording frame clock to
the source video file and source-local frame index needed to decode that frame.
It intentionally does not store review state, edits, or downstream status.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc
import pyarrow.parquet as pq


MODULE_NAME = "fisheye.utils.build_recording_frame_index"
SCHEMA_VERSION = "palette.recording_frame_index.v1"
MANIFEST_SCHEMA_VERSION = "palette.recording_frame_index_manifest.v1"
ARTIFACT_ROLE = "palette_derived_convenience_index"

FRAME_ID_BASE_CHOICES = ("auto", "zero", "one")

TABLE_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("recording_id", pa.string()),
        ("producer", pa.string()),
        ("recording_folder", pa.string()),
        ("source_layout", pa.string()),
        ("recording_backend_mode", pa.string()),
        ("camera_serial", pa.string()),
        ("recording_frame_id", pa.int64()),
        ("parent_frame_index", pa.int64()),
        ("clip_index", pa.int32()),
        ("clip_id", pa.string()),
        ("clip_local_frame_index", pa.int64()),
        ("timestamp", pa.int64()),
        ("timestamp_sys", pa.int64()),
        ("video_path", pa.string()),
        ("metadata_path", pa.string()),
        ("keyframe_path", pa.string()),
        ("clip_manifest_path", pa.string()),
        ("clip_directory", pa.string()),
        ("clip_recording_folder", pa.string()),
    ]
)

TABLE_COLUMNS = [field.name for field in TABLE_SCHEMA]


@dataclass(frozen=True)
class MetadataRows:
    path: Path
    fieldnames: tuple[str, ...]
    frame_ids: list[int]
    timestamps: list[int | None]
    timestamps_sys: list[int | None]

    @property
    def row_count(self) -> int:
        return len(self.frame_ids)


@dataclass(frozen=True)
class BuildOutputs:
    parquet_path: Path
    manifest_path: Path
    csv_path: Path | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _safe_replace_table(path: Path, write_fn: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    write_fn(tmp)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _camera_serial_from_name(path: Path) -> str | None:
    match = re.search(r"Cam(\d+)", path.name)
    return match.group(1) if match else None


def _resolve_path(recording_dir: Path, value: Any) -> Path:
    if value is None:
        raise ValueError("Expected path value, got None")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved_root = recording_dir.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Recording-relative path escapes recording root: {value}")
    return resolved


def _optional_resolve_path(recording_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _resolve_path(recording_dir, text)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _load_metadata_csv(path: Path) -> MetadataRows:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    frame_ids: list[int] = []
    timestamps: list[int | None] = []
    timestamps_sys: list[int | None] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: {path}")
        fieldnames = tuple(str(name) for name in reader.fieldnames)
        if "recording_frame_id" in fieldnames:
            frame_key = "recording_frame_id"
        elif "frame_id" in fieldnames:
            frame_key = "frame_id"
        else:
            raise ValueError(f"Metadata CSV must include frame_id or recording_frame_id: {path}")
        for row_number, row in enumerate(reader, start=2):
            try:
                frame_ids.append(int(str(row.get(frame_key)).strip()))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {frame_key} at {path}:{row_number}: {row.get(frame_key)!r}"
                ) from exc
            timestamps.append(_as_optional_int(row.get("timestamp")))
            timestamps_sys.append(_as_optional_int(row.get("timestamp_sys")))
    return MetadataRows(
        path=path,
        fieldnames=fieldnames,
        frame_ids=frame_ids,
        timestamps=timestamps,
        timestamps_sys=timestamps_sys,
    )


def _frame_id_offset(first_recording_frame_id: int, frame_id_base: str) -> int:
    if frame_id_base == "zero":
        return 0
    if frame_id_base == "one":
        return 1
    if frame_id_base != "auto":
        raise ValueError(f"Unsupported frame_id_base: {frame_id_base}")
    return 0 if int(first_recording_frame_id) == 0 else 1


def _slice_gap_count(values: Sequence[int]) -> int:
    if len(values) <= 1:
        return 0
    gaps = 0
    previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current != previous + 1:
            gaps += 1
        previous = current
    return gaps


def _file_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    exists = path.exists()
    payload: dict[str, Any] = {"path": str(path), "exists": bool(exists)}
    if exists:
        stat = path.stat()
        payload.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return payload


def _new_columns() -> dict[str, list[Any]]:
    return {name: [] for name in TABLE_COLUMNS}


def _append_frame_rows(
    *,
    columns: dict[str, list[Any]],
    metadata: MetadataRows,
    recording_dir: Path,
    session_id: str,
    recording_id: str,
    producer: str,
    source_layout: str,
    recording_backend_mode: str,
    camera_serial: str,
    clip_index: int,
    clip_id: str,
    video_path: Path,
    keyframe_path: Path | None,
    clip_manifest_path: Path | None,
    clip_directory: str,
    clip_recording_folder: Path,
    frame_id_base: str,
) -> dict[str, Any]:
    if metadata.row_count == 0:
        raise ValueError(f"Metadata CSV has no rows: {metadata.path}")
    offset = _frame_id_offset(metadata.frame_ids[0], frame_id_base)
    gaps = _slice_gap_count(metadata.frame_ids)
    for local_index, recording_frame_id in enumerate(metadata.frame_ids):
        parent_frame_index = int(recording_frame_id) - offset
        columns["session_id"].append(session_id)
        columns["recording_id"].append(recording_id)
        columns["producer"].append(producer)
        columns["recording_folder"].append(str(recording_dir))
        columns["source_layout"].append(source_layout)
        columns["recording_backend_mode"].append(recording_backend_mode)
        columns["camera_serial"].append(camera_serial)
        columns["recording_frame_id"].append(int(recording_frame_id))
        columns["parent_frame_index"].append(parent_frame_index)
        columns["clip_index"].append(int(clip_index))
        columns["clip_id"].append(clip_id)
        columns["clip_local_frame_index"].append(int(local_index))
        columns["timestamp"].append(metadata.timestamps[local_index])
        columns["timestamp_sys"].append(metadata.timestamps_sys[local_index])
        columns["video_path"].append(str(video_path))
        columns["metadata_path"].append(str(metadata.path))
        columns["keyframe_path"].append(str(keyframe_path) if keyframe_path is not None else None)
        columns["clip_manifest_path"].append(
            str(clip_manifest_path) if clip_manifest_path is not None else None
        )
        columns["clip_directory"].append(clip_directory)
        columns["clip_recording_folder"].append(str(clip_recording_folder))
    return {
        "metadata_path": str(metadata.path),
        "rows": int(metadata.row_count),
        "first_recording_frame_id": int(metadata.frame_ids[0]),
        "last_recording_frame_id": int(metadata.frame_ids[-1]),
        "recording_frame_id_gaps": int(gaps),
        "parent_frame_index_offset": int(offset),
    }


def _iter_clip_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("clips")
    if raw_rows is None:
        raw_rows = payload.get("rows")
    if raw_rows is None:
        raw_rows = payload.get("camera_artifacts")
    if not isinstance(raw_rows, list):
        raise ValueError("recording_clip_index JSON must include a clips, rows, or camera_artifacts list")

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("recording_clip_index row is not an object")
        if isinstance(raw.get("camera_artifacts"), list):
            base = {key: value for key, value in raw.items() if key != "camera_artifacts"}
            for artifact in raw["camera_artifacts"]:
                if not isinstance(artifact, Mapping):
                    raise ValueError("camera_artifacts item is not an object")
                merged = {**base, **dict(artifact)}
                rows.append(merged)
        else:
            rows.append(dict(raw))
    return rows


def _validate_expected_row(
    *,
    row: Mapping[str, Any],
    metadata_summary: Mapping[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    clip_id = row.get("clip_id")
    expected_count = _as_optional_int(row.get("frame_count"))
    if expected_count is not None:
        checks.append(
            {
                "status": "ok" if expected_count == metadata_summary["rows"] else "fail",
                "code": "metadata_rows_match_clip_index_frame_count",
                "clip_id": clip_id,
                "expected": expected_count,
                "observed": metadata_summary["rows"],
            }
        )
    expected_first = _as_optional_int(row.get("first_recording_frame_id"))
    if expected_first is not None:
        checks.append(
            {
                "status": "ok"
                if expected_first == metadata_summary["first_recording_frame_id"]
                else "fail",
                "code": "first_recording_frame_id_matches_clip_index",
                "clip_id": clip_id,
                "expected": expected_first,
                "observed": metadata_summary["first_recording_frame_id"],
            }
        )
    expected_last = _as_optional_int(row.get("last_recording_frame_id"))
    if expected_last is not None:
        checks.append(
            {
                "status": "ok"
                if expected_last == metadata_summary["last_recording_frame_id"]
                else "fail",
                "code": "last_recording_frame_id_matches_clip_index",
                "clip_id": clip_id,
                "expected": expected_last,
                "observed": metadata_summary["last_recording_frame_id"],
            }
        )
    checks.append(
        {
            "status": "ok" if int(metadata_summary["recording_frame_id_gaps"]) == 0 else "fail",
            "code": "clip_recording_frame_id_continuity",
            "clip_id": clip_id,
            "recording_frame_id_gaps": int(metadata_summary["recording_frame_id_gaps"]),
        }
    )


def _build_from_clips(
    *,
    recording_dir: Path,
    clip_index_json: Path,
    frame_id_base: str,
) -> tuple[pa.Table, dict[str, Any]]:
    payload = _read_json(clip_index_json)
    if not isinstance(payload, Mapping):
        raise ValueError(f"recording_clip_index JSON is not an object: {clip_index_json}")
    rows = _iter_clip_rows(payload)
    if not rows:
        raise ValueError(f"recording_clip_index has no rows: {clip_index_json}")

    columns = _new_columns()
    checks: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    per_input: list[dict[str, Any]] = []
    camera_serials: set[str] = set()
    per_camera_previous: dict[str, int] = {}

    recording_id = str(payload.get("recording_id") or recording_dir.name)
    session_id = str(payload.get("session_id") or recording_id)
    producer = str(payload.get("producer") or "orange_rolling_clip")
    backend_mode = str(payload.get("recording_backend_mode") or payload.get("mode") or "rolling_clips")

    for row in sorted(rows, key=lambda item: (str(item.get("camera_serial") or ""), int(item.get("clip_index") or 0))):
        if not row.get("metadata_path"):
            raise ValueError(f"Clip row is missing metadata_path: {row}")
        if not row.get("video_path"):
            raise ValueError(f"Clip row is missing video_path: {row}")
        metadata_path = _resolve_path(recording_dir, row["metadata_path"])
        video_path = _resolve_path(recording_dir, row["video_path"])
        keyframe_path = _optional_resolve_path(recording_dir, row.get("keyframe_path"))
        clip_manifest_path = _optional_resolve_path(recording_dir, row.get("clip_manifest_path"))
        clip_id = str(row.get("clip_id") or f"clip_{int(row.get('clip_index') or 0):06d}")
        clip_index = int(row.get("clip_index") or 0)
        clip_directory = str(row.get("clip_directory") or f"clips/{clip_id}")
        clip_recording_folder = _resolve_path(recording_dir, clip_directory)
        camera_serial = str(row.get("camera_serial") or _camera_serial_from_name(video_path) or "")
        if not camera_serial:
            raise ValueError(f"Could not resolve camera serial for clip row: {row}")
        metadata = _load_metadata_csv(metadata_path)
        summary = _append_frame_rows(
            columns=columns,
            metadata=metadata,
            recording_dir=recording_dir,
            session_id=str(row.get("session_id") or session_id),
            recording_id=str(row.get("recording_id") or recording_id),
            producer=producer,
            source_layout="rolling_clips",
            recording_backend_mode=str(row.get("recording_backend_mode") or backend_mode),
            camera_serial=camera_serial,
            clip_index=clip_index,
            clip_id=clip_id,
            video_path=video_path,
            keyframe_path=keyframe_path,
            clip_manifest_path=clip_manifest_path,
            clip_directory=clip_directory,
            clip_recording_folder=clip_recording_folder,
            frame_id_base=frame_id_base,
        )
        _validate_expected_row(row=row, metadata_summary=summary, checks=checks)
        previous = per_camera_previous.get(camera_serial)
        if previous is not None:
            checks.append(
                {
                    "status": "ok"
                    if int(summary["first_recording_frame_id"]) == previous + 1
                    else "fail",
                    "code": "inter_clip_recording_frame_id_continuity",
                    "camera_serial": camera_serial,
                    "clip_id": clip_id,
                    "previous_last_recording_frame_id": previous,
                    "current_first_recording_frame_id": int(summary["first_recording_frame_id"]),
                }
            )
        per_camera_previous[camera_serial] = int(summary["last_recording_frame_id"])
        camera_serials.add(camera_serial)
        per_input.append({"clip_id": clip_id, "camera_serial": camera_serial, **summary})
        for path in (metadata_path, video_path, keyframe_path, clip_manifest_path):
            stats = _file_stats(path)
            if stats is not None:
                source_files.append(stats)

    table = pa.Table.from_pydict(columns, schema=TABLE_SCHEMA)
    details = {
        "source_layout": "rolling_clips",
        "recording_id": recording_id,
        "session_id": session_id,
        "camera_serials": sorted(camera_serials),
        "checks": checks,
        "source_files": source_files,
        "inputs": per_input,
        "recording_clip_index_json": str(clip_index_json),
        "recording_clip_index_csv": str(recording_dir / "recording_clip_index.csv")
        if (recording_dir / "recording_clip_index.csv").exists()
        else None,
    }
    return table, details


def _single_video_bundles(recording_dir: Path) -> list[dict[str, Any]]:
    cams_dir = recording_dir / "cams"
    search_dir = cams_dir if cams_dir.exists() else recording_dir
    bundles: list[dict[str, Any]] = []
    for metadata_path in sorted(search_dir.glob("Cam*_meta.csv")):
        base_name = metadata_path.name[: -len("_meta.csv")]
        video_path = metadata_path.with_name(f"{base_name}.mp4")
        keyframe_path = metadata_path.with_name(f"{base_name}_keyframe.json")
        if not video_path.exists():
            raise FileNotFoundError(f"Video for metadata CSV not found: {video_path}")
        bundles.append(
            {
                "metadata_path": metadata_path,
                "video_path": video_path,
                "keyframe_path": keyframe_path if keyframe_path.exists() else None,
                "camera_serial": _camera_serial_from_name(video_path),
            }
        )
    return bundles


def _build_from_single_video(
    *,
    recording_dir: Path,
    frame_id_base: str,
) -> tuple[pa.Table, dict[str, Any]]:
    bundles = _single_video_bundles(recording_dir)
    if not bundles:
        raise FileNotFoundError(
            f"No recording_clip_index.json and no Cam*_meta.csv files found under {recording_dir}/cams"
        )
    columns = _new_columns()
    checks: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    per_input: list[dict[str, Any]] = []
    camera_serials: set[str] = set()
    recording_id = recording_dir.name
    session_id = recording_id
    for bundle in bundles:
        camera_serial = str(bundle["camera_serial"] or "")
        if not camera_serial:
            raise ValueError(f"Could not resolve camera serial from {bundle['video_path']}")
        metadata = _load_metadata_csv(bundle["metadata_path"])
        summary = _append_frame_rows(
            columns=columns,
            metadata=metadata,
            recording_dir=recording_dir,
            session_id=session_id,
            recording_id=recording_id,
            producer="palette_single_video_frame_index",
            source_layout="single_video",
            recording_backend_mode="single_video",
            camera_serial=camera_serial,
            clip_index=0,
            clip_id="full_video",
            video_path=bundle["video_path"].resolve(),
            keyframe_path=bundle["keyframe_path"].resolve() if bundle["keyframe_path"] is not None else None,
            clip_manifest_path=None,
            clip_directory="cams",
            clip_recording_folder=(recording_dir / "cams").resolve()
            if (recording_dir / "cams").exists()
            else recording_dir.resolve(),
            frame_id_base=frame_id_base,
        )
        checks.append(
            {
                "status": "ok" if int(summary["recording_frame_id_gaps"]) == 0 else "fail",
                "code": "single_video_recording_frame_id_continuity",
                "camera_serial": camera_serial,
                "recording_frame_id_gaps": int(summary["recording_frame_id_gaps"]),
            }
        )
        camera_serials.add(camera_serial)
        per_input.append({"clip_id": "full_video", "camera_serial": camera_serial, **summary})
        for path in (bundle["metadata_path"], bundle["video_path"], bundle["keyframe_path"]):
            stats = _file_stats(path)
            if stats is not None:
                source_files.append(stats)
    table = pa.Table.from_pydict(columns, schema=TABLE_SCHEMA)
    details = {
        "source_layout": "single_video",
        "recording_id": recording_id,
        "session_id": session_id,
        "camera_serials": sorted(camera_serials),
        "checks": checks,
        "source_files": source_files,
        "inputs": per_input,
        "recording_clip_index_json": None,
        "recording_clip_index_csv": None,
    }
    return table, details


def _manifest_for_table(
    *,
    table: pa.Table,
    recording_dir: Path,
    outputs: BuildOutputs,
    details: Mapping[str, Any],
    frame_id_base: str,
    dry_run: bool,
    duration_seconds: float,
) -> dict[str, Any]:
    row_count = int(table.num_rows)
    if row_count:
        frame_range = pc.min_max(table.column("recording_frame_id")).as_py()
        frame_min = int(frame_range["min"])
        frame_max = int(frame_range["max"])
    else:
        frame_min = None
        frame_max = None
    checks = list(details.get("checks") or [])
    checks.append(
        {
            "status": "ok" if row_count > 0 else "fail",
            "code": "recording_frame_index_nonempty",
            "row_count": row_count,
        }
    )
    failures = [item for item in checks if isinstance(item, Mapping) and item.get("status") != "ok"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "frame_index_schema_version": SCHEMA_VERSION,
        "generated_by": MODULE_NAME,
        "generated_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "artifact_role": ARTIFACT_ROLE,
        "source_authority": (
            "recording_clip_index + per_clip_metadata_csv"
            if details.get("source_layout") == "rolling_clips"
            else "single_video_metadata_csv"
        ),
        "status": "ok" if not failures else "fail",
        "dry_run": bool(dry_run),
        "recording_folder": str(recording_dir),
        "source_layout": details.get("source_layout"),
        "recording_id": details.get("recording_id"),
        "session_id": details.get("session_id"),
        "recording_clip_index_json": details.get("recording_clip_index_json"),
        "recording_clip_index_csv": details.get("recording_clip_index_csv"),
        "recording_frame_index_path": str(outputs.parquet_path),
        "recording_frame_index_manifest_path": str(outputs.manifest_path),
        "recording_frame_index_csv_path": str(outputs.csv_path) if outputs.csv_path is not None else None,
        "row_count": row_count,
        "columns": TABLE_COLUMNS,
        "camera_serials": list(details.get("camera_serials") or []),
        "recording_frame_id_min": frame_min,
        "recording_frame_id_max": frame_max,
        "frame_id_base": frame_id_base,
        "checks": checks,
        "failure_count": len(failures),
        "inputs": details.get("inputs") or [],
        "source_files": details.get("source_files") or [],
        "duration_seconds": float(duration_seconds),
    }


def build_recording_frame_index(
    recording_dir: str | Path,
    *,
    output_parquet: str | Path | None = None,
    output_manifest: str | Path | None = None,
    output_csv: str | Path | None = None,
    write_csv: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    frame_id_base: str = "auto",
) -> dict[str, Any]:
    """Build and optionally write a recording-frame index sidecar."""
    started = time.perf_counter()
    if frame_id_base not in FRAME_ID_BASE_CHOICES:
        raise ValueError(f"frame_id_base must be one of {FRAME_ID_BASE_CHOICES}")
    root = Path(recording_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Recording folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Recording path is not a directory: {root}")
    parquet_path = Path(output_parquet).expanduser().resolve() if output_parquet else root / "recording_frame_index.parquet"
    manifest_path = (
        Path(output_manifest).expanduser().resolve()
        if output_manifest
        else root / "recording_frame_index_manifest.json"
    )
    csv_path = (
        Path(output_csv).expanduser().resolve()
        if output_csv
        else root / "recording_frame_index.csv"
        if write_csv
        else None
    )
    outputs = BuildOutputs(parquet_path=parquet_path, manifest_path=manifest_path, csv_path=csv_path)

    clip_index_json = root / "recording_clip_index.json"
    if clip_index_json.exists():
        table, details = _build_from_clips(
            recording_dir=root,
            clip_index_json=clip_index_json,
            frame_id_base=frame_id_base,
        )
    else:
        table, details = _build_from_single_video(recording_dir=root, frame_id_base=frame_id_base)

    duration_seconds = time.perf_counter() - started
    manifest = _manifest_for_table(
        table=table,
        recording_dir=root,
        outputs=outputs,
        details=details,
        frame_id_base=frame_id_base,
        dry_run=dry_run,
        duration_seconds=duration_seconds,
    )

    if not dry_run:
        if manifest["status"] != "ok":
            raise RuntimeError(
                "Refusing to write recording frame index because validation failed. "
                "Run with --dry-run --json to inspect checks."
            )
        _safe_replace_table(
            parquet_path,
            lambda tmp: pq.write_table(table, tmp, compression="zstd"),
            overwrite=overwrite,
        )
        if csv_path is not None:
            _safe_replace_table(
                csv_path,
                lambda tmp: pacsv.write_csv(table, tmp),
                overwrite=overwrite,
            )
        _write_json(manifest_path, manifest, overwrite=overwrite)

    return {
        "status": manifest["status"],
        "generated_by": MODULE_NAME,
        "dry_run": bool(dry_run),
        "recording_folder": str(root),
        "source_layout": details.get("source_layout"),
        "row_count": int(table.num_rows),
        "camera_serials": list(details.get("camera_serials") or []),
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "wrote_parquet": bool(not dry_run),
        "wrote_manifest": bool(not dry_run),
        "wrote_csv": bool(not dry_run and csv_path is not None),
        "failure_count": int(manifest["failure_count"]),
        "recording_frame_id_min": manifest["recording_frame_id_min"],
        "recording_frame_id_max": manifest["recording_frame_id_max"],
        "checks": manifest["checks"],
        "manifest": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build recording_frame_index.parquet from Orange rolling clips or a single-video "
            "Cam*_meta.csv bundle."
        )
    )
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--write-csv", action="store_true", help="Also write recording_frame_index.csv.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output sidecars.")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate the table without writing files.")
    parser.add_argument(
        "--frame-id-base",
        choices=FRAME_ID_BASE_CHOICES,
        default="auto",
        help="How to derive parent_frame_index from recording_frame_id.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON summary.")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"source_layout: {result.get('source_layout')}")
    print(f"recording_folder: {result.get('recording_folder')}")
    print(f"row_count: {result.get('row_count')}")
    print(f"camera_serials: {result.get('camera_serials')}")
    print(
        "recording_frame_id_range: "
        f"{result.get('recording_frame_id_min')}..{result.get('recording_frame_id_max')}"
    )
    print(f"parquet_path: {result.get('parquet_path')}")
    print(f"manifest_path: {result.get('manifest_path')}")
    if result.get("csv_path"):
        print(f"csv_path: {result.get('csv_path')}")
    print(f"wrote_parquet: {result.get('wrote_parquet')}")
    print(f"wrote_manifest: {result.get('wrote_manifest')}")
    print(f"wrote_csv: {result.get('wrote_csv')}")
    print(f"failure_count: {result.get('failure_count')}")
    failures = [
        check
        for check in list(result.get("checks") or [])
        if isinstance(check, Mapping) and check.get("status") != "ok"
    ]
    for check in failures[:10]:
        print(f"failure: {check}")
    if len(failures) > 10:
        print(f"... {len(failures) - 10} more failures")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_recording_frame_index(
        args.recording_dir,
        output_parquet=args.output_parquet,
        output_manifest=args.output_manifest,
        output_csv=args.output_csv,
        write_csv=bool(args.write_csv or args.output_csv is not None),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        frame_id_base=args.frame_id_base,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

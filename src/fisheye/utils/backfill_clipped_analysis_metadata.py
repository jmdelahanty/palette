"""Backfill clipped-analysis Zarr and registry source metadata.

This utility repairs active metadata for clipped/rolling-clip analysis stores.
It does not rewrite model outputs or finalized detect collections. By default it
is a dry run; pass ``--apply`` to modify the Zarr root metadata and registry.
Pass ``--rewrite-frame-index-paths`` when the recording-level frame index still
contains stale absolute paths after a recording-store relocation.
Pass ``--source-video-path`` to repair the live root and ``raw_video`` fields
used by Crimson and Palette while preserving captured environment provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic


CLIPPED_SOURCE_LAYOUT = "rolling_clips"
DEFAULT_FRAME_INDEX_SCHEMA = "palette.recording_frame_index.v1"
PATH_COLUMNS = (
    "recording_folder",
    "video_path",
    "metadata_path",
    "keyframe_path",
    "clip_manifest_path",
    "clip_recording_folder",
)
SUPPORTED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True)
class RepairContext:
    zarr_path: Path
    recording_root: Path
    manifest_path: Path
    frame_index_path: Path
    recording_id: str
    session_id: str
    source_frame_index_schema: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _zarr_json_path(zarr_path: Path) -> Path:
    path = zarr_path / "zarr.json"
    if not path.exists():
        raise FileNotFoundError(f"Zarr metadata not found: {path}")
    return path


def _load_zarr_attrs(zarr_path: Path) -> dict[str, Any]:
    payload = _read_json(_zarr_json_path(zarr_path))
    attrs = payload.get("attributes")
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        raise ValueError(f"Zarr root attributes are not a JSON object: {zarr_path / 'zarr.json'}")
    return dict(attrs)


def _write_zarr_attrs(zarr_path: Path, attrs: Mapping[str, Any]) -> None:
    path = _zarr_json_path(zarr_path)
    payload = _read_json(path)
    payload["attributes"] = json_attr_safe(dict(attrs))
    write_json_atomic(path, payload)


def _infer_recording_root(zarr_path: Path) -> Path:
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent.resolve()
    return zarr_path.parent.resolve()


def _resolve_recording_relative(recording_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        relocated = recording_root / path.name
        if relocated.exists():
            return relocated.resolve()
        return path
    return (recording_root / path).resolve()


def _load_context(zarr_path: Path, recording_root: Path | None) -> RepairContext:
    resolved_zarr = zarr_path.expanduser().resolve()
    root = recording_root.expanduser().resolve() if recording_root else _infer_recording_root(resolved_zarr)
    manifest_path = root / "recording_frame_index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"recording_frame_index_manifest.json not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"recording_frame_index_manifest.json is not an object: {manifest_path}")
    frame_index_path = _resolve_recording_relative(
        root,
        manifest.get("recording_frame_index_path") or "recording_frame_index.parquet",
    )
    if not frame_index_path.exists():
        default_path = root / "recording_frame_index.parquet"
        if default_path.exists():
            frame_index_path = default_path.resolve()
        else:
            raise FileNotFoundError(f"recording_frame_index.parquet not found: {frame_index_path}")
    recording_id = str(manifest.get("recording_id") or root.name)
    session_id = str(manifest.get("session_id") or recording_id)
    schema = str(manifest.get("frame_index_schema_version") or DEFAULT_FRAME_INDEX_SCHEMA)
    return RepairContext(
        zarr_path=resolved_zarr,
        recording_root=root,
        manifest_path=manifest_path,
        frame_index_path=frame_index_path,
        recording_id=recording_id,
        session_id=session_id,
        source_frame_index_schema=schema,
    )


def _wanted_attrs(context: RepairContext, manifest: Mapping[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "zarr_purpose": "analysis",
        "recording_id": context.recording_id,
        "session_id": context.session_id,
        "recording_name": context.recording_root.name,
        "recording_path": str(context.recording_root),
        "source_layout": CLIPPED_SOURCE_LAYOUT,
        "recording_frame_index_path": str(context.frame_index_path),
        "source_recording_frame_index_path": str(context.frame_index_path),
        "recording_frame_index_manifest_path": str(context.manifest_path),
        "recording_frame_index_schema": context.source_frame_index_schema,
        "source_frame_index_schema": context.source_frame_index_schema,
    }
    for source_key, target_key in (
        ("row_count", "recording_frame_index_row_count"),
        ("recording_frame_id_min", "recording_frame_id_min"),
        ("recording_frame_id_max", "recording_frame_id_max"),
    ):
        if source_key in manifest:
            attrs[target_key] = manifest[source_key]
    if "camera_serials" in manifest:
        attrs["camera_serials"] = manifest["camera_serials"]
    return attrs


def _attr_diff(current: Mapping[str, Any], wanted: Mapping[str, Any], *, force: bool) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key, wanted_value in wanted.items():
        current_value = current.get(key)
        if current_value == wanted_value:
            continue
        current_text = "" if current_value is None else str(current_value)
        should_update = force or current_value in (None, "") or current_text.startswith("/nvme1/")
        if key == "zarr_purpose" and current_value == "production" and wanted_value == "analysis":
            should_update = True
        if key in {"recording_path", "recording_frame_index_path", "source_recording_frame_index_path"}:
            should_update = should_update or not Path(str(current_value)).exists()
        diff[key] = {
            "current": current_value,
            "wanted": wanted_value,
            "will_update": bool(should_update),
        }
    return diff


def _apply_attr_diff(current: Mapping[str, Any], diff: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    updated = dict(current)
    for key, item in diff.items():
        if item.get("will_update"):
            updated[key] = item.get("wanted")
    updated["clipped_analysis_metadata_backfilled_at_utc"] = utc_now()
    updated["clipped_analysis_metadata_backfill_tool"] = __name__
    return updated


def _validated_source_video_path(source_video_path: Path) -> Path:
    resolved = source_video_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source video not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported source video extension {resolved.suffix!r}: {resolved}"
        )
    return resolved


def _video_location_plan(zarr_path: Path, source_video_path: Path) -> dict[str, Any]:
    resolved_video = _validated_source_video_path(source_video_path)
    root_attrs = _load_zarr_attrs(zarr_path)
    raw_video_path = zarr_path / "raw_video"
    raw_attrs = _load_zarr_attrs(raw_video_path)

    source_metadata = root_attrs.get("source_video_metadata")
    if source_metadata is None:
        source_metadata = {}
    if not isinstance(source_metadata, Mapping):
        raise ValueError("Zarr root source_video_metadata is not an object.")

    wanted = str(resolved_video)
    current = {
        "root.source_path": root_attrs.get("source_path"),
        "root.source_video_path": root_attrs.get("source_video_path"),
        "root.source_video_metadata.source_path": source_metadata.get("source_path"),
        "raw_video.source_path": raw_attrs.get("source_path"),
    }
    changes = {
        key: {"current": value, "wanted": wanted}
        for key, value in current.items()
        if value != wanted
    }
    return {
        "source_video_path": wanted,
        "changes": changes,
        "will_update": bool(changes),
    }


def _apply_video_location_plan(
    zarr_path: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if not plan.get("will_update"):
        return {**dict(plan), "updated": False}

    wanted = str(plan["source_video_path"])
    changes = plan.get("changes")
    if not isinstance(changes, Mapping):
        raise ValueError("Video-location plan changes are not an object.")

    root_attrs = _load_zarr_attrs(zarr_path)
    raw_video_path = zarr_path / "raw_video"
    raw_attrs = _load_zarr_attrs(raw_video_path)

    source_metadata = root_attrs.get("source_video_metadata")
    if source_metadata is None:
        source_metadata = {}
    if not isinstance(source_metadata, Mapping):
        raise ValueError("Zarr root source_video_metadata is not an object.")

    # Write raw_video first because Crimson prefers this field when present.
    # Each metadata file is replaced atomically by write_json_atomic.
    updated_raw_attrs = dict(raw_attrs)
    updated_raw_attrs["source_path"] = wanted
    _write_zarr_attrs(raw_video_path, updated_raw_attrs)

    repaired_at = utc_now()
    updated_source_metadata = dict(source_metadata)
    updated_source_metadata["source_path"] = wanted
    updated_root_attrs = dict(root_attrs)
    updated_root_attrs["source_path"] = wanted
    updated_root_attrs["source_video_path"] = wanted
    updated_root_attrs["source_video_metadata"] = updated_source_metadata
    updated_root_attrs["source_video_location_repair"] = {
        "schema_id": "palette.source_video_location_repair.v1",
        "repaired_at_utc": repaired_at,
        "tool": __name__,
        "source_video_path": wanted,
        "previous_live_fields": {
            str(key): item.get("current")
            for key, item in changes.items()
            if isinstance(item, Mapping)
        },
        "historical_environment_provenance_preserved": True,
    }
    _write_zarr_attrs(zarr_path, updated_root_attrs)
    return {**dict(plan), "updated": True, "repaired_at_utc": repaired_at}


def _iter_path_strings(table: pa.Table, columns: Iterable[str]) -> Iterable[str]:
    for name in columns:
        if name not in table.column_names:
            continue
        for value in table[name].to_pylist():
            if value:
                yield str(value)


def _replace_prefix(value: Any, old_root: str, new_root: str) -> Any:
    if not isinstance(value, str):
        return value
    if value == old_root:
        return new_root
    prefix = old_root.rstrip("/") + "/"
    if value.startswith(prefix):
        return new_root.rstrip("/") + "/" + value[len(prefix) :]
    return value


def _rewrite_mapping_paths(payload: Any, old_root: str, new_root: str) -> Any:
    if isinstance(payload, dict):
        return {key: _rewrite_mapping_paths(value, old_root, new_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_rewrite_mapping_paths(value, old_root, new_root) for value in payload]
    return _replace_prefix(payload, old_root, new_root)


def _path_rewrite_plan(frame_index_path: Path, *, old_root: str, new_root: str) -> dict[str, Any]:
    table = pq.read_table(frame_index_path)
    counts: dict[str, int] = {}
    for name in PATH_COLUMNS:
        if name not in table.column_names:
            continue
        count = 0
        for value in table[name].to_pylist():
            if isinstance(value, str) and _replace_prefix(value, old_root, new_root) != value:
                count += 1
        if count:
            counts[name] = count
    return {
        "frame_index_path": str(frame_index_path),
        "row_count": table.num_rows,
        "old_root": old_root,
        "new_root": new_root,
        "columns_with_rewrites": counts,
        "rewrite_count": int(sum(counts.values())),
    }


def _rewrite_frame_index_paths(frame_index_path: Path, *, old_root: str, new_root: str) -> dict[str, Any]:
    table = pq.read_table(frame_index_path)
    arrays: list[pa.Array | pa.ChunkedArray] = []
    changed: dict[str, int] = {}
    for field in table.schema:
        column = table[field.name]
        if field.name not in PATH_COLUMNS:
            arrays.append(column)
            continue
        values = column.to_pylist()
        rewritten = [_replace_prefix(value, old_root, new_root) for value in values]
        count = sum(1 for before, after in zip(values, rewritten) if before != after)
        if count:
            changed[field.name] = int(count)
            arrays.append(pa.array(rewritten, type=field.type))
        else:
            arrays.append(column)
    rewritten_table = pa.Table.from_arrays(arrays, schema=table.schema)
    tmp_path = frame_index_path.with_name(f".{frame_index_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(rewritten_table, tmp_path)
    os.replace(tmp_path, frame_index_path)
    return {
        "frame_index_path": str(frame_index_path),
        "row_count": table.num_rows,
        "columns_rewritten": changed,
        "rewrite_count": int(sum(changed.values())),
    }


def _infer_old_root(context: RepairContext, manifest: Mapping[str, Any]) -> str | None:
    manifest_path = manifest.get("recording_frame_index_path")
    if isinstance(manifest_path, str) and manifest_path.startswith("/nvme1/"):
        return str(Path(manifest_path).parent)
    table = pq.read_table(context.frame_index_path, columns=[name for name in ("video_path", "metadata_path") if name])
    for value in _iter_path_strings(table, ("video_path", "metadata_path")):
        if value.startswith("/nvme1/"):
            marker = f"/{context.recording_root.name}/"
            if marker in value:
                return value.split(marker, 1)[0] + f"/{context.recording_root.name}"
    return None


def _registry_row(conn: sqlite3.Connection, zarr_path: Path) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT dataset_id, recording_id, zarr_use, source_layout,
               source_recording_frame_index_path, source_frame_index_schema
        FROM datasets
        WHERE zarr_path = ?
        ORDER BY last_seen_utc DESC
        LIMIT 1;
        """,
        (str(zarr_path),),
    ).fetchone()


def _registry_plan(registry_path: Path, context: RepairContext) -> dict[str, Any]:
    with sqlite3.connect(str(registry_path)) as conn:
        row = _registry_row(conn, context.zarr_path)
        if row is None:
            return {"registry_path": str(registry_path), "status": "missing_dataset_row"}
        wanted = {
            "recording_id": context.recording_id,
            "zarr_use": "analysis",
            "source_layout": CLIPPED_SOURCE_LAYOUT,
            "source_recording_frame_index_path": str(context.frame_index_path),
            "source_frame_index_schema": context.source_frame_index_schema,
        }
        changes = {
            key: {"current": row[key], "wanted": value}
            for key, value in wanted.items()
            if row[key] != value
        }
        return {
            "registry_path": str(registry_path),
            "status": "ok",
            "dataset_id": row["dataset_id"],
            "changes": changes,
        }


def _apply_registry_update(registry_path: Path, context: RepairContext) -> dict[str, Any]:
    with sqlite3.connect(str(registry_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = _registry_row(conn, context.zarr_path)
        if row is None:
            return {"registry_path": str(registry_path), "status": "missing_dataset_row"}
        now = utc_now()
        conn.execute(
            """
            UPDATE datasets
            SET recording_id = ?,
                zarr_use = ?,
                source_layout = ?,
                source_recording_frame_index_path = ?,
                source_frame_index_schema = ?,
                last_seen_utc = ?
            WHERE dataset_id = ?;
            """,
            (
                context.recording_id,
                "analysis",
                CLIPPED_SOURCE_LAYOUT,
                str(context.frame_index_path),
                context.source_frame_index_schema,
                now,
                row["dataset_id"],
            ),
        )
        conn.commit()
        return {"registry_path": str(registry_path), "status": "updated", "dataset_id": row["dataset_id"]}


def backfill_clipped_analysis_metadata(
    zarr_path: Path,
    *,
    recording_root: Path | None = None,
    source_video_path: Path | None = None,
    registry_path: Path | None = None,
    rewrite_frame_index_paths: bool = False,
    old_root: str | None = None,
    new_root: str | None = None,
    force: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    context = _load_context(zarr_path, recording_root)
    manifest = _read_json(context.manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"recording_frame_index_manifest.json is not an object: {context.manifest_path}")
    current_attrs = _load_zarr_attrs(context.zarr_path)
    wanted_attrs = _wanted_attrs(context, manifest)
    attr_diff = _attr_diff(current_attrs, wanted_attrs, force=force)
    video_location = (
        _video_location_plan(context.zarr_path, source_video_path)
        if source_video_path is not None
        else None
    )

    path_plan: dict[str, Any] | None = None
    if rewrite_frame_index_paths:
        resolved_old_root = old_root or _infer_old_root(context, manifest)
        if not resolved_old_root:
            raise ValueError("Could not infer old root for frame-index path rewrite; pass --old-root.")
        resolved_new_root = new_root or str(context.recording_root)
        path_plan = _path_rewrite_plan(context.frame_index_path, old_root=resolved_old_root, new_root=resolved_new_root)

    registry_result: dict[str, Any] | None = None
    if registry_path is not None:
        registry_result = _registry_plan(registry_path.expanduser().resolve(), context)

    result: dict[str, Any] = {
        "status": "planned",
        "apply": bool(apply),
        "zarr_path": str(context.zarr_path),
        "recording_root": str(context.recording_root),
        "recording_id": context.recording_id,
        "frame_index_path": str(context.frame_index_path),
        "attr_changes": attr_diff,
        "video_location": video_location,
        "path_rewrite": path_plan,
        "registry": registry_result,
    }
    if not apply:
        return result

    if any(item.get("will_update") for item in attr_diff.values()):
        _write_zarr_attrs(context.zarr_path, _apply_attr_diff(current_attrs, attr_diff))
        result["zarr_attrs_updated"] = True
    else:
        result["zarr_attrs_updated"] = False

    if video_location is not None:
        result["video_location"] = _apply_video_location_plan(
            context.zarr_path,
            video_location,
        )

    if rewrite_frame_index_paths and path_plan is not None and path_plan.get("rewrite_count", 0):
        result["path_rewrite"] = _rewrite_frame_index_paths(
            context.frame_index_path,
            old_root=str(path_plan["old_root"]),
            new_root=str(path_plan["new_root"]),
        )
        manifest_updated = _rewrite_mapping_paths(manifest, str(path_plan["old_root"]), str(path_plan["new_root"]))
        manifest_updated["rewritten_for_recording_store_relocation_at_utc"] = utc_now()
        write_json_atomic(context.manifest_path, manifest_updated)
        result["manifest_paths_updated"] = True
    else:
        result["manifest_paths_updated"] = False

    if registry_path is not None:
        result["registry"] = _apply_registry_update(registry_path.expanduser().resolve(), context)

    result["status"] = "applied"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Clipped analysis Zarr path to repair.")
    parser.add_argument("--recording-root", type=Path, help="Override recording root.")
    parser.add_argument(
        "--source-video-path",
        type=Path,
        help=(
            "Repair live root/raw_video source paths to this existing video while "
            "preserving historical environment provenance."
        ),
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path to update.")
    parser.add_argument("--rewrite-frame-index-paths", action="store_true", help="Rewrite stale active paths inside recording_frame_index.parquet and manifest.")
    parser.add_argument("--old-root", help="Old recording root prefix for frame-index path rewrite.")
    parser.add_argument("--new-root", help="New recording root prefix for frame-index path rewrite. Defaults to the resolved recording root.")
    parser.add_argument("--force", action="store_true", help="Overwrite conflicting non-empty attrs.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run planning only.")
    args = parser.parse_args(argv)

    result = backfill_clipped_analysis_metadata(
        args.zarr_path,
        recording_root=args.recording_root,
        source_video_path=args.source_video_path,
        registry_path=args.registry,
        rewrite_frame_index_paths=args.rewrite_frame_index_paths,
        old_root=args.old_root,
        new_root=args.new_root,
        force=args.force,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Materialize clip-local raw detections in one recording-ordered quality source.

The source is an immutable, indexed-sharded staging authority for the modern
collection detection-quality stage.  It retains exact clip-local lineage while
rebasing frame indices onto the complete recording timeline.  Publication is
dry-run by default and streams one complete output shard at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fisheye.shared.detect_quality_contract import (
    CLIPPED_DETECT_QUALITY_SOURCE_SCHEMA,
    FULL_FRAME_GEOMETRY_SCHEMA,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_PROVENANCE_ATTR,
    is_run_complete,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA


SOURCE_SCHEMA = CLIPPED_DETECT_QUALITY_SOURCE_SCHEMA
SOURCE_FAMILY = "detect_collection_sources"
DEFAULT_SHARD_ROWS = 131_072
DEFAULT_INNER_ROWS = 16_384


@dataclass(frozen=True)
class ClipSource:
    clip_id: str
    clip_index: int
    camera_serial: str
    detect_run: str
    group_path: str
    group: Any
    source_video_width: int
    source_video_height: int
    parent_frames: np.ndarray
    start: int
    stop: int

    @property
    def rows(self) -> int:
        return self.stop - self.start


def _safe_name(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or "/" in text or text in {".", ".."}:
        raise ValueError(f"{label} must be a safe single group name.")
    return text


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON document must be an object: {path}")
    return payload


def _group_at(root: Any, path: str) -> Any:
    group = root
    for part in str(path).strip("/").split("/"):
        group = group[part]
    return group


def _positive_int(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _required_source_geometry(group: Any, group_path: str) -> tuple[int, int]:
    width = _positive_int(group.attrs.get("source_video_width"))
    height = _positive_int(group.attrs.get("source_video_height"))
    if width is None or height is None:
        raise ValueError(
            f"Raw detection run {group_path} is missing required full-frame geometry attrs "
            "source_video_width/source_video_height."
        )
    return width, height


def _existing_geometry_pair(
    attrs: Any,
    width_key: str,
    height_key: str,
    *,
    label: str,
) -> tuple[int, int] | None:
    raw_width = attrs.get(width_key)
    raw_height = attrs.get(height_key)
    if raw_width is None and raw_height is None:
        return None
    width = _positive_int(raw_width)
    height = _positive_int(raw_height)
    if width is None or height is None:
        raise ValueError(
            f"Existing {label} geometry is incomplete or non-positive: "
            f"{width_key}={raw_width!r}, {height_key}={raw_height!r}."
        )
    return width, height


def _validate_parent_geometry(
    root: Any,
    *,
    width: int,
    height: int,
) -> None:
    expected = (int(width), int(height))
    candidates: list[tuple[str, tuple[int, int] | None]] = []
    for width_key, height_key in (
        ("width", "height"),
        ("source_video_width", "source_video_height"),
        ("source_full_width", "source_full_height"),
    ):
        candidates.append(
            (
                f"root attrs {width_key}/{height_key}",
                _existing_geometry_pair(
                    root.attrs,
                    width_key,
                    height_key,
                    label="root",
                ),
            )
        )
    metadata = root.attrs.get("source_video_metadata")
    if isinstance(metadata, Mapping):
        candidates.append(
            (
                "root source_video_metadata",
                _existing_geometry_pair(
                    metadata,
                    "width",
                    "height",
                    label="root source_video_metadata",
                ),
            )
        )
    raw = root.get("raw_video")
    if raw is not None:
        for width_key, height_key in (
            ("width", "height"),
            ("source_video_width", "source_video_height"),
            ("video_width", "video_height"),
        ):
            candidates.append(
                (
                    f"raw_video attrs {width_key}/{height_key}",
                    _existing_geometry_pair(
                        raw.attrs,
                        width_key,
                        height_key,
                        label="raw_video",
                    ),
                )
            )
    for label, observed in candidates:
        if observed is not None and observed != expected:
            raise ValueError(
                f"Existing {label} geometry {observed[0]}x{observed[1]} disagrees "
                f"with validated detection geometry {expected[0]}x{expected[1]}."
            )


def _stamp_parent_geometry(
    root: Any,
    *,
    width: int,
    height: int,
    source_group_path: str,
) -> None:
    """Stamp parent compatibility metadata from one validated clip collection."""

    _validate_parent_geometry(root, width=width, height=height)
    metadata = root.attrs.get("source_video_metadata")
    source_video_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    source_video_metadata.update({"width": int(width), "height": int(height)})
    shared_attrs = {
        "width": int(width),
        "height": int(height),
        "source_video_width": int(width),
        "source_video_height": int(height),
        "full_frame_geometry_schema": FULL_FRAME_GEOMETRY_SCHEMA,
        "full_frame_geometry_source": source_group_path,
    }
    root.attrs.update({**shared_attrs, "source_video_metadata": source_video_metadata})
    raw = root.require_group("raw_video")
    raw.attrs.update(shared_attrs)


def _frame_index_path(plan: Mapping[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
    else:
        recording_dir = Path(str(plan.get("recording_dir") or "")).expanduser().resolve()
        manifest_path = recording_dir / "recording_frame_index_manifest.json"
        manifest = _read_json(manifest_path)
        raw = Path(str(manifest.get("recording_frame_index_path") or "recording_frame_index.parquet"))
        path = raw.expanduser().resolve() if raw.is_absolute() else (recording_dir / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical recording frame index is missing: {path}")
    return path


def _frame_maps(
    path: Path,
    units: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    required = (
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
    )
    table = pq.read_table(path, columns=list(required)).combine_chunks()
    camera_col = pc.cast(table["camera_serial"], "string")
    clip_col = pc.cast(table["clip_id"], "string")
    maps: dict[tuple[str, str], np.ndarray] = {}
    complete: list[np.ndarray] = []
    for unit in sorted(
        units,
        key=lambda row: (int(row.get("clip_index") or 0), str(row.get("camera_serial") or "")),
    ):
        key = (str(unit.get("camera_serial") or ""), str(unit.get("clip_id") or ""))
        if not all(key) or key in maps:
            raise ValueError(f"Invalid or duplicate clip/camera pair in plan: {key!r}")
        mask = pc.and_(pc.equal(camera_col, key[0]), pc.equal(clip_col, key[1]))
        local = np.asarray(
            pc.filter(table["clip_local_frame_index"], mask).to_numpy(), dtype=np.int64
        )
        parent = np.asarray(
            pc.filter(table["parent_frame_index"], mask).to_numpy(), dtype=np.int64
        )
        order = np.argsort(local, kind="stable")
        local = local[order]
        parent = parent[order]
        if not np.array_equal(local, np.arange(local.size, dtype=np.int64)):
            raise ValueError(f"Clip-local frame map is not contiguous for {key!r}.")
        expected = unit.get("frame_count")
        if expected is not None and int(expected) != int(parent.size):
            raise ValueError(f"Frame-count mismatch for {key!r}: {expected} != {parent.size}.")
        maps[key] = parent
        complete.append(parent)
    recording_frames = np.concatenate(complete) if complete else np.empty((0,), dtype=np.int64)
    if not np.array_equal(recording_frames, np.arange(recording_frames.size, dtype=np.int64)):
        raise ValueError("Selected clips do not cover one complete ordered parent-frame timeline.")
    return maps, int(recording_frames.size)


def _effective_shard_rows(shard_rows: int, inner_rows: int) -> int:
    if shard_rows <= 0 or inner_rows <= 0:
        raise ValueError("shard_rows and inner_rows must be positive.")
    return int(math.ceil(shard_rows / inner_rows) * inner_rows)


def _build_sources(
    root: Any,
    units: Sequence[Mapping[str, Any]],
    frame_maps: Mapping[tuple[str, str], np.ndarray],
) -> list[ClipSource]:
    sources: list[ClipSource] = []
    cursor = 0
    bbox_contract: tuple[np.dtype[Any], tuple[int, ...]] | None = None
    geometry_contract: tuple[int, int] | None = None
    for unit in sorted(
        units,
        key=lambda row: (int(row.get("clip_index") or 0), str(row.get("camera_serial") or "")),
    ):
        paths = unit.get("zarr_paths")
        names = unit.get("run_names")
        if not isinstance(paths, Mapping) or not isinstance(names, Mapping):
            raise ValueError("Detection plan work unit is missing run/path mappings.")
        group_path = str(paths.get("detect_target_group_path") or "")
        group = _group_at(root, group_path)
        if not is_run_complete(group):
            raise ValueError(f"Raw detection run is incomplete: {group_path}")
        missing = [name for name in ("frame_indices", "bbox_norm_coords", "instance_key") if name not in group]
        if missing:
            raise ValueError(f"Raw detection run {group_path} is missing arrays: {missing}")
        rows = int(group["frame_indices"].shape[0])
        if tuple(group["bbox_norm_coords"].shape) != (rows, 4):
            raise ValueError(f"Raw detection bbox shape is invalid: {group_path}")
        if tuple(group["instance_key"].shape) != (rows,):
            raise ValueError(f"Raw detection instance_key shape is invalid: {group_path}")
        if np.dtype(group["instance_key"].dtype) != np.dtype(np.uint64):
            raise ValueError(f"Raw detection instance_key must be uint64: {group_path}")
        current_contract = (np.dtype(group["bbox_norm_coords"].dtype), (4,))
        if bbox_contract is None:
            bbox_contract = current_contract
        elif current_contract != bbox_contract:
            raise ValueError("Clip-local bbox arrays do not share one dtype/shape contract.")
        source_video_width, source_video_height = _required_source_geometry(
            group,
            group_path,
        )
        current_geometry = (source_video_width, source_video_height)
        if geometry_contract is None:
            geometry_contract = current_geometry
        elif current_geometry != geometry_contract:
            raise ValueError(
                "Clip-local raw detection runs disagree on full-frame geometry: "
                f"expected {geometry_contract[0]}x{geometry_contract[1]}, observed "
                f"{current_geometry[0]}x{current_geometry[1]} at {group_path}."
            )
        key = (str(unit.get("camera_serial") or ""), str(unit.get("clip_id") or ""))
        parent_frames = frame_maps[key]
        sources.append(
            ClipSource(
                clip_id=key[1],
                clip_index=int(unit.get("clip_index") or 0),
                camera_serial=key[0],
                detect_run=str(names.get("detect") or ""),
                group_path=group_path,
                group=group,
                source_video_width=source_video_width,
                source_video_height=source_video_height,
                parent_frames=parent_frames,
                start=cursor,
                stop=cursor + rows,
            )
        )
        cursor += rows
    return sources


def _source_segments(
    sources: Sequence[ClipSource], start: int, stop: int
) -> list[tuple[ClipSource, int, int]]:
    result: list[tuple[ClipSource, int, int]] = []
    for source in sources:
        overlap_start = max(start, source.start)
        overlap_stop = min(stop, source.stop)
        if overlap_start < overlap_stop:
            result.append(
                (source, overlap_start - source.start, overlap_stop - source.start)
            )
    if sum(local_stop - local_start for _, local_start, local_stop in result) != stop - start:
        raise RuntimeError(f"Could not resolve complete source coverage for rows {start}:{stop}.")
    return result


def _read_values(
    sources: Sequence[ClipSource], start: int, stop: int, name: str
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for source, local_start, local_stop in _source_segments(sources, start, stop):
        if name == "frame_indices":
            local = np.asarray(
                source.group["frame_indices"][local_start:local_stop], dtype=np.int64
            )
            if local.size and (int(local.min()) < 0 or int(local.max()) >= source.parent_frames.size):
                raise ValueError(f"Detection frame index is outside clip bounds: {source.group_path}")
            pieces.append(source.parent_frames[local])
        elif name == "source_clip_indices":
            pieces.append(np.full(local_stop - local_start, source.clip_index, dtype=np.int32))
        elif name == "source_clip_local_frame_indices":
            pieces.append(
                np.asarray(source.group["frame_indices"][local_start:local_stop], dtype=np.int32)
            )
        elif name == "source_clip_detect_row_index":
            pieces.append(np.arange(local_start, local_stop, dtype=np.int32))
        else:
            pieces.append(np.asarray(source.group[name][local_start:local_stop]))
    return np.concatenate(pieces, axis=0)


def _digest_array(array: Any, *, block_rows: int) -> str:
    digest = hashlib.sha256()
    for start in range(0, int(array.shape[0]), block_rows):
        values = np.asarray(array[start : min(start + block_rows, int(array.shape[0]))])
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def materialize_clipped_detect_quality_source(
    zarr_path: str | Path,
    *,
    plan_path: str | Path,
    output_run: str,
    recording_frame_index: str | Path | None = None,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    inner_rows: int = DEFAULT_INNER_ROWS,
    apply: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    run_name = _safe_name(output_run, label="output_run")
    plan = _read_json(plan_file)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported clipped detection plan: {plan_file}")
    planned_archive = Path(str(plan.get("analysis_zarr") or "")).expanduser().resolve()
    if planned_archive != archive:
        raise ValueError(f"Detection plan targets {planned_archive}, not {archive}.")
    raw_units = plan.get("work_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("Detection plan has no work units.")
    units = [row for row in raw_units if isinstance(row, Mapping)]
    if len(units) != len(raw_units):
        raise ValueError("Detection plan contains a non-object work unit.")
    frame_index = _frame_index_path(
        plan,
        Path(recording_frame_index) if recording_frame_index is not None else None,
    )
    frame_maps, frame_count = _frame_maps(frame_index, units)
    root = open_zarr_root(archive, mode="r")
    sources = _build_sources(root, units, frame_maps)
    source_video_width = sources[0].source_video_width
    source_video_height = sources[0].source_video_height
    row_count = sum(source.rows for source in sources)
    outer = _effective_shard_rows(int(shard_rows), int(inner_rows))
    slices = [
        {
            "clip_id": source.clip_id,
            "clip_index": source.clip_index,
            "camera_serial": source.camera_serial,
            "detect_run": source.detect_run,
            "detect_group_path": source.group_path,
            "source_video_width": source.source_video_width,
            "source_video_height": source.source_video_height,
            "start": source.start,
            "stop": source.stop,
            "row_count": source.rows,
        }
        for source in sources
    ]
    result: dict[str, Any] = {
        "status": "planned" if not apply else "running",
        "schema": SOURCE_SCHEMA,
        "zarr_path": str(archive),
        "plan_path": str(plan_file),
        "recording_frame_index": str(frame_index),
        "output_run": run_name,
        "output_group_path": f"{SOURCE_FAMILY}/{run_name}",
        "recording_frame_count": frame_count,
        "source_video_width": source_video_width,
        "source_video_height": source_video_height,
        "row_count": row_count,
        "source_slices": slices,
        "shard_rows": outer,
        "inner_rows": int(inner_rows),
        "promote": bool(promote),
    }
    if not apply:
        return result

    write_root = open_zarr_root(archive, mode="a")
    parent = require_runs_parent(write_root, SOURCE_FAMILY)
    if run_name in parent:
        raise ValueError(f"{SOURCE_FAMILY}/{run_name} already exists.")
    target = parent.create_group(run_name)
    mark_run_started(target, run_name=run_name, stage="detect_quality_source")
    if promote:
        note_pending_latest(parent, run_name)
    try:
        target.attrs.update(
            {
                "schema_id": SOURCE_SCHEMA,
                "artifact_mutability": "immutable_snapshot",
                "source_detect_collection_id": run_name,
                "source_detect_collection_path": f"{SOURCE_FAMILY}/{run_name}",
                "source_workflow_id": str(plan.get("workflow_id") or ""),
                "source_plan_path": str(plan_file),
                "recording_frame_index_path": str(frame_index),
                "recording_frame_count": frame_count,
                "source_video_width": source_video_width,
                "source_video_height": source_video_height,
                "width": source_video_width,
                "height": source_video_height,
                "full_frame_geometry_schema": FULL_FRAME_GEOMETRY_SCHEMA,
                "full_frame_geometry_source": "validated_source_detect_runs",
                "source_row_count": row_count,
                "frame_index_semantics": "recording_parent_frame_index_0_based",
                "row_identity": "instance_key",
                "storage_layout": "indexed_sharding_v1",
                "row_shard_rows": outer,
                "row_chunk_rows": int(inner_rows),
                "source_slices": slices,
            }
        )
        specs = {
            "frame_indices": (np.dtype(np.int64), ()),
            "bbox_norm_coords": (np.dtype(sources[0].group["bbox_norm_coords"].dtype), (4,)),
            "instance_key": (np.dtype(np.uint64), ()),
            "source_clip_indices": (np.dtype(np.int32), ()),
            "source_clip_local_frame_indices": (np.dtype(np.int32), ()),
            "source_clip_detect_row_index": (np.dtype(np.int32), ()),
        }
        arrays: dict[str, Any] = {}
        for name, (dtype, trailing) in specs.items():
            arrays[name] = target.create_array(
                name,
                shape=(row_count, *trailing),
                dtype=dtype,
                chunks=(int(inner_rows), *trailing),
                shards=(outer, *trailing),
            )
        source_hashes = {name: hashlib.sha256() for name in specs}
        for start in range(0, row_count, outer):
            stop = min(start + outer, row_count)
            for name, array in arrays.items():
                values = _read_values(sources, start, stop, name)
                array[start:stop] = values
                source_hashes[name].update(np.ascontiguousarray(values).view(np.uint8))
        digests = {name: digest.hexdigest() for name, digest in source_hashes.items()}
        for name, array in arrays.items():
            observed = _digest_array(array, block_rows=outer)
            if observed != digests[name]:
                raise RuntimeError(f"Decoded output digest mismatch for {name}.")
        keys = np.asarray(arrays["instance_key"][:], dtype=np.uint64)
        if int(np.unique(keys).size) != row_count:
            raise RuntimeError("Recording-order quality source instance_key values are not unique.")
        frames = np.asarray(arrays["frame_indices"][:], dtype=np.int64)
        if frames.size > 1 and np.any(np.diff(frames) < 0):
            raise RuntimeError("Recording-order quality source frames are not monotonic.")
        provenance = build_writer_run_provenance(
            command="fisheye.utils.materialize_clipped_detect_quality_source",
            params={
                "output_run": run_name,
                "shard_rows": outer,
                "inner_rows": int(inner_rows),
                "source_video_width": source_video_width,
                "source_video_height": source_video_height,
                "promote": bool(promote),
            },
            input_run_ids={"detect_runs": slices, "plan_path": str(plan_file)},
            cwd=Path.cwd(),
        )
        target.attrs[RUN_PROVENANCE_ATTR] = provenance
        target.attrs["decoded_array_sha256"] = digests
        target.attrs["source_validation"] = {
            "status": "complete",
            "row_count": row_count,
            "recording_frame_count": frame_count,
            "source_slice_count": len(slices),
            "source_video_width": source_video_width,
            "source_video_height": source_video_height,
            "full_frame_geometry_uniform": True,
            "instance_key_unique": True,
            "frames_recording_ordered": True,
            "arrays_indexed_sharded": True,
        }
        _stamp_parent_geometry(
            write_root,
            width=source_video_width,
            height=source_video_height,
            source_group_path=f"{SOURCE_FAMILY}/{run_name}",
        )
        mark_run_complete(
            target,
            parent_group=parent if promote else None,
            run_name=run_name,
            run_provenance=provenance,
        )
        result.update({"status": "complete", "decoded_array_sha256": digests})
        return result
    except Exception as exc:
        mark_run_failed(
            target,
            parent_group=parent if promote else None,
            run_name=run_name,
            error=str(exc),
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--recording-frame-index", type=Path, default=None)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--inner-rows", type=int, default=DEFAULT_INNER_ROWS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = materialize_clipped_detect_quality_source(
        args.zarr_path,
        plan_path=args.plan,
        output_run=args.output_run,
        recording_frame_index=args.recording_frame_index,
        shard_rows=args.shard_rows,
        inner_rows=args.inner_rows,
        apply=args.apply,
        promote=not args.no_promote,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

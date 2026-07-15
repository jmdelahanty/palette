#!/usr/bin/env python3
"""Publish a finalized clipped detection collection as one recording-level run.

The selected per-clip runs remain immutable source evidence.  The publisher
streams their canonical ``instances`` and ``source_detections`` surfaces into
one indexed-sharded ``refined_detect_runs/<run>`` snapshot, rebases frame and
row identities into recording-level domains, validates the written values, and
only then promotes the new run.  Default operation is a read-only dry run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_PROVENANCE_ATTR,
    is_run_complete,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
)
from fisheye.utils.validate_refined_detect_run import validate_refined_detect_run


SNAPSHOT_SCHEMA = "palette.refined_detect_collection_snapshot.v1"
DEFAULT_SHARD_ROWS = 131_072
DEFAULT_ROW_CHUNK = 1_024
DEFAULT_LINEAGE_CHUNK = 16_384

INSTANCE_COPY_ARRAYS = (
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "class_ids",
    "confidence_scores",
    "instance_key",
    "instance_key_origin_codes",
    "manual_edit_flags",
    "reason_bytes",
    "source_kind_codes",
)
SOURCE_COPY_ARRAYS = (
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "class_ids",
    "confidence_scores",
    "decision_codes",
    "instance_key",
    "reason_bytes",
)


@dataclass(frozen=True)
class OutputArrayPlan:
    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]
    shards: tuple[int, ...]
    source: str


@dataclass
class ClipContext:
    clip_id: str
    clip_index: int
    camera_serial: str
    refined_group_path: str
    refined: Any
    instances: Any
    source_detections: Any
    parent_frames: np.ndarray
    recording_frame_ids: np.ndarray
    instance_offset: int = 0
    source_offset: int = 0
    source_local_ids_sorted: np.ndarray | None = None
    source_local_positions: np.ndarray | None = None
    refined_local_ids_sorted: np.ndarray | None = None
    refined_local_positions: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        return int(self.parent_frames.shape[0])

    @property
    def instance_count(self) -> int:
        return int(self.instances["frame_indices"].shape[0])

    @property
    def source_count(self) -> int:
        return int(self.source_detections["source_detect_row_index"].shape[0])


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _group_at(root: Any, group_path: str) -> Any:
    group = root
    for part in str(group_path).strip("/").split("/"):
        group = group[part]
    return group


def _resolve_collection(root: Any, collection_id: str | None) -> tuple[str, str, Any]:
    parent = root.get("refined_detect_runs")
    if parent is None:
        raise ValueError("Archive is missing refined_detect_runs.")
    resolved = str(collection_id or parent.attrs.get("latest_collection") or "").strip()
    if not resolved or "/" in resolved:
        raise ValueError("No safe finalized clipped collection id was provided or selected.")
    path = f"experiment_index/finalized_runs/{resolved}"
    try:
        collection = _group_at(root, path)
    except Exception as exc:
        raise ValueError(f"Finalized clipped collection does not exist: {path}") from exc
    if str(collection.attrs.get("schema_version") or "") != (
        "palette.refined_detect_clip_collection.v1"
    ):
        raise ValueError(f"Unsupported clipped collection schema at {path}.")
    return resolved, path, collection


def _resolve_frame_index_path(
    root: Any,
    collection: Any,
    explicit: str | Path | None,
) -> Path:
    candidates = [
        explicit,
        root.attrs.get("recording_frame_index_path"),
        root.attrs.get("source_recording_frame_index_path"),
        (collection.attrs.get("recording_frame_index_validation") or {}).get("path"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(str(candidate)).expanduser().resolve()
            if path.is_file():
                return path
    raise ValueError("A readable canonical recording_frame_index.parquet is required.")


def _frame_maps(
    frame_index_path: Path,
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], int]:
    required = (
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
        "recording_frame_id",
    )
    table = pq.read_table(frame_index_path, columns=list(required)).combine_chunks()
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"Recording frame index is missing columns: {missing}.")

    cameras = sorted({str(item.get("camera_serial") or "") for item in selected})
    if len(cameras) != 1 or not cameras[0]:
        raise ValueError(
            "Recording-level refined-detection snapshot v1 requires exactly one camera."
        )

    camera_col = pc.cast(table["camera_serial"], "string")
    clip_col = pc.cast(table["clip_id"], "string")
    maps: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    all_parent: list[np.ndarray] = []
    ordered = sorted(
        selected,
        key=lambda item: (
            int(item.get("clip_index") or 0),
            str(item.get("camera_serial") or ""),
        ),
    )
    for item in ordered:
        camera = str(item.get("camera_serial") or "")
        clip = str(item.get("clip_id") or "")
        key = (camera, clip)
        if key in maps:
            raise ValueError(f"Collection repeats camera/clip work unit {key}.")
        mask = pc.and_(pc.equal(camera_col, camera), pc.equal(clip_col, clip))
        local = np.asarray(
            pc.filter(table["clip_local_frame_index"], mask).to_numpy(), dtype=np.int64
        )
        parent = np.asarray(
            pc.filter(table["parent_frame_index"], mask).to_numpy(), dtype=np.int64
        )
        recording_ids = np.asarray(
            pc.filter(table["recording_frame_id"], mask).to_numpy(), dtype=np.int64
        )
        if local.size == 0:
            raise ValueError(f"Recording frame index has no rows for {key}.")
        order = np.argsort(local, kind="stable")
        local = local[order]
        parent = parent[order]
        recording_ids = recording_ids[order]
        if not np.array_equal(local, np.arange(local.shape[0], dtype=np.int64)):
            raise ValueError(f"Clip-local frame mapping is not contiguous for {key}.")
        expected_count = item.get("frame_count")
        if expected_count is not None and int(expected_count) != int(local.shape[0]):
            raise ValueError(
                f"Collection frame_count disagrees with frame index for {key}: "
                f"{expected_count} != {local.shape[0]}."
            )
        if np.any(parent < 0) or np.any(recording_ids <= 0):
            raise ValueError(f"Recording-level frame identities are invalid for {key}.")
        maps[key] = (parent, recording_ids)
        all_parent.append(parent)

    concatenated_parent = np.concatenate(all_parent)
    if int(np.unique(concatenated_parent).shape[0]) != int(concatenated_parent.shape[0]):
        raise ValueError("Selected clips repeat parent_frame_index values.")
    if not np.array_equal(
        concatenated_parent,
        np.arange(concatenated_parent.shape[0], dtype=np.int64),
    ):
        raise ValueError(
            "Selected clips must cover the complete recording parent-frame timeline "
            "in collection order."
        )
    return maps, int(concatenated_parent.shape[0])


def _require_complete(group: Any, *, path: str) -> None:
    if not is_run_complete(group):
        raise ValueError(f"Selected source refined-detection run is incomplete: {path}.")


def _require_source_array(group: Any, name: str, *, label: str) -> Any:
    if name not in group:
        raise ValueError(f"{label} is missing required canonical array {name}.")
    return group[name]


def _same_array_contract(arrays: Sequence[Any], *, label: str) -> tuple[np.dtype[Any], tuple[int, ...]]:
    first = arrays[0]
    dtype = np.dtype(first.dtype)
    trailing = tuple(int(value) for value in first.shape[1:])
    for array in arrays[1:]:
        if np.dtype(array.dtype) != dtype or tuple(int(value) for value in array.shape[1:]) != trailing:
            raise ValueError(f"Selected clips disagree on array contract for {label}.")
    return dtype, trailing


def _prepare_local_identity(context: ClipContext) -> None:
    source_ids = np.asarray(
        context.source_detections["source_detect_row_index"][:], dtype=np.int64
    ).reshape(-1)
    if int(np.unique(source_ids).shape[0]) != int(source_ids.shape[0]):
        raise ValueError(
            f"{context.refined_group_path}/source_detections has duplicate source row ids."
        )
    source_order = np.argsort(source_ids, kind="stable")
    context.source_local_ids_sorted = source_ids[source_order]
    context.source_local_positions = source_order.astype(np.int64, copy=False)

    refined_ids = np.asarray(
        context.instances["refined_row_ids"][:], dtype=np.int64
    ).reshape(-1)
    if np.any(refined_ids < 0) or int(np.unique(refined_ids).shape[0]) != int(
        refined_ids.shape[0]
    ):
        raise ValueError(
            f"{context.refined_group_path}/instances has invalid local refined row ids."
        )
    refined_order = np.argsort(refined_ids, kind="stable")
    context.refined_local_ids_sorted = refined_ids[refined_order]
    context.refined_local_positions = refined_order.astype(np.int64, copy=False)


def _map_local_ids(
    values: np.ndarray,
    *,
    sorted_ids: np.ndarray | None,
    sorted_positions: np.ndarray | None,
    offset: int,
    label: str,
    allow_negative: bool,
) -> np.ndarray:
    if sorted_ids is None or sorted_positions is None:
        raise RuntimeError(f"Local identity lookup was not prepared for {label}.")
    requested = np.asarray(values, dtype=np.int64).reshape(-1)
    result = np.full(requested.shape, -1, dtype=np.int64)
    valid = requested >= 0
    if not allow_negative and not bool(np.all(valid)):
        raise ValueError(f"{label} contains negative local identities.")
    positions = np.searchsorted(sorted_ids, requested[valid])
    in_range = positions < sorted_ids.shape[0]
    matched = np.zeros(positions.shape, dtype=bool)
    if bool(np.any(in_range)):
        matched[in_range] = sorted_ids[positions[in_range]] == requested[valid][in_range]
    if not bool(np.all(matched)):
        missing = requested[valid][~matched]
        raise ValueError(
            f"{label} references unknown local identities, e.g. "
            f"{[int(value) for value in missing[:5].tolist()]}"
        )
    result[valid] = int(offset) + sorted_positions[positions]
    return result


def _build_contexts(
    root: Any,
    selected: Sequence[Mapping[str, Any]],
    maps: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    prepare_identity: bool,
) -> list[ClipContext]:
    ordered = sorted(
        selected,
        key=lambda item: (int(item.get("clip_index") or 0), str(item.get("camera_serial") or "")),
    )
    contexts: list[ClipContext] = []
    instance_offset = 0
    source_offset = 0
    for item in ordered:
        path = str(item.get("refined_group_path") or "").strip("/")
        refined = _group_at(root, path)
        _require_complete(refined, path=path)
        if "instances" not in refined or "source_detections" not in refined:
            raise ValueError(f"Selected refined run lacks canonical surfaces: {path}.")
        instances = refined["instances"]
        source = refined["source_detections"]
        for name in (*INSTANCE_COPY_ARRAYS, "refined_row_ids", "frame_counts", "frame_offsets", "frame_indices", "source_detect_row_index"):
            _require_source_array(instances, name, label=f"{path}/instances")
        for name in (*SOURCE_COPY_ARRAYS, "frame_indices", "resolved_refined_row_id", "source_detect_row_index"):
            _require_source_array(source, name, label=f"{path}/source_detections")
        key = (str(item.get("camera_serial") or ""), str(item.get("clip_id") or ""))
        parent, recording_ids = maps[key]
        context = ClipContext(
            clip_id=key[1],
            clip_index=int(item.get("clip_index") or 0),
            camera_serial=key[0],
            refined_group_path=path,
            refined=refined,
            instances=instances,
            source_detections=source,
            parent_frames=parent,
            recording_frame_ids=recording_ids,
            instance_offset=instance_offset,
            source_offset=source_offset,
        )
        if int(instances["frame_counts"].shape[0]) != context.frame_count:
            raise ValueError(f"{path}/instances/frame_counts does not cover its clip.")
        if int(instances["frame_offsets"].shape[0]) != context.frame_count + 1:
            raise ValueError(f"{path}/instances/frame_offsets has an invalid length.")
        if prepare_identity:
            _prepare_local_identity(context)
        contexts.append(context)
        instance_offset += context.instance_count
        source_offset += context.source_count
    return contexts


def _aligned_shards(chunks: tuple[int, ...], shard_rows: int) -> tuple[int, ...]:
    outer_rows = int(math.ceil(int(shard_rows) / int(chunks[0])) * int(chunks[0]))
    return (outer_rows, *chunks[1:])


def _array_plan(
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | str,
    inner_rows: int,
    shard_rows: int,
    source: str,
) -> OutputArrayPlan:
    chunks = (max(1, min(int(inner_rows), max(1, int(shape[0])))), *shape[1:])
    return OutputArrayPlan(
        path=path,
        shape=shape,
        dtype=str(np.dtype(dtype)),
        chunks=chunks,
        shards=_aligned_shards(chunks, int(shard_rows)),
        source=source,
    )


def _build_array_plans(
    contexts: Sequence[ClipContext],
    *,
    recording_frame_count: int,
    shard_rows: int,
) -> tuple[OutputArrayPlan, ...]:
    instance_count = sum(context.instance_count for context in contexts)
    source_count = sum(context.source_count for context in contexts)
    if source_count > int(np.iinfo(np.int32).max):
        raise ValueError(
            "Recording-level source detection row count exceeds the current int32 "
            "refined-detection identity contract."
        )
    plans: list[OutputArrayPlan] = []
    for name in INSTANCE_COPY_ARRAYS:
        arrays = [context.instances[name] for context in contexts]
        dtype, trailing = _same_array_contract(arrays, label=f"instances/{name}")
        plans.append(
            _array_plan(
                path=f"instances/{name}",
                shape=(instance_count, *trailing),
                dtype=dtype,
                inner_rows=(DEFAULT_LINEAGE_CHUNK if name.startswith("instance_key") else DEFAULT_ROW_CHUNK),
                shard_rows=shard_rows,
                source=f"concatenate selected instances/{name}",
            )
        )
    for name, dtype in (
        ("refined_row_ids", np.int64),
        ("source_refined_row_ids", np.int64),
        ("source_detect_row_index", np.int32),
        ("source_clip_detect_row_index", np.int64),
        ("frame_indices", np.int64),
        ("source_frame_indices", np.int64),
        ("source_recording_frame_ids", np.int64),
        ("source_clip_indices", np.int64),
        ("source_clip_local_frame_indices", np.int64),
    ):
        plans.append(
            _array_plan(
                path=f"instances/{name}",
                shape=(instance_count,),
                dtype=dtype,
                inner_rows=DEFAULT_LINEAGE_CHUNK,
                shard_rows=shard_rows,
                source="recording-level identity rebase",
            )
        )
    plans.extend(
        (
            _array_plan(
                path="instances/frame_counts",
                shape=(recording_frame_count,),
                dtype=np.int32,
                inner_rows=DEFAULT_LINEAGE_CHUNK,
                shard_rows=shard_rows,
                source="concatenate clip frame_counts on parent timeline",
            ),
            _array_plan(
                path="instances/frame_offsets",
                shape=(recording_frame_count + 1,),
                dtype=np.int64,
                inner_rows=DEFAULT_LINEAGE_CHUNK,
                shard_rows=shard_rows,
                source="cumulative sum of recording frame_counts",
            ),
        )
    )

    for name in SOURCE_COPY_ARRAYS:
        arrays = [context.source_detections[name] for context in contexts]
        dtype, trailing = _same_array_contract(arrays, label=f"source_detections/{name}")
        plans.append(
            _array_plan(
                path=f"source_detections/{name}",
                shape=(source_count, *trailing),
                dtype=dtype,
                inner_rows=(DEFAULT_LINEAGE_CHUNK if name == "instance_key" else DEFAULT_ROW_CHUNK),
                shard_rows=shard_rows,
                source=f"concatenate selected source_detections/{name}",
            )
        )
    for name, dtype in (
        ("source_detect_row_index", np.int32),
        ("source_clip_detect_row_index", np.int64),
        ("resolved_refined_row_id", np.int64),
        ("source_resolved_refined_row_id", np.int64),
        ("frame_indices", np.int64),
        ("source_frame_indices", np.int64),
        ("source_recording_frame_ids", np.int64),
        ("source_clip_indices", np.int64),
        ("source_clip_local_frame_indices", np.int64),
    ):
        plans.append(
            _array_plan(
                path=f"source_detections/{name}",
                shape=(source_count,),
                dtype=dtype,
                inner_rows=DEFAULT_LINEAGE_CHUNK,
                shard_rows=shard_rows,
                source="recording-level identity rebase",
            )
        )
    return tuple(sorted(plans, key=lambda plan: plan.path))


def _create_array(
    parent: Any,
    name: str,
    plan: OutputArrayPlan,
    *,
    template: Any | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "shape": plan.shape,
        "dtype": np.dtype(plan.dtype),
        "chunks": plan.chunks,
        "shards": plan.shards,
        "overwrite": True,
    }
    if template is not None:
        fill_value = getattr(template, "fill_value", None)
        if fill_value is not None:
            kwargs["fill_value"] = fill_value
        compressors = getattr(template, "compressors", None)
        if compressors:
            kwargs["compressors"] = compressors
        filters = getattr(template, "filters", None)
        if filters:
            kwargs["filters"] = filters
        serializer = getattr(template, "serializer", None)
        if serializer is not None:
            kwargs["serializer"] = serializer
    return parent.create_array(name, **kwargs)


def _digest_block(digest: Any, values: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(values).view(np.uint8))


def _write_concatenated(
    destination: Any,
    contexts: Sequence[ClipContext],
    *,
    domain: str,
    loader: Callable[[ClipContext, int, int], np.ndarray],
) -> dict[str, Any]:
    total = int(destination.shape[0])
    trailing = tuple(int(value) for value in destination.shape[1:])
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    outer_rows = int(destination.shards[0])
    if domain == "instances":
        offsets = [int(context.instance_offset) for context in contexts]
        counts = [int(context.instance_count) for context in contexts]
    elif domain == "source_detections":
        offsets = [int(context.source_offset) for context in contexts]
        counts = [int(context.source_count) for context in contexts]
    elif domain == "frames":
        counts = [int(context.frame_count) for context in contexts]
        offsets = []
        running_offset = 0
        for count in counts:
            offsets.append(running_offset)
            running_offset += count
    else:
        raise ValueError(f"Unsupported concatenation domain: {domain}.")
    for start in range(0, total, outer_rows):
        stop = min(start + outer_rows, total)
        buffer = np.empty((stop - start, *trailing), dtype=np.dtype(destination.dtype))
        filled = 0
        for context, offset, count in zip(contexts, offsets, counts, strict=True):
            overlap_start = max(start, offset)
            overlap_stop = min(stop, offset + count)
            if overlap_start >= overlap_stop:
                continue
            local_start = overlap_start - offset
            local_stop = overlap_stop - offset
            values = np.asarray(loader(context, local_start, local_stop), dtype=destination.dtype)
            expected_shape = (local_stop - local_start, *trailing)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Loader produced {values.shape}, expected {expected_shape} for "
                    f"{getattr(destination, 'path', '<array>')}."
                )
            target_start = overlap_start - start
            target_stop = overlap_stop - start
            buffer[target_start:target_stop] = values
            filled += int(overlap_stop - overlap_start)
        if filled != stop - start:
            raise RuntimeError(
                f"Concatenation did not fill destination rows {start}:{stop}; filled={filled}."
            )
        _digest_block(source_digest, buffer)
        destination[start:stop, ...] = buffer
        reread = np.asarray(destination[start:stop, ...])
        _digest_block(destination_digest, reread)
    source_sha256 = source_digest.hexdigest()
    destination_sha256 = destination_digest.hexdigest()
    if source_sha256 != destination_sha256:
        raise RuntimeError(
            f"Decoded digest mismatch for {getattr(destination, 'path', '<array>')}."
        )
    return {
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "exact": True,
    }


def _frame_lookup(
    context: ClipContext,
    local_frames: np.ndarray,
    *,
    recording_ids: bool,
) -> np.ndarray:
    local = np.asarray(local_frames, dtype=np.int64).reshape(-1)
    if local.size and (int(local.min()) < 0 or int(local.max()) >= context.frame_count):
        raise ValueError(
            f"{context.refined_group_path} contains frames outside its clip mapping."
        )
    source = context.recording_frame_ids if recording_ids else context.parent_frames
    return source[local]


def _loader_for_instances(name: str) -> Callable[[ClipContext, int, int], np.ndarray]:
    if name in INSTANCE_COPY_ARRAYS:
        return lambda context, start, stop: np.asarray(context.instances[name][start:stop, ...])
    if name == "refined_row_ids":
        return lambda context, start, stop: context.instance_offset + np.arange(start, stop, dtype=np.int64)
    if name == "source_refined_row_ids":
        return lambda context, start, stop: np.asarray(
            context.instances["refined_row_ids"][start:stop], dtype=np.int64
        )
    if name == "source_clip_detect_row_index":
        return lambda context, start, stop: np.asarray(
            context.instances["source_detect_row_index"][start:stop], dtype=np.int64
        )
    if name == "source_detect_row_index":
        return lambda context, start, stop: _map_local_ids(
            np.asarray(context.instances["source_detect_row_index"][start:stop], dtype=np.int64),
            sorted_ids=context.source_local_ids_sorted,
            sorted_positions=context.source_local_positions,
            offset=context.source_offset,
            label=f"{context.refined_group_path}/instances/source_detect_row_index",
            allow_negative=True,
        ).astype(np.int32)
    if name in {"frame_indices", "source_frame_indices"}:
        return lambda context, start, stop: _frame_lookup(
            context,
            np.asarray(context.instances["frame_indices"][start:stop], dtype=np.int64),
            recording_ids=False,
        )
    if name == "source_recording_frame_ids":
        return lambda context, start, stop: _frame_lookup(
            context,
            np.asarray(context.instances["frame_indices"][start:stop], dtype=np.int64),
            recording_ids=True,
        )
    if name == "source_clip_indices":
        return lambda context, start, stop: np.full(stop - start, context.clip_index, dtype=np.int64)
    if name == "source_clip_local_frame_indices":
        return lambda context, start, stop: np.asarray(
            context.instances["frame_indices"][start:stop], dtype=np.int64
        )
    raise KeyError(name)


def _loader_for_source(name: str) -> Callable[[ClipContext, int, int], np.ndarray]:
    if name in SOURCE_COPY_ARRAYS:
        return lambda context, start, stop: np.asarray(
            context.source_detections[name][start:stop, ...]
        )
    if name == "source_detect_row_index":
        return lambda context, start, stop: (
            context.source_offset + np.arange(start, stop, dtype=np.int64)
        ).astype(np.int32)
    if name == "source_clip_detect_row_index":
        return lambda context, start, stop: np.asarray(
            context.source_detections["source_detect_row_index"][start:stop], dtype=np.int64
        )
    if name == "resolved_refined_row_id":
        return lambda context, start, stop: _map_local_ids(
            np.asarray(
                context.source_detections["resolved_refined_row_id"][start:stop], dtype=np.int64
            ),
            sorted_ids=context.refined_local_ids_sorted,
            sorted_positions=context.refined_local_positions,
            offset=context.instance_offset,
            label=f"{context.refined_group_path}/source_detections/resolved_refined_row_id",
            allow_negative=True,
        )
    if name == "source_resolved_refined_row_id":
        return lambda context, start, stop: np.asarray(
            context.source_detections["resolved_refined_row_id"][start:stop], dtype=np.int64
        )
    if name in {"frame_indices", "source_frame_indices"}:
        return lambda context, start, stop: _frame_lookup(
            context,
            np.asarray(context.source_detections["frame_indices"][start:stop], dtype=np.int64),
            recording_ids=False,
        )
    if name == "source_recording_frame_ids":
        return lambda context, start, stop: _frame_lookup(
            context,
            np.asarray(context.source_detections["frame_indices"][start:stop], dtype=np.int64),
            recording_ids=True,
        )
    if name == "source_clip_indices":
        return lambda context, start, stop: np.full(stop - start, context.clip_index, dtype=np.int64)
    if name == "source_clip_local_frame_indices":
        return lambda context, start, stop: np.asarray(
            context.source_detections["frame_indices"][start:stop], dtype=np.int64
        )
    raise KeyError(name)


def _write_frame_counts(
    destination: Any,
    contexts: Sequence[ClipContext],
) -> dict[str, Any]:
    return _write_concatenated(
        destination,
        contexts,
        domain="frames",
        loader=lambda context, start, stop: np.asarray(
            context.instances["frame_counts"][start:stop], dtype=np.int32
        ),
    )


def _write_frame_offsets(destination: Any, frame_counts: Any) -> dict[str, Any]:
    total_offsets = int(destination.shape[0])
    outer_rows = int(destination.shards[0])
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    running = 0
    for start in range(0, total_offsets, outer_rows):
        stop = min(start + outer_rows, total_offsets)
        buffer = np.empty(stop - start, dtype=np.int64)
        if start == 0:
            buffer[0] = 0
            if stop > 1:
                counts = np.asarray(frame_counts[0 : stop - 1], dtype=np.int64)
                buffer[1:] = np.cumsum(counts, dtype=np.int64)
        else:
            counts = np.asarray(frame_counts[start - 1 : stop - 1], dtype=np.int64)
            buffer[:] = running + np.cumsum(counts, dtype=np.int64)
        running = int(buffer[-1])
        _digest_block(source_digest, buffer)
        destination[start:stop] = buffer
        reread = np.asarray(destination[start:stop], dtype=np.int64)
        _digest_block(destination_digest, reread)
    source_sha256 = source_digest.hexdigest()
    destination_sha256 = destination_digest.hexdigest()
    if source_sha256 != destination_sha256:
        raise RuntimeError("Decoded digest mismatch for instances/frame_offsets.")
    return {
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "exact": True,
        "final_offset": int(running),
    }


def _copy_schema_attrs(contexts: Sequence[ClipContext], target: Any) -> None:
    instances = target.require_group("instances")
    source = target.require_group("source_detections")
    for destination, source_group in (
        (instances, contexts[0].instances),
        (source, contexts[0].source_detections),
    ):
        for name, value in dict(source_group.attrs).items():
            if name.startswith("instance_key_backfill_"):
                continue
            destination.attrs[name] = value
        destination.attrs.update(
            {
                "reason_authority": "reason_bytes",
                "reason_fallback_order": ["reason_bytes", "detection_source"],
                "reason_encoding": "utf8-null-terminated",
                "reason_bytes_null_terminated": True,
                "row_frame_domain": "recording_parent_frame_index",
                "source_frame_indices_semantics": "recording_parent_frame_index",
                "source_recording_frame_ids_semantics": (
                    "session_continuous_recording_frame_id_1_based"
                ),
                "source_clip_local_frame_indices_semantics": (
                    "zero_based_frame_index_within_selected_clip"
                ),
            }
        )
    instances.attrs.update(
        {
            "row_sort_order": ["frame_indices", "refined_row_ids"],
            "refined_row_ids_scope": "recording_snapshot",
            "source_refined_row_ids_scope": "source_clip_refined_run",
            "source_detect_row_index_scope": "recording_snapshot_source_projection",
            "source_clip_detect_row_index_scope": "source_clip_detect_run",
            "instance_key_frame_domain": "recording_parent_frame_index",
        }
    )
    source.attrs.update(
        {
            "source_detect_row_index_scope": "recording_snapshot_source_projection",
            "source_clip_detect_row_index_scope": "source_clip_detect_run",
            "resolved_refined_row_id_scope": "recording_snapshot",
            "source_resolved_refined_row_id_scope": "source_clip_refined_run",
        }
    )


def _template_for_plan(contexts: Sequence[ClipContext], path: str) -> Any | None:
    group_name, name = path.split("/", 1)
    if group_name == "instances" and name in INSTANCE_COPY_ARRAYS:
        return contexts[0].instances[name]
    if group_name == "source_detections" and name in SOURCE_COPY_ARRAYS:
        return contexts[0].source_detections[name]
    if path == "instances/frame_counts":
        return contexts[0].instances["frame_counts"]
    if path == "instances/frame_offsets":
        return contexts[0].instances["frame_offsets"]
    return None


def publish_clipped_refined_detect_snapshot(
    *,
    zarr_path: str | Path,
    output_run: str,
    collection_id: str | None = None,
    recording_frame_index: str | Path | None = None,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    apply: bool = False,
    promote: bool = True,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    if not output_run or "/" in output_run:
        raise ValueError("output_run must be a safe single group name.")
    if int(shard_rows) <= 0:
        raise ValueError("shard_rows must be positive.")
    root = open_zarr_root(archive, mode="a" if apply else "r")
    resolved_collection, collection_path, collection = _resolve_collection(root, collection_id)
    selected = [
        dict(item)
        for item in list(collection.attrs.get("selected_runs") or [])
        if isinstance(item, Mapping)
    ]
    if not selected:
        raise ValueError(f"Finalized collection has no selected runs: {collection_path}.")
    frame_index_path = _resolve_frame_index_path(root, collection, recording_frame_index)
    maps, recording_frame_count = _frame_maps(frame_index_path, selected)
    contexts = _build_contexts(root, selected, maps, prepare_identity=bool(apply))
    plans = _build_array_plans(
        contexts,
        recording_frame_count=recording_frame_count,
        shard_rows=int(shard_rows),
    )
    instance_count = sum(context.instance_count for context in contexts)
    source_count = sum(context.source_count for context in contexts)
    result: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "planned" if not apply else "running",
        "zarr_path": str(archive),
        "collection_id": resolved_collection,
        "collection_path": collection_path,
        "output_run": output_run,
        "output_run_path": f"refined_detect_runs/{output_run}",
        "recording_frame_index": str(frame_index_path),
        "recording_frame_index_sha256": _sha256_file(frame_index_path),
        "recording_frame_count": recording_frame_count,
        "instance_count": instance_count,
        "source_detection_count": source_count,
        "selected_run_count": len(contexts),
        "selected_run_paths": [context.refined_group_path for context in contexts],
        "shard_rows": int(shard_rows),
        "promote": bool(promote),
        "arrays": [asdict(plan) for plan in plans],
        "omitted_legacy_array_paths": [
            f"{group_name}/reason" for group_name in ("instances", "source_detections")
        ],
    }
    if not apply:
        return result

    parent = root.require_group("refined_detect_runs")
    if output_run in parent:
        raise ValueError(f"Output run already exists: refined_detect_runs/{output_run}.")
    target = parent.create_group(output_run)
    mark_run_started(target, run_name=output_run, stage="refined_detect_collection_snapshot")
    if promote:
        note_pending_latest(parent, output_run)
    try:
        _copy_schema_attrs(contexts, target)
        target.attrs.update(
            {
                "snapshot_schema": SNAPSHOT_SCHEMA,
                "artifact_mutability": "immutable_snapshot",
                "snapshot_source_collection": resolved_collection,
                "snapshot_source_collection_path": collection_path,
                "source_detect_collection_id": resolved_collection,
                "source_detect_collection_path": collection_path,
                "source_detect_path": collection_path,
                "source_detect_run": resolved_collection,
                "source_quality_run": resolved_collection,
                "refined_family_path": "refined_detect_runs",
                "curated_primary_surface": "instances",
                "row_identity_policy": "stable_sparse_refined_row_id",
                "reason_authority": "reason_bytes",
                "reason_fallback_order": ["reason_bytes", "detection_source"],
                "tabular_storage_layout": "indexed_sharding_v1",
                "tabular_shard_rows": int(shard_rows),
                "recording_frame_index_path": str(frame_index_path),
                "recording_frame_index_sha256": _sha256_file(frame_index_path),
                "frame_indices_semantics": "recording_parent_frame_index_0_based",
                "source_recording_frame_ids_semantics": (
                    "session_continuous_recording_frame_id_1_based"
                ),
                "selected_source_refined_run_paths": [
                    context.refined_group_path for context in contexts
                ],
                "summary_statistics": {
                    "recording_frames": recording_frame_count,
                    "instances": instance_count,
                    "source_detections": source_count,
                    "source_clip_runs": len(contexts),
                },
                "coverage_comparison": {
                    "recording_frame_count": recording_frame_count,
                    "instance_count": instance_count,
                    "source_detection_count": source_count,
                    "complete_parent_frame_timeline": True,
                },
            }
        )
        instances = target["instances"]
        source_detections = target["source_detections"]
        destinations: dict[str, Any] = {}
        for plan in plans:
            group_name, name = plan.path.split("/", 1)
            destination_parent = instances if group_name == "instances" else source_detections
            destinations[plan.path] = _create_array(
                destination_parent,
                name,
                plan,
                template=_template_for_plan(contexts, plan.path),
            )

        validation: list[dict[str, Any]] = []
        for path in sorted(destinations):
            destination = destinations[path]
            group_name, name = path.split("/", 1)
            if path == "instances/frame_counts":
                item = _write_frame_counts(destination, contexts)
            elif path == "instances/frame_offsets":
                item = _write_frame_offsets(destination, destinations["instances/frame_counts"])
                if int(item["final_offset"]) != instance_count:
                    raise RuntimeError(
                        "Recording-level frame counts do not sum to the instance row count."
                    )
            elif group_name == "instances":
                item = _write_concatenated(
                    destination,
                    contexts,
                    domain="instances",
                    loader=_loader_for_instances(name),
                )
            else:
                item = _write_concatenated(
                    destination,
                    contexts,
                    domain="source_detections",
                    loader=_loader_for_source(name),
                )
            validation.append({"path": path, **item})

        keys = np.asarray(instances["instance_key"][:], dtype=np.uint64)
        if int(np.unique(keys).shape[0]) != instance_count:
            raise RuntimeError("Published recording-level instance_key values are not unique.")
        if not np.array_equal(
            np.asarray(instances["source_frame_indices"][:], dtype=np.int64),
            np.asarray(instances["frame_indices"][:], dtype=np.int64),
        ):
            raise RuntimeError("Published source_frame_indices disagree with parent frame_indices.")

        target.attrs["tabular_snapshot_validation"] = {
            "status": "complete",
            "array_count": len(validation),
            "sharded_array_count": len(validation),
            "decoded_exact": True,
            "instance_key_unique": True,
            "complete_parent_frame_timeline": True,
        }
        provenance = build_writer_run_provenance(
            command="fisheye.utils.publish_clipped_refined_detect_snapshot",
            params={
                "zarr_path": str(archive),
                "collection_id": resolved_collection,
                "output_run": output_run,
                "recording_frame_index": str(frame_index_path),
                "shard_rows": int(shard_rows),
                "promote": bool(promote),
            },
            input_run_ids={
                "source_collection": collection_path,
                "selected_refined_runs": [context.refined_group_path for context in contexts],
            },
            cwd=Path.cwd(),
        )
        target.attrs[RUN_PROVENANCE_ATTR] = provenance

        contract_validation = validate_refined_detect_run(
            archive,
            target_group_path=f"refined_detect_runs/{output_run}",
        )
        if contract_validation.get("status") != "ok":
            raise RuntimeError(
                "Published recording-level refined-detection contract validation failed: "
                f"{contract_validation.get('errors')}"
            )
        target.attrs["collection_snapshot_completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        mark_run_complete(
            target,
            parent_group=parent if promote else None,
            run_name=output_run,
            run_provenance=provenance,
        )
        result.update(
            {
                "status": "complete",
                "validation": validation,
                "contract_validation": contract_validation,
            }
        )
        return result
    except Exception as exc:
        mark_run_failed(
            target,
            parent_group=parent if promote else None,
            run_name=output_run,
            error=str(exc),
        )
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--collection-id", default=None)
    parser.add_argument("--recording-frame-index", type=Path, default=None)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = publish_clipped_refined_detect_snapshot(
        zarr_path=args.zarr_path,
        output_run=args.output_run,
        collection_id=args.collection_id,
        recording_frame_index=args.recording_frame_index,
        shard_rows=int(args.shard_rows),
        apply=bool(args.apply),
        promote=not bool(args.no_promote),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: {result['collection_path']} -> "
            f"{result['output_run_path']} rows={result['instance_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

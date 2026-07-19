#!/usr/bin/env python3
"""Publish an immutable keypoint snapshot from one base and keyed replacements."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from fisheye.shared.keyed_replacement import (
    REPLACEMENT_SOURCE_BASE,
    KeyedReplacementPlan,
    build_keyed_replacement_plan,
)
from fisheye.shared.row_lineage import ROW_IDENTITY_ARRAYS, ROW_IDENTITY_MODE_SCHEMA
from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.run_provenance import (
    build_writer_run_provenance,
    json_ready,
    stable_json,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr.chunk_profiles import (
    create_geometry_preload_array,
    geometry_preload_chunks_for_shape,
)
from fisheye.shared.zarr.columnar import pick_shards
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_PROVENANCE_ATTR,
    is_run_complete_in_parent,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from fisheye.utils.finalize_keypoint_shards import (
    COMPAT_ATTRS,
    COPY_IF_CONSISTENT_ATTRS,
    DEFAULT_CANONICAL_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_CANONICAL_KEYPOINT_ROW_SHARD_ROWS,
    GEOMETRY_PRELOAD_ROW_ARRAYS,
    KEYPOINT_OUTPUT_PARENT,
    KEYPOINT_SHARD_PARENT,
    OPTIONAL_ROW_ARRAYS,
    REQUIRED_ROW_ARRAYS,
    SOURCE_CROP_ROW_IDS_ARRAY,
)


KEYPOINT_COMPACTION_SCHEMA_ID = "palette.keypoint_keyed_compaction"
KEYPOINT_COMPACTION_SCHEMA_VERSION = 1
KEYPOINT_COMPACTION_METHOD = "fisheye.utils.compact_keypoint_deltas"
DEFAULT_COPY_BATCH_ROWS = 131_072

_TARGET_LINEAGE_ARRAYS = frozenset(
    {
        "frame_indices",
        SOURCE_CROP_ROW_IDS_ARRAY,
        "detection_indices",
        "instance_key",
        "source_frame_indices",
        "source_clip_indices",
        "source_clip_local_frame_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
        "detection_source",
    }
)
_PREDICTION_ARRAYS = tuple(
    name
    for name in REQUIRED_ROW_ARRAYS
    if name not in _TARGET_LINEAGE_ARRAYS and name != "heading_usable"
)
_SEMANTIC_ATTRS = tuple(
    dict.fromkeys(
        (
            *COMPAT_ATTRS,
            "model_input_transform",
            "model_input_transform_name",
            "model_input_shape_hw",
            "native_roi_shape_hw",
        )
    )
)
_SEMANTIC_PARAMETER_NAMES = (
    "confidence_threshold",
    "iou_threshold",
    "max_det",
    "imgsz",
    "pose_schema",
    "n_keypoints",
    "model_kpt_shape",
    "input_mode_requested",
    "input_mode_effective",
    "model_input_transform",
)


class KeypointCompactionError(RuntimeError):
    """Raised when exact keypoint snapshot publication cannot be proven safe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array(group: Any, name: str, *, dtype: Any | None = None) -> np.ndarray:
    values = np.asarray(group[name][:])
    return values.astype(dtype, copy=False) if dtype is not None else values


def _require_complete(parent: Any, name: str, *, label: str) -> Any:
    if name not in parent:
        raise KeypointCompactionError(f"{label} {name!r} does not exist.")
    group = parent[name]
    if not is_run_complete_in_parent(parent, group):
        raise KeypointCompactionError(f"{label} {name!r} is not complete.")
    return group


def _require_unique_keys(group: Any, *, label: str, row_count: int) -> np.ndarray:
    if "instance_key" not in group:
        raise KeypointCompactionError(f"{label} is missing instance_key.")
    array = group["instance_key"]
    if np.dtype(array.dtype) != np.dtype(np.uint64) or tuple(array.shape) != (row_count,):
        raise KeypointCompactionError(
            f"{label}/instance_key must be uint64[{row_count}], got {array.dtype}{array.shape}."
        )
    keys = _array(group, "instance_key", dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise KeypointCompactionError(f"{label}/instance_key is not unique.")
    return keys


def _source_signatures(root: Any, run_group: Any, *, label: str) -> tuple[np.ndarray, str]:
    crop_name = str(run_group.attrs.get("source_crop_run") or "").strip()
    crop_parent = root.get("crop_runs")
    if not crop_name or crop_parent is None:
        raise KeypointCompactionError(f"{label} has no resolvable source_crop_run.")
    crop = _require_complete(crop_parent, crop_name, label="Source crop run")
    crop_rows = int(crop["instance_key"].shape[0]) if "instance_key" in crop else -1
    _require_unique_keys(crop, label=f"crop_runs/{crop_name}", row_count=crop_rows)
    if ROW_SOURCE_SIGNATURE_ARRAY not in crop:
        raise KeypointCompactionError(
            f"crop_runs/{crop_name} lacks {ROW_SOURCE_SIGNATURE_ARRAY}; keyed reuse is unsafe."
        )
    validate_row_source_signature_array(crop[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=crop_rows)
    source_rows = _array(run_group, SOURCE_CROP_ROW_IDS_ARRAY, dtype=np.int64).reshape(-1)
    if source_rows.size and (source_rows.min() < 0 or source_rows.max() >= crop_rows):
        raise KeypointCompactionError(f"{label} references an out-of-range crop row.")
    run_keys = _require_unique_keys(run_group, label=label, row_count=source_rows.shape[0])
    crop_keys = _array(crop, "instance_key", dtype=np.uint64).reshape(-1)
    if not np.array_equal(run_keys, crop_keys[source_rows]):
        raise KeypointCompactionError(f"{label} keys do not exactly match its source crop rows.")
    signatures = np.asarray(crop[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8)[source_rows]
    return signatures, load_row_source_signature_spec(crop.attrs).spec_digest


def _semantic_value(group: Any, name: str) -> object:
    return group.attrs.get(name, None)


def _input_artifact_sha256(group: Any, *, role: str, label: str) -> str:
    fingerprints: set[str] = set()
    for attr_name in ("run_provenance", "cli_provenance"):
        provenance = group.attrs.get(attr_name)
        if not isinstance(provenance, Mapping):
            continue
        artifacts = provenance.get("input_artifacts")
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or str(artifact.get("role") or "") != role:
                continue
            sha256 = str(artifact.get("sha256") or "").strip().lower()
            if len(sha256) == 64:
                fingerprints.add(sha256)
    if len(fingerprints) != 1:
        raise KeypointCompactionError(
            f"{label} must expose exactly one {role!r} SHA-256 fingerprint; found {len(fingerprints)}."
        )
    return next(iter(fingerprints))


def _semantic_sources(root: Any, base: Any) -> list[Any]:
    raw = base.attrs.get("source_keypoint_shard_runs")
    if not isinstance(raw, (list, tuple)) or not raw:
        return [base]
    parent = root.get(KEYPOINT_SHARD_PARENT)
    if parent is None:
        raise KeypointCompactionError("Base keypoint semantic source shards are unavailable.")
    groups: list[Any] = []
    for value in raw:
        name = str(value).split("/")[-1]
        groups.append(_require_complete(parent, name, label="Base keypoint source shard"))
    return groups


def _parameters(group: Any) -> dict[str, object]:
    raw = group.attrs.get("parameters")
    values = raw if isinstance(raw, Mapping) else {}
    return {name: values.get(name) for name in _SEMANTIC_PARAMETER_NAMES}


def _validate_semantics(root: Any, base: Any, replacements: Sequence[Any]) -> str:
    for name in COMPAT_ATTRS:
        if name not in base.attrs:
            raise KeypointCompactionError(f"Base keypoint run lacks compatibility attr {name!r}.")
    base_sources = _semantic_sources(root, base)
    base_fingerprints = {
        _input_artifact_sha256(
            source,
            role="keypoint_model",
            label="Base keypoint semantic source",
        )
        for source in base_sources
    }
    if len(base_fingerprints) != 1:
        raise KeypointCompactionError("Base keypoint source shards used different model fingerprints.")
    base_parameter_payloads = {stable_json(_parameters(source)) for source in base_sources}
    if len(base_parameter_payloads) != 1:
        raise KeypointCompactionError("Base keypoint source shards used different inference parameters.")
    base_parameter_payload = next(iter(base_parameter_payloads))
    model_sha256 = next(iter(base_fingerprints))
    for index, replacement in enumerate(replacements):
        for name in _SEMANTIC_ATTRS:
            base_value = _semantic_value(base, name)
            replacement_value = _semantic_value(replacement, name)
            if name in COMPAT_ATTRS and name not in replacement.attrs:
                raise KeypointCompactionError(
                    f"Replacement {index} lacks compatibility attr {name!r}."
                )
            if stable_json(base_value) != stable_json(replacement_value):
                raise KeypointCompactionError(
                    f"Keypoint semantic attr {name!r} differs between base and replacement {index}."
                )
        if _input_artifact_sha256(
            replacement,
            role="keypoint_model",
            label=f"Replacement {index}",
        ) != model_sha256:
            raise KeypointCompactionError(
                f"Keypoint model fingerprint differs for replacement {index}."
            )
        if stable_json(_parameters(replacement)) != base_parameter_payload:
            raise KeypointCompactionError(
                f"Keypoint inference parameters differ for replacement {index}."
            )
    return model_sha256


def _validate_row_array_contract(base: Any, replacements: Sequence[Any]) -> None:
    for name in _PREDICTION_ARRAYS:
        if name not in base:
            raise KeypointCompactionError(f"Base keypoint run lacks {name!r}.")
        expected_shape = tuple(int(value) for value in base[name].shape[1:])
        expected_dtype = np.dtype(base[name].dtype)
        for index, replacement in enumerate(replacements):
            if name not in replacement:
                raise KeypointCompactionError(f"Replacement {index} lacks {name!r}.")
            array = replacement[name]
            if tuple(int(value) for value in array.shape[1:]) != expected_shape or np.dtype(array.dtype) != expected_dtype:
                raise KeypointCompactionError(
                    f"Replacement {index}/{name} shape or dtype differs from the base."
                )


def _resolve_replacements(root: Any, names: Sequence[str], target_crop: Any, target_crop_name: str) -> list[Any]:
    parent = root.get(KEYPOINT_SHARD_PARENT)
    if names and parent is None:
        raise KeypointCompactionError(f"Missing {KEYPOINT_SHARD_PARENT}.")
    if len(set(names)) != len(names):
        raise KeypointCompactionError("Duplicate replacement run names were supplied.")
    groups: list[Any] = []
    target_keys = _array(target_crop, "instance_key", dtype=np.uint64).reshape(-1)
    target_rows = target_keys.shape[0]
    for name in names:
        group = _require_complete(parent, str(name), label="Replacement keypoint run")
        if normalize_attr(group.attrs.get("incremental_materialization_role")) != "delta_replacement_rows":
            raise KeypointCompactionError(
                f"{KEYPOINT_SHARD_PARENT}/{name} is not a delta_replacement_rows run."
            )
        if str(group.attrs.get("source_crop_run") or "") != target_crop_name:
            raise KeypointCompactionError(
                f"{KEYPOINT_SHARD_PARENT}/{name} is not bound to crop_runs/{target_crop_name}."
            )
        row_count = int(group["instance_key"].shape[0]) if "instance_key" in group else -1
        keys = _require_unique_keys(group, label=f"{KEYPOINT_SHARD_PARENT}/{name}", row_count=row_count)
        if SOURCE_CROP_ROW_IDS_ARRAY not in group:
            raise KeypointCompactionError(f"{KEYPOINT_SHARD_PARENT}/{name} lacks source_crop_row_ids.")
        rows = _array(group, SOURCE_CROP_ROW_IDS_ARRAY, dtype=np.int64).reshape(-1)
        if rows.shape != (row_count,) or (rows.size and (rows.min() < 0 or rows.max() >= target_rows)):
            raise KeypointCompactionError(f"{KEYPOINT_SHARD_PARENT}/{name} has invalid target crop rows.")
        if not np.array_equal(keys, target_keys[rows]):
            raise KeypointCompactionError(
                f"{KEYPOINT_SHARD_PARENT}/{name} keys do not match the target crop rows."
            )
        if ROW_SOURCE_SIGNATURE_ARRAY not in group:
            raise KeypointCompactionError(
                f"{KEYPOINT_SHARD_PARENT}/{name} lacks the signed crop rows used for inference."
            )
        validate_row_source_signature_array(
            group[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=row_count
        )
        replacement_spec = load_row_source_signature_spec(group.attrs)
        target_spec = load_row_source_signature_spec(target_crop.attrs)
        replacement_signatures = _array(
            group, ROW_SOURCE_SIGNATURE_ARRAY, dtype=np.uint8
        )
        target_signatures = _array(
            target_crop, ROW_SOURCE_SIGNATURE_ARRAY, dtype=np.uint8
        )
        if (
            replacement_spec.spec_digest != target_spec.spec_digest
            or not np.array_equal(replacement_signatures, target_signatures[rows])
        ):
            raise KeypointCompactionError(
                f"{KEYPOINT_SHARD_PARENT}/{name} was inferred from stale or incompatible crop rows."
            )
        groups.append(group)
    return groups


def _read_rows_into(source: Any, source_rows: np.ndarray, output: np.ndarray, output_rows: np.ndarray) -> None:
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    destinations = np.asarray(output_rows, dtype=np.int64).reshape(-1)
    if rows.shape != destinations.shape:
        raise KeypointCompactionError("Source and destination row maps differ.")
    if not rows.size:
        return
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    destinations = destinations[order]
    start = 0
    while start < rows.shape[0]:
        stop = start + 1
        while stop < rows.shape[0] and rows[stop] == rows[stop - 1] + 1:
            stop += 1
        first = int(rows[start])
        last = int(rows[stop - 1]) + 1
        values = np.asarray(source[first:last])
        output[destinations[start:stop]] = values[rows[start:stop] - first]
        start = stop


def _gather_mapped_rows(
    name: str,
    *,
    base: Any,
    replacements: Sequence[Any],
    plan: KeyedReplacementPlan,
    start: int,
    stop: int,
) -> np.ndarray:
    source = base[name]
    output = np.empty((stop - start, *tuple(int(v) for v in source.shape[1:])), dtype=source.dtype)
    run_indices = plan.source_run_indices[start:stop]
    source_rows = plan.source_row_indices[start:stop]
    base_positions = np.flatnonzero(run_indices == REPLACEMENT_SOURCE_BASE)
    _read_rows_into(source, source_rows[base_positions], output, base_positions)
    for run_index, replacement in enumerate(replacements):
        positions = np.flatnonzero(run_indices == run_index)
        _read_rows_into(replacement[name], source_rows[positions], output, positions)
    return output


def _chunks_for(source: Any, shape: tuple[int, ...], *, geometry: bool) -> tuple[int, ...] | None:
    if not shape:
        return None
    if geometry:
        return geometry_preload_chunks_for_shape(shape)
    raw = getattr(source, "chunks", None)
    if raw:
        return tuple(max(1, min(int(shape[i]), int(value))) for i, value in enumerate(raw))
    return (max(1, min(1024, shape[0])), *tuple(max(1, value) for value in shape[1:]))


def _create_empty_like(
    group: Any,
    name: str,
    source: Any,
    *,
    row_count: int,
    geometry: bool,
    shard_rows: int,
) -> Any:
    shape = (int(row_count), *tuple(int(value) for value in source.shape[1:]))
    chunks = _chunks_for(source, shape, geometry=geometry)
    shards = pick_shards(shape, chunks, shard_rows=int(shard_rows)) if chunks else None
    kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": source.dtype,
        "overwrite": True,
    }
    if chunks is not None:
        kwargs["chunks"] = chunks
    if shards is not None:
        kwargs["shards"] = shards
    if geometry:
        return create_geometry_preload_array(group, name, **kwargs)
    return group.create_array(name, **kwargs)


def _write_batched(array: Any, values: Any, *, batch_rows: int) -> str:
    digest = hashlib.sha256()
    row_count = int(array.shape[0])
    for start in range(0, row_count, int(batch_rows)):
        stop = min(start + int(batch_rows), row_count)
        batch = np.ascontiguousarray(values(start, stop))
        array[start:stop] = batch
        persisted = np.ascontiguousarray(array[start:stop])
        if not np.array_equal(batch, persisted, equal_nan=True):
            raise KeypointCompactionError("Keypoint output readback differs from the written batch.")
        digest.update(persisted.view(np.uint8))
    return digest.hexdigest()


def _snapshot_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def compact_keypoint_deltas(
    *,
    zarr_path: str | Path,
    base_run: str,
    target_crop_run: str,
    replacement_runs: Sequence[str] = (),
    output_run: str | None = None,
    dry_run: bool = False,
    row_shard_rows: int = DEFAULT_CANONICAL_KEYPOINT_ROW_SHARD_ROWS,
    frame_shard_rows: int = DEFAULT_CANONICAL_KEYPOINT_FRAME_SHARD_ROWS,
    before_publish: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Create one complete immutable keypoint snapshot in target-crop order."""

    archive = Path(zarr_path).expanduser().resolve()
    root = open_zarr_root(archive, mode="r" if dry_run else "a")
    if getattr(getattr(root, "metadata", None), "zarr_format", None) != 3:
        raise KeypointCompactionError("Keyed keypoint compaction requires Zarr v3.")
    keypoint_parent = root.get(KEYPOINT_OUTPUT_PARENT)
    crop_parent = root.get("crop_runs")
    if keypoint_parent is None or crop_parent is None:
        raise KeypointCompactionError("Archive lacks keypoints_runs or crop_runs.")
    base = _require_complete(keypoint_parent, str(base_run), label="Base keypoint run")
    target = _require_complete(crop_parent, str(target_crop_run), label="Target crop run")
    target_rows = int(target["instance_key"].shape[0]) if "instance_key" in target else -1
    target_keys = _require_unique_keys(target, label=f"crop_runs/{target_crop_run}", row_count=target_rows)
    if ROW_SOURCE_SIGNATURE_ARRAY not in target:
        raise KeypointCompactionError("Target crop lacks signed source rows.")
    validate_row_source_signature_array(target[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=target_rows)
    target_signatures = _array(target, ROW_SOURCE_SIGNATURE_ARRAY, dtype=np.uint8)
    target_spec = load_row_source_signature_spec(target.attrs)
    for required in ("frame_indices", "frame_counts", "detection_indices", "detection_source"):
        if required not in target:
            raise KeypointCompactionError(f"Target crop lacks required lineage array {required!r}.")

    base_rows = int(base["instance_key"].shape[0]) if "instance_key" in base else -1
    base_keys = _require_unique_keys(base, label=f"{KEYPOINT_OUTPUT_PARENT}/{base_run}", row_count=base_rows)
    if SOURCE_CROP_ROW_IDS_ARRAY not in base:
        raise KeypointCompactionError("Base keypoint run lacks source_crop_row_ids.")
    base_signatures, base_spec_digest = _source_signatures(
        root, base, label=f"{KEYPOINT_OUTPUT_PARENT}/{base_run}"
    )
    replacements = _resolve_replacements(root, replacement_runs, target, str(target_crop_run))
    model_sha256 = _validate_semantics(root, base, replacements)
    _validate_row_array_contract(base, replacements)
    replacement_keys = [
        _array(group, "instance_key", dtype=np.uint64).reshape(-1) for group in replacements
    ]
    plan = build_keyed_replacement_plan(
        target_instance_keys=target_keys,
        target_source_signatures=target_signatures,
        target_signature_spec_digest=target_spec.spec_digest,
        base_instance_keys=base_keys,
        base_source_signatures=base_signatures,
        base_signature_spec_digest=base_spec_digest,
        replacement_instance_keys=replacement_keys,
    )
    output_name = output_run or f"keypoints_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not output_name or "/" in output_name:
        raise KeypointCompactionError("output_run must be one non-empty group name.")
    result: dict[str, Any] = {
        "ok": True,
        "status": "would_write" if dry_run else "written",
        "zarr_path": str(archive),
        "output_run": output_name,
        "output_path": f"{KEYPOINT_OUTPUT_PARENT}/{output_name}",
        "base_run": str(base_run),
        "target_crop_run": str(target_crop_run),
        "replacement_runs": [str(value) for value in replacement_runs],
        "plan": plan.summary(),
        "storage": {
            "layout": "indexed_sharding_v1",
            "row_shard_rows": int(row_shard_rows),
            "frame_shard_rows": int(frame_shard_rows),
            "write_mode": "bounded_whole_outer_shards",
        },
    }
    if dry_run:
        return result

    parent = require_runs_parent(root, KEYPOINT_OUTPUT_PARENT, completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE)
    if output_name in parent:
        raise KeypointCompactionError(f"{KEYPOINT_OUTPUT_PARENT}/{output_name} already exists.")
    expected_parent_state = {
        "latest": parent.attrs.get("latest"),
        RUN_LATEST_COMPLETE_ATTR: parent.attrs.get(RUN_LATEST_COMPLETE_ATTR),
    }
    target_lineage_names = tuple(
        name
        for name in (*sorted(_TARGET_LINEAGE_ARRAYS), "frame_counts")
        if name != SOURCE_CROP_ROW_IDS_ARRAY and name in target
    )
    target_lineage_snapshot = [np.asarray(target[name][:]) for name in target_lineage_names]
    source_digest = _snapshot_digest(
        target_keys,
        target_signatures,
        *target_lineage_snapshot,
        base_keys,
        base_signatures,
        *replacement_keys,
    )
    run_group = parent.create_group(output_name)
    mark_run_started(run_group, run_name=output_name, stage="keypoints")
    note_pending_latest(parent, output_name)
    started = time.perf_counter()
    try:
        digests: dict[str, str] = {}
        output_arrays: dict[str, Any] = {}
        for name in _PREDICTION_ARRAYS:
            output_arrays[name] = _create_empty_like(
                run_group,
                name,
                base[name],
                row_count=target_rows,
                geometry=name in GEOMETRY_PRELOAD_ROW_ARRAYS,
                shard_rows=int(row_shard_rows),
            )
            physical_rows = int((output_arrays[name].shards or output_arrays[name].chunks)[0])
            digests[name] = _write_batched(
                output_arrays[name],
                lambda start, stop, n=name: _gather_mapped_rows(
                    n, base=base, replacements=replacements, plan=plan, start=start, stop=stop
                ),
                batch_rows=physical_rows,
            )

        lineage_names = [
            name
            for name in dict.fromkeys((*REQUIRED_ROW_ARRAYS, *OPTIONAL_ROW_ARRAYS, *ROW_IDENTITY_ARRAYS))
            if name in _TARGET_LINEAGE_ARRAYS and name != SOURCE_CROP_ROW_IDS_ARRAY and name in target
        ]
        for name in lineage_names:
            source = target[name]
            output = _create_empty_like(
                run_group,
                name,
                source,
                row_count=target_rows,
                geometry=True,
                shard_rows=int(row_shard_rows),
            )
            physical_rows = int((output.shards or output.chunks)[0])
            digests[name] = _write_batched(
                output,
                lambda start, stop, s=source: np.asarray(s[start:stop]),
                batch_rows=physical_rows,
            )
        source_crop_rows = np.arange(target_rows, dtype=np.int64)
        crop_ids_source = base[SOURCE_CROP_ROW_IDS_ARRAY]
        crop_ids_output = _create_empty_like(
            run_group,
            SOURCE_CROP_ROW_IDS_ARRAY,
            crop_ids_source,
            row_count=target_rows,
            geometry=True,
            shard_rows=int(row_shard_rows),
        )
        digests[SOURCE_CROP_ROW_IDS_ARRAY] = _write_batched(
            crop_ids_output,
            lambda start, stop: source_crop_rows[start:stop],
            batch_rows=int((crop_ids_output.shards or crop_ids_output.chunks)[0]),
        )

        success = np.asarray(run_group["detection_success"][:], dtype=bool).reshape(-1)
        detection_source = np.asarray(run_group["detection_source"][:], dtype=np.int8).reshape(-1)
        heading_finite = np.asarray(run_group["heading_finite"][:], dtype=bool).reshape(-1)
        heading_usable = success & (detection_source == 0) & heading_finite
        heading_output = _create_empty_like(
            run_group,
            "heading_usable",
            base["heading_usable"],
            row_count=target_rows,
            geometry=False,
            shard_rows=int(row_shard_rows),
        )
        digests["heading_usable"] = _write_batched(
            heading_output,
            lambda start, stop: heading_usable[start:stop],
            batch_rows=int((heading_output.shards or heading_output.chunks)[0]),
        )

        frame_counts = _array(target, "frame_counts", dtype=np.int32).reshape(-1)
        frame_indices = _array(target, "frame_indices", dtype=np.int64).reshape(-1)
        keypoint_count = int(base["keypoints_roi"].shape[1])
        n_keypoints = np.bincount(
            frame_indices,
            weights=success.astype(np.int64) * keypoint_count,
            minlength=frame_counts.shape[0],
        ).astype(np.int32)
        for name, values in (
            ("frame_counts", frame_counts),
            ("n_rois", frame_counts),
            ("n_keypoints", n_keypoints),
        ):
            source = base[name] if name in base else target["frame_counts"]
            output = _create_empty_like(
                run_group,
                name,
                source,
                row_count=values.shape[0],
                geometry=True,
                shard_rows=int(frame_shard_rows),
            )
            digests[name] = _write_batched(
                output,
                lambda start, stop, data=values: data[start:stop],
                batch_rows=int((output.shards or output.chunks)[0]),
            )

        summary_statistics = {
            "total_rois": int(target_rows),
            "keypoint_count": int(keypoint_count),
            "successful_detections": int(np.count_nonzero(success)),
            "failed_detections": int(target_rows - np.count_nonzero(success)),
            "success_rate_percent": round(
                float(np.count_nonzero(success) / target_rows * 100.0) if target_rows else 0.0,
                2,
            ),
            "mean_confidence": float(np.nanmean(np.asarray(run_group["confidence"][:]))) if target_rows else 0.0,
        }
        copied_attrs = {
            name: json_ready(base.attrs.get(name))
            for name in (*COMPAT_ATTRS, *COPY_IF_CONSISTENT_ATTRS)
            if name in base.attrs
        }
        created_at = _utc_now()
        run_group.attrs.update(
            {
                **copied_attrs,
                "schema_id": KEYPOINT_COMPACTION_SCHEMA_ID,
                "schema_version": KEYPOINT_COMPACTION_SCHEMA_VERSION,
                "method": KEYPOINT_COMPACTION_METHOD,
                "run_semantics": "immutable_keypoint_snapshot",
                "artifact_mutability": "raw_immutable",
                "created_at_utc": created_at,
                "keypoints_timestamp_utc": created_at,
                "source_keypoint_base_run": str(base_run),
                "source_keypoint_replacement_runs": [str(value) for value in replacement_runs],
                "source_keypoint_replacement_run_paths": [
                    f"{KEYPOINT_SHARD_PARENT}/{value}" for value in replacement_runs
                ],
                "source_crop_run": str(target_crop_run),
                "source_crop_runs": [str(target_crop_run)],
                "compaction_plan": plan.summary(),
                "source_keypoint_model_sha256": model_sha256,
                "compaction_array_sha256": digests,
                "instance_key_status": "present",
                "row_identity_mode": "instance_key",
                "row_identity_mode_schema": ROW_IDENTITY_MODE_SCHEMA,
                "instance_key_alignment_status": "exact_full_target_crop",
                "keypoints_processed": int(target_rows),
                "summary_statistics": summary_statistics,
                "keypoint_storage_layout": "indexed_sharding_v1",
                "keypoint_storage_policy": "immutable_snapshot_indexed_sharding_v1",
                "keypoint_roi_shard_rows": int(row_shard_rows),
                "keypoint_frame_shard_rows": int(frame_shard_rows),
                "stage_selector_eligible": True,
                "output_parent": KEYPOINT_OUTPUT_PARENT,
                "run_group_parent": KEYPOINT_OUTPUT_PARENT,
            }
        )
        provenance = build_writer_run_provenance(
            command=KEYPOINT_COMPACTION_METHOD,
            params={
                "zarr_path": str(archive),
                "base_run": str(base_run),
                "target_crop_run": str(target_crop_run),
                "replacement_runs": [str(value) for value in replacement_runs],
                "output_run": output_name,
                "row_shard_rows": int(row_shard_rows),
                "frame_shard_rows": int(frame_shard_rows),
            },
            input_run_ids={
                "keypoint_base": str(base_run),
                "keypoint_replacements": [f"{KEYPOINT_SHARD_PARENT}/{value}" for value in replacement_runs],
                "crop": str(target_crop_run),
            },
            input_artifacts=[
                {
                    "role": "keypoint_model",
                    "path": str(base.attrs.get("model_path") or ""),
                    "sha256": model_sha256,
                    "fingerprint_source": "validated_compaction_inputs",
                }
            ],
            cwd=Path.cwd(),
        )
        run_group.attrs[RUN_PROVENANCE_ATTR] = provenance
        if before_publish is not None:
            before_publish()
        refreshed_target_keys = _array(root[f"crop_runs/{target_crop_run}"], "instance_key", dtype=np.uint64)
        refreshed_target_signatures = _array(root[f"crop_runs/{target_crop_run}"], ROW_SOURCE_SIGNATURE_ARRAY, dtype=np.uint8)
        refreshed_base = root[f"{KEYPOINT_OUTPUT_PARENT}/{base_run}"]
        refreshed_target = root[f"crop_runs/{target_crop_run}"]
        refreshed_replacements = _resolve_replacements(
            root,
            replacement_runs,
            refreshed_target,
            str(target_crop_run),
        )
        if _validate_semantics(root, refreshed_base, refreshed_replacements) != model_sha256:
            raise KeypointCompactionError("Keypoint model identity changed before publication.")
        _validate_row_array_contract(refreshed_base, refreshed_replacements)
        refreshed_base_keys = _array(refreshed_base, "instance_key", dtype=np.uint64)
        refreshed_base_signatures, _ = _source_signatures(
            root, refreshed_base, label=f"{KEYPOINT_OUTPUT_PARENT}/{base_run}"
        )
        refreshed_replacement_keys = [
            _array(group, "instance_key", dtype=np.uint64)
            for group in refreshed_replacements
        ]
        if source_digest != _snapshot_digest(
            refreshed_target_keys,
            refreshed_target_signatures,
            *[np.asarray(refreshed_target[name][:]) for name in target_lineage_names],
            refreshed_base_keys,
            refreshed_base_signatures,
            *refreshed_replacement_keys,
        ):
            raise KeypointCompactionError("A compaction input changed before publication.")
        if any(parent.attrs.get(key) != value for key, value in expected_parent_state.items()):
            raise KeypointCompactionError("Keypoint selection state changed before publication.")
        if not np.array_equal(np.asarray(run_group["instance_key"][:], dtype=np.uint64), target_keys):
            raise KeypointCompactionError("Published keypoint keys do not equal the target crop keys.")
        mark_run_complete(
            run_group,
            parent_group=parent,
            run_name=output_name,
            run_provenance=provenance,
        )
        root.attrs["current_keypoint_group_path"] = run_group.path
        duration = time.perf_counter() - started
        run_group.attrs["compaction_duration_seconds"] = float(duration)
        result["compaction_duration_seconds"] = float(duration)
        return result
    except Exception as exc:
        mark_run_failed(run_group, parent_group=parent, run_name=output_name, error=str(exc))
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--target-crop-run", required=True)
    parser.add_argument("--replacement-run", action="append", default=[])
    parser.add_argument("--output-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--row-shard-rows", type=int, default=DEFAULT_CANONICAL_KEYPOINT_ROW_SHARD_ROWS)
    parser.add_argument("--frame-shard-rows", type=int, default=DEFAULT_CANONICAL_KEYPOINT_FRAME_SHARD_ROWS)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = compact_keypoint_deltas(
            zarr_path=args.zarr_path,
            base_run=args.base_run,
            target_crop_run=args.target_crop_run,
            replacement_runs=args.replacement_run,
            output_run=args.output_run,
            dry_run=bool(args.dry_run),
            row_shard_rows=int(args.row_shard_rows),
            frame_shard_rows=int(args.frame_shard_rows),
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['output_path']} ({result['plan']['target_row_count']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

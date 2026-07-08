#!/usr/bin/env python3
"""Import per-clip refined subject-mask packages into one canonical run."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.mask_store import update_mask_storage_attrs
from fisheye.shared.run_provenance import (
    CLI_RUN_PROVENANCE_ATTR,
    RUN_PROVENANCE_ATTR,
    build_run_provenance,
    json_ready,
)
from fisheye.shared.subject_mask_chunks import (
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_chunks,
)
from fisheye.shared.zarr_run_completion import mark_run_complete, require_runs_parent
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID as CLIP_PACKAGE_SCHEMA_ID


IMPORT_SCHEMA_ID = "palette_refined_subject_mask_clip_package_import_v1"
SKIPPED_DERIVED_GROUPS = {"mask_bitpacked", "mask_rle"}
SKIPPED_REGENERATED_ARRAY_NAMES = {"reason_bytes"}
COLLECTION_INHERITED_ATTR_DROP_EXACT = {
    "clip_package_host",
    "clip_package_lsb_jobid",
    "clip_package_source_zarr_path",
    "clip_package_staged_zarr_path",
    "clip_package_subject_shard_run",
    "clip_package_target_crop_run",
    "source_roi_cache_canonical_path",
    "source_roi_cache_key",
    "source_roi_cache_path",
}
COLLECTION_LIST_ATTRS = {
    "clip_package_host": "clip_package_hosts",
    "clip_package_lsb_jobid": "clip_package_lsb_jobids",
    "source_roi_cache_canonical_path": "source_roi_cache_canonical_paths",
    "source_roi_cache_key": "source_roi_cache_keys",
    "source_roi_cache_path": "source_roi_cache_paths",
    "source_subject_mask_shard_crop_runs": "source_subject_mask_shard_crop_runs",
    "source_subject_mask_shard_run_paths": "source_subject_mask_shard_run_paths",
    "source_subject_mask_shard_runs": "source_subject_mask_shard_runs",
}


@dataclass(frozen=True)
class ClipPackage:
    package_path: Path
    extract_dir: Path
    manifest: Mapping[str, Any]
    run_name: str
    group: zarr.Group
    source_crop_row_ids: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.source_crop_row_ids.shape[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_extract_package_tar(tar: tarfile.TarFile, target: Path) -> None:
    target_root = target.resolve()
    members = tar.getmembers()
    for member in members:
        member_path = (target_root / member.name).resolve()
        if member_path != target_root and target_root not in member_path.parents:
            raise ValueError(f"Unsafe package member path escapes extraction root: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing package link member: {member.name!r}")
    tar.extractall(target, members=members)


def _extract_package(package_path: Path, extract_root: Path) -> tuple[Path, Mapping[str, Any]]:
    package_path = package_path.expanduser().resolve()
    if not package_path.is_file():
        raise ValueError(f"Package path is not a file: {package_path}")
    target = extract_root / package_path.stem.replace(".tar", "")
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(package_path, "r:gz") as tar:
        _safe_extract_package_tar(tar, target)
    manifest_path = target / "package.json"
    if not manifest_path.is_file():
        raise ValueError(f"Package {package_path} is missing package.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_id") != CLIP_PACKAGE_SCHEMA_ID:
        raise ValueError(
            f"Package {package_path} has schema_id={manifest.get('schema_id')!r}; "
            f"expected {CLIP_PACKAGE_SCHEMA_ID!r}."
        )
    return target, manifest


def _load_package(package_path: Path, extract_root: Path) -> ClipPackage:
    extract_dir, manifest = _extract_package(package_path, extract_root)
    run_group_path = PurePosixPath(str(manifest.get("run_group_path") or ""))
    if len(run_group_path.parts) != 2 or run_group_path.parts[0] != "refined_subject_masks_runs":
        raise ValueError(f"Package {package_path} has invalid run_group_path={str(run_group_path)!r}.")
    run_name = run_group_path.parts[1]
    group_path = extract_dir / run_group_path.parts[0] / run_name
    if not group_path.is_dir():
        raise ValueError(f"Package {package_path} is missing run group {run_group_path}.")
    group = zarr.open_group(str(group_path), mode="r", use_consolidated=False)
    if "masks_roi" not in group:
        raise ValueError(f"Package {package_path} run {run_group_path} is missing dense masks_roi.")
    if "source_crop_row_ids" not in group:
        raise ValueError(f"Package {package_path} run {run_group_path} is missing source_crop_row_ids.")
    source_crop_row_ids = np.asarray(group["source_crop_row_ids"][:], dtype=np.int64).reshape(-1)
    row_count = int(group["masks_roi"].shape[0])
    if int(source_crop_row_ids.shape[0]) != row_count:
        raise ValueError(
            f"Package {package_path} source_crop_row_ids has {int(source_crop_row_ids.shape[0])} rows "
            f"but masks_roi has {row_count} rows."
        )
    return ClipPackage(
        package_path=package_path.expanduser().resolve(),
        extract_dir=extract_dir,
        manifest=manifest,
        run_name=run_name,
        group=group,
        source_crop_row_ids=source_crop_row_ids,
    )


def _is_group(value: object) -> bool:
    return isinstance(value, zarr.Group)


def _is_array(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and not _is_group(value)


def _iter_array_paths(group: zarr.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key in sorted(group.keys()):
        path = f"{prefix}/{key}" if prefix else str(key)
        if path.split("/", 1)[0] in SKIPPED_DERIVED_GROUPS:
            continue
        if PurePosixPath(path).name in SKIPPED_REGENERATED_ARRAY_NAMES:
            continue
        child = group[key]
        if _is_group(child):
            paths.extend(_iter_array_paths(child, path))
        elif _is_array(child):
            paths.append(path)
    return paths


def _iter_group_paths(group: zarr.Group, prefix: str = "") -> list[str]:
    paths = [prefix] if prefix else [""]
    for key in sorted(group.keys()):
        path = f"{prefix}/{key}" if prefix else str(key)
        if path.split("/", 1)[0] in SKIPPED_DERIVED_GROUPS:
            continue
        child = group[key]
        if _is_group(child):
            paths.extend(_iter_group_paths(child, path))
    return paths


def _get_node(group: zarr.Group, path: str) -> Any:
    node: Any = group
    for part in PurePosixPath(path).parts:
        if not part:
            continue
        node = node[part]
    return node


def _require_group(root: zarr.Group, path: str) -> zarr.Group:
    group = root
    for part in PurePosixPath(path).parts:
        if not part:
            continue
        group = group.require_group(part)
    return group


def _copy_group_attrs(source: zarr.Group, target: zarr.Group) -> None:
    target.attrs.update(dict(json_ready(dict(source.attrs))))


def _is_collection_inherited_attr(key: str) -> bool:
    return key in COLLECTION_INHERITED_ATTR_DROP_EXACT


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[str] = []
        for item in value:
            out.extend(_as_string_list(item))
        return out
    text = str(value)
    return [text] if text else []


def _dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def sanitize_collection_run_attrs(dest_run: zarr.Group, packages: Sequence[ClipPackage]) -> None:
    """Remove singleton shard attrs and replace useful ones with collection lists."""

    for key in list(dest_run.attrs.keys()):
        if _is_collection_inherited_attr(str(key)):
            del dest_run.attrs[key]

    for package_attr, collection_attr in COLLECTION_LIST_ATTRS.items():
        values: list[str] = []
        for package in packages:
            values.extend(_as_string_list(package.group.attrs.get(package_attr)))
        values = _dedupe_preserving_order(values)
        if values:
            dest_run.attrs[collection_attr] = values

    roi_cache_used = [bool(package.group.attrs.get("source_roi_cache_used")) for package in packages]
    if any(roi_cache_used):
        dest_run.attrs["source_roi_cache_used"] = bool(all(roi_cache_used))
        dest_run.attrs["source_roi_cache_package_count"] = int(sum(1 for value in roi_cache_used if value))


def _array_chunks(array: Any, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    normalized = tuple(max(1, min(int(chunk), int(dim))) for chunk, dim in zip(tuple(chunks), shape))
    return normalized if len(normalized) == len(shape) else None


def _create_array_like(
    parent: zarr.Group,
    name: str,
    source_array: Any,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
) -> Any:
    chunks = chunks or _array_chunks(source_array, shape)
    kwargs: dict[str, Any] = {
        "shape": shape,
        "dtype": source_array.dtype,
        "overwrite": True,
    }
    if chunks is not None:
        kwargs["chunks"] = chunks
    fill_value = getattr(source_array, "fill_value", None)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    return parent.create_array(name, **kwargs)


def _write_runs_by_mapping(
    dest: Any,
    source: Any,
    *,
    local_rows: np.ndarray,
    dest_rows: np.ndarray,
) -> None:
    if int(local_rows.shape[0]) == 0:
        return
    start = 0
    count = int(local_rows.shape[0])
    while start < count:
        stop = start + 1
        while (
            stop < count
            and int(local_rows[stop]) == int(local_rows[stop - 1]) + 1
            and int(dest_rows[stop]) == int(dest_rows[stop - 1]) + 1
        ):
            stop += 1
        src_start = int(local_rows[start])
        src_stop = int(local_rows[stop - 1]) + 1
        dst_start = int(dest_rows[start])
        dst_stop = int(dest_rows[stop - 1]) + 1
        dest[dst_start:dst_stop] = source[src_start:src_stop]
        start = stop


def _contiguous_runs(local_rows: np.ndarray, dest_rows: np.ndarray) -> list[tuple[int, int, int, int]]:
    if int(local_rows.shape[0]) == 0:
        return []
    runs: list[tuple[int, int, int, int]] = []
    start = 0
    count = int(local_rows.shape[0])
    while start < count:
        stop = start + 1
        while (
            stop < count
            and int(local_rows[stop]) == int(local_rows[stop - 1]) + 1
            and int(dest_rows[stop]) == int(dest_rows[stop - 1]) + 1
        ):
            stop += 1
        runs.append(
            (
                int(local_rows[start]),
                int(local_rows[stop - 1]) + 1,
                int(dest_rows[start]),
                int(dest_rows[stop - 1]) + 1,
            )
        )
        start = stop
    return runs


def _numpy_block_dtype(dtype: Any) -> Any:
    try:
        return np.dtype(dtype)
    except TypeError:
        return object


def _empty_chunk_block(shape: tuple[int, ...], dtype: Any, fill_value: Any) -> np.ndarray:
    block_dtype = _numpy_block_dtype(dtype)
    block = np.empty(shape, dtype=block_dtype)
    if fill_value is None:
        if block_dtype == object:
            block.fill("")
        else:
            block.fill(0)
    else:
        block[...] = fill_value
    return block


def _copy_row_aligned_array_by_chunk(
    dest: Any,
    packages: Sequence[ClipPackage],
    *,
    array_path: str,
    row_maps: Mapping[Path, tuple[np.ndarray, np.ndarray]],
    total_rows: int,
    array_copy_workers: int,
) -> None:
    """Copy one row-aligned array with one writer per physical row chunk.

    Multiple clip packages can contribute rows to the same destination chunk at
    clip boundaries. Building each destination chunk in memory and writing it
    once avoids Zarr chunk-level read-modify-write races.
    """

    row_chunk = int(getattr(dest, "chunks", (0,))[0] or total_rows or 1)
    row_chunk = max(1, row_chunk)
    fill_value = getattr(dest, "fill_value", None)
    dtype = getattr(dest, "dtype", None)

    def write_dest_chunk(dst_start: int, dst_stop: int) -> None:
        block_shape = (int(dst_stop) - int(dst_start), *tuple(int(value) for value in dest.shape[1:]))
        block = _empty_chunk_block(block_shape, dtype, fill_value)
        wrote_any = False
        for package in packages:
            local_rows, dest_rows = row_maps[package.package_path]
            mask = (dest_rows >= dst_start) & (dest_rows < dst_stop)
            if not bool(np.any(mask)):
                continue
            package_array = _get_node(package.group, array_path)
            selected_local_rows = local_rows[mask]
            selected_dest_rows = dest_rows[mask]
            for src_start, src_stop, run_dst_start, run_dst_stop in _contiguous_runs(
                selected_local_rows,
                selected_dest_rows,
            ):
                rel_start = int(run_dst_start) - int(dst_start)
                rel_stop = int(run_dst_stop) - int(dst_start)
                block[rel_start:rel_stop] = package_array[src_start:src_stop]
                wrote_any = True
        if not wrote_any:
            return
        dest[dst_start:dst_stop] = block

    chunks = [(start, min(start + row_chunk, total_rows)) for start in range(0, int(total_rows), row_chunk)]
    workers = max(1, int(array_copy_workers))
    if workers == 1 or len(chunks) <= 1:
        for start, stop in chunks:
            write_dest_chunk(start, stop)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(write_dest_chunk, start, stop) for start, stop in chunks]
        for future in futures:
            future.result()


def _is_contour_array_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return len(parts) >= 4 and parts[-2] == "contours" and parts[-1] in {"ptr", "len", "points_xy"}


def _dense_masks_roi_chunks(array_path: str, dest_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if array_path != "masks_roi" or len(dest_shape) != 4:
        return None
    return refined_subject_mask_storage_chunks(
        int(dest_shape[0]),
        int(dest_shape[2]),
        int(dest_shape[3]),
    )


def _contour_components(group: zarr.Group) -> list[str]:
    components = group.get("components")
    if not isinstance(components, zarr.Group):
        return []
    out: list[str] = []
    for name in sorted(components.keys()):
        comp = components[name]
        if not isinstance(comp, zarr.Group):
            continue
        contours = comp.get("contours")
        if isinstance(contours, zarr.Group) and all(key in contours for key in ("ptr", "len", "points_xy")):
            out.append(str(name))
    return out


def _merge_contours(
    packages: Sequence[ClipPackage],
    dest_run: zarr.Group,
    *,
    component_name: str,
    row_maps: Mapping[Path, tuple[np.ndarray, np.ndarray]],
    total_rows: int,
    source_mask_run: str,
) -> dict[str, object]:
    ptr = np.full((total_rows,), -1, dtype=np.int64)
    length = np.zeros((total_rows,), dtype=np.int32)
    segments_by_row: list[np.ndarray | None] = [None] * int(total_rows)
    points: list[np.ndarray] = []
    first_contours: zarr.Group | None = None
    for package in packages:
        comp = package.group["components"][component_name]
        contours = comp["contours"]
        if first_contours is None:
            first_contours = contours
        local_rows, dest_rows = row_maps[package.package_path]
        local_ptr = np.asarray(contours["ptr"][:], dtype=np.int64)
        local_len = np.asarray(contours["len"][:], dtype=np.int32)
        local_points = np.asarray(contours["points_xy"][:], dtype=np.float32).reshape(-1, 2)
        for src_row, dst_row in zip(local_rows.tolist(), dest_rows.tolist(), strict=True):
            n_points = int(local_len[int(src_row)])
            if n_points <= 0:
                continue
            src_offset = int(local_ptr[int(src_row)])
            if src_offset < 0:
                raise ValueError(
                    f"{package.package_path} component {component_name} has len={n_points} "
                    f"but ptr={src_offset} at local row {int(src_row)}."
                )
            segment = np.asarray(local_points[src_offset : src_offset + n_points], dtype=np.float32)
            segments_by_row[int(dst_row)] = segment

    offset = 0
    for row_idx, segment in enumerate(segments_by_row):
        if segment is None or int(segment.shape[0]) == 0:
            continue
        ptr[row_idx] = np.int64(offset)
        length[row_idx] = np.int32(segment.shape[0])
        points.append(segment)
        offset += int(segment.shape[0])

    points_xy = (
        np.concatenate(points, axis=0).astype(np.float32, copy=False)
        if points
        else np.empty((0, 2), dtype=np.float32)
    )
    comp_dest = dest_run["components"][component_name]
    contours_dest = comp_dest.require_group("contours")
    if first_contours is not None:
        contours_dest.attrs.update(dict(json_ready(dict(first_contours.attrs))))
    contours_dest.attrs["source_mask_run"] = str(source_mask_run)
    contours_dest.attrs["cache_coverage"] = "full_indexed_rows"
    contours_dest.attrs["merged_from_clip_packages"] = True
    contours_dest.attrs["generated_at_utc"] = _utc_now()
    chunk_rois = refined_subject_mask_metric_row_chunk(total_rows)
    contours_dest.create_array("ptr", data=ptr, chunks=(chunk_rois,), overwrite=True)
    contours_dest.create_array("len", data=length, chunks=(chunk_rois,), overwrite=True)
    contours_dest.create_array(
        "points_xy",
        data=points_xy,
        chunks=(max(1, min(4096, int(points_xy.shape[0]))), 2),
        overwrite=True,
    )
    return {
        "component": str(component_name),
        "status": "written",
        "roi_count": int(total_rows),
        "contour_count": int(np.count_nonzero(length > 0)),
        "point_count": int(points_xy.shape[0]),
    }


def _build_row_maps(packages: Sequence[ClipPackage]) -> tuple[np.ndarray, dict[Path, tuple[np.ndarray, np.ndarray]]]:
    all_ids = np.concatenate([package.source_crop_row_ids for package in packages], axis=0)
    unique_ids, counts = np.unique(all_ids, return_counts=True)
    if int(unique_ids.shape[0]) != int(all_ids.shape[0]):
        duplicate = int(unique_ids[np.flatnonzero(counts > 1)[0]])
        raise ValueError(f"Duplicate source_crop_row_ids across clip packages: {duplicate}")
    sorted_ids = np.sort(all_ids, kind="stable")
    dest_by_crop_row = {int(crop_row_id): int(row_idx) for row_idx, crop_row_id in enumerate(sorted_ids.tolist())}
    row_maps: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for package in packages:
        local_order = np.argsort(package.source_crop_row_ids, kind="stable").astype(np.int64, copy=False)
        dest_rows = np.asarray(
            [dest_by_crop_row[int(package.source_crop_row_ids[int(local_row)])] for local_row in local_order.tolist()],
            dtype=np.int64,
        )
        row_maps[package.package_path] = (local_order, dest_rows)
    return sorted_ids.astype(np.int64, copy=False), row_maps


def _validate_package_schema(packages: Sequence[ClipPackage]) -> None:
    if not packages:
        raise ValueError("No clip packages supplied.")
    reference = packages[0].group
    reference_labels = list(reference.attrs.get("mask_labels") or [])
    reference_schema = str(reference.attrs.get("label_schema_id") or "")
    reference_shape = tuple(int(value) for value in reference["masks_roi"].shape[1:])
    reference_paths = set(_iter_array_paths(reference))
    for package in packages[1:]:
        labels = list(package.group.attrs.get("mask_labels") or [])
        if labels != reference_labels:
            raise ValueError(
                f"Package {package.package_path} mask_labels {labels!r} do not match "
                f"{reference_labels!r}."
            )
        schema = str(package.group.attrs.get("label_schema_id") or "")
        if schema != reference_schema:
            raise ValueError(
                f"Package {package.package_path} label_schema_id={schema!r} does not match "
                f"{reference_schema!r}."
            )
        shape = tuple(int(value) for value in package.group["masks_roi"].shape[1:])
        if shape != reference_shape:
            raise ValueError(f"Package {package.package_path} masks_roi row shape {shape} != {reference_shape}.")
        paths = set(_iter_array_paths(package.group))
        if paths != reference_paths:
            missing = sorted(reference_paths - paths)
            extra = sorted(paths - reference_paths)
            raise ValueError(
                f"Package {package.package_path} array paths do not match reference; "
                f"missing={missing[:5]!r}, extra={extra[:5]!r}."
            )


def _copy_group_tree_attrs(reference: zarr.Group, dest: zarr.Group) -> None:
    for group_path in _iter_group_paths(reference):
        source_group = _get_node(reference, group_path) if group_path else reference
        target_group = _require_group(dest, group_path)
        _copy_group_attrs(source_group, target_group)


def import_refined_subject_mask_clip_packages(
    *,
    zarr_path: Path,
    package_paths: Sequence[Path],
    output_run: str,
    overwrite: bool = False,
    expected_target_crop_run: str | None = None,
    array_copy_workers: int = 1,
) -> dict[str, Any]:
    zarr_path = zarr_path.expanduser().resolve()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="palette_refined_subject_mask_package_import_") as tmp:
        extract_root = Path(tmp)
        packages = [_load_package(path, extract_root) for path in package_paths]
        _validate_package_schema(packages)
        sorted_crop_rows, row_maps = _build_row_maps(packages)

        root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
        parent = require_runs_parent(root, "refined_subject_masks_runs")
        if output_run in parent:
            if not overwrite:
                raise ValueError(f"refined_subject_masks_runs/{output_run} already exists. Pass --overwrite.")
            del parent[output_run]
        dest_run = parent.create_group(output_run)
        try:
            reference = packages[0].group
            _copy_group_tree_attrs(reference, dest_run)
            sanitize_collection_run_attrs(dest_run, packages)
            dest_run.attrs["palette_run_completion_status"] = "running"

            total_rows = int(sorted_crop_rows.shape[0])
            target_crop_runs = sorted({str(package.group.attrs.get("source_crop_run") or "") for package in packages})
            if expected_target_crop_run and target_crop_runs != [str(expected_target_crop_run)]:
                raise ValueError(
                    f"Package source_crop_run values {target_crop_runs!r} do not match expected "
                    f"{expected_target_crop_run!r}."
                )

            for array_path in _iter_array_paths(reference):
                if _is_contour_array_path(array_path):
                    continue
                source_array = _get_node(reference, array_path)
                parent_path = str(PurePosixPath(array_path).parent)
                if parent_path == ".":
                    parent_path = ""
                array_name = PurePosixPath(array_path).name
                dest_parent = _require_group(dest_run, parent_path)
                source_shape = tuple(int(value) for value in source_array.shape)
                if source_shape and source_shape[0] == packages[0].row_count:
                    dest_shape = (total_rows, *source_shape[1:])
                    dest_array = _create_array_like(
                        dest_parent,
                        array_name,
                        source_array,
                        shape=dest_shape,
                        chunks=_dense_masks_roi_chunks(array_path, dest_shape),
                    )
                    for package in packages:
                        package_array = _get_node(package.group, array_path)
                        if int(package_array.shape[0]) != package.row_count:
                            raise ValueError(
                                f"{package.package_path}:{array_path} has first dimension "
                                f"{int(package_array.shape[0])}, expected {package.row_count}."
                            )
                    _copy_row_aligned_array_by_chunk(
                        dest_array,
                        packages,
                        array_path=array_path,
                        row_maps=row_maps,
                        total_rows=total_rows,
                        array_copy_workers=array_copy_workers,
                    )
                else:
                    data = np.asarray(source_array[:])
                    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
                    chunks = _array_chunks(source_array, tuple(int(value) for value in data.shape))
                    if chunks is not None:
                        kwargs["chunks"] = chunks
                    dest_parent.create_array(array_name, **kwargs)

            dest_run["source_crop_row_ids"][:] = sorted_crop_rows
        except Exception:
            if output_run in parent:
                del parent[output_run]
            raise

        contour_components = sorted(set.intersection(*[set(_contour_components(package.group)) for package in packages]))
        contour_summaries = [
            _merge_contours(
                packages,
                dest_run,
                component_name=component,
                row_maps=row_maps,
                total_rows=total_rows,
                source_mask_run=output_run,
            )
            for component in contour_components
        ]
        if contour_summaries:
            dest_run.attrs["component_contours_status"] = "computed"
            dest_run.attrs["component_contours_components"] = [item["component"] for item in contour_summaries]
            dest_run.attrs["component_contours_summary"] = list(json_ready(contour_summaries))

        update_mask_storage_attrs(dest_run, has_dense=True, has_rle=False, has_bitpacked=False)
        dest_run.attrs["method"] = "refined_subject_mask_clip_package_import_v1"
        dest_run.attrs["source_refined_subject_mask_clip_package_runs"] = [package.run_name for package in packages]
        dest_run.attrs["source_refined_subject_mask_clip_package_paths"] = [str(package.package_path) for package in packages]
        dest_run.attrs["source_crop_run"] = target_crop_runs[0] if len(target_crop_runs) == 1 else ""
        dest_run.attrs["source_crop_run_values"] = target_crop_runs
        dest_run.attrs["row_merge_key"] = "source_crop_row_ids"
        dest_run.attrs["row_merge_order"] = "ascending_source_crop_row_ids"
        dest_run.attrs["array_copy_workers"] = int(max(1, array_copy_workers))
        dest_run.attrs["array_copy_strategy"] = "chunk_owned_parallel" if int(array_copy_workers) > 1 else "chunk_owned_serial"
        dest_run.attrs["import_schema_id"] = IMPORT_SCHEMA_ID
        dest_run.attrs["created_at_utc"] = _utc_now()
        dest_run.attrs["created_utc"] = dest_run.attrs["created_at_utc"]
        dest_run.attrs["duration_seconds"] = float(time.perf_counter() - started)
        dest_run.attrs["summary_statistics"] = {
            **dict(dest_run.attrs.get("summary_statistics") or {}),
            "rows_total": total_rows,
            "imported_clip_package_count": int(len(packages)),
        }
        for component_name in list(dest_run.attrs.get("mask_labels") or []):
            component_group = dest_run.get(f"components/{component_name}")
            if isinstance(component_group, zarr.Group):
                labels = read_reason_labels(component_group)
                if labels is not None and int(labels.shape[0]) == total_rows:
                    write_reason_columns(
                        component_group,
                        labels,
                        chunk_size=refined_subject_mask_metric_row_chunk(total_rows),
                        include_reason_text=True,
                        overwrite=True,
                    )

        package_artifacts = [
            {
                "artifact_path": str(package.package_path),
                "schema_id": str(package.manifest.get("schema_id") or ""),
                "run_group_path": str(package.manifest.get("run_group_path") or ""),
                "row_count": int(package.row_count),
            }
            for package in packages
        ]
        run_provenance = build_run_provenance(
            command="fisheye.utils.import_refined_subject_mask_clip_packages",
            params={
                "zarr_path": str(zarr_path),
                "output_run": str(output_run),
                "package_count": int(len(packages)),
                "expected_target_crop_run": expected_target_crop_run,
                "row_merge_key": "source_crop_row_ids",
                "array_copy_workers": int(max(1, array_copy_workers)),
            },
            input_run_ids={
                "refined_subject_mask_clip_packages": [package.run_name for package in packages],
                "target_crop_run": target_crop_runs[0] if len(target_crop_runs) == 1 else target_crop_runs,
            },
            input_artifacts=package_artifacts,
            cwd=Path.cwd(),
        )
        dest_run.attrs[RUN_PROVENANCE_ATTR] = dict(run_provenance)
        dest_run.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(run_provenance)
        mark_run_complete(dest_run, parent_group=parent, run_name=output_run, run_provenance=run_provenance)
        parent.attrs["refined_subject_mask_review_status_latest"] = output_run

        return {
            "schema_id": IMPORT_SCHEMA_ID,
            "status": "ok",
            "zarr_path": str(zarr_path),
            "output_run": str(output_run),
            "row_count": total_rows,
            "source_crop_row_min": int(sorted_crop_rows.min()) if sorted_crop_rows.size else None,
            "source_crop_row_max": int(sorted_crop_rows.max()) if sorted_crop_rows.size else None,
            "package_count": int(len(packages)),
            "packages": package_artifacts,
            "component_contours": contour_summaries,
            "duration_seconds": float(dest_run.attrs["duration_seconds"]),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--package", dest="packages", action="append", required=True, type=Path)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--expected-target-crop-run")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--array-copy-workers",
        type=int,
        default=1,
        help=(
            "Number of chunk-owned row-copy workers for row-aligned arrays. "
            "Each task writes a whole destination row chunk to avoid Zarr RMW races."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = import_refined_subject_mask_clip_packages(
        zarr_path=args.zarr,
        package_paths=args.packages,
        output_run=args.output_run,
        overwrite=bool(args.overwrite),
        expected_target_crop_run=args.expected_target_crop_run,
        array_copy_workers=max(1, int(args.array_copy_workers)),
    )
    if args.json:
        print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    else:
        print(
            f"Imported {result['package_count']} refined subject-mask clip packages "
            f"into refined_subject_masks_runs/{result['output_run']} ({result['row_count']} rows)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

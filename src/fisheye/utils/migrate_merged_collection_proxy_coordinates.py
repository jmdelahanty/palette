#!/usr/bin/env python3
"""Publish canonical coordinates for validated merged collection-proxy rows.

Dry-run is the default.  The migration is intentionally narrow: it accepts
only the versioned merged clipped-collection proxy schema, replays every merged
row through its persisted source proxy row, and then through the exact refined
detection row before adding acquisition-frame and source-camera geometry
authority to the existing auxiliary rowset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.detection.detect_yolo import (
    _publish_detection_frame_evidence,
    _restore_detection_coordinate_checkpoints,
)
from fisheye.shared.observation_coordinate_publication import (
    COLLECTION_PROXY_ACQUISITION_MAPPING_ATTR,
    MERGED_COLLECTION_PROXY_SCHEMA,
    MERGED_COLLECTION_PROXY_SOURCE_KIND,
    derive_detection_source_camera_geometry,
    load_persisted_collection_proxy_observation_geometry,
    publish_collection_proxy_acquisition_mapping,
    publish_detection_observation_geometry,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


MIGRATION_SCHEMA_ID = "palette.merged_collection_proxy_coordinate_migration.v1"


@dataclass(frozen=True)
class MigrationPlan:
    zarr_path: str
    crop_run: str
    camera_id: str
    row_count: int
    source_proxy_run_count: int
    source_refined_run_count: int
    status: str


def _open(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _array(group: zarr.Group, name: str, dtype: Any | None = None) -> np.ndarray:
    if name not in group:
        raise ValueError(f"{group.path}/{name} is required.")
    values = np.asarray(group[name][:])
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _value_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare exact numeric values while permitting documented storage casts."""

    if left.shape != right.shape or left.dtype.kind not in "iufc" or right.dtype.kind not in "iufc":
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _positions_for_unique_ids(
    available_ids: np.ndarray,
    requested_ids: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    if (
        available_ids.ndim != 1
        or requested_ids.ndim != 1
        or available_ids.dtype.kind not in "iu"
        or requested_ids.dtype.kind not in "iu"
    ):
        raise ValueError(f"{label} must use one-dimensional integer identifiers.")
    order = np.argsort(available_ids, kind="stable")
    sorted_ids = available_ids[order]
    if sorted_ids.size > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"{label} contains duplicate identifiers.")
    locations = np.searchsorted(sorted_ids, requested_ids)
    valid = locations < sorted_ids.size
    if np.any(valid):
        valid[valid] &= sorted_ids[locations[valid]] == requested_ids[valid]
    if not np.all(valid):
        raise ValueError(f"{label} does not contain every requested identifier.")
    return order[locations]


def _validate_source_rows(
    root: zarr.Group,
    rowset: zarr.Group,
) -> tuple[int, int, int]:
    attrs = rowset.attrs
    if (
        attrs.get("schema") != MERGED_COLLECTION_PROXY_SCHEMA
        or attrs.get("crop_proxy_schema") != MERGED_COLLECTION_PROXY_SCHEMA
        or attrs.get("source_kind") != MERGED_COLLECTION_PROXY_SOURCE_KIND
    ):
        raise ValueError("Selected crop run is not the exact merged proxy schema.")
    source_runs = attrs.get("source_proxy_crop_runs")
    source_refined_paths = attrs.get("source_refined_run_paths")
    if not isinstance(source_runs, list) or not source_runs:
        raise ValueError("source_proxy_crop_runs must be a nonempty list.")
    if not isinstance(source_refined_paths, list) or not source_refined_paths:
        raise ValueError("source_refined_run_paths must be a nonempty list.")
    if len(source_refined_paths) != len(source_runs):
        raise ValueError(
            "Merged proxy source runs and refined paths must be one-to-one."
        )

    merged_keys = _array(rowset, "instance_key")
    merged_frames = _array(rowset, "frame_indices")
    merged_source_frames = _array(rowset, "source_frame_indices")
    merged_bbox = _array(rowset, "bbox_norm_coords")
    run_indices = _array(rowset, "source_proxy_crop_run_index", np.int64)
    row_indices = _array(rowset, "source_proxy_crop_row_ids", np.int64)
    row_count = int(merged_keys.shape[0])
    if (
        merged_keys.shape != (row_count,)
        or merged_frames.shape != (row_count,)
        or merged_source_frames.shape != (row_count,)
        or merged_bbox.shape != (row_count, 4)
        or run_indices.shape != (row_count,)
        or row_indices.shape != (row_count,)
        or not np.array_equal(
            merged_frames.astype(np.int64),
            merged_source_frames.astype(np.int64),
        )
    ):
        raise ValueError("Merged proxy arrays are not exactly row-aligned.")

    seen_refined: list[str] = []
    verified_rows = 0
    for source_index, source_name in enumerate(source_runs):
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("source_proxy_crop_runs contains an invalid name.")
        merged_rows = np.flatnonzero(run_indices == source_index)
        if merged_rows.size == 0:
            raise ValueError(f"Source proxy run {source_name!r} has no merged rows.")
        source_path = f"crop_runs/{source_name}"
        if source_path not in root:
            raise ValueError(f"Source proxy rowset is missing: {source_path}")
        source = root[source_path]
        source_rows = row_indices[merged_rows]
        source_count = int(source["instance_key"].shape[0])
        if np.any(source_rows < 0) or np.any(source_rows >= source_count):
            raise ValueError(f"Merged source row ids exceed {source_path}.")
        comparisons = [
            (merged_keys[merged_rows], _array(source, "instance_key")[source_rows]),
            (merged_frames[merged_rows], _array(source, "frame_indices")[source_rows]),
            (
                merged_source_frames[merged_rows],
                _array(source, "source_frame_indices")[source_rows],
            ),
        ]
        if "bbox_norm_coords" in source:
            comparisons.append(
                (
                    merged_bbox[merged_rows],
                    _array(source, "bbox_norm_coords")[source_rows],
                )
            )
        if any(not _equal(left, right) for left, right in comparisons):
            raise ValueError(f"Merged values disagree with exact source proxy {source_path}.")

        refined_paths = source.attrs.get("source_refined_run_paths")
        expected_refined_path = str(source_refined_paths[source_index])
        if refined_paths is None:
            refined_path = expected_refined_path
        elif isinstance(refined_paths, list) and len(refined_paths) == 1:
            refined_path = str(refined_paths[0])
            if refined_path != expected_refined_path:
                raise ValueError(
                    f"{source_path} refined source disagrees with merged lineage."
                )
        else:
            raise ValueError(f"{source_path} has ambiguous refined-row lineage.")
        if refined_path not in root:
            raise ValueError(f"Refined source rowset is missing: {refined_path}")
        if refined_path not in seen_refined:
            seen_refined.append(refined_path)
        refined_run = root[refined_path]
        if "instances" not in refined_run:
            raise ValueError(f"Refined source instances are missing: {refined_path}")
        refined = refined_run["instances"]
        source_refined_rows = _array(source, "source_refined_row_ids", np.int64)[
            source_rows
        ]
        refined_ids = _array(refined, "refined_row_ids")
        refined_positions = _positions_for_unique_ids(
            refined_ids,
            source_refined_rows,
            label=f"Refined row identifiers at {refined_path}",
        )
        source_local_frames = _array(
            source,
            "source_clip_local_frame_indices",
        )[source_rows]
        refined_comparisons = (
            (
                merged_keys[merged_rows],
                _array(refined, "instance_key")[refined_positions],
            ),
            (
                source_local_frames,
                _array(refined, "frame_indices")[refined_positions],
            ),
            (
                merged_bbox[merged_rows],
                _array(refined, "bbox_norm_coords")[refined_positions],
            ),
        )
        if any(not _value_equal(left, right) for left, right in refined_comparisons):
            raise ValueError(
                f"Merged values disagree with exact refined source {refined_path}."
            )
        verified_rows += int(merged_rows.size)

    if verified_rows != row_count or set(seen_refined) != set(source_refined_paths):
        raise ValueError("Merged proxy source coverage is incomplete or disagrees with attrs.")
    return row_count, len(source_runs), len(seen_refined)


def plan_migration(zarr_path: Path, crop_run: str) -> MigrationPlan:
    path = Path(zarr_path).expanduser().resolve()
    root = _open(path, mode="r")
    rowset_path = f"crop_runs/{crop_run}"
    if rowset_path not in root:
        raise ValueError(f"Merged proxy crop run is missing: {rowset_path}")
    rowset = root[rowset_path]
    row_count, source_count, refined_count = _validate_source_rows(root, rowset)
    _, acquisition = load_persisted_acquisition_camera_authority(root)
    status = "would_publish_canonical_coordinates"
    if (
        COLLECTION_PROXY_ACQUISITION_MAPPING_ATTR in rowset.attrs
        and rowset.attrs.get("coordinate_contract") == "canonical_v2"
    ):
        load_persisted_collection_proxy_observation_geometry(root, rowset_path)
        status = "already_canonical"
    elif COLLECTION_PROXY_ACQUISITION_MAPPING_ATTR in rowset.attrs:
        status = "would_resume_partial_coordinate_publication"
    return MigrationPlan(
        zarr_path=str(path),
        crop_run=crop_run,
        camera_id=acquisition.record.camera_id,
        row_count=row_count,
        source_proxy_run_count=source_count,
        source_refined_run_count=refined_count,
        status=status,
    )


def _create_or_verify_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
) -> bool:
    values = np.asarray(values)
    if name in group:
        current = np.asarray(group[name][:])
        if not _equal(current, values):
            raise ValueError(f"Existing {group.path}/{name} disagrees with migration values.")
        return False
    create_geometry_preload_array(group, name, data=values, overwrite=False)
    return True


def apply_migration(
    zarr_path: Path,
    crop_run: str,
    *,
    consolidate_metadata: bool = True,
) -> MigrationPlan:
    plan = plan_migration(zarr_path, crop_run)
    if plan.status == "already_canonical":
        return plan
    root = _open(Path(plan.zarr_path), mode="a")
    rowset = root[f"crop_runs/{crop_run}"]
    _validate_source_rows(root, rowset)
    _, acquisition = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=plan.camera_id,
    )
    attrs_snapshot = dict(rowset.attrs)
    created_arrays: list[str] = []
    frame_checkpoints: tuple[Any, ...] = ()
    had_frame_group = "coordinate_frames" in rowset
    had_normalized_frame = (
        had_frame_group and "source_camera_normalized" in rowset["coordinate_frames"]
    )
    had_transform_group = "coordinate_transforms" in rowset
    had_transform_matrix = (
        had_transform_group
        and "source_camera_normalized_to_image" in rowset["coordinate_transforms"]
    )
    had_transform_authority = (
        had_transform_group
        and "source_camera_normalized_to_image_authority"
        in rowset["coordinate_transforms"]
    )
    try:
        evidence, frame_checkpoints = _publish_detection_frame_evidence(
            root,
            rowset,
            acquisition_frame=acquisition,
        )
        bbox_norm = np.asarray(rowset["bbox_norm_coords"][:])
        bbox_img, centers = derive_detection_source_camera_geometry(
            bbox_norm,
            frame_evidence=evidence,
        )
        for name, values in (
            (
                "source_acquisition_frame_index",
                np.asarray(rowset["frame_indices"][:], dtype=np.int64),
            ),
            ("bbox_img_xyxy", bbox_img),
            ("centers_img_xy", centers),
        ):
            if _create_or_verify_array(rowset, name, values):
                created_arrays.append(name)
        mapping = publish_collection_proxy_acquisition_mapping(
            rowset,
            acquisition_frame=acquisition,
        )
        publish_detection_observation_geometry(
            rowset,
            rowset["instance_key"],
            rowset["source_acquisition_frame_index"],
            rowset["bbox_norm_coords"],
            rowset["bbox_img_xyxy"],
            rowset["centers_img_xy"],
            frame_evidence=evidence,
            source_lineage_records=(mapping,),
        )
        rowset.attrs["coordinate_contract"] = "canonical_v2"
        load_persisted_collection_proxy_observation_geometry(
            root,
            f"crop_runs/{crop_run}",
        )
    except BaseException as exc:
        for name in reversed(created_arrays):
            if name in rowset:
                del rowset[name]
        for name in tuple(rowset.attrs.keys()):
            del rowset.attrs[name]
        rowset.attrs.update(attrs_snapshot)
        if frame_checkpoints:
            failures = _restore_detection_coordinate_checkpoints(
                frame_checkpoints,
                cause=exc,
            )
            if failures:
                raise RuntimeError(
                    f"Coordinate migration rollback was incomplete: {failures!r}"
                ) from exc
        if "coordinate_transforms" in rowset:
            transforms = rowset["coordinate_transforms"]
            if (
                not had_transform_authority
                and "source_camera_normalized_to_image_authority" in transforms
            ):
                del transforms["source_camera_normalized_to_image_authority"]
            if (
                not had_transform_matrix
                and "source_camera_normalized_to_image" in transforms
            ):
                del transforms["source_camera_normalized_to_image"]
            if not had_transform_group:
                del rowset["coordinate_transforms"]
        if "coordinate_frames" in rowset:
            frames = rowset["coordinate_frames"]
            if not had_normalized_frame and "source_camera_normalized" in frames:
                del frames["source_camera_normalized"]
            if not had_frame_group:
                del rowset["coordinate_frames"]
        raise
    if consolidate_metadata:
        consolidate_metadata_capture_expected_warnings(Path(plan.zarr_path))
    return MigrationPlan(**{**asdict(plan), "status": "published_canonical_coordinates"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            result = apply_migration(
                args.zarr_path,
                args.crop_run,
                consolidate_metadata=not args.no_consolidate,
            )
        else:
            result = plan_migration(args.zarr_path, args.crop_run)
        payload = {"schema_id": MIGRATION_SCHEMA_ID, **asdict(result)}
        code = 0
    except Exception as exc:
        payload = {
            "schema_id": MIGRATION_SCHEMA_ID,
            "zarr_path": str(args.zarr_path),
            "crop_run": args.crop_run,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        code = 1
    print(json.dumps(payload, sort_keys=True) if args.json else payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Repair missing full-frame geometry on a completed clipped quality source.

This is a compatibility-only metadata repair for historical v1 source snapshots.
It validates the immutable arrays and their complete raw-detection lineage before
adding the geometry contract that current collection quality consumers require.
No array payload is rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.cluster.lsf import write_json_snapshot
from fisheye.shared.detect_quality_contract import FULL_FRAME_GEOMETRY_SCHEMA
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import is_run_complete
from fisheye.utils.materialize_clipped_detect_quality_source import (
    SOURCE_FAMILY,
    _build_sources,
    _frame_index_path,
    _frame_maps,
    _group_at,
    _read_json,
    _safe_name,
    _stamp_parent_geometry,
    _validate_parent_geometry,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA


REPAIR_SCHEMA = "palette.clipped_detect_quality_source_geometry_repair.v1"
SUPPORTED_SOURCE_SCHEMAS = {
    "palette.clipped_detect_quality_source.v1",
    "palette.clipped_detect_quality_source.v2",
}
ARRAY_CONTRACTS = {
    "frame_indices": (np.dtype(np.int64), ()),
    "bbox_norm_coords": (None, (4,)),
    "instance_key": (np.dtype(np.uint64), ()),
    "source_clip_indices": (np.dtype(np.int32), ()),
    "source_clip_local_frame_indices": (np.dtype(np.int32), ()),
    "source_clip_detect_row_index": (np.dtype(np.int32), ()),
}


def _expected_slices(sources: Sequence[Any]) -> list[dict[str, Any]]:
    return [
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


def _validate_slices(observed: object, expected: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(observed, list) or len(observed) != len(expected):
        raise ValueError("Completed quality source does not have the expected source_slices.")
    required = (
        "clip_id",
        "clip_index",
        "camera_serial",
        "detect_run",
        "detect_group_path",
        "start",
        "stop",
        "row_count",
    )
    for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
        if not isinstance(actual, Mapping):
            raise ValueError(f"source_slices[{index}] is not an object.")
        mismatches = {
            key: {"expected": wanted[key], "observed": actual.get(key)}
            for key in required
            if actual.get(key) != wanted[key]
        }
        for key in ("source_video_width", "source_video_height"):
            if actual.get(key) is not None and actual.get(key) != wanted[key]:
                mismatches[key] = {"expected": wanted[key], "observed": actual.get(key)}
        if mismatches:
            raise ValueError(
                f"source_slices[{index}] disagrees with the raw detection lineage: "
                + json.dumps(mismatches, sort_keys=True)
            )


def _validate_arrays(source: Any, *, row_count: int) -> None:
    digests = source.attrs.get("decoded_array_sha256")
    if not isinstance(digests, Mapping):
        raise ValueError("Completed quality source has no decoded_array_sha256 contract.")
    for name, (dtype, trailing_shape) in ARRAY_CONTRACTS.items():
        if name not in source:
            raise ValueError(f"Completed quality source is missing array {name!r}.")
        array = source[name]
        expected_shape = (int(row_count), *trailing_shape)
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Completed quality source array {name!r} has shape {tuple(array.shape)}, "
                f"expected {expected_shape}."
            )
        if dtype is not None and np.dtype(array.dtype) != dtype:
            raise ValueError(
                f"Completed quality source array {name!r} has dtype {array.dtype}, "
                f"expected {dtype}."
            )
        digest = str(digests.get(name) or "")
        if len(digest) != 64:
            raise ValueError(f"Completed quality source array {name!r} has no SHA-256 digest.")


def repair_clipped_detect_quality_source_geometry(
    zarr_path: str | Path,
    *,
    plan_path: str | Path,
    source_run: str,
    recording_frame_index: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    plan_file = Path(plan_path).expanduser().resolve()
    run_name = _safe_name(source_run, label="source_run")
    plan = _read_json(plan_file)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported clipped detection plan: {plan_file}")
    planned_archive = Path(str(plan.get("analysis_zarr") or "")).expanduser().resolve()
    if planned_archive != archive:
        raise ValueError(f"Detection plan targets {planned_archive}, not {archive}.")
    raw_units = plan.get("work_units")
    if not isinstance(raw_units, list) or not raw_units or not all(
        isinstance(unit, Mapping) for unit in raw_units
    ):
        raise ValueError("Detection plan requires object-valued work units.")
    units = list(raw_units)
    frame_index = _frame_index_path(
        plan,
        Path(recording_frame_index) if recording_frame_index is not None else None,
    )
    frame_maps, frame_count = _frame_maps(frame_index, units)
    root = open_zarr_root(archive, mode="r")
    sources = _build_sources(root, units, frame_maps)
    expected_slices = _expected_slices(sources)
    row_count = sum(source.rows for source in sources)
    width = sources[0].source_video_width
    height = sources[0].source_video_height
    source_group_path = f"{SOURCE_FAMILY}/{run_name}"
    source = _group_at(root, source_group_path)
    if not is_run_complete(source):
        raise ValueError(f"Quality source is not complete: {source_group_path}")
    source_schema = str(source.attrs.get("schema_id") or "")
    if source_schema not in SUPPORTED_SOURCE_SCHEMAS:
        raise ValueError(f"Unsupported quality source schema {source_schema!r}.")
    if int(source.attrs.get("source_row_count") or -1) != row_count:
        raise ValueError("Quality source row count disagrees with its raw detection lineage.")
    if int(source.attrs.get("recording_frame_count") or -1) != frame_count:
        raise ValueError("Quality source frame count disagrees with the canonical frame index.")
    _validate_slices(source.attrs.get("source_slices"), expected_slices)
    _validate_arrays(source, row_count=row_count)
    validation = source.attrs.get("source_validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "complete":
        raise ValueError("Quality source does not have a complete source_validation record.")
    _validate_parent_geometry(root, width=width, height=height)

    repair_record = {
        "schema": REPAIR_SCHEMA,
        "status": "complete",
        "source_schema_preserved": source_schema,
        "source_group_path": source_group_path,
        "source_plan_path": str(plan_file),
        "recording_frame_index_path": str(frame_index),
        "source_video_width": width,
        "source_video_height": height,
        "source_row_count": row_count,
        "recording_frame_count": frame_count,
        "source_slice_count": len(expected_slices),
        "array_payload_rewritten": False,
    }
    report = {
        "status": "planned" if not apply else "complete",
        "repair": repair_record,
        "zarr_path": str(archive),
    }
    if not apply:
        return report

    write_root = open_zarr_root(archive, mode="a")
    write_source = _group_at(write_root, source_group_path)
    existing_repair = write_source.attrs.get("full_frame_geometry_repair")
    if existing_repair is not None and existing_repair != repair_record:
        raise ValueError("Quality source contains a conflicting geometry-repair record.")
    repaired_validation = dict(validation)
    repaired_validation.update(
        {
            "source_video_width": width,
            "source_video_height": height,
            "full_frame_geometry_uniform": True,
        }
    )
    write_source.attrs.update(
        {
            "source_video_width": width,
            "source_video_height": height,
            "width": width,
            "height": height,
            "full_frame_geometry_schema": FULL_FRAME_GEOMETRY_SCHEMA,
            "full_frame_geometry_source": "validated_source_detect_runs_metadata_repair",
            "source_slices": expected_slices,
            "source_validation": repaired_validation,
            "full_frame_geometry_repair": repair_record,
        }
    )
    _stamp_parent_geometry(
        write_root,
        width=width,
        height=height,
        source_group_path=source_group_path,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--recording-frame-index", type=Path, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = repair_clipped_detect_quality_source_geometry(
        args.zarr_path,
        plan_path=args.plan,
        source_run=args.source_run,
        recording_frame_index=args.recording_frame_index,
        apply=args.apply,
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json.expanduser().resolve(), report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"{report['status']}: {report['repair']['source_group_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Add stable observation keys to a refined subject-mask run after exact lineage validation.

The repair is additive and dry-run by default.  It never reads dense mask
pixels.  The selected refined-keypoint and refined-subject-mask row lineages
must match block-for-block before ``instance_key`` is copied through a
same-directory temporary indexed-sharded array and atomically published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.zarr_run_completion import is_run_complete


BACKFILL_SCHEMA = "palette.refined_subject_mask_instance_key_backfill.v1"
DEFAULT_BLOCK_ROWS = 131_072
DEFAULT_INNER_ROWS = 16_384
LINEAGE_ARRAYS = (
    "detection_indices",
    "detection_source",
    "frame_counts",
    "frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_crop_row_ids",
    "source_detect_row_index",
    "source_frame_indices",
    "source_refined_row_ids",
)


def _open(path: Path, *, mode: str) -> Any:
    return zarr.open_group(str(path), mode=mode, use_consolidated=False)


def _selected_complete_run(
    root: Any,
    parent_name: str,
    explicit_run: str | None,
) -> tuple[str, Any]:
    parent = root.get(parent_name)
    if parent is None:
        raise ValueError(f"Archive is missing {parent_name}.")
    run_name = str(
        explicit_run or parent.attrs.get("latest_complete") or parent.attrs.get("latest") or ""
    ).strip()
    if not run_name or "/" in run_name or run_name not in parent:
        raise ValueError(f"{parent_name} has no resolvable selected run.")
    run = parent[run_name]
    if not is_run_complete(run):
        raise ValueError(f"Selected run is not complete: {parent_name}/{run_name}.")
    return run_name, run


def _update_digest(digest: Any, values: np.ndarray) -> None:
    digest.update(np.ascontiguousarray(values).view(np.uint8))


def _validate_lineage(
    keypoints: Any,
    masks: Any,
    *,
    block_rows: int,
) -> tuple[list[dict[str, Any]], int]:
    reports: list[dict[str, Any]] = []
    row_count: int | None = None
    for name in LINEAGE_ARRAYS:
        if name not in keypoints or name not in masks:
            raise ValueError(f"Exact mask/keypoint lineage requires shared array {name}.")
        left = keypoints[name]
        right = masks[name]
        if tuple(int(value) for value in left.shape) != tuple(int(value) for value in right.shape):
            raise ValueError(
                f"Lineage shape mismatch for {name}: {left.shape} != {right.shape}."
            )
        if np.dtype(left.dtype) != np.dtype(right.dtype):
            raise ValueError(
                f"Lineage dtype mismatch for {name}: {left.dtype} != {right.dtype}."
            )
        if name != "frame_counts":
            current_rows = int(left.shape[0])
            if row_count is None:
                row_count = current_rows
            elif current_rows != row_count:
                raise ValueError(
                    f"Lineage row count mismatch within keypoint/mask surfaces for {name}."
                )
        digest = hashlib.sha256()
        for start in range(0, int(left.shape[0]), int(block_rows)):
            stop = min(start + int(block_rows), int(left.shape[0]))
            left_values = np.asarray(left[start:stop, ...])
            right_values = np.asarray(right[start:stop, ...])
            if not np.array_equal(left_values, right_values):
                unequal = np.flatnonzero(left_values != right_values)
                first = start + int(unequal[0]) if unequal.size else start
                raise ValueError(f"Lineage mismatch for {name} at flattened row {first}.")
            _update_digest(digest, left_values)
        reports.append(
            {
                "array": name,
                "shape": [int(value) for value in left.shape],
                "dtype": str(left.dtype),
                "sha256": digest.hexdigest(),
                "exact": True,
            }
        )
    if row_count is None:
        raise ValueError("Could not resolve mask/keypoint row count.")
    return reports, row_count


def _existing_key_status(keypoints: Any, masks: Any, *, block_rows: int) -> str:
    if "instance_key" not in keypoints:
        raise ValueError("Selected refined-keypoint run is missing required instance_key.")
    keys = keypoints["instance_key"]
    if int(keys.ndim) != 1:
        raise ValueError("Selected refined-keypoint instance_key must be one-dimensional.")
    if "instance_key" not in masks:
        return "add"
    existing = masks["instance_key"]
    if tuple(existing.shape) != tuple(keys.shape) or np.dtype(existing.dtype) != np.dtype(keys.dtype):
        raise ValueError("Existing refined-mask instance_key has an incompatible contract.")
    for start in range(0, int(keys.shape[0]), int(block_rows)):
        stop = min(start + int(block_rows), int(keys.shape[0]))
        if not np.array_equal(
            np.asarray(existing[start:stop], dtype=np.uint64),
            np.asarray(keys[start:stop], dtype=np.uint64),
        ):
            raise ValueError("Existing refined-mask instance_key disagrees with exact keypoint lineage.")
    return "verify_existing"


def _copy_keys_atomically(
    *,
    zarr_path: Path,
    mask_run_path: str,
    keypoint_run_path: str,
    row_count: int,
    block_rows: int,
    inner_rows: int,
) -> dict[str, Any]:
    root = _open(zarr_path, mode="a")
    masks = root[mask_run_path]
    keypoints = root[keypoint_run_path]
    if "instance_key" in masks:
        return {"action": "verified_existing", "rows": row_count}
    temp_name = f"_instance_key_backfill_{uuid4().hex}"
    chunks = (max(1, min(int(inner_rows), max(1, row_count))),)
    outer_rows = int(math.ceil(int(block_rows) / chunks[0]) * chunks[0])
    destination = masks.create_array(
        temp_name,
        shape=(row_count,),
        dtype=np.uint64,
        chunks=chunks,
        shards=(outer_rows,),
        overwrite=False,
    )
    source_digest = hashlib.sha256()
    destination_digest = hashlib.sha256()
    for start in range(0, row_count, outer_rows):
        stop = min(start + outer_rows, row_count)
        values = np.asarray(keypoints["instance_key"][start:stop], dtype=np.uint64)
        _update_digest(source_digest, values)
        destination[start:stop] = values
        reread = np.asarray(destination[start:stop], dtype=np.uint64)
        _update_digest(destination_digest, reread)
    if source_digest.hexdigest() != destination_digest.hexdigest():
        raise RuntimeError("Temporary refined-mask instance_key failed decoded validation.")
    del destination
    del masks
    del keypoints
    del root

    mask_dir = zarr_path.joinpath(*mask_run_path.split("/"))
    temp_dir = mask_dir / temp_name
    final_dir = mask_dir / "instance_key"
    if final_dir.exists():
        raise FileExistsError(f"Refined-mask instance_key appeared during repair: {final_dir}")
    os.replace(temp_dir, final_dir)

    verify_root = _open(zarr_path, mode="r")
    verify_masks = verify_root[mask_run_path]
    verify_keys = verify_root[keypoint_run_path]["instance_key"]
    for start in range(0, row_count, outer_rows):
        stop = min(start + outer_rows, row_count)
        if not np.array_equal(
            np.asarray(verify_masks["instance_key"][start:stop], dtype=np.uint64),
            np.asarray(verify_keys[start:stop], dtype=np.uint64),
        ):
            raise RuntimeError("Published refined-mask instance_key failed reread validation.")
    return {
        "action": "written",
        "rows": row_count,
        "chunks": list(chunks),
        "shards": [outer_rows],
        "sha256": source_digest.hexdigest(),
    }


def backfill_refined_subject_mask_instance_keys(
    *,
    zarr_path: str | Path,
    keypoint_run: str | None = None,
    mask_run: str | None = None,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    inner_rows: int = DEFAULT_INNER_ROWS,
    apply: bool = False,
) -> dict[str, Any]:
    archive = Path(zarr_path).expanduser().resolve()
    if int(block_rows) <= 0 or int(inner_rows) <= 0:
        raise ValueError("block_rows and inner_rows must be positive.")
    root = _open(archive, mode="r")
    keypoint_name, keypoints = _selected_complete_run(
        root, "refined_keypoints_runs", keypoint_run
    )
    mask_name, masks = _selected_complete_run(
        root, "refined_subject_masks_runs", mask_run
    )
    keypoint_path = f"refined_keypoints_runs/{keypoint_name}"
    mask_path = f"refined_subject_masks_runs/{mask_name}"
    lineage, row_count = _validate_lineage(
        keypoints,
        masks,
        block_rows=int(block_rows),
    )
    if "instance_key" not in keypoints:
        raise ValueError("Selected refined-keypoint run is missing required instance_key.")
    if int(keypoints["instance_key"].shape[0]) != row_count:
        raise ValueError("Refined-keypoint instance_key length does not match exact row lineage.")
    key_values = np.asarray(keypoints["instance_key"][:], dtype=np.uint64)
    if int(np.unique(key_values).shape[0]) != row_count:
        raise ValueError("Selected refined-keypoint instance_key values are not unique.")
    planned_action = _existing_key_status(
        keypoints,
        masks,
        block_rows=int(block_rows),
    )
    result: dict[str, Any] = {
        "schema": BACKFILL_SCHEMA,
        "status": "planned" if not apply else "running",
        "zarr_path": str(archive),
        "keypoint_run": keypoint_name,
        "keypoint_run_path": keypoint_path,
        "mask_run": mask_name,
        "mask_run_path": mask_path,
        "row_count": row_count,
        "lineage_validation": lineage,
        "planned_action": planned_action,
        "block_rows": int(block_rows),
        "inner_rows": int(inner_rows),
    }
    if not apply:
        return result

    copy_result = _copy_keys_atomically(
        zarr_path=archive,
        mask_run_path=mask_path,
        keypoint_run_path=keypoint_path,
        row_count=row_count,
        block_rows=int(block_rows),
        inner_rows=int(inner_rows),
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    write_root = _open(archive, mode="a")
    write_masks = write_root[mask_path]
    write_masks.attrs.update(
        {
            "instance_key_available": True,
            "instance_key_policy": "copied_from_exact_refined_keypoint_lineage",
            "instance_key_source_run_path": keypoint_path,
            "instance_key_lineage_arrays_validated": list(LINEAGE_ARRAYS),
            "instance_key_lineage_validation_status": "exact",
            "instance_key_backfill_schema": BACKFILL_SCHEMA,
            "instance_key_backfill_completed_at_utc": completed_at,
        }
    )
    result.update(
        {
            "status": "complete",
            "completed_at_utc": completed_at,
            "copy": copy_result,
        }
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--keypoint-run", default=None)
    parser.add_argument("--mask-run", default=None)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--inner-rows", type=int, default=DEFAULT_INNER_ROWS)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = backfill_refined_subject_mask_instance_keys(
        zarr_path=args.zarr_path,
        keypoint_run=args.keypoint_run,
        mask_run=args.mask_run,
        block_rows=int(args.block_rows),
        inner_rows=int(args.inner_rows),
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: {result['keypoint_run_path']} -> "
            f"{result['mask_run_path']}/instance_key rows={result['row_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

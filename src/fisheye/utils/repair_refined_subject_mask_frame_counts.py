#!/usr/bin/env python3
"""Transactionally repair recording-wide refined subject-mask frame counts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.utils.import_refined_subject_mask_clip_packages import (
    FRAME_COUNTS_INNER_ROWS,
    FRAME_COUNTS_SHARD_ROWS,
    compute_recording_frame_counts,
    resolve_recording_frame_count,
    write_recording_frame_counts,
)


REPAIR_SCHEMA = "palette.refined_subject_mask_frame_counts_repair.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_run(root: zarr.Group, run_name: str) -> zarr.Group:
    parent = root.get("refined_subject_masks_runs")
    if not isinstance(parent, zarr.Group) or run_name not in parent:
        raise ValueError(f"Missing refined_subject_masks_runs/{run_name}.")
    run = parent[run_name]
    if not isinstance(run, zarr.Group):
        raise ValueError(f"refined_subject_masks_runs/{run_name} is not a group.")
    if str(run.attrs.get("palette_run_completion_status") or "") != "complete":
        raise ValueError(f"Refusing to repair non-complete refined subject-mask run {run_name!r}.")
    return run


def _existing_summary(run: zarr.Group) -> dict[str, Any]:
    array = run.get("frame_counts")
    if array is None:
        return {"present": False, "shape": None, "sum": None}
    shape = tuple(int(value) for value in array.shape)
    values = np.asarray(array[:], dtype=np.int64).reshape(-1)
    return {
        "present": True,
        "shape": list(shape),
        "sum": int(values.sum(dtype=np.int64)),
        "chunks": [int(value) for value in (getattr(array, "chunks", None) or ())],
    }


def _write_receipt(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def repair_frame_counts(
    *,
    zarr_path: Path,
    run_name: str,
    execute: bool,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    zarr_path = zarr_path.expanduser().resolve()
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = _resolve_run(root, run_name)
    source_crop_run = str(run.attrs.get("source_crop_run") or "")
    if not source_crop_run:
        raise ValueError(f"refined_subject_masks_runs/{run_name} has no source_crop_run.")
    recording_frame_count, authorities = resolve_recording_frame_count(
        root,
        run,
        source_crop_run=source_crop_run,
    )
    frame_indices = run.get("frame_indices")
    if frame_indices is None:
        raise ValueError(f"refined_subject_masks_runs/{run_name} is missing frame_indices.")
    expected = compute_recording_frame_counts(
        frame_indices,
        recording_frame_count=recording_frame_count,
    )
    row_count = int(frame_indices.shape[0])
    if int(expected.sum(dtype=np.int64)) != row_count:
        raise ValueError(
            f"Rebuilt frame_counts sum {int(expected.sum(dtype=np.int64))} != row count {row_count}."
        )
    before = _existing_summary(run)
    existing = run.get("frame_counts")
    already_correct = bool(
        existing is not None
        and tuple(int(value) for value in existing.shape) == (recording_frame_count,)
        and np.array_equal(np.asarray(existing[:], dtype=np.int32).reshape(-1), expected)
    )
    base_report: dict[str, Any] = {
        "schema": REPAIR_SCHEMA,
        "created_at_utc": _utc_now(),
        "mode": "execute" if execute else "dry_run",
        "status": "already_correct" if already_correct else ("would_repair" if not execute else "running"),
        "analysis_zarr": str(zarr_path),
        "run_name": str(run_name),
        "run_path": f"refined_subject_masks_runs/{run_name}",
        "source_crop_run": source_crop_run,
        "recording_frame_count": int(recording_frame_count),
        "row_count": row_count,
        "expected_sum": int(expected.sum(dtype=np.int64)),
        "frame_count_authorities": dict(authorities),
        "before": before,
    }
    if already_correct or not execute:
        _write_receipt(receipt_path, base_report)
        return base_report

    candidate_name = "frame_counts_repair_candidate"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_name = f"frame_counts_repair_backup_{timestamp}"
    run_path = zarr_path / "refined_subject_masks_runs" / run_name
    frame_counts_path = run_path / "frame_counts"
    candidate_path = run_path / candidate_name
    backup_path = run_path / backup_name

    del existing
    del run
    del root
    gc.collect()

    writable_root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    writable_run = _resolve_run(writable_root, run_name)
    if candidate_name in writable_run:
        del writable_run[candidate_name]
    write_recording_frame_counts(
        writable_run,
        expected,
        name=candidate_name,
        source_candidates=authorities,
    )
    del writable_run
    del writable_root
    gc.collect()

    old_moved = False
    new_published = False
    try:
        if frame_counts_path.exists():
            os.replace(frame_counts_path, backup_path)
            old_moved = True
        os.replace(candidate_path, frame_counts_path)
        new_published = True

        verify_root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
        verify_run = _resolve_run(verify_root, run_name)
        observed = np.asarray(verify_run["frame_counts"][:], dtype=np.int32).reshape(-1)
        if not np.array_equal(observed, expected):
            raise ValueError("Published frame_counts failed exact post-swap verification.")
        repair_record = {
            "schema": REPAIR_SCHEMA,
            "created_at_utc": _utc_now(),
            "old_shape": before.get("shape"),
            "old_sum": before.get("sum"),
            "new_shape": [int(recording_frame_count)],
            "new_sum": int(observed.sum(dtype=np.int64)),
            "generation": "bincount_of_existing_frame_indices_v1",
            "frame_count_authorities": dict(authorities),
        }
        history = list(verify_run.attrs.get("maintenance_repairs") or [])
        history.append(repair_record)
        verify_run.attrs["maintenance_repairs"] = history
        verify_run.attrs["frame_counts_generation"] = "bincount_of_assembled_frame_indices_v1"
        verify_run.attrs["recording_frame_count"] = int(recording_frame_count)
        verify_run.attrs["frame_counts_authorities"] = dict(authorities)
        if str(verify_run.attrs.get("palette_run_completion_status") or "") != "complete":
            raise ValueError("Repair unexpectedly changed the run completion status.")
        del verify_run
        del verify_root
        gc.collect()
        if backup_path.exists():
            shutil.rmtree(backup_path)
    except Exception:
        if new_published and frame_counts_path.exists():
            shutil.rmtree(frame_counts_path)
        if old_moved and backup_path.exists():
            os.replace(backup_path, frame_counts_path)
        raise

    report = {
        **base_report,
        "status": "repaired",
        "completed_at_utc": _utc_now(),
        "after": {
            "shape": [int(recording_frame_count)],
            "sum": int(expected.sum(dtype=np.int64)),
            "dtype": "int32",
            "logical_chunk_rows": min(FRAME_COUNTS_INNER_ROWS, int(recording_frame_count)),
            "shard_rows_requested": FRAME_COUNTS_SHARD_ROWS,
            "exact_bincount_match": True,
        },
        "publication": "same_filesystem_atomic_directory_swap",
        "backup_removed_after_verification": True,
    }
    _write_receipt(receipt_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    report = repair_frame_counts(
        zarr_path=args.zarr,
        run_name=str(args.run),
        execute=bool(args.execute),
        receipt_path=args.receipt,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

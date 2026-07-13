"""Promote a completed refined subject-mask run by validated copy, not recompute."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.subject_mask_registry_status import (
    emit_refined_subject_mask_stage_completion,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_ATTR,
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_STATUS_COMPLETE,
    is_run_complete_in_parent,
)
from fisheye.utils.run_subject_mask_batch_pipeline import (
    _commit_run_group_publish,
    _prepare_run_group_publish,
    _refresh_subject_mask_registry_views,
)
from fisheye.utils.validate_refined_subject_mask_contract import (
    validate_refined_subject_mask_contract,
)


REPORT_SCHEMA = "palette.refined_subject_mask_copy_promotion.v1"
RUN_PARENT = "refined_subject_masks_runs"
REVIEW_POINTER = "refined_subject_mask_review_status_latest"


@dataclass(frozen=True)
class TreeInventory:
    file_count: int
    apparent_bytes: int
    path_size_sha256: str
    metadata_sha256: str
    metadata_file_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "file_count": self.file_count,
            "apparent_bytes": self.apparent_bytes,
            "path_size_sha256": self.path_size_sha256,
            "metadata_sha256": self.metadata_sha256,
            "metadata_file_count": self.metadata_file_count,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_name(run_name: str) -> str:
    name = str(run_name).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("run name must be one non-empty Zarr path component")
    return name


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _attrs_from_metadata(group_path: Path) -> dict[str, Any]:
    payload = _read_json(group_path / "zarr.json")
    attrs = payload.get("attributes")
    if not isinstance(attrs, dict):
        raise ValueError(f"Zarr group attributes are missing: {group_path / 'zarr.json'}")
    return dict(attrs)


def _validate_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path.expanduser().resolve())
    corrected = payload.get("corrected_validation")
    dense = payload.get("dense_content_audit")
    checks = {
        "classification_pass": payload.get("classification") == "pass",
        "finalizer_rerun_not_required": payload.get("finalizer_rerun_required") is False,
        "publication_rerun_not_required": payload.get("publication_rerun_required") is False,
        "corrected_validation_pass": isinstance(corrected, Mapping)
        and corrected.get("all_checks_pass") is True,
        "dense_content_audit_pass": isinstance(dense, Mapping)
        and dense.get("all_checks_pass") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Promotion evidence is not a complete pass: {checks}")
    return {
        "path": str(path.expanduser().resolve()),
        "schema_id": payload.get("schema_id"),
        "classification": payload.get("classification"),
        "lsf_exit_interpretation": payload.get("lsf_exit_interpretation"),
        "checks": checks,
    }


def _tree_inventory(root: Path) -> TreeInventory:
    path_size = hashlib.sha256()
    metadata = hashlib.sha256()
    file_count = 0
    apparent_bytes = 0
    metadata_file_count = 0
    for directory, directory_names, file_names in os.walk(root):
        directory_path = Path(directory)
        directory_names.sort()
        for name in directory_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(f"Run-group trees may not contain symlink directories: {candidate}")
        for name in sorted(file_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(f"Run-group trees may not contain symlink files: {candidate}")
            relative = candidate.relative_to(root).as_posix()
            size = int(candidate.stat().st_size)
            file_count += 1
            apparent_bytes += size
            path_size.update(relative.encode("utf-8"))
            path_size.update(b"\0")
            path_size.update(str(size).encode("ascii"))
            path_size.update(b"\0")
            if name == "zarr.json":
                metadata_file_count += 1
                metadata.update(relative.encode("utf-8"))
                metadata.update(b"\0")
                with candidate.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        metadata.update(block)
                metadata.update(b"\0")
    return TreeInventory(
        file_count=file_count,
        apparent_bytes=apparent_bytes,
        path_size_sha256=path_size.hexdigest(),
        metadata_sha256=metadata.hexdigest(),
        metadata_file_count=metadata_file_count,
    )


def _arrays(group: zarr.Group, prefix: str = "") -> dict[str, zarr.Array]:
    result: dict[str, zarr.Array] = {}
    for name, member in group.members():
        path = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(member, zarr.Array):
            result[path] = member
        elif isinstance(member, zarr.Group):
            result.update(_arrays(member, path))
    return result


def _array_values_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _row_step(array: zarr.Array) -> int:
    chunks = tuple(int(value) for value in array.chunks)
    first = max(1, chunks[0]) if chunks else 1
    if array.ndim >= 3 and tuple(int(value) for value in array.shape[-2:]) == (512, 512):
        return min(64, first)
    return min(65536, first)


def _compare_array_groups(source_path: Path, copied_path: Path) -> dict[str, Any]:
    source = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    copied = zarr.open_group(str(copied_path), mode="r", use_consolidated=False)
    source_arrays = _arrays(source)
    copied_arrays = _arrays(copied)
    if source_arrays.keys() != copied_arrays.keys():
        raise ValueError(
            "Copied run array paths differ: "
            f"missing_in_copy={sorted(source_arrays.keys() - copied_arrays.keys())}, "
            f"extra_in_copy={sorted(copied_arrays.keys() - source_arrays.keys())}"
        )
    compared_chunks = 0
    for path in sorted(source_arrays):
        left = source_arrays[path]
        right = copied_arrays[path]
        if tuple(left.shape) != tuple(right.shape) or str(left.dtype) != str(right.dtype):
            raise ValueError(f"Copied array shape/dtype differs: {path}")
        if tuple(left.chunks) != tuple(right.chunks):
            raise ValueError(f"Copied array chunking differs: {path}")
        if left.ndim == 0:
            compared_chunks += 1
            if not _array_values_equal(np.asarray(left[()]), np.asarray(right[()])):
                raise ValueError(f"Copied scalar values differ: {path}")
            continue
        step = _row_step(left)
        for start in range(0, int(left.shape[0]), step):
            stop = min(int(left.shape[0]), start + step)
            compared_chunks += 1
            if not _array_values_equal(
                np.asarray(left[start:stop]),
                np.asarray(right[start:stop]),
            ):
                raise ValueError(f"Copied array values differ: {path} rows {start}:{stop}")
    return {
        "array_count": len(source_arrays),
        "compared_chunks": compared_chunks,
        "all_values_equal": True,
    }


def _contract_summary(zarr_path: Path, run_name: str) -> dict[str, Any]:
    summary = dict(
        validate_refined_subject_mask_contract(zarr_path, run_name=run_name)
    )
    if not bool(summary.get("valid")):
        raise ValueError(
            f"Refined subject-mask contract is invalid for {zarr_path}: "
            f"{summary.get('errors')}"
        )
    return summary


def _source_contract_summary(zarr_path: Path, run_name: str) -> dict[str, Any]:
    """Validate source-local surfaces while allowing isolated-artifact lineage context."""

    summary = dict(
        validate_refined_subject_mask_contract(zarr_path, run_name=run_name)
    )
    if bool(summary.get("valid")):
        summary["target_context_validation_deferred"] = False
        return summary
    errors = list(summary.get("errors") or [])
    error_codes = {
        str(item.get("code"))
        for item in errors
        if isinstance(item, Mapping)
    }
    if error_codes and error_codes <= {"missing_source_crop_run"}:
        summary["target_context_validation_deferred"] = True
        summary["target_context_deferred_reason"] = (
            "isolated published artifact does not carry canonical crop_runs; "
            "the same run must pass the full contract after canonical publication"
        )
        return summary
    raise ValueError(
        f"Refined subject-mask source contract is invalid for {zarr_path}: {errors}"
    )


def _storage_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: stats.get(key)
        for key in (
            "file_count",
            "metadata_file_count",
            "payload_file_count",
            "apparent_bytes",
            "allocated_bytes",
            "array_count",
            "metadata_error_count",
            "stat_error_count",
            "scan_duration_seconds",
        )
    }


def _row_count(group_path: Path) -> int:
    metadata = _read_json(group_path / "masks_roi" / "zarr.json")
    shape = metadata.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError(f"masks_roi must declare a four-dimensional shape: {group_path}")
    return int(shape[0])


def _parent_pointer_snapshot(parent: zarr.Group) -> dict[str, Any]:
    return {
        key: parent.attrs.get(key)
        for key in ("latest", RUN_LATEST_COMPLETE_ATTR, REVIEW_POINTER)
    }


def _restore_parent_pointers(parent: zarr.Group, snapshot: Mapping[str, Any]) -> None:
    for key, value in snapshot.items():
        if value is None:
            if key in parent.attrs:
                del parent.attrs[key]
        else:
            parent.attrs[key] = value


def _promote_pointers_and_registry(
    *,
    target_zarr: Path,
    run_name: str,
    registry_path: Path | None,
) -> dict[str, Any]:
    root = zarr.open_group(str(target_zarr), mode="a", use_consolidated=False)
    parent = root[RUN_PARENT]
    run = parent[run_name]
    if not is_run_complete_in_parent(parent, run, legacy_default=False):
        raise ValueError(f"Copied run is not complete: {RUN_PARENT}/{run_name}")
    before = _parent_pointer_snapshot(parent)
    parent.attrs.update(
        {
            "latest": run_name,
            RUN_LATEST_COMPLETE_ATTR: run_name,
            REVIEW_POINTER: run_name,
        }
    )
    registry_refresh: dict[str, Any]
    try:
        if registry_path is None:
            registry_refresh = {
                "registry_refresh_status": "skipped",
                "reason": "no_registry",
            }
        else:
            emitted = emit_refined_subject_mask_stage_completion(
                root,
                target_zarr,
                run_group=run,
                run_name=run_name,
                source="copy_promoted_completed_refined_subject_mask_run",
                registry=registry_path,
                invalidate_on_ok=True,
            )
            if not emitted:
                raise RuntimeError("Failed to emit refined-subject-mask registry completion")
            registry_refresh = _refresh_subject_mask_registry_views(
                registry_path=registry_path,
                zarr_path=target_zarr,
            )
            if registry_refresh.get("registry_refresh_status") != "ok":
                raise RuntimeError(f"Subject-mask registry refresh failed: {registry_refresh}")
    except Exception:
        _restore_parent_pointers(parent, before)
        raise
    return {
        "before": before,
        "after": _parent_pointer_snapshot(parent),
        "registry": registry_refresh,
    }


def promote_refined_subject_mask_run(
    *,
    source_zarr: Path,
    source_run: str,
    target_zarr: Path,
    expected_rows: int | None = None,
    evidence_json: Path | None = None,
    registry_path: Path | None = None,
    apply: bool = False,
    resume_existing: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_zarr = source_zarr.expanduser().resolve()
    target_zarr = target_zarr.expanduser().resolve()
    source_run = _validate_run_name(source_run)
    source_path = source_zarr / RUN_PARENT / source_run
    target_parent_path = target_zarr / RUN_PARENT
    target_path = target_parent_path / source_run
    if source_zarr == target_zarr:
        raise ValueError("source and target Zarrs must differ")
    for required in (
        source_zarr / "zarr.json",
        source_path / "zarr.json",
        target_zarr / "zarr.json",
        target_parent_path / "zarr.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_attrs = _attrs_from_metadata(source_path)
    if source_attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError(f"Source run is not marked complete: {source_path}")
    declared_name = source_attrs.get("palette_run_name")
    if declared_name is not None and str(declared_name) != source_run:
        raise ValueError(
            f"Source run name attribute differs: {declared_name!r} != {source_run!r}"
        )
    target_parent_attrs = _attrs_from_metadata(target_parent_path)
    if int(target_parent_attrs.get(COMPLETION_EPOCH_ATTR, -1)) < int(
        COMPLETION_EPOCH_REQUIRE_PROVENANCE
    ):
        raise ValueError("Target refined-subject-mask parent is not provenance-strict")
    rows = _row_count(source_path)
    if expected_rows is not None and rows != int(expected_rows):
        raise ValueError(f"Source rows {rows} != expected rows {int(expected_rows)}")
    evidence = _validate_evidence(evidence_json)
    source_contract = _source_contract_summary(source_zarr, source_run)

    plan = {
        "schema": REPORT_SCHEMA,
        "status": "planned",
        "apply": bool(apply),
        "source_zarr": str(source_zarr),
        "source_run": source_run,
        "source_path": str(source_path),
        "target_zarr": str(target_zarr),
        "target_path": str(target_path),
        "expected_rows": expected_rows,
        "source_rows": rows,
        "target_preexisting": target_path.exists(),
        "resume_existing": bool(resume_existing),
        "evidence": evidence,
        "source_contract": source_contract,
        "registry_path": None if registry_path is None else str(registry_path),
    }
    if target_path.exists() and not resume_existing:
        raise FileExistsError(f"Canonical target already exists: {target_path}")
    if not apply:
        return plan

    copied = False
    publish_plan = None
    if target_path.exists():
        copied_path = target_path
        publish_summary = {
            "mode": "resume_existing",
            "copy_seconds": 0.0,
        }
    else:
        publish_plan = _prepare_run_group_publish(
            staged_parent=source_zarr / RUN_PARENT,
            target_parent=target_parent_path,
            run_name=source_run,
            overwrite=False,
        )
        copied_path = publish_plan.tmp_path
        publish_summary = {
            "mode": "copy_then_atomic_rename",
            "copy_seconds": publish_plan.copy_duration_seconds,
            "publish_backend": publish_plan.publish_backend,
            "source_storage": _storage_summary(publish_plan.storage_stats),
            "temporary_path": str(publish_plan.tmp_path),
        }

    try:
        source_inventory = _tree_inventory(source_path)
        copied_inventory = _tree_inventory(copied_path)
        if source_inventory != copied_inventory:
            raise ValueError(
                "Copied tree inventory or Zarr metadata differs: "
                f"source={source_inventory.to_json()} copied={copied_inventory.to_json()}"
            )
        array_validation = _compare_array_groups(source_path, copied_path)
        if publish_plan is not None:
            _commit_run_group_publish(publish_plan)
            copied = True
        target_contract = _contract_summary(target_zarr, source_run)
        pointer_registry = _promote_pointers_and_registry(
            target_zarr=target_zarr,
            run_name=source_run,
            registry_path=(
                None if registry_path is None else registry_path.expanduser().resolve()
            ),
        )
    except Exception:
        if publish_plan is not None and publish_plan.tmp_path.exists():
            shutil.rmtree(publish_plan.tmp_path)
        raise

    completed = _utc_now()
    report = {
        **plan,
        "status": "promoted",
        "apply": True,
        "copied": copied,
        "publish": publish_summary,
        "source_inventory": source_inventory.to_json(),
        "target_inventory": copied_inventory.to_json(),
        "array_validation": array_validation,
        "target_contract": target_contract,
        "pointers_and_registry": pointer_registry,
        "host": socket.gethostname(),
        "lsb_job_id": os.environ.get("LSB_JOBID"),
        "completed_at_utc": completed,
        "duration_seconds": float(time.perf_counter() - started),
    }
    receipt_path = target_parent_path / ".imports" / f"{source_run}.json"
    report["receipt_path"] = str(receipt_path)
    write_json_atomic(receipt_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", required=True, type=Path)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-zarr", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--evidence-json", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--resume-existing", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = promote_refined_subject_mask_run(
        source_zarr=args.source_zarr,
        source_run=args.source_run,
        target_zarr=args.target_zarr,
        expected_rows=args.expected_rows,
        evidence_json=args.evidence_json,
        registry_path=args.registry,
        apply=bool(args.apply),
        resume_existing=bool(args.resume_existing),
    )
    if args.output_json is not None:
        write_json_atomic(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA",
    "TreeInventory",
    "main",
    "promote_refined_subject_mask_run",
]

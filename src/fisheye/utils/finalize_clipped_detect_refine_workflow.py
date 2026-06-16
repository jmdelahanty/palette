"""Finalize a clipped detect/refine workflow into a manifest-backed collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA
from fisheye.utils.system import get_environment_summary, get_git_info, get_platform_info
from fisheye.utils.validate_refined_detect_run import validate_refined_detect_run


FINALIZER_SCHEMA = "palette.clipped_detect_refine_finalizer.v1"
COLLECTION_SCHEMA = "palette.refined_detect_clip_collection.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode=mode)


def _load_plan(plan_path: Path) -> dict[str, Any]:
    payload = _read_json(plan_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Plan JSON must be an object: {plan_path}")
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError(f"Expected {PLAN_SCHEMA}, got {payload.get('schema_version')!r}: {plan_path}")
    units = payload.get("work_units")
    if not isinstance(units, list) or not units:
        raise ValueError(f"Plan has no work_units: {plan_path}")
    return payload


def _load_submission_manifest(path: Optional[Path]) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Submission manifest must be an object: {path}")
    return payload


def _resolve_recording_frame_index(plan: Mapping[str, Any]) -> Path | None:
    recording_dir = plan.get("recording_dir")
    if not recording_dir:
        return None
    recording_path = Path(str(recording_dir)).expanduser().resolve()
    manifest_path = recording_path / "recording_frame_index_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"recording_frame_index_manifest.json is not an object: {manifest_path}")
    raw_path = manifest.get("recording_frame_index_path") or "recording_frame_index.parquet"
    frame_index_path = Path(str(raw_path)).expanduser()
    if frame_index_path.is_absolute():
        resolved = frame_index_path.resolve()
        relocated = recording_path / frame_index_path.name
        if relocated.exists() and not resolved.is_relative_to(recording_path):
            return relocated.resolve()
        if resolved.exists():
            return resolved
        if relocated.exists():
            return relocated.resolve()
        return resolved
    return (recording_path / frame_index_path).resolve()


def _validate_recording_frame_index(
    frame_index_path: Path | None,
    work_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if frame_index_path is None:
        return None, ["recording_frame_index.parquet not available from plan recording_dir"], []
    if not frame_index_path.exists():
        return None, [f"recording_frame_index.parquet not found: {frame_index_path}"], []

    required_columns = [
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "recording_frame_id",
    ]
    table = pq.read_table(frame_index_path, columns=required_columns).combine_chunks()
    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "path": str(frame_index_path),
        "row_count": int(table.num_rows),
        "work_unit_count": len(work_units),
        "units": [],
    }
    if table.num_rows == 0:
        errors.append(f"recording_frame_index.parquet has zero rows: {frame_index_path}")
        return summary, errors, warnings

    camera_col = pc.cast(table["camera_serial"], "string")
    clip_col = pc.cast(table["clip_id"], "string")
    local_col = table["clip_local_frame_index"]
    recording_frame_col = table["recording_frame_id"]
    observed_pairs: set[tuple[str, str]] = set()

    for unit in work_units:
        clip_id = str(unit.get("clip_id") or "")
        camera_serial = str(unit.get("camera_serial") or "")
        expected_frame_count = unit.get("frame_count")
        mask = pc.and_(pc.equal(clip_col, clip_id), pc.equal(camera_col, camera_serial))
        local_values = pc.filter(local_col, mask).to_pylist()
        recording_frame_values = pc.filter(recording_frame_col, mask).to_pylist()
        observed_pairs.add((camera_serial, clip_id))
        unit_summary = {
            "clip_id": clip_id,
            "camera_serial": camera_serial,
            "frame_index_rows": len(local_values),
            "expected_frame_count": expected_frame_count,
        }
        summary["units"].append(unit_summary)
        if expected_frame_count is not None and len(local_values) != int(expected_frame_count):
            errors.append(
                f"{clip_id} cam{camera_serial}: recording_frame_index row count {len(local_values)} "
                f"does not match planned frame_count {expected_frame_count}"
            )
        if local_values:
            local_ints = sorted(int(value) for value in local_values)
            expected = list(range(len(local_ints)))
            if local_ints != expected:
                errors.append(
                    f"{clip_id} cam{camera_serial}: clip_local_frame_index is not contiguous 0..N-1"
                )
        if len(recording_frame_values) != len(set(recording_frame_values)):
            errors.append(f"{clip_id} cam{camera_serial}: duplicate recording_frame_id values")

    expected_pairs = {
        (str(unit.get("camera_serial") or ""), str(unit.get("clip_id") or ""))
        for unit in work_units
    }
    table_pairs = set(zip(camera_col.to_pylist(), clip_col.to_pylist()))
    missing_pairs = sorted(expected_pairs - table_pairs)
    unexpected_pairs = sorted(table_pairs - expected_pairs)
    if missing_pairs:
        errors.append(f"recording_frame_index missing planned clip-camera pairs: {missing_pairs[:20]}")
    if unexpected_pairs:
        warnings.append(f"recording_frame_index has clip-camera pairs outside selected plan: {unexpected_pairs[:20]}")

    for camera_serial in sorted({camera for camera, _clip in expected_pairs}):
        camera_mask = pc.equal(camera_col, camera_serial)
        recording_ids = [int(value) for value in pc.filter(recording_frame_col, camera_mask).to_pylist()]
        if len(recording_ids) != len(set(recording_ids)):
            errors.append(f"camera {camera_serial}: duplicate recording_frame_id values across frame index")
        sorted_ids = sorted(recording_ids)
        if sorted_ids:
            expected_ids = list(range(sorted_ids[0], sorted_ids[-1] + 1))
            if sorted_ids != expected_ids:
                errors.append(f"camera {camera_serial}: recording_frame_id values are not contiguous")

    return summary, errors, warnings


def _submission_statuses_by_work_unit(
    submission_manifest: Optional[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if not submission_manifest:
        return {}
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for item in submission_manifest.get("work_items", []):
        if not isinstance(item, Mapping):
            continue
        work_unit_id = str(item.get("work_unit_id") or "")
        if not work_unit_id:
            continue
        stage_records: list[dict[str, Any]] = []
        for stage in item.get("stages", []):
            if isinstance(stage, Mapping):
                stage_records.append(dict(stage))
        by_unit[work_unit_id] = stage_records
    return by_unit


def _validate_stage_statuses(
    *,
    work_unit_id: str,
    stage_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    if not stage_records:
        return reports, [f"{work_unit_id}: no stage records found in submission manifest"]
    for stage in stage_records:
        stage_name = str(stage.get("stage") or "unknown")
        status_path = Path(str(stage.get("status_json") or ""))
        record: dict[str, Any] = {
            "stage": stage_name,
            "status_json": str(status_path),
            "status": None,
        }
        if not status_path.exists():
            errors.append(f"{work_unit_id}: missing stage status JSON for {stage_name}: {status_path}")
            record["status"] = "missing"
            reports.append(record)
            continue
        try:
            payload = _read_json(status_path)
        except Exception as exc:
            errors.append(f"{work_unit_id}: unreadable stage status JSON for {stage_name}: {exc}")
            record["status"] = "unreadable"
            reports.append(record)
            continue
        if not isinstance(payload, Mapping):
            errors.append(f"{work_unit_id}: stage status JSON is not an object for {stage_name}")
            record["status"] = "invalid"
            reports.append(record)
            continue
        status = str(payload.get("status") or "")
        record.update(
            {
                "status": status,
                "job_id": payload.get("job_id"),
                "host": payload.get("host"),
                "stage_seconds": payload.get("stage_seconds"),
            }
        )
        if status != "ok":
            errors.append(f"{work_unit_id}: stage {stage_name} status is {status!r}, expected 'ok'")
        reports.append(record)
    return reports, errors


def _collection_group_path(collection_id: str) -> str:
    return f"experiment_index/finalized_runs/{collection_id}"


def _delete_if_present(group: zarr.Group, name: str) -> None:
    if name in group:
        del group[name]


def _write_collection_group(
    root: zarr.Group,
    *,
    collection_id: str,
    manifest: Mapping[str, Any],
    overwrite: bool,
) -> str:
    finalized = root.require_group("experiment_index").require_group("finalized_runs")
    if collection_id in finalized:
        if not overwrite:
            raise FileExistsError(f"Finalized collection already exists: {_collection_group_path(collection_id)}")
        _delete_if_present(finalized, collection_id)
    group = finalized.require_group(collection_id)
    for key, value in manifest.items():
        group.attrs[key] = value
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is None:
        raise ValueError("Cannot finalize clipped collection: refined_detect_runs is missing.")
    refined_parent.attrs["latest_collection"] = collection_id
    refined_parent.attrs["latest_collection_path"] = _collection_group_path(collection_id)
    return _collection_group_path(collection_id)


def finalize_clipped_detect_refine_workflow(
    plan_json: str | Path,
    *,
    submission_manifest: str | Path | None = None,
    collection_id: str | None = None,
    apply: bool = False,
    overwrite: bool = False,
    require_stage_status: Optional[bool] = None,
    require_recording_frame_index: bool = True,
) -> dict[str, Any]:
    plan_path = Path(plan_json).expanduser().resolve()
    plan = _load_plan(plan_path)
    submission_path = Path(submission_manifest).expanduser().resolve() if submission_manifest else None
    submission = _load_submission_manifest(submission_path)
    require_status = bool(submission_path) if require_stage_status is None else bool(require_stage_status)
    statuses_by_unit = _submission_statuses_by_work_unit(submission)
    zarr_path = Path(str(plan["analysis_zarr"])).expanduser().resolve()
    root = _open_root(zarr_path, mode="a" if apply else "r")
    workflow_id = str(plan.get("workflow_id") or "clipped_detect_refine")
    resolved_collection_id = collection_id or workflow_id

    errors: list[str] = []
    warnings: list[str] = []
    selected_runs: list[dict[str, Any]] = []
    unit_reports: list[dict[str, Any]] = []
    frame_index_summary, frame_index_errors, frame_index_warnings = _validate_recording_frame_index(
        _resolve_recording_frame_index(plan),
        [unit for unit in plan["work_units"] if isinstance(unit, Mapping)],
    )
    if require_recording_frame_index:
        errors.extend(frame_index_errors)
    else:
        warnings.extend(frame_index_errors)
    warnings.extend(frame_index_warnings)

    for unit in plan["work_units"]:
        if not isinstance(unit, Mapping):
            errors.append("plan contains a non-object work unit")
            continue
        work_unit_id = str(unit.get("work_unit_id") or "")
        clip_id = str(unit.get("clip_id") or "")
        camera_serial = str(unit.get("camera_serial") or "")
        target_group_path = str(unit.get("zarr_paths", {}).get("refined_group_path") or "")
        expected_frame_count = unit.get("frame_count")
        report: dict[str, Any] = {
            "work_unit_id": work_unit_id,
            "clip_id": clip_id,
            "camera_serial": camera_serial,
            "expected_frame_count": expected_frame_count,
            "refined_group_path": target_group_path,
        }

        if require_status:
            stage_reports, stage_errors = _validate_stage_statuses(
                work_unit_id=work_unit_id,
                stage_records=statuses_by_unit.get(work_unit_id, []),
            )
            report["stage_reports"] = stage_reports
            errors.extend(stage_errors)

        validation = validate_refined_detect_run(
            zarr_path,
            target_group_path=target_group_path,
        )
        report["refined_validation"] = validation
        if validation.get("status") != "ok":
            errors.append(f"{work_unit_id}: refined validation failed for {target_group_path}")
        frame_counts_length = validation.get("row_counts", {}).get("frame_counts")
        if expected_frame_count is not None and frame_counts_length is not None:
            if int(expected_frame_count) != int(frame_counts_length):
                errors.append(
                    f"{work_unit_id}: refined frame_counts length {frame_counts_length} "
                    f"does not match planned frame_count {expected_frame_count}"
                )
        elif expected_frame_count is not None:
            errors.append(f"{work_unit_id}: refined validation did not report frame_counts length")

        unit_reports.append(report)
        if validation.get("status") == "ok":
            selected_runs.append(
                {
                    "work_unit_id": work_unit_id,
                    "clip_id": clip_id,
                    "clip_index": unit.get("clip_index"),
                    "camera_serial": camera_serial,
                    "frame_count": expected_frame_count,
                    "detect_run": unit.get("run_names", {}).get("detect"),
                    "detect_quality_run": unit.get("run_names", {}).get("detect_quality"),
                    "refined_detect_run": unit.get("run_names", {}).get("refined_detect"),
                    "detect_group_path": unit.get("zarr_paths", {}).get("detect_target_group_path"),
                    "refined_group_path": target_group_path,
                    "source": unit.get("source"),
                }
            )

    camera_serials = sorted({str(row["camera_serial"]) for row in selected_runs})
    clip_ids = sorted({str(row["clip_id"]) for row in selected_runs})
    manifest = {
        "schema_version": COLLECTION_SCHEMA,
        "collection_id": resolved_collection_id,
        "collection_kind": "refined_detect_clip_collection",
        "workflow_id": workflow_id,
        "recording_id": plan.get("recording_id"),
        "analysis_zarr": str(zarr_path),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "submission_manifest_path": str(submission_path) if submission_path else None,
        "submission_manifest_sha256": _sha256_file(submission_path) if submission_path else None,
        "created_at_utc": _utc_now(),
        "work_unit_count": len(plan["work_units"]),
        "selected_run_count": len(selected_runs),
        "recording_frame_index_validation": frame_index_summary,
        "clip_count": len(clip_ids),
        "camera_serials": camera_serials,
        "clip_ids": clip_ids,
        "selected_runs": selected_runs,
        "provenance": {
            "command": " ".join(sys.argv) if sys.argv else None,
            "git": get_git_info(),
            "platform": get_platform_info(disk_path=str(zarr_path)),
            "environment": get_environment_summary(),
            "parameters": {
                "apply": bool(apply),
                "overwrite": bool(overwrite),
                "require_stage_status": bool(require_status),
            },
        },
    }

    collection_path = _collection_group_path(resolved_collection_id)
    status = "ok" if not errors and len(selected_runs) == len(plan["work_units"]) else "failed"
    if status == "failed" and apply:
        warnings.append("apply requested but collection was not written because finalizer validation failed")
    if status == "ok" and apply:
        collection_path = _write_collection_group(
            root,
            collection_id=resolved_collection_id,
            manifest=manifest,
            overwrite=overwrite,
        )

    return {
        "status": status,
        "schema_version": FINALIZER_SCHEMA,
        "collection_schema_version": COLLECTION_SCHEMA,
        "collection_id": resolved_collection_id,
        "collection_path": collection_path,
        "applied": bool(status == "ok" and apply),
        "analysis_zarr": str(zarr_path),
        "plan_path": str(plan_path),
        "submission_manifest_path": str(submission_path) if submission_path else None,
        "work_unit_count": len(plan["work_units"]),
        "selected_run_count": len(selected_runs),
        "unit_reports": unit_reports,
        "manifest": manifest,
        "errors": errors,
        "warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and optionally finalize a clipped detect/refine collection."
    )
    parser.add_argument("plan_json", type=Path, help="Plan JSON from plan_clipped_detect_refine_workflow")
    parser.add_argument("--submission-manifest", type=Path, default=None, help="Optional LSF submission manifest")
    parser.add_argument("--collection-id", default=None, help="Collection id; default is plan workflow_id")
    parser.add_argument("--apply", action="store_true", help="Write experiment_index/finalized_runs/<collection_id>")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing finalized collection")
    parser.add_argument(
        "--require-stage-status",
        action="store_true",
        help="Require stage status JSONs from --submission-manifest. Enabled automatically when a submission manifest is provided.",
    )
    parser.add_argument(
        "--no-require-stage-status",
        action="store_true",
        help="Do not require stage status JSONs even when --submission-manifest is provided.",
    )
    parser.add_argument(
        "--no-require-recording-frame-index",
        action="store_true",
        help="Do not fail if recording_frame_index.parquet is missing or inconsistent.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write finalizer report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.require_stage_status and args.no_require_stage_status:
        parser.error("choose at most one of --require-stage-status and --no-require-stage-status")
    require_status: Optional[bool]
    if args.require_stage_status:
        require_status = True
    elif args.no_require_stage_status:
        require_status = False
    else:
        require_status = None
    result = finalize_clipped_detect_refine_workflow(
        args.plan_json,
        submission_manifest=args.submission_manifest,
        collection_id=args.collection_id,
        apply=args.apply,
        overwrite=args.overwrite,
        require_stage_status=require_status,
        require_recording_frame_index=not args.no_require_recording_frame_index,
    )
    if args.output_json is not None:
        _write_json(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

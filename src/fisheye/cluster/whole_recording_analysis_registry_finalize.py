"""Serially reconcile exact keypoint and subject-mask runs for a combined DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.keypoints.registry_finalize import finalize_registry as finalize_keypoints
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.cluster.whole_recording_analysis import PLAN_SCHEMA
from fisheye.registry.db import Registry
from fisheye.shared.subject_mask_registry_status import (
    emit_refined_subject_mask_stage_completion,
    emit_subject_mask_stage_completion,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


REPORT_SCHEMA = "palette.whole_recording_analysis_registry_finalizer.v1"


def _require_complete(root: Any, parent_name: str, run_name: str) -> Any:
    parent = root.get(parent_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"Missing planned run {parent_name}/{run_name}")
    run = parent[run_name]
    if not is_run_complete_in_parent(parent, run):
        raise RuntimeError(f"Planned run is incomplete: {parent_name}/{run_name}")
    return run


def finalize_registry(
    run_root: Path,
    *,
    registry_path: Path,
    apply: bool,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    registry_path = registry_path.expanduser().resolve()
    payload = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported combined plan: {run_root / 'plan.json'}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Combined plan has no targets.")

    keypoint_report = finalize_keypoints(
        Path(str(payload["keypoint_plan_path"])).parent,
        registry_path=registry_path,
        apply=apply,
    )
    mask_reports: list[dict[str, Any]] = []
    for raw in targets:
        if not isinstance(raw, Mapping):
            raise ValueError("Combined plan target must be an object.")
        zarr_path = Path(str(raw["analysis_zarr"])).expanduser().resolve()
        subject_names = raw.get("subject_masks")
        if not isinstance(subject_names, Mapping):
            raise ValueError("Combined target lacks subject_masks run names.")
        subject_run_name = str(subject_names["subject_mask_run"])
        refined_run_name = str(subject_names["refined_subject_mask_run"])
        refined_keypoint_run = str(raw["refined_keypoint_run"])
        root = open_zarr_group_direct(zarr_path, mode="r")
        subject_run = _require_complete(root, "subject_mask_runs", subject_run_name)
        refined_run = _require_complete(
            root,
            "refined_subject_masks_runs",
            refined_run_name,
        )
        actual_keypoint_group = normalize_attr(
            refined_run.attrs.get("assignment_keypoint_group")
        )
        actual_keypoint_run = normalize_attr(
            refined_run.attrs.get("assignment_keypoints_run")
            or refined_run.attrs.get("assignment_keypoint_run")
        )
        expected_assignment = ("refined_keypoints_runs", refined_keypoint_run)
        if (actual_keypoint_group, actual_keypoint_run) != expected_assignment:
            raise RuntimeError(
                "Refined subject-mask assignment lineage mismatch: actual "
                f"{(actual_keypoint_group, actual_keypoint_run)}, expected "
                f"{expected_assignment}."
            )

        report: dict[str, Any] = {
            "target_id": str(raw.get("target_id") or ""),
            "analysis_zarr": str(zarr_path),
            "subject_mask_run": subject_run_name,
            "refined_subject_mask_run": refined_run_name,
            "assignment_keypoint_group": actual_keypoint_group,
            "assignment_keypoint_run": actual_keypoint_run,
            "status": "validated",
        }
        if apply:
            if not emit_subject_mask_stage_completion(
                root,
                zarr_path,
                run_group=subject_run,
                run_name=subject_run_name,
                source="serial_whole_recording_analysis_finalizer",
                registry=registry_path,
                invalidate_on_ok=True,
            ):
                raise RuntimeError("Subject-mask stage reconciliation returned false.")
            if not emit_refined_subject_mask_stage_completion(
                root,
                zarr_path,
                run_group=refined_run,
                run_name=refined_run_name,
                source="serial_whole_recording_analysis_finalizer",
                registry=registry_path,
                invalidate_on_ok=True,
            ):
                raise RuntimeError("Refined subject-mask reconciliation returned false.")
            registry = Registry(registry_path)
            try:
                row = registry.conn.execute(
                    "SELECT dataset_id, recording_id, zarr_use FROM datasets "
                    "WHERE zarr_path = ? ORDER BY dataset_id LIMIT 1;",
                    (str(zarr_path),),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"Dataset is not registered: {zarr_path}")
                report["subject_mask_performance_rows"] = int(
                    registry.refresh_subject_mask_performance_for_dataset(
                        str(row["dataset_id"]),
                        zarr_path=zarr_path,
                        recording_id=row["recording_id"],
                        zarr_use=row["zarr_use"],
                    )
                )
                report["subject_mask_component_quality_rows"] = int(
                    registry.refresh_subject_mask_component_quality_for_dataset(
                        str(row["dataset_id"]),
                        zarr_path=zarr_path,
                        recording_id=row["recording_id"],
                        zarr_use=row["zarr_use"],
                    )
                )
            finally:
                registry.close()
            report["status"] = "reconciled"
        mask_reports.append(report)

    registry = Registry(registry_path)
    try:
        integrity = str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    finally:
        registry.close()
    if apply and integrity != "ok":
        raise RuntimeError(f"Registry integrity check failed: {integrity}")
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok",
        "apply": bool(apply),
        "registry": str(registry_path),
        "registry_integrity": integrity,
        "keypoints": keypoint_report,
        "subject_masks": mask_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = finalize_registry(
        args.run_root,
        registry_path=args.registry,
        apply=bool(args.apply),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "finalize_registry", "main"]

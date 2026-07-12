"""Serially reconcile exact whole-recording keypoint runs into the registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.keypoints.whole_recording import PLAN_SCHEMA
from fisheye.cluster.lsf import write_json_snapshot
from fisheye.registry.db import Registry
from fisheye.registry.inline_refresh import refresh_keypoint_performance_details
from fisheye.registry.stage_complete import emit_stage_completion
from fisheye.refinement import refine_keypoints as refine_mod
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent


REPORT_SCHEMA = "palette.whole_recording_keypoint_registry_finalizer.v1"
_DISABLE_REGISTRY_WRITES_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Plan field {field!r} must be an object.")
    return dict(value)


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _check_registry_integrity(registry_path: Path) -> str:
    registry = Registry(registry_path)
    registry.conn.execute("PRAGMA busy_timeout=60000;")
    try:
        return str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    finally:
        registry.close()


def _require_complete_run(root: Any, *, parent_name: str, run_name: str) -> Any:
    parent = root.get(parent_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"Missing planned run {parent_name}/{run_name}")
    run_group = parent[run_name]
    if not is_run_complete_in_parent(parent, run_group):
        raise RuntimeError(f"Planned run is not complete: {parent_name}/{run_name}")
    return run_group


def _keypoint_statistics(run_group: Any) -> dict[str, Any]:
    summary = run_group.attrs.get("summary_statistics")
    summary_mapping = dict(summary) if isinstance(summary, Mapping) else {}
    total = _safe_int(summary_mapping.get("total_rois"))
    successful = _safe_int(summary_mapping.get("successful_detections"))
    failed = _safe_int(summary_mapping.get("failed_detections"))
    coverage = _safe_float(summary_mapping.get("success_rate_percent"))
    if total is None:
        try:
            total = int(run_group["keypoints_roi"].shape[0])
        except Exception:
            total = None
    if successful is None and total is not None:
        try:
            successful = int(run_group["detection_success"][:].sum())
        except Exception:
            successful = None
    if failed is None and total is not None and successful is not None:
        failed = total - successful
    if coverage is None and total and successful is not None:
        coverage = float(successful) / float(total) * 100.0
    return {
        "total_rois": total,
        "successful_detections": successful,
        "failed_detections": failed,
        "coverage_pct": coverage,
        "summary_statistics": summary_mapping,
    }


def _refined_statistics(run_group: Any) -> dict[str, Any]:
    summary = run_group.attrs.get("summary_statistics")
    summary_mapping = dict(summary) if isinstance(summary, Mapping) else {}
    total = _safe_int(summary_mapping.get("total_rois"))
    refined_success = _safe_int(summary_mapping.get("refined_success"))
    usable = _safe_int(summary_mapping.get("usable_keypoints"))
    coverage = _safe_float(summary_mapping.get("pass_rate_percent"))
    if total is None:
        try:
            total = int(run_group["keypoints_roi"].shape[0])
        except Exception:
            total = None
    if coverage is None and total and refined_success is not None:
        coverage = float(refined_success) / float(total) * 100.0
    return {
        "total_rois": total,
        "refined_success": refined_success,
        "usable_keypoints": usable,
        "coverage_pct": coverage,
        "summary_statistics": summary_mapping,
    }


def _target_rows(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_targets = plan.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Whole-recording plan has no targets.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(raw_targets, start=1):
        planned = _mapping(raw, field=f"targets[{index}]")
        target = _mapping(planned.get("target"), field=f"targets[{index}].target")
        run_names = _mapping(
            planned.get("run_names"),
            field=f"targets[{index}].run_names",
        )
        model = _mapping(planned.get("model"), field=f"targets[{index}].model")
        cache = _mapping(planned.get("cache"), field=f"targets[{index}].cache")
        row = {
            "target_id": str(target.get("target_id") or ""),
            "recording_id": str(target.get("recording_id") or ""),
            "analysis_zarr": str(target.get("analysis_zarr") or ""),
            "keypoint_run": str(run_names.get("keypoint_run") or ""),
            "refined_keypoint_run": str(
                run_names.get("refined_keypoint_run") or ""
            ),
            "model_set_id": str(model.get("set_id") or ""),
            "model_run_id": str(model.get("run_id") or ""),
            "model_path": str(model.get("model_path") or ""),
            "crop_run": str(cache.get("crop_run") or ""),
        }
        if any(not value for value in row.values()):
            raise ValueError(f"Whole-recording plan target {index} is incomplete: {row}")
        rows.append(row)
    return rows


def finalize_registry(
    run_root: Path,
    *,
    registry_path: Path,
    apply: bool,
) -> dict[str, Any]:
    resolved_run_root = run_root.expanduser().resolve()
    resolved_registry = registry_path.expanduser().resolve()
    plan_path = resolved_run_root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(
            f"Unsupported or missing whole-recording plan schema in {plan_path}."
        )
    targets = _target_rows(plan)
    integrity_before = (
        _check_registry_integrity(resolved_registry)
        if resolved_registry.exists()
        else "missing"
    )
    if apply and integrity_before != "ok":
        raise RuntimeError(
            f"Registry failed integrity_check before finalization: {integrity_before}"
        )

    finalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    previous_disable = os.environ.pop(_DISABLE_REGISTRY_WRITES_ENV, None)
    previous_registry = os.environ.get("PALETTE_REGISTRY_PATH")
    os.environ["PALETTE_REGISTRY_PATH"] = str(resolved_registry)
    try:
        for target in targets:
            try:
                zarr_path = Path(target["analysis_zarr"]).expanduser().resolve()
                root = open_zarr_group_direct(zarr_path, mode="r")
                keypoint_run = _require_complete_run(
                    root,
                    parent_name="keypoints_runs",
                    run_name=target["keypoint_run"],
                )
                refined_run = _require_complete_run(
                    root,
                    parent_name=refine_mod.REFINED_KEYPOINT_GROUP,
                    run_name=target["refined_keypoint_run"],
                )
                actual_model_set = normalize_attr(
                    keypoint_run.attrs.get("model_resolution_selected_set_id")
                )
                actual_model_run = normalize_attr(
                    keypoint_run.attrs.get("model_resolution_selected_run_id")
                )
                actual_model_path = normalize_attr(
                    keypoint_run.attrs.get("model_resolution_selected_model_path")
                )
                actual_crop_run = normalize_attr(
                    keypoint_run.attrs.get("source_crop_run")
                )
                expected_model_path = str(
                    Path(target["model_path"]).expanduser().resolve()
                )
                normalized_actual_model_path = (
                    str(Path(actual_model_path).expanduser().resolve())
                    if actual_model_path
                    else None
                )
                if (
                    actual_model_set != target["model_set_id"]
                    or actual_model_run != target["model_run_id"]
                    or normalized_actual_model_path != expected_model_path
                ):
                    raise RuntimeError(
                        "Keypoint model lineage mismatch: actual "
                        f"set/run/path={(actual_model_set, actual_model_run, normalized_actual_model_path)}, "
                        "expected "
                        f"{(target['model_set_id'], target['model_run_id'], expected_model_path)}."
                    )
                if actual_crop_run != target["crop_run"]:
                    raise RuntimeError(
                        f"Keypoint run sources crop {actual_crop_run!r}, expected "
                        f"{target['crop_run']!r}."
                    )
                refined_source = normalize_attr(
                    refined_run.attrs.get("source_keypoints_run")
                ) or normalize_attr(refined_run.attrs.get("source_keypoint_run"))
                if refined_source != target["keypoint_run"]:
                    raise RuntimeError(
                        f"Refined run {target['refined_keypoint_run']!r} sources "
                        f"{refined_source!r}, expected {target['keypoint_run']!r}."
                    )
                keypoint_stats = _keypoint_statistics(keypoint_run)
                refined_stats = _refined_statistics(refined_run)
                dataset_id: str | None = None
                recording_id: str | None = None
                if apply:
                    refresh_details = refresh_keypoint_performance_details(
                        root=root,
                        zarr_path=zarr_path,
                        run_name=target["keypoint_run"],
                        registry_path=resolved_registry,
                        console=None,
                    )
                    if refresh_details.get("keypoint_performance_refresh_status") != "ok":
                        raise RuntimeError(
                            "Keypoint performance refresh failed: "
                            f"{refresh_details.get('keypoint_performance_refresh_reason')}"
                        )
                    wrote_keypoints = emit_stage_completion(
                        root,
                        zarr_path,
                        step_name="keypoints",
                        status="ok",
                        source="serial_whole_recording_keypoint_finalizer",
                        run_name=target["keypoint_run"],
                        method=normalize_attr(keypoint_run.attrs.get("method"))
                        or "yolo_pose",
                        coverage_pct=keypoint_stats["coverage_pct"],
                        details_json={
                            "reason": "serial_whole_recording_keypoint_finalizer",
                            "source_crop_run": normalize_attr(
                                keypoint_run.attrs.get("source_crop_run")
                            ),
                            **keypoint_stats,
                            **refresh_details,
                            "source_run_root": str(resolved_run_root),
                        },
                        registry=resolved_registry,
                        auto_registry_from_env=False,
                        trigger_run_name=target["keypoint_run"],
                    )
                    if not wrote_keypoints:
                        raise RuntimeError("Keypoint stage-status reconciliation returned false.")

                    context = refine_mod._resolve_status_context_from_root(
                        root,
                        str(zarr_path),
                    )
                    if context is None:
                        context = refine_mod._resolve_status_context(str(zarr_path))
                    if context is None:
                        raise RuntimeError(
                            f"Could not resolve refined-keypoint registry context for {zarr_path}"
                        )
                    review_status = refined_run.attrs.get("keypoint_review_status")
                    wrote_refined = refine_mod._emit_refined_keypoint_status(
                        context=context,
                        status="ok",
                        run_name=target["refined_keypoint_run"],
                        method=normalize_attr(refined_run.attrs.get("method"))
                        or "refine_keypoints",
                        coverage_pct=refined_stats["coverage_pct"],
                        review_status_json=(
                            dict(review_status)
                            if isinstance(review_status, Mapping)
                            else None
                        ),
                        details={
                            "reason": "serial_whole_recording_keypoint_finalizer",
                            "source_keypoints_run": refined_source,
                            "source_crop_run": normalize_attr(
                                refined_run.attrs.get("source_crop_run")
                            ),
                            **refined_stats,
                            "source_run_root": str(resolved_run_root),
                        },
                        console=None,
                    )
                    if not wrote_refined:
                        raise RuntimeError(
                            "Refined-keypoint stage-status reconciliation returned false."
                        )
                    dataset_id = context.dataset_id
                    recording_id = context.recording_id

                finalized.append(
                    {
                        **target,
                        "dataset_id": dataset_id,
                        "resolved_recording_id": recording_id,
                        "keypoint_statistics": keypoint_stats,
                        "refined_keypoint_statistics": refined_stats,
                        "registry_status": "updated" if apply else "validated",
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "target_id": target.get("target_id"),
                        "analysis_zarr": target.get("analysis_zarr"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if apply:
                    break
    finally:
        if previous_disable is not None:
            os.environ[_DISABLE_REGISTRY_WRITES_ENV] = previous_disable
        if previous_registry is None:
            os.environ.pop("PALETTE_REGISTRY_PATH", None)
        else:
            os.environ["PALETTE_REGISTRY_PATH"] = previous_registry

    integrity_after = (
        _check_registry_integrity(resolved_registry)
        if resolved_registry.exists()
        else "missing"
    )
    if apply and integrity_after != "ok":
        errors.append(
            {
                "target_id": None,
                "analysis_zarr": None,
                "error": (
                    "RuntimeError: Registry failed integrity_check after finalization: "
                    f"{integrity_after}"
                ),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not errors and len(finalized) == len(targets) else "error",
        "apply": bool(apply),
        "run_root": str(resolved_run_root),
        "plan_path": str(plan_path),
        "registry_path": str(resolved_registry),
        "target_count": len(targets),
        "finalized_count": len(finalized),
        "registry_integrity_before": integrity_before,
        "registry_integrity_after": integrity_after,
        "finished_at_utc": utc_now(),
        "errors": errors,
        "finalized": finalized,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize_registry(
        args.run_root,
        registry_path=args.registry,
        apply=bool(args.apply),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json.expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "finalize_registry", "main"]

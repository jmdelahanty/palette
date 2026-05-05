"""Batch-run subject-mask inference and smart finalization on recording Zarrs.

The utility is intentionally conservative:

* default mode is dry-run;
* existing subject-mask and refined-subject-mask runs are not overwritten;
* smoke archives are excluded by default;
* each archive is processed serially while the underlying stages retain their
  own provenance capture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import zarr


RAW_COMPONENTS = ("subject_body", "eyes_union", "swim_bladder")
REFINED_COMPONENTS = ("subject_body", "eye_left", "eye_right", "swim_bladder")


@dataclass(frozen=True)
class ArchivePlan:
    zarr_path: str
    subject_run: str
    refined_run: str
    crop_run: Optional[str]
    assignment_keypoint_group: Optional[str]
    assignment_keypoint_run: Optional[str]
    has_subject_runs: bool
    has_refined_subject_runs: bool
    run_inference: bool
    run_finalization: bool
    skip_reason: str = ""


@dataclass
class ArchiveResult:
    zarr_path: str
    subject_run: str
    refined_run: str
    planned_inference: bool
    planned_finalization: bool
    inference_status: str = "not_requested"
    finalization_status: str = "not_requested"
    validation_status: str = "not_run"
    error: str = ""


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _attrs(group_path: Path) -> dict[str, Any]:
    payload = _read_json(group_path / "zarr.json")
    raw = payload.get("attributes") if isinstance(payload, dict) else None
    return dict(raw) if isinstance(raw, dict) else {}


def _child_groups(group_path: Path) -> list[str]:
    if not group_path.is_dir():
        return []
    return sorted(path.name for path in group_path.iterdir() if path.is_dir())


def _latest_group_name(parent_path: Path) -> Optional[str]:
    attrs = _attrs(parent_path)
    for key in ("latest", "latest_materialized", "detect_review_status_latest"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (parent_path / value).is_dir():
            return value
    children = _child_groups(parent_path)
    return children[-1] if children else None


def _discover_analysis_zarrs(roots: Sequence[Path], *, include_smoke: bool) -> list[Path]:
    seen: set[str] = set()
    zarrs: list[Path] = []
    for root in roots:
        root = root.expanduser()
        candidates = [root] if root.name.endswith("_analysis.zarr") else sorted(root.rglob("*_analysis.zarr"))
        for candidate in candidates:
            if not include_smoke and "/smoke/" in str(candidate):
                continue
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            zarrs.append(candidate)
    return sorted(zarrs)


def _zarr_paths_from_report(report_path: Path) -> list[Path]:
    payload = _read_json(report_path)
    if payload is None:
        raise ValueError(f"Could not read JSON report: {report_path}")

    rows = payload.get("results")
    if not isinstance(rows, list):
        rows = payload.get("plans")
    if not isinstance(rows, list):
        raise ValueError(f"Report {report_path} does not contain a results or plans list.")

    seen: set[str] = set()
    zarr_paths: list[Path] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("zarr_path")
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        zarr_paths.append(path)
    if not zarr_paths:
        raise ValueError(f"Report {report_path} did not contain any zarr_path entries.")
    return zarr_paths


def _resolve_crop_run(zarr_path: Path) -> Optional[str]:
    return _latest_group_name(zarr_path / "crop_runs")


def _resolve_assignment_keypoints(zarr_path: Path) -> tuple[Optional[str], Optional[str]]:
    refined_parent = zarr_path / "refined_keypoints_runs"
    refined_latest = _latest_group_name(refined_parent)
    if refined_latest:
        return "refined_keypoints_runs", refined_latest
    keypoint_parent = zarr_path / "keypoints_runs"
    keypoint_latest = _latest_group_name(keypoint_parent)
    if keypoint_latest:
        return "keypoints_runs", keypoint_latest
    return None, None


def build_archive_plan(
    zarr_path: Path,
    *,
    subject_run_name: str,
    refined_run_name: str,
    force_inference: bool,
    force_finalization: bool,
) -> ArchivePlan:
    subject_parent = zarr_path / "subject_mask_runs"
    refined_parent = zarr_path / "refined_subject_masks_runs"
    subject_children = _child_groups(subject_parent)
    refined_children = _child_groups(refined_parent)
    crop_run = _resolve_crop_run(zarr_path)
    keypoint_group, keypoint_run = _resolve_assignment_keypoints(zarr_path)

    has_subject_runs = bool(subject_children)
    has_refined_subject_runs = bool(refined_children)
    run_inference = bool(force_inference or not has_subject_runs)
    run_finalization = bool(force_finalization or not has_refined_subject_runs)

    skip_reasons: list[str] = []
    if crop_run is None:
        run_inference = False
        run_finalization = False
        skip_reasons.append("missing_crop_run")
    if keypoint_group is None or keypoint_run is None:
        run_inference = False
        skip_reasons.append("missing_keypoint_assignment_source")
    if has_subject_runs and not force_inference:
        skip_reasons.append("subject_mask_runs_present")
    if has_refined_subject_runs and not force_finalization:
        skip_reasons.append("refined_subject_masks_runs_present")
    if run_finalization and not run_inference and not has_subject_runs:
        run_finalization = False
        skip_reasons.append("cannot_finalize_without_subject_mask_run")

    return ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run=subject_run_name,
        refined_run=refined_run_name,
        crop_run=crop_run,
        assignment_keypoint_group=keypoint_group,
        assignment_keypoint_run=keypoint_run,
        has_subject_runs=has_subject_runs,
        has_refined_subject_runs=has_refined_subject_runs,
        run_inference=run_inference,
        run_finalization=run_finalization,
        skip_reason=", ".join(skip_reasons),
    )


def _selected_subject_run_for_finalization(plan: ArchivePlan) -> str:
    zarr_path = Path(plan.zarr_path)
    if plan.run_inference:
        return plan.subject_run
    latest = _latest_group_name(zarr_path / "subject_mask_runs")
    if latest is None:
        raise RuntimeError(f"{zarr_path} has no subject_mask_runs to finalize.")
    return latest


def _inference_command(args: argparse.Namespace, plan: ArchivePlan) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fisheye.segmentation.infer_unet_subject_masks",
        plan.zarr_path,
        "--resolve-model-from-registry",
        "--registry",
        str(args.registry),
        "--model-coverage-class",
        args.model_coverage_class,
        "--model-component-coverage-key",
        args.model_component_coverage_key,
        "--model-label-schema-id",
        args.model_label_schema_id,
        "--run-name",
        plan.subject_run,
        "--crop-run",
        str(plan.crop_run),
        "--assignment-keypoint-group",
        str(plan.assignment_keypoint_group),
        "--assignment-keypoint-run",
        str(plan.assignment_keypoint_run),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mask-probs-dtype",
        args.mask_probs_dtype,
        "--mask-probs-chunk-rois",
        str(args.mask_probs_chunk_rois),
        "--no-write-masks-roi",
        "--async-output",
        "--output-queue-size",
        str(args.output_queue_size),
        "--no-progress",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _finalization_command(args: argparse.Namespace, plan: ArchivePlan) -> list[str]:
    subject_run = _selected_subject_run_for_finalization(plan)
    cmd = [
        sys.executable,
        "-m",
        "fisheye.refinement.finalize_subject_masks",
        plan.zarr_path,
        "--subject-run",
        subject_run,
        "--run-name",
        plan.refined_run,
        "--components",
        "subject_body",
        "eyes_union",
        "swim_bladder",
        "--chunk-size",
        str(args.finalize_chunk_size),
        "--metric-level",
        args.metric_level,
        "--execution-backend",
        args.finalize_execution_backend,
        "--scheduler",
        args.finalize_scheduler,
        "--assignment-keypoint-group",
        str(plan.assignment_keypoint_group),
        "--assignment-keypoints-run",
        str(plan.assignment_keypoint_run),
        "--json",
    ]
    if args.finalize_num_workers is not None:
        cmd.extend(["--num-workers", str(args.finalize_num_workers)])
    if args.write_eye_geometry:
        cmd.append("--write-eye-geometry")
    if args.write_component_contours:
        cmd.append("--write-component-contours")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _run_command(cmd: Sequence[str], *, dry_run: bool) -> str:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return "planned"
    completed = subprocess.run(list(cmd), check=False)
    return "ok" if completed.returncode == 0 else f"failed_exit_{completed.returncode}"


def _open_group(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def validate_outputs(zarr_path: Path, *, subject_run: str, refined_run: str) -> tuple[str, str]:
    root = _open_group(zarr_path)
    details: list[str] = []
    subject_parent = root.get("subject_mask_runs")
    if subject_parent is None or subject_run not in subject_parent:
        return "failed", f"missing subject_mask_runs/{subject_run}"
    subject = subject_parent[subject_run]
    raw_labels = tuple(str(label) for label in subject.attrs.get("mask_labels", ()))
    if any(label not in raw_labels for label in RAW_COMPONENTS):
        return "failed", f"subject mask labels {raw_labels!r} missing {RAW_COMPONENTS!r}"
    if "mask_probs_roi" not in subject:
        return "failed", f"subject_mask_runs/{subject_run} missing mask_probs_roi"
    details.append(f"subject_mask_labels={raw_labels}")

    refined_parent = root.get("refined_subject_masks_runs")
    if refined_parent is None or refined_run not in refined_parent:
        return "failed", f"missing refined_subject_masks_runs/{refined_run}"
    refined = refined_parent[refined_run]
    refined_labels = tuple(str(label) for label in refined.attrs.get("mask_labels", ()))
    if any(label not in refined_labels for label in REFINED_COMPONENTS):
        return "failed", f"refined mask labels {refined_labels!r} missing {REFINED_COMPONENTS!r}"
    if "masks_roi" not in refined:
        return "failed", f"refined_subject_masks_runs/{refined_run} missing masks_roi"
    for component in REFINED_COMPONENTS:
        if f"components/{component}" not in refined:
            return "failed", f"refined_subject_masks_runs/{refined_run} missing components/{component}"
    details.append(f"refined_mask_labels={refined_labels}")
    return "ok", "; ".join(details)


def _write_json_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plans": [asdict(plan) for plan in plans],
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown_report(path: Path, *, plans: Sequence[ArchivePlan], results: Sequence[ArchiveResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Subject Mask Batch Pipeline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Planned archives: {len(plans)}",
        f"- Results: {len(results)}",
        f"- Inference requested: {sum(1 for plan in plans if plan.run_inference)}",
        f"- Finalization requested: {sum(1 for plan in plans if plan.run_finalization)}",
        f"- Validation ok: {sum(1 for result in results if result.validation_status == 'ok')}",
        f"- Errors: {sum(1 for result in results if result.error)}",
        "",
        "## Results",
        "",
        "| Zarr | Inference | Finalization | Validation | Error |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | {} |".format(
                Path(result.zarr_path).name,
                result.inference_status,
                result.finalization_status,
                result.validation_status,
                (result.error or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/nvme1/recordings")])
    parser.add_argument(
        "--roots-from-report",
        type=Path,
        help="Use unique zarr_path entries from a previous JSON report instead of positional roots.",
    )
    parser.add_argument("--apply", action="store_true", help="Run commands. Default is dry-run planning.")
    parser.add_argument("--pilot-size", type=int, default=None, help="Limit to the first N eligible archives.")
    parser.add_argument("--include-smoke", action="store_true", help="Include /smoke/ analysis Zarrs.")
    parser.add_argument("--run-label", default=f"batch_{_utc_now_compact()}")
    parser.add_argument("--registry", type=Path, default=Path("/nvme1/palette_registry.sqlite"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-probs-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--mask-probs-chunk-rois", type=int, default=32)
    parser.add_argument("--output-queue-size", type=int, default=2)
    parser.add_argument("--model-coverage-class", default="dense_all_components")
    parser.add_argument("--model-component-coverage-key", default="body+eyes+swim_bladder")
    parser.add_argument("--model-label-schema-id", default="subject_v1_union")
    parser.add_argument("--metric-level", choices=("cheap", "full"), default="cheap")
    parser.add_argument("--finalize-chunk-size", type=int, default=64)
    parser.add_argument("--finalize-execution-backend", choices=("serial_driver", "dask_worker_chunks"), default="dask_worker_chunks")
    parser.add_argument("--finalize-scheduler", choices=("single-threaded", "threads", "processes", "distributed"), default="processes")
    parser.add_argument("--finalize-num-workers", type=int, default=48)
    parser.add_argument("--write-eye-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-component-contours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-inference", action="store_true", help="Run inference even if subject_mask_runs already exists.")
    parser.add_argument("--force-finalization", action="store_true", help="Run finalization even if refined_subject_masks_runs already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Pass overwrite through to child stages.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    parser.add_argument("--consolidate-metadata", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    dry_run = not bool(args.apply)

    subject_run = f"subject_masks_unet_registry_{args.run_label}"
    refined_run = f"refined_subject_masks_smart_finalizer_{args.run_label}"
    root_inputs = (
        _zarr_paths_from_report(args.roots_from_report)
        if args.roots_from_report is not None
        else args.roots
    )
    all_plans = [
        build_archive_plan(
            zarr_path,
            subject_run_name=subject_run,
            refined_run_name=refined_run,
            force_inference=bool(args.force_inference),
            force_finalization=bool(args.force_finalization),
        )
        for zarr_path in _discover_analysis_zarrs(root_inputs, include_smoke=bool(args.include_smoke))
    ]
    plans = [plan for plan in all_plans if plan.run_inference or plan.run_finalization]
    if args.pilot_size is not None:
        plans = plans[: max(0, int(args.pilot_size))]

    print(f"analysis_archives_discovered: {len(all_plans)}")
    print(f"archives_selected: {len(plans)}")
    print(f"mode: {'apply' if args.apply else 'dry-run'}")
    print(f"subject_run: {subject_run}")
    print(f"refined_run: {refined_run}")

    results: list[ArchiveResult] = []
    exit_code = 0
    for idx, plan in enumerate(plans, start=1):
        print(f"\n[{idx}/{len(plans)}] {plan.zarr_path}", flush=True)
        result = ArchiveResult(
            zarr_path=plan.zarr_path,
            subject_run=plan.subject_run,
            refined_run=plan.refined_run,
            planned_inference=plan.run_inference,
            planned_finalization=plan.run_finalization,
        )
        try:
            if plan.run_inference:
                result.inference_status = _run_command(_inference_command(args, plan), dry_run=dry_run)
                if result.inference_status != "ok" and not dry_run:
                    raise RuntimeError(f"inference {result.inference_status}")
            if plan.run_finalization:
                result.finalization_status = _run_command(_finalization_command(args, plan), dry_run=dry_run)
                if result.finalization_status != "ok" and not dry_run:
                    raise RuntimeError(f"finalization {result.finalization_status}")
            if dry_run:
                result.validation_status = "planned"
            else:
                validation_subject_run = plan.subject_run if plan.run_inference else _selected_subject_run_for_finalization(plan)
                status, detail = validate_outputs(
                    Path(plan.zarr_path),
                    subject_run=validation_subject_run,
                    refined_run=plan.refined_run,
                )
                result.validation_status = status
                if status != "ok":
                    raise RuntimeError(detail)
                if args.consolidate_metadata:
                    zarr.consolidate_metadata(plan.zarr_path)
        except Exception as exc:
            result.error = str(exc)
            exit_code = 1
            print(f"error: {exc}", file=sys.stderr, flush=True)
            if not bool(args.continue_on_error):
                results.append(result)
                break
        results.append(result)

    if args.json_report:
        _write_json_report(args.json_report, plans=plans, results=results)
    if args.markdown_report:
        _write_markdown_report(args.markdown_report, plans=plans, results=results)

    print("\nsummary:")
    print(json.dumps({"plans": len(plans), "results": len(results), "exit_code": exit_code}, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

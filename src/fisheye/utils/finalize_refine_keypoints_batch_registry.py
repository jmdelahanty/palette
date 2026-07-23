"""Serial registry finalizer for refined-keypoint LSF batches.

Parallel refined-keypoint array tasks write Zarr run groups independently.  The
SQLite registry update is intentionally deferred to this single-process
finalizer to avoid concurrent registry writers from many LSF tasks.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.refinement import refine_keypoints as refine_mod
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    is_run_complete_in_parent,
)


_DISABLE_REGISTRY_WRITES_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"


@dataclass(frozen=True)
class FinalizedRefinedKeypoints:
    zarr_path: Path
    run_name: str
    source_keypoints_run: str | None
    dataset_id: str
    recording_id: str | None
    coverage_pct: float | None
    total_rois: int | None
    refined_success: int | None
    usable_keypoints: int | None


def _load_targets(run_root: Path) -> list[Path]:
    target_file = run_root / "zarr_paths.txt"
    if not target_file.exists():
        raise FileNotFoundError(f"Missing target list: {target_file}")
    targets: list[Path] = []
    for line in target_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        targets.append(Path(value).expanduser().resolve())
    if not targets:
        raise RuntimeError(f"No zarr paths found in {target_file}")
    return targets


def _as_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_keypoints_run(run_group: Any) -> str | None:
    attrs = getattr(run_group, "attrs", {})
    return normalize_attr(attrs.get("source_keypoints_run")) or normalize_attr(
        attrs.get("source_keypoint_run")
    )


def _completed_at(run_group: Any) -> str:
    attrs = getattr(run_group, "attrs", {})
    return (
        normalize_attr(attrs.get(RUN_COMPLETED_AT_ATTR))
        or normalize_attr(attrs.get("completed_at_utc"))
        or normalize_attr(attrs.get("created_at_utc"))
        or ""
    )


def _run_sort_key(run_name: str, run_group: Any) -> tuple[str, str]:
    return (_completed_at(run_group), run_name)


def _iter_group_keys(group: Any) -> list[str]:
    try:
        return sorted(str(name) for name in group.group_keys())
    except Exception:
        return []


def _select_refined_run(
    root: Any,
    *,
    requested_keypoint_run: str | None,
) -> tuple[str, Any]:
    parent = root.get(refine_mod.REFINED_KEYPOINT_GROUP)
    if parent is None:
        parent = root.get(refine_mod.LEGACY_KEYPOINT_GROUP)
    if parent is None:
        raise RuntimeError("No refined_keypoints_runs parent found")

    candidates: list[tuple[str, Any]] = []
    latest = normalize_attr(parent.attrs.get("latest"))
    if latest and latest in parent:
        candidates.append((latest, parent[latest]))

    for run_name in _iter_group_keys(parent):
        if latest and run_name == latest:
            continue
        candidates.append((run_name, parent[run_name]))

    matching: list[tuple[str, Any]] = []
    for run_name, run_group in candidates:
        attrs = getattr(run_group, "attrs", {})
        source_run = _source_keypoints_run(run_group)
        if requested_keypoint_run and source_run != requested_keypoint_run:
            continue
        if attrs.get("stage_selector_eligible") is not True:
            continue
        if (
            attrs.get("coordinate_contract")
            == refine_mod._LEGACY_UNVERIFIED_OUTPUT_CONTRACT
            or attrs.get("legacy_unverified_diagnostic_output") is True
        ):
            continue
        if not is_run_complete_in_parent(parent, run_group):
            continue
        matching.append((run_name, run_group))

    if not matching:
        source_suffix = (
            f" for source keypoint run {requested_keypoint_run!r}"
            if requested_keypoint_run
            else ""
        )
        raise RuntimeError(
            "No complete selector-eligible refined keypoint run found"
            f"{source_suffix}"
        )

    matching.sort(key=lambda item: _run_sort_key(item[0], item[1]))
    return matching[-1]


def _stats_for_run(run_group: Any) -> tuple[dict[str, Any], float | None, int | None, int | None, int | None]:
    attrs = getattr(run_group, "attrs", {})
    stats = _as_mapping(attrs.get("summary_statistics")) or {}
    total_rois = _safe_int(stats.get("total_rois"))
    if total_rois is None:
        try:
            total_rois = int(run_group["keypoints_roi"].shape[0])
        except Exception:
            total_rois = None
    refined_success = _safe_int(stats.get("refined_success"))
    usable_keypoints = _safe_int(stats.get("usable_keypoints"))
    coverage_pct = _safe_float(stats.get("pass_rate_percent"))
    if coverage_pct is None and total_rois and refined_success is not None:
        coverage_pct = float(refined_success) / float(total_rois) * 100.0
    return stats, coverage_pct, total_rois, refined_success, usable_keypoints


def _check_registry_integrity(registry_path: Path) -> str:
    registry = Registry(registry_path)
    registry.conn.execute("PRAGMA busy_timeout=60000;")
    try:
        return str(registry.conn.execute("PRAGMA integrity_check;").fetchone()[0])
    finally:
        registry.close()


def _default_registry_path() -> Path:
    return RegistryPaths.from_env(Path.cwd()).path


def finalize_refine_keypoints_batch_registry(
    run_root: str | Path,
    *,
    registry_path: str | Path,
    keypoint_run: str | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Refresh refined-keypoint registry status from completed Zarr runs."""

    run_root_path = Path(run_root).expanduser().resolve()
    if not run_root_path.exists():
        raise FileNotFoundError(run_root_path)

    registry_file = Path(registry_path).expanduser().resolve()
    targets = _load_targets(run_root_path)
    integrity_before = _check_registry_integrity(registry_file) if registry_file.exists() else "missing"
    if apply and integrity_before != "ok":
        raise RuntimeError(
            f"Registry failed integrity_check before finalization: {integrity_before}"
        )

    finalized: list[FinalizedRefinedKeypoints] = []
    errors: list[dict[str, Any]] = []
    upserted_status_rows = 0

    previous_disable_value = os.environ.pop(_DISABLE_REGISTRY_WRITES_ENV, None)
    try:
        for zarr_path in targets:
            try:
                root = open_zarr_group_direct(zarr_path, mode="r")
                run_name, run_group = _select_refined_run(
                    root,
                    requested_keypoint_run=keypoint_run,
                )
                stats, coverage_pct, total_rois, refined_success, usable_keypoints = _stats_for_run(
                    run_group
                )
                source_run = _source_keypoints_run(run_group)
                review_status = _as_mapping(run_group.attrs.get("keypoint_review_status"))
                context = None
                if apply:
                    context = refine_mod._resolve_status_context_from_root(root, str(zarr_path))
                    if context is None:
                        context = refine_mod._resolve_status_context(str(zarr_path))
                    if context is None:
                        raise RuntimeError(f"Could not resolve registry dataset context for {zarr_path}")

                    wrote = refine_mod._emit_refined_keypoint_status(
                        context=context,
                        status="ok",
                        run_name=run_name,
                        method=normalize_attr(run_group.attrs.get("method"))
                        or "refine_keypoints",
                        coverage_pct=coverage_pct,
                        review_status_json=review_status,
                        details={
                            "reason": "serial_refine_keypoints_batch_registry_finalizer",
                            "source_keypoints_run": source_run,
                            "source_crop_run": normalize_attr(
                                run_group.attrs.get("source_crop_run")
                            ),
                            "source_detect_run": normalize_attr(
                                run_group.attrs.get("source_detect_run")
                            ),
                            "source_refined_run": normalize_attr(
                                run_group.attrs.get("source_refined_run")
                            ),
                            "total_rois": total_rois,
                            "refined_success": refined_success,
                            "usable_keypoints": usable_keypoints,
                            "pass_rate_percent": coverage_pct,
                            "summary_statistics": stats,
                            "source_run_root": str(run_root_path),
                        },
                        console=None,
                    )
                    if not wrote:
                        raise RuntimeError(
                            f"Step-status sync returned false for {zarr_path} run {run_name!r}"
                        )
                    upserted_status_rows += 1

                finalized.append(
                    FinalizedRefinedKeypoints(
                        zarr_path=zarr_path,
                        run_name=run_name,
                        source_keypoints_run=source_run,
                        dataset_id=context.dataset_id if context is not None else "",
                        recording_id=context.recording_id if context is not None else None,
                        coverage_pct=coverage_pct,
                        total_rois=total_rois,
                        refined_success=refined_success,
                        usable_keypoints=usable_keypoints,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "zarr_path": str(zarr_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if apply:
                    raise
    finally:
        if previous_disable_value is not None:
            os.environ[_DISABLE_REGISTRY_WRITES_ENV] = previous_disable_value

    integrity_after = _check_registry_integrity(registry_file) if registry_file.exists() else "missing"
    if apply and integrity_after != "ok":
        raise RuntimeError(
            f"Registry failed integrity_check after finalization: {integrity_after}"
        )

    return {
        "schema": "palette.refine_keypoints_batch_registry_finalizer.v1",
        "status": "ok" if not errors else "error",
        "apply": bool(apply),
        "run_root": str(run_root_path),
        "registry_path": str(registry_file),
        "keypoint_run": keypoint_run,
        "target_count": len(targets),
        "finalized_count": len(finalized),
        "upserted_status_rows": upserted_status_rows,
        "registry_integrity_before": integrity_before,
        "registry_integrity_after": integrity_after,
        "finished_at_utc": utc_now(),
        "errors": errors,
        "finalized": [
            {
                "zarr_path": str(item.zarr_path),
                "run_name": item.run_name,
                "source_keypoints_run": item.source_keypoints_run,
                "dataset_id": item.dataset_id or None,
                "recording_id": item.recording_id,
                "coverage_pct": item.coverage_pct,
                "total_rois": item.total_rois,
                "refined_success": item.refined_success,
                "usable_keypoints": item.usable_keypoints,
            }
            for item in finalized
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serially finalize registry rows for a refined-keypoint batch.",
    )
    parser.add_argument("run_root", type=Path, help="Batch run root directory.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry SQLite path. Defaults to PALETTE_REGISTRY_PATH/config/default.",
    )
    parser.add_argument(
        "--keypoint-run",
        default=None,
        help="Only finalize refined runs sourced from this keypoints_runs child.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write registry rows. Without this, only validate and report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the finalizer report JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize_refine_keypoints_batch_registry(
        args.run_root,
        registry_path=args.registry or _default_registry_path(),
        keypoint_run=args.keypoint_run,
        apply=bool(args.apply),
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill crop row-identity arrays from refined-detect source instances.

Default mode is dry-run. Use --apply to write arrays and metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.row_lineage import normalize_chunks_for_data


IDENTITY_ARRAY_NAMES = ("source_refined_row_ids", "source_detect_row_index")
BACKFILL_POLICY = "mapped_from_detection_indices_unmapped_minus_one"


@dataclass(frozen=True)
class CropIdentityPayload:
    source_refined_row_ids: np.ndarray
    source_detect_row_index: np.ndarray
    valid_mask: np.ndarray

    @property
    def crop_rows(self) -> int:
        return int(self.source_refined_row_ids.shape[0])

    @property
    def mappable_rows(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def unmappable_rows(self) -> int:
        return int(self.crop_rows - self.mappable_rows)


@dataclass(frozen=True)
class ArrayPlan:
    name: str
    action: str
    existing: bool


@dataclass(frozen=True)
class CropPlan:
    zarr_path: Path
    crop_run: str
    source_path: str
    crop_rows: int
    source_rows: int
    mappable_rows: int
    unmappable_rows: int
    array_plans: tuple[ArrayPlan, ...]
    status: str
    message: str = ""

    @property
    def needs_write(self) -> bool:
        return any(plan.action in {"write", "overwrite"} for plan in self.array_plans)


def _resolve_roots(paths: Sequence[Path]) -> list[Path]:
    if paths:
        return [Path(path).expanduser() for path in paths]
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root).expanduser()]
    return [Path("/nvme1/recordings")]


def _iter_zarr(roots: Sequence[Path], *, recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = Path(root).expanduser()
        if root.suffix == ".zarr" and root.is_dir():
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(path for path in root.rglob("*.zarr") if path.is_dir())
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        else:
            candidates = []
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


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


def _array_shape(array_path: Path) -> Optional[tuple[int, ...]]:
    payload = _read_json(array_path / "zarr.json")
    raw = payload.get("shape") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _latest_crop_name(crop_parent_path: Path) -> Optional[str]:
    attrs = _attrs(crop_parent_path)
    for key in ("latest_materialized", "latest", "latest_any", "crop_review_status_latest"):
        value = attrs.get(key)
        if isinstance(value, str) and value and (crop_parent_path / value).is_dir():
            return value
    children = sorted(path.name for path in crop_parent_path.iterdir() if path.is_dir())
    return children[-1] if children else None


def _select_crop_runs(
    zarr_path: Path,
    *,
    requested: Sequence[str],
    limit: str,
) -> list[str]:
    crop_parent = zarr_path / "crop_runs"
    if not crop_parent.is_dir():
        return []
    if requested:
        return [name for name in requested if (crop_parent / name).is_dir()]
    if limit == "all":
        return sorted(path.name for path in crop_parent.iterdir() if path.is_dir())
    latest = _latest_crop_name(crop_parent)
    return [latest] if latest else []


def _infer_zarr_use(zarr_path: Path) -> str:
    attrs = _attrs(zarr_path)
    for key in ("zarr_use", "zarr_purpose"):
        raw = attrs.get(key)
        if isinstance(raw, str) and raw.lower() in {"analysis", "training"}:
            return raw.lower()
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def build_crop_identity_payload(
    detection_indices: np.ndarray,
    refined_row_ids: np.ndarray,
    source_detect_row_index: np.ndarray,
) -> CropIdentityPayload:
    detection_indices = np.asarray(detection_indices, dtype=np.int64).reshape(-1)
    refined_row_ids = np.asarray(refined_row_ids, dtype=np.int64).reshape(-1)
    source_detect_row_index = np.asarray(source_detect_row_index, dtype=np.int64).reshape(-1)
    if refined_row_ids.shape != source_detect_row_index.shape:
        raise ValueError(
            "refined_row_ids and source_detect_row_index must have matching shape, "
            f"got {refined_row_ids.shape} and {source_detect_row_index.shape}"
        )

    valid = (detection_indices >= 0) & (detection_indices < refined_row_ids.shape[0])
    source_refined = np.full(detection_indices.shape, -1, dtype=np.int64)
    source_detect = np.full(detection_indices.shape, -1, dtype=np.int32)
    if np.any(valid):
        valid_source_rows = detection_indices[valid]
        source_refined[valid] = refined_row_ids[valid_source_rows]
        source_detect[valid] = source_detect_row_index[valid_source_rows].astype(np.int32, copy=False)
    return CropIdentityPayload(
        source_refined_row_ids=source_refined,
        source_detect_row_index=source_detect,
        valid_mask=valid,
    )


def _open_array(array_path: Path) -> np.ndarray:
    return np.asarray(zarr.open_array(str(array_path), mode="r")[:])


def _open_group_direct(group_path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(group_path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(group_path), mode=mode, consolidated=False)


def _existing_array_equal(array_path: Path, desired: np.ndarray) -> tuple[bool, bool]:
    if not (array_path / "zarr.json").is_file():
        return False, False
    shape = _array_shape(array_path)
    if shape != tuple(desired.shape):
        return True, False
    try:
        existing = _open_array(array_path)
    except Exception:
        return True, False
    return True, bool(np.array_equal(existing, desired))


def plan_crop_backfill(
    zarr_path: Path,
    crop_run: str,
    *,
    overwrite: bool,
) -> CropPlan:
    crop_path = zarr_path / "crop_runs" / crop_run
    crop_attrs = _attrs(crop_path)
    source_path_raw = crop_attrs.get("detection_source_path")
    source_path = str(source_path_raw).strip("/") if source_path_raw is not None else ""
    if not source_path:
        return CropPlan(
            zarr_path=zarr_path,
            crop_run=crop_run,
            source_path="",
            crop_rows=0,
            source_rows=0,
            mappable_rows=0,
            unmappable_rows=0,
            array_plans=(),
            status="skipped",
            message="crop run has no detection_source_path",
        )

    source_group_path = zarr_path / source_path
    required_paths = {
        "detection_indices": crop_path / "detection_indices",
        "refined_row_ids": source_group_path / "refined_row_ids",
        "source_detect_row_index": source_group_path / "source_detect_row_index",
    }
    missing = [name for name, path in required_paths.items() if not (path / "zarr.json").is_file()]
    if missing:
        return CropPlan(
            zarr_path=zarr_path,
            crop_run=crop_run,
            source_path=source_path,
            crop_rows=0,
            source_rows=0,
            mappable_rows=0,
            unmappable_rows=0,
            array_plans=(),
            status="skipped",
            message=f"missing required arrays: {', '.join(missing)}",
        )

    detection_indices = _open_array(required_paths["detection_indices"])
    refined_row_ids = _open_array(required_paths["refined_row_ids"])
    source_detect_row_index = _open_array(required_paths["source_detect_row_index"])
    payload = build_crop_identity_payload(detection_indices, refined_row_ids, source_detect_row_index)

    desired_by_name = {
        "source_refined_row_ids": payload.source_refined_row_ids,
        "source_detect_row_index": payload.source_detect_row_index,
    }
    plans: list[ArrayPlan] = []
    for name, desired in desired_by_name.items():
        exists, equal = _existing_array_equal(crop_path / name, desired)
        if equal:
            action = "keep"
        elif exists and overwrite:
            action = "overwrite"
        elif exists:
            action = "skip_mismatch"
        else:
            action = "write"
        plans.append(ArrayPlan(name=name, action=action, existing=exists))

    if any(plan.action in {"write", "overwrite"} for plan in plans):
        status = "planned"
    elif any(plan.action == "skip_mismatch" for plan in plans):
        status = "skipped"
    else:
        status = "up_to_date"

    return CropPlan(
        zarr_path=zarr_path,
        crop_run=crop_run,
        source_path=source_path,
        crop_rows=payload.crop_rows,
        source_rows=int(np.asarray(refined_row_ids).reshape(-1).shape[0]),
        mappable_rows=payload.mappable_rows,
        unmappable_rows=payload.unmappable_rows,
        array_plans=tuple(plans),
        status=status,
        message="",
    )


def _write_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    chunks = normalize_chunks_for_data(None, data.shape, default_chunk_len=1000)
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    try:
        group.create_array(name, **kwargs)
    except TypeError:
        kwargs.pop("chunks", None)
        group.create_array(name, **kwargs)


def apply_crop_backfill(plan: CropPlan) -> int:
    crop_path = plan.zarr_path / "crop_runs" / plan.crop_run
    source_group_path = plan.zarr_path / plan.source_path
    detection_indices = _open_array(crop_path / "detection_indices")
    refined_row_ids = _open_array(source_group_path / "refined_row_ids")
    source_detect_row_index = _open_array(source_group_path / "source_detect_row_index")
    payload = build_crop_identity_payload(detection_indices, refined_row_ids, source_detect_row_index)
    desired_by_name = {
        "source_refined_row_ids": payload.source_refined_row_ids,
        "source_detect_row_index": payload.source_detect_row_index,
    }

    crop_group = _open_group_direct(crop_path, mode="a")
    writes = 0
    for array_plan in plan.array_plans:
        if array_plan.action not in {"write", "overwrite"}:
            continue
        _write_array(crop_group, array_plan.name, desired_by_name[array_plan.name])
        writes += 1

    if writes:
        crop_group.attrs["source_refined_row_ids_available"] = True
        crop_group.attrs["source_refined_row_id_policy"] = BACKFILL_POLICY
        crop_group.attrs["source_detect_row_index_available"] = True
        crop_group.attrs["row_lineage_backfill"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "utility": "fisheye.utils.backfill_crop_row_lineage",
            "source_path": plan.source_path,
            "crop_rows": int(payload.crop_rows),
            "source_rows": int(np.asarray(refined_row_ids).reshape(-1).shape[0]),
            "mappable_rows": int(payload.mappable_rows),
            "unmappable_rows": int(payload.unmappable_rows),
            "unmappable_value": -1,
            "policy": BACKFILL_POLICY,
        }
    return writes


def _format_array_actions(plan: CropPlan) -> str:
    if not plan.array_plans:
        return "-"
    return ",".join(f"{item.name}:{item.action}" for item in plan.array_plans)


def _plan_record(plan: CropPlan, *, applied: bool) -> dict[str, Any]:
    return {
        "zarr_path": str(plan.zarr_path),
        "crop_run": plan.crop_run,
        "source_path": plan.source_path,
        "status": plan.status,
        "message": plan.message,
        "crop_rows": plan.crop_rows,
        "source_rows": plan.source_rows,
        "mappable_rows": plan.mappable_rows,
        "unmappable_rows": plan.unmappable_rows,
        "array_plans": [
            {"name": item.name, "action": item.action, "existing": item.existing}
            for item in plan.array_plans
        ],
        "applied": bool(applied),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--crop-run",
        action="append",
        default=[],
        help="Crop run(s) to backfill (default: latest crop run). May be specified multiple times.",
    )
    parser.add_argument(
        "--limit",
        choices=["latest", "all"],
        default="latest",
        help="When --crop-run is not set, inspect latest crop run or all crop runs (default: latest).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing identity arrays when their values differ from the source mapping.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes only (default behavior).")
    parser.add_argument("--apply", action="store_true", help="Write missing or selected row-lineage arrays.")
    parser.add_argument(
        "--consolidate-metadata",
        action="store_true",
        help="After --apply, run zarr.consolidate_metadata on each modified archive.",
    )
    parser.add_argument("--json-report", type=Path, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        raise SystemExit("Choose either --apply or --dry-run, not both.")
    if args.consolidate_metadata and not args.apply:
        raise SystemExit("--consolidate-metadata requires --apply.")

    apply = bool(args.apply)
    roots = _resolve_roots(args.paths)
    plans: list[tuple[CropPlan, bool]] = []
    modified_zarrs: set[Path] = set()
    scanned_zarrs = 0
    scanned_crop_runs = 0
    errors = 0

    for zarr_path in _iter_zarr(roots, recursive=bool(args.recursive)):
        if args.zarr_use != "any" and _infer_zarr_use(zarr_path) != args.zarr_use:
            continue
        scanned_zarrs += 1
        for crop_run in _select_crop_runs(zarr_path, requested=args.crop_run, limit=args.limit):
            scanned_crop_runs += 1
            try:
                plan = plan_crop_backfill(zarr_path, crop_run, overwrite=bool(args.overwrite))
            except Exception as exc:
                errors += 1
                plan = CropPlan(
                    zarr_path=zarr_path,
                    crop_run=crop_run,
                    source_path="",
                    crop_rows=0,
                    source_rows=0,
                    mappable_rows=0,
                    unmappable_rows=0,
                    array_plans=(),
                    status="error",
                    message=str(exc),
                )

            applied = False
            if apply and plan.needs_write:
                try:
                    writes = apply_crop_backfill(plan)
                    applied = writes > 0
                    if applied:
                        modified_zarrs.add(zarr_path)
                except Exception as exc:
                    errors += 1
                    plan = CropPlan(
                        zarr_path=plan.zarr_path,
                        crop_run=plan.crop_run,
                        source_path=plan.source_path,
                        crop_rows=plan.crop_rows,
                        source_rows=plan.source_rows,
                        mappable_rows=plan.mappable_rows,
                        unmappable_rows=plan.unmappable_rows,
                        array_plans=plan.array_plans,
                        status="error",
                        message=str(exc),
                    )
                    applied = False

            mode = "apply" if applied else "plan"
            if plan.status in {"planned", "skipped", "error"} or applied:
                print(
                    f"{mode}: {zarr_path}:{plan.crop_run} status={plan.status} "
                    f"rows={plan.crop_rows} source_rows={plan.source_rows} "
                    f"mappable={plan.mappable_rows} unmappable={plan.unmappable_rows} "
                    f"arrays={_format_array_actions(plan)}"
                    + (f" message={plan.message}" if plan.message else "")
                )
            plans.append((plan, applied))

    consolidated = 0
    if args.consolidate_metadata and modified_zarrs:
        for zarr_path in sorted(modified_zarrs):
            try:
                zarr.consolidate_metadata(str(zarr_path))
                consolidated += 1
            except Exception as exc:
                errors += 1
                print(f"error: failed to consolidate metadata for {zarr_path}: {exc}")

    planned_writes = sum(1 for plan, _ in plans if plan.needs_write)
    applied_writes = sum(1 for _, applied in plans if applied)
    skipped = sum(1 for plan, _ in plans if plan.status == "skipped")
    up_to_date = sum(1 for plan, _ in plans if plan.status == "up_to_date")
    total_crop_rows = sum(plan.crop_rows for plan, _ in plans)
    total_mappable_rows = sum(plan.mappable_rows for plan, _ in plans)
    total_unmappable_rows = sum(plan.unmappable_rows for plan, _ in plans)

    print(f"zarr_scanned: {scanned_zarrs}")
    print(f"crop_runs_scanned: {scanned_crop_runs}")
    print(f"planned_crop_run_writes: {planned_writes}")
    if apply:
        print(f"applied_crop_run_writes: {applied_writes}")
    print(f"up_to_date: {up_to_date}")
    print(f"skipped: {skipped}")
    print(f"total_crop_rows: {total_crop_rows}")
    print(f"total_mappable_rows: {total_mappable_rows}")
    print(f"total_unmappable_rows: {total_unmappable_rows}")
    if args.consolidate_metadata:
        print(f"metadata_consolidated: {consolidated}")
    if errors:
        print(f"errors: {errors}")

    if args.json_report:
        report_path = args.json_report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "mode": "apply" if apply else "dry-run",
            "summary": {
                "zarr_scanned": scanned_zarrs,
                "crop_runs_scanned": scanned_crop_runs,
                "planned_crop_run_writes": planned_writes,
                "applied_crop_run_writes": applied_writes if apply else 0,
                "up_to_date": up_to_date,
                "skipped": skipped,
                "total_crop_rows": total_crop_rows,
                "total_mappable_rows": total_mappable_rows,
                "total_unmappable_rows": total_unmappable_rows,
                "metadata_consolidated": consolidated,
                "errors": errors,
            },
            "crop_runs": [_plan_record(plan, applied=applied) for plan, applied in plans],
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    if errors:
        return 1
    if not apply and planned_writes:
        print("Dry-run only. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

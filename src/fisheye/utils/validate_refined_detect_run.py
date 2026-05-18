"""Validate a refined-detect run before downstream/finalizer use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.shared.refined_detect_curation import (
    has_curated_refined_detect_surface,
    has_curated_refined_source_detections_projection,
)
from fisheye.shared.refined_detect_identity import collect_refined_detect_identity_validation
from fisheye.shared.type_conversions import normalize_attr


VALIDATION_SCHEMA = "palette.refined_detect_run_validation.v1"


def _normalize_group_path(path: str) -> str:
    value = str(path or "").strip().strip("/")
    if not value:
        raise ValueError("group path must be non-empty")
    parts = Path(value).parts
    if any(part == ".." for part in parts):
        raise ValueError(f"group path must not contain '..': {path!r}")
    return value


def _join_group_path(*parts: str) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(zarr_path), mode="r")


def _resolve_target_path(
    *,
    target_group_path: Optional[str],
    refined_family_path: Optional[str],
    run_name: Optional[str],
) -> str:
    if target_group_path:
        value = _normalize_group_path(target_group_path)
        parts = value.split("/")
        if len(parts) < 2:
            raise ValueError("--target-group-path must end in '<family>/<run_name>'")
        return value
    if not refined_family_path or not run_name:
        raise ValueError("provide --target-group-path or both --refined-family-path and --run-name")
    return _join_group_path(_normalize_group_path(refined_family_path), str(run_name).strip())


def _group_exists(root: zarr.Group, path: str) -> bool:
    try:
        root[path]
        return True
    except Exception:
        return False


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _array_chunks(array: Any) -> tuple[int, ...] | None:
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return None
    try:
        return tuple(int(value) for value in chunks)
    except TypeError:
        return None


def _bbox_chunk_validation(refined: zarr.Group) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for group_name in ("instances", "source_detections"):
        group = refined.get(group_name)
        if group is None:
            continue
        for array_name in ("bbox_img_xyxy", "bbox_norm_coords"):
            if array_name not in group:
                continue
            array = group[array_name]
            shape = tuple(int(value) for value in getattr(array, "shape", ()))
            chunks = _array_chunks(array)
            status = "pass"
            message = None
            if len(shape) != 2 or shape[1] != 4:
                status = "warning"
                message = f"{group_name}/{array_name} has unexpected shape {shape}; expected [N, 4]"
            elif chunks is None:
                status = "warning"
                message = f"{group_name}/{array_name} chunk shape is unavailable"
            elif len(chunks) != 2 or chunks[1] != 4:
                status = "warning"
                message = (
                    f"{group_name}/{array_name} chunk shape {chunks} splits the fixed-width bbox columns; "
                    "preferred chunk shape is [row_chunk, 4]"
                )
            reports.append(
                {
                    "group": group_name,
                    "array": array_name,
                    "shape": list(shape),
                    "chunks": list(chunks) if chunks is not None else None,
                    "status": status,
                    "message": message,
                }
            )
    return reports


def validate_refined_detect_run(
    zarr_path: str | Path,
    *,
    target_group_path: Optional[str] = None,
    refined_family_path: Optional[str] = None,
    run_name: Optional[str] = None,
    require_source_detections: bool = True,
    require_source_detect_exists: bool = True,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    root = _open_root(archive_path)
    resolved_path = _resolve_target_path(
        target_group_path=target_group_path,
        refined_family_path=refined_family_path,
        run_name=run_name,
    )
    validations: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    try:
        refined = root[resolved_path]
    except Exception as exc:
        return {
            "status": "failed",
            "schema_version": VALIDATION_SCHEMA,
            "zarr_path": str(archive_path),
            "target_group_path": resolved_path,
            "errors": [f"refined group not found: {exc}"],
            "warnings": [],
            "validations": [],
        }

    surface_present = has_curated_refined_detect_surface(refined)
    validations.append({"name": "curated_surface_present", "status": "pass" if surface_present else "fail"})
    if not surface_present:
        errors.append("refined run is missing a curated detect surface")

    source_projection_present = has_curated_refined_source_detections_projection(refined)
    source_projection_status = "pass" if source_projection_present else "fail"
    if not require_source_detections and not source_projection_present:
        source_projection_status = "skip"
    validations.append(
        {
            "name": "source_detections_projection_present",
            "status": source_projection_status,
            "required": bool(require_source_detections),
        }
    )
    if require_source_detections and not source_projection_present:
        errors.append("refined run is missing source_detections projection")

    identity_validation = collect_refined_detect_identity_validation(refined)
    validations.append(
        {
            "name": "identity_validation",
            "status": "pass" if identity_validation.get("ok") else "fail",
            "error_count": identity_validation.get("error_count"),
            "warning_count": identity_validation.get("warning_count"),
        }
    )
    if not identity_validation.get("ok"):
        errors.append("refined detect identity validation failed")

    source_detect_path = normalize_attr(refined.attrs.get("source_detect_path"))
    source_detect_run = normalize_attr(refined.attrs.get("source_detect_run"))
    source_quality_run = normalize_attr(refined.attrs.get("source_quality_run"))
    refined_family = normalize_attr(refined.attrs.get("refined_family_path"))
    source_exists = bool(source_detect_path and _group_exists(root, source_detect_path))
    validations.append(
        {
            "name": "source_detect_path_exists",
            "status": "pass" if source_exists else ("skip" if not require_source_detect_exists else "fail"),
            "required": bool(require_source_detect_exists),
            "source_detect_path": source_detect_path,
        }
    )
    if require_source_detect_exists and not source_exists:
        errors.append("source_detect_path does not resolve to an existing group")
    if not source_detect_run:
        warnings.append("source_detect_run attr is missing")
    if not source_quality_run:
        warnings.append("source_quality_run attr is missing")
    if refined_family and not resolved_path.startswith(f"{refined_family}/"):
        warnings.append("refined_family_path attr does not match target_group_path prefix")

    instances = refined.get("instances")
    source_detections = refined.get("source_detections")
    summary_statistics = _optional_dict(refined.attrs.get("summary_statistics"))
    coverage_comparison = _optional_dict(refined.attrs.get("coverage_comparison"))
    row_counts = {
        "instances": int(instances["frame_indices"].shape[0])
        if instances is not None and "frame_indices" in instances
        else None,
        "frame_counts": int(instances["frame_counts"].shape[0])
        if instances is not None and "frame_counts" in instances
        else None,
        "source_detections": int(source_detections["source_detect_row_index"].shape[0])
        if source_detections is not None and "source_detect_row_index" in source_detections
        else None,
    }
    bbox_chunk_validation = _bbox_chunk_validation(refined)
    for item in bbox_chunk_validation:
        if item["status"] != "pass" and item.get("message"):
            warnings.append(str(item["message"]))

    return {
        "status": "ok" if not errors else "failed",
        "schema_version": VALIDATION_SCHEMA,
        "zarr_path": str(archive_path),
        "target_group_path": resolved_path,
        "run_name": resolved_path.rsplit("/", 1)[-1],
        "source_detect_run": source_detect_run,
        "source_detect_path": source_detect_path,
        "source_quality_run": source_quality_run,
        "refined_family_path": refined_family,
        "row_counts": row_counts,
        "bbox_chunk_validation": bbox_chunk_validation,
        "summary_statistics": summary_statistics,
        "coverage_comparison": coverage_comparison,
        "identity_validation": identity_validation,
        "validations": validations,
        "errors": errors,
        "warnings": warnings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a refined-detect run group.")
    parser.add_argument("zarr_path", type=Path, help="Analysis Zarr path")
    parser.add_argument("--target-group-path", default=None, help="Full refined run group path")
    parser.add_argument("--refined-family-path", default=None, help="Refined run family path")
    parser.add_argument("--run-name", default=None, help="Refined run name")
    parser.add_argument(
        "--no-require-source-detections",
        action="store_true",
        help="Do not fail if source_detections projection is missing",
    )
    parser.add_argument(
        "--no-require-source-detect",
        action="store_true",
        help="Do not fail if source_detect_path does not resolve",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = validate_refined_detect_run(
        args.zarr_path,
        target_group_path=args.target_group_path,
        refined_family_path=args.refined_family_path,
        run_name=args.run_name,
        require_source_detections=not args.no_require_source_detections,
        require_source_detect_exists=not args.no_require_source_detect,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validation and readback helpers for bout classification runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import read_columnar_dataset
from fisheye.utils.zarr_io import open_zarr_root

BOUT_CLASSIFICATION_SCHEMA_ID = "analysis.bout_classification_runs"
BOUT_CLASSIFICATION_SCHEMA_VERSION = 1
PER_BOUT_SCHEMA_ID = f"{BOUT_CLASSIFICATION_SCHEMA_ID}.per_bout"

REQUIRED_RUN_ATTRS = (
    "schema_id",
    "schema_version",
    "classifier_family",
    "classifier_name",
    "source_mode",
    "row_axis",
    "invalid_window_policy",
    "source_refs",
    "parameters",
)
RECOMMENDED_RUN_ATTRS = (
    "classifier_version",
    "adapter_method",
    "adapter_method_version",
    "tail_angle_conversion",
    "trajectory_conversion",
    "invalid_frame_policy",
    "provenance",
)
REQUIRED_PER_BOUT_FIELDS = (
    "source_bout_id",
    "start_frame",
    "end_frame",
    "window_start_frame",
    "window_end_frame",
    "HB1_frame",
    "HB1_offset_frames",
    "category_id",
    "category_label_bytes",
    "subcategory_id",
    "sign",
    "probability",
    "tail_valid_fraction",
    "traj_valid_fraction",
    "max_consecutive_tail_invalid",
    "max_consecutive_traj_invalid",
    "source_window_valid",
    "classified",
    "valid",
    "failure_reason_bytes",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _require_group(parent: zarr.Group, name: str) -> zarr.Group:
    group = parent.get(name)
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Missing required group: {parent.name}/{name}")
    return group


def resolve_bout_classification_run(
    root: zarr.Group,
    run_name: str = "latest",
) -> tuple[zarr.Group, str, str]:
    """Resolve a bout-classification run by name, path, or ``latest``."""

    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "bout_classification_runs")
    spec = str(run_name or "latest").strip().strip("/")
    if spec.startswith("analysis/bout_classification_runs/"):
        parts = spec.split("/")
        resolved = parts[2] if len(parts) >= 3 else ""
    elif spec == "latest":
        resolved = parent.attrs.get("latest")
    else:
        resolved = spec
    if not resolved or str(resolved) not in parent:
        raise ValueError(f"Bout classification run {run_name!r} not found.")
    resolved_name = str(resolved)
    return parent[resolved_name], resolved_name, f"analysis/bout_classification_runs/{resolved_name}"


def load_bout_classification_table(run_group: zarr.Group) -> np.ndarray:
    """Load the `per_bout` columnar table for a resolved classification run."""

    per_bout = run_group.get("per_bout")
    if not isinstance(per_bout, zarr.Group):
        raise ValueError(f"Missing required group: {run_group.name}/per_bout")
    return read_columnar_dataset(per_bout)


def _decode_string_column(values: np.ndarray) -> list[str]:
    labels: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            labels.append(value.rstrip(b"\x00").decode("utf-8", "replace"))
        elif isinstance(value, str):
            labels.append(value.rstrip("\x00"))
        else:
            labels.append(str(value))
    return labels


def _count_strings(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = int(counts.get(key, 0) + 1)
    return dict(sorted(counts.items()))


def _probability_summary(values: np.ndarray) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"finite_count": 0, "min": math.nan, "mean": math.nan, "max": math.nan}
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def validate_bout_classification_run(
    root: zarr.Group,
    run_name: str = "latest",
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Validate the generic bout-classification run contract.

    Non-strict mode treats newer provenance/conversion attrs as warnings so
    early canary runs remain readable. Strict mode requires those recommended
    fields too.
    """

    errors: list[str] = []
    warnings: list[str] = []
    try:
        run_group, resolved_name, run_path = resolve_bout_classification_run(root, run_name)
    except Exception as exc:
        return {
            "ok": False,
            "run_name": str(run_name),
            "run_path": None,
            "errors": [str(exc)],
            "warnings": [],
        }

    attrs = run_group.attrs
    for attr_name in REQUIRED_RUN_ATTRS:
        if attr_name not in attrs:
            errors.append(f"missing required run attr: {attr_name}")
    for attr_name in RECOMMENDED_RUN_ATTRS:
        if attr_name not in attrs:
            warnings.append(f"missing recommended run attr: {attr_name}")

    if attrs.get("schema_id") != BOUT_CLASSIFICATION_SCHEMA_ID:
        errors.append(
            f"schema_id {attrs.get('schema_id')!r} != {BOUT_CLASSIFICATION_SCHEMA_ID!r}"
        )
    if int(attrs.get("schema_version", -1)) != BOUT_CLASSIFICATION_SCHEMA_VERSION:
        errors.append(
            "schema_version "
            f"{attrs.get('schema_version')!r} != {BOUT_CLASSIFICATION_SCHEMA_VERSION}"
        )
    if attrs.get("row_axis") not in {None, "swim_bout_rows"}:
        errors.append(f"row_axis must be 'swim_bout_rows', got {attrs.get('row_axis')!r}")

    per_bout = run_group.get("per_bout")
    table: Optional[np.ndarray] = None
    if not isinstance(per_bout, zarr.Group):
        errors.append("missing required per_bout group")
    else:
        field_names = list(per_bout.attrs.get("field_names", []))
        if per_bout.attrs.get("storage_layout") != "columnar":
            errors.append("per_bout.storage_layout must be 'columnar'")
        missing_fields = [field for field in REQUIRED_PER_BOUT_FIELDS if field not in field_names]
        for field in missing_fields:
            errors.append(f"missing required per_bout field: {field}")
        for field in field_names:
            if field not in per_bout:
                errors.append(f"per_bout field listed but missing array: {field}")
        if not missing_fields and not errors:
            try:
                table = read_columnar_dataset(per_bout)
            except Exception as exc:
                errors.append(f"failed to read per_bout table: {exc}")

    if table is not None:
        row_count = int(table.shape[0])
        if int(attrs.get("source_bout_count", row_count)) != row_count:
            errors.append(
                f"source_bout_count attr {attrs.get('source_bout_count')!r} != per_bout rows {row_count}"
            )
        classified = np.asarray(table["classified"], dtype=bool)
        valid = np.asarray(table["valid"], dtype=bool)
        source_window_valid = np.asarray(table["source_window_valid"], dtype=bool)
        if np.any(classified & ~source_window_valid):
            errors.append("classified rows must have source_window_valid=true")
        if np.any(valid & ~classified):
            errors.append("valid rows must have classified=true")
        if np.any(classified & (np.asarray(table["category_id"]) < 0)):
            errors.append("classified rows must have non-negative category_id")
        if np.any(classified & ~np.isfinite(np.asarray(table["probability"], dtype=np.float64))):
            errors.append("classified rows must have finite probability")
        skipped = ~classified
        if np.any(skipped & (np.asarray(table["category_id"]) != -1)):
            warnings.append("unclassified rows should use category_id=-1")
        if int(attrs.get("classified_bout_count", int(np.count_nonzero(classified)))) != int(np.count_nonzero(classified)):
            errors.append("classified_bout_count attr does not match per_bout classified rows")
        if int(attrs.get("valid_source_window_count", int(np.count_nonzero(source_window_valid)))) != int(np.count_nonzero(source_window_valid)):
            errors.append("valid_source_window_count attr does not match per_bout source_window_valid rows")

    strict_errors = errors + warnings if strict else errors
    return {
        "ok": not strict_errors,
        "strict": bool(strict),
        "run_name": resolved_name,
        "run_path": run_path,
        "schema_id": attrs.get("schema_id"),
        "schema_version": attrs.get("schema_version"),
        "errors": errors,
        "warnings": warnings,
    }


def summarize_bout_classification_run(
    root: zarr.Group,
    run_name: str = "latest",
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Return a JSON-safe summary plus validation result for one run."""

    validation = validate_bout_classification_run(root, run_name, strict=strict)
    if not validation["ok"] and validation.get("run_path") is None:
        return validation
    run_group, resolved_name, run_path = resolve_bout_classification_run(root, run_name)
    table = load_bout_classification_table(run_group)
    labels = _decode_string_column(table["category_label_bytes"])
    reasons = _decode_string_column(table["failure_reason_bytes"])
    classified = np.asarray(table["classified"], dtype=bool)
    source_window_valid = np.asarray(table["source_window_valid"], dtype=bool)
    valid = np.asarray(table["valid"], dtype=bool)
    summary = {
        "ok": bool(validation["ok"]),
        "validation": validation,
        "run_name": resolved_name,
        "run_path": run_path,
        "schema_id": run_group.attrs.get("schema_id"),
        "schema_version": run_group.attrs.get("schema_version"),
        "classifier_family": run_group.attrs.get("classifier_family"),
        "classifier_name": run_group.attrs.get("classifier_name"),
        "classifier_version": run_group.attrs.get("classifier_version"),
        "source_bout_count": int(table.shape[0]),
        "source_window_valid_count": int(np.count_nonzero(source_window_valid)),
        "classified_bout_count": int(np.count_nonzero(classified)),
        "valid_bout_count": int(np.count_nonzero(valid)),
        "skipped_bout_count": int(table.shape[0] - np.count_nonzero(classified)),
        "category_counts": _count_strings(labels),
        "failure_reason_counts": _count_strings(reasons),
        "probability": _probability_summary(table["probability"][classified]),
        "invalid_window_policy": run_group.attrs.get("invalid_window_policy"),
        "source_refs": run_group.attrs.get("source_refs", {}),
        "parameters": run_group.attrs.get("parameters", {}),
    }
    return _json_safe(summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and summarize Palette bout-classification runs.")
    parser.add_argument("zarr_path", type=Path, help="Palette Zarr archive.")
    parser.add_argument("--run", default="latest", help="Run name/path or latest.")
    parser.add_argument("--strict", action="store_true", help="Treat recommended contract fields as required.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root = open_zarr_root(args.zarr_path, mode="r")
    summary = summarize_bout_classification_run(root, args.run, strict=bool(args.strict))
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit subject-mask training manifest sources against their source Zarrs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.utils.export_subject_mask_training_zarr import (
    SUBJECT_STAGE_CHOICES,
    TARGET_SCHEMAS,
)

OpenZarr = Callable[[Path], zarr.Group]


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, np.ndarray):
        return _normalize_str_list(value.reshape(-1).tolist())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [text for item in value if (text := str(item).strip())]
    return []


def _open_group(path: Path) -> zarr.Group:
    return zarr.open_group(str(path.expanduser().resolve()), mode="r", use_consolidated=False)


def _is_group(value: Any) -> bool:
    return isinstance(value, zarr.Group)


def _is_array(value: Any) -> bool:
    return isinstance(value, zarr.Array)


def _source_name(source: Mapping[str, Any], index: int) -> str:
    return _as_text(source.get("name")) or _as_text(source.get("dataset_id")) or f"source_{index}"


def _source_zarr_path(source: Mapping[str, Any]) -> Optional[Path]:
    value = source.get("zarr_path") or source.get("source_zarr") or source.get("source_zarr_path")
    text = _as_text(value)
    return Path(text).expanduser() if text else None


def _source_stage_group(source: Mapping[str, Any]) -> str:
    return (
        _as_text(source.get("source_stage_group"))
        or _as_text(source.get("stage_group"))
        or "subject_mask_runs"
    )


def _source_subject_run(source: Mapping[str, Any]) -> Optional[str]:
    return (
        _as_text(source.get("source_subject_mask_run"))
        or _as_text(source.get("source_subject_run"))
        or _as_text(source.get("subject_mask_run"))
        or _as_text(source.get("subject_run"))
    )


def _source_crop_run(source: Mapping[str, Any]) -> Optional[str]:
    return _as_text(source.get("source_crop_run")) or _as_text(source.get("crop_run"))


def _expected_available_components(source: Mapping[str, Any]) -> list[str]:
    return sorted(set(_normalize_str_list(source.get("available_components"))))


def _component_quality_rows(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = source.get("component_quality")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _review_payload(subject_group: zarr.Group, component_name: str) -> Mapping[str, Any]:
    reviews = subject_group.attrs.get("component_review_statuses")
    if not isinstance(reviews, Mapping):
        return {}
    payload = reviews.get(component_name)
    return payload if isinstance(payload, Mapping) else {}


def _review_state(subject_group: zarr.Group, component_name: str) -> Optional[str]:
    return _as_text(_review_payload(subject_group, component_name).get("state"))


def _review_intended_use(subject_group: zarr.Group, component_name: str) -> Optional[str]:
    return _as_text(_review_payload(subject_group, component_name).get("intended_use"))


def _read_available_channels(
    subject_group: zarr.Group,
    *,
    labels: Sequence[str],
    context: str,
    errors: list[str],
) -> Optional[np.ndarray]:
    if "available_channels" not in subject_group:
        errors.append(f"{context}: missing required array available_channels")
        return None
    try:
        available = np.asarray(subject_group["available_channels"][:], dtype=np.bool_).reshape(-1)
    except Exception as exc:
        errors.append(f"{context}: failed reading available_channels ({exc})")
        return None
    if int(available.shape[0]) != len(labels):
        errors.append(
            f"{context}: available_channels length mismatch "
            f"({available.shape[0]} != mask_labels {len(labels)})"
        )
        return None
    return available


def _mask_present_rates_from_metrics(
    subject_group: zarr.Group,
    *,
    labels: Sequence[str],
    total_rows: int,
    context: str,
    warnings: list[str],
) -> dict[str, float]:
    metrics = subject_group.get("metrics")
    if not _is_group(metrics) or "mask_present" not in metrics:
        return {}
    try:
        present = np.asarray(metrics["mask_present"][:], dtype=np.bool_)
    except Exception as exc:
        warnings.append(f"{context}: failed reading metrics/mask_present; rate parity skipped ({exc})")
        return {}
    if present.ndim != 2 or int(present.shape[0]) != total_rows or int(present.shape[1]) != len(labels):
        warnings.append(
            f"{context}: metrics/mask_present shape {tuple(present.shape)} does not match "
            f"(rows={total_rows}, channels={len(labels)}); rate parity skipped"
        )
        return {}
    if total_rows <= 0:
        return {str(label): 0.0 for label in labels}
    return {
        str(label): float(np.count_nonzero(present[:, index])) / float(total_rows)
        for index, label in enumerate(labels)
    }


def _mask_present_rates_from_masks(
    subject_group: zarr.Group,
    *,
    labels: Sequence[str],
    total_rows: int,
    context: str,
    warnings: list[str],
) -> dict[str, float]:
    if "masks_roi" not in subject_group:
        return {}
    try:
        masks = np.asarray(subject_group["masks_roi"][:], dtype=np.uint8)
    except Exception as exc:
        warnings.append(f"{context}: failed reading masks_roi; rate parity skipped ({exc})")
        return {}
    if masks.ndim != 4 or int(masks.shape[0]) != total_rows or int(masks.shape[1]) != len(labels):
        warnings.append(
            f"{context}: masks_roi shape {tuple(masks.shape)} does not match "
            f"(rows={total_rows}, channels={len(labels)}); rate parity skipped"
        )
        return {}
    if total_rows <= 0:
        return {str(label): 0.0 for label in labels}
    present = np.any(masks > 0, axis=(2, 3))
    return {
        str(label): float(np.count_nonzero(present[:, index])) / float(total_rows)
        for index, label in enumerate(labels)
    }


def _resolve_subject_run(
    root: zarr.Group,
    *,
    stage_group: str,
    run_name: Optional[str],
    context: str,
    errors: list[str],
) -> tuple[Optional[str], Optional[zarr.Group]]:
    if stage_group not in SUBJECT_STAGE_CHOICES:
        errors.append(
            f"{context}: unsupported source_stage_group {stage_group!r}; "
            f"expected one of {sorted(SUBJECT_STAGE_CHOICES)}"
        )
        return None, None
    parent = root.get(stage_group)
    if not _is_group(parent):
        errors.append(f"{context}: missing required group {stage_group}")
        return None, None
    resolved = run_name
    if resolved is None:
        latest = _as_text(parent.attrs.get("latest"))
        if latest and latest in parent:
            resolved = latest
    if resolved is None:
        keys = sorted(str(key) for key in parent.keys())
        if keys:
            resolved = keys[-1]
    if resolved is None:
        errors.append(f"{context}: no subject-mask runs found under {stage_group}")
        return None, None
    if resolved not in parent:
        errors.append(f"{context}: source run not found: {stage_group}/{resolved}")
        return resolved, None
    run_group = parent[resolved]
    if not _is_group(run_group):
        errors.append(f"{context}: expected group at {stage_group}/{resolved}")
        return resolved, None
    return resolved, run_group


def _resolve_crop_group(
    root: zarr.Group,
    subject_group: zarr.Group,
    *,
    requested_crop_run: Optional[str],
    context: str,
    warnings: list[str],
    errors: list[str],
) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = root.get("crop_runs")
    if not _is_group(parent):
        errors.append(f"{context}: missing required group crop_runs")
        return None, None

    attr_crop = _as_text(subject_group.attrs.get("source_crop_run"))
    if requested_crop_run and attr_crop and requested_crop_run != attr_crop:
        errors.append(
            f"{context}: manifest source_crop_run {requested_crop_run!r} does not match "
            f"run attr source_crop_run {attr_crop!r}"
        )
    crop_run = requested_crop_run or attr_crop
    if crop_run is None:
        latest = _as_text(parent.attrs.get("latest"))
        if latest and latest in parent:
            crop_run = latest
            warnings.append(f"{context}: source_crop_run missing; falling back to crop_runs latest {latest!r}")
    if crop_run is None:
        keys = sorted(str(key) for key in parent.keys())
        if keys:
            crop_run = keys[-1]
            warnings.append(f"{context}: source_crop_run missing; falling back to last crop run {crop_run!r}")
    if crop_run is None:
        errors.append(f"{context}: no crop runs found under crop_runs")
        return None, None
    if crop_run not in parent:
        errors.append(f"{context}: crop run not found: crop_runs/{crop_run}")
        return crop_run, None
    crop_group = parent[crop_run]
    if not _is_group(crop_group):
        errors.append(f"{context}: expected group at crop_runs/{crop_run}")
        return crop_run, None
    return crop_run, crop_group


def _audit_component_quality(
    *,
    source: Mapping[str, Any],
    subject_group: zarr.Group,
    stage_group: str,
    labels: Sequence[str],
    available: np.ndarray,
    total_rows: int,
    rate_by_component: Mapping[str, float],
    allow_unapproved_refined: bool,
    context: str,
    warnings: list[str],
    errors: list[str],
) -> None:
    label_set = set(labels)
    available_by_component = {
        str(label): bool(available[index])
        for index, label in enumerate(labels)
    }

    has_expected_available_components = "available_components" in source
    expected_available = _expected_available_components(source)
    actual_available = sorted(component for component, is_available in available_by_component.items() if is_available)
    if has_expected_available_components and expected_available != actual_available:
        errors.append(
            f"{context}: available_components mismatch "
            f"(manifest={expected_available!r}, disk={actual_available!r})"
        )
    elif not has_expected_available_components:
        warnings.append(f"{context}: manifest source missing available_components; availability parity is partial")

    quality_rows = _component_quality_rows(source)
    if not quality_rows:
        warnings.append(f"{context}: manifest source has no component_quality rows; registry parity is partial")

    seen_components: set[str] = set()
    for row in quality_rows:
        component = _as_text(row.get("component_name"))
        if component is None:
            errors.append(f"{context}: component_quality row missing component_name")
            continue
        seen_components.add(component)
        if component not in label_set:
            errors.append(f"{context}: registry component {component!r} is absent from disk mask_labels")
            continue

        expected_available_value = row.get("available")
        if expected_available_value is not None:
            expected_bool = _as_bool(expected_available_value)
            actual_bool = available_by_component[component]
            if expected_bool != actual_bool:
                errors.append(
                    f"{context}: component {component!r} availability mismatch "
                    f"(registry={expected_bool}, disk={actual_bool})"
                )

        expected_state = _as_text(row.get("review_state"))
        actual_state = _review_state(subject_group, component)
        if expected_state is not None and expected_state != actual_state:
            errors.append(
                f"{context}: component {component!r} review_state mismatch "
                f"(registry={expected_state!r}, disk={actual_state!r})"
            )

        expected_intended_use = _as_text(row.get("review_intended_use"))
        actual_intended_use = _review_intended_use(subject_group, component)
        if expected_intended_use is not None and expected_intended_use != actual_intended_use:
            errors.append(
                f"{context}: component {component!r} review_intended_use mismatch "
                f"(registry={expected_intended_use!r}, disk={actual_intended_use!r})"
            )

        expected_rate = _as_float(row.get("rows_with_component_mask_rate"))
        actual_rate = rate_by_component.get(component)
        if expected_rate is not None:
            if actual_rate is None:
                warnings.append(f"{context}: component {component!r} mask-present rate not checked on disk")
            elif abs(float(expected_rate) - float(actual_rate)) > 1e-6:
                errors.append(
                    f"{context}: component {component!r} rows_with_component_mask_rate mismatch "
                    f"(registry={float(expected_rate):.6f}, disk={float(actual_rate):.6f})"
                )

    missing_quality = sorted(label_set - seen_components)
    if quality_rows and missing_quality:
        warnings.append(
            f"{context}: component_quality missing disk label(s) {missing_quality!r}; "
            "registry may be incomplete for this run"
        )

    if stage_group == "refined_subject_masks_runs" and not allow_unapproved_refined:
        for component, is_available in available_by_component.items():
            if not is_available:
                continue
            state = _review_state(subject_group, component)
            if state != "approved":
                errors.append(
                    f"{context}: refined component {component!r} is not approved for training export "
                    f"(disk review_state={state or 'missing'!r})"
                )

    if bool(source.get("canonical_latest_requires_assembly")):
        warnings.append(
            f"{context}: manifest says canonical latest exportable components require assembly; "
            "the selected single source run is still audited, but it may not represent every latest component"
        )


def _audit_one_source(
    source: Mapping[str, Any],
    *,
    index: int,
    open_zarr: OpenZarr,
    allow_unapproved_refined: bool,
    read_masks_for_rates: bool,
) -> dict[str, Any]:
    name = _source_name(source, index)
    errors: list[str] = []
    warnings: list[str] = []
    zarr_path = _source_zarr_path(source)
    stage_group = _source_stage_group(source)
    requested_run = _source_subject_run(source)
    requested_crop_run = _source_crop_run(source)
    context = f"{name}"

    if zarr_path is None:
        errors.append(f"{context}: missing zarr_path")
        return {
            "name": name,
            "valid": False,
            "zarr_path": None,
            "source_stage_group": stage_group,
            "source_subject_mask_run": requested_run,
            "source_crop_run": requested_crop_run,
            "total_samples": None,
            "mask_labels": [],
            "available_components": [],
            "errors": errors,
            "warnings": warnings,
        }

    try:
        root = open_zarr(zarr_path)
    except Exception as exc:
        errors.append(f"{context}: failed opening source zarr {zarr_path} ({exc})")
        return {
            "name": name,
            "valid": False,
            "zarr_path": str(zarr_path),
            "source_stage_group": stage_group,
            "source_subject_mask_run": requested_run,
            "source_crop_run": requested_crop_run,
            "total_samples": None,
            "mask_labels": [],
            "available_components": [],
            "errors": errors,
            "warnings": warnings,
        }

    resolved_run, subject_group = _resolve_subject_run(
        root,
        stage_group=stage_group,
        run_name=requested_run,
        context=context,
        errors=errors,
    )
    if subject_group is None:
        return {
            "name": name,
            "valid": False,
            "zarr_path": str(zarr_path),
            "source_stage_group": stage_group,
            "source_subject_mask_run": resolved_run or requested_run,
            "source_crop_run": requested_crop_run,
            "total_samples": None,
            "mask_labels": [],
            "available_components": [],
            "errors": errors,
            "warnings": warnings,
        }

    subject_context = f"{context}: {stage_group}/{resolved_run}"
    labels = _normalize_str_list(subject_group.attrs.get("mask_labels"))
    if not labels:
        errors.append(f"{subject_context}: missing usable mask_labels attr")

    schema_id = _as_text(subject_group.attrs.get("label_schema_id"))
    expected_schema_id = _as_text(source.get("label_schema_id"))
    if expected_schema_id is not None and schema_id != expected_schema_id:
        errors.append(
            f"{subject_context}: label_schema_id mismatch "
            f"(manifest={expected_schema_id!r}, disk={schema_id!r})"
        )
    if schema_id not in TARGET_SCHEMAS:
        errors.append(
            f"{subject_context}: unsupported label_schema_id {schema_id!r}; "
            f"expected one of {sorted(TARGET_SCHEMAS)}"
        )
    elif labels and labels != list(TARGET_SCHEMAS[schema_id]):
        errors.append(
            f"{subject_context}: mask_labels mismatch for {schema_id} "
            f"(disk={labels!r}, expected={list(TARGET_SCHEMAS[schema_id])!r})"
        )

    masks = subject_group.get("masks_roi")
    total_rows: Optional[int] = None
    if not _is_array(masks):
        errors.append(f"{subject_context}: missing required array masks_roi")
    else:
        shape = tuple(int(dim) for dim in masks.shape)
        if len(shape) != 4:
            errors.append(f"{subject_context}: masks_roi must be 4D (N,C,H,W), got {shape}")
        else:
            total_rows = shape[0]
            if labels and shape[1] != len(labels):
                errors.append(
                    f"{subject_context}: masks_roi channel count mismatch "
                    f"({shape[1]} != mask_labels {len(labels)})"
                )

    manifest_total = _as_int(source.get("total_samples"))
    if manifest_total is not None and total_rows is not None and manifest_total != total_rows:
        errors.append(
            f"{subject_context}: total_samples mismatch "
            f"(manifest={manifest_total}, disk={total_rows})"
        )

    available = (
        _read_available_channels(subject_group, labels=labels, context=subject_context, errors=errors)
        if labels
        else None
    )
    if available is None:
        available_components: list[str] = []
    else:
        available_components = sorted(
            str(label)
            for index_, label in enumerate(labels)
            if index_ < int(available.shape[0]) and bool(available[index_])
        )

    resolved_crop_run, crop_group = _resolve_crop_group(
        root,
        subject_group,
        requested_crop_run=requested_crop_run,
        context=subject_context,
        warnings=warnings,
        errors=errors,
    )
    if crop_group is not None and total_rows is not None:
        roi = crop_group.get("roi_images")
        if not _is_array(roi):
            errors.append(f"{subject_context}: crop_runs/{resolved_crop_run} missing roi_images")
        elif int(roi.shape[0]) != total_rows:
            errors.append(
                f"{subject_context}: row mismatch between crop_runs/{resolved_crop_run}/roi_images "
                f"({roi.shape[0]}) and masks_roi ({total_rows})"
            )

    rate_by_component: dict[str, float] = {}
    if labels and total_rows is not None:
        rate_by_component = _mask_present_rates_from_metrics(
            subject_group,
            labels=labels,
            total_rows=total_rows,
            context=subject_context,
            warnings=warnings,
        )
        if not rate_by_component and read_masks_for_rates:
            rate_by_component = _mask_present_rates_from_masks(
                subject_group,
                labels=labels,
                total_rows=total_rows,
                context=subject_context,
                warnings=warnings,
            )

    if labels and available is not None and total_rows is not None:
        _audit_component_quality(
            source=source,
            subject_group=subject_group,
            stage_group=stage_group,
            labels=labels,
            available=available,
            total_rows=total_rows,
            rate_by_component=rate_by_component,
            allow_unapproved_refined=allow_unapproved_refined,
            context=subject_context,
            warnings=warnings,
            errors=errors,
        )

    return {
        "name": name,
        "valid": not errors,
        "zarr_path": str(zarr_path),
        "source_stage_group": stage_group,
        "source_subject_mask_run": resolved_run,
        "source_crop_run": resolved_crop_run,
        "label_schema_id": schema_id,
        "total_samples": total_rows,
        "mask_labels": labels,
        "available_components": available_components,
        "component_rate_parity_checked": sorted(rate_by_component),
        "errors": errors,
        "warnings": warnings,
    }


def audit_selected_subject_mask_training_sources(
    selected_sources: Sequence[Mapping[str, Any]],
    *,
    open_zarr: Optional[OpenZarr] = None,
    allow_unapproved_refined: bool = False,
    read_masks_for_rates: bool = False,
) -> dict[str, Any]:
    """Audit selected subject-mask source rows against the source Zarr archives."""

    opener = open_zarr or _open_group
    source_summaries = [
        _audit_one_source(
            source,
            index=index,
            open_zarr=opener,
            allow_unapproved_refined=allow_unapproved_refined,
            read_masks_for_rates=read_masks_for_rates,
        )
        for index, source in enumerate(selected_sources, start=1)
        if isinstance(source, Mapping)
    ]
    errors = [
        message
        for source_summary in source_summaries
        for message in source_summary.get("errors", [])
    ]
    warnings = [
        message
        for source_summary in source_summaries
        for message in source_summary.get("warnings", [])
    ]
    return {
        "valid": not errors,
        "source_count": int(len(source_summaries)),
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
        "errors": errors,
        "warnings": warnings,
        "sources": source_summaries,
    }


def audit_subject_mask_training_manifest(
    manifest_path: Path | str,
    *,
    open_zarr: Optional[OpenZarr] = None,
    allow_unapproved_refined: bool = False,
    read_masks_for_rates: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest root must be an object: {path}")
    selected_sources = payload.get("selected_sources")
    if not isinstance(selected_sources, list) or not selected_sources:
        raise ValueError(f"Manifest missing non-empty selected_sources: {path}")
    summary = audit_selected_subject_mask_training_sources(
        selected_sources,
        open_zarr=open_zarr,
        allow_unapproved_refined=allow_unapproved_refined,
        read_masks_for_rates=read_masks_for_rates,
    )
    summary["manifest_path"] = str(path)
    summary["set_id"] = _as_text(payload.get("set_id"))
    summary["task"] = _as_text(payload.get("task"))
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="Subject-mask training manifest JSON.")
    parser.add_argument("--manifest", dest="manifest_opt", type=Path, help="Subject-mask training manifest JSON.")
    parser.add_argument(
        "--allow-unapproved-refined",
        action="store_true",
        help="Do not fail refined_subject_masks_runs sources whose available components are not approved.",
    )
    parser.add_argument(
        "--read-masks-for-rates",
        action="store_true",
        help="Read masks_roi to check component mask-present rates when metrics/mask_present is absent.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON summary.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest_path = args.manifest_opt or args.manifest
    if manifest_path is None:
        parser.error("manifest path is required")

    summary = audit_subject_mask_training_manifest(
        manifest_path,
        allow_unapproved_refined=bool(args.allow_unapproved_refined),
        read_masks_for_rates=bool(args.read_masks_for_rates),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif summary["valid"]:
        print(
            "Subject-mask training source audit passed "
            f"({summary['source_count']} sources, {summary['warning_count']} warnings)."
        )
    else:
        print("Subject-mask training source audit failed.")
        for message in summary["errors"]:
            print(f"- {message}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

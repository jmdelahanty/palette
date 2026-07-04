#!/usr/bin/env python3
"""Backfill subject_mask_tuning.eyes_union from legacy eye_mask_tuning."""

from __future__ import annotations

from fisheye.shared.zarr_helpers import infer_zarr_use as _infer_zarr_use
from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import zarr

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_io import open_zarr_root


TUNING_KEY = "subject_mask_tuning"
EYE_TUNING_KEY = "eye_mask_tuning"
SUBJECT_TUNING_VERSION = "2.0"
SUPPORTED_COMPONENTS = {"subject_body", "eyes_union", "eye_left", "eye_right", "swim_bladder"}


def _normalize_component_name(value: object) -> Optional[str]:
    text = str(value).strip().lower().replace(" ", "_")
    if not text:
        return None
    aliases = {
        "subject": "subject_body",
        "body": "subject_body",
        "eyes": "eyes_union",
        "eye_union": "eyes_union",
        "eye-union": "eyes_union",
        "eye_left": "eye_left",
        "left_eye": "eye_left",
        "left-eye": "eye_left",
        "eye_right": "eye_right",
        "right_eye": "eye_right",
        "right-eye": "eye_right",
        "swimbladder": "swim_bladder",
        "swim-bladder": "swim_bladder",
    }
    return aliases.get(text, text if text in SUPPORTED_COMPONENTS else None)


def _normalize_subject_tuning_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}

    components_raw = raw.get("components")
    if isinstance(components_raw, dict):
        components: dict[str, dict[str, Any]] = {}
        for key, value in components_raw.items():
            normalized = _normalize_component_name(key)
            if normalized is None or not isinstance(value, dict):
                continue
            components[normalized] = dict(value)
        payload = dict(raw)
        payload["version"] = str(payload.get("version") or SUBJECT_TUNING_VERSION)
        payload["components"] = components
        return payload

    tuned = raw.get("tuned_parameters")
    if not isinstance(tuned, dict):
        return {"version": SUBJECT_TUNING_VERSION, "components": {}}
    context = dict(raw.get("context")) if isinstance(raw.get("context"), dict) else {}
    return {
        "version": SUBJECT_TUNING_VERSION,
        "components": {
            "subject_body": {
                "method": str(raw.get("method") or "traditional_subject_mask_seed"),
                "version": str(raw.get("version") or "1.0"),
                "tuned_timestamp": raw.get("tuned_timestamp"),
                "tuned_parameters": dict(tuned),
                "context": context,
            }
        },
        "migrated_from_flat": True,
    }


def _build_eyes_union_entry(eye_tuning: dict[str, Any]) -> dict[str, Any]:
    tuned_parameters = eye_tuning.get("tuned_parameters")
    if not isinstance(tuned_parameters, dict):
        raise ValueError("analysis_metadata.attrs['eye_mask_tuning'] missing dict tuned_parameters.")
    context = dict(eye_tuning.get("context")) if isinstance(eye_tuning.get("context"), dict) else {}
    if "component_name" in context and "storage_component_name" not in context:
        context["storage_component_name"] = context.pop("component_name")
    context.setdefault("storage_component_name", "eyes_union")
    context.setdefault("migration_source_key", EYE_TUNING_KEY)
    return {
        "method": str(eye_tuning.get("method") or "global_threshold_otsu"),
        "subject_method_family": "eye_mask_threshold_lr_v1",
        "version": str(eye_tuning.get("version") or "1.0"),
        "tuned_timestamp": eye_tuning.get("tuned_timestamp") or _utc_now(),
        "tuned_parameters": dict(tuned_parameters),
        "context": context,
        "output_labels": ["eye_left", "eye_right"],
        "storage_component": "eyes_union",
        "migrated_from_eye_mask_tuning": True,
    }


def backfill_subject_mask_tuning(
    zarr_path: Path | str,
    *,
    overwrite: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r+" if apply else "r")
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return {"zarr_path": str(zarr_path), "status": "missing_analysis_metadata"}

    metadata = dict(analysis.attrs) if analysis.attrs else {}
    eye_tuning = metadata.get(EYE_TUNING_KEY)
    if not isinstance(eye_tuning, dict):
        return {"zarr_path": str(zarr_path), "status": "missing_eye_mask_tuning"}

    subject_payload = _normalize_subject_tuning_payload(metadata.get(TUNING_KEY))
    components = dict(subject_payload.get("components", {}))
    existing_entry = components.get("eyes_union")
    if isinstance(existing_entry, dict) and not overwrite:
        return {"zarr_path": str(zarr_path), "status": "exists", "component": "eyes_union"}

    eyes_union_entry = _build_eyes_union_entry(eye_tuning)
    summary = {
        "zarr_path": str(zarr_path),
        "status": "would_update" if not apply else "updated",
        "component": "eyes_union",
        "tuned_timestamp": eyes_union_entry.get("tuned_timestamp"),
    }
    if not apply:
        return summary

    components["eyes_union"] = eyes_union_entry
    subject_payload["version"] = SUBJECT_TUNING_VERSION
    subject_payload["components"] = components
    if overwrite or not subject_payload.get("latest_component"):
        subject_payload["latest_component"] = "eyes_union"
        subject_payload["latest_timestamp"] = eyes_union_entry.get("tuned_timestamp")
    subject_payload["migration_sources"] = sorted(
        {str(item) for item in list(subject_payload.get("migration_sources", [])) + [EYE_TUNING_KEY]}
    )

    metadata[TUNING_KEY] = subject_payload
    analysis.attrs.update(metadata)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or roots containing .zarr archives.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search roots for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["training", "analysis", "any"],
        default="training",
        help="Archive scope filter (default: training).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing subject eyes_union tuning entry.")
    parser.add_argument("--apply", action="store_true", help="Write the migrated subject_mask_tuning payload.")
    args = parser.parse_args(argv)

    matched = False
    exit_code = 0
    for zarr_path in _iter_zarr(list(args.paths), args.recursive):
        try:
            root = open_zarr_root(zarr_path, mode="r")
        except Exception as exc:
            print(f"ERROR {zarr_path}: failed to open archive ({exc})")
            exit_code = 1
            continue
        inferred_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "any" and inferred_use is not None and inferred_use != args.zarr_use:
            continue
        matched = True
        try:
            result = backfill_subject_mask_tuning(
                zarr_path,
                overwrite=bool(args.overwrite),
                apply=bool(args.apply),
            )
            print(f"{result['status'].upper()} {zarr_path}")
        except Exception as exc:
            print(f"ERROR {zarr_path}: {exc}")
            exit_code = 1
    if not matched:
        print("No matching zarr archives found.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

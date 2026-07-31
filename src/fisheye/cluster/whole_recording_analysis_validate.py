"""Independently validate exact outputs from a whole-recording analysis DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fisheye.cluster.lsf import write_json_snapshot
from fisheye.cluster.whole_recording_analysis import PLAN_SCHEMA
from fisheye.shared.composite_subject_mask import (
    COMPOSITE_SUBJECT_MASK_STORAGE_MODE,
    CompositeSubjectMaskArray,
)
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import is_run_complete_in_parent
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    validate_subject_mask_bundle_candidate,
)
from fisheye.utils.validate_refined_subject_mask_contract import (
    validate_refined_subject_mask_contract,
)

REPORT_SCHEMA = "palette.whole_recording_analysis_validation.v1"
RAW_LABELS = ("subject_body", "eyes_union", "swim_bladder")
REFINED_LABELS = ("subject_body", "eye_left", "eye_right", "swim_bladder")


def _labels(value: object) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _uniform_indices(total_rows: int, sample_rows: int) -> np.ndarray:
    if total_rows <= 0:
        return np.zeros((0,), dtype=np.int64)
    if total_rows <= sample_rows:
        return np.arange(total_rows, dtype=np.int64)
    return np.unique(np.linspace(0, total_rows - 1, num=sample_rows, dtype=np.int64))


def _read_rows(array: Any, row_indices: np.ndarray) -> np.ndarray:
    selection = row_indices.tolist()
    selection_key = (selection,) + (slice(None),) * (len(array.shape) - 1)
    oindex = getattr(array, "oindex", None)
    if oindex is not None:
        return np.asarray(oindex[selection_key])
    return np.asarray(array[selection_key])


def _require_complete_run(root: Any, parent_name: str, run_name: str) -> Any:
    parent = root.get(parent_name)
    if parent is None or run_name not in parent:
        raise RuntimeError(f"Missing planned run {parent_name}/{run_name}")
    run = parent[run_name]
    if not is_run_complete_in_parent(parent, run):
        raise RuntimeError(f"Planned run is incomplete: {parent_name}/{run_name}")
    return run


def _validate_raw_masks(
    root: Any, run: Any, *, run_name: str, sample_rows: int
) -> dict[str, Any]:
    labels = _labels(run.attrs.get("mask_labels"))
    if labels != RAW_LABELS:
        raise RuntimeError(
            f"Raw mask labels {labels!r} do not match expected {RAW_LABELS!r}."
        )
    probabilities = run.get("mask_probs_roi")
    if (
        probabilities is None
        and str(run.attrs.get("subject_mask_storage_mode") or "").strip()
        == COMPOSITE_SUBJECT_MASK_STORAGE_MODE
    ):
        probabilities = CompositeSubjectMaskArray.open(
            root,
            run,
            run_name=run_name,
            verify_identity=True,
        )
    if probabilities is None:
        raise RuntimeError("Raw run is missing mask_probs_roi.")
    shape = tuple(int(value) for value in probabilities.shape)
    if len(shape) != 4 or shape[0] <= 0 or shape[1] != len(RAW_LABELS):
        raise RuntimeError(f"Invalid raw probability shape: {shape!r}.")
    if np.dtype(probabilities.dtype) != np.dtype(np.uint8):
        raise RuntimeError(
            f"Raw probability dtype must be uint8, got {probabilities.dtype}."
        )
    indices = _uniform_indices(shape[0], sample_rows)
    sampled = _read_rows(probabilities, indices)
    component_max = sampled.reshape(sampled.shape[0], sampled.shape[1], -1).max(
        axis=(0, 2)
    )
    if np.any(component_max <= 1):
        raise RuntimeError(
            "Sampled encoded probabilities are empty or binary for components: "
            + ", ".join(
                label
                for label, maximum in zip(labels, component_max, strict=True)
                if int(maximum) <= 1
            )
        )
    metrics = run.get("metrics/mask_present")
    if metrics is None:
        raise RuntimeError("Raw run is missing metrics/mask_present.")
    present = np.asarray(metrics[:], dtype=bool)
    if present.shape != (shape[0], len(labels)):
        raise RuntimeError(
            "Raw metrics/mask_present shape does not match probabilities: "
            f"{present.shape!r} versus {(shape[0], len(labels))!r}."
        )
    present_counts = present.sum(axis=0, dtype=np.int64)
    if np.any(present_counts <= 0):
        raise RuntimeError("At least one raw component has no present masks.")
    return {
        "shape": list(shape),
        "dtype": str(np.dtype(probabilities.dtype)),
        "sample_indices": indices.tolist(),
        "sample_component_max": {
            label: int(value)
            for label, value in zip(labels, component_max, strict=True)
        },
        "mask_present_counts": {
            label: int(value)
            for label, value in zip(labels, present_counts, strict=True)
        },
    }


def _validate_refined_masks(run: Any, *, sample_rows: int) -> dict[str, Any]:
    labels = _labels(run.attrs.get("mask_labels"))
    if labels != REFINED_LABELS:
        raise RuntimeError(
            f"Refined mask labels {labels!r} do not match expected {REFINED_LABELS!r}."
        )
    masks = run.get("masks_roi")
    if masks is None:
        raise RuntimeError("Refined run is missing authoritative dense masks_roi.")
    shape = tuple(int(value) for value in masks.shape)
    if len(shape) != 4 or shape[0] <= 0 or shape[1] != len(REFINED_LABELS):
        raise RuntimeError(f"Invalid refined mask shape: {shape!r}.")
    if np.dtype(masks.dtype) != np.dtype(np.uint8):
        raise RuntimeError(f"Refined masks_roi dtype must be uint8, got {masks.dtype}.")
    indices = _uniform_indices(shape[0], sample_rows)
    sampled = _read_rows(masks, indices)
    unique_values = np.unique(sampled)
    if not set(int(value) for value in unique_values).issubset({0, 1}):
        raise RuntimeError(
            f"Refined masks_roi sample is not binary: {unique_values.tolist()!r}."
        )
    sampled_present = sampled.reshape(sampled.shape[0], sampled.shape[1], -1).any(
        axis=(0, 2)
    )
    if not np.all(sampled_present):
        raise RuntimeError(
            "Sampled refined masks are empty for components: "
            + ", ".join(
                label
                for label, is_present in zip(labels, sampled_present, strict=True)
                if not bool(is_present)
            )
        )
    metrics = run.get("metrics/mask_present")
    if metrics is None:
        raise RuntimeError("Refined run is missing metrics/mask_present.")
    present = np.asarray(metrics[:], dtype=bool)
    if present.shape != (shape[0], len(labels)):
        raise RuntimeError(
            "Refined metrics/mask_present shape does not match masks_roi: "
            f"{present.shape!r} versus {(shape[0], len(labels))!r}."
        )
    present_counts = present.sum(axis=0, dtype=np.int64)
    if np.any(present_counts <= 0):
        raise RuntimeError("At least one refined component has no present masks.")
    return {
        "shape": list(shape),
        "dtype": str(np.dtype(masks.dtype)),
        "sample_indices": indices.tolist(),
        "sample_unique_values": [int(value) for value in unique_values],
        "sample_component_nonzero": {
            label: bool(value)
            for label, value in zip(labels, sampled_present, strict=True)
        },
        "mask_present_counts": {
            label: int(value)
            for label, value in zip(labels, present_counts, strict=True)
        },
        "authoritative_surface": "masks_roi",
    }


def _validate_target(
    raw: Mapping[str, Any],
    *,
    sample_rows: int,
    open_root_fn: Callable[..., Any],
    contract_validator_fn: Callable[..., Mapping[str, Any]],
    bundle_validator_fn: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    zarr_path = Path(str(raw["analysis_zarr"])).expanduser().resolve()
    subject_names = raw.get("subject_masks")
    if not isinstance(subject_names, Mapping):
        raise ValueError("Combined target lacks subject_masks run names.")
    subject_run_name = str(subject_names["subject_mask_run"])
    refined_run_name = str(subject_names["refined_subject_mask_run"])
    quality_run_name = str(subject_names.get("subject_mask_quality_run") or "")
    bundle_id = str(subject_names.get("subject_mask_bundle_id") or "")
    keypoint_run_name = str(raw["keypoint_run"])
    refined_keypoint_run_name = str(raw["refined_keypoint_run"])
    root = open_root_fn(zarr_path, mode="r")
    _require_complete_run(root, "keypoints_runs", keypoint_run_name)
    _require_complete_run(
        root,
        "refined_keypoints_runs",
        refined_keypoint_run_name,
    )
    subject_run = _require_complete_run(root, "subject_mask_runs", subject_run_name)
    refined_run = _require_complete_run(
        root,
        "refined_subject_masks_runs",
        refined_run_name,
    )
    if quality_run_name:
        _require_complete_run(
            root,
            "subject_mask_quality_runs",
            quality_run_name,
        )
    actual_assignment = (
        normalize_attr(refined_run.attrs.get("assignment_keypoint_group")),
        normalize_attr(
            refined_run.attrs.get("assignment_keypoints_run")
            or refined_run.attrs.get("assignment_keypoint_run")
        ),
    )
    expected_assignment = ("refined_keypoints_runs", refined_keypoint_run_name)
    if actual_assignment != expected_assignment:
        raise RuntimeError(
            "Refined subject-mask assignment lineage mismatch: "
            f"{actual_assignment!r} versus {expected_assignment!r}."
        )
    bundle = None
    if bundle_id:
        bundle = dict(bundle_validator_fn(analysis_zarr=zarr_path, bundle_id=bundle_id))
        contract = {"valid": True, "source": "subject_mask_bundle_v1"}
    else:
        contract = dict(contract_validator_fn(zarr_path, run_name=refined_run_name))
        if not bool(contract.get("valid")):
            raise RuntimeError(
                "Refined subject-mask contract validation failed: "
                + json.dumps(contract.get("errors") or [], sort_keys=True)
            )
    return {
        "target_id": str(raw.get("target_id") or ""),
        "analysis_zarr": str(zarr_path),
        "keypoint_run": keypoint_run_name,
        "refined_keypoint_run": refined_keypoint_run_name,
        "subject_mask_run": subject_run_name,
        "refined_subject_mask_run": refined_run_name,
        "subject_mask_quality_run": quality_run_name or None,
        "subject_mask_bundle_id": bundle_id or None,
        "assignment_keypoint_group": actual_assignment[0],
        "assignment_keypoint_run": actual_assignment[1],
        "raw_masks": _validate_raw_masks(
            root,
            subject_run,
            run_name=subject_run_name,
            sample_rows=sample_rows,
        ),
        "refined_masks": _validate_refined_masks(
            refined_run,
            sample_rows=sample_rows,
        ),
        "refined_contract": contract,
        "subject_mask_bundle": bundle,
        "status": "ok",
    }


def validate_analysis_plan(
    plan_path: Path,
    *,
    sample_rows: int = 32,
    open_root_fn: Callable[..., Any] = open_zarr_group_direct,
    contract_validator_fn: Callable[..., Mapping[str, Any]] = (
        validate_refined_subject_mask_contract
    ),
    bundle_validator_fn: Callable[..., Mapping[str, Any]] = (
        validate_subject_mask_bundle_candidate
    ),
) -> dict[str, Any]:
    if int(sample_rows) <= 0:
        raise ValueError("sample_rows must be positive.")
    plan_path = plan_path.expanduser().resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported combined plan: {plan_path}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Combined plan has no targets.")
    reports: list[dict[str, Any]] = []
    for raw in targets:
        if not isinstance(raw, Mapping):
            raise ValueError("Combined plan target must be an object.")
        try:
            reports.append(
                _validate_target(
                    raw,
                    sample_rows=int(sample_rows),
                    open_root_fn=open_root_fn,
                    contract_validator_fn=contract_validator_fn,
                    bundle_validator_fn=bundle_validator_fn,
                )
            )
        except Exception as exc:
            reports.append(
                {
                    "target_id": str(raw.get("target_id") or ""),
                    "analysis_zarr": str(raw.get("analysis_zarr") or ""),
                    "status": "invalid",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    invalid = [report for report in reports if report["status"] != "ok"]
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not invalid else "invalid",
        "plan_path": str(plan_path),
        "sample_rows_requested": int(sample_rows),
        "target_count": len(reports),
        "ok_count": len(reports) - len(invalid),
        "invalid_count": len(invalid),
        "targets": reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_path", type=Path)
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    report = validate_analysis_plan(
        args.plan_path,
        sample_rows=int(args.sample_rows),
    )
    if args.output_json is not None:
        write_json_snapshot(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_SCHEMA",
    "main",
    "validate_analysis_plan",
]

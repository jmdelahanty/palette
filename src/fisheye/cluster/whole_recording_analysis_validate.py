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
from fisheye.shared.zarr.subject_mask_schema import (
    SUBJECT_V1_LR_COMPONENT_SCHEMA,
    SUBJECT_V1_UNION_COMPONENT_SCHEMA,
    SubjectMaskComponentSchema,
    resolve_subject_mask_component_schema,
)
from fisheye.utils.validate_refined_subject_mask_contract import (
    validate_refined_subject_mask_contract,
)

REPORT_SCHEMA = "palette.whole_recording_analysis_validation.v2"
RAW_LABELS = SUBJECT_V1_UNION_COMPONENT_SCHEMA.labels
REFINED_LABELS = SUBJECT_V1_LR_COMPONENT_SCHEMA.labels


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


def _component_presence_report(
    labels: tuple[str, ...],
    present: np.ndarray,
    *,
    required_labels: tuple[str, ...],
) -> dict[str, Any]:
    required = set(required_labels)
    unknown_required = required.difference(labels)
    if unknown_required:
        raise RuntimeError(
            "Component presence policy references unknown labels: "
            + ", ".join(sorted(unknown_required))
        )
    values = np.asarray(present, dtype=bool)
    if values.shape != (len(labels),):
        raise RuntimeError(
            "Component presence summary has invalid shape: "
            f"{values.shape!r} versus {(len(labels),)!r}."
        )
    missing_required = [
        label
        for label, is_present in zip(labels, values, strict=True)
        if label in required and not bool(is_present)
    ]
    absent_optional = [
        label
        for label, is_present in zip(labels, values, strict=True)
        if label not in required and not bool(is_present)
    ]
    return {
        "status": "failed" if missing_required else "passed",
        "required_components": [label for label in labels if label in required],
        "optional_components": [label for label in labels if label not in required],
        "missing_required_components": missing_required,
        "absent_optional_components": absent_optional,
    }


def _mapping_attr(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _component_presence_contract(
    run: Any,
) -> tuple[tuple[str, ...], SubjectMaskComponentSchema, dict[str, Any]]:
    labels = _labels(run.attrs.get("mask_labels"))
    if not labels:
        raise RuntimeError("Subject-mask run is missing mask_labels.")

    component_registry = _mapping_attr(run.attrs.get("component_registry"))
    if component_registry is not None:
        registry_labels = _labels(component_registry.get("labels"))
        if registry_labels != labels:
            raise RuntimeError(
                "Subject-mask component_registry labels do not match mask_labels: "
                f"{registry_labels!r} versus {labels!r}."
            )

    logical_schema = _mapping_attr(run.attrs.get("logical_schema"))
    logical_components = (
        _mapping_attr(logical_schema.get("components"))
        if logical_schema is not None
        else None
    )
    if logical_components is not None:
        logical_labels = _labels(logical_components.get("labels"))
        if logical_labels != labels:
            raise RuntimeError(
                "Subject-mask logical-schema components do not match mask_labels: "
                f"{logical_labels!r} versus {labels!r}."
            )

    declared_schema_id = str(
        normalize_attr(run.attrs.get("label_schema_id")) or ""
    ).strip()
    try:
        schema = resolve_subject_mask_component_schema(
            schema_id=declared_schema_id or None,
            labels=labels,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    available_array = run.get("available_channels")
    if available_array is None:
        raise RuntimeError("Subject-mask run is missing available_channels.")
    available = np.asarray(available_array[:])
    if available.dtype != np.dtype(bool) or available.shape != (len(labels),):
        raise RuntimeError(
            "Subject-mask available_channels must be bool with shape "
            f"{(len(labels),)!r}; got {available.dtype} {available.shape!r}."
        )
    unavailable_required = [
        label
        for label, is_available in zip(labels, available, strict=True)
        if label in schema.required_labels and not bool(is_available)
    ]
    if unavailable_required:
        raise RuntimeError(
            "Required subject-mask schema components are marked unavailable: "
            + ", ".join(unavailable_required)
        )

    schema_basis = "label_schema_id" if declared_schema_id else "exact_component_labels"
    return (
        labels,
        schema,
        {
            **schema.as_manifest(),
            "resolution_basis": schema_basis,
            "available_channels": {
                label: bool(value)
                for label, value in zip(labels, available, strict=True)
            },
            "component_registry_present": component_registry is not None,
            "logical_schema_components_present": logical_components is not None,
        },
    )


def _component_completeness_report(
    component_presence: Mapping[str, Any],
    sample_presence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failures = [
        {
            "code": "required_component_has_no_present_masks",
            "scope": "all_rows",
            "component": str(component),
        }
        for component in component_presence.get("missing_required_components", ())
    ]
    sample_observations = [
        {
            "code": "required_component_not_observed_in_uniform_sample",
            "scope": "uniform_sample",
            "component": str(component),
        }
        for component in (
            sample_presence.get("missing_required_components", ())
            if sample_presence is not None
            else ()
        )
    ]
    return {
        "status": "failed" if failures else "passed",
        "publication_blocking": False,
        "failure_count": len(failures),
        "failures": failures,
        "sample_observation_count": len(sample_observations),
        "sample_observations": sample_observations,
    }


def _aggregate_component_completeness(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    sample_observations: list[dict[str, Any]] = []
    for entry in entries:
        stage = str(entry.get("stage") or "")
        clip_id = str(entry.get("clip_id") or "") or None
        report = entry.get("report")
        if not isinstance(report, Mapping):
            raise TypeError("Component completeness entry is missing its report.")
        completeness = report.get("component_completeness")
        if not isinstance(completeness, Mapping):
            raise TypeError("Mask report is missing component_completeness.")
        context = {"stage": stage}
        if clip_id is not None:
            context["clip_id"] = clip_id
        for finding in completeness.get("failures", ()):
            failures.append({**context, **dict(finding)})
        for finding in completeness.get("sample_observations", ()):
            sample_observations.append({**context, **dict(finding)})
    return {
        "status": "failed" if failures else "passed",
        "publication_blocking": False,
        "failure_count": len(failures),
        "failures": failures,
        "sample_observation_count": len(sample_observations),
        "sample_observations": sample_observations,
    }


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
    labels, component_schema, schema_report = _component_presence_contract(run)
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
    if len(shape) != 4 or shape[0] <= 0 or shape[1] != len(labels):
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
    sample_presence = _component_presence_report(
        labels,
        component_max > 1,
        required_labels=component_schema.required_labels,
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
    component_presence = _component_presence_report(
        labels,
        present_counts > 0,
        required_labels=component_schema.required_labels,
    )
    return {
        "component_schema": schema_report,
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
        "component_presence_policy": component_presence,
        "sample_component_presence_policy": sample_presence,
        "component_completeness": _component_completeness_report(
            component_presence,
            sample_presence,
        ),
    }


def _validate_refined_masks(run: Any, *, sample_rows: int) -> dict[str, Any]:
    labels, component_schema, schema_report = _component_presence_contract(run)
    masks = run.get("masks_roi")
    if masks is None:
        raise RuntimeError("Refined run is missing authoritative dense masks_roi.")
    shape = tuple(int(value) for value in masks.shape)
    if len(shape) != 4 or shape[0] <= 0 or shape[1] != len(labels):
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
    sample_presence = _component_presence_report(
        labels,
        sampled_present,
        required_labels=component_schema.required_labels,
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
    component_presence = _component_presence_report(
        labels,
        present_counts > 0,
        required_labels=component_schema.required_labels,
    )
    return {
        "component_schema": schema_report,
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
        "component_presence_policy": component_presence,
        "sample_component_presence_policy": sample_presence,
        "component_completeness": _component_completeness_report(
            component_presence,
            sample_presence,
        ),
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
    raw_report = _validate_raw_masks(
        root,
        subject_run,
        run_name=subject_run_name,
        sample_rows=sample_rows,
    )
    refined_report = _validate_refined_masks(
        refined_run,
        sample_rows=sample_rows,
    )
    component_completeness = _aggregate_component_completeness(
        (
            {"stage": "raw_subject_masks", "report": raw_report},
            {"stage": "refined_subject_masks", "report": refined_report},
        )
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
        "raw_masks": raw_report,
        "refined_masks": refined_report,
        "subject_mask_component_completeness": component_completeness,
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
    completeness_failed = [
        report
        for report in reports
        if report.get("status") == "ok"
        and isinstance(report.get("subject_mask_component_completeness"), Mapping)
        and report["subject_mask_component_completeness"].get("status") == "failed"
    ]
    return {
        "schema": REPORT_SCHEMA,
        "status": "ok" if not invalid else "invalid",
        "plan_path": str(plan_path),
        "sample_rows_requested": int(sample_rows),
        "target_count": len(reports),
        "ok_count": len(reports) - len(invalid),
        "invalid_count": len(invalid),
        "component_completeness_failed_target_count": len(completeness_failed),
        "component_completeness_status": (
            "failed" if completeness_failed else "passed"
        ),
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

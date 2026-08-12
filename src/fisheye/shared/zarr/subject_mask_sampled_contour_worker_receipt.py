"""Receipts for worker-produced sampled subject-mask contours.

Refinement workers already compute fixed-count sampled contours while their
final dense masks are local.  This module seals those derived rows against the
worker's dense-mask receipt so recording publication can assemble the cache
without decoding the dense authority a second time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fisheye.shared.refined_subject_component_contours import (
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    SAMPLED_COMPONENT_CONTOUR_CANONICALIZATION,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_ID,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_VERSION,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_subject_mask_extensions import (
    SubjectMaskSampledContourProfile,
    default_subject_mask_sampled_contour_profile,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
    subject_mask_array_unit_document,
)

SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask.sampled_contour_worker_receipt"
)
SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_VERSION = 1
SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_ID = (
    "palette.subject_mask.sampled_contour_worker_assembly"
)
SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_VERSION = 2
DEFAULT_RECEIPT_UNIT_ROWS = 1024
_FIELDS = ("points_xy", "valid", "source_point_count")


def _strict_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be strict canonical JSON: {exc}.") from exc


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_run_path(value: Any) -> str:
    path = str(value).strip().strip("/")
    parts = Path(path).parts
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Sampled-contour worker source_run_path is unsafe.")
    return path


def _component_registry(run: Any) -> SubjectMaskComponentRegistry:
    labels = run.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)):
        raise ValueError("Refined worker lacks its component registry.")
    return SubjectMaskComponentRegistry(tuple(str(value) for value in labels))


def sampled_contour_worker_arrays(
    run: Any,
    *,
    components: SubjectMaskComponentRegistry,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for component in components.labels:
        prefix = f"components/{component}/sampled_contours"
        if prefix not in run:
            raise ValueError(
                f"Refined worker lacks sampled contours for {component!r}."
            )
        group = run[prefix]
        for field in _FIELDS:
            path = f"{prefix}/{field}"
            if field not in group:
                raise ValueError(f"Refined worker lacks sampled-contour {path!r}.")
            arrays[path] = group[field]
    return arrays


def _profile_group_errors(
    run: Any,
    *,
    components: SubjectMaskComponentRegistry,
    profile: SubjectMaskSampledContourProfile,
) -> list[str]:
    errors: list[str] = []
    expected_common = {
        "schema_id": "sampled_component_contours_v1",
        "contour_schema_id": "sampled_component_contours_v1",
        "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
        "point_order": "xy",
        "source_contour_method": DEFAULT_CONTOUR_METHOD,
        "source_contour_method_version": DEFAULT_CONTOUR_METHOD_VERSION,
        "sampling_method": "closed_arc_length_uniform",
        "sampling_method_version": 2,
        "publication_profile_id": SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_ID,
        "publication_profile_version": (
            SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_VERSION
        ),
        "point_canonicalization": SAMPLED_COMPONENT_CONTOUR_CANONICALIZATION,
        "winding": "clockwise_in_roi_y_down",
        "start_point": "topmost_then_leftmost_vertex",
        "duplicate_closing_point": False,
        "boundary_policy": DEFAULT_BOUNDARY_POLICY,
        "min_source_points": 2,
    }
    for component in components.labels:
        component_group = run[f"components/{component}"]
        if "contours" in component_group:
            errors.append(
                f"full ragged contours are forbidden by the default profile: {component}"
            )
        sampled = component_group.get("sampled_contours")
        if sampled is None:
            errors.append(f"sampled contours are absent for {component}")
            continue
        for name, expected in expected_common.items():
            if sampled.attrs.get(name) != expected:
                errors.append(
                    f"sampled-contour attribute {name!r} differs for {component}"
                )
        if sampled.attrs.get("source_component") != component:
            errors.append(f"sampled-contour source component differs for {component}")
        if sampled.attrs.get("sample_count") != profile.sample_counts[component]:
            errors.append(f"sampled-contour count differs for {component}")
    return errors


def _array_contract_errors(
    arrays: Mapping[str, Any],
    *,
    row_count: int,
    components: SubjectMaskComponentRegistry,
    profile: SubjectMaskSampledContourProfile,
) -> list[str]:
    errors: list[str] = []
    for component in components.labels:
        prefix = f"components/{component}/sampled_contours"
        sample_count = profile.sample_counts[component]
        expected = {
            f"{prefix}/points_xy": ((row_count, sample_count, 2), np.dtype("float32")),
            f"{prefix}/valid": ((row_count,), np.dtype("bool")),
            f"{prefix}/source_point_count": ((row_count,), np.dtype("int32")),
        }
        for path, (shape, dtype) in expected.items():
            value = arrays.get(path)
            if value is None:
                errors.append(f"sampled-contour array is absent at {path}")
                continue
            if tuple(int(item) for item in value.shape) != shape:
                errors.append(f"sampled-contour shape differs at {path}")
            if np.dtype(value.dtype) != dtype:
                errors.append(f"sampled-contour dtype differs at {path}")
    return errors


def _semantic_errors(
    arrays: Mapping[str, Any],
    *,
    components: SubjectMaskComponentRegistry,
) -> list[str]:
    errors: list[str] = []
    for component in components.labels:
        prefix = f"components/{component}/sampled_contours"
        points = np.asarray(arrays[f"{prefix}/points_xy"][:], dtype=np.float32)
        valid = np.asarray(arrays[f"{prefix}/valid"][:], dtype=bool)
        counts = np.asarray(arrays[f"{prefix}/source_point_count"][:], dtype=np.int32)
        finite = np.isfinite(points).all(axis=(1, 2))
        all_nan = np.isnan(points).all(axis=(1, 2))
        if bool(np.any(valid & ~finite)):
            errors.append(
                f"valid sampled contours contain non-finite points: {component}"
            )
        if bool(np.any(~valid & ~all_nan)):
            errors.append(
                f"invalid sampled contours do not use all-NaN encoding: {component}"
            )
        if bool(np.any(counts < 0)) or bool(np.any(valid & (counts < 2))):
            errors.append(f"sampled-contour source counts differ: {component}")
    return errors


def _worker_dense_binding(
    worker_receipt: Mapping[str, Any],
    *,
    run_path: str,
    row_count: int,
) -> dict[str, object]:
    receipt = _strict_copy(worker_receipt, name="worker semantic receipt")
    payload = receipt.get("payload") if isinstance(receipt, Mapping) else None
    arrays = payload.get("arrays") if isinstance(payload, Mapping) else None
    dense = arrays.get("masks_roi") if isinstance(arrays, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not _is_sha256(receipt.get("payload_digest"))
        or not isinstance(payload, Mapping)
        or receipt.get("payload_digest") != canonical_json_sha256(payload)
        or payload.get("stage_kind") != "refined_subject_mask"
        or payload.get("run_path") != run_path
        or not isinstance(dense, Mapping)
        or dense.get("shape", [None])[0] != row_count
        or dense.get("dtype") != "uint8"
        or dense.get("digest_algorithm") != SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM
        or not _is_sha256(dense.get("units_digest"))
    ):
        raise ValueError("Sampled contours do not bind a valid refined worker receipt.")
    return {
        "worker_receipt_payload_digest": receipt["payload_digest"],
        "dense_array_path": "masks_roi",
        "dense_array_shape": list(dense["shape"]),
        "dense_array_dtype": dense["dtype"],
        "dense_array_units_digest": dense["units_digest"],
    }


def build_subject_mask_sampled_contour_worker_receipt(
    run: Any,
    *,
    global_start_row: int,
    worker_receipt: Mapping[str, Any],
    producer_commit: str,
    unit_rows: int = DEFAULT_RECEIPT_UNIT_ROWS,
) -> dict[str, object]:
    """Seal one worker's existing sampled contours against its dense receipt."""

    start = int(global_start_row)
    if type(global_start_row) is not int or start < 0:
        raise ValueError("global_start_row must be a nonnegative exact integer.")
    commit = str(producer_commit).strip()
    if not commit:
        raise ValueError("producer_commit must be nonempty.")
    rows = int(run["masks_roi"].shape[0])
    if rows <= 0:
        raise ValueError("Sampled-contour worker receipts require nonempty rows.")
    run_path = _safe_run_path(run.path)
    components = _component_registry(run)
    profile = default_subject_mask_sampled_contour_profile(components)
    arrays = sampled_contour_worker_arrays(run, components=components)
    errors = [
        *_profile_group_errors(run, components=components, profile=profile),
        *_array_contract_errors(
            arrays, row_count=rows, components=components, profile=profile
        ),
        *_semantic_errors(arrays, components=components),
    ]
    if run.attrs.get("sampled_component_contours_status") != "computed":
        errors.append("sampled-contour worker status is not computed")
    if run.attrs.get("derived_mask_caches_stale") is not False:
        errors.append("sampled-contour worker derived caches are stale")
    if run.attrs.get("contours_stale") is not False:
        errors.append("sampled-contour worker contour cache is stale")
    if errors:
        raise ValueError("; ".join(errors))
    dense_binding = _worker_dense_binding(
        worker_receipt, run_path=run_path, row_count=rows
    )
    array_document = subject_mask_array_unit_document(
        arrays, tuple(sorted(arrays)), unit_rows=int(unit_rows)
    )
    payload = {
        "kind": "fixed_count_sampled_contours",
        "source_run_path": run_path,
        "global_row_interval": {"start_row": start, "stop_row": start + rows},
        "local_row_count": rows,
        "component_registry": components.as_manifest(),
        "contour_profile": profile.as_manifest(components=components),
        "source_dense_worker": dense_binding,
        "arrays": array_document,
        "array_document_digest": canonical_json_sha256(array_document),
        "freshness": {
            "sampled_component_contours_status": "computed",
            "derived_mask_caches_stale": False,
            "contours_stale": False,
        },
        "full_contours_policy": "forbidden_default_profile",
        "producer_commit": commit,
    }
    receipt = {
        "schema_id": SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    validate_subject_mask_sampled_contour_worker_receipt(
        receipt, run=run, worker_receipt=worker_receipt, verify_values=False
    )
    return receipt


def write_subject_mask_sampled_contour_worker_receipt(
    run: Any,
    *,
    destination: Path,
    global_start_row: int,
    worker_receipt: Mapping[str, Any],
    producer_commit: str,
    unit_rows: int = DEFAULT_RECEIPT_UNIT_ROWS,
) -> dict[str, object]:
    receipt = build_subject_mask_sampled_contour_worker_receipt(
        run,
        global_start_row=global_start_row,
        worker_receipt=worker_receipt,
        producer_commit=producer_commit,
        unit_rows=unit_rows,
    )
    target = destination.expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(receipt) + b"\n")
    os.replace(temporary, target)
    return receipt


def load_subject_mask_sampled_contour_worker_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Sampled-contour worker receipt must be one JSON object.")
    return value


def validate_subject_mask_sampled_contour_worker_receipt(
    receipt: Mapping[str, Any],
    *,
    run: Any | None = None,
    worker_receipt: Mapping[str, Any] | None = None,
    verify_values: bool = True,
) -> dict[str, object]:
    canonical = _strict_copy(receipt, name="sampled-contour worker receipt")
    if not isinstance(canonical, dict) or set(canonical) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        raise ValueError("Sampled-contour worker receipt fields are not exact.")
    payload = canonical.get("payload")
    if (
        canonical.get("schema_id")
        != SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_ID
        or canonical.get("schema_version")
        != SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_VERSION
        or canonical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, dict)
        or canonical.get("payload_digest") != canonical_json_sha256(payload)
    ):
        raise ValueError("Sampled-contour worker receipt envelope differs.")
    expected_fields = {
        "kind",
        "source_run_path",
        "global_row_interval",
        "local_row_count",
        "component_registry",
        "contour_profile",
        "source_dense_worker",
        "arrays",
        "array_document_digest",
        "freshness",
        "full_contours_policy",
        "producer_commit",
    }
    if set(payload) != expected_fields:
        raise ValueError("Sampled-contour worker payload fields are not exact.")
    interval = payload.get("global_row_interval")
    rows = payload.get("local_row_count")
    if (
        payload.get("kind") != "fixed_count_sampled_contours"
        or not isinstance(interval, dict)
        or set(interval) != {"start_row", "stop_row"}
        or type(rows) is not int
        or rows <= 0
        or type(interval.get("start_row")) is not int
        or type(interval.get("stop_row")) is not int
        or interval["start_row"] < 0
        or interval["stop_row"] - interval["start_row"] != rows
        or payload.get("full_contours_policy") != "forbidden_default_profile"
        or payload.get("freshness")
        != {
            "sampled_component_contours_status": "computed",
            "derived_mask_caches_stale": False,
            "contours_stale": False,
        }
        or not str(payload.get("producer_commit") or "").strip()
    ):
        raise ValueError("Sampled-contour worker identity or freshness differs.")
    run_path = _safe_run_path(payload.get("source_run_path"))
    registry = payload.get("component_registry")
    labels = registry.get("labels") if isinstance(registry, Mapping) else None
    if not isinstance(labels, list):
        raise ValueError("Sampled-contour worker component registry is absent.")
    components = SubjectMaskComponentRegistry(tuple(str(value) for value in labels))
    if registry != components.as_manifest():
        raise ValueError("Sampled-contour worker component registry is not canonical.")
    profile = default_subject_mask_sampled_contour_profile(components)
    if payload.get("contour_profile") != profile.as_manifest(components=components):
        raise ValueError("Sampled-contour worker profile differs from the default.")
    arrays_document = payload.get("arrays")
    if not isinstance(arrays_document, Mapping) or payload.get(
        "array_document_digest"
    ) != canonical_json_sha256(arrays_document):
        raise ValueError("Sampled-contour worker array digest differs.")
    expected_paths = {
        f"components/{component}/sampled_contours/{field}"
        for component in components.labels
        for field in _FIELDS
    }
    if set(arrays_document) != expected_paths:
        raise ValueError("Sampled-contour worker array inventory differs.")
    for path, record in arrays_document.items():
        component = path.split("/")[1]
        field = path.rsplit("/", 1)[-1]
        sample_count = profile.sample_counts[component]
        expected_shape = {
            "points_xy": [rows, sample_count, 2],
            "valid": [rows],
            "source_point_count": [rows],
        }[field]
        expected_dtype = {
            "points_xy": "float32",
            "valid": "bool",
            "source_point_count": "int32",
        }[field]
        units = record.get("units") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "shape",
                "dtype",
                "digest_algorithm",
                "unit_count",
                "units_digest",
                "units",
            }
            or record.get("digest_algorithm")
            != SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM
            or record.get("shape") != expected_shape
            or record.get("dtype") != expected_dtype
            or not _is_sha256(record.get("units_digest"))
            or not isinstance(units, list)
            or not units
            or record.get("unit_count") != len(units)
            or record.get("units_digest") != canonical_json_sha256(units)
        ):
            raise ValueError(f"Sampled-contour worker array record differs at {path}.")
        cursor = 0
        for unit in units:
            if (
                not isinstance(unit, Mapping)
                or set(unit) != {"start_row", "stop_row", "decoded_bytes", "sha256"}
                or type(unit.get("start_row")) is not int
                or type(unit.get("stop_row")) is not int
                or unit["start_row"] != cursor
                or not (cursor < unit["stop_row"] <= rows)
                or type(unit.get("decoded_bytes")) is not int
                or unit["decoded_bytes"] <= 0
                or not _is_sha256(unit.get("sha256"))
            ):
                raise ValueError(
                    f"Sampled-contour worker array units differ at {path}."
                )
            cursor = unit["stop_row"]
        if cursor != rows:
            raise ValueError(
                f"Sampled-contour worker array coverage is incomplete at {path}."
            )
    dense_binding = payload.get("source_dense_worker")
    if (
        not isinstance(dense_binding, Mapping)
        or set(dense_binding)
        != {
            "worker_receipt_payload_digest",
            "dense_array_path",
            "dense_array_shape",
            "dense_array_dtype",
            "dense_array_units_digest",
        }
        or dense_binding.get("dense_array_path") != "masks_roi"
        or dense_binding.get("dense_array_shape", [None])[0] != rows
        or dense_binding.get("dense_array_dtype") != "uint8"
        or not _is_sha256(dense_binding.get("worker_receipt_payload_digest"))
        or not _is_sha256(dense_binding.get("dense_array_units_digest"))
    ):
        raise ValueError("Sampled-contour dense-worker binding differs.")
    if worker_receipt is not None:
        expected_dense = _worker_dense_binding(
            worker_receipt, run_path=run_path, row_count=rows
        )
        if dense_binding != expected_dense:
            raise ValueError("Sampled-contour worker receipt binds another dense run.")
    if run is not None:
        if (
            _safe_run_path(run.path) != run_path
            or int(run["masks_roi"].shape[0]) != rows
        ):
            raise ValueError("Sampled-contour receipt belongs to another worker run.")
        if _component_registry(run) != components:
            raise ValueError("Sampled-contour worker components differ.")
        arrays = sampled_contour_worker_arrays(run, components=components)
        errors = [
            *_profile_group_errors(run, components=components, profile=profile),
            *_array_contract_errors(
                arrays, row_count=rows, components=components, profile=profile
            ),
        ]
        if verify_values:
            observed = subject_mask_array_unit_document(
                arrays,
                tuple(sorted(arrays)),
                unit_rows=max(
                    1,
                    int(
                        arrays_document[next(iter(sorted(arrays_document)))]["units"][
                            0
                        ]["stop_row"]
                    ),
                ),
            )
            if observed != arrays_document:
                errors.append("sampled-contour worker logical values differ")
            errors.extend(_semantic_errors(arrays, components=components))
        if run.attrs.get("sampled_component_contours_status") != "computed":
            errors.append("sampled-contour worker status is not computed")
        if run.attrs.get("derived_mask_caches_stale") is not False:
            errors.append("sampled-contour worker derived caches are stale")
        if run.attrs.get("contours_stale") is not False:
            errors.append("sampled-contour worker contour cache is stale")
        if errors:
            raise ValueError("; ".join(errors))
    return canonical


def build_subject_mask_sampled_contour_worker_assembly(
    receipts: Sequence[Mapping[str, Any]],
    *,
    source_producer_evidence: Mapping[str, Any],
    n_rois: int,
    components: SubjectMaskComponentRegistry,
    producer_commit: str,
) -> dict[str, object]:
    """Validate complete worker coverage and return compact manifest evidence."""

    validated = [
        validate_subject_mask_sampled_contour_worker_receipt(
            receipt, verify_values=False
        )
        for receipt in receipts
    ]
    ordered = sorted(
        validated, key=lambda item: item["payload"]["global_row_interval"]["start_row"]
    )
    workers = source_producer_evidence.get("workers")
    if not isinstance(workers, list) or len(workers) != len(ordered):
        raise ValueError("Sampled-contour receipts do not cover the refined workers.")
    source_by_path = {
        str(worker.get("run_path")): worker
        for worker in workers
        if isinstance(worker, Mapping)
    }
    cursor = 0
    intervals: list[dict[str, int]] = []
    payload_digests: list[str] = []
    dense_receipt_digests: list[str] = []
    profile = default_subject_mask_sampled_contour_profile(components).as_manifest(
        components=components
    )
    for receipt in ordered:
        payload = receipt["payload"]
        interval = payload["global_row_interval"]
        if interval["start_row"] != cursor or interval["stop_row"] <= cursor:
            raise ValueError(
                "Sampled-contour worker intervals must be contiguous and ordered."
            )
        if payload["component_registry"] != components.as_manifest():
            raise ValueError("Sampled-contour worker component registries differ.")
        if payload["contour_profile"] != profile:
            raise ValueError("Sampled-contour worker profiles differ.")
        if payload["producer_commit"] != producer_commit:
            raise ValueError("Sampled-contour worker producer commits differ.")
        source = source_by_path.get(payload["source_run_path"])
        dense_digest = payload["source_dense_worker"]["worker_receipt_payload_digest"]
        if (
            not isinstance(source, Mapping)
            or source.get("global_row_interval") != interval
            or source.get("worker_receipt_payload_digest") != dense_digest
        ):
            raise ValueError(
                "Sampled-contour worker does not bind the recording source receipt."
            )
        intervals.append(dict(interval))
        payload_digests.append(receipt["payload_digest"])
        dense_receipt_digests.append(dense_digest)
        cursor = interval["stop_row"]
    if cursor != int(n_rois):
        raise ValueError("Sampled-contour workers do not cover every recording row.")
    document = {
        "schema_id": SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_VERSION,
        "worker_count": len(ordered),
        "global_row_count": int(n_rois),
        "complete_contiguous_row_coverage": True,
        "component_registry_digest": canonical_json_sha256(components.as_manifest()),
        "contour_profile_digest": canonical_json_sha256(profile),
        "worker_intervals": intervals,
        "worker_receipt_payload_digests": payload_digests,
        "worker_receipts_digest": canonical_json_sha256(payload_digests),
        "dense_worker_receipt_payload_digests": dense_receipt_digests,
        "dense_worker_receipts_digest": canonical_json_sha256(dense_receipt_digests),
        "source_producer_evidence_digest": canonical_json_sha256(
            source_producer_evidence
        ),
        "full_contours_policy": "forbidden_default_profile",
        "producer_commit": str(producer_commit),
    }
    return _strict_copy(document, name="sampled-contour worker assembly")


def validate_subject_mask_sampled_contour_worker_assembly(
    value: Mapping[str, Any],
    *,
    n_rois: int,
    components: SubjectMaskComponentRegistry,
) -> dict[str, object]:
    document = _strict_copy(value, name="sampled-contour worker assembly")
    expected_fields = {
        "schema_id",
        "schema_version",
        "worker_count",
        "global_row_count",
        "complete_contiguous_row_coverage",
        "component_registry_digest",
        "contour_profile_digest",
        "worker_intervals",
        "worker_receipt_payload_digests",
        "worker_receipts_digest",
        "dense_worker_receipt_payload_digests",
        "dense_worker_receipts_digest",
        "source_producer_evidence_digest",
        "full_contours_policy",
        "producer_commit",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError("Sampled-contour worker assembly fields are not exact.")
    intervals = document.get("worker_intervals")
    receipt_digests = document.get("worker_receipt_payload_digests")
    dense_digests = document.get("dense_worker_receipt_payload_digests")
    count = document.get("worker_count")
    if (
        document.get("schema_id")
        != SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_ID
        or document.get("schema_version")
        != SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_VERSION
        or type(count) is not int
        or count <= 0
        or document.get("global_row_count") != int(n_rois)
        or document.get("complete_contiguous_row_coverage") is not True
        or not isinstance(intervals, list)
        or not isinstance(receipt_digests, list)
        or not isinstance(dense_digests, list)
        or len(intervals) != count
        or len(receipt_digests) != count
        or len(dense_digests) != count
        or not all(_is_sha256(value) for value in receipt_digests)
        or not all(_is_sha256(value) for value in dense_digests)
        or document.get("worker_receipts_digest")
        != canonical_json_sha256(receipt_digests)
        or document.get("dense_worker_receipts_digest")
        != canonical_json_sha256(dense_digests)
        or not _is_sha256(document.get("source_producer_evidence_digest"))
        or document.get("component_registry_digest")
        != canonical_json_sha256(components.as_manifest())
        or document.get("contour_profile_digest")
        != canonical_json_sha256(
            default_subject_mask_sampled_contour_profile(components).as_manifest(
                components=components
            )
        )
        or document.get("full_contours_policy") != "forbidden_default_profile"
        or not str(document.get("producer_commit") or "").strip()
    ):
        raise ValueError("Sampled-contour worker assembly declaration differs.")
    cursor = 0
    for interval in intervals:
        if (
            not isinstance(interval, dict)
            or set(interval) != {"start_row", "stop_row"}
            or type(interval.get("start_row")) is not int
            or type(interval.get("stop_row")) is not int
            or interval["start_row"] != cursor
            or interval["stop_row"] <= cursor
            or interval["stop_row"] > int(n_rois)
        ):
            raise ValueError("Sampled-contour worker assembly coverage differs.")
        cursor = interval["stop_row"]
    if cursor != int(n_rois):
        raise ValueError("Sampled-contour worker assembly coverage is incomplete.")
    return document


__all__ = [
    "DEFAULT_RECEIPT_UNIT_ROWS",
    "SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_ID",
    "SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_ASSEMBLY_SCHEMA_VERSION",
    "SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_SAMPLED_CONTOUR_WORKER_RECEIPT_SCHEMA_VERSION",
    "build_subject_mask_sampled_contour_worker_assembly",
    "build_subject_mask_sampled_contour_worker_receipt",
    "load_subject_mask_sampled_contour_worker_receipt",
    "sampled_contour_worker_arrays",
    "validate_subject_mask_sampled_contour_worker_assembly",
    "validate_subject_mask_sampled_contour_worker_receipt",
    "write_subject_mask_sampled_contour_worker_receipt",
]

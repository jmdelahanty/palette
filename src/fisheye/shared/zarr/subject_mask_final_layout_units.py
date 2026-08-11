"""Immutable reusable final-layout units for subject-mask core publication.

Clip workers may prepare complete outer-shard row bands using the exact
recording-level storage plan.  The recording finalizer validates the sealed
packages, copies their already encoded objects, and recomputes only row bands
that cross worker boundaries.  Packages are transport artifacts: they never
own run metadata, completion state, consolidated metadata, or selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import (
    array_metadata_declaration_from_plan,
    create_array_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_file, utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskDimensions,
    SubjectMaskProbabilityEncoding,
)
from fisheye.shared.zarr.subject_mask_storage import (
    plan_raw_subject_mask_storage,
    plan_refined_subject_mask_publication_storage,
)

SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_ID = (
    "palette.subject_mask.final_layout_unit_package"
)
SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION = 1
SUBJECT_MASK_FINAL_LAYOUT_UNIT_RECEIPT = "receipt.json"
SUBJECT_MASK_FINAL_LAYOUT_UNIT_ARRAY = "payload.zarr/payload"
_ARRAY_ATTRIBUTES = {
    "benchmark_only": True,
    "selector_eligible": False,
    "artifact_class": "subject_mask_scientific_core",
}
_FICLONE = 0x40049409


@dataclass(frozen=True)
class EncodedObject:
    source_path: Path
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class EncodedPhysicalUnit:
    package_path: Path
    start_row: int
    stop_row: int
    logical_sha256: str
    objects: tuple[EncodedObject, ...]


@dataclass(frozen=True)
class FinalLayoutUnitAdoption:
    """Validated complete-unit ownership for one final payload array."""

    units: Mapping[int, EncodedPhysicalUnit]
    boundary_starts: tuple[int, ...]
    package_count: int
    encoded_object_count: int
    encoded_bytes: int


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return text


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r}.")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, target)


def _dimensions_document(dimensions: SubjectMaskDimensions) -> dict[str, int]:
    return {
        "n_frames": int(dimensions.n_frames),
        "n_rois": int(dimensions.n_rois),
        "n_channels": int(dimensions.n_channels),
        "roi_height": int(dimensions.roi_height),
        "roi_width": int(dimensions.roi_width),
    }


def _payload_contract_and_plan(
    *,
    kind: str,
    dimensions: SubjectMaskDimensions,
    profile: StorageProfile,
) -> tuple[str, Any, StoragePlan]:
    if kind == "raw_probability_uint8":
        path = "mask_probs_roi"
        schema = RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
        plans = plan_raw_subject_mask_storage(
            dimensions,
            encoding=SubjectMaskProbabilityEncoding.LINEAR_UINT8_0_255,
            profile=profile,
        )
    elif kind == "refined_dense_core":
        path = "masks_roi"
        schema = REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
        plans = plan_refined_subject_mask_publication_storage(
            dimensions,
            profile=profile,
        )
    else:
        raise ValueError(f"Unsupported subject-mask final-layout kind {kind!r}.")
    entries = [entry for entry in plans.entries if entry.rule.path == path]
    if len(entries) != 1:
        raise RuntimeError(
            f"Storage planner did not resolve one payload plan for {path}."
        )
    binding = next(binding for binding in schema.bindings if binding.path == path)
    contract = schema.contracts.resolve(
        binding.contract_id,
        binding.contract_version,
    )
    return path, contract, entries[0].plan


def subject_mask_final_layout_payload_plan(
    *,
    kind: str,
    dimensions: SubjectMaskDimensions,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> dict[str, object]:
    """Return the frozen payload plan workers and the finalizer must share."""

    path, _contract, plan = _payload_contract_and_plan(
        kind=kind,
        dimensions=dimensions,
        profile=profile,
    )
    return {
        "kind": kind,
        "array_path": path,
        "dimensions": _dimensions_document(dimensions),
        "storage_profile": profile.as_manifest(),
        "storage_plan": plan.as_dict(),
        "storage_plan_digest": canonical_json_sha256(plan.as_dict()),
    }


def _physical_row_ranges(plan: StoragePlan) -> tuple[tuple[int, int], ...]:
    unit = plan.shard_shape or plan.chunk_shape
    if unit is None or int(unit[0]) <= 0:
        raise ValueError("Subject-mask final payload requires a physical row unit.")
    return tuple(
        (start, min(int(plan.logical_shape[0]), start + int(unit[0])))
        for start in range(0, int(plan.logical_shape[0]), int(unit[0]))
    )


def _object_records(array_path: Path, *, row_unit_index: int) -> list[dict[str, Any]]:
    chunk_root = array_path / "c" / str(int(row_unit_index))
    if not chunk_root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in chunk_root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": path.relative_to(array_path).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def build_subject_mask_final_layout_unit_package(
    *,
    source_array: Any,
    source_crop_row_ids: Any,
    destination: Path,
    kind: str,
    dimensions: SubjectMaskDimensions,
    global_start_row: int,
    source_run_path: str,
    worker_receipt_payload_digest: str,
    producer_commit: str,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> dict[str, Any]:
    """Seal reusable complete final-layout row bands for one worker interval."""

    target = destination.expanduser().resolve()
    payload_path, contract, plan = _payload_contract_and_plan(
        kind=kind,
        dimensions=dimensions,
        profile=profile,
    )
    shape = tuple(int(value) for value in source_array.shape)
    dtype = np.dtype(source_array.dtype)
    local_rows = int(shape[0])
    start = int(global_start_row)
    stop = start + local_rows
    if start < 0 or stop > int(dimensions.n_rois):
        raise ValueError(
            "Final-layout package worker interval leaves the recording row domain."
        )
    if shape[1:] != tuple(int(value) for value in plan.logical_shape[1:]):
        raise ValueError(
            "Final-layout package payload trailing shape differs from the plan."
        )
    if dtype != np.dtype(plan.logical_dtype):
        raise ValueError("Final-layout package payload dtype differs from the plan.")
    rows = np.asarray(source_crop_row_ids[:], dtype=np.int64).reshape(-1)
    if not np.array_equal(rows, np.arange(start, stop, dtype=np.int64)):
        raise ValueError(
            "Final-layout package source_crop_row_ids must equal its exact global interval."
        )
    worker_digest = _require_sha256(
        worker_receipt_payload_digest,
        name="worker_receipt_payload_digest",
    )
    commit = str(producer_commit).strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("producer_commit must be a full lowercase Git commit.")
    source_path = str(source_run_path).strip().strip("/")
    if not source_path:
        raise ValueError("source_run_path cannot be empty.")

    if target.exists():
        existing = validate_subject_mask_final_layout_unit_package(target)
        expected = {
            "kind": kind,
            "array_path": payload_path,
            "dimensions": _dimensions_document(dimensions),
            "global_row_interval": {"start_row": start, "stop_row": stop},
            "source_run_path": source_path,
            "worker_receipt_payload_digest": worker_digest,
            "producer_commit": commit,
            "storage_plan_digest": canonical_json_sha256(plan.as_dict()),
        }
        payload = existing["payload"]
        if any(payload.get(key) != value for key, value in expected.items()):
            raise FileExistsError(
                f"Existing final-layout package has another identity: {target}"
            )
        return existing

    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        payload_store = temporary / "payload.zarr"
        root = zarr.open_group(str(payload_store), mode="w", zarr_format=3)
        destination_array = create_array_from_plan(
            root,
            name="payload",
            contract=contract,
            plan=plan,
            fill_value=0,
            attributes=_ARRAY_ATTRIBUTES,
        )
        array_path = payload_store / "payload"
        unit_rows = int((plan.shard_shape or plan.chunk_shape or (0,))[0])
        units: list[dict[str, Any]] = []
        boundary_ranges: list[dict[str, int]] = []
        trailing = (slice(None),) * (len(shape) - 1)
        for unit_start, unit_stop in _physical_row_ranges(plan):
            overlap_start = max(start, unit_start)
            overlap_stop = min(stop, unit_stop)
            if overlap_start >= overlap_stop:
                continue
            if unit_start < start or unit_stop > stop:
                boundary_ranges.append(
                    {"start_row": overlap_start, "stop_row": overlap_stop}
                )
                continue
            local_start = unit_start - start
            local_stop = unit_stop - start
            values = np.ascontiguousarray(
                np.asarray(source_array[(slice(local_start, local_stop), *trailing)])
            )
            destination_array[(slice(unit_start, unit_stop), *trailing)] = values
            unit_index = unit_start // unit_rows
            objects = _object_records(array_path, row_unit_index=unit_index)
            units.append(
                {
                    "unit_index": int(unit_index),
                    "start_row": int(unit_start),
                    "stop_row": int(unit_stop),
                    "logical_sha256": hashlib.sha256(values.view(np.uint8)).hexdigest(),
                    "objects": objects,
                }
            )
        metadata_path = array_path / "zarr.json"
        metadata = _strict_json(metadata_path)
        package_payload: dict[str, Any] = {
            "kind": kind,
            "array_path": payload_path,
            "dimensions": _dimensions_document(dimensions),
            "global_row_interval": {"start_row": start, "stop_row": stop},
            "source_run_path": source_path,
            "worker_receipt_payload_digest": worker_digest,
            "producer_commit": commit,
            "storage_profile": profile.as_manifest(),
            "storage_plan": plan.as_dict(),
            "storage_plan_digest": canonical_json_sha256(plan.as_dict()),
            "physical_unit_rows": unit_rows,
            "array_relative_path": SUBJECT_MASK_FINAL_LAYOUT_UNIT_ARRAY,
            "array_metadata_digest": canonical_json_sha256(metadata),
            "complete_unit_count": len(units),
            "encoded_object_count": sum(len(unit["objects"]) for unit in units),
            "encoded_bytes": sum(
                int(item["size_bytes"]) for unit in units for item in unit["objects"]
            ),
            "units": units,
            "boundary_ranges": boundary_ranges,
        }
        receipt = {
            "schema_id": SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": utc_now(),
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": canonical_json_sha256(package_payload),
            "payload": package_payload,
        }
        _write_json_atomic(
            temporary / SUBJECT_MASK_FINAL_LAYOUT_UNIT_RECEIPT,
            receipt,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
        return validate_subject_mask_final_layout_unit_package(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_subject_mask_final_layout_unit_package(
    package: Path,
    *,
    verify_object_digests: bool = True,
) -> dict[str, Any]:
    """Deeply validate one immutable final-layout unit package and its objects."""

    root = package.expanduser().resolve()
    receipt_path = root / SUBJECT_MASK_FINAL_LAYOUT_UNIT_RECEIPT
    if not receipt_path.is_file():
        raise ValueError(f"Final-layout package lacks {receipt_path.name}: {root}")
    receipt = _strict_json(receipt_path)
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_id",
        "schema_version",
        "status",
        "created_at_utc",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        raise ValueError("Final-layout package receipt envelope is not exact.")
    if (
        receipt.get("schema_id") != SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_ID
        or receipt.get("schema_version")
        != SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        raise ValueError("Final-layout package receipt identity is unsupported.")
    payload = receipt.get("payload")
    expected_payload_fields = {
        "kind",
        "array_path",
        "dimensions",
        "global_row_interval",
        "source_run_path",
        "worker_receipt_payload_digest",
        "producer_commit",
        "storage_profile",
        "storage_plan",
        "storage_plan_digest",
        "physical_unit_rows",
        "array_relative_path",
        "array_metadata_digest",
        "complete_unit_count",
        "encoded_object_count",
        "encoded_bytes",
        "units",
        "boundary_ranges",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_fields:
        raise ValueError("Final-layout package payload fields are not exact.")
    if receipt.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Final-layout package payload digest differs.")
    _require_sha256(
        payload.get("worker_receipt_payload_digest"),
        name="worker_receipt_payload_digest",
    )
    producer_commit = str(payload.get("producer_commit") or "")
    if len(producer_commit) != 40 or any(
        char not in "0123456789abcdef" for char in producer_commit
    ):
        raise ValueError("Final-layout package producer_commit is invalid.")
    source_run_path = str(payload.get("source_run_path") or "").strip().strip("/")
    if not source_run_path or source_run_path != payload.get("source_run_path"):
        raise ValueError("Final-layout package source_run_path is invalid.")
    if payload.get("storage_plan_digest") != canonical_json_sha256(
        payload.get("storage_plan")
    ):
        raise ValueError("Final-layout package storage-plan digest differs.")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "n_frames",
        "n_rois",
        "n_channels",
        "roi_height",
        "roi_width",
    }:
        raise ValueError("Final-layout package dimensions are not exact.")
    if any(
        type(dimensions[name]) is not int or int(dimensions[name]) <= 0
        for name in dimensions
    ):
        raise ValueError("Final-layout package dimensions must be positive integers.")
    storage_plan = payload.get("storage_plan")
    if not isinstance(storage_plan, Mapping):
        raise ValueError("Final-layout package storage plan must be an object.")
    expected_shape = [
        int(dimensions["n_rois"]),
        int(dimensions["n_channels"]),
        int(dimensions["roi_height"]),
        int(dimensions["roi_width"]),
    ]
    if (
        storage_plan.get("logical_shape") != expected_shape
        or storage_plan.get("logical_dtype") != "uint8"
    ):
        raise ValueError("Final-layout package dimensions/storage plan differ.")
    relative_array = PurePosixPath(str(payload.get("array_relative_path") or ""))
    if (
        relative_array.as_posix() != SUBJECT_MASK_FINAL_LAYOUT_UNIT_ARRAY
        or relative_array.is_absolute()
        or ".." in relative_array.parts
    ):
        raise ValueError("Final-layout package array path is unsafe.")
    array_path = root.joinpath(*relative_array.parts)
    metadata_path = array_path / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError("Final-layout package array metadata is absent.")
    metadata = _strict_json(metadata_path)
    if payload.get("array_metadata_digest") != canonical_json_sha256(metadata):
        raise ValueError("Final-layout package array metadata digest differs.")
    interval = payload.get("global_row_interval")
    if not isinstance(interval, Mapping) or set(interval) != {"start_row", "stop_row"}:
        raise ValueError("Final-layout package row interval is not exact.")
    start = interval.get("start_row")
    stop = interval.get("stop_row")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise ValueError("Final-layout package row interval is invalid.")
    unit_rows = payload.get("physical_unit_rows")
    if type(unit_rows) is not int or unit_rows <= 0:
        raise ValueError("Final-layout package physical_unit_rows is invalid.")
    outer_shape = storage_plan.get("shard_shape") or storage_plan.get("chunk_shape")
    if (
        not isinstance(outer_shape, list)
        or not outer_shape
        or outer_shape[0] != unit_rows
    ):
        raise ValueError(
            "Final-layout package physical row unit differs from its plan."
        )
    units = payload.get("units")
    if not isinstance(units, list):
        raise ValueError("Final-layout package units must be an array.")
    seen_starts: set[int] = set()
    object_count = 0
    encoded_bytes = 0
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != {
            "unit_index",
            "start_row",
            "stop_row",
            "logical_sha256",
            "objects",
        }:
            raise ValueError("Final-layout package unit fields are not exact.")
        unit_start = unit.get("start_row")
        unit_stop = unit.get("stop_row")
        unit_index = unit.get("unit_index")
        if (
            type(unit_start) is not int
            or type(unit_stop) is not int
            or type(unit_index) is not int
            or unit_start in seen_starts
            or unit_start % unit_rows != 0
            or unit_index != unit_start // unit_rows
            or unit_start < start
            or unit_stop > stop
            or not unit_start < unit_stop <= unit_start + unit_rows
        ):
            raise ValueError("Final-layout package unit interval is invalid.")
        seen_starts.add(unit_start)
        _require_sha256(unit.get("logical_sha256"), name="unit logical_sha256")
        objects = unit.get("objects")
        if not isinstance(objects, list):
            raise ValueError("Final-layout package unit objects must be an array.")
        seen_paths: set[str] = set()
        required_prefix = f"c/{unit_index}/"
        for item in objects:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise ValueError("Final-layout encoded-object fields are not exact.")
            relative = PurePosixPath(str(item.get("path") or ""))
            relative_text = relative.as_posix()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative_text.startswith(required_prefix)
                or relative_text in seen_paths
            ):
                raise ValueError("Final-layout encoded-object path is unsafe.")
            seen_paths.add(relative_text)
            size = item.get("size_bytes")
            if type(size) is not int or size < 0:
                raise ValueError("Final-layout encoded-object size is invalid.")
            expected_sha = _require_sha256(
                item.get("sha256"), name="encoded-object sha256"
            )
            source = array_path.joinpath(*relative.parts)
            if not source.is_file() or int(source.stat().st_size) != size:
                raise ValueError("Final-layout encoded object is absent or changed.")
            if verify_object_digests and sha256_file(source) != expected_sha:
                raise ValueError("Final-layout encoded-object digest differs.")
            object_count += 1
            encoded_bytes += size
    if payload.get("complete_unit_count") != len(units):
        raise ValueError("Final-layout complete-unit count differs.")
    if payload.get("encoded_object_count") != object_count:
        raise ValueError("Final-layout encoded-object count differs.")
    if payload.get("encoded_bytes") != encoded_bytes:
        raise ValueError("Final-layout encoded byte count differs.")
    expected_unit_intervals: list[tuple[int, int]] = []
    expected_boundaries: list[dict[str, int]] = []
    for unit_start in range(0, int(dimensions["n_rois"]), unit_rows):
        unit_stop = min(int(dimensions["n_rois"]), unit_start + unit_rows)
        overlap_start = max(start, unit_start)
        overlap_stop = min(stop, unit_stop)
        if overlap_start >= overlap_stop:
            continue
        if start <= unit_start and unit_stop <= stop:
            expected_unit_intervals.append((unit_start, unit_stop))
        else:
            expected_boundaries.append(
                {"start_row": overlap_start, "stop_row": overlap_stop}
            )
    observed_unit_intervals = [
        (int(unit["start_row"]), int(unit["stop_row"])) for unit in units
    ]
    if observed_unit_intervals != expected_unit_intervals:
        raise ValueError("Final-layout package complete-unit coverage differs.")
    if payload.get("boundary_ranges") != expected_boundaries:
        raise ValueError("Final-layout package boundary coverage differs.")
    return receipt


def prepare_subject_mask_final_layout_unit_adoption(
    package_paths: Sequence[Path],
    *,
    kind: str,
    dimensions: SubjectMaskDimensions,
    plan: StoragePlan,
    source_validation_receipt: Mapping[str, Any],
    require_complete_eligible_units: bool,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
) -> FinalLayoutUnitAdoption:
    """Preflight package ownership and return encoded units safe to adopt."""

    payload_path, contract, expected_plan = _payload_contract_and_plan(
        kind=kind,
        dimensions=dimensions,
        profile=profile,
    )
    if plan.as_dict() != expected_plan.as_dict():
        raise ValueError("Final-layout adoption requires the published HTTP plan.")
    producer = source_validation_receipt.get("payload", {}).get("producer_evidence")
    workers = producer.get("workers") if isinstance(producer, Mapping) else None
    if not isinstance(workers, list) or not workers:
        raise ValueError("Final-layout adoption requires recording worker evidence.")
    expected_by_digest: dict[str, tuple[int, int, str]] = {}
    for worker in workers:
        if not isinstance(worker, Mapping):
            raise ValueError("Recording worker evidence is malformed.")
        digest = _require_sha256(
            worker.get("worker_receipt_payload_digest"),
            name="worker_receipt_payload_digest",
        )
        interval = worker.get("global_row_interval")
        if not isinstance(interval, Mapping):
            raise ValueError("Recording worker interval is malformed.")
        expected_by_digest[digest] = (
            int(interval["start_row"]),
            int(interval["stop_row"]),
            str(worker.get("run_path") or "").strip().strip("/"),
        )

    plan_digest = canonical_json_sha256(plan.as_dict())
    expected_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        **array_metadata_declaration_from_plan(
            contract=contract,
            plan=plan,
            fill_value=0,
            attributes=_ARRAY_ATTRIBUTES,
        ),
    }
    observed_workers: set[str] = set()
    units: dict[int, EncodedPhysicalUnit] = {}
    encoded_object_count = 0
    encoded_bytes = 0
    for package_path in package_paths:
        receipt = validate_subject_mask_final_layout_unit_package(
            package_path,
            verify_object_digests=False,
        )
        payload = receipt["payload"]
        if (
            payload.get("kind") != kind
            or payload.get("array_path") != payload_path
            or payload.get("dimensions") != _dimensions_document(dimensions)
            or payload.get("storage_plan") != plan.as_dict()
            or payload.get("storage_plan_digest") != plan_digest
        ):
            raise ValueError(
                "Final-layout package contract differs from the destination."
            )
        package_array_path = (
            package_path.expanduser().resolve() / SUBJECT_MASK_FINAL_LAYOUT_UNIT_ARRAY
        )
        observed_metadata = _strict_json(package_array_path / "zarr.json")
        observed_metadata = {
            key: value
            for key, value in observed_metadata.items()
            if key != "consolidated_metadata"
        }
        if observed_metadata != expected_metadata:
            raise ValueError(
                "Final-layout package encoded metadata differs from the destination."
            )
        worker_digest = str(payload["worker_receipt_payload_digest"])
        if worker_digest not in expected_by_digest or worker_digest in observed_workers:
            raise ValueError(
                "Final-layout package worker binding is absent or duplicated."
            )
        expected_start, expected_stop, expected_run = expected_by_digest[worker_digest]
        if (
            payload.get("global_row_interval")
            != {
                "start_row": expected_start,
                "stop_row": expected_stop,
            }
            or payload.get("source_run_path") != expected_run
        ):
            raise ValueError("Final-layout package worker identity differs.")
        observed_workers.add(worker_digest)
        array_path = package_array_path
        for item in payload["units"]:
            start = int(item["start_row"])
            if start in units:
                raise ValueError("Final-layout physical unit has multiple owners.")
            objects = tuple(
                EncodedObject(
                    source_path=array_path.joinpath(*PurePosixPath(obj["path"]).parts),
                    relative_path=str(obj["path"]),
                    size_bytes=int(obj["size_bytes"]),
                    sha256=str(obj["sha256"]),
                )
                for obj in item["objects"]
            )
            units[start] = EncodedPhysicalUnit(
                package_path=package_path.expanduser().resolve(),
                start_row=start,
                stop_row=int(item["stop_row"]),
                logical_sha256=str(item["logical_sha256"]),
                objects=objects,
            )
            encoded_object_count += len(objects)
            encoded_bytes += sum(obj.size_bytes for obj in objects)
    if observed_workers != set(expected_by_digest):
        raise ValueError("Final-layout packages do not cover every recording worker.")

    expected_starts: set[int] = set()
    boundary_starts: list[int] = []
    for start, stop in _physical_row_ranges(plan):
        owners = [
            digest
            for digest, (worker_start, worker_stop, _run) in expected_by_digest.items()
            if worker_start <= start and stop <= worker_stop
        ]
        if len(owners) == 1:
            expected_starts.add(start)
        elif not owners:
            boundary_starts.append(start)
        else:  # pragma: no cover - recording receipt already forbids overlap
            raise ValueError("Final-layout row band has multiple worker owners.")
    if require_complete_eligible_units and set(units) != expected_starts:
        missing = sorted(expected_starts - set(units))[:8]
        unexpected = sorted(set(units) - expected_starts)[:8]
        raise ValueError(
            "Final-layout complete-unit coverage differs: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if not set(units).issubset(expected_starts):
        raise ValueError("Final-layout package includes a cross-worker boundary unit.")
    return FinalLayoutUnitAdoption(
        units=units,
        boundary_starts=tuple(boundary_starts),
        package_count=len(package_paths),
        encoded_object_count=encoded_object_count,
        encoded_bytes=encoded_bytes,
    )


def copy_encoded_physical_unit(
    unit: EncodedPhysicalUnit,
    *,
    destination_array_path: Path,
) -> None:
    """Copy and re-verify one complete encoded physical unit."""

    target_root = destination_array_path.expanduser().resolve()
    for item in unit.objects:
        relative = PurePosixPath(item.relative_path)
        target = target_root.joinpath(*relative.parts)
        if target.exists():
            raise FileExistsError(
                f"Encoded destination object already exists: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        cloned = False
        try:
            import fcntl

            with (
                item.source_path.open("rb") as source_handle,
                target.open("xb") as target_handle,
            ):
                fcntl.ioctl(target_handle.fileno(), _FICLONE, source_handle.fileno())
            cloned = True
        except (ImportError, OSError):
            if target.exists():
                target.unlink()
        if not cloned:
            shutil.copyfile(item.source_path, target)
        if (
            int(target.stat().st_size) != item.size_bytes
            or sha256_file(target) != item.sha256
        ):
            raise RuntimeError(f"Copied encoded subject-mask object differs: {target}")


__all__ = [
    "EncodedPhysicalUnit",
    "FinalLayoutUnitAdoption",
    "SUBJECT_MASK_FINAL_LAYOUT_UNIT_ARRAY",
    "SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_ID",
    "SUBJECT_MASK_FINAL_LAYOUT_UNIT_PACKAGE_SCHEMA_VERSION",
    "build_subject_mask_final_layout_unit_package",
    "copy_encoded_physical_unit",
    "prepare_subject_mask_final_layout_unit_adoption",
    "subject_mask_final_layout_payload_plan",
    "validate_subject_mask_final_layout_unit_package",
]

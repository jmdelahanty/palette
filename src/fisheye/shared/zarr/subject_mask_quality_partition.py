"""Immutable worker partitions for recording-scale subject-mask quality.

Quality values are observation-local, so clip/window workers may compute them
independently.  The receipts in this module bind each partition to the exact
refined-mask worker receipt and global row interval.  Recording publication
may then concatenate the small quality arrays without recomputing topology
over the complete dense mask surface.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.subject_mask_quality_producer import (
    SUBJECT_V1_LR_COMPONENTS,
    SubjectMaskQualityPayload,
    SubjectV1LrObservationQualityPolicy,
    compute_subject_mask_quality_block,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM,
    subject_mask_array_unit_document,
)

SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask_quality.partition_receipt"
)
SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION = 2
SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_ID = (
    "palette.subject_mask_quality.partition_assembly"
)
SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_VERSION = 2
DEFAULT_PARTITION_UNIT_ROWS = 1024
DEFAULT_PARTITION_COMPUTE_BLOCK_BYTES = 64 * 1024 * 1024

_PARTITION_ARRAY_PATHS = tuple(
    path
    for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
    if path != "frame_row_offsets"
)


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
        raise ValueError("Quality partition source_run_path is unsafe.")
    return path


def _dense_worker_binding(
    worker_receipt: Mapping[str, Any], *, run_path: str, row_count: int
) -> dict[str, object]:
    receipt = _strict_copy(worker_receipt, name="refined worker receipt")
    payload = receipt.get("payload") if isinstance(receipt, Mapping) else None
    arrays = payload.get("arrays") if isinstance(payload, Mapping) else None
    dense = arrays.get("masks_roi") if isinstance(arrays, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or receipt.get("payload_digest") != canonical_json_sha256(payload)
        or not isinstance(payload, Mapping)
        or payload.get("stage_kind") != "refined_subject_mask"
        or payload.get("run_path") != run_path
        or not isinstance(dense, Mapping)
        or dense.get("shape", [None])[0] != row_count
        or dense.get("dtype") != "uint8"
        or dense.get("digest_algorithm") != SUBJECT_MASK_ARRAY_UNIT_DIGEST_ALGORITHM
        or not _is_sha256(dense.get("units_digest"))
        or not isinstance(dense.get("units"), list)
        or not dense["units"]
        or dense.get("unit_count") != len(dense["units"])
        or dense.get("units_digest") != canonical_json_sha256(dense["units"])
    ):
        raise ValueError("Quality partition lacks a valid refined worker binding.")
    return {
        "worker_receipt_payload_digest": receipt["payload_digest"],
        "dense_array_path": "masks_roi",
        "dense_array_shape": list(dense["shape"]),
        "dense_array_dtype": "uint8",
        "dense_array_units_digest": dense["units_digest"],
        "dense_array_unit_count": dense["unit_count"],
        "dense_array_units": [dict(unit) for unit in dense["units"]],
    }


class _DenseWorkerValueVerifier:
    """Verify required QC reads against the producer's ordered unit hashes."""

    def __init__(self, binding: Mapping[str, Any]) -> None:
        self._units = tuple(binding["dense_array_units"])
        self._shape = tuple(int(value) for value in binding["dense_array_shape"])
        self._cursor = 0
        self._unit_index = 0
        self._unit_digest = hashlib.sha256()
        row_bytes = int(np.prod(self._shape[1:], dtype=np.int64))
        cursor = 0
        for unit in self._units:
            if not isinstance(unit, Mapping) or set(unit) != {
                "start_row",
                "stop_row",
                "decoded_bytes",
                "sha256",
            }:
                raise ValueError("Dense worker receipt unit fields are not exact.")
            start = unit.get("start_row")
            stop = unit.get("stop_row")
            if (
                type(start) is not int
                or type(stop) is not int
                or start != cursor
                or not (start < stop <= self._shape[0])
                or unit.get("decoded_bytes") != (stop - start) * row_bytes
                or not _is_sha256(unit.get("sha256"))
            ):
                raise ValueError("Dense worker receipt unit coverage differs.")
            cursor = stop
        if cursor != self._shape[0]:
            raise ValueError("Dense worker receipt units do not cover every row.")

    def append(self, start_row: int, values: np.ndarray[Any, Any]) -> None:
        if int(start_row) != self._cursor:
            raise ValueError("Dense QC reads are not ordered and contiguous.")
        block = np.asarray(values)
        if (
            block.dtype != np.dtype(np.uint8)
            or tuple(block.shape[1:]) != self._shape[1:]
        ):
            raise ValueError("Dense QC read shape or dtype differs from its receipt.")
        offset = 0
        while offset < int(block.shape[0]):
            if self._unit_index >= len(self._units):
                raise ValueError("Dense QC reads exceed their receipt coverage.")
            unit = self._units[self._unit_index]
            take = min(
                int(unit["stop_row"]) - self._cursor,
                int(block.shape[0]) - offset,
            )
            part = np.ascontiguousarray(block[offset : offset + take])
            self._unit_digest.update(part.view(np.uint8))
            self._cursor += take
            offset += take
            if self._cursor == int(unit["stop_row"]):
                if self._unit_digest.hexdigest() != unit["sha256"]:
                    raise ValueError(
                        "QC source masks differ from their worker receipt."
                    )
                self._unit_index += 1
                self._unit_digest = hashlib.sha256()

    def finish(self) -> None:
        if self._cursor != self._shape[0] or self._unit_index != len(self._units):
            raise ValueError("Dense QC verification coverage is incomplete.")


def _effective_block_rows(run: Any, budget_bytes: int) -> int:
    masks = run["masks_roi"]
    shape = tuple(int(value) for value in masks.shape)
    row_bytes = int(np.prod(shape[1:], dtype=np.int64))
    budget_rows = max(1, int(budget_bytes) // max(1, row_bytes))
    chunks = getattr(masks, "chunks", None)
    if not isinstance(chunks, (tuple, list)) or not chunks:
        return budget_rows
    physical_row_chunk = int(chunks[0])
    if physical_row_chunk <= 0:
        return budget_rows
    if budget_rows < physical_row_chunk:
        return min(int(shape[0]), physical_row_chunk)
    return max(
        physical_row_chunk,
        (budget_rows // physical_row_chunk) * physical_row_chunk,
    )


def _write_arrays(directory: Path, arrays: Mapping[str, np.ndarray]) -> None:
    directory.mkdir(parents=True)
    for path in _PARTITION_ARRAY_PATHS:
        with (directory / f"{path}.npy").open("wb") as handle:
            np.save(handle, np.ascontiguousarray(arrays[path]), allow_pickle=False)


def load_subject_mask_quality_partition_arrays(
    partition_root: Path,
) -> dict[str, np.ndarray]:
    root = partition_root.expanduser().resolve()
    return {
        path: np.load(root / "arrays" / f"{path}.npy", mmap_mode="r")
        for path in _PARTITION_ARRAY_PATHS
    }


def compute_subject_mask_quality_partition(
    run: Any,
    *,
    source_acquisition_frame_index: np.ndarray,
    global_start_row: int,
    global_frame_start: int,
    global_frame_stop: int,
    work_unit_id: str,
    work_unit_index: int,
    source_worker_receipt: Mapping[str, Any],
    producer_commit: str,
    destination: Path,
    compute_workers: int = 4,
    source_compute_block_bytes: int = DEFAULT_PARTITION_COMPUTE_BLOCK_BYTES,
    receipt_unit_rows: int = DEFAULT_PARTITION_UNIT_ROWS,
) -> dict[str, object]:
    """Compute and atomically seal one observation-local quality partition."""

    start = int(global_start_row)
    frame_start = int(global_frame_start)
    frame_stop = int(global_frame_stop)
    if type(global_start_row) is not int or start < 0:
        raise ValueError("global_start_row must be a nonnegative exact integer.")
    if (
        type(global_frame_start) is not int
        or type(global_frame_stop) is not int
        or not (0 <= frame_start < frame_stop)
    ):
        raise ValueError("Quality partition frame interval is invalid.")
    if type(work_unit_index) is not int or work_unit_index < 0:
        raise ValueError("work_unit_index must be a nonnegative exact integer.")
    if type(compute_workers) is not int or compute_workers <= 0:
        raise ValueError("compute_workers must be a positive exact integer.")
    if type(source_compute_block_bytes) is not int or source_compute_block_bytes <= 0:
        raise ValueError("source_compute_block_bytes must be positive.")
    if type(receipt_unit_rows) is not int or receipt_unit_rows <= 0:
        raise ValueError("receipt_unit_rows must be positive.")
    unit_id = str(work_unit_id).strip()
    commit = str(producer_commit).strip()
    if not unit_id or not commit:
        raise ValueError("work_unit_id and producer_commit must be nonempty.")

    masks = run["masks_roi"]
    rows = int(masks.shape[0])
    if rows <= 0 or tuple(int(value) for value in masks.shape[1:])[:1] != (4,):
        raise ValueError("Quality partitions require nonempty masks_roi[N,4,H,W].")
    components = SubjectMaskComponentRegistry(
        tuple(str(value) for value in run.attrs["mask_labels"])
    )
    available = np.asarray(run["available_channels"][:], dtype=bool)
    frames = np.asarray(source_acquisition_frame_index, dtype=np.int64)
    if frames.shape != (rows,):
        raise ValueError("Quality partition frame identity must have shape [N].")
    if (
        np.any(frames < frame_start)
        or np.any(frames >= frame_stop)
        or np.any(frames[1:] < frames[:-1])
    ):
        raise ValueError("Quality partition rows escape their frame interval.")
    keys = np.asarray(run["instance_key"][:], dtype=np.uint64)
    if keys.shape != (rows,) or np.unique(keys).size != rows:
        raise ValueError("Quality partition instance keys must be unique uint64[N].")
    run_path = _safe_run_path(run.path)
    dense_binding = _dense_worker_binding(
        source_worker_receipt, run_path=run_path, row_count=rows
    )
    dense_verifier = _DenseWorkerValueVerifier(dense_binding)
    policy = SubjectV1LrObservationQualityPolicy()
    profile = quality_profile_for_policy(policy)
    block_rows = _effective_block_rows(run, int(source_compute_block_bytes))
    block_count = max(1, math.ceil(rows / block_rows))
    effective_workers = min(int(compute_workers), block_count)

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    started = time.perf_counter()
    try:
        arrays: dict[str, np.ndarray] = {
            "instance_key": keys,
            "source_mask_row_ids": np.arange(start, start + rows, dtype=np.int64),
            "source_acquisition_frame_index": frames,
            "component_metric_values": np.empty((rows, 4, 8), dtype=np.float32),
            "component_metric_valid": np.empty((rows, 4, 8), dtype=bool),
            "observation_metric_values": np.empty((rows, 7), dtype=np.float32),
            "observation_metric_valid": np.empty((rows, 7), dtype=bool),
            "component_quality_flags": np.empty((rows, 4), dtype=np.uint16),
            "observation_quality_flags": np.empty((rows,), dtype=np.uint16),
            "proposed_component_usable": np.empty((rows, 4), dtype=bool),
            "proposed_observation_usable": np.empty((rows,), dtype=bool),
        }

        def store(
            block_start: int, block_stop: int, payload: SubjectMaskQualityPayload
        ) -> None:
            for path, values in payload.as_arrays().items():
                arrays[path][block_start:block_stop] = values

        pending: deque[tuple[int, int, Future[SubjectMaskQualityPayload]]] = deque()
        executor = (
            None
            if effective_workers == 1
            else ThreadPoolExecutor(
                max_workers=effective_workers,
                thread_name_prefix="subject-mask-quality-partition",
            )
        )
        try:
            for block_start in range(0, rows, block_rows):
                block_stop = min(rows, block_start + block_rows)
                block = np.ascontiguousarray(masks[block_start:block_stop])
                dense_verifier.append(block_start, block)
                if executor is None:
                    store(
                        block_start,
                        block_stop,
                        compute_subject_mask_quality_block(
                            block,
                            available_channels=available,
                            components=components,
                            policy=policy,
                        ),
                    )
                else:
                    pending.append(
                        (
                            block_start,
                            block_stop,
                            executor.submit(
                                compute_subject_mask_quality_block,
                                block,
                                available_channels=available,
                                components=components,
                                policy=policy,
                            ),
                        )
                    )
                    if len(pending) >= effective_workers:
                        pending_start, pending_stop, future = pending.popleft()
                        store(pending_start, pending_stop, future.result())
            while pending:
                pending_start, pending_stop, future = pending.popleft()
                store(pending_start, pending_stop, future.result())
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        dense_verifier.finish()

        _write_arrays(temporary / "arrays", arrays)
        array_document = subject_mask_array_unit_document(
            arrays, _PARTITION_ARRAY_PATHS, unit_rows=int(receipt_unit_rows)
        )
        payload = {
            "kind": "subject_mask_quality_observation_partition",
            "work_unit": {
                "work_unit_id": unit_id,
                "work_unit_index": int(work_unit_index),
                "global_frame_interval": {
                    "start_frame": frame_start,
                    "stop_frame": frame_stop,
                },
                "global_row_interval": {
                    "start_row": start,
                    "stop_row": start + rows,
                },
            },
            "local_row_count": rows,
            "source_run_path": run_path,
            "source_dense_worker": dense_binding,
            "source_dense_verification": {
                "mode": "required_qc_read_compared_to_worker_unit_receipt_v1",
                "status": "passed",
            },
            "component_registry": components.as_manifest(),
            "quality_profile": profile.as_manifest(),
            "quality_policy": policy.as_manifest(),
            "arrays": array_document,
            "array_document_digest": canonical_json_sha256(array_document),
            "execution": {
                "compute_block_rows": block_rows,
                "compute_block_count": block_count,
                "compute_workers_requested": int(compute_workers),
                "compute_workers_effective": effective_workers,
                "receipt_unit_rows": int(receipt_unit_rows),
                "source_compute_block_bytes_budget": int(source_compute_block_bytes),
            },
            "producer_commit": commit,
        }
        receipt = {
            "schema_id": SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION,
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        (temporary / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
        result = {
            "status": "complete",
            "receipt_payload_digest": receipt["payload_digest"],
            "duration_seconds": float(time.perf_counter() - started),
        }
        (temporary / "result.json").write_bytes(canonical_json_bytes(result) + b"\n")
        validate_subject_mask_quality_partition(
            temporary,
            source_worker_receipt=source_worker_receipt,
            verify_values=True,
        )
        os.replace(temporary, destination)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_subject_mask_quality_partition(
    partition_root: Path,
    *,
    source_worker_receipt: Mapping[str, Any] | None = None,
    verify_values: bool = True,
) -> dict[str, Any]:
    root = partition_root.expanduser().resolve()
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        raise ValueError("Quality partition receipt fields are not exact.")
    payload = receipt.get("payload")
    if (
        receipt.get("schema_id") != SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID
        or receipt.get("schema_version")
        != SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION
        or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, dict)
        or receipt.get("payload_digest") != canonical_json_sha256(payload)
    ):
        raise ValueError("Quality partition receipt envelope differs.")
    expected_fields = {
        "kind",
        "work_unit",
        "local_row_count",
        "source_run_path",
        "source_dense_worker",
        "source_dense_verification",
        "component_registry",
        "quality_profile",
        "quality_policy",
        "arrays",
        "array_document_digest",
        "execution",
        "producer_commit",
    }
    if set(payload) != expected_fields:
        raise ValueError("Quality partition payload fields are not exact.")
    work_unit = payload.get("work_unit")
    frame_interval = (
        work_unit.get("global_frame_interval")
        if isinstance(work_unit, Mapping)
        else None
    )
    row_interval = (
        work_unit.get("global_row_interval") if isinstance(work_unit, Mapping) else None
    )
    rows = payload.get("local_row_count")
    start_row = (
        row_interval.get("start_row") if isinstance(row_interval, Mapping) else None
    )
    stop_row = (
        row_interval.get("stop_row") if isinstance(row_interval, Mapping) else None
    )
    start_frame = (
        frame_interval.get("start_frame")
        if isinstance(frame_interval, Mapping)
        else None
    )
    stop_frame = (
        frame_interval.get("stop_frame")
        if isinstance(frame_interval, Mapping)
        else None
    )
    if (
        payload.get("kind") != "subject_mask_quality_observation_partition"
        or not isinstance(work_unit, Mapping)
        or set(work_unit)
        != {
            "work_unit_id",
            "work_unit_index",
            "global_frame_interval",
            "global_row_interval",
        }
        or type(work_unit.get("work_unit_id")) is not str
        or not work_unit["work_unit_id"].strip()
        or type(work_unit.get("work_unit_index")) is not int
        or work_unit["work_unit_index"] < 0
        or not isinstance(frame_interval, Mapping)
        or set(frame_interval) != {"start_frame", "stop_frame"}
        or type(start_frame) is not int
        or type(stop_frame) is not int
        or not (0 <= start_frame < stop_frame)
        or not isinstance(row_interval, Mapping)
        or set(row_interval) != {"start_row", "stop_row"}
        or type(start_row) is not int
        or type(stop_row) is not int
        or start_row < 0
        or type(rows) is not int
        or rows <= 0
        or stop_row - start_row != rows
        or not str(payload.get("producer_commit") or "").strip()
    ):
        raise ValueError("Quality partition work-unit identity differs.")
    policy = SubjectV1LrObservationQualityPolicy()
    profile = quality_profile_for_policy(policy)
    component_registry = payload.get("component_registry")
    if (
        component_registry
        != SubjectMaskComponentRegistry(
            tuple(component_registry.get("labels", ()))
            if isinstance(component_registry, Mapping)
            else ()
        ).as_manifest()
        or tuple(component_registry.get("labels", ())) != SUBJECT_V1_LR_COMPONENTS
        or payload.get("quality_policy") != policy.as_manifest()
        or payload.get("quality_profile") != profile.as_manifest()
    ):
        raise ValueError("Quality partition component or quality contract differs.")
    source_binding = payload.get("source_dense_worker")
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding)
        != {
            "worker_receipt_payload_digest",
            "dense_array_path",
            "dense_array_shape",
            "dense_array_dtype",
            "dense_array_units_digest",
            "dense_array_unit_count",
            "dense_array_units",
        }
        or not _is_sha256(source_binding.get("worker_receipt_payload_digest"))
        or source_binding.get("dense_array_path") != "masks_roi"
        or source_binding.get("dense_array_shape", [None])[0] != rows
        or source_binding.get("dense_array_dtype") != "uint8"
        or not _is_sha256(source_binding.get("dense_array_units_digest"))
        or not isinstance(source_binding.get("dense_array_units"), list)
        or source_binding.get("dense_array_unit_count")
        != len(source_binding.get("dense_array_units", []))
        or source_binding.get("dense_array_units_digest")
        != canonical_json_sha256(source_binding.get("dense_array_units", []))
    ):
        raise ValueError("Quality partition dense-worker binding differs.")
    _DenseWorkerValueVerifier(source_binding)
    if payload.get("source_dense_verification") != {
        "mode": "required_qc_read_compared_to_worker_unit_receipt_v1",
        "status": "passed",
    }:
        raise ValueError("Quality partition lacks verified dense-source evidence.")
    source_run_path = _safe_run_path(payload.get("source_run_path"))
    if source_worker_receipt is not None and payload.get(
        "source_dense_worker"
    ) != _dense_worker_binding(
        source_worker_receipt, run_path=source_run_path, row_count=rows
    ):
        raise ValueError("Quality partition binds another refined worker.")
    arrays = load_subject_mask_quality_partition_arrays(root)
    execution = payload.get("execution")
    if (
        not isinstance(execution, Mapping)
        or set(execution)
        != {
            "compute_block_rows",
            "compute_block_count",
            "compute_workers_requested",
            "compute_workers_effective",
            "receipt_unit_rows",
            "source_compute_block_bytes_budget",
        }
        or any(
            type(execution.get(name)) is not int or execution[name] <= 0
            for name in execution
        )
        or execution["compute_workers_effective"]
        > execution["compute_workers_requested"]
        or execution["compute_block_count"]
        != math.ceil(rows / execution["compute_block_rows"])
    ):
        raise ValueError("Quality partition execution declaration differs.")
    document = payload.get("arrays")
    if not isinstance(document, Mapping) or set(document) != set(
        _PARTITION_ARRAY_PATHS
    ):
        raise ValueError("Quality partition array inventory differs.")
    if payload.get("array_document_digest") != canonical_json_sha256(document):
        raise ValueError("Quality partition array-document digest differs.")
    if verify_values:
        observed = subject_mask_array_unit_document(
            arrays,
            _PARTITION_ARRAY_PATHS,
            unit_rows=execution["receipt_unit_rows"],
        )
        if observed != document:
            raise ValueError("Quality partition decoded arrays differ from receipt.")
    start = int(start_row)
    if not np.array_equal(
        np.asarray(arrays["source_mask_row_ids"]),
        np.arange(start, start + rows, dtype=np.int64),
    ):
        raise ValueError("Quality partition source-row identity differs.")
    frames = np.asarray(arrays["source_acquisition_frame_index"])
    keys = np.asarray(arrays["instance_key"])
    if (
        frames.dtype != np.dtype(np.int64)
        or frames.shape != (rows,)
        or np.any(frames < start_frame)
        or np.any(frames >= stop_frame)
        or np.any(frames[1:] < frames[:-1])
    ):
        raise ValueError("Quality partition frame identity differs.")
    if (
        keys.dtype != np.dtype(np.uint64)
        or keys.shape != (rows,)
        or np.unique(keys).size != rows
    ):
        raise ValueError("Quality partition instance identity differs.")
    return receipt


def build_subject_mask_quality_partition_assembly(
    receipts: Sequence[Mapping[str, Any]],
    *,
    n_rois: int,
    producer_commit: str,
    source_producer_evidence: Mapping[str, Any],
) -> dict[str, object]:
    if not receipts:
        raise ValueError("Quality partition assembly requires workers.")
    canonical = [
        _strict_copy(receipt, name="quality partition receipt") for receipt in receipts
    ]
    ordered = sorted(
        canonical,
        key=lambda item: item["payload"]["work_unit"]["global_row_interval"][
            "start_row"
        ],
    )
    cursor = 0
    first = ordered[0]["payload"]
    evidence = _strict_copy(
        source_producer_evidence, name="quality source producer evidence"
    )
    evidence_workers = (
        evidence.get("workers") if isinstance(evidence, Mapping) else None
    )
    if not isinstance(evidence_workers, list) or not evidence_workers:
        raise ValueError("Quality assembly source producer evidence lacks workers.")
    expected_workers = {
        str(worker.get("worker_receipt_payload_digest")): worker
        for worker in evidence_workers
        if isinstance(worker, Mapping)
    }
    if len(expected_workers) != len(evidence_workers):
        raise ValueError("Quality assembly source worker identities are not unique.")
    observed_worker_digests: set[str] = set()
    worker_records: list[dict[str, object]] = []
    for receipt in ordered:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt)
            != {
                "schema_id",
                "schema_version",
                "digest_algorithm",
                "payload_digest",
                "payload",
            }
            or receipt.get("schema_id")
            != SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID
            or receipt.get("schema_version")
            != SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION
            or receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
            or not isinstance(receipt.get("payload"), Mapping)
            or receipt.get("payload_digest")
            != canonical_json_sha256(receipt.get("payload"))
        ):
            raise ValueError("Quality partition receipt envelope differs.")
        payload = receipt["payload"]
        interval = payload["work_unit"]["global_row_interval"]
        if interval["start_row"] != cursor:
            raise ValueError("Quality partitions have a gap, overlap, or reordering.")
        if (
            payload["component_registry"] != first["component_registry"]
            or payload["quality_profile"] != first["quality_profile"]
            or payload["quality_policy"] != first["quality_policy"]
            or payload["producer_commit"] != str(producer_commit)
        ):
            raise ValueError("Quality partition contracts or producer commits differ.")
        worker_digest = payload["source_dense_worker"]["worker_receipt_payload_digest"]
        evidence_worker = expected_workers.get(str(worker_digest))
        if (
            not isinstance(evidence_worker, Mapping)
            or evidence_worker.get("run_path") != payload["source_run_path"]
            or evidence_worker.get("global_row_interval") != interval
            or worker_digest in observed_worker_digests
        ):
            raise ValueError("Quality partition binds another refined worker assembly.")
        observed_worker_digests.add(str(worker_digest))
        cursor = int(interval["stop_row"])
        worker_records.append(
            {
                "receipt_payload_digest": receipt["payload_digest"],
                "work_unit": payload["work_unit"],
                "source_run_path": payload["source_run_path"],
                "source_dense_worker": payload["source_dense_worker"],
                "source_dense_verification": payload["source_dense_verification"],
                "array_document_digest": payload["array_document_digest"],
            }
        )
    if cursor != int(n_rois):
        raise ValueError("Quality partitions do not cover every recording row.")
    if observed_worker_digests != set(expected_workers):
        raise ValueError("Quality partitions do not cover every refined worker.")
    payload = {
        "kind": "ordered_complete_quality_partition_assembly",
        "n_rois": int(n_rois),
        "worker_count": len(worker_records),
        "component_registry": first["component_registry"],
        "quality_profile": first["quality_profile"],
        "quality_policy": first["quality_policy"],
        "producer_commit": str(producer_commit),
        "source_producer_evidence_digest": canonical_json_sha256(evidence),
        "workers": worker_records,
        "workers_digest": canonical_json_sha256(worker_records),
    }
    return {
        "schema_id": SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_subject_mask_quality_partition_assembly(
    assembly: Mapping[str, Any], *, n_rois: int
) -> None:
    canonical = _strict_copy(assembly, name="quality partition assembly")
    payload = canonical.get("payload") if isinstance(canonical, Mapping) else None
    expected_payload_fields = {
        "kind",
        "n_rois",
        "worker_count",
        "component_registry",
        "quality_profile",
        "quality_policy",
        "producer_commit",
        "source_producer_evidence_digest",
        "workers",
        "workers_digest",
    }
    if (
        not isinstance(canonical, Mapping)
        or set(canonical)
        != {
            "schema_id",
            "schema_version",
            "digest_algorithm",
            "payload_digest",
            "payload",
        }
        or canonical.get("schema_id")
        != SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_ID
        or canonical.get("schema_version")
        != SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_VERSION
        or canonical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, Mapping)
        or set(payload) != expected_payload_fields
        or canonical.get("payload_digest") != canonical_json_sha256(payload)
        or payload.get("kind") != "ordered_complete_quality_partition_assembly"
        or payload.get("n_rois") != int(n_rois)
        or payload.get("worker_count") != len(payload.get("workers", []))
        or payload.get("workers_digest")
        != canonical_json_sha256(payload.get("workers"))
        or not _is_sha256(payload.get("source_producer_evidence_digest"))
    ):
        raise ValueError("Quality partition assembly differs.")
    workers = payload["workers"]
    policy = SubjectV1LrObservationQualityPolicy()
    profile = quality_profile_for_policy(policy)
    component_registry = payload.get("component_registry")
    if (
        not workers
        or not str(payload.get("producer_commit") or "").strip()
        or not isinstance(component_registry, Mapping)
        or tuple(component_registry.get("labels", ())) != SUBJECT_V1_LR_COMPONENTS
        or component_registry
        != SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS).as_manifest()
        or payload.get("quality_policy") != policy.as_manifest()
        or payload.get("quality_profile") != profile.as_manifest()
    ):
        raise ValueError("Quality partition assembly is incomplete.")
    cursor = 0
    seen_units: set[tuple[str, int]] = set()
    for worker in workers:
        if not isinstance(worker, Mapping) or set(worker) != {
            "receipt_payload_digest",
            "work_unit",
            "source_run_path",
            "source_dense_worker",
            "source_dense_verification",
            "array_document_digest",
        }:
            raise ValueError("Quality partition assembly worker fields differ.")
        unit = worker.get("work_unit")
        row_interval = (
            unit.get("global_row_interval") if isinstance(unit, Mapping) else None
        )
        unit_identity = (
            (
                unit.get("work_unit_id"),
                unit.get("work_unit_index"),
            )
            if isinstance(unit, Mapping)
            else (None, None)
        )
        if (
            not _is_sha256(worker.get("receipt_payload_digest"))
            or not _is_sha256(worker.get("array_document_digest"))
            or not isinstance(row_interval, Mapping)
            or set(row_interval) != {"start_row", "stop_row"}
            or type(row_interval.get("start_row")) is not int
            or type(row_interval.get("stop_row")) is not int
            or row_interval["start_row"] != cursor
            or row_interval["stop_row"] <= cursor
            or unit_identity in seen_units
            or worker.get("source_dense_verification")
            != {
                "mode": "required_qc_read_compared_to_worker_unit_receipt_v1",
                "status": "passed",
            }
        ):
            raise ValueError("Quality partition assembly coverage differs.")
        seen_units.add(unit_identity)
        cursor = row_interval["stop_row"]
    if cursor != int(n_rois):
        raise ValueError("Quality partition assembly does not cover all rows.")


__all__ = [
    "DEFAULT_PARTITION_COMPUTE_BLOCK_BYTES",
    "DEFAULT_PARTITION_UNIT_ROWS",
    "SUBJECT_MASK_QUALITY_PARTITION_ASSEMBLY_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID",
    "build_subject_mask_quality_partition_assembly",
    "compute_subject_mask_quality_partition",
    "load_subject_mask_quality_partition_arrays",
    "validate_subject_mask_quality_partition",
    "validate_subject_mask_quality_partition_assembly",
]

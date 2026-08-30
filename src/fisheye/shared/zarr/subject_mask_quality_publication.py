"""Bounded selector-ineligible publication for subject-mask quality v1."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import StoragePlan
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, StorageProfile
from fisheye.shared.zarr.subject_mask_quality_manifest import (
    SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE,
    build_subject_mask_quality_run_manifest,
    subject_mask_quality_output_write_units,
    validate_subject_mask_quality_publication,
)
from fisheye.shared.zarr.subject_mask_quality_producer import (
    SUBJECT_V1_LR_COMPONENTS,
    SubjectMaskQualityPayload,
    SubjectV1LrObservationQualityPolicy,
    compute_subject_mask_quality_block,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.subject_mask_quality_partition import (
    validate_subject_mask_quality_partition_assembly,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualityAnySourceReference,
    SubjectMaskQualityComposableSourceReference,
    SubjectMaskQualityDimensions,
    SubjectMaskQualityProfile,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_quality_storage import (
    SubjectMaskQualityStoragePlanSet,
    plan_subject_mask_quality_storage,
)
from fisheye.shared.zarr.subject_mask_final_layout_units import (
    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM,
    SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM,
)
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)

SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_ID = (
    "palette.subject_mask_quality.shadow_publication"
)
SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_VERSION = 1
SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_ID = (
    "palette.subject_mask_quality.write_receipt"
)
SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_VERSION = 4
DEFAULT_SUBJECT_MASK_QUALITY_SHADOW_ROOT = Path(
    "/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/"
    "subject_mask_quality"
)
DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES = 64 * 1024 * 1024
_SELECTOR_ATTRIBUTES = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)
_SOURCE_PATHS = {
    "masks_roi",
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_row_offsets",
    "available_channels",
}


@dataclass(frozen=True)
class SubjectMaskQualityShadowPublication:
    output_path: Path
    run_id: str
    dimensions: SubjectMaskQualityDimensions
    components: SubjectMaskComponentRegistry
    profile: SubjectMaskQualityProfile
    policy: SubjectV1LrObservationQualityPolicy
    source: SubjectMaskQualityAnySourceReference
    source_manifest: Mapping[str, Any]
    plans: SubjectMaskQualityStoragePlanSet
    manifest: Mapping[str, Any]
    write_receipt: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


@dataclass(frozen=True)
class SubjectMaskQualityComposableSourceExpectation:
    """Refined-v5 source identity validated during required quality compute."""

    run_name: str
    manifest_digest: str
    component_registry_digest: str
    source_array_logical_identities: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        run_name = str(self.run_name).strip()
        if not run_name or "/" in run_name:
            raise ValueError("run_name must be one nonempty archive group name.")
        object.__setattr__(self, "run_name", run_name)
        for name in ("manifest_digest", "component_registry_digest"):
            value = str(getattr(self, name)).strip()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
            object.__setattr__(self, name, value)
        identities = {
            str(path): dict(record)
            for path, record in self.source_array_logical_identities.items()
        }
        if set(identities) != _SOURCE_PATHS:
            raise ValueError(
                "Composable quality source must bind every exact refined source array."
            )
        object.__setattr__(self, "source_array_logical_identities", identities)

    def as_source_reference(self) -> SubjectMaskQualityComposableSourceReference:
        return SubjectMaskQualityComposableSourceReference(
            run_name=self.run_name,
            manifest_digest=self.manifest_digest,
            component_registry_digest=self.component_registry_digest,
            source_array_logical_identities=self.source_array_logical_identities,
        )


def _validate_receipt_bound_composable_quality_source(
    *,
    expectation: SubjectMaskQualityComposableSourceExpectation,
    source_manifest: Mapping[str, Any],
    worker_assembly: Mapping[str, Any],
) -> SubjectMaskQualityComposableSourceReference:
    """Join QC worker evidence to the exact published refined logical identity."""

    validate_subject_mask_quality_partition_assembly(
        worker_assembly,
        n_rois=int(
            expectation.source_array_logical_identities["masks_roi"]["shape"][0]
        ),
    )
    payload = source_manifest.get("payload")
    logical = payload.get("logical_content") if isinstance(payload, Mapping) else None
    document = logical.get("document") if isinstance(logical, Mapping) else None
    arrays = document.get("arrays") if isinstance(document, Mapping) else None
    dependencies = (
        payload.get("coordinate_dependencies") if isinstance(payload, Mapping) else None
    )
    dependency_document = (
        dependencies.get("document") if isinstance(dependencies, Mapping) else None
    )
    recording_assembly = (
        dependency_document.get("recording_assembly")
        if isinstance(dependency_document, Mapping)
        else None
    )
    assembly_payload = worker_assembly.get("payload")
    if (
        canonical_json_sha256(source_manifest) != expectation.manifest_digest
        or not isinstance(arrays, Mapping)
        or {
            path: dict(arrays[path])
            for path in expectation.source_array_logical_identities
            if path in arrays
        }
        != {
            path: dict(record)
            for path, record in expectation.source_array_logical_identities.items()
        }
        or not isinstance(recording_assembly, Mapping)
        or not isinstance(assembly_payload, Mapping)
        or recording_assembly.get("producer_evidence_digest")
        != assembly_payload.get("source_producer_evidence_digest")
    ):
        raise ValueError(
            "Quality partitions do not bind the published refined composable identity."
        )
    return expectation.as_source_reference()


class _ComposableMaskIdentityVerifier:
    def __init__(self, record: Mapping[str, Any], *, shape: tuple[int, ...]) -> None:
        expected_fields = {
            "shape",
            "dtype",
            "digest_algorithm",
            "identity_unit_rows",
            "unit_count",
            "units_digest",
            "units",
        }
        units = record.get("units")
        if (
            set(record) != expected_fields
            or record.get("shape") != list(shape)
            or record.get("dtype") != "uint8"
            or record.get("digest_algorithm")
            != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_ALGORITHM
            or record.get("identity_unit_rows")
            != SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
            or not isinstance(units, list)
            or not units
            or record.get("unit_count") != len(units)
            or record.get("units_digest") != canonical_json_sha256(units)
        ):
            raise ValueError("Composable quality mask identity is invalid.")
        cursor = 0
        trailing_values = int(np.prod(shape[1:], dtype=np.int64))
        for index, unit in enumerate(units):
            if not isinstance(unit, Mapping) or set(unit) != {
                "start_row",
                "stop_row",
                "decoded_bytes",
                "sha256",
            }:
                raise ValueError("Composable quality mask unit is invalid.")
            start = int(unit.get("start_row", -1))
            stop = int(unit.get("stop_row", -1))
            expected_stop = min(
                start + SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS,
                int(shape[0]),
            )
            sha256 = str(unit.get("sha256", ""))
            if (
                start != cursor
                or stop != expected_stop
                or int(unit.get("decoded_bytes", -1))
                != (stop - start) * trailing_values * np.dtype(np.uint8).itemsize
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
                or index != start // SUBJECT_MASK_COMPOSABLE_LOGICAL_IDENTITY_UNIT_ROWS
            ):
                raise ValueError("Composable quality mask unit coverage is invalid.")
            cursor = stop
        if cursor != int(shape[0]):
            raise ValueError("Composable quality mask identity is incomplete.")
        self._units = units
        self._cursor = 0
        self._unit_index = 0
        self._digest = hashlib.sha256()

    def append(self, start_row: int, values: np.ndarray[Any, Any]) -> None:
        if int(start_row) != self._cursor:
            raise ValueError("Composable quality mask reads are not contiguous.")
        offset = 0
        while offset < int(values.shape[0]):
            if self._unit_index >= len(self._units):
                raise ValueError("Composable quality mask input exceeds its identity.")
            unit = self._units[self._unit_index]
            stop = int(unit["stop_row"])
            take = min(stop - self._cursor, int(values.shape[0]) - offset)
            part = np.ascontiguousarray(values[offset : offset + take])
            self._digest.update(part.view(np.uint8))
            self._cursor += int(take)
            offset += int(take)
            if self._cursor == stop:
                if self._digest.hexdigest() != unit.get("sha256"):
                    raise ValueError(
                        "Quality source masks differ from the composable core identity."
                    )
                self._unit_index += 1
                self._digest = hashlib.sha256()

    def finish(self) -> None:
        if self._unit_index != len(self._units) or self._cursor != int(
            self._units[-1]["stop_row"]
        ):
            raise ValueError("Composable quality mask identity coverage is incomplete.")


def require_safe_subject_mask_quality_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_SUBJECT_MASK_QUALITY_SHADOW_ROOT,
) -> Path:
    root = shadow_root.expanduser().resolve()
    output = destination.expanduser().resolve()
    if output == root:
        raise ValueError("Subject-mask quality destination cannot equal its root.")
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Subject-mask quality destination must be below shadow_root."
        ) from exc
    if output.exists():
        raise FileExistsError(
            f"Subject-mask quality destination already exists: {output}"
        )
    return output


def require_local_subject_mask_quality_scratch_root(
    scratch_root: Path | None,
) -> Path:
    """Resolve a conventional node-local scratch root and reject shared mounts."""

    candidate = Path(tempfile.gettempdir()) if scratch_root is None else scratch_root
    resolved = candidate.expanduser().resolve()
    allowed_roots = {
        Path(tempfile.gettempdir()).expanduser().resolve(),
        Path("/tmp"),
        Path("/nvme1"),
        Path("/scratch"),
        Path("/lscratch"),
        Path("/local"),
        Path("/dev/shm"),
    }
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(
            "Subject-mask quality scratch_root must be node-local; use /tmp, "
            "$TMPDIR, /nvme1, or a conventional local scratch mount."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _shape_dtype(value: Any, *, name: str) -> tuple[tuple[int, ...], np.dtype[Any]]:
    try:
        shape = tuple(int(item) for item in value.shape)
        dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} lacks exact shape or dtype metadata.") from exc
    return shape, dtype


def _read_rows(value: Any, start: int, stop: int) -> np.ndarray:
    return np.asarray(value[slice(int(start), int(stop))])


def _effective_compute_block_rows(
    *,
    n_channels: int,
    roi_height: int,
    roi_width: int,
    budget_bytes: int,
) -> int:
    budget = int(budget_bytes)
    if budget <= 0:
        raise ValueError("source_compute_block_bytes must be positive.")
    bytes_per_row = int(n_channels) * int(roi_height) * int(roi_width)
    return max(1, budget // max(1, bytes_per_row))


def _write_by_physical_units(
    destination: Any,
    values: Any,
    *,
    plan: StoragePlan,
) -> int:
    unit_shape = plan.shard_shape or plan.chunk_shape
    if unit_shape is None:
        raise ValueError("Subject-mask quality does not support scalar arrays.")
    unit_rows = max(1, int(unit_shape[0]))
    shape, _dtype = _shape_dtype(values, name=plan.array_name)
    trailing = (slice(None),) * (len(shape) - 1)
    writes = 0
    for start in range(0, shape[0], unit_rows):
        stop = min(start + unit_rows, shape[0])
        selection = (slice(start, stop), *trailing)
        destination[selection] = np.asarray(values[selection])
        writes += 1
    return writes


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return result


def subject_mask_quality_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: SubjectMaskQualityStoragePlanSet,
    archive_root_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"subject_mask_quality_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path = metadata_path / relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")

    archive_root = (
        archive_root_metadata
        if archive_root_metadata is not None
        else _read_strict_json(output_path / "zarr.json")
    )
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Subject-mask quality shadow lacks consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Subject-mask quality consolidated envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Subject-mask quality consolidated map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(
                f"Subject-mask quality consolidated metadata lacks {full_path!r}."
            )
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def _create_scratch_arrays(
    scratch_path: Path,
    plans: SubjectMaskQualityStoragePlanSet,
) -> dict[str, np.memmap[Any, Any]]:
    arrays: dict[str, np.memmap[Any, Any]] = {}
    for entry in plans.entries:
        plan = entry.plan
        arrays[entry.rule.path] = np.lib.format.open_memmap(
            scratch_path / f"{entry.rule.path}.npy",
            mode="w+",
            dtype=np.dtype(plan.logical_dtype),
            shape=plan.logical_shape,
        )
    return arrays


def _close_scratch_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    for value in arrays.values():
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            value.flush()
            mmap.close()


def _validate_source_surface(
    source_arrays: Mapping[str, Any],
    *,
    n_frames: int,
    components: SubjectMaskComponentRegistry,
    source: (
        SubjectMaskQualitySourceReference
        | SubjectMaskQualityComposableSourceExpectation
    ),
) -> tuple[SubjectMaskQualityDimensions, np.ndarray]:
    if not _SOURCE_PATHS <= set(source_arrays):
        missing = sorted(_SOURCE_PATHS - set(source_arrays))
        raise ValueError(f"Source mask snapshot lacks arrays: {missing!r}.")
    if tuple(components.labels) != SUBJECT_V1_LR_COMPONENTS:
        raise ValueError("Quality v1 requires canonical subject_v1_lr components.")
    mask_shape, mask_dtype = _shape_dtype(source_arrays["masks_roi"], name="masks_roi")
    if len(mask_shape) != 4 or mask_shape[1] != len(components.labels):
        raise ValueError("Source masks_roi must have shape (N,4,H,W).")
    if mask_dtype != np.dtype(np.uint8):
        raise ValueError("Source masks_roi must be uint8.")
    n_rois, n_channels, height, width = mask_shape
    expected = {
        "instance_key": ((n_rois,), np.dtype(np.uint64)),
        "source_crop_row_ids": ((n_rois,), np.dtype(np.int64)),
        "source_acquisition_frame_index": ((n_rois,), np.dtype(np.int64)),
        "frame_row_offsets": ((int(n_frames) + 1,), np.dtype(np.int64)),
        "available_channels": ((n_channels,), np.dtype(bool)),
    }
    for path, (shape, dtype) in expected.items():
        observed_shape, observed_dtype = _shape_dtype(source_arrays[path], name=path)
        if observed_shape != shape or observed_dtype != dtype:
            raise ValueError(f"Source {path} must be {dtype}{shape}.")
    available = np.asarray(source_arrays["available_channels"][...], dtype=bool)
    source_frames = np.asarray(
        source_arrays["source_acquisition_frame_index"][...], dtype=np.int64
    )
    expected_offsets = derive_subject_mask_frame_row_offsets(
        source_frames,
        n_frames=int(n_frames),
    )
    source_offsets = np.asarray(source_arrays["frame_row_offsets"][...], dtype=np.int64)
    if not np.array_equal(source_offsets, expected_offsets):
        raise ValueError(
            "Source frame_row_offsets do not exactly index "
            "source_acquisition_frame_index."
        )
    if source.component_registry_digest != canonical_json_sha256(
        components.as_manifest()
    ):
        raise ValueError("Source component registry digest mismatch.")
    profile = quality_profile_for_policy(SubjectV1LrObservationQualityPolicy())
    dimensions = SubjectMaskQualityDimensions(
        n_frames=int(n_frames),
        n_rois=n_rois,
        n_channels=n_channels,
        roi_height=height,
        roi_width=width,
        n_component_metrics=len(profile.component_metrics),
        n_observation_metrics=len(profile.observation_metrics),
    )
    return dimensions, available


def _write_receipt(
    *,
    plans: SubjectMaskQualityStoragePlanSet,
    block_rows: int,
    block_bytes_budget: int,
    block_count: int,
    compute_workers_requested: int,
    compute_workers_effective: int,
    source_mode: str,
    worker_assembly: Mapping[str, Any] | None,
    receipt_bound_composable: bool,
) -> dict[str, object]:
    if receipt_bound_composable:
        block_rows = 0
        block_bytes_budget = 0
        block_count = 0
        compute_workers_requested = 0
        compute_workers_effective = 0
    return {
        "schema_id": SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_ID,
        "schema_version": (
            SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_VERSION
            if receipt_bound_composable
            else 3
        ),
        "source_compute_block_rows": int(block_rows),
        "source_compute_block_bytes_budget": int(block_bytes_budget),
        "source_compute_block_count": int(block_count),
        "source_compute_workers_requested": int(compute_workers_requested),
        "source_compute_workers_effective": int(compute_workers_effective),
        "source_compute_execution": (
            "receipt_bound_partitions_with_verified_worker_units_v2"
            if receipt_bound_composable
            else (
                "receipt_bound_partitions_with_ordered_source_verification_v1"
                if source_mode == "receipt_bound_quality_partitions"
                else (
                    "ordered_inline_single_worker_v1"
                    if int(compute_workers_effective) == 1
                    else "bounded_thread_pool_ordered_single_writer_v1"
                )
            )
        ),
        "source_mode": source_mode,
        "worker_assembly": (
            dict(worker_assembly) if worker_assembly is not None else None
        ),
        "output_write_unit": "complete_outer_shard_or_unsharded_chunk",
        "output_array_write_units": subject_mask_quality_output_write_units(plans),
        "scratch_surface": "node_local_npy_memmap_deleted_after_publication",
        "parallel_write_policy": (
            "single_writer_v1_future_workers_require_disjoint_whole_shards"
        ),
    }


def validate_subject_mask_quality_shadow_publication(
    publication: SubjectMaskQualityShadowPublication,
) -> tuple[str, ...]:
    try:
        direct, consolidated = subject_mask_quality_metadata_declaration_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
        run = zarr.open_group(
            str(
                publication.output_path
                / "subject_mask_quality_runs"
                / publication.run_id
            ),
            mode="r",
            use_consolidated=False,
        )
        arrays = {
            path: run[path] for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
        }
    except (OSError, TypeError, ValueError) as exc:
        return (f"subject-mask quality shadow reopen failed: {exc}",)
    errors = list(
        validate_subject_mask_quality_publication(
            publication.manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
            source_manifest=publication.source_manifest,
        )
    )
    if run.attrs.get("status") != "complete":
        errors.append("subject-mask quality shadow status is not complete")
    if run.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        errors.append("subject-mask quality completion contract is absent")
    if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "complete":
        errors.append("subject-mask quality completion marker is not complete")
    if run.attrs.get("stage_selector_eligible") is not False:
        errors.append("subject-mask quality shadow is selector eligible")
    if run.attrs.get(SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE) != dict(
        publication.manifest
    ):
        errors.append("subject-mask quality persisted manifest differs")
    family = zarr.open_group(
        str(publication.output_path / "subject_mask_quality_runs"),
        mode="r",
        use_consolidated=False,
    )
    selected = [
        name
        for name in _SELECTOR_ATTRIBUTES
        if family.attrs.get(name) == publication.run_id
    ]
    if selected:
        errors.append(f"subject-mask quality shadow is selected by {selected!r}")
    return tuple(errors)


def publish_selector_ineligible_subject_mask_quality_snapshot(
    source_mask_arrays: Mapping[str, Any],
    *,
    n_frames: int,
    components: SubjectMaskComponentRegistry,
    source: (
        SubjectMaskQualitySourceReference
        | SubjectMaskQualityComposableSourceExpectation
    ),
    source_manifest: Mapping[str, Any],
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_SUBJECT_MASK_QUALITY_SHADOW_ROOT,
    scratch_root: Path | None = None,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    policy: SubjectV1LrObservationQualityPolicy = (
        SubjectV1LrObservationQualityPolicy()
    ),
    source_compute_block_bytes: int = DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    compute_workers: int = 1,
    precomputed_arrays: Mapping[str, Any] | None = None,
    worker_assembly: Mapping[str, Any] | None = None,
    created_by: str = "subject_mask_quality_shadow",
) -> SubjectMaskQualityShadowPublication:
    """Compute, write, consolidate, and gate one immutable QC snapshot."""

    output_path = require_safe_subject_mask_quality_shadow_destination(
        destination, shadow_root=shadow_root
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
    if type(compute_workers) is not int or compute_workers <= 0:
        raise ValueError("compute_workers must be a positive integer.")
    adopting_partitions = precomputed_arrays is not None or worker_assembly is not None
    if (precomputed_arrays is None) != (worker_assembly is None):
        raise ValueError(
            "precomputed_arrays and worker_assembly must be supplied together."
        )
    resolved_run_id = str(run_id).strip()
    if not resolved_run_id or "/" in resolved_run_id:
        raise ValueError("run_id must be one nonempty archive group name.")
    if canonical_json_sha256(source_manifest) != source.manifest_digest:
        raise ValueError("Source mask manifest differs from the bound digest.")

    base_dimensions, available = _validate_source_surface(
        source_mask_arrays,
        n_frames=int(n_frames),
        components=components,
        source=source,
    )
    composable_expectation = (
        source
        if isinstance(source, SubjectMaskQualityComposableSourceExpectation)
        else None
    )
    profile = quality_profile_for_policy(policy)
    dimensions = SubjectMaskQualityDimensions(
        n_frames=base_dimensions.n_frames,
        n_rois=base_dimensions.n_rois,
        n_channels=base_dimensions.n_channels,
        roi_height=base_dimensions.roi_height,
        roi_width=base_dimensions.roi_width,
        n_component_metrics=len(profile.component_metrics),
        n_observation_metrics=len(profile.observation_metrics),
    )
    plans = plan_subject_mask_quality_storage(dimensions, profile=storage_profile)
    if worker_assembly is not None:
        validate_subject_mask_quality_partition_assembly(
            worker_assembly, n_rois=dimensions.n_rois
        )
        expected_precomputed = set(SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths) - {
            "frame_row_offsets"
        }
        if set(precomputed_arrays or {}) != expected_precomputed:
            raise ValueError("Precomputed quality partition arrays are incomplete.")
    receipt_bound_composable = bool(
        adopting_partitions and composable_expectation is not None
    )
    receipt_bound_source = (
        _validate_receipt_bound_composable_quality_source(
            expectation=composable_expectation,
            source_manifest=source_manifest,
            worker_assembly=worker_assembly,
        )
        if receipt_bound_composable
        and composable_expectation is not None
        and worker_assembly is not None
        else None
    )
    block_rows = _effective_compute_block_rows(
        n_channels=dimensions.n_channels,
        roi_height=dimensions.roi_height,
        roi_width=dimensions.roi_width,
        budget_bytes=int(source_compute_block_bytes),
    )
    block_count = max(1, math.ceil(dimensions.n_rois / block_rows))
    effective_compute_workers = (
        1 if adopting_partitions else min(int(compute_workers), int(block_count))
    )
    receipt = _write_receipt(
        plans=plans,
        block_rows=block_rows,
        block_bytes_budget=int(source_compute_block_bytes),
        block_count=block_count,
        compute_workers_requested=int(compute_workers),
        compute_workers_effective=effective_compute_workers,
        source_mode=(
            "receipt_bound_quality_partitions"
            if adopting_partitions
            else "inline_dense_quality_compute"
        ),
        worker_assembly=worker_assembly,
        receipt_bound_composable=receipt_bound_composable,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_scratch_root = require_local_subject_mask_quality_scratch_root(
        scratch_root
    )
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    with tempfile.TemporaryDirectory(
        prefix="palette_subject_mask_quality_", dir=str(resolved_scratch_root)
    ) as temporary:
        scratch_path = Path(temporary)
        scratch_arrays = _create_scratch_arrays(scratch_path, plans)
        source_digest = hashlib.sha256()
        composable_mask_verifier = (
            _ComposableMaskIdentityVerifier(
                composable_expectation.source_array_logical_identities["masks_roi"],
                shape=tuple(
                    int(value) for value in source_mask_arrays["masks_roi"].shape
                ),
            )
            if composable_expectation is not None and not receipt_bound_composable
            else None
        )
        identity_digests = {
            path: hashlib.sha256()
            for path in (
                "instance_key",
                "source_crop_row_ids",
                "source_acquisition_frame_index",
            )
        }
        phase_started = time.perf_counter()
        pending: deque[
            tuple[
                int,
                int,
                np.ndarray,
                np.ndarray,
                Future[SubjectMaskQualityPayload],
            ]
        ] = deque()

        def write_payload(
            start: int,
            stop: int,
            keys: np.ndarray,
            frames: np.ndarray,
            payload: SubjectMaskQualityPayload,
        ) -> None:
            scratch_arrays["instance_key"][start:stop] = keys
            scratch_arrays["source_mask_row_ids"][start:stop] = np.arange(
                start, stop, dtype=np.int64
            )
            scratch_arrays["source_acquisition_frame_index"][start:stop] = frames
            for path, values in payload.as_arrays().items():
                scratch_arrays[path][start:stop] = values

        def process_blocks(executor: ThreadPoolExecutor | None) -> None:
            for start in range(0, dimensions.n_rois, block_rows):
                stop = min(start + block_rows, dimensions.n_rois)
                masks = np.ascontiguousarray(
                    _read_rows(source_mask_arrays["masks_roi"], start, stop)
                )
                source_digest.update(masks.view(np.uint8))
                if composable_mask_verifier is not None:
                    composable_mask_verifier.append(start, masks)
                keys = _read_rows(source_mask_arrays["instance_key"], start, stop)
                source_crop_row_ids = _read_rows(
                    source_mask_arrays["source_crop_row_ids"], start, stop
                )
                frames = _read_rows(
                    source_mask_arrays["source_acquisition_frame_index"], start, stop
                )
                for path, values in (
                    ("instance_key", keys),
                    ("source_crop_row_ids", source_crop_row_ids),
                    ("source_acquisition_frame_index", frames),
                ):
                    identity_digests[path].update(
                        np.ascontiguousarray(values).view(np.uint8)
                    )
                if adopting_partitions:
                    continue
                if executor is None:
                    payload = compute_subject_mask_quality_block(
                        masks,
                        available_channels=available,
                        components=components,
                        policy=policy,
                    )
                    write_payload(start, stop, keys, frames, payload)
                    continue
                future = executor.submit(
                    compute_subject_mask_quality_block,
                    masks,
                    available_channels=available,
                    components=components,
                    policy=policy,
                )
                pending.append((start, stop, keys, frames, future))
                if len(pending) >= effective_compute_workers:
                    (
                        pending_start,
                        pending_stop,
                        pending_keys,
                        pending_frames,
                        pending_future,
                    ) = pending.popleft()
                    write_payload(
                        pending_start,
                        pending_stop,
                        pending_keys,
                        pending_frames,
                        pending_future.result(),
                    )
            while pending:
                (
                    pending_start,
                    pending_stop,
                    pending_keys,
                    pending_frames,
                    pending_future,
                ) = pending.popleft()
                write_payload(
                    pending_start,
                    pending_stop,
                    pending_keys,
                    pending_frames,
                    pending_future.result(),
                )

        if receipt_bound_composable:
            assert precomputed_arrays is not None
        elif effective_compute_workers == 1:
            process_blocks(None)
        else:
            with ThreadPoolExecutor(
                max_workers=effective_compute_workers,
                thread_name_prefix="subject-mask-quality",
            ) as compute_executor:
                process_blocks(compute_executor)
        if receipt_bound_composable:
            for path in identity_digests:
                values = np.ascontiguousarray(source_mask_arrays[path][...])
                identity_digests[path].update(values.view(np.uint8))
        observed_source_digests = {
            **(
                {}
                if receipt_bound_composable
                else {"masks_roi": source_digest.hexdigest()}
            ),
            **{path: digest.hexdigest() for path, digest in identity_digests.items()},
            "frame_row_offsets": hashlib.sha256(
                np.ascontiguousarray(source_mask_arrays["frame_row_offsets"][...]).view(
                    np.uint8
                )
            ).hexdigest(),
            "available_channels": hashlib.sha256(
                np.ascontiguousarray(available).view(np.uint8)
            ).hexdigest(),
        }
        if composable_mask_verifier is not None:
            composable_mask_verifier.finish()
        if composable_expectation is None and observed_source_digests != dict(
            source.source_array_values_sha256
        ):
            mismatches = sorted(
                path
                for path, observed in observed_source_digests.items()
                if source.source_array_values_sha256.get(path) != observed
            )
            raise ValueError(
                "Exact refined source-array digest mismatch for: "
                + ", ".join(mismatches)
            )
        if composable_expectation is not None:
            mismatches: list[str] = []
            for path, observed in observed_source_digests.items():
                if path == "masks_roi":
                    continue
                expected = composable_expectation.source_array_logical_identities[path]
                observed_shape, observed_dtype = _shape_dtype(
                    source_mask_arrays[path], name=path
                )
                if (
                    set(expected)
                    != {
                        "shape",
                        "dtype",
                        "digest_algorithm",
                        "sha256",
                    }
                    or expected.get("shape") != list(observed_shape)
                    or expected.get("dtype") != str(observed_dtype)
                    or expected.get("digest_algorithm")
                    != SUBJECT_MASK_ARRAY_DIGEST_ALGORITHM
                    or expected.get("sha256") != observed
                ):
                    mismatches.append(path)
            if mismatches:
                raise ValueError(
                    "Exact refined composable source-array mismatch for: "
                    + ", ".join(sorted(mismatches))
                )
            resolved_source: SubjectMaskQualityAnySourceReference = (
                receipt_bound_source
                if receipt_bound_source is not None
                else SubjectMaskQualitySourceReference(
                    run_name=composable_expectation.run_name,
                    manifest_digest=composable_expectation.manifest_digest,
                    dense_array_values_sha256=observed_source_digests["masks_roi"],
                    component_registry_digest=(
                        composable_expectation.component_registry_digest
                    ),
                    source_array_values_sha256=observed_source_digests,
                )
            )
        else:
            assert isinstance(source, SubjectMaskQualitySourceReference)
            resolved_source = source
        source_verification_seconds = time.perf_counter() - phase_started
        if adopting_partitions:
            phase_seconds[
                (
                    "receipt_bound_source_verification"
                    if receipt_bound_composable
                    else "ordered_source_verification"
                )
            ] = source_verification_seconds
        if adopting_partitions:
            assert precomputed_arrays is not None
            phase_started = time.perf_counter()
            for path in sorted(precomputed_arrays):
                source_value = precomputed_arrays[path]
                destination_value = scratch_arrays[path]
                source_shape, source_dtype = _shape_dtype(source_value, name=path)
                destination_shape, destination_dtype = _shape_dtype(
                    destination_value, name=path
                )
                if (
                    source_shape != destination_shape
                    or source_dtype != destination_dtype
                ):
                    raise ValueError(
                        f"Precomputed quality partition array differs at {path}."
                    )
                destination_value[...] = np.asarray(source_value[...])
            phase_seconds["receipt_bound_partition_adoption"] = (
                time.perf_counter() - phase_started
            )
        frames = scratch_arrays["source_acquisition_frame_index"]
        scratch_arrays["frame_row_offsets"][...] = (
            derive_subject_mask_frame_row_offsets(frames, n_frames=dimensions.n_frames)
        )
        phase_seconds["bounded_compute_to_scratch"] = (
            source_verification_seconds
            + phase_seconds.get("receipt_bound_partition_adoption", 0.0)
            if adopting_partitions
            else time.perf_counter() - phase_started
        )

        source_evidence = {
            "instance_key": scratch_arrays["instance_key"],
            "source_acquisition_frame_index": scratch_arrays[
                "source_acquisition_frame_index"
            ],
            "available_channels": available,
        }
        SUBJECT_MASK_QUALITY_SCHEMA_V1.require(
            scratch_arrays,
            dimensions=dimensions,
            components=components,
            profile=profile,
            source_mask_arrays=source_evidence,
        )

        root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
        root.attrs.update(
            {
                "benchmark_only": True,
                "canonical": False,
                "registry_registered": False,
                "selector_eligible": False,
                "schema_id": SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_ID,
                "schema_version": SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "created_by": str(created_by),
            }
        )
        family = require_runs_parent(
            root,
            "subject_mask_quality_runs",
            completion_epoch=COMPLETION_EPOCH_STRICT,
        )
        family.attrs.update(
            {
                "benchmark_only": True,
                "selector_eligible": False,
                "selection_contract": "none_shadow_direct_path_only",
            }
        )
        run = family.create_group(resolved_run_id)
        mark_run_started(
            run,
            run_name=resolved_run_id,
            stage="subject_mask_quality",
        )
        run.attrs.update(
            {
                "status": "running",
                "stage_selector_eligible": False,
                "shadow_only": True,
                "artifact_class": "observation_local_quality_diagnostics",
                "logical_schema": SUBJECT_MASK_QUALITY_SCHEMA_V1.as_manifest(
                    dimensions=dimensions,
                    components=components,
                    profile=profile,
                    source=resolved_source,
                ),
                "storage_plan": plans.as_manifest(),
                "source_refined_subject_mask_snapshot": resolved_source.as_manifest(),
                "policy": policy.as_manifest(),
                "write_receipt": receipt,
            }
        )

        destination_arrays: dict[str, Any] = {}
        bindings = {
            binding.path: binding for binding in SUBJECT_MASK_QUALITY_SCHEMA_V1.bindings
        }
        phase_started = time.perf_counter()
        physical_write_counts: dict[str, int] = {}
        for entry in plans.entries:
            path = entry.rule.path
            binding = bindings[path]
            contract = SUBJECT_MASK_QUALITY_SCHEMA_V1.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            destination_array = create_array_from_plan(
                run,
                name=path,
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={
                    "benchmark_only": True,
                    "selector_eligible": False,
                    "artifact_class": "observation_local_quality_diagnostics",
                },
            )
            physical_write_counts[path] = _write_by_physical_units(
                destination_array,
                scratch_arrays[path],
                plan=entry.plan,
            )
            destination_arrays[path] = destination_array
        phase_seconds["physical_unit_publication"] = time.perf_counter() - phase_started
        run.attrs["physical_write_counts"] = physical_write_counts

        phase_started = time.perf_counter()
        consolidate_metadata_capture_expected_warnings(output_path)
        direct, consolidated = subject_mask_quality_metadata_declaration_maps(
            output_path,
            run_id=resolved_run_id,
            plans=plans,
        )
        phase_seconds["first_consolidation"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        manifest = build_subject_mask_quality_run_manifest(
            run_id=resolved_run_id,
            dimensions=dimensions,
            components=components,
            profile=profile,
            policy=policy,
            source=resolved_source,
            source_manifest=source_manifest,
            storage_plan=plans,
            arrays=scratch_arrays,
            source_arrays=source_evidence,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            write_receipt=receipt,
        )
        run.attrs[SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE] = manifest
        phase_seconds["build_manifest"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        consolidate_metadata_capture_expected_warnings(output_path)
        direct, consolidated = subject_mask_quality_metadata_declaration_maps(
            output_path,
            run_id=resolved_run_id,
            plans=plans,
        )
        errors = validate_subject_mask_quality_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
            source_manifest=source_manifest,
        )
        phase_seconds["final_consolidation_and_gate"] = (
            time.perf_counter() - phase_started
        )
        if errors:
            run.attrs.update(
                {
                    "status": "failed",
                    "stage_selector_eligible": False,
                    "publication_errors": list(errors),
                }
            )
            mark_run_failed(
                run,
                run_name=resolved_run_id,
                error="; ".join(errors),
            )
            raise RuntimeError(
                "Subject-mask quality publication failed: " + "; ".join(errors)
            )

        # Seal only after the complete logical, physical, direct/consolidated,
        # source-binding, and manifest gates above have passed.  Lifecycle
        # fields are deliberately outside the scientific metadata digest.
        run.attrs["status"] = "complete"
        mark_run_complete(run, run_name=resolved_run_id)
        consolidate_metadata_capture_expected_warnings(output_path)
        final_direct, final_consolidated = (
            subject_mask_quality_metadata_declaration_maps(
                output_path,
                run_id=resolved_run_id,
                plans=plans,
            )
        )
        final_errors = validate_subject_mask_quality_publication(
            manifest,
            direct_metadata_declarations=final_direct,
            consolidated_metadata_declarations=final_consolidated,
            arrays=destination_arrays,
            source_manifest=source_manifest,
        )
        if final_errors:
            mark_run_failed(
                run,
                run_name=resolved_run_id,
                error="; ".join(final_errors),
            )
            consolidate_metadata_capture_expected_warnings(output_path)
            raise RuntimeError(
                "Subject-mask quality completion seal failed: "
                + "; ".join(final_errors)
            )

        publication = SubjectMaskQualityShadowPublication(
            output_path=output_path,
            run_id=resolved_run_id,
            dimensions=dimensions,
            components=components,
            profile=profile,
            policy=policy,
            source=resolved_source,
            source_manifest=dict(source_manifest),
            plans=plans,
            manifest=manifest,
            write_receipt=receipt,
            phase_seconds=phase_seconds,
            elapsed_seconds=time.perf_counter() - started,
        )
        reopen_errors = validate_subject_mask_quality_shadow_publication(publication)
        if reopen_errors:
            run.attrs.update(
                {
                    "status": "failed",
                    "stage_selector_eligible": False,
                    "publication_errors": list(reopen_errors),
                }
            )
            mark_run_failed(
                run,
                run_name=resolved_run_id,
                error="; ".join(reopen_errors),
            )
            consolidate_metadata_capture_expected_warnings(output_path)
            raise RuntimeError(
                "Reopened subject-mask quality publication failed: "
                + "; ".join(reopen_errors)
            )
        _close_scratch_arrays(scratch_arrays)
    return publication


__all__ = [
    "DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES",
    "DEFAULT_SUBJECT_MASK_QUALITY_SHADOW_ROOT",
    "SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_SHADOW_SCHEMA_VERSION",
    "SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_VERSION",
    "SubjectMaskQualityComposableSourceExpectation",
    "SubjectMaskQualityShadowPublication",
    "publish_selector_ineligible_subject_mask_quality_snapshot",
    "require_local_subject_mask_quality_scratch_root",
    "require_safe_subject_mask_quality_shadow_destination",
    "subject_mask_quality_metadata_declaration_maps",
    "validate_subject_mask_quality_shadow_publication",
]

"""Bounded selector-ineligible publication for subject-mask quality v1."""

from __future__ import annotations

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
    SubjectV1LrObservationQualityPolicy,
    compute_subject_mask_quality_block,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualityDimensions,
    SubjectMaskQualityProfile,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_quality_storage import (
    SubjectMaskQualityStoragePlanSet,
    plan_subject_mask_quality_storage,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
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
SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_VERSION = 1
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
    "source_acquisition_frame_index",
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
    source: SubjectMaskQualitySourceReference
    source_manifest: Mapping[str, Any]
    plans: SubjectMaskQualityStoragePlanSet
    manifest: Mapping[str, Any]
    write_receipt: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


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
    if not any(
        resolved == root or root in resolved.parents for root in allowed_roots
    ):
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
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"subject_mask_quality_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path = metadata_path / relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")

    archive_root = _read_strict_json(output_path / "zarr.json")
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
    source: SubjectMaskQualitySourceReference,
) -> tuple[SubjectMaskQualityDimensions, np.ndarray]:
    if not _SOURCE_PATHS <= set(source_arrays):
        missing = sorted(_SOURCE_PATHS - set(source_arrays))
        raise ValueError(f"Source mask snapshot lacks arrays: {missing!r}.")
    if tuple(components.labels) != SUBJECT_V1_LR_COMPONENTS:
        raise ValueError("Quality v1 requires canonical subject_v1_lr components.")
    mask_shape, mask_dtype = _shape_dtype(
        source_arrays["masks_roi"], name="masks_roi"
    )
    if len(mask_shape) != 4 or mask_shape[1] != len(components.labels):
        raise ValueError("Source masks_roi must have shape (N,4,H,W).")
    if mask_dtype != np.dtype(np.uint8):
        raise ValueError("Source masks_roi must be uint8.")
    n_rois, n_channels, height, width = mask_shape
    expected = {
        "instance_key": ((n_rois,), np.dtype(np.uint64)),
        "source_acquisition_frame_index": ((n_rois,), np.dtype(np.int64)),
        "available_channels": ((n_channels,), np.dtype(bool)),
    }
    for path, (shape, dtype) in expected.items():
        observed_shape, observed_dtype = _shape_dtype(source_arrays[path], name=path)
        if observed_shape != shape or observed_dtype != dtype:
            raise ValueError(f"Source {path} must be {dtype}{shape}.")
    available = np.asarray(source_arrays["available_channels"][...], dtype=bool)
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
) -> dict[str, object]:
    return {
        "schema_id": SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_QUALITY_WRITE_RECEIPT_SCHEMA_VERSION,
        "source_compute_block_rows": int(block_rows),
        "source_compute_block_bytes_budget": int(block_bytes_budget),
        "source_compute_block_count": int(block_count),
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
    source: SubjectMaskQualitySourceReference,
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
    created_by: str = "subject_mask_quality_shadow",
) -> SubjectMaskQualityShadowPublication:
    """Compute, write, consolidate, and gate one immutable QC snapshot."""

    output_path = require_safe_subject_mask_quality_shadow_destination(
        destination, shadow_root=shadow_root
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
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
    plans = plan_subject_mask_quality_storage(
        dimensions, profile=storage_profile
    )
    block_rows = _effective_compute_block_rows(
        n_channels=dimensions.n_channels,
        roi_height=dimensions.roi_height,
        roi_width=dimensions.roi_width,
        budget_bytes=int(source_compute_block_bytes),
    )
    block_count = max(1, math.ceil(dimensions.n_rois / block_rows))
    receipt = _write_receipt(
        plans=plans,
        block_rows=block_rows,
        block_bytes_budget=int(source_compute_block_bytes),
        block_count=block_count,
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
        phase_started = time.perf_counter()
        for start in range(0, dimensions.n_rois, block_rows):
            stop = min(start + block_rows, dimensions.n_rois)
            masks = np.ascontiguousarray(
                _read_rows(source_mask_arrays["masks_roi"], start, stop)
            )
            source_digest.update(masks.view(np.uint8))
            keys = _read_rows(source_mask_arrays["instance_key"], start, stop)
            frames = _read_rows(
                source_mask_arrays["source_acquisition_frame_index"], start, stop
            )
            payload = compute_subject_mask_quality_block(
                masks,
                available_channels=available,
                components=components,
                policy=policy,
            )
            scratch_arrays["instance_key"][start:stop] = keys
            scratch_arrays["source_mask_row_ids"][start:stop] = np.arange(
                start, stop, dtype=np.int64
            )
            scratch_arrays["source_acquisition_frame_index"][start:stop] = frames
            for path, values in payload.as_arrays().items():
                scratch_arrays[path][start:stop] = values
        if source_digest.hexdigest() != source.dense_array_values_sha256:
            raise ValueError("Decoded dense source-mask digest mismatch.")
        frames = scratch_arrays["source_acquisition_frame_index"]
        scratch_arrays["frame_row_offsets"][...] = (
            derive_subject_mask_frame_row_offsets(
                frames, n_frames=dimensions.n_frames
            )
        )
        phase_seconds["bounded_compute_to_scratch"] = (
            time.perf_counter() - phase_started
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
        family = root.create_group("subject_mask_quality_runs")
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
                    source=source,
                ),
                "storage_plan": plans.as_manifest(),
                "source_refined_subject_mask_snapshot": source.as_manifest(),
                "policy": policy.as_manifest(),
                "write_receipt": receipt,
            }
        )

        destination_arrays: dict[str, Any] = {}
        bindings = {
            binding.path: binding
            for binding in SUBJECT_MASK_QUALITY_SCHEMA_V1.bindings
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
        phase_seconds["physical_unit_publication"] = (
            time.perf_counter() - phase_started
        )
        run.attrs["physical_write_counts"] = physical_write_counts
        run.attrs["status"] = "complete"
        mark_run_complete(run, run_name=resolved_run_id)

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
            source=source,
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

        publication = SubjectMaskQualityShadowPublication(
            output_path=output_path,
            run_id=resolved_run_id,
            dimensions=dimensions,
            components=components,
            profile=profile,
            policy=policy,
            source=source,
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
    "SubjectMaskQualityShadowPublication",
    "publish_selector_ineligible_subject_mask_quality_snapshot",
    "require_local_subject_mask_quality_scratch_root",
    "require_safe_subject_mask_quality_shadow_destination",
    "subject_mask_quality_metadata_declaration_maps",
    "validate_subject_mask_quality_shadow_publication",
]

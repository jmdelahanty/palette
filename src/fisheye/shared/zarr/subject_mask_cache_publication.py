"""Bounded publication for immutable recording-level subject-mask caches.

The current publisher normally assembles receipt-bound fixed-count contours
that refinement workers computed beside their final dense rows.  Explicit
repair and compatibility calls may still regenerate them from complete dense
``masks_roi``.  Both paths write the same access-aware Zarr v3 contract and
remain selector-ineligible until the enclosing bundle is activated.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.refined_subject_component_contours import (
    DEFAULT_BOUNDARY_POLICY,
    DEFAULT_CONTOUR_COORDINATE_SPACE,
    DEFAULT_CONTOUR_METHOD,
    DEFAULT_CONTOUR_METHOD_VERSION,
    SAMPLED_COMPONENT_CONTOUR_CANONICALIZATION,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_ID,
    SAMPLED_COMPONENT_CONTOUR_PUBLICATION_PROFILE_VERSION,
    extract_largest_external_contour,
    sample_contours_fixed_k,
)
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.refined_subject_mask_extensions import (
    SubjectMaskDerivedCacheKind,
    SubjectMaskDerivedCacheReceipt,
    SubjectMaskSampledContourProfile,
    default_subject_mask_sampled_contour_profile,
    published_subject_mask_cache_extension_manifest,
    validate_published_subject_mask_cache_extension,
)
from fisheye.shared.zarr.storage_profiles import (
    SUBJECT_MASK_PRESENTATION_CANDIDATE_V1,
    StorageProfile,
    storage_profile_from_manifest,
)
from fisheye.shared.zarr.subject_mask_cache_storage import (
    SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND,
    SubjectMaskSampledContourStoragePlanSet,
    plan_subject_mask_sampled_contour_storage,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    validate_persisted_subject_mask_core_publication,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_schema import (
    REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)
from fisheye.shared.zarr.subject_mask_sampled_contour_worker_receipt import (
    validate_subject_mask_sampled_contour_worker_assembly,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)

SUBJECT_MASK_CACHE_FAMILY = "subject_mask_cache_runs"
SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_ID = (
    "palette.subject_mask.derived_cache_run_manifest"
)
SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_VERSION = 2
SUBJECT_MASK_CACHE_RUN_MANIFEST_SUPPORTED_VERSIONS = (1, 2)
SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_ID = (
    "palette.subject_mask.derived_cache_publication"
)
SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_VERSION = 1
SUBJECT_MASK_CACHE_METADATA_DIGEST_SCOPE = (
    "exact_run_subtree_declarations_redacting_manifest_and_lifecycle_v1"
)
SUBJECT_MASK_CACHE_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_logical_values_v1"
SUBJECT_MASK_CACHE_GENERATOR_ID = "palette_subject_mask_sampled_contours"
SUBJECT_MASK_CACHE_GENERATOR_VERSION = 1
DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_PATHS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_row_offsets",
    "source_crop_xywh",
)
_LIFECYCLE_METADATA_ATTRS = {
    SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE,
    "status",
    RUN_COMPLETION_STATUS_ATTR,
    RUN_COMPLETED_AT_ATTR,
    "palette_run_failed_at_utc",
    "palette_run_error",
    "atomic_publication_owner_uuid",
    "atomic_publication_tombstone",
    "cluster_output_staging",
    "publication_status",
    "subject_mask_bundle_selector_eligible",
}


@dataclass(frozen=True)
class SubjectMaskCachePublication:
    output_path: Path
    run_id: str
    dimensions: SubjectMaskDimensions
    components: SubjectMaskComponentRegistry
    contour_profile: SubjectMaskSampledContourProfile
    plans: SubjectMaskSampledContourStoragePlanSet
    manifest: Mapping[str, Any]
    phase_seconds: Mapping[str, float]
    elapsed_seconds: float


def _compute_sampled_contour_block(
    source_root: str,
    refined_run_id: str,
    start: int,
    stop: int,
    component_labels: Sequence[str],
    sample_counts: Mapping[str, int],
) -> tuple[int, int, dict[str, np.ndarray]]:
    """Compute one independent dense-row block without writing Zarr."""

    import cv2

    # One Python process owns one CPU worker slot.  OpenCV otherwise creates a
    # host-sized native pool in every process, causing severe oversubscription
    # under LSF even though findContours itself does not benefit from it.
    cv2.setNumThreads(1)
    run = zarr.open_group(
        str(Path(source_root) / "refined_subject_masks_runs" / str(refined_run_id)),
        mode="r",
        use_consolidated=False,
    )
    masks = np.asarray(run["masks_roi"][int(start) : int(stop)])
    payload: dict[str, np.ndarray] = {}
    for component_index, component in enumerate(component_labels):
        contours = [
            extract_largest_external_contour(mask) for mask in masks[:, component_index]
        ]
        points, valid, source_counts = sample_contours_fixed_k(
            contours,
            sample_count=int(sample_counts[component]),
            min_points=2,
        )
        prefix = f"components/{component}/sampled_contours"
        payload[f"{prefix}/points_xy"] = points
        payload[f"{prefix}/valid"] = valid
        payload[f"{prefix}/source_point_count"] = source_counts
    return int(start), int(stop), payload


def _store_sampled_contour_block(
    memmaps: Mapping[str, np.memmap],
    result: tuple[int, int, Mapping[str, np.ndarray]],
) -> None:
    start, stop, payload = result
    for path, values in payload.items():
        memmaps[path][start:stop] = values


def _generate_sampled_contours(
    *,
    source_root: Path,
    source_run: Any,
    refined_run_id: str,
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    contour_profile: SubjectMaskSampledContourProfile,
    block_rows: int,
    compute_workers: int,
    memmaps: Mapping[str, np.memmap],
) -> int:
    ranges = tuple(
        (start, min(dimensions.n_rois, start + int(block_rows)))
        for start in range(0, dimensions.n_rois, int(block_rows))
    )
    effective_workers = min(max(1, int(compute_workers)), max(1, len(ranges)))
    sample_counts = {
        component: int(contour_profile.sample_counts[component])
        for component in components.labels
    }
    if effective_workers == 1:
        import cv2

        previous_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        masks_roi = source_run["masks_roi"]
        try:
            for start, stop in ranges:
                masks = np.asarray(masks_roi[start:stop])
                payload: dict[str, np.ndarray] = {}
                for component_index, component in enumerate(components.labels):
                    contours = [
                        extract_largest_external_contour(mask)
                        for mask in masks[:, component_index]
                    ]
                    points, valid, source_counts = sample_contours_fixed_k(
                        contours,
                        sample_count=sample_counts[component],
                        min_points=2,
                    )
                    prefix = f"components/{component}/sampled_contours"
                    payload[f"{prefix}/points_xy"] = points
                    payload[f"{prefix}/valid"] = valid
                    payload[f"{prefix}/source_point_count"] = source_counts
                _store_sampled_contour_block(memmaps, (start, stop, payload))
        finally:
            cv2.setNumThreads(previous_cv2_threads)
        return effective_workers

    range_iterator = iter(ranges)
    max_pending = effective_workers * 2
    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        pending = set()
        for _ in range(min(max_pending, len(ranges))):
            start, stop = next(range_iterator)
            pending.add(
                pool.submit(
                    _compute_sampled_contour_block,
                    str(source_root),
                    refined_run_id,
                    start,
                    stop,
                    components.labels,
                    sample_counts,
                )
            )
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                _store_sampled_contour_block(memmaps, future.result())
                try:
                    start, stop = next(range_iterator)
                except StopIteration:
                    continue
                pending.add(
                    pool.submit(
                        _compute_sampled_contour_block,
                        str(source_root),
                        refined_run_id,
                        start,
                        stop,
                        components.labels,
                        sample_counts,
                    )
                )
    return effective_workers


def _safe_run_id(value: object, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved or "/" in resolved or resolved in {".", ".."}:
        raise ValueError(f"{name} must be one safe nonempty run id.")
    return resolved


def _require_sha256(value: object, *, name: str) -> str:
    resolved = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(resolved):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return resolved


def _dimensions_from_manifest(value: object) -> SubjectMaskDimensions:
    if not isinstance(value, Mapping):
        raise TypeError("Subject-mask cache dimensions must be an object.")
    dimensions = SubjectMaskDimensions(
        n_frames=value.get("n_frames"),
        n_rois=value.get("n_rois"),
        n_channels=value.get("n_channels"),
        roi_height=value.get("roi_height"),
        roi_width=value.get("roi_width"),
    )
    if dict(value) != dimensions.as_manifest():
        raise ValueError("Subject-mask cache dimensions are not canonical.")
    return dimensions


def _components_from_manifest(value: object) -> SubjectMaskComponentRegistry:
    if not isinstance(value, Mapping) or not isinstance(value.get("labels"), list):
        raise TypeError("Subject-mask cache components must be an object.")
    components = SubjectMaskComponentRegistry(tuple(value["labels"]))
    if dict(value) != components.as_manifest():
        raise ValueError("Subject-mask cache components are not canonical.")
    return components


def _contour_profile_from_manifest(
    value: object,
    *,
    components: SubjectMaskComponentRegistry,
) -> SubjectMaskSampledContourProfile:
    if not isinstance(value, Mapping):
        raise TypeError("Sampled-contour profile must be an object.")
    default = value.get("default_cache")
    counts = (
        default.get("component_sample_counts") if isinstance(default, Mapping) else None
    )
    if not isinstance(counts, Mapping):
        raise ValueError("Sampled-contour profile lacks component sample counts.")
    profile = SubjectMaskSampledContourProfile(
        {str(label): count for label, count in counts.items()}
    )
    if dict(value) != profile.as_manifest(components=components):
        raise ValueError("Sampled-contour profile differs from its frozen builder.")
    return profile


def _source_refined_context(
    source_root: Path,
    *,
    refined_run_id: str,
) -> tuple[
    Any,
    Mapping[str, Any],
    SubjectMaskDimensions,
    SubjectMaskComponentRegistry,
    dict[str, object],
]:
    errors = validate_persisted_subject_mask_core_publication(
        source_root,
        family="refined_subject_masks_runs",
        run_id=refined_run_id,
    )
    if errors:
        raise ValueError("Invalid refined source: " + "; ".join(errors))
    run = zarr.open_group(
        str(source_root / "refined_subject_masks_runs" / refined_run_id),
        mode="r",
        use_consolidated=False,
    )
    manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError("Refined source run_manifest is absent.")
    manifest_errors = validate_subject_mask_core_run_manifest(manifest)
    if manifest_errors:
        raise ValueError(
            "Invalid refined source manifest: " + "; ".join(manifest_errors)
        )
    payload = manifest["payload"]
    if payload.get("kind") != "refined_dense_core":
        raise ValueError("Sampled contours require a refined dense core source.")
    logical = payload["logical_schema"]
    dimensions = _dimensions_from_manifest(logical["dimensions"])
    components = _components_from_manifest(logical["components"])
    arrays = payload["logical_content"]["document"]["arrays"]
    source_hashes = {
        path: _require_sha256(arrays[path]["sha256"], name=f"source {path}")
        for path in ("masks_roi", *_IDENTITY_PATHS)
    }
    source = {
        "stage": "refined_subject_masks",
        "run_name": refined_run_id,
        "run_path": f"refined_subject_masks_runs/{refined_run_id}",
        "manifest_schema_id": manifest["schema_id"],
        "manifest_schema_version": manifest["schema_version"],
        "manifest_payload_digest": manifest["payload_digest"],
        "manifest_document_digest": canonical_json_sha256(manifest),
        "dense_array_values_sha256": source_hashes["masks_roi"],
        "component_registry_digest": canonical_json_sha256(components.as_manifest()),
        "row_identity_array_values_sha256": {
            path: source_hashes[path] for path in _IDENTITY_PATHS
        },
        "authority": "dense_masks_roi",
    }
    return run, manifest, dimensions, components, source


def _group_for_path(group: Any, path: str) -> tuple[Any, str]:
    parts = path.split("/")
    target = group
    for part in parts[:-1]:
        target = target.require_group(part)
    return target, parts[-1]


def _array_attributes(*, component: str, field: str) -> dict[str, object]:
    return {
        "benchmark_only": True,
        "selector_eligible": False,
        "artifact_class": "subject_mask_derived_presentation_cache",
        "authority": "derived_from_dense_masks_roi",
        "authoritative_pixels": False,
        "cache_kind": SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS.value,
        "component": component,
        "field": field,
    }


def _sampled_group_attributes(
    *, component: str, sample_count: int, source_run_id: str
) -> dict[str, object]:
    return {
        "schema_id": "sampled_component_contours_v1",
        "contour_schema_id": "sampled_component_contours_v1",
        "coordinate_space": DEFAULT_CONTOUR_COORDINATE_SPACE,
        "point_order": "xy",
        "source_component": component,
        "source_mask_run": source_run_id,
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
        "sample_count": sample_count,
        "boundary_policy": DEFAULT_BOUNDARY_POLICY,
        "min_source_points": 2,
        "cache_coverage": "full_recording_indexed_rows",
        "surface_role": "canonical_derived_display_cache",
        "authoritative_pixels": False,
    }


def _write_memmap_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> dict[str, object]:
    unit = plan.shard_shape or plan.chunk_shape
    if unit is None:
        raise ValueError("Sampled-contour cache arrays cannot be scalar.")
    unit_rows = max(1, int(unit[0]))
    trailing = (slice(None),) * (values.ndim - 1)
    digest = hashlib.sha256()
    samples: list[dict[str, object]] = []
    starts = tuple(range(0, int(values.shape[0]), unit_rows))
    for index, start in enumerate(starts):
        stop = min(int(values.shape[0]), start + unit_rows)
        selection = (slice(start, stop), *trailing)
        block = np.ascontiguousarray(values[selection])
        digest.update(block.view(np.uint8))
        destination[selection] = block
        if index == 0 or index == len(starts) - 1:
            samples.append(
                {
                    "start_row": start,
                    "stop_row": stop,
                    "sha256": hashlib.sha256(block.view(np.uint8)).hexdigest(),
                }
            )
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "digest_algorithm": SUBJECT_MASK_CACHE_ARRAY_DIGEST_ALGORITHM,
        "sha256": digest.hexdigest(),
        "physical_write_count": len(starts),
        "bounded_reopen_samples": samples,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def _metadata_paths(plans: SubjectMaskSampledContourStoragePlanSet) -> tuple[str, ...]:
    paths = {"", "components"}
    for component in plans.components.labels:
        paths.add(f"components/{component}")
        paths.add(f"components/{component}/sampled_contours")
    paths.update(entry.rule.path for entry in plans.entries)
    return tuple(sorted(paths, key=lambda path: (path.count("/"), path)))


def _metadata_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: SubjectMaskSampledContourStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    prefix = f"{SUBJECT_MASK_CACHE_FAMILY}/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in _metadata_paths(plans):
        path = output_path / prefix
        if relative:
            path = path / relative
        direct[relative] = _strict_json(path / "zarr.json")
    root = _strict_json(output_path / "zarr.json")
    envelope = root.get("consolidated_metadata")
    flattened = envelope.get("metadata") if isinstance(envelope, Mapping) else None
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("kind") != "inline"
        or envelope.get("must_understand") is not False
        or not isinstance(flattened, Mapping)
    ):
        raise ValueError("Sampled-contour publication lacks exact inline metadata.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in direct:
        key = prefix if not relative else f"{prefix}/{relative}"
        value = flattened.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"Consolidated metadata lacks {key!r}.")
        consolidated[relative] = dict(value)
    return direct, consolidated


def _normalized_metadata_document(
    declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for path, declaration in declarations.items():
        normalized = metadata_without_empty_group_consolidation(declaration, path=path)
        if path == "":
            attributes = normalized.get("attributes")
            if not isinstance(attributes, Mapping):
                raise ValueError("Cache run metadata attributes are absent.")
            redacted = dict(attributes)
            for name in _LIFECYCLE_METADATA_ATTRS:
                redacted.pop(name, None)
            normalized["attributes"] = redacted
        result[path] = normalized
    return result


def _metadata_digest(
    direct: Mapping[str, Mapping[str, Any]],
    consolidated: Mapping[str, Mapping[str, Any]],
) -> str:
    direct_normalized = _normalized_metadata_document(direct)
    consolidated_normalized = _normalized_metadata_document(consolidated)
    if direct_normalized != consolidated_normalized:
        raise ValueError("Direct and consolidated cache metadata differ.")
    return canonical_json_sha256(direct_normalized)


def _logical_content(
    write_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    arrays = {
        path: {
            "shape": list(receipt["shape"]),
            "dtype": str(receipt["dtype"]),
            "digest_algorithm": str(receipt["digest_algorithm"]),
            "sha256": str(receipt["sha256"]),
        }
        for path, receipt in sorted(write_receipts.items())
    }
    document = {
        "schema_id": "palette.subject_mask.sampled_contour_logical_content",
        "schema_version": 1,
        "arrays": arrays,
    }
    return {
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "document": document,
        "digest": canonical_json_sha256(document),
    }


def _cache_extension(
    *,
    logical_content: Mapping[str, Any],
    source: Mapping[str, Any],
    components: SubjectMaskComponentRegistry,
    generated_at_utc: str,
) -> dict[str, object]:
    arrays = logical_content["document"]["arrays"]
    receipts: list[SubjectMaskDerivedCacheReceipt] = []
    for component in components.labels:
        prefix = f"components/{component}/sampled_contours"
        component_document = {
            field: arrays[f"{prefix}/{field}"]
            for field in ("points_xy", "valid", "source_point_count")
        }
        receipts.append(
            SubjectMaskDerivedCacheReceipt(
                cache_kind=SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS,
                cache_path=prefix,
                source_dense_core_manifest_digest=str(
                    source["manifest_document_digest"]
                ),
                source_dense_array_values_sha256=str(
                    source["dense_array_values_sha256"]
                ),
                component_registry_digest=str(source["component_registry_digest"]),
                logical_content_digest=canonical_json_sha256(component_document),
                generator_id=SUBJECT_MASK_CACHE_GENERATOR_ID,
                generator_version=SUBJECT_MASK_CACHE_GENERATOR_VERSION,
                generated_at_utc=generated_at_utc,
            )
        )
    return published_subject_mask_cache_extension_manifest(tuple(receipts))


def build_subject_mask_cache_run_manifest(
    *,
    run_id: str,
    dimensions: SubjectMaskDimensions,
    components: SubjectMaskComponentRegistry,
    contour_profile: SubjectMaskSampledContourProfile,
    source: Mapping[str, Any],
    plans: SubjectMaskSampledContourStoragePlanSet,
    logical_content: Mapping[str, Any],
    cache_extension: Mapping[str, Any],
    write_receipts: Mapping[str, Mapping[str, object]],
    metadata_digest: str,
    source_compute_block_bytes: int,
    effective_compute_block_rows: int,
    requested_compute_workers: int,
    effective_compute_workers: int,
    source_mode: str,
    worker_assembly: Mapping[str, Any] | None,
) -> dict[str, object]:
    resolved_run = _safe_run_id(run_id, name="run_id")
    payload = {
        "run_id": resolved_run,
        "stage_family": SUBJECT_MASK_CACHE_FAMILY,
        "kind": SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND,
        "publication": {
            "completion_contract": RUN_COMPLETION_CONTRACT,
            "completion_status": RUN_STATUS_COMPLETE,
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_digest_scope": SUBJECT_MASK_CACHE_METADATA_DIGEST_SCOPE,
            "metadata_digest": _require_sha256(metadata_digest, name="metadata_digest"),
        },
        "dimensions": dimensions.as_manifest(),
        "components": components.as_manifest(),
        "contour_profile": contour_profile.as_manifest(components=components),
        "source_refined_subject_mask_snapshot": dict(source),
        "storage_plan": plans.as_manifest(),
        "logical_content": dict(logical_content),
        "cache_extension": dict(cache_extension),
        "write_receipt": {
            "generation": (
                "receipt_bound_worker_sampled_contour_assembly_v1"
                if source_mode == "receipt_bound_worker_arrays"
                else "bounded_dense_rows_to_disjoint_node_local_memmap_ranges_v1"
            ),
            "compute_backend": (
                "precomputed_worker_sampled_contours"
                if source_mode == "receipt_bound_worker_arrays"
                else (
                    "serial_blocks"
                    if int(effective_compute_workers) == 1
                    else "process_blocks"
                )
            ),
            "source_compute_block_bytes": int(source_compute_block_bytes),
            "effective_compute_block_rows": int(effective_compute_block_rows),
            "requested_compute_workers": int(requested_compute_workers),
            "effective_compute_workers": int(effective_compute_workers),
            "publication": "one_process_owns_every_complete_output_shard",
            "full_dense_equivalence": True,
            "physical_write_counts": {
                path: int(receipt["physical_write_count"])
                for path, receipt in sorted(write_receipts.items())
            },
            "bounded_reopen_samples": {
                path: list(receipt["bounded_reopen_samples"])
                for path, receipt in sorted(write_receipts.items())
            },
            "source_mode": str(source_mode),
            "worker_assembly": (
                dict(worker_assembly) if worker_assembly is not None else None
            ),
        },
    }
    manifest = {
        "schema_id": SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(manifest)
    return manifest


def validate_subject_mask_cache_run_manifest(
    manifest: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("subject-mask cache manifest envelope fields are not exact")
    payload = manifest.get("payload")
    if (
        manifest.get("schema_id") != SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version")
        not in SUBJECT_MASK_CACHE_RUN_MANIFEST_SUPPORTED_VERSIONS
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(payload, Mapping)
    ):
        return (*errors, "subject-mask cache manifest envelope mismatch")
    try:
        if manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("subject-mask cache payload digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask cache manifest is not strict JSON: {exc}")
    expected_payload = {
        "run_id",
        "stage_family",
        "kind",
        "publication",
        "dimensions",
        "components",
        "contour_profile",
        "source_refined_subject_mask_snapshot",
        "storage_plan",
        "logical_content",
        "cache_extension",
        "write_receipt",
    }
    if set(payload) != expected_payload:
        errors.append("subject-mask cache payload fields are not exact")
    try:
        _safe_run_id(payload.get("run_id"), name="run_id")
    except ValueError as exc:
        errors.append(str(exc))
    if (
        payload.get("stage_family") != SUBJECT_MASK_CACHE_FAMILY
        or payload.get("kind") != SUBJECT_MASK_SAMPLED_CONTOUR_STAGE_KIND
    ):
        errors.append("subject-mask cache family/kind mismatch")

    dimensions: SubjectMaskDimensions | None = None
    components: SubjectMaskComponentRegistry | None = None
    plans: SubjectMaskSampledContourStoragePlanSet | None = None
    try:
        dimensions = _dimensions_from_manifest(payload.get("dimensions"))
        components = _components_from_manifest(payload.get("components"))
        components.require_dimensions(dimensions)
        contour_profile = _contour_profile_from_manifest(
            payload.get("contour_profile"), components=components
        )
        storage = payload.get("storage_plan")
        if not isinstance(storage, Mapping) or not isinstance(
            storage.get("storage_profile"), Mapping
        ):
            raise ValueError("Subject-mask cache storage plan/profile is absent.")
        storage_profile = storage_profile_from_manifest(storage["storage_profile"])
        plans = plan_subject_mask_sampled_contour_storage(
            dimensions,
            components=components,
            contour_profile=contour_profile,
            profile=storage_profile,
        )
        if dict(storage) != plans.as_manifest():
            errors.append("subject-mask cache storage plan differs from planner output")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))

    source = payload.get("source_refined_subject_mask_snapshot")
    expected_source_fields = {
        "stage",
        "run_name",
        "run_path",
        "manifest_schema_id",
        "manifest_schema_version",
        "manifest_payload_digest",
        "manifest_document_digest",
        "dense_array_values_sha256",
        "component_registry_digest",
        "row_identity_array_values_sha256",
        "authority",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source_fields:
        errors.append("subject-mask cache source binding is not exact")
    else:
        if (
            source.get("stage") != "refined_subject_masks"
            or source.get("run_path")
            != f"refined_subject_masks_runs/{source.get('run_name')}"
            or source.get("authority") != "dense_masks_roi"
        ):
            errors.append("subject-mask cache source path/authority mismatch")
        for field in (
            "manifest_payload_digest",
            "manifest_document_digest",
            "dense_array_values_sha256",
            "component_registry_digest",
        ):
            try:
                _require_sha256(source.get(field), name=f"source {field}")
            except ValueError as exc:
                errors.append(str(exc))
        identities = source.get("row_identity_array_values_sha256")
        if not isinstance(identities, Mapping) or set(identities) != set(
            _IDENTITY_PATHS
        ):
            errors.append("subject-mask cache source row identities are not exact")
        else:
            for path in _IDENTITY_PATHS:
                try:
                    _require_sha256(identities.get(path), name=f"source {path}")
                except ValueError as exc:
                    errors.append(str(exc))
        if source_manifest is not None:
            source_errors = validate_subject_mask_core_run_manifest(source_manifest)
            if source_errors:
                errors.extend(f"source: {error}" for error in source_errors)
            else:
                source_payload = source_manifest["payload"]
                source_arrays = source_payload["logical_content"]["document"]["arrays"]
                expected_source = {
                    "stage": "refined_subject_masks",
                    "run_name": source_payload["run_id"],
                    "run_path": (
                        f"refined_subject_masks_runs/{source_payload['run_id']}"
                    ),
                    "manifest_schema_id": source_manifest["schema_id"],
                    "manifest_schema_version": source_manifest["schema_version"],
                    "manifest_payload_digest": source_manifest["payload_digest"],
                    "manifest_document_digest": canonical_json_sha256(source_manifest),
                    "dense_array_values_sha256": source_arrays["masks_roi"]["sha256"],
                    "component_registry_digest": canonical_json_sha256(
                        source_payload["logical_schema"]["components"]
                    ),
                    "row_identity_array_values_sha256": {
                        path: source_arrays[path]["sha256"] for path in _IDENTITY_PATHS
                    },
                    "authority": "dense_masks_roi",
                }
                if dict(source) != expected_source:
                    errors.append("subject-mask cache source binding differs")

    publication = payload.get("publication")
    expected_publication = {
        "completion_contract": RUN_COMPLETION_CONTRACT,
        "completion_status": RUN_STATUS_COMPLETE,
        "stage_selector_eligible": False,
        "metadata_state": "direct_and_consolidated_validated",
        "metadata_digest_scope": SUBJECT_MASK_CACHE_METADATA_DIGEST_SCOPE,
        "metadata_digest": (
            publication.get("metadata_digest")
            if isinstance(publication, Mapping)
            else None
        ),
    }
    if (
        not isinstance(publication, Mapping)
        or dict(publication) != expected_publication
    ):
        errors.append("subject-mask cache publication declaration differs")
    else:
        try:
            _require_sha256(publication.get("metadata_digest"), name="metadata_digest")
        except ValueError as exc:
            errors.append(str(exc))

    logical = payload.get("logical_content")
    expected_paths = set(plans.by_path()) if plans is not None else set()
    if not isinstance(logical, Mapping) or set(logical) != {
        "digest_algorithm",
        "document",
        "digest",
    }:
        errors.append("subject-mask cache logical content is not exact")
    else:
        document = logical.get("document")
        arrays = document.get("arrays") if isinstance(document, Mapping) else None
        if (
            logical.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
            or not isinstance(document, Mapping)
            or set(document) != {"schema_id", "schema_version", "arrays"}
            or document.get("schema_id")
            != "palette.subject_mask.sampled_contour_logical_content"
            or document.get("schema_version") != 1
            or not isinstance(arrays, Mapping)
            or set(arrays) != expected_paths
        ):
            errors.append("subject-mask cache logical array inventory differs")
        else:
            for path, entry in plans.by_path().items():
                value = arrays[path]
                expected_fields = {
                    "shape",
                    "dtype",
                    "digest_algorithm",
                    "sha256",
                }
                if (
                    not isinstance(value, Mapping)
                    or set(value) != expected_fields
                    or value.get("shape") != list(entry.plan.logical_shape)
                    or value.get("dtype") != entry.plan.logical_dtype
                    or value.get("digest_algorithm")
                    != SUBJECT_MASK_CACHE_ARRAY_DIGEST_ALGORITHM
                ):
                    errors.append(
                        f"subject-mask cache array declaration differs at {path}"
                    )
                else:
                    try:
                        _require_sha256(value.get("sha256"), name=f"array {path}")
                    except ValueError as exc:
                        errors.append(str(exc))
            if logical.get("digest") != canonical_json_sha256(document):
                errors.append("subject-mask cache logical content digest mismatch")

    extension = payload.get("cache_extension")
    if not isinstance(extension, Mapping):
        errors.append("subject-mask cache extension must be an object")
    else:
        errors.extend(validate_published_subject_mask_cache_extension(extension))
        if (
            components is not None
            and isinstance(logical, Mapping)
            and isinstance(source, Mapping)
        ):
            receipts = extension.get("receipts")
            observed_paths = (
                {
                    receipt["payload"]["cache_path"]
                    for receipt in receipts
                    if isinstance(receipt, Mapping)
                    and isinstance(receipt.get("payload"), Mapping)
                }
                if isinstance(receipts, list)
                else set()
            )
            expected_receipt_paths = {
                f"components/{component}/sampled_contours"
                for component in components.labels
            }
            if observed_paths != expected_receipt_paths:
                errors.append("subject-mask cache receipts do not cover components")
            logical_document = logical.get("document")
            logical_arrays = (
                logical_document.get("arrays")
                if isinstance(logical_document, Mapping)
                else None
            )
            for receipt in receipts if isinstance(receipts, list) else ():
                receipt_payload = (
                    receipt.get("payload") if isinstance(receipt, Mapping) else None
                )
                if not isinstance(receipt_payload, Mapping):
                    continue
                cache_path = receipt_payload.get("cache_path")
                if not isinstance(cache_path, str) or not isinstance(
                    logical_arrays, Mapping
                ):
                    continue
                source_binding = receipt_payload.get("source")
                expected_receipt_source = {
                    "schema_id": REFINED_SUBJECT_MASK_CORE_SCHEMA_ID,
                    "schema_version": 1,
                    "dense_core_manifest_digest": source.get(
                        "manifest_document_digest"
                    ),
                    "dense_array_values_sha256": source.get(
                        "dense_array_values_sha256"
                    ),
                    "component_registry_digest": source.get(
                        "component_registry_digest"
                    ),
                }
                if source_binding != expected_receipt_source:
                    errors.append(
                        f"subject-mask cache receipt source differs at {cache_path}"
                    )
                component_document = {
                    field: logical_arrays.get(f"{cache_path}/{field}")
                    for field in ("points_xy", "valid", "source_point_count")
                }
                if receipt_payload.get("logical_content_digest") != (
                    canonical_json_sha256(component_document)
                ):
                    errors.append(
                        f"subject-mask cache receipt content differs at {cache_path}"
                    )
                if receipt_payload.get("generator") != {
                    "id": SUBJECT_MASK_CACHE_GENERATOR_ID,
                    "version": SUBJECT_MASK_CACHE_GENERATOR_VERSION,
                }:
                    errors.append(
                        f"subject-mask cache receipt generator differs at {cache_path}"
                    )

    manifest_version = int(manifest["schema_version"])
    write_receipt = payload.get("write_receipt")
    expected_write_fields = {
        "generation",
        "compute_backend",
        "source_compute_block_bytes",
        "effective_compute_block_rows",
        "requested_compute_workers",
        "effective_compute_workers",
        "publication",
        "full_dense_equivalence",
        "physical_write_counts",
        "bounded_reopen_samples",
    }
    if manifest_version >= 2:
        expected_write_fields.update({"source_mode", "worker_assembly"})
    if (
        not isinstance(write_receipt, Mapping)
        or set(write_receipt) != expected_write_fields
        or write_receipt.get("publication")
        != "one_process_owns_every_complete_output_shard"
        or write_receipt.get("full_dense_equivalence") is not True
        or not isinstance(write_receipt.get("physical_write_counts"), Mapping)
        or set(write_receipt["physical_write_counts"]) != expected_paths
        or not isinstance(write_receipt.get("bounded_reopen_samples"), Mapping)
        or set(write_receipt["bounded_reopen_samples"]) != expected_paths
    ):
        errors.append("subject-mask cache write receipt differs")
    elif manifest_version == 1 or write_receipt.get("source_mode") == (
        "dense_authority_recompute"
    ):
        if (
            write_receipt.get("generation")
            != "bounded_dense_rows_to_disjoint_node_local_memmap_ranges_v1"
            or (
                manifest_version >= 2
                and write_receipt.get("worker_assembly") is not None
            )
            or not isinstance(write_receipt.get("source_compute_block_bytes"), int)
            or int(write_receipt["source_compute_block_bytes"]) <= 0
            or not isinstance(write_receipt.get("effective_compute_block_rows"), int)
            or int(write_receipt["effective_compute_block_rows"]) <= 0
            or not isinstance(write_receipt.get("requested_compute_workers"), int)
            or int(write_receipt["requested_compute_workers"]) <= 0
            or not isinstance(write_receipt.get("effective_compute_workers"), int)
            or int(write_receipt["effective_compute_workers"]) <= 0
            or int(write_receipt["effective_compute_workers"])
            > int(write_receipt["requested_compute_workers"])
            or write_receipt.get("compute_backend")
            != (
                "serial_blocks"
                if int(write_receipt["effective_compute_workers"]) == 1
                else "process_blocks"
            )
        ):
            errors.append("subject-mask cache dense generation execution differs")
    elif write_receipt.get("source_mode") == "receipt_bound_worker_arrays":
        assembly = write_receipt.get("worker_assembly")
        if (
            write_receipt.get("generation")
            != "receipt_bound_worker_sampled_contour_assembly_v1"
            or write_receipt.get("compute_backend")
            != "precomputed_worker_sampled_contours"
            or write_receipt.get("source_compute_block_bytes") != 0
            or write_receipt.get("effective_compute_block_rows") != 0
            or write_receipt.get("requested_compute_workers") != 0
            or write_receipt.get("effective_compute_workers") != 0
            or not isinstance(assembly, Mapping)
            or dimensions is None
            or components is None
        ):
            errors.append("subject-mask cache worker assembly execution differs")
        else:
            try:
                validate_subject_mask_sampled_contour_worker_assembly(
                    assembly,
                    n_rois=dimensions.n_rois,
                    components=components,
                )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
    else:
        errors.append("subject-mask cache source mode is unsupported")
    return tuple(errors)


def _validate_metadata_and_samples(
    output_path: Path,
    *,
    run_id: str,
    plans: SubjectMaskSampledContourStoragePlanSet,
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    direct, consolidated = _metadata_maps(output_path, run_id=run_id, plans=plans)
    if (
        _metadata_digest(direct, consolidated)
        != manifest["payload"]["publication"]["metadata_digest"]
    ):
        errors.append("subject-mask cache metadata digest mismatch")
    run = zarr.open_group(
        str(output_path / SUBJECT_MASK_CACHE_FAMILY / run_id),
        mode="r",
        use_consolidated=False,
    )
    samples = manifest["payload"]["write_receipt"]["bounded_reopen_samples"]
    for entry in plans.entries:
        path = entry.rule.path
        array = run[path]
        contract = entry.rule.contract
        contract_errors = contract.validate_observation(
            array,
            dimensions={
                **plans.dimensions.contract_dimensions,
                "n_samples": entry.rule.sample_count,
            },
        )
        errors.extend(f"{path}: {error}" for error in contract_errors)
        metadata_errors = validate_array_metadata_declaration_from_plan(
            direct[path],
            contract=contract,
            plan=entry.plan,
            fill_value=0,
        )
        errors.extend(f"{path}: {error}" for error in metadata_errors)
        trailing = (slice(None),) * (array.ndim - 1)
        for sample in samples[path]:
            start = int(sample["start_row"])
            stop = int(sample["stop_row"])
            block = np.ascontiguousarray(
                np.asarray(array[(slice(start, stop), *trailing)])
            )
            if hashlib.sha256(block.view(np.uint8)).hexdigest() != sample["sha256"]:
                errors.append(f"{path}: bounded reopen sample differs")
    return tuple(errors)


def publish_selector_ineligible_subject_mask_sampled_contours(
    *,
    refined_snapshot_root: Path,
    refined_run_id: str,
    destination: Path,
    cache_run_id: str,
    contour_profile: SubjectMaskSampledContourProfile | None = None,
    storage_profile: StorageProfile = SUBJECT_MASK_PRESENTATION_CANDIDATE_V1,
    source_compute_block_bytes: int = DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES,
    compute_workers: int = 1,
    precomputed_arrays: Mapping[str, Any] | None = None,
    worker_assembly: Mapping[str, Any] | None = None,
    created_by: str = SUBJECT_MASK_CACHE_GENERATOR_ID,
) -> SubjectMaskCachePublication:
    """Assemble or explicitly regenerate one recording-level contour cache."""

    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    source_root = refined_snapshot_root.expanduser().resolve()
    resolved_source_run = _safe_run_id(refined_run_id, name="refined_run_id")
    output_path = destination.expanduser().resolve()
    resolved_cache_run = _safe_run_id(cache_run_id, name="cache_run_id")
    if output_path.exists():
        raise FileExistsError(f"Subject-mask cache destination exists: {output_path}")
    adopting_workers = precomputed_arrays is not None or worker_assembly is not None
    if (precomputed_arrays is None) != (worker_assembly is None):
        raise ValueError(
            "precomputed_arrays and worker_assembly must be supplied together."
        )
    if adopting_workers:
        budget = 0
        requested_compute_workers = 0
    else:
        budget = int(source_compute_block_bytes)
        if budget <= 0:
            raise ValueError("source_compute_block_bytes must be positive.")
        requested_compute_workers = int(compute_workers)
        if requested_compute_workers <= 0:
            raise ValueError("compute_workers must be positive.")

    phase = time.perf_counter()
    source_run, source_manifest, dimensions, components, source = (
        _source_refined_context(source_root, refined_run_id=resolved_source_run)
    )
    contour_profile = contour_profile or default_subject_mask_sampled_contour_profile(
        components
    )
    plans = plan_subject_mask_sampled_contour_storage(
        dimensions,
        components=components,
        contour_profile=contour_profile,
        profile=storage_profile,
    )
    if worker_assembly is not None:
        validate_subject_mask_sampled_contour_worker_assembly(
            worker_assembly,
            n_rois=dimensions.n_rois,
            components=components,
        )
        expected_paths = set(plans.by_path())
        if set(precomputed_arrays or {}) != expected_paths:
            raise ValueError(
                "Precomputed sampled-contour array inventory differs from policy."
            )
        for path, entry in plans.by_path().items():
            value = precomputed_arrays[path]  # type: ignore[index]
            if (
                tuple(int(item) for item in value.shape)
                != tuple(entry.plan.logical_shape)
                or str(np.dtype(value.dtype)) != entry.plan.logical_dtype
            ):
                raise ValueError(
                    f"Precomputed sampled-contour array contract differs at {path}."
                )
    phase_seconds["source_validation_and_storage_plan"] = time.perf_counter() - phase

    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_ID,
            "schema_version": SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_VERSION,
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    family = root.create_group(SUBJECT_MASK_CACHE_FAMILY)
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = family.create_group(resolved_cache_run)
    mark_run_started(
        run, run_name=resolved_cache_run, stage="subject_mask_derived_cache"
    )
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "artifact_class": "subject_mask_derived_presentation_cache",
            "authoritative_pixels": False,
            "cache_profile": contour_profile.as_manifest(components=components),
            "storage_plan": plans.as_manifest(),
            "source_refined_subject_mask_snapshot": source,
        }
    )

    destination_arrays: dict[str, Any] = {}
    for entry in plans.entries:
        group, leaf = _group_for_path(run, entry.rule.path)
        if entry.rule.field == "points_xy":
            group.attrs.update(
                _sampled_group_attributes(
                    component=entry.rule.component,
                    sample_count=entry.rule.sample_count,
                    source_run_id=resolved_source_run,
                )
            )
        destination_arrays[entry.rule.path] = create_array_from_plan(
            group,
            name=leaf,
            contract=entry.rule.contract,
            plan=entry.plan,
            fill_value=0,
            attributes=_array_attributes(
                component=entry.rule.component, field=entry.rule.field
            ),
        )

    write_receipts: dict[str, dict[str, object]] = {}
    generated_at = utc_now()
    bytes_per_source_row = (
        dimensions.n_channels * dimensions.roi_height * dimensions.roi_width
    )
    block_rows = (
        0 if adopting_workers else max(1, budget // max(1, bytes_per_source_row))
    )
    temp_parent = output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    try:
        by_path = plans.by_path()
        if adopting_workers:
            assert precomputed_arrays is not None
            phase = time.perf_counter()
            for path in sorted(precomputed_arrays):
                write_receipts[path] = _write_memmap_by_physical_units(
                    destination_arrays[path],
                    precomputed_arrays[path],
                    plan=by_path[path].plan,
                )
            effective_compute_workers = 0
            phase_seconds["receipt_bound_worker_contour_assembly"] = (
                time.perf_counter() - phase
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=f".{resolved_cache_run}.contour_memmap.", dir=temp_parent
            ) as temp_dir_text:
                temp_dir = Path(temp_dir_text)
                memmaps: dict[str, np.memmap] = {}
                for component in components.labels:
                    sample_count = contour_profile.sample_counts[component]
                    prefix = f"components/{component}/sampled_contours"
                    memmaps[f"{prefix}/points_xy"] = np.lib.format.open_memmap(
                        temp_dir / f"{component}.points.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(dimensions.n_rois, sample_count, 2),
                    )
                    memmaps[f"{prefix}/valid"] = np.lib.format.open_memmap(
                        temp_dir / f"{component}.valid.npy",
                        mode="w+",
                        dtype=bool,
                        shape=(dimensions.n_rois,),
                    )
                    memmaps[f"{prefix}/source_point_count"] = np.lib.format.open_memmap(
                        temp_dir / f"{component}.source_count.npy",
                        mode="w+",
                        dtype=np.int32,
                        shape=(dimensions.n_rois,),
                    )

                phase = time.perf_counter()
                effective_compute_workers = _generate_sampled_contours(
                    source_root=source_root,
                    source_run=source_run,
                    refined_run_id=resolved_source_run,
                    dimensions=dimensions,
                    components=components,
                    contour_profile=contour_profile,
                    block_rows=block_rows,
                    compute_workers=requested_compute_workers,
                    memmaps=memmaps,
                )
                for memmap in memmaps.values():
                    memmap.flush()
                phase_seconds["bounded_dense_contour_generation"] = (
                    time.perf_counter() - phase
                )

                phase = time.perf_counter()
                for path in sorted(memmaps):
                    write_receipts[path] = _write_memmap_by_physical_units(
                        destination_arrays[path],
                        memmaps[path],
                        plan=by_path[path].plan,
                    )
                phase_seconds["physical_unit_publication"] = time.perf_counter() - phase
                del memmaps

        phase = time.perf_counter()
        consolidate_metadata_capture_expected_warnings(output_path)
        direct, consolidated = _metadata_maps(
            output_path, run_id=resolved_cache_run, plans=plans
        )
        metadata_digest = _metadata_digest(direct, consolidated)
        phase_seconds["first_consolidation"] = time.perf_counter() - phase

        logical_content = _logical_content(write_receipts)
        cache_extension = _cache_extension(
            logical_content=logical_content,
            source=source,
            components=components,
            generated_at_utc=generated_at,
        )
        manifest = build_subject_mask_cache_run_manifest(
            run_id=resolved_cache_run,
            dimensions=dimensions,
            components=components,
            contour_profile=contour_profile,
            source=source,
            plans=plans,
            logical_content=logical_content,
            cache_extension=cache_extension,
            write_receipts=write_receipts,
            metadata_digest=metadata_digest,
            source_compute_block_bytes=budget,
            effective_compute_block_rows=block_rows,
            requested_compute_workers=requested_compute_workers,
            effective_compute_workers=effective_compute_workers,
            source_mode=(
                "receipt_bound_worker_arrays"
                if adopting_workers
                else "dense_authority_recompute"
            ),
            worker_assembly=worker_assembly,
        )
        errors = validate_subject_mask_cache_run_manifest(
            manifest, source_manifest=source_manifest
        )
        if errors:
            raise RuntimeError("; ".join(errors))
        run.attrs[SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE] = manifest

        phase = time.perf_counter()
        consolidate_metadata_capture_expected_warnings(output_path)
        metadata_errors = _validate_metadata_and_samples(
            output_path,
            run_id=resolved_cache_run,
            plans=plans,
            manifest=manifest,
        )
        phase_seconds["final_metadata_and_sample_gate"] = time.perf_counter() - phase
        if metadata_errors:
            raise RuntimeError("; ".join(metadata_errors))

        writable = zarr.open_group(
            str(output_path / SUBJECT_MASK_CACHE_FAMILY / resolved_cache_run),
            mode="a",
            use_consolidated=False,
        )
        writable.attrs["status"] = "complete"
        mark_run_complete(writable, run_name=resolved_cache_run)
        consolidate_metadata_capture_expected_warnings(output_path)
        final_errors = validate_persisted_subject_mask_cache_publication(
            output_path,
            run_id=resolved_cache_run,
            source_manifest=source_manifest,
        )
        if final_errors:
            raise RuntimeError("; ".join(final_errors))
    except BaseException as exc:
        failed = zarr.open_group(
            str(output_path / SUBJECT_MASK_CACHE_FAMILY / resolved_cache_run),
            mode="a",
            use_consolidated=False,
        )
        failed.attrs["status"] = "failed"
        failed.attrs["stage_selector_eligible"] = False
        mark_run_failed(failed, run_name=resolved_cache_run, error=str(exc))
        consolidate_metadata_capture_expected_warnings(output_path)
        raise

    return SubjectMaskCachePublication(
        output_path=output_path,
        run_id=resolved_cache_run,
        dimensions=dimensions,
        components=components,
        contour_profile=contour_profile,
        plans=plans,
        manifest=manifest,
        phase_seconds=dict(phase_seconds),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def validate_persisted_subject_mask_cache_publication(
    output_path: Path,
    *,
    run_id: str,
    source_manifest: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate one sealed cache without opening its full contour payload."""

    errors: list[str] = []
    resolved_run = _safe_run_id(run_id, name="run_id")
    try:
        run = zarr.open_group(
            str(output_path / SUBJECT_MASK_CACHE_FAMILY / resolved_run),
            mode="r",
            use_consolidated=False,
        )
        manifest = run.attrs.get(SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE)
        if not isinstance(manifest, Mapping):
            return ("subject-mask cache run_manifest is absent",)
        errors.extend(
            validate_subject_mask_cache_run_manifest(
                manifest, source_manifest=source_manifest
            )
        )
        payload = manifest.get("payload")
        if not isinstance(payload, Mapping):
            return tuple(errors)
        dimensions = _dimensions_from_manifest(payload["dimensions"])
        components = _components_from_manifest(payload["components"])
        contour_profile = _contour_profile_from_manifest(
            payload["contour_profile"], components=components
        )
        storage = payload["storage_plan"]
        profile = storage_profile_from_manifest(storage["storage_profile"])
        plans = plan_subject_mask_sampled_contour_storage(
            dimensions,
            components=components,
            contour_profile=contour_profile,
            profile=profile,
        )
        errors.extend(
            _validate_metadata_and_samples(
                output_path,
                run_id=resolved_run,
                plans=plans,
                manifest=manifest,
            )
        )
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            errors.append("subject-mask cache run is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("subject-mask cache run is not selector-ineligible")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return tuple(errors)


__all__ = [
    "DEFAULT_SOURCE_COMPUTE_BLOCK_BYTES",
    "SUBJECT_MASK_CACHE_FAMILY",
    "SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_ID",
    "SUBJECT_MASK_CACHE_PUBLICATION_SCHEMA_VERSION",
    "SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE",
    "SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_CACHE_RUN_MANIFEST_SCHEMA_VERSION",
    "SubjectMaskCachePublication",
    "build_subject_mask_cache_run_manifest",
    "publish_selector_ineligible_subject_mask_sampled_contours",
    "validate_persisted_subject_mask_cache_publication",
    "validate_subject_mask_cache_run_manifest",
]

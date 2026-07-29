"""Selector-ineligible geometry-only crop shadow publication from refined v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.crop_manifest import (
    CROP_RUN_MANIFEST_ATTRIBUTE,
    CropPixelAuthority,
    CropRefinedSourceIdentity,
    build_crop_row_source_signatures,
    build_crop_run_manifest,
    build_coordinate_crop_run_manifest,
    crop_logical_content_digest,
    crop_refined_source_identity_from_refined_manifest,
    crop_row_signature_manifest,
    validate_crop_publication,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    CropSizeMode,
    derive_crop_placement_geometry,
)
from fisheye.shared.zarr.crop_storage import (
    CropGeometryStoragePlanSet,
    plan_crop_geometry_storage,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


DEFAULT_CROP_GEOMETRY_SHADOW_ROOT = Path("/tmp/palette-crop-geometry-shadows")
CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_ID = (
    "palette.crop_geometry.shadow_publication"
)
CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedCropGeometrySnapshot:
    """Validated arrays and immutable upstream evidence ready for publication."""

    dimensions: CropDimensions
    policy: CropGeometryPolicy
    source: CropRefinedSourceIdentity
    pixel_authority: CropPixelAuthority
    arrays: Mapping[str, np.ndarray]
    source_manifest: Mapping[str, Any]
    source_arrays: Mapping[str, Any]


@dataclass(frozen=True)
class CropGeometryShadowPublication:
    """One completed standalone crop shadow and its validation evidence."""

    output_path: Path
    run_id: str
    dimensions: CropDimensions
    plans: CropGeometryStoragePlanSet
    manifest: Mapping[str, object]
    arrays: Mapping[str, Any]
    source_manifest: Mapping[str, Any]
    source_arrays: Mapping[str, Any]
    receipt: Mapping[str, object]


def require_safe_crop_geometry_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_CROP_GEOMETRY_SHADOW_ROOT,
) -> Path:
    path = destination.expanduser().resolve()
    root = shadow_root.expanduser().resolve()
    root_is_safe = root.is_relative_to(Path("/tmp").resolve()) or any(
        marker in root.parts for marker in (".palette_scratch", ".palette_benchmarks")
    )
    if not root_is_safe:
        raise ValueError(
            "Crop shadow roots must be below /tmp, .palette_scratch, or "
            ".palette_benchmarks."
        )
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Crop shadow destination must be a child of {root}.")
    if path.suffix != ".zarr":
        raise ValueError("Crop shadow destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Crop shadow destination already exists: {path}")
    if any(part.endswith("_analysis.zarr") for part in path.parts[:-1]):
        raise ValueError("Crop shadow cannot be nested in a recording archive.")
    return path


def _values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def prepare_crop_geometry_from_refined_source(
    source: BoundRefinedDetectionCropSource,
    *,
    policy: CropGeometryPolicy,
    pixel_authority: CropPixelAuthority,
    roi_sizes_full: np.ndarray | None = None,
) -> PreparedCropGeometrySnapshot:
    """Copy exact refined rows and derive policy-owned crop geometry/signatures."""

    refined_dimensions = source.dimensions
    dimensions = CropDimensions(
        n_frames=refined_dimensions.n_frames,
        n_instances=refined_dimensions.n_instances,
        source_width=refined_dimensions.source_width,
        source_height=refined_dimensions.source_height,
    )
    identity = crop_refined_source_identity_from_refined_manifest(
        source.manifest,
        logical_content_digest=source.logical_content_digest,
    )
    if source.run_id != identity.run_id:
        raise ValueError("Bound refined run differs from its source identity.")
    if pixel_authority.recording_identity != identity.recording_identity:
        raise ValueError("Pixel and refined sources bind different recordings.")
    if (
        pixel_authority.n_frames != dimensions.n_frames
        or pixel_authority.source_width != dimensions.source_width
        or pixel_authority.source_height != dimensions.source_height
    ):
        raise ValueError("Pixel authority dimensions differ from refined source.")

    source_arrays = source.arrays
    rows = dimensions.n_instances
    if policy.size_mode is CropSizeMode.FIXED_PER_RUN:
        if roi_sizes_full is not None:
            raise ValueError(
                "fixed_per_run derives roi_sizes_full from policy; explicit sizes "
                "are forbidden."
            )
        sizes = np.repeat(
            np.asarray(policy.fixed_size_wh, dtype=np.int32).reshape(1, 2),
            rows,
            axis=0,
        )
    else:
        if roi_sizes_full is None:
            raise ValueError("variable_per_row requires explicit roi_sizes_full.")
        raw_sizes = np.asarray(roi_sizes_full)
        if raw_sizes.dtype != np.dtype(np.int32) or raw_sizes.shape != (rows, 2):
            raise ValueError(
                "variable roi_sizes_full must have exact int32 shape (N, 2)."
            )
        sizes = np.array(raw_sizes, copy=True, order="C")

    bbox_img = np.array(
        _values(source_arrays["instances/bbox_img_xyxy"]),
        copy=True,
        order="C",
    )
    centers = np.array(
        _values(source_arrays["instances/centers_img_xy"]),
        copy=True,
        order="C",
    )
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    frame_indices = np.asarray(
        _values(source_arrays["instances/frame_indices"]),
        dtype=np.int64,
    )
    arrays: dict[str, np.ndarray] = {
        "instance_key": np.array(
            _values(source_arrays["instances/instance_key"]), copy=True, order="C"
        ),
        "source_refined_row_ids": np.array(
            _values(source_arrays["instances/refined_row_ids"]),
            copy=True,
            order="C",
        ),
        "frame_indices": np.array(frame_indices, copy=True, order="C"),
        "source_acquisition_frame_index": np.array(
            _values(source_arrays["instances/source_acquisition_frame_index"]),
            copy=True,
            order="C",
        ),
        "frame_row_offsets": np.array(
            _values(source_arrays["instances/frame_row_offsets"]),
            copy=True,
            order="C",
        ),
        "bbox_norm_coords": np.array(
            _values(source_arrays["instances/bbox_norm_coords"]),
            copy=True,
            order="C",
        ),
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "roi_coordinates_full": coordinates,
        "roi_sizes_full": sizes,
        "source_crop_xywh": source_crop,
        "bbox_roi_xyxy": bbox_roi,
    }
    signatures = build_crop_row_source_signatures(
        arrays,
        source=identity,
        policy=policy,
        pixel_authority=pixel_authority,
    )
    arrays["source_row_signature"] = signatures.signatures
    CROP_GEOMETRY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        policy=policy,
    )
    return PreparedCropGeometrySnapshot(
        dimensions=dimensions,
        policy=policy,
        source=identity,
        pixel_authority=pixel_authority,
        arrays=arrays,
        source_manifest=source.manifest,
        source_arrays=source.arrays,
    )


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    if plan.chunk_shape is None:
        raise ValueError("Crop arrays cannot be scalars.")
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def crop_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: CropGeometryStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract exact direct and root-consolidated declarations for one run."""

    relative_paths = ("", *(entry.rule.path for entry in plans.entries))
    run_prefix = f"crop_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path = metadata_path / relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")

    archive_root = _read_strict_json(output_path / "zarr.json")
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Crop shadow lacks root consolidated metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Crop consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Crop consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Crop consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def validate_crop_geometry_shadow_publication(
    publication: CropGeometryShadowPublication,
) -> tuple[str, ...]:
    """Reopen metadata and re-run the complete crop publication gate."""

    try:
        direct, consolidated = crop_metadata_declaration_maps(
            publication.output_path,
            run_id=publication.run_id,
            plans=publication.plans,
        )
    except (OSError, TypeError, ValueError) as exc:
        return (f"crop shadow metadata reopen failed: {exc}",)
    return validate_crop_publication(
        publication.manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=publication.arrays,
        source_manifest=publication.source_manifest,
        source_arrays=publication.source_arrays,
    )


def publish_selector_ineligible_crop_geometry_snapshot(
    prepared: PreparedCropGeometrySnapshot,
    *,
    destination: Path,
    run_id: str,
    shadow_root: Path = DEFAULT_CROP_GEOMETRY_SHADOW_ROOT,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    created_by: str = "crop_geometry_shadow",
    coordinate_catalog: bool = False,
) -> CropGeometryShadowPublication:
    """Write, consolidate, and fully validate one standalone crop shadow."""

    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")

    output_path = require_safe_crop_geometry_shadow_destination(
        destination,
        shadow_root=shadow_root,
    )
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")
    CROP_GEOMETRY_SCHEMA_V1.require(
        prepared.arrays,
        dimensions=prepared.dimensions,
        policy=prepared.policy,
    )
    plans = plan_crop_geometry_storage(prepared.dimensions, profile=profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    family = root.create_group("crop_runs")
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = family.create_group(str(run_id))
    signature_spec = build_crop_row_source_signatures(
        prepared.arrays,
        source=prepared.source,
        policy=prepared.policy,
        pixel_authority=prepared.pixel_authority,
    ).spec
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "artifact_class": "geometry_only_analysis",
            "logical_schema": CROP_GEOMETRY_SCHEMA_V1.as_manifest(
                dimensions=prepared.dimensions,
                policy=prepared.policy,
            ),
            "storage_plan": plans.as_manifest(),
            "source_refined_snapshot": prepared.source.as_manifest(),
            "source_pixel_authority": prepared.pixel_authority.as_manifest(),
            "row_signature": crop_row_signature_manifest(signature_spec),
        }
    )
    destination_arrays: dict[str, Any] = {}
    write_records: list[dict[str, object]] = []
    per_array_write_seconds: dict[str, float] = {}
    try:
        bindings = {
            binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings
        }
        phase_started = time.perf_counter()
        for entry in plans.entries:
            path = entry.rule.path
            values = np.asarray(prepared.arrays[path])
            binding = bindings[path]
            contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            array_started = time.perf_counter()
            array = create_array_from_plan(
                run,
                name=path,
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={
                    "benchmark_only": True,
                    "selector_eligible": False,
                    "artifact_class": "geometry_only_analysis",
                },
            )
            _write_by_physical_units(array, values, plan=entry.plan)
            per_array_write_seconds[path] = time.perf_counter() - array_started
            destination_arrays[path] = array
            write_records.append(
                {
                    "path": path,
                    "logical_shape": list(entry.plan.logical_shape),
                    "logical_dtype": entry.plan.logical_dtype,
                    "chunk_shape": list(entry.plan.chunk_shape or ()),
                    "shard_shape": (
                        None
                        if entry.plan.shard_shape is None
                        else list(entry.plan.shard_shape)
                    ),
                    "write_ownership": entry.plan.write_ownership,
                }
            )
        phase_seconds["create_and_write_arrays"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        CROP_GEOMETRY_SCHEMA_V1.require(
            destination_arrays,
            dimensions=prepared.dimensions,
            policy=prepared.policy,
        )
        source_hashes = {
            path: sha256_array(np.asarray(values))
            for path, values in prepared.arrays.items()
        }
        destination_hashes = {
            path: sha256_array(np.asarray(array[...]))
            for path, array in destination_arrays.items()
        }
        if source_hashes != destination_hashes:
            raise RuntimeError("Crop shadow decoded values differ from prepared rows.")
        logical_content_digest = crop_logical_content_digest(
            destination_arrays,
            dimensions=prepared.dimensions,
            policy=prepared.policy,
            source=prepared.source,
            pixel_authority=prepared.pixel_authority,
        )
        phase_seconds["validate_decoded_values"] = time.perf_counter() - phase_started

        run.attrs["status"] = "complete"
        phase_started = time.perf_counter()
        first_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        phase_seconds["consolidate_before_manifest"] = (
            time.perf_counter() - phase_started
        )
        direct, consolidated = crop_metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        phase_started = time.perf_counter()
        manifest_builder = (
            build_coordinate_crop_run_manifest
            if coordinate_catalog
            else build_crop_run_manifest
        )
        manifest = manifest_builder(
            run_id=str(run_id),
            dimensions=prepared.dimensions,
            policy=prepared.policy,
            storage_plan=plans,
            arrays=destination_arrays,
            source=prepared.source,
            pixel_authority=prepared.pixel_authority,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=False,
        )
        run.attrs[CROP_RUN_MANIFEST_ATTRIBUTE] = manifest
        phase_seconds["build_manifest"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        second_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        phase_seconds["consolidate_after_manifest"] = (
            time.perf_counter() - phase_started
        )
        direct, consolidated = crop_metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        phase_started = time.perf_counter()
        errors = validate_crop_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
            source_manifest=prepared.source_manifest,
            source_arrays=prepared.source_arrays,
        )
        if errors:
            raise RuntimeError(
                "Crop shadow publication validation failed: " + "; ".join(errors)
            )
        phase_seconds["validate_publication"] = time.perf_counter() - phase_started

        receipt: dict[str, object] = {
            "schema_id": CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "artifact_class": "geometry_only_analysis",
            "output_path": str(output_path),
            "run_id": str(run_id),
            "run_manifest_digest": manifest["payload_digest"],
            "logical_content_digest": logical_content_digest,
            "source_refined_manifest_digest": prepared.source.run_manifest_digest,
            "source_pixel_authority_manifest_digest": (
                prepared.pixel_authority.authority_manifest_digest
            ),
            "crop_policy_digest": prepared.policy.payload_digest,
            "storage_profile_id": profile.profile_id,
            "logical_hashes": destination_hashes,
            "writes": write_records,
            "per_array_write_seconds": per_array_write_seconds,
            "phase_seconds": phase_seconds,
            "consolidation": {
                "before_manifest": first_consolidation,
                "after_manifest": second_consolidation,
            },
            "publication_seconds": float(time.perf_counter() - started),
            "production_state_changes": [],
        }
        with (output_path / "shadow_publication_receipt.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                receipt,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        return CropGeometryShadowPublication(
            output_path=output_path,
            run_id=str(run_id),
            dimensions=prepared.dimensions,
            plans=plans,
            manifest=manifest,
            arrays=destination_arrays,
            source_manifest=prepared.source_manifest,
            source_arrays=prepared.source_arrays,
            receipt=receipt,
        )
    except Exception as exc:
        run.attrs["status"] = "failed"
        run.attrs["stage_selector_eligible"] = False
        run.attrs["publication_failure"] = str(exc)
        raise


def publish_refined_crop_geometry_shadow(
    source: BoundRefinedDetectionCropSource,
    *,
    policy: CropGeometryPolicy,
    pixel_authority: CropPixelAuthority,
    destination: Path,
    run_id: str,
    roi_sizes_full: np.ndarray | None = None,
    shadow_root: Path = DEFAULT_CROP_GEOMETRY_SHADOW_ROOT,
    profile: StorageProfile = PUBLISHED_HTTP_V1,
    coordinate_catalog: bool = False,
) -> CropGeometryShadowPublication:
    """Prepare and publish a fresh geometry-only shadow from one bound source."""

    prepared = prepare_crop_geometry_from_refined_source(
        source,
        policy=policy,
        pixel_authority=pixel_authority,
        roi_sizes_full=roi_sizes_full,
    )
    return publish_selector_ineligible_crop_geometry_snapshot(
        prepared,
        destination=destination,
        run_id=run_id,
        shadow_root=shadow_root,
        profile=profile,
        coordinate_catalog=coordinate_catalog,
    )


__all__ = [
    "CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_ID",
    "CROP_GEOMETRY_SHADOW_RECEIPT_SCHEMA_VERSION",
    "DEFAULT_CROP_GEOMETRY_SHADOW_ROOT",
    "CropGeometryShadowPublication",
    "PreparedCropGeometrySnapshot",
    "crop_metadata_declaration_maps",
    "prepare_crop_geometry_from_refined_source",
    "publish_refined_crop_geometry_shadow",
    "publish_selector_ineligible_crop_geometry_snapshot",
    "require_safe_crop_geometry_shadow_destination",
    "validate_crop_geometry_shadow_publication",
]

"""Fresh selector-ineligible publisher for refined-detection v1 shadows.

This module is deliberately unable to write into a recording archive. It only
creates a new standalone Zarr store below an explicit safe shadow root, never
updates a selector or registry, and validates the complete publication before
returning success. Shadows are integration artifacts, not profile-promotion
evidence or production authorities.
"""

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
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    build_refined_detection_run_manifest,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_storage import (
    REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    RefinedDetectionStoragePlanSet,
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.refined_detection_transition import (
    RefinedDetectionTransitionResult,
)
from fisheye.shared.zarr.storage_profiles import StorageProfile
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


DEFAULT_REFINED_DETECTION_SHADOW_ROOT = Path("/tmp/palette-refined-detection-shadows")
SHADOW_RECEIPT_SCHEMA_ID = "palette.refined_detection.shadow_publication"
SHADOW_RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefinedDetectionShadowPublication:
    """Completed standalone shadow and its validated evidence."""

    output_path: Path
    run_id: str
    manifest: Mapping[str, object]
    receipt: Mapping[str, object]


def require_safe_refined_detection_shadow_destination(
    destination: Path,
    *,
    shadow_root: Path = DEFAULT_REFINED_DETECTION_SHADOW_ROOT,
) -> Path:
    """Require a fresh child path in a shadow/benchmark namespace."""

    path = destination.expanduser().resolve()
    root = shadow_root.expanduser().resolve()
    temporary_root = Path("/tmp").resolve()
    root_is_safe = root.is_relative_to(temporary_root) or (
        ".palette_benchmarks" in root.parts
    )
    if not root_is_safe:
        raise ValueError(
            "Shadow roots must be below /tmp or a .palette_benchmarks namespace."
        )
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Shadow destination must be a child of {root}.")
    if path.suffix != ".zarr":
        raise ValueError("Shadow destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Shadow destination already exists: {path}")
    if any(part.endswith("_analysis.zarr") for part in path.parts[:-1]):
        raise ValueError("Shadow publication cannot be nested in a recording archive.")
    return path


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    """Write one complete outer shard or regular chunk at a time."""

    if plan.chunk_shape is None:
        raise ValueError("Refined shadow arrays cannot be scalars.")
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


def _metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: RefinedDetectionStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract matching direct and archive-root consolidated declarations."""

    relative_paths = (
        "",
        "instances",
        "source_detections",
        *(entry.rule.path for entry in plans.entries),
    )
    run_prefix = f"refined_detect_runs/{run_id}"
    direct: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        metadata_path = output_path / run_prefix
        if relative:
            metadata_path = metadata_path / relative
        direct[relative] = _read_strict_json(metadata_path / "zarr.json")

    archive_root = _read_strict_json(output_path / "zarr.json")
    consolidated_envelope = archive_root.get("consolidated_metadata")
    if not isinstance(consolidated_envelope, Mapping):
        raise ValueError("Shadow archive lacks root consolidated_metadata.")
    if (
        consolidated_envelope.get("kind") != "inline"
        or consolidated_envelope.get("must_understand") is not False
    ):
        raise ValueError("Shadow archive consolidated metadata envelope is invalid.")
    flattened = consolidated_envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Shadow archive consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def _logical_hashes(arrays: Mapping[str, Any]) -> dict[str, str]:
    return {
        path: sha256_array(
            np.asarray(array if isinstance(array, np.ndarray) else array[...])
        )
        for path, array in sorted(arrays.items())
    }


def _validate_transition_source_matches_canonical(
    transition: RefinedDetectionTransitionResult,
    canonical_source: CanonicalDetectionShadowPublication,
) -> tuple[str, ...]:
    """Prove the refined source-audit projection binds one canonical rowset."""

    errors: list[str] = []
    if transition.dimensions.n_frames != canonical_source.dimensions.n_frames:
        errors.append("refined and canonical source frame counts differ")
    if (
        transition.dimensions.n_source_detections
        != canonical_source.dimensions.n_instances
    ):
        errors.append("refined and canonical source row counts differ")
    comparisons = (
        ("frame_indices", "frame_indices"),
        ("source_acquisition_frame_index", "source_acquisition_frame_index"),
        ("instance_key", "instance_key"),
        ("bbox_norm_coords", "bbox_norm_coords"),
        ("bbox_img_xyxy", "bbox_img_xyxy"),
        ("centers_img_xy", "centers_img_xy"),
        ("scores", "scores"),
        ("class_ids", "class_ids"),
        ("frame_row_offsets", "frame_row_offsets"),
    )
    for refined_name, canonical_name in comparisons:
        refined_path = f"source_detections/{refined_name}"
        canonical_path = f"instances/{canonical_name}"
        if refined_path not in transition.arrays:
            errors.append(f"refined source evidence lacks {refined_path!r}")
            continue
        if canonical_path not in canonical_source.arrays:
            errors.append(f"canonical source evidence lacks {canonical_path!r}")
            continue
        refined_values = np.asarray(transition.arrays[refined_path])
        canonical_values = np.asarray(canonical_source.arrays[canonical_path][...])
        if not np.array_equal(refined_values, canonical_values):
            errors.append(
                f"refined source evidence differs from canonical {canonical_path!r}"
            )
    source_rows = np.asarray(
        transition.arrays.get("source_detections/source_detect_row_index", []),
        dtype=np.int64,
    )
    if not np.array_equal(
        source_rows,
        np.arange(canonical_source.dimensions.n_instances, dtype=np.int64),
    ):
        errors.append("refined source row identities are not canonical row positions")
    return tuple(dict.fromkeys(errors))


def publish_refined_detection_shadow(
    transition: RefinedDetectionTransitionResult,
    *,
    destination: Path,
    run_id: str,
    lineage: RefinedDetectionSnapshotLineage,
    canonical_source: CanonicalDetectionShadowPublication,
    shadow_root: Path = DEFAULT_REFINED_DETECTION_SHADOW_ROOT,
    profile: StorageProfile = REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
) -> RefinedDetectionShadowPublication:
    """Write and fully validate one standalone full-acquisition shadow."""

    output_path = require_safe_refined_detection_shadow_destination(
        destination,
        shadow_root=shadow_root,
    )
    if (
        transition.dimensions.lineage_profile
        is not RefinedDetectionLineageProfile.FULL_ACQUISITION
    ):
        raise ValueError(
            "The first shadow publisher supports only full-acquisition transitions."
        )
    if transition.report.get("status") != "contract_ready":
        raise ValueError("Shadow publication requires a contract-ready transition.")
    if transition.report.get("selector_eligible") is not False:
        raise ValueError("Transition report must remain selector-ineligible.")
    REFINED_DETECTION_SCHEMA_V1.require(
        transition.arrays,
        dimensions=transition.dimensions,
    )
    canonical_errors = validate_canonical_detection_shadow_publication(canonical_source)
    if canonical_errors:
        raise ValueError(
            "Canonical source shadow is invalid: " + "; ".join(canonical_errors)
        )
    source_errors = _validate_transition_source_matches_canonical(
        transition,
        canonical_source,
    )
    if source_errors:
        raise ValueError(
            "Refined source audit does not match canonical evidence: "
            + "; ".join(source_errors)
        )
    source = canonical_source.refined_source_identity()
    plans = plan_refined_detection_storage(
        transition.dimensions,
        profile=profile,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": SHADOW_RECEIPT_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
        }
    )
    family = root.create_group("refined_detect_runs")
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": "none_shadow_direct_path_only",
        }
    )
    run = family.create_group(str(run_id))
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "shadow_only": True,
            "logical_schema": REFINED_DETECTION_SCHEMA_V1.as_manifest(
                dimensions=transition.dimensions,
            ),
            "storage_plan": plans.as_manifest(),
            "transition_report": dict(transition.report),
        }
    )
    groups = {
        "instances": run.create_group("instances"),
        "source_detections": run.create_group("source_detections"),
    }
    destination_arrays: dict[str, Any] = {}
    write_records: list[dict[str, object]] = []
    try:
        binding_by_path = {
            binding.path: binding
            for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(
                transition.dimensions
            )
        }
        for entry in plans.entries:
            path = entry.rule.path
            group_name, leaf = path.split("/", 1)
            binding = binding_by_path[path]
            contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            values = np.asarray(transition.arrays[path])
            fill_value: object = False if values.dtype == np.dtype(np.bool_) else 0
            destination_array = create_array_from_plan(
                groups[group_name],
                name=leaf,
                contract=contract,
                plan=entry.plan,
                fill_value=fill_value,
                attributes={
                    "shadow_only": True,
                    "selector_eligible": False,
                },
            )
            _write_by_physical_units(
                destination_array,
                values,
                plan=entry.plan,
            )
            destination_arrays[path] = destination_array
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

        destination_issues = REFINED_DETECTION_SCHEMA_V1.validate(
            destination_arrays,
            dimensions=transition.dimensions,
        )
        if destination_issues:
            raise RuntimeError(
                "Decoded shadow arrays violate refined v1: "
                + "; ".join(
                    f"{issue.code} at {issue.path}: {issue.message}"
                    for issue in destination_issues
                )
            )
        source_hashes = _logical_hashes(transition.arrays)
        destination_hashes = _logical_hashes(destination_arrays)
        if source_hashes != destination_hashes:
            raise RuntimeError(
                "Decoded shadow arrays differ from the transition result."
            )

        run.attrs["status"] = "complete"
        first_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        direct, consolidated = _metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        manifest = build_refined_detection_run_manifest(
            run_id=str(run_id),
            dimensions=transition.dimensions,
            storage_plan=plans,
            lineage=lineage,
            source=source,
            instance_reason_codes=transition.instance_reason_codes,
            source_reason_codes=transition.source_reason_codes,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=False,
        )
        run.attrs["run_manifest"] = manifest
        second_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        direct, consolidated = _metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        publication_errors = validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
        )
        if publication_errors:
            raise RuntimeError(
                "Shadow publication validation failed: " + "; ".join(publication_errors)
            )
        receipt: dict[str, object] = {
            "schema_id": SHADOW_RECEIPT_SCHEMA_ID,
            "schema_version": SHADOW_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "output_path": str(output_path),
            "run_id": str(run_id),
            "source_manifest_digest": source.run_manifest_digest,
            "source_shadow_path": str(canonical_source.output_path),
            "refined_manifest_digest": manifest["payload_digest"],
            "storage_profile_id": profile.profile_id,
            "logical_hashes": destination_hashes,
            "writes": write_records,
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
        return RefinedDetectionShadowPublication(
            output_path=output_path,
            run_id=str(run_id),
            manifest=manifest,
            receipt=receipt,
        )
    except Exception as exc:
        run.attrs["status"] = "failed"
        run.attrs["stage_selector_eligible"] = False
        run.attrs["shadow_failure"] = str(exc)
        raise


__all__ = [
    "DEFAULT_REFINED_DETECTION_SHADOW_ROOT",
    "SHADOW_RECEIPT_SCHEMA_ID",
    "SHADOW_RECEIPT_SCHEMA_VERSION",
    "RefinedDetectionShadowPublication",
    "publish_refined_detection_shadow",
    "require_safe_refined_detection_shadow_destination",
]

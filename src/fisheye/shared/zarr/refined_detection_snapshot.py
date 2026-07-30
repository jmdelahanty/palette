"""Shared immutable publisher for selector-ineligible refined snapshots.

The publisher owns the physical write, consolidation, and fail-closed
publication validation used by shadow and compaction workflows.  It can only
create a fresh standalone store below an explicit scratch or benchmark root;
it has no selector, registry, or in-place archive mutation capability.
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
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionBoundClipEvidence,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceCollectionIdentity,
    RefinedDetectionSourceIdentity,
    build_coordinate_refined_detection_run_manifest,
    build_refined_detection_run_manifest,
    refined_detection_logical_content_digest,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionClippedBinding,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_storage import (
    RefinedDetectionStoragePlanSet,
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_ID = (
    "palette.refined_detection.snapshot_publication"
)
REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_VERSION = 1
DEFAULT_REFINED_DETECTION_SNAPSHOT_ROOT = Path(
    "/tmp/palette-refined-detection-snapshots"
)


@dataclass(frozen=True)
class RefinedDetectionSnapshotPublication:
    """One fully validated immutable standalone snapshot publication."""

    output_path: Path
    run_id: str
    dimensions: RefinedDetectionDimensions
    manifest: Mapping[str, object]
    arrays: Mapping[str, Any]
    receipt: Mapping[str, object]


def require_safe_refined_detection_snapshot_destination(
    destination: Path,
    *,
    safe_root: Path = DEFAULT_REFINED_DETECTION_SNAPSHOT_ROOT,
) -> Path:
    """Require a fresh child in an unmistakable noncanonical namespace."""

    path = destination.expanduser().resolve()
    root = safe_root.expanduser().resolve()
    temporary_root = Path("/tmp").resolve()
    root_is_safe = root.is_relative_to(temporary_root) or any(
        marker in root.parts for marker in (".palette_scratch", ".palette_benchmarks")
    )
    if not root_is_safe:
        raise ValueError(
            "Snapshot roots must be below /tmp, .palette_scratch, or "
            ".palette_benchmarks."
        )
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Snapshot destination must be a child of {root}.")
    if path.suffix != ".zarr":
        raise ValueError("Snapshot destination must use a .zarr suffix.")
    if path.exists():
        raise FileExistsError(f"Snapshot destination already exists: {path}")
    if any(part.endswith("_analysis.zarr") for part in path.parts[:-1]):
        raise ValueError(
            "Snapshot publication cannot be nested in a recording archive."
        )
    return path


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    """Write one complete outer shard or ordinary chunk per assignment."""

    if plan.chunk_shape is None:
        raise ValueError("Refined snapshot arrays cannot be scalars.")
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


def refined_detection_metadata_declaration_maps(
    output_path: Path,
    *,
    run_id: str,
    plans: RefinedDetectionStoragePlanSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract direct and archive-root consolidated declarations for one run."""

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
    envelope = archive_root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ValueError("Snapshot archive lacks root consolidated_metadata.")
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ValueError("Snapshot consolidated metadata envelope is invalid.")
    flattened = envelope.get("metadata")
    if not isinstance(flattened, Mapping):
        raise ValueError("Snapshot consolidated metadata map is missing.")
    consolidated: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        full_path = run_prefix if not relative else f"{run_prefix}/{relative}"
        declaration = flattened.get(full_path)
        if not isinstance(declaration, Mapping):
            raise ValueError(f"Consolidated metadata lacks {full_path!r}.")
        consolidated[relative] = dict(declaration)
    return direct, consolidated


def refined_detection_logical_hashes(
    arrays: Mapping[str, Any],
) -> dict[str, str]:
    """Hash every exact logical array after decoding."""

    return {
        path: sha256_array(
            np.asarray(array if isinstance(array, np.ndarray) else array[...])
        )
        for path, array in sorted(arrays.items())
    }


def publish_selector_ineligible_refined_detection_snapshot(
    *,
    dimensions: RefinedDetectionDimensions,
    arrays: Mapping[str, Any],
    instance_reason_codes: Mapping[int, str],
    source_reason_codes: Mapping[int, str],
    destination: Path,
    run_id: str,
    lineage: RefinedDetectionSnapshotLineage,
    source: RefinedDetectionSourceIdentity | RefinedDetectionSourceCollectionIdentity,
    created_by: str,
    publication_kind: str,
    safe_root: Path = DEFAULT_REFINED_DETECTION_SNAPSHOT_ROOT,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
    clipped_binding: RefinedDetectionClippedBinding | None = None,
    clipped_source_evidence: tuple[RefinedDetectionBoundClipEvidence, ...]
    | None = None,
    run_attributes: Mapping[str, Any] | None = None,
    selection_contract: str = "none_direct_path_only",
    coordinate_catalog: bool = False,
) -> RefinedDetectionSnapshotPublication:
    """Create, consolidate, and validate a fresh immutable full snapshot."""

    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")

    output_path = require_safe_refined_detection_snapshot_destination(
        destination,
        safe_root=safe_root,
    )
    is_clipped = (
        dimensions.lineage_profile
        is RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
    )
    if is_clipped != (clipped_binding is not None):
        raise ValueError(
            "clipped_binding must be present exactly for clipped recording snapshots."
        )
    if is_clipped != (clipped_source_evidence is not None):
        raise ValueError(
            "clipped_source_evidence must be present exactly for clipped recording "
            "snapshots."
        )
    if is_clipped and not isinstance(source, RefinedDetectionSourceCollectionIdentity):
        raise TypeError("Clipped snapshots require a source collection identity.")
    if not is_clipped and not isinstance(source, RefinedDetectionSourceIdentity):
        raise TypeError("Full-acquisition snapshots require one source run identity.")
    if (
        not str(created_by).strip()
        or not str(publication_kind).strip()
        or not str(selection_contract).strip()
    ):
        raise ValueError(
            "created_by, publication_kind, and selection_contract cannot be empty."
        )
    if (parent_manifest is None) != (parent_arrays is None):
        raise ValueError("parent_manifest and parent_arrays must be supplied together.")
    if (lineage.parent_run_id is None) != (parent_manifest is None):
        raise ValueError("Successor lineage and parent evidence must agree.")
    REFINED_DETECTION_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        clipped_binding=clipped_binding,
    )

    phase_seconds: dict[str, float] = {}
    phase_started = time.perf_counter()
    plans = plan_refined_detection_storage(dimensions, profile=profile)
    phase_seconds["plan_storage"] = time.perf_counter() - phase_started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "benchmark_only": True,
            "canonical": False,
            "registry_registered": False,
            "selector_eligible": False,
            "schema_id": REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_ID,
            "schema_version": REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_VERSION,
            "publication_kind": str(publication_kind),
            "created_at_utc": utc_now(),
            "created_by": str(created_by),
        }
    )
    family = root.create_group("refined_detect_runs")
    family.attrs.update(
        {
            "benchmark_only": True,
            "selector_eligible": False,
            "selection_contract": str(selection_contract),
        }
    )
    run = family.create_group(str(run_id))
    attrs = {
        "status": "running",
        "stage_selector_eligible": False,
        "publication_kind": str(publication_kind),
        "logical_schema": REFINED_DETECTION_SCHEMA_V1.as_manifest(
            dimensions=dimensions,
            clipped_binding=clipped_binding,
        ),
        "storage_plan": plans.as_manifest(),
    }
    if run_attributes:
        overlap = set(attrs).intersection(run_attributes)
        if overlap:
            raise ValueError(
                f"run_attributes cannot replace reserved fields: {overlap}"
            )
        attrs.update(dict(run_attributes))
    run.attrs.update(attrs)
    groups = {
        "instances": run.create_group("instances"),
        "source_detections": run.create_group("source_detections"),
    }
    destination_arrays: dict[str, Any] = {}
    write_records: list[dict[str, object]] = []
    per_array_write_seconds: dict[str, float] = {}
    try:
        binding_by_path = {
            binding.path: binding
            for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions)
        }
        phase_started = time.perf_counter()
        for entry in plans.entries:
            path = entry.rule.path
            group_name, leaf = path.split("/", 1)
            binding = binding_by_path[path]
            contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            values = np.asarray(arrays[path])
            fill_value: object = False if values.dtype == np.dtype(np.bool_) else 0
            array_started = time.perf_counter()
            destination_array = create_array_from_plan(
                groups[group_name],
                name=leaf,
                contract=contract,
                plan=entry.plan,
                fill_value=fill_value,
                attributes={
                    "benchmark_only": True,
                    "selector_eligible": False,
                    "publication_kind": str(publication_kind),
                },
            )
            _write_by_physical_units(destination_array, values, plan=entry.plan)
            per_array_write_seconds[path] = time.perf_counter() - array_started
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
        phase_seconds["create_and_write_arrays"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        destination_issues = REFINED_DETECTION_SCHEMA_V1.validate(
            destination_arrays,
            dimensions=dimensions,
            clipped_binding=clipped_binding,
        )
        if destination_issues:
            raise RuntimeError(
                "Decoded snapshot arrays violate refined v1: "
                + "; ".join(
                    f"{issue.code} at {issue.path}: {issue.message}"
                    for issue in destination_issues
                )
            )
        phase_seconds["validate_decoded_schema"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        source_hashes = refined_detection_logical_hashes(arrays)
        destination_hashes = refined_detection_logical_hashes(destination_arrays)
        if source_hashes != destination_hashes:
            raise RuntimeError("Decoded snapshot arrays differ from resolved input.")
        logical_content_digest = refined_detection_logical_content_digest(
            destination_arrays,
            dimensions=dimensions,
            clipped_binding=clipped_binding,
        )
        phase_seconds["logical_hash_equivalence"] = time.perf_counter() - phase_started

        run.attrs["status"] = "complete"
        phase_started = time.perf_counter()
        first_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        phase_seconds["consolidate_before_manifest"] = (
            time.perf_counter() - phase_started
        )
        phase_started = time.perf_counter()
        direct, consolidated = refined_detection_metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        manifest_builder = (
            build_coordinate_refined_detection_run_manifest
            if coordinate_catalog
            else build_refined_detection_run_manifest
        )
        manifest = manifest_builder(
            run_id=str(run_id),
            dimensions=dimensions,
            storage_plan=plans,
            lineage=lineage,
            source=source,
            instance_reason_codes=instance_reason_codes,
            source_reason_codes=source_reason_codes,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=False,
            clipped_binding=clipped_binding,
        )
        run.attrs["run_manifest"] = manifest
        phase_seconds["build_manifest"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        second_consolidation = consolidate_metadata_capture_expected_warnings(
            output_path
        )
        phase_seconds["consolidate_after_manifest"] = (
            time.perf_counter() - phase_started
        )
        direct, consolidated = refined_detection_metadata_declaration_maps(
            output_path,
            run_id=str(run_id),
            plans=plans,
        )
        phase_started = time.perf_counter()
        publication_errors = validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
            parent_manifest=parent_manifest,
            parent_arrays=parent_arrays,
            clipped_source_evidence=clipped_source_evidence,
        )
        if publication_errors:
            raise RuntimeError(
                "Snapshot publication validation failed: "
                + "; ".join(publication_errors)
            )
        phase_seconds["validate_publication"] = time.perf_counter() - phase_started
        receipt: dict[str, object] = {
            "schema_id": REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_ID,
            "schema_version": REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "publication_kind": str(publication_kind),
            "output_path": str(output_path),
            "run_id": str(run_id),
            "refined_manifest_digest": manifest["payload_digest"],
            "storage_profile_id": profile.profile_id,
            "logical_hashes": destination_hashes,
            "logical_content_digest": logical_content_digest,
            "writes": write_records,
            "per_array_write_seconds": per_array_write_seconds,
            "phase_seconds": phase_seconds,
            "consolidation": {
                "before_manifest": first_consolidation,
                "after_manifest": second_consolidation,
            },
            "publication_seconds": float(time.perf_counter() - total_started),
            "production_state_changes": [],
        }
        with (output_path / "snapshot_publication_receipt.json").open(
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
        return RefinedDetectionSnapshotPublication(
            output_path=output_path,
            run_id=str(run_id),
            dimensions=dimensions,
            manifest=manifest,
            arrays=destination_arrays,
            receipt=receipt,
        )
    except Exception as exc:
        run.attrs["status"] = "failed"
        run.attrs["stage_selector_eligible"] = False
        run.attrs["publication_failure"] = str(exc)
        raise


__all__ = [
    "DEFAULT_REFINED_DETECTION_SNAPSHOT_ROOT",
    "REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_ID",
    "REFINED_DETECTION_SNAPSHOT_RECEIPT_SCHEMA_VERSION",
    "RefinedDetectionSnapshotPublication",
    "publish_selector_ineligible_refined_detection_snapshot",
    "refined_detection_logical_hashes",
    "refined_detection_metadata_declaration_maps",
    "require_safe_refined_detection_snapshot_destination",
]

"""Write one selector-ineligible native canonical detection candidate locally."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.detection.clipped_native_binding import (
    BoundClippedCanonicalDetection,
)
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION,
    build_coordinate_canonical_detection_run_manifest,
    build_native_canonical_detection_run_manifest,
    build_native_detection_source_evidence,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)


NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID = (
    "palette.native_canonical_detection.candidate"
)
NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION = 1

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class NativeCanonicalDetectionCandidate:
    """One complete local candidate and its frozen publication evidence."""

    output_path: Path
    run_id: str
    plans: CanonicalDetectionStoragePlanSet
    manifest: Mapping[str, object]
    arrays: Mapping[str, Any]
    receipt: Mapping[str, object]


def _require_run_id(value: str) -> str:
    normalized = str(value).strip()
    if not _RUN_ID_RE.fullmatch(normalized):
        raise ValueError("run_id must be one safe nonempty group name.")
    return normalized


def _require_node_local_destination(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Native detection candidate already exists: {path}")
    if path.suffix != ".zarr":
        raise ValueError("Native detection candidate must use a .zarr suffix.")
    if str(path).startswith(("/groups/", "/nrs/")):
        raise ValueError("Native detection candidates must be built on node-local storage.")
    if path in {Path("/").resolve(), Path("/tmp").resolve(), Path("/scratch").resolve()}:
        raise ValueError("Native detection candidate destination is not a bounded child path.")
    return path


def _write_by_physical_units(destination: Any, values: np.ndarray, *, plan: Any) -> None:
    if plan.chunk_shape is None:
        raise ValueError("Canonical detection arrays cannot be scalars.")
    unit_rows = int(
        plan.shard_shape[0] if plan.shard_shape is not None else plan.chunk_shape[0]
    )
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), unit_rows):
        stop = min(start + unit_rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def _strict_json_file(path: Path, value: Mapping[str, object]) -> None:
    canonical_json_bytes(value)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def validate_native_canonical_detection_candidate(
    candidate: NativeCanonicalDetectionCandidate,
) -> tuple[str, ...]:
    """Reopen direct/consolidated metadata and validate the complete candidate."""

    errors: list[str] = []
    run_path = candidate.output_path / "detect_runs" / candidate.run_id
    try:
        run = zarr.open_group(str(run_path), mode="r", use_consolidated=False)
        arrays = {
            path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            candidate.output_path,
            run_id=candidate.run_id,
            plans=candidate.plans,
        )
        errors.extend(
            validate_canonical_detection_publication(
                candidate.manifest,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                arrays=arrays,
            )
        )
        if dict(run.attrs.get("run_manifest") or {}) != dict(candidate.manifest):
            errors.append("persisted native run manifest differs from the candidate")
        if run.attrs.get("status") != "complete":
            errors.append("native candidate status is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("native candidate became selector eligible")
        family = zarr.open_group(
            str(candidate.output_path / "detect_runs"),
            mode="r",
            use_consolidated=False,
        )
        for selector in (
            "latest",
            "latest_complete",
            "latest_pending",
            "authoritative_run",
        ):
            if family.attrs.get(selector) == candidate.run_id:
                errors.append(f"native candidate is referenced by {selector}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return tuple(dict.fromkeys(errors))


def write_native_clipped_detection_candidate(
    bound: BoundClippedCanonicalDetection,
    *,
    destination: Path,
    run_id: str,
    recording_identity: str,
    producer_id: str,
    producer_version: str,
    source_frame_authority: Mapping[str, Any],
    source_pixel_authority: Mapping[str, Any],
    model_artifact_sha256: str,
    run_provenance: Mapping[str, Any],
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    coordinate_catalog: bool = False,
) -> NativeCanonicalDetectionCandidate:
    """Materialize one complete native canonical candidate on local storage.

    The production-compatible default remains native manifest v2. Benchmark
    companions may opt into manifest v3, which changes only the persisted
    manifest/catalog envelope and retains the same logical arrays and physical
    storage plan.
    """

    output_path = _require_node_local_destination(destination)
    normalized_run_id = _require_run_id(run_id)
    if type(coordinate_catalog) is not bool:
        raise TypeError("coordinate_catalog must be an exact bool.")
    if str(recording_identity).strip() != str(
        bound.binding_evidence["document"]["recording_identity"]
    ):
        raise ValueError(
            "recording_identity differs from the clipped binding evidence."
        )
    provenance = dict(run_provenance)
    existing_binding = provenance.get("clipped_detection_binding")
    binding_document = dict(bound.binding_evidence)
    if existing_binding is not None and existing_binding != binding_document:
        raise ValueError(
            "run_provenance carries different clipped detection binding evidence."
        )
    provenance["clipped_detection_binding"] = binding_document
    canonical_json_bytes(provenance)

    source_evidence = build_native_detection_source_evidence(
        dimensions=bound.dimensions,
        recording_identity=str(recording_identity),
        producer_id=producer_id,
        producer_version=producer_version,
        source_frame_authority=source_frame_authority,
        source_pixel_authority=source_pixel_authority,
        model_artifact_sha256=model_artifact_sha256,
        run_provenance=provenance,
    )
    plans = plan_canonical_detection_storage(bound.dimensions, profile=profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root = zarr.open_group(str(output_path), mode="w-", zarr_format=3)
    root.attrs.update(
        {
            "schema_id": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID,
            "schema_version": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION,
            "recording_id": str(recording_identity),
            "stage_selector_eligible": False,
            "registry_registered": False,
            "created_at_utc": utc_now(),
        }
    )
    family = root.create_group("detect_runs")
    family.attrs.update(
        {
            "stage_selector_eligible": False,
            "selection_contract": "none_native_candidate_direct_path_only",
        }
    )
    run = family.create_group(normalized_run_id)
    run.attrs.update(
        {
            "status": "running",
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "running",
            "stage_selector_eligible": False,
            "immutable_snapshot": True,
            "native_production_candidate": True,
            "logical_schema": CANONICAL_DETECTION_SCHEMA_V1.as_manifest(
                dimensions=bound.dimensions
            ),
            "storage_plan": plans.as_manifest(),
            "source_evidence": source_evidence,
        }
    )
    instances = run.create_group("instances")
    destination_arrays: dict[str, Any] = {}
    writes: list[dict[str, object]] = []
    try:
        binding_by_path = {
            binding.path: binding for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
        }
        for entry in plans.entries:
            path = entry.rule.path
            values = np.asarray(bound.arrays[path])
            binding = binding_by_path[path]
            contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            array = create_array_from_plan(
                instances,
                name=path.split("/", 1)[1],
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={
                    "selector_eligible": False,
                    "native_production_candidate": True,
                },
            )
            _write_by_physical_units(array, values, plan=entry.plan)
            destination_arrays[path] = array
            writes.append(
                {
                    "path": path,
                    "logical_shape": list(values.shape),
                    "logical_dtype": str(values.dtype),
                    "chunk_shape": list(entry.plan.chunk_shape or ()),
                    "shard_shape": (
                        None
                        if entry.plan.shard_shape is None
                        else list(entry.plan.shard_shape)
                    ),
                    "write_ownership": entry.plan.write_ownership,
                }
            )

        CANONICAL_DETECTION_SCHEMA_V1.require(
            destination_arrays,
            dimensions=bound.dimensions,
        )
        source_hashes = {
            path: sha256_array(np.asarray(values))
            for path, values in bound.arrays.items()
        }
        destination_hashes = {
            path: sha256_array(np.asarray(array[...]))
            for path, array in destination_arrays.items()
        }
        if source_hashes != destination_hashes:
            raise RuntimeError("Native candidate decoded values differ from bound input.")

        run.attrs["status"] = "complete"
        run.attrs["palette_run_completion_status"] = "complete"
        first_consolidation = consolidate_metadata_capture_expected_warnings(output_path)
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            output_path,
            run_id=normalized_run_id,
            plans=plans,
        )
        if coordinate_catalog:
            manifest = build_coordinate_canonical_detection_run_manifest(
                run_id=normalized_run_id,
                dimensions=bound.dimensions,
                storage_plan=plans,
                arrays=destination_arrays,
                source_evidence=source_evidence,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                source_evidence_kind="native_detection",
                selector_eligible=False,
            )
            expected_manifest_version = (
                CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
            )
        else:
            manifest = build_native_canonical_detection_run_manifest(
                run_id=normalized_run_id,
                dimensions=bound.dimensions,
                storage_plan=plans,
                arrays=destination_arrays,
                source_evidence=source_evidence,
                direct_metadata_declarations=direct,
                consolidated_metadata_declarations=consolidated,
                selector_eligible=False,
            )
            expected_manifest_version = (
                CANONICAL_DETECTION_NATIVE_RUN_MANIFEST_SCHEMA_VERSION
            )
        if manifest["schema_version"] != expected_manifest_version:
            raise RuntimeError(
                "Native candidate received the wrong run-manifest version."
            )
        run.attrs["run_manifest"] = manifest
        second_consolidation = consolidate_metadata_capture_expected_warnings(output_path)
        direct, consolidated = canonical_detection_metadata_declaration_maps(
            output_path,
            run_id=normalized_run_id,
            plans=plans,
        )
        errors = validate_canonical_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=destination_arrays,
        )
        if errors:
            raise RuntimeError(
                "Native candidate publication validation failed: " + "; ".join(errors)
            )
        receipt: dict[str, object] = {
            "schema_id": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID,
            "schema_version": NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION,
            "status": "complete",
            "stage_selector_eligible": False,
            "registry_registered": False,
            "output_path": str(output_path),
            "run_id": normalized_run_id,
            "native_run_manifest_schema_version": manifest["schema_version"],
            "coordinate_catalog": coordinate_catalog,
            "logical_schema_version": CANONICAL_DETECTION_SCHEMA_V1.schema_version,
            "storage_profile_id": plans.profile.profile_id,
            "run_manifest_digest": manifest["payload_digest"],
            "clipped_binding_digest": bound.binding_evidence["digest"],
            "logical_hashes": destination_hashes,
            "writes": writes,
            "consolidation": {
                "before_manifest": first_consolidation,
                "after_manifest": second_consolidation,
            },
            "publication_seconds": float(time.perf_counter() - started),
            "production_state_changes": [],
        }
        _strict_json_file(
            output_path / "native_detection_candidate_receipt.json",
            receipt,
        )
        candidate = NativeCanonicalDetectionCandidate(
            output_path=output_path,
            run_id=normalized_run_id,
            plans=plans,
            manifest=manifest,
            arrays=destination_arrays,
            receipt=receipt,
        )
        candidate_errors = validate_native_canonical_detection_candidate(candidate)
        if candidate_errors:
            raise RuntimeError(
                "Native candidate reopen validation failed: "
                + "; ".join(candidate_errors)
            )
        return candidate
    except Exception as exc:
        run.attrs["status"] = "failed"
        run.attrs["palette_run_completion_status"] = "failed"
        run.attrs["stage_selector_eligible"] = False
        run.attrs["native_candidate_failure"] = str(exc)
        raise


__all__ = [
    "NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_ID",
    "NATIVE_CANONICAL_DETECTION_CANDIDATE_SCHEMA_VERSION",
    "NativeCanonicalDetectionCandidate",
    "validate_native_canonical_detection_candidate",
    "write_native_clipped_detection_candidate",
]

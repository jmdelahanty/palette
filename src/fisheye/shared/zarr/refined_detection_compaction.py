"""Local immutable compaction of frozen refined-detection delta generations.

Compaction is deliberately a copy-on-publish operation: it decodes an exact
immutable base plus a verified frozen delta prefix, resolves a complete
refined-v1 rowset, and writes a fresh selector-ineligible standalone Zarr.
Neither the base, delta lineage, production archive, registry, nor selectors
are mutated by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.benchmark_runtime import (
    local_environment_manifest,
    peak_rss_bytes,
    storage_stats,
    utc_now,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_delta_storage import (
    FrozenRefinedDetectionDeltaGeneration,
    read_frozen_refined_detection_delta_generation,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
    refined_detection_logical_content_digest,
    validate_refined_detection_reason_code_coverage,
    validate_refined_detection_run_manifest,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    RefinedDetectionSnapshotPublication,
    publish_selector_ineligible_refined_detection_snapshot,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)


REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_ID = (
    "palette.refined_detection.compaction_benchmark"
)
REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_VERSION = 1
REFINED_DETECTION_COMPACTION_RECEIPT_NAME = "compaction_benchmark_receipt.json"
DEFAULT_REFINED_DETECTION_COMPACTION_ROOT = Path(
    "/tmp/palette-refined-detection-compactions"
)


class RefinedDetectionCompactionError(ValueError):
    """Raised before or during a fail-closed compaction."""


@dataclass(frozen=True)
class RefinedDetectionCompactionResult:
    """Validated successor publication and its structured timing evidence."""

    publication: RefinedDetectionSnapshotPublication
    frozen_generation: FrozenRefinedDetectionDeltaGeneration
    receipt: Mapping[str, object]


def _manifest_payload(manifest: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    errors = validate_refined_detection_run_manifest(manifest)
    if errors:
        raise RefinedDetectionCompactionError(
            f"{name} manifest is invalid: " + "; ".join(errors)
        )
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise RefinedDetectionCompactionError(f"{name} manifest lacks payload.")
    return payload


def _dimensions_from_manifest(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> RefinedDetectionDimensions:
    logical = payload.get("logical_schema")
    dimensions = logical.get("dimensions") if isinstance(logical, Mapping) else None
    if not isinstance(dimensions, Mapping):
        raise RefinedDetectionCompactionError(f"{name} lacks logical dimensions.")
    try:
        result = RefinedDetectionDimensions(
            n_frames=dimensions["n_frames"],
            n_instances=dimensions["n_instances"],
            n_source_detections=dimensions["n_source_detections"],
            source_width=dimensions["source_width"],
            source_height=dimensions["source_height"],
            lineage_profile=dimensions["lineage_profile"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RefinedDetectionCompactionError(
            f"{name} logical dimensions are invalid."
        ) from exc
    if result.lineage_profile is not RefinedDetectionLineageProfile.FULL_ACQUISITION:
        raise RefinedDetectionCompactionError(
            "Delta compaction currently supports full-acquisition snapshots only."
        )
    return result


def _snapshot_lineage(payload: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    value = payload.get("snapshot_lineage")
    if not isinstance(value, Mapping):
        raise RefinedDetectionCompactionError(f"{name} lacks snapshot_lineage.")
    allocator = value.get("refined_row_id_allocator")
    key_allocator = value.get("manual_instance_key_allocator")
    if not isinstance(allocator, Mapping) or not isinstance(key_allocator, Mapping):
        raise RefinedDetectionCompactionError(f"{name} lineage allocators are invalid.")
    return value


def _source_identity(
    payload: Mapping[str, Any], *, name: str
) -> RefinedDetectionSourceIdentity:
    value = payload.get("source_detection")
    if not isinstance(value, Mapping):
        raise RefinedDetectionCompactionError(f"{name} lacks source_detection.")
    try:
        source = RefinedDetectionSourceIdentity(
            run_id=value["run_id"],
            run_manifest_digest=value["run_manifest_digest"],
            logical_content_digest=value["logical_content_digest"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RefinedDetectionCompactionError(
            f"{name} source_detection is invalid."
        ) from exc
    if source.as_manifest() != dict(value):
        raise RefinedDetectionCompactionError(
            f"{name} source_detection is not canonical."
        )
    return source


def _reason_codes(
    payload: Mapping[str, Any],
    *,
    registry_name: str,
) -> dict[int, str]:
    registries = payload.get("reason_registries")
    registry = (
        registries.get(registry_name) if isinstance(registries, Mapping) else None
    )
    codes = registry.get("codes") if isinstance(registry, Mapping) else None
    if not isinstance(codes, Mapping):
        raise RefinedDetectionCompactionError(
            f"Base manifest lacks {registry_name} reason codes."
        )
    return {int(code): str(label) for code, label in codes.items()}


def _decode_exact_arrays(
    arrays: Mapping[str, Any],
    *,
    dimensions: RefinedDetectionDimensions,
    name: str,
) -> dict[str, np.ndarray]:
    expected = REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    if set(arrays) != set(expected):
        raise RefinedDetectionCompactionError(
            f"{name} arrays do not match the exact refined-v1 path set."
        )
    decoded = {
        path: np.asarray(
            arrays[path] if isinstance(arrays[path], np.ndarray) else arrays[path][...]
        )
        for path in expected
    }
    issues = REFINED_DETECTION_SCHEMA_V1.validate(
        decoded,
        dimensions=dimensions,
    )
    if issues:
        raise RefinedDetectionCompactionError(
            f"{name} arrays violate refined v1: "
            + "; ".join(
                f"{issue.code} at {issue.path}: {issue.message}" for issue in issues
            )
        )
    return decoded


def _receipt_envelope(payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_id": REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": dict(payload),
    }


def validate_refined_detection_compaction_receipt(
    receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the exact sidecar receipt envelope and safety assertions."""

    errors: list[str] = []
    if set(receipt) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("compaction receipt envelope has an unexpected field set")
    if receipt.get("schema_id") != REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_ID:
        errors.append("compaction receipt schema_id mismatch")
    if (
        receipt.get("schema_version")
        != REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_VERSION
    ):
        errors.append("compaction receipt schema_version mismatch")
    if receipt.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("compaction receipt digest algorithm mismatch")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "compaction receipt payload must be an object")
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        return (*errors, f"compaction receipt is not strict JSON: {exc}")
    if receipt.get("payload_digest") != expected_digest:
        errors.append("compaction receipt payload digest mismatch")
    if payload.get("status") != "complete":
        errors.append("compaction receipt status must be complete")
    if payload.get("selector_eligible") is not False:
        errors.append("compaction receipt must remain selector-ineligible")
    if payload.get("production_state_changes") != []:
        errors.append("compaction receipt must report zero production state changes")
    return tuple(errors)


def compact_frozen_refined_detection_delta_generation(
    *,
    delta_root: Any,
    delta_lineage_id: str,
    generation_ordinal: int,
    base_manifest: Mapping[str, Any],
    base_arrays: Mapping[str, Any],
    destination: Path,
    run_id: str,
    snapshot_id: str,
    created_by: str,
    safe_root: Path = DEFAULT_REFINED_DETECTION_COMPACTION_ROOT,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
    parent_logical_content_digest: str | None = None,
) -> RefinedDetectionCompactionResult:
    """Resolve a frozen prefix into a fresh local immutable successor.

    ``base_*`` must be the exact snapshot bound by the delta lineage.  For the
    first compaction, the immediate output parent is that base.  Later prefix
    compactions may provide a newer ``parent_*`` for cross-snapshot retirement
    checks while resolution still starts from the lineage-bound base.
    """

    if not (
        (
            parent_manifest is None
            and parent_arrays is None
            and parent_logical_content_digest is None
        )
        or (
            parent_manifest is not None
            and parent_arrays is not None
            and parent_logical_content_digest is not None
        )
    ):
        raise RefinedDetectionCompactionError(
            "A newer immediate parent requires its manifest, arrays, and "
            "logical-content digest together."
        )
    compaction_started = time.perf_counter()
    rss_before = peak_rss_bytes()
    phase_seconds: dict[str, float] = {}

    phase_started = time.perf_counter()
    base_payload = _manifest_payload(base_manifest, name="base")
    base_dimensions = _dimensions_from_manifest(base_payload, name="base")
    decoded_base = _decode_exact_arrays(
        base_arrays,
        dimensions=base_dimensions,
        name="base",
    )
    reason_errors = validate_refined_detection_reason_code_coverage(
        base_manifest,
        decoded_base,
    )
    if reason_errors:
        raise RefinedDetectionCompactionError(
            "Base reason-code coverage is invalid: " + "; ".join(reason_errors)
        )
    base_lineage = _snapshot_lineage(base_payload, name="base")
    base_allocator = base_lineage["refined_row_id_allocator"]
    base_key_allocator = base_lineage["manual_instance_key_allocator"]
    source = _source_identity(base_payload, name="base")
    instance_reason_codes = _reason_codes(
        base_payload,
        registry_name="instances",
    )
    source_reason_codes = _reason_codes(
        base_payload,
        registry_name="source_detections",
    )
    base_logical_content_digest = refined_detection_logical_content_digest(
        decoded_base,
        dimensions=base_dimensions,
    )
    phase_seconds["read_and_validate_base"] = time.perf_counter() - phase_started

    immediate_parent_manifest = (
        base_manifest if parent_manifest is None else parent_manifest
    )
    immediate_parent_arrays: Mapping[str, Any]
    if parent_arrays is None:
        immediate_parent_arrays = decoded_base
        parent_payload = base_payload
        decoded_parent = decoded_base
        immediate_parent_content_digest = base_logical_content_digest
    else:
        immediate_parent_arrays = parent_arrays
        phase_started = time.perf_counter()
        parent_payload = _manifest_payload(parent_manifest, name="parent")
        parent_dimensions = _dimensions_from_manifest(parent_payload, name="parent")
        decoded_parent = _decode_exact_arrays(
            immediate_parent_arrays,
            dimensions=parent_dimensions,
            name="parent",
        )
        immediate_parent_content_digest = refined_detection_logical_content_digest(
            decoded_parent,
            dimensions=parent_dimensions,
        )
        if immediate_parent_content_digest != parent_logical_content_digest:
            raise RefinedDetectionCompactionError(
                "Immediate parent arrays differ from their supplied logical-content "
                "digest."
            )
        if (
            parent_dimensions.n_frames != base_dimensions.n_frames
            or parent_dimensions.n_source_detections
            != base_dimensions.n_source_detections
            or parent_dimensions.source_width != base_dimensions.source_width
            or parent_dimensions.source_height != base_dimensions.source_height
        ):
            raise RefinedDetectionCompactionError(
                "Immediate parent dimensions differ from the lineage base."
            )
        if _source_identity(parent_payload, name="parent") != source:
            raise RefinedDetectionCompactionError(
                "Immediate parent canonical source differs from the lineage base."
            )
        phase_seconds["read_and_validate_immediate_parent"] = (
            time.perf_counter() - phase_started
        )
    parent_lineage = _snapshot_lineage(parent_payload, name="parent")
    if parent_lineage["lineage_id"] != base_lineage["lineage_id"]:
        raise RefinedDetectionCompactionError(
            "Immediate parent lineage differs from the delta base lineage."
        )

    phase_started = time.perf_counter()
    frozen = read_frozen_refined_detection_delta_generation(
        delta_root,
        delta_lineage_id=delta_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    phase_seconds["read_and_verify_frozen_delta_prefix"] = (
        time.perf_counter() - phase_started
    )
    binding = frozen.binding
    expected_base_path = f"refined_detect_runs/{base_payload['run_id']}"
    if (
        binding.base_run_path != expected_base_path
        or binding.base_snapshot_id != base_lineage["snapshot_id"]
        or binding.base_manifest_digest != base_manifest.get("payload_digest")
        or binding.base_logical_content_digest != base_logical_content_digest
        or binding.recording_identity != base_key_allocator["recording_identity"]
        or binding.base_next_refined_row_id != base_allocator["next_id"]
    ):
        raise RefinedDetectionCompactionError(
            "Frozen delta lineage does not bind the supplied immutable base."
        )

    phase_started = time.perf_counter()
    resolution = frozen.resolve(
        base_dimensions=base_dimensions,
        base_arrays=decoded_base,
        base_instance_reason_codes=instance_reason_codes,
        base_source_reason_codes=source_reason_codes,
    )
    phase_seconds["resolve_sort_and_rebuild_offsets"] = (
        time.perf_counter() - phase_started
    )
    if (
        resolution.report.get("schema_id")
        != "palette.refined_detection.delta_resolution"
        or resolution.report.get("compaction_required") is not True
    ):
        raise RefinedDetectionCompactionError(
            "Delta resolution did not produce a complete resolved snapshot."
        )

    successor_lineage = RefinedDetectionSnapshotLineage(
        lineage_id=str(base_lineage["lineage_id"]),
        snapshot_id=snapshot_id,
        recording_identity=str(base_key_allocator["recording_identity"]),
        next_refined_row_id=resolution.next_refined_row_id,
        parent_run_id=str(parent_payload["run_id"]),
        parent_manifest_digest=str(immediate_parent_manifest["payload_digest"]),
    )
    phase_started = time.perf_counter()
    publication = publish_selector_ineligible_refined_detection_snapshot(
        dimensions=resolution.dimensions,
        arrays=resolution.arrays,
        instance_reason_codes=resolution.instance_reason_codes,
        source_reason_codes=resolution.source_reason_codes,
        destination=destination,
        run_id=run_id,
        lineage=successor_lineage,
        source=source,
        created_by=created_by,
        publication_kind="delta_compaction_local_candidate",
        safe_root=safe_root,
        profile=profile,
        parent_manifest=immediate_parent_manifest,
        parent_arrays=decoded_parent,
        run_attributes={
            "compaction_provenance_state": "sidecar_bound_selector_ineligible",
            "delta_lineage_id": binding.delta_lineage_id,
            "delta_generation_ordinal": frozen.generation_ordinal,
        },
    )
    phase_seconds["publish_validate_immutable_snapshot"] = (
        time.perf_counter() - phase_started
    )

    environment = dict(local_environment_manifest())
    environment.update(
        {
            "storage_tier": "local_noncanonical_scratch",
            "destination_root": str(safe_root.expanduser().resolve()),
            "timing_scope": "single_process_copy_compute_publish_without_network_copy",
        }
    )
    partition_digests = {
        path: str(manifest["payload_digest"])
        for path, manifest in sorted(frozen.partition_manifests.items())
    }
    generation_payload = frozen.generation_manifest["payload"]
    payload: dict[str, object] = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "created_by": str(created_by),
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "local_store": True,
        "base": {
            "run_path": binding.base_run_path,
            "snapshot_id": binding.base_snapshot_id,
            "run_manifest_digest": binding.base_manifest_digest,
            "logical_content_digest": binding.base_logical_content_digest,
        },
        "immediate_parent": {
            "run_id": parent_payload["run_id"],
            "snapshot_id": parent_lineage["snapshot_id"],
            "run_manifest_digest": immediate_parent_manifest["payload_digest"],
            "logical_content_digest": immediate_parent_content_digest,
        },
        "delta": {
            "delta_lineage_id": binding.delta_lineage_id,
            "generation_ordinal": frozen.generation_ordinal,
            "generation_manifest_digest": frozen.generation_manifest["payload_digest"],
            "generation_content_digest": generation_payload[
                "generation_content_digest"
            ],
            "partition_manifest_digests": partition_digests,
            "partition_count_in_prefix": len(partition_digests),
            "event_count_in_prefix": sum(batch.row_count for batch in frozen.batches),
        },
        "output": {
            "path": str(publication.output_path),
            "run_id": publication.run_id,
            "snapshot_id": snapshot_id,
            "run_manifest_digest": publication.manifest["payload_digest"],
            "storage_profile_id": profile.profile_id,
            "n_frames": resolution.dimensions.n_frames,
            "n_instances": resolution.dimensions.n_instances,
            "n_source_detections": resolution.dimensions.n_source_detections,
            "logical_hashes": publication.receipt["logical_hashes"],
            "logical_content_digest": publication.receipt["logical_content_digest"],
            "storage_stats_before_compaction_receipt": storage_stats(
                publication.output_path
            ),
        },
        "resolution_report": dict(resolution.report),
        "phase_seconds": {
            **phase_seconds,
            "snapshot_publication": publication.receipt["phase_seconds"],
            "per_array_write": publication.receipt["per_array_write_seconds"],
            "total_before_receipt": time.perf_counter() - compaction_started,
        },
        "memory": {
            "process_peak_rss_bytes_before": rss_before,
            "process_peak_rss_bytes_after": peak_rss_bytes(),
            "measurement_semantics": "process_lifetime_high_water_not_incremental_rss",
        },
        "environment": environment,
        "validation": {
            "frozen_delta_digests_recomputed": True,
            "base_unchanged_by_api_contract": True,
            "decoded_output_equals_resolution": True,
            "direct_consolidated_metadata_equivalent": True,
            "successor_identity_validated_against_immediate_parent": True,
            "selector_eligible": False,
        },
        "provenance_status": (
            "sidecar_binds_base_delta_and_output;_production_promotion_requires_"
            "manifest_bound_compaction_derivation"
        ),
        "production_state_changes": [],
    }
    receipt = _receipt_envelope(payload)
    receipt_errors = validate_refined_detection_compaction_receipt(receipt)
    if receipt_errors:
        raise RuntimeError(
            "Generated compaction receipt is invalid: " + "; ".join(receipt_errors)
        )
    receipt_path = publication.output_path / REFINED_DETECTION_COMPACTION_RECEIPT_NAME
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(
            receipt,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    return RefinedDetectionCompactionResult(
        publication=publication,
        frozen_generation=frozen,
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_REFINED_DETECTION_COMPACTION_ROOT",
    "REFINED_DETECTION_COMPACTION_RECEIPT_NAME",
    "REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_ID",
    "REFINED_DETECTION_COMPACTION_RECEIPT_SCHEMA_VERSION",
    "RefinedDetectionCompactionError",
    "RefinedDetectionCompactionResult",
    "compact_frozen_refined_detection_delta_generation",
    "validate_refined_detection_compaction_receipt",
]

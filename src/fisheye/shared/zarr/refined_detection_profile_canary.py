"""Paired immutable refined-detection physical-profile canary publication."""

from __future__ import annotations

import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time
from typing import Any, Mapping
import uuid

import zarr

from fisheye.shared.zarr.benchmark_fixture import (
    freeze_tree,
    inventory_tree,
    thaw_tree_for_cleanup,
)
from fisheye.shared.zarr.benchmark_publication import publish_benchmark_candidate
from fisheye.shared.zarr.benchmark_runtime import sha256_file, utc_now
from fisheye.shared.zarr.canonical_detection_shadow import (
    CanonicalDetectionShadowPublication,
    publish_legacy_canonical_detection_shadow,
    validate_canonical_detection_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    validate_refined_detection_publication,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
)
from fisheye.shared.zarr.refined_detection_shadow import (
    _validate_transition_source_matches_canonical,
    publish_refined_detection_shadow,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    refined_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.refined_detection_storage import (
    REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    REFINED_DETECTION_REGULAR_CONTROL_V1,
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_accept_all_refined_detection_root,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)


REFINED_DETECTION_PROFILE_CANARY_SCHEMA_ID = (
    "palette.refined_detection.physical_profile_canary"
)
REFINED_DETECTION_PROFILE_CANARY_SCHEMA_VERSION = 1
_SELECTOR_ATTRIBUTES = {"latest", "latest_complete", "authoritative_run"}
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _write_strict_json(path: Path, value: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _require_safe_destination(destination: Path) -> Path:
    resolved = destination.expanduser().resolve()
    if ".palette_benchmarks" not in resolved.parts:
        raise ValueError("Canary destination must be in .palette_benchmarks.")
    if any(part.endswith("_analysis.zarr") for part in resolved.parts):
        raise ValueError("Canary destination cannot be inside a recording archive.")
    if resolved.exists():
        raise FileExistsError(f"Canary destination already exists: {resolved}")
    return resolved


def _require_safe_scratch_root(scratch_root: Path) -> Path:
    resolved = scratch_root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Scratch root does not exist: {resolved}")
    if resolved in {Path("/").resolve(), Path("/tmp").resolve()}:
        raise ValueError("Scratch root is too broad.")
    if ".palette_benchmarks" in resolved.parts or "groups" in resolved.parts:
        raise ValueError(
            "Scratch root must be node-local, not shared publication data."
        )
    return resolved


def _payload_stats(path: Path) -> dict[str, int]:
    payload_files = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.name not in {"zarr.json", "shadow_publication_receipt.json"}
    ]
    return {
        "payload_object_count": len(payload_files),
        "payload_apparent_bytes": sum(
            int(candidate.stat().st_size) for candidate in payload_files
        ),
    }


def _macos_path(path: Path) -> str | None:
    prefix = "/groups/johnson/johnsonlab"
    value = str(path)
    if not value.startswith(prefix + "/"):
        return None
    return "/Volumes/johnsonlab" + value[len(prefix) :]


def _current_git_identity() -> dict[str, str]:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
    }


def _require_selector_ineligible(
    path: Path,
    *,
    family_name: str,
    run_id: str,
) -> None:
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    family = root[family_name]
    run = family[run_id]
    if _SELECTOR_ATTRIBUTES & set(root.attrs):
        raise RuntimeError(f"Canary root unexpectedly contains a selector: {path}")
    if _SELECTOR_ATTRIBUTES & set(family.attrs):
        raise RuntimeError(f"Canary family unexpectedly contains a selector: {path}")
    if (
        root.attrs.get("benchmark_only") is not True
        or root.attrs.get("selector_eligible") is not False
        or root.attrs.get("registry_registered") is not False
        or run.attrs.get("status") != "complete"
        or run.attrs.get("stage_selector_eligible") is not False
    ):
        raise RuntimeError(f"Canary selector-safety envelope is invalid: {path}")


def _published_canonical(
    path: Path,
    *,
    source: CanonicalDetectionShadowPublication,
) -> CanonicalDetectionShadowPublication:
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    run = root[f"detect_runs/{source.run_id}"]
    arrays = {
        array_path: run[array_path]
        for array_path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
    }
    return CanonicalDetectionShadowPublication(
        output_path=path,
        run_id=source.run_id,
        dimensions=source.dimensions,
        plans=source.plans,
        manifest=dict(run.attrs["run_manifest"]),
        arrays=arrays,
        receipt=_read_strict_json(path / "shadow_publication_receipt.json"),
    )


def _validate_published_refined(
    path: Path,
    *,
    run_id: str,
    transition: Any,
    profile: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plans = plan_refined_detection_storage(
        transition.dimensions,
        profile=profile,
    )
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    run = root[f"refined_detect_runs/{run_id}"]
    arrays = {
        array_path: run[array_path]
        for array_path in REFINED_DETECTION_SCHEMA_V1.binding_paths_for(
            transition.dimensions
        )
    }
    direct, consolidated = refined_detection_metadata_declaration_maps(
        path,
        run_id=run_id,
        plans=plans,
    )
    manifest = dict(run.attrs["run_manifest"])
    errors = validate_refined_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
    )
    if errors:
        raise RuntimeError(
            f"Published refined canary {run_id!r} is invalid: " + "; ".join(errors)
        )
    return manifest, _read_strict_json(path / "shadow_publication_receipt.json")


def publish_refined_detection_profile_canary(
    *,
    source_group_path: Path,
    source_fixture_manifest_path: Path,
    source_run_id: str,
    recording_identity: str,
    destination: Path,
    scratch_root: Path,
    canary_id: str,
    crimson_implementation_commit: str,
    crimson_evidence_commit: str,
    crimson_evidence_sha256: str,
    access_aware_profile: StorageProfile = (
        REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1
    ),
    require_object_gate: bool = True,
) -> dict[str, object]:
    """Build on local scratch, copy back, and validate one physical pair."""

    normalized_canary_id = str(canary_id).strip()
    if not normalized_canary_id or "/" in normalized_canary_id:
        raise ValueError("canary_id must be one nonempty path-safe component.")
    if not _GIT_SHA1_RE.fullmatch(str(crimson_implementation_commit)):
        raise ValueError("crimson_implementation_commit must be a full SHA-1.")
    if not _GIT_SHA1_RE.fullmatch(str(crimson_evidence_commit)):
        raise ValueError("crimson_evidence_commit must be a full SHA-1.")
    if not _SHA256_RE.fullmatch(str(crimson_evidence_sha256)):
        raise ValueError("crimson_evidence_sha256 must be a lowercase SHA-256.")
    if access_aware_profile not in {
        REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
        DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    }:
        raise ValueError(
            "Canary access_aware_profile must be the frozen evidence candidate "
            "or the physically identical promoted detection profile."
        )
    promoted_profile_canary = (
        access_aware_profile == DETECTION_PUBLISHED_ACCESS_AWARE_V1
    )
    final_root = _require_safe_destination(destination)
    scratch = _require_safe_scratch_root(scratch_root)
    source_path = source_group_path.expanduser().resolve()
    fixture_manifest_path = source_fixture_manifest_path.expanduser().resolve()
    fixture_manifest = _read_strict_json(fixture_manifest_path)
    if fixture_manifest.get("benchmark_only") is not True:
        raise ValueError("Source fixture must be explicitly benchmark-only.")
    for field in ("canonical", "registry_registered", "selector_eligible"):
        if fixture_manifest.get(field) is not False:
            raise ValueError(f"Source fixture must declare {field}=false.")
    if (
        fixture_manifest_path.parent
        / str(fixture_manifest.get("copied_zarr_relative_path"))
    ).resolve() != source_path:
        raise ValueError("Source fixture manifest does not bind source_group_path.")
    source_inventory = inventory_tree(source_path)
    declared_inventory = fixture_manifest.get("copied_inventory")
    if not isinstance(declared_inventory, Mapping) or any(
        declared_inventory.get(field) != observed
        for field, observed in (
            ("file_count", source_inventory.file_count),
            ("apparent_bytes", source_inventory.apparent_bytes),
            ("tree_sha256", source_inventory.tree_sha256),
        )
    ):
        raise ValueError("Source fixture inventory no longer matches its manifest.")

    session = scratch / f"palette_refined_profile_canary_{uuid.uuid4().hex}"
    staged_source = session / "source.zarr"
    local_root = session / ".palette_benchmarks" / str(canary_id)
    local_root.mkdir(parents=True)
    started = time.perf_counter()
    copy_started = time.perf_counter()
    shutil.copytree(source_path, staged_source)
    source_copy_seconds = float(time.perf_counter() - copy_started)
    staged_inventory = inventory_tree(staged_source)
    if (
        staged_inventory.file_count != source_inventory.file_count
        or staged_inventory.apparent_bytes != source_inventory.apparent_bytes
        or staged_inventory.tree_sha256 != source_inventory.tree_sha256
    ):
        raise RuntimeError("Node-local staged source differs from shared fixture.")

    canonical_run_id = f"detect_canonical_{normalized_canary_id}"
    regular_run_id = f"refined_detect_regular_{normalized_canary_id}"
    candidate_run_id = f"refined_detect_access_aware_{normalized_canary_id}"
    canonical = publish_legacy_canonical_detection_shadow(
        source_group_path=staged_source,
        source_evidence_group_path=source_path,
        recording_identity=str(recording_identity),
        source_run_id=str(source_run_id),
        destination=local_root / "canonical_source.zarr",
        run_id=canonical_run_id,
        shadow_root=local_root,
    )
    transition = build_accept_all_refined_detection_root(
        canonical.arrays,
        dimensions=canonical.dimensions,
        recording_identity=str(recording_identity),
    )
    lineage = RefinedDetectionSnapshotLineage(
        lineage_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{recording_identity}:refined_detection_v1_profile_canary",
            )
        ),
        snapshot_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{recording_identity}:{normalized_canary_id}:logical_snapshot",
            )
        ),
        recording_identity=str(recording_identity),
        next_refined_row_id=transition.dimensions.n_instances,
    )
    regular = publish_refined_detection_shadow(
        transition,
        destination=local_root / "regular.zarr",
        run_id=regular_run_id,
        lineage=lineage,
        canonical_source=canonical,
        shadow_root=local_root,
        profile=REFINED_DETECTION_REGULAR_CONTROL_V1,
    )
    candidate = publish_refined_detection_shadow(
        transition,
        destination=local_root / "access_aware.zarr",
        run_id=candidate_run_id,
        lineage=lineage,
        canonical_source=canonical,
        shadow_root=local_root,
        profile=access_aware_profile,
    )
    if regular.receipt["logical_hashes"] != candidate.receipt["logical_hashes"]:
        raise RuntimeError("Local regular/candidate decoded hashes differ.")

    final_paths = {
        "canonical": final_root / "candidates/canonical_source.zarr",
        "regular": final_root / "candidates/regular.zarr",
        "access_aware": final_root / "candidates/access_aware.zarr",
    }
    publications = {
        "canonical": publish_benchmark_candidate(
            source=canonical.output_path,
            destination=final_paths["canonical"],
            workflow_root=final_root,
        ),
        "regular": publish_benchmark_candidate(
            source=regular.output_path,
            destination=final_paths["regular"],
            workflow_root=final_root,
        ),
        "access_aware": publish_benchmark_candidate(
            source=candidate.output_path,
            destination=final_paths["access_aware"],
            workflow_root=final_root,
        ),
    }

    published_canonical = _published_canonical(
        final_paths["canonical"],
        source=canonical,
    )
    canonical_errors = validate_canonical_detection_shadow_publication(
        published_canonical
    )
    if canonical_errors:
        raise RuntimeError(
            "Published canonical canary is invalid: " + "; ".join(canonical_errors)
        )
    regular_manifest, regular_receipt = _validate_published_refined(
        final_paths["regular"],
        run_id=regular_run_id,
        transition=transition,
        profile=REFINED_DETECTION_REGULAR_CONTROL_V1,
    )
    candidate_manifest, candidate_receipt = _validate_published_refined(
        final_paths["access_aware"],
        run_id=candidate_run_id,
        transition=transition,
        profile=access_aware_profile,
    )
    if regular_receipt["logical_hashes"] != candidate_receipt["logical_hashes"]:
        raise RuntimeError("Published regular/candidate decoded hashes differ.")
    _require_selector_ineligible(
        final_paths["canonical"],
        family_name="detect_runs",
        run_id=canonical_run_id,
    )
    _require_selector_ineligible(
        final_paths["regular"],
        family_name="refined_detect_runs",
        run_id=regular_run_id,
    )
    _require_selector_ineligible(
        final_paths["access_aware"],
        family_name="refined_detect_runs",
        run_id=candidate_run_id,
    )
    for refined_path in (final_paths["regular"], final_paths["access_aware"]):
        source_errors = _validate_transition_source_matches_canonical(
            transition,
            published_canonical,
        )
        if source_errors:
            raise RuntimeError(
                f"Published refined source binding failed for {refined_path}: "
                + "; ".join(source_errors)
            )

    regular_stats = _payload_stats(final_paths["regular"])
    candidate_stats = _payload_stats(final_paths["access_aware"])
    regular_plans = plan_refined_detection_storage(
        transition.dimensions,
        profile=REFINED_DETECTION_REGULAR_CONTROL_V1,
    )
    candidate_plans = plan_refined_detection_storage(
        transition.dimensions,
        profile=access_aware_profile,
    )
    planned_object_ratio = (
        candidate_plans.estimated_payload_objects
        / regular_plans.estimated_payload_objects
    )
    actual_object_ratio = (
        candidate_stats["payload_object_count"] / regular_stats["payload_object_count"]
    )
    if require_object_gate and (
        planned_object_ratio > 0.25 or actual_object_ratio > 0.25
    ):
        raise RuntimeError(
            "Access-aware canary did not achieve the required 4x payload-object "
            "reduction."
        )
    payload: dict[str, object] = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "canary_id": normalized_canary_id,
        "purpose": "paired_refined_detection_physical_profile_gate",
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "profile_promoted": promoted_profile_canary,
        "palette": _current_git_identity(),
        "crimson_prerequisite": {
            "implementation_commit": str(crimson_implementation_commit),
            "evidence_commit": str(crimson_evidence_commit),
            "evidence_sha256": str(crimson_evidence_sha256),
            "real_shadow_gate": "pass",
        },
        "source_fixture": {
            "path": str(source_path),
            "fixture_manifest_path": str(fixture_manifest_path),
            "fixture_manifest_sha256": sha256_file(fixture_manifest_path),
            "source_run_id": str(source_run_id),
            "recording_identity": str(recording_identity),
            "inventory": source_inventory.as_manifest(),
            "node_local_copy_verified": True,
        },
        "logical_snapshot": {
            "construction": "canonical_detection_v1_accept_all_root",
            "dimensions": transition.dimensions.as_manifest(),
            "instance_reason_codes": {"0": "none"},
            "source_reason_codes": {"0": "none"},
            "logical_hashes": regular_receipt["logical_hashes"],
            "lineage_id": lineage.lineage_id,
            "snapshot_id": lineage.snapshot_id,
        },
        "artifacts": {
            "canonical": {
                "server_path": str(final_paths["canonical"]),
                "macos_path": _macos_path(final_paths["canonical"]),
                "run_id": canonical_run_id,
                "manifest_digest": published_canonical.manifest["payload_digest"],
                "publication": publications["canonical"],
                **_payload_stats(final_paths["canonical"]),
            },
            "regular": {
                "server_path": str(final_paths["regular"]),
                "macos_path": _macos_path(final_paths["regular"]),
                "run_id": regular_run_id,
                "manifest_digest": regular_manifest["payload_digest"],
                "storage_profile_id": regular_plans.profile.profile_id,
                "planned_payload_objects": regular_plans.estimated_payload_objects,
                "publication": publications["regular"],
                **regular_stats,
            },
            "access_aware": {
                "server_path": str(final_paths["access_aware"]),
                "macos_path": _macos_path(final_paths["access_aware"]),
                "run_id": candidate_run_id,
                "manifest_digest": candidate_manifest["payload_digest"],
                "storage_profile_id": candidate_plans.profile.profile_id,
                "planned_payload_objects": candidate_plans.estimated_payload_objects,
                "publication": publications["access_aware"],
                **candidate_stats,
            },
        },
        "pair_validation": {
            "exact_decoded_logical_hash_equality": True,
            "canonical_source_audit_equality": True,
            "direct_consolidated_metadata_equivalence": True,
            "codec_and_crc_validation": True,
            "offsets_begin_at_zero_and_end_at_row_count": True,
            "selector_attributes_absent": True,
            "production_state_changes": [],
            "payload_object_gate_required": bool(require_object_gate),
            "payload_object_gate_passed": (
                planned_object_ratio <= 0.25 and actual_object_ratio <= 0.25
            ),
            "planned_payload_object_ratio_candidate_over_regular": (
                planned_object_ratio
            ),
            "actual_payload_object_ratio_candidate_over_regular": (actual_object_ratio),
        },
        "construction": {
            "source_to_node_local_copy_seconds": source_copy_seconds,
            "node_local_source_inventory": staged_inventory.as_manifest(),
            "workflow_seconds": float(time.perf_counter() - started),
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "publication_lifecycle": (
                "shared_source_to_node_local_scratch_compute_then_verified_"
                "atomic_candidate_copyback"
            ),
        },
        "gate": {
            "palette_publication_gate": "pass",
            "crimson_physical_measurement_required": not promoted_profile_canary,
            "promotion_decision": (
                "promoted_profile_verification"
                if promoted_profile_canary
                else "deferred"
            ),
        },
    }
    envelope: dict[str, object] = {
        "schema_id": REFINED_DETECTION_PROFILE_CANARY_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_PROFILE_CANARY_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
    }
    _write_strict_json(final_root / "canary_manifest.json", envelope)
    freeze_tree(final_root)
    thaw_tree_for_cleanup(session)
    shutil.rmtree(session)
    return envelope


__all__ = [
    "REFINED_DETECTION_PROFILE_CANARY_SCHEMA_ID",
    "REFINED_DETECTION_PROFILE_CANARY_SCHEMA_VERSION",
    "publish_refined_detection_profile_canary",
]

"""Atomically publish one immutable reviewed training-artifact candidate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
from uuid import uuid4

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    tree_inventory,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.keypoint_manual_review_qc import (
    manual_keypoint_qc_policy_from_manifest,
    validate_manual_keypoint_review_derivation,
)
from fisheye.shared.refined_subject_mask_mutation import (
    REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
    REFINED_SUBJECT_MASK_SEALED_SNAPSHOT,
    refined_subject_mask_lifecycle_state,
    require_approved_refined_subject_mask_review,
    stamp_refined_subject_mask_sealed_snapshot,
)
from fisheye.shared.zarr.benchmark_runtime import utc_now
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
    validate_refined_keypoint_run_manifest,
)
from fisheye.shared.zarr.training_review_artifact_publication import (
    TRAINING_REVIEW_ARTIFACT_SCHEMA_ID,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
    open_zarr_group_direct,
)


REVIEWED_TRAINING_ARTIFACT_SCHEMA_ID = "palette.reviewed_training_artifact"
REVIEWED_TRAINING_ARTIFACT_SCHEMA_VERSION = 1
REVIEWED_TRAINING_ARTIFACT_RECEIPT = "reviewed_training_artifact_receipt.json"
KEYPOINT_COMPACTION_RECEIPT_SCHEMA_ID = "palette.refined_keypoint.delta_compaction"
KEYPOINT_COMPACTION_RECEIPT_SCHEMA_VERSION = 2


def _safe_run_id(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text or "/" in text or text.startswith("."):
        raise ValueError(f"{name} must be one safe non-hidden group name.")
    return text


def _bounded_node_local_scratch(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Node-local scratch root not found: {resolved}")
    if resolved in {
        Path("/").resolve(),
        Path("/tmp").resolve(),
        Path("/scratch").resolve(),
    } or str(resolved).startswith(("/groups/", "/nrs/")):
        raise ValueError(
            "Reviewed training publication requires one bounded node-local "
            "scratch directory, not a shared filesystem or broad root."
        )
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _validated_compaction_receipt(
    *,
    source_archive: Path,
    compacted_archive: Path,
    compacted_run_id: str,
) -> dict[str, Any]:
    receipt_path = compacted_archive.with_name(
        compacted_archive.name + ".compaction_receipt.json"
    )
    receipt = _read_json_object(receipt_path)
    if set(receipt) != {"schema_id", "schema_version", "payload_digest", "payload"}:
        raise ValueError("Keypoint compaction receipt envelope is not exact.")
    if (
        receipt.get("schema_id") != KEYPOINT_COMPACTION_RECEIPT_SCHEMA_ID
        or receipt.get("schema_version") != KEYPOINT_COMPACTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("Keypoint compaction receipt schema is unsupported.")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Keypoint compaction receipt payload is missing.")
    if receipt.get("payload_digest") != canonical_json_sha256(payload):
        raise ValueError("Keypoint compaction receipt digest is invalid.")
    expected_payload_fields = {
        "status",
        "created_at_utc",
        "created_by",
        "source_archive",
        "base",
        "delta",
        "output",
        "production_state_changes",
    }
    if set(payload) != expected_payload_fields or payload.get("status") != "complete":
        raise ValueError("Keypoint compaction receipt payload is not exact.")
    base = payload.get("base")
    delta = payload.get("delta")
    output = payload.get("output")
    if (
        not isinstance(base, Mapping)
        or set(base) != {"run_path", "run_id", "manifest_digest", "snapshot_id"}
        or not isinstance(delta, Mapping)
        or set(delta)
        != {
            "delta_run",
            "generation",
            "generation_sha256",
            "overlay_sha256",
            "partition_count",
            "event_count",
            "review_qc_policy_digest",
            "review_derivation",
        }
        or not isinstance(output, Mapping)
        or set(output)
        != {
            "path",
            "run_id",
            "manifest_digest",
            "snapshot_id",
            "edited_instance_keys",
            "stage_selector_eligible",
        }
        or payload.get("production_state_changes") != []
    ):
        raise ValueError("Keypoint compaction receipt nested envelope is not exact.")
    derivation_errors = validate_manual_keypoint_review_derivation(
        delta.get("review_derivation")
    )
    if derivation_errors:
        raise ValueError(
            "Keypoint compaction receipt review derivation is invalid: "
            + "; ".join(derivation_errors)
        )
    review_policy = manual_keypoint_qc_policy_from_manifest(
        delta["review_derivation"].get("review_qc_policy")
    )
    derivation = delta["review_derivation"]
    if (
        delta.get("review_qc_policy_digest") != review_policy.policy_digest
        or derivation.get("delta_run") != delta.get("delta_run")
        or derivation.get("generation") != delta.get("generation")
        or derivation.get("generation_sha256") != delta.get("generation_sha256")
        or derivation.get("overlay_sha256") != delta.get("overlay_sha256")
        or derivation.get("partition_count") != delta.get("partition_count")
        or derivation.get("event_count") != delta.get("event_count")
        or derivation.get("base_run_path") != base.get("run_path")
    ):
        raise ValueError(
            "Keypoint compaction receipt QC/derivation evidence is inconsistent."
        )
    if (
        Path(str(payload.get("source_archive") or "")).expanduser().resolve()
        != source_archive
        or Path(str(output.get("path") or "")).expanduser().resolve()
        != compacted_archive
        or output.get("run_id") != compacted_run_id
        or output.get("stage_selector_eligible") is not False
    ):
        raise ValueError(
            "Keypoint compaction receipt does not bind the requested inputs."
        )
    return dict(receipt)


def _validate_source_and_compaction(
    *,
    source_archive: Path,
    compacted_archive: Path,
    compacted_run_id: str,
    refined_subject_mask_run_id: str,
) -> dict[str, Any]:
    source = open_zarr_group_direct(source_archive, mode="r")
    review_receipt = source.attrs.get("training_review_artifact")
    if (
        not isinstance(review_receipt, Mapping)
        or review_receipt.get("schema_id") != TRAINING_REVIEW_ARTIFACT_SCHEMA_ID
        or source.attrs.get("training_artifact_status") != "review_active"
        or source.attrs.get("stage_selector_eligible") is not False
    ):
        raise ValueError("Source is not an active selector-ineligible review artifact.")
    review_payload = review_receipt.get("payload")
    if not isinstance(review_payload, Mapping) or review_receipt.get(
        "payload_digest"
    ) != canonical_json_sha256(review_payload):
        raise ValueError("Source review-artifact receipt digest is invalid.")

    compaction_receipt = _validated_compaction_receipt(
        source_archive=source_archive,
        compacted_archive=compacted_archive,
        compacted_run_id=compacted_run_id,
    )
    compaction_payload = compaction_receipt["payload"]
    base_evidence = compaction_payload["base"]
    delta_evidence = compaction_payload["delta"]
    base_run_path = str(base_evidence["run_path"])
    base_run = source[base_run_path]
    base_manifest = base_run.attrs.get(REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if (
        not isinstance(base_manifest, Mapping)
        or base_evidence["run_id"] != base_run_path.rsplit("/", 1)[-1]
        or base_evidence["manifest_digest"] != canonical_json_sha256(base_manifest)
        or base_manifest.get("payload", {})
        .get("snapshot_identity", {})
        .get("snapshot_id")
        != base_evidence["snapshot_id"]
    ):
        raise ValueError(
            "Compaction receipt base evidence differs from the review source."
        )
    generation = source[
        "edit_delta_runs/"
        f"{delta_evidence['delta_run']}/generations/{delta_evidence['generation']}"
    ]
    if (
        generation.attrs.get("status") != "frozen"
        or generation.attrs.get("generation_sha256")
        != delta_evidence["generation_sha256"]
        or generation.attrs.get("review_qc_policy_digest")
        != delta_evidence["review_qc_policy_digest"]
        or generation.attrs.get("review_qc_policy")
        != delta_evidence["review_derivation"]["review_qc_policy"]
    ):
        raise ValueError("Compaction receipt delta generation differs from the source.")
    compacted = open_zarr_group_direct(compacted_archive, mode="r")
    compacted_run = compacted[f"refined_keypoints_runs/{compacted_run_id}"]
    compacted_manifest = compacted_run.attrs.get(
        REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE
    )
    if not isinstance(compacted_manifest, Mapping):
        raise ValueError("Compacted refined-keypoint run has no strict manifest.")
    manifest_errors = validate_refined_keypoint_run_manifest(compacted_manifest)
    if manifest_errors:
        raise ValueError(
            "Compacted refined-keypoint manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    compaction_output = compaction_receipt["payload"]["output"]
    if (
        compacted_run.attrs.get("status") != "complete"
        or compacted_run.attrs.get("stage_selector_eligible") is not False
        or compacted_run.attrs.get("artifact_mutability") != "immutable_snapshot"
        or compaction_output.get("manifest_digest")
        != canonical_json_sha256(compacted_manifest)
        or compacted_run.attrs.get("review_derivation")
        != delta_evidence["review_derivation"]
    ):
        raise ValueError("Compacted refined-keypoint run differs from its receipt.")

    mask_run = source[f"refined_subject_masks_runs/{refined_subject_mask_run_id}"]
    if (
        refined_subject_mask_lifecycle_state(mask_run)
        != REFINED_SUBJECT_MASK_EDITABLE_DRAFT
        or mask_run.attrs.get("stage_selector_eligible") is not False
        or "masks_roi" not in mask_run
    ):
        raise ValueError("Refined subject-mask source is not a dense editable draft.")
    require_approved_refined_subject_mask_review(mask_run)

    return {
        "review_receipt_digest": str(review_receipt["payload_digest"]),
        "compaction_receipt": compaction_receipt,
        "compacted_manifest_digest": canonical_json_sha256(compacted_manifest),
        "mask_run": refined_subject_mask_run_id,
    }


def _validate_reviewed_candidate(
    archive: Path,
    *,
    compacted_run_id: str,
    compacted_manifest_digest: str,
    refined_subject_mask_run_id: str,
) -> dict[str, Any]:
    root = open_zarr_group_direct(archive, mode="r")
    if (
        root.attrs.get("training_artifact_status") != "reviewed_immutable_candidate"
        or root.attrs.get("artifact_mutability") != "immutable_snapshot"
        or root.attrs.get("stage_selector_eligible") is not False
        or root.attrs.get("metadata_read_mode") != "consolidated_immutable_publication"
    ):
        raise RuntimeError("Reviewed training candidate root state is invalid.")
    keypoints = root[f"refined_keypoints_runs/{compacted_run_id}"]
    manifest = keypoints.attrs.get(REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
    if (
        not isinstance(manifest, Mapping)
        or canonical_json_sha256(manifest) != compacted_manifest_digest
        or keypoints.attrs.get("artifact_mutability") != "immutable_snapshot"
        or keypoints.attrs.get("stage_selector_eligible") is not False
    ):
        raise RuntimeError("Reviewed candidate keypoint successor is invalid.")
    masks = root[f"refined_subject_masks_runs/{refined_subject_mask_run_id}"]
    if (
        refined_subject_mask_lifecycle_state(masks)
        != REFINED_SUBJECT_MASK_SEALED_SNAPSHOT
        or masks.attrs.get("stage_selector_eligible") is not False
        or "masks_roi" not in masks
    ):
        raise RuntimeError("Reviewed candidate dense subject masks are not sealed.")
    require_approved_refined_subject_mask_review(masks)
    root_document = _read_json_object(archive / "zarr.json")
    consolidated = root_document.get("consolidated_metadata")
    metadata = (
        consolidated.get("metadata") if isinstance(consolidated, Mapping) else None
    )
    if not isinstance(metadata, Mapping):
        raise RuntimeError("Reviewed training candidate lacks consolidated metadata.")
    keypoint_declaration = metadata.get(f"refined_keypoints_runs/{compacted_run_id}")
    mask_declaration = metadata.get(
        f"refined_subject_masks_runs/{refined_subject_mask_run_id}"
    )
    if not isinstance(keypoint_declaration, Mapping) or not isinstance(
        mask_declaration, Mapping
    ):
        raise RuntimeError(
            "Reviewed candidate consolidated run declarations are missing."
        )
    keypoint_attrs = keypoint_declaration.get("attributes")
    mask_attrs = mask_declaration.get("attributes")
    consolidated_manifest = (
        keypoint_attrs.get(REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE)
        if isinstance(keypoint_attrs, Mapping)
        else None
    )
    if (
        not isinstance(keypoint_attrs, Mapping)
        or not isinstance(consolidated_manifest, Mapping)
        or canonical_json_sha256(consolidated_manifest) != compacted_manifest_digest
        or not isinstance(mask_attrs, Mapping)
        or mask_attrs.get("refined_subject_mask_lifecycle")
        != masks.attrs.get("refined_subject_mask_lifecycle")
    ):
        raise RuntimeError("Reviewed candidate consolidated declarations are stale.")
    receipt = root.attrs.get("reviewed_training_artifact")
    if not isinstance(receipt, Mapping) or receipt.get("schema_id") != (
        REVIEWED_TRAINING_ARTIFACT_SCHEMA_ID
    ):
        raise RuntimeError("Reviewed training candidate receipt is missing.")
    payload = receipt.get("payload")
    if not isinstance(payload, Mapping) or receipt.get(
        "payload_digest"
    ) != canonical_json_sha256(payload):
        raise RuntimeError("Reviewed training candidate receipt digest is invalid.")
    return {
        "compacted_refined_keypoint_run": compacted_run_id,
        "sealed_refined_subject_mask_run": refined_subject_mask_run_id,
        "receipt_digest": str(receipt["payload_digest"]),
        "stage_selector_eligible": False,
    }


def publish_reviewed_training_artifact_candidate(
    *,
    source_review_archive: Path,
    compacted_keypoint_archive: Path,
    compacted_keypoint_run_id: str,
    refined_subject_mask_run_id: str,
    destination: Path,
    scratch_root: Path,
    created_by: str,
) -> Mapping[str, Any]:
    """Publish one immutable copy without mutating the active review artifact."""

    source = source_review_archive.expanduser().resolve()
    compacted = compacted_keypoint_archive.expanduser().resolve()
    target = destination.expanduser().resolve()
    scratch = _bounded_node_local_scratch(scratch_root)
    keypoint_run = _safe_run_id(
        compacted_keypoint_run_id, name="compacted_keypoint_run_id"
    )
    mask_run = _safe_run_id(
        refined_subject_mask_run_id, name="refined_subject_mask_run_id"
    )
    if not source.is_dir() or not compacted.is_dir():
        raise FileNotFoundError(
            "Source review and compacted keypoint archives are required."
        )
    if target.suffix != ".zarr":
        raise ValueError("Reviewed training destination must end in .zarr.")
    if target.exists():
        raise FileExistsError(f"Reviewed training destination exists: {target}")
    if not str(created_by).strip():
        raise ValueError("created_by cannot be empty.")

    evidence = _validate_source_and_compaction(
        source_archive=source,
        compacted_archive=compacted,
        compacted_run_id=keypoint_run,
        refined_subject_mask_run_id=mask_run,
    )
    source_inventory_before = tree_inventory(source, hash_content=True)

    with tempfile.TemporaryDirectory(
        prefix="palette-reviewed-training-artifact-", dir=str(scratch)
    ) as temporary:
        local = Path(temporary) / target.name
        shutil.copytree(source, local)
        source_inventory_after = tree_inventory(source, hash_content=True)
        local_source_inventory = tree_inventory(local, hash_content=True)
        if (
            source_inventory_before != source_inventory_after
            or source_inventory_before != local_source_inventory
        ):
            raise RuntimeError(
                "Review artifact changed while it was being snapshotted."
            )

        local_keypoint_parent = local / "refined_keypoints_runs"
        local_keypoint_run = local_keypoint_parent / keypoint_run
        if local_keypoint_run.exists():
            raise FileExistsError(
                f"Reviewed artifact already contains refined keypoint run {keypoint_run}."
            )
        shutil.copytree(
            compacted / "refined_keypoints_runs" / keypoint_run,
            local_keypoint_run,
        )
        source_compacted_inventory = tree_inventory(
            compacted / "refined_keypoints_runs" / keypoint_run,
            hash_content=True,
        )
        if tree_inventory(local_keypoint_run, hash_content=True) != (
            source_compacted_inventory
        ):
            raise RuntimeError("Imported compacted keypoint run changed during copy.")

        local_root = open_zarr_group_direct(local, mode="a")
        local_masks = local_root[f"refined_subject_masks_runs/{mask_run}"]
        stamp_refined_subject_mask_sealed_snapshot(local_masks)
        payload = json_attr_safe(
            {
                "status": "reviewed_immutable_candidate",
                "created_at_utc": utc_now(),
                "created_by": str(created_by),
                "source_review_archive": str(source),
                "source_review_receipt_digest": evidence["review_receipt_digest"],
                "keypoint_compaction": {
                    "archive": str(compacted),
                    "run_id": keypoint_run,
                    "manifest_digest": evidence["compacted_manifest_digest"],
                    "receipt_digest": evidence["compaction_receipt"]["payload_digest"],
                },
                "sealed_refined_subject_mask_run": mask_run,
                "artifact_mutability": "immutable_snapshot",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "production_state_changes": [],
            }
        )
        receipt = {
            "schema_id": REVIEWED_TRAINING_ARTIFACT_SCHEMA_ID,
            "schema_version": REVIEWED_TRAINING_ARTIFACT_SCHEMA_VERSION,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        local_root.attrs.update(
            {
                "training_artifact_status": "reviewed_immutable_candidate",
                "artifact_mutability": "immutable_snapshot",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "metadata_read_mode": "consolidated_immutable_publication",
                "reviewed_training_artifact": receipt,
            }
        )
        write_json_atomic(local / REVIEWED_TRAINING_ARTIFACT_RECEIPT, receipt)
        consolidate_metadata_capture_expected_warnings(local)
        local_state = _validate_reviewed_candidate(
            local,
            compacted_run_id=keypoint_run,
            compacted_manifest_digest=str(evidence["compacted_manifest_digest"]),
            refined_subject_mask_run_id=mask_run,
        )
        local_inventory = tree_inventory(local, hash_content=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        hidden = target.with_name(
            f".{target.name}.publish_tmp.{os.getpid()}.{uuid4().hex}"
        )
        with archive_metadata_publication_lock(target):
            if target.exists() or hidden.exists():
                raise FileExistsError(
                    f"Reviewed training target became occupied: {target}"
                )
            try:
                shutil.copytree(local, hidden)
                if tree_inventory(hidden, hash_content=True) != local_inventory:
                    raise RuntimeError("Published reviewed artifact copy changed.")
                _validate_reviewed_candidate(
                    hidden,
                    compacted_run_id=keypoint_run,
                    compacted_manifest_digest=str(
                        evidence["compacted_manifest_digest"]
                    ),
                    refined_subject_mask_run_id=mask_run,
                )
                hidden.rename(target)
            except BaseException:
                if hidden.exists():
                    shutil.rmtree(hidden)
                raise

    final_state = _validate_reviewed_candidate(
        target,
        compacted_run_id=keypoint_run,
        compacted_manifest_digest=str(evidence["compacted_manifest_digest"]),
        refined_subject_mask_run_id=mask_run,
    )
    return {
        "status": "complete",
        "destination": str(target),
        **local_state,
        "final_receipt_digest": final_state["receipt_digest"],
        "source_review_artifact_mutated": False,
        "stage_selector_eligible": False,
        "production_state_changes": [],
    }


__all__ = [
    "REVIEWED_TRAINING_ARTIFACT_RECEIPT",
    "REVIEWED_TRAINING_ARTIFACT_SCHEMA_ID",
    "REVIEWED_TRAINING_ARTIFACT_SCHEMA_VERSION",
    "publish_reviewed_training_artifact_candidate",
]

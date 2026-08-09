"""Publish one self-contained, mutable training-review artifact atomically.

The scientific prediction bases remain immutable.  Keypoint review writes to
an instance-key-bound delta generation, while refined subject masks use their
canonical dense editable-draft lifecycle.  The source training artifact and
all production selectors remain untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np

from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    tree_inventory,
)
from fisheye.refinement.finalize_subject_masks import finalize_subject_masks
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.keypoint_manual_review_qc import (
    build_default_manual_keypoint_qc_policy,
    manual_keypoint_qc_policy_from_manifest,
)
from fisheye.shared.mask_probability_encoding import (
    PROBABILITY_ENCODING_LINEAR_UINT8,
    probabilities_encoding_from_attrs,
)
from fisheye.shared.refined_subject_mask_mutation import (
    REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
    refined_subject_mask_lifecycle_state,
)
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.tabular_deltas import create_delta_generation, instance_key_digest
from fisheye.shared.zarr.benchmark_runtime import sha256_array, sha256_file, utc_now
from fisheye.shared.zarr.keypoint_bundle_production_publication import (
    publish_keypoint_v2_production_candidate_chain,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    initial_refined_keypoint_snapshot_identity,
)
from fisheye.shared.zarr.training_keypoint_review_publication import (
    publish_training_keypoint_review_candidate_chain,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_FAMILY,
    SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE,
    publish_subject_mask_bundle_candidate,
    validate_subject_mask_bundle_candidate,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    publish_selector_ineligible_subject_mask_core_snapshot,
    validate_persisted_subject_mask_core_publication,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)
from fisheye.shared.zarr_helpers import (
    archive_metadata_publication_lock,
    consolidate_metadata_capture_expected_warnings,
    open_zarr_group_direct,
)
from fisheye.shared.zarr_run_completion import is_run_complete


TRAINING_REVIEW_ARTIFACT_SCHEMA_ID = "palette.training_review_artifact"
TRAINING_REVIEW_ARTIFACT_SCHEMA_VERSION = 2
TRAINING_REVIEW_ARTIFACT_RECEIPT = "training_review_artifact_receipt.json"
TRAINING_REVIEW_ARTIFACT_POLICY = (
    "node_local_candidate_bases_plus_editable_surfaces_then_atomic_tree_v1"
)
TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID = (
    "palette.training_keypoint_review_artifact"
)
TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_VERSION = 1
TRAINING_KEYPOINT_REVIEW_ARTIFACT_RECEIPT = (
    "training_keypoint_review_artifact_receipt.json"
)
TRAINING_KEYPOINT_REVIEW_ARTIFACT_POLICY = (
    "node_local_immutable_keypoint_bases_plus_instance_key_deltas_v1"
)


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
            "Training review publication requires one bounded node-local scratch "
            "directory, not a shared filesystem or broad root."
        )
    return resolved


def _metadata_sha256(archive: Path, run_path: str) -> str:
    path = archive / run_path / "zarr.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    return sha256_file(path)


def _recording_identity(archive: Path) -> str:
    root = open_zarr_group_direct(archive, mode="r")
    value = str(root.attrs.get("recording_id") or "").strip()
    if not value:
        raise ValueError("Training artifact lacks recording_id.")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise ValueError("Source must be a training-purpose Zarr artifact.")
    return value


def _initial_refined_identity(
    *,
    recording_identity: str,
    terminal_metadata_sha256: str,
    refined_run_id: str,
) -> Any:
    lineage_id = str(
        uuid5(
            NAMESPACE_URL,
            f"palette:{recording_identity}:training-keypoint-review-lineage-v1",
        )
    )
    snapshot_id = str(
        uuid5(
            NAMESPACE_URL,
            "palette:training-keypoint-review-snapshot-v1:"
            f"{recording_identity}:{terminal_metadata_sha256}:{refined_run_id}",
        )
    )
    return initial_refined_keypoint_snapshot_identity(
        recording_identity=recording_identity,
        lineage_id=lineage_id,
        snapshot_id=snapshot_id,
    )


def _publish_subject_mask_contract_chain(
    *,
    archive: Path,
    working_root: Path,
    recording_identity: str,
    crop_run_id: str,
    terminal_run_id: str,
    terminal_metadata_sha256: str,
    raw_run_id: str,
    canonical_refined_run_id: str,
    quality_run_id: str,
    bundle_id: str,
    editable_refined_run_id: str,
    created_by: str,
) -> dict[str, Any]:
    """Publish raw/refined/quality authorities beside one editable draft."""

    root = open_zarr_group_direct(archive, mode="r")
    crop = root[f"crop_runs/{crop_run_id}"]
    terminal = root[f"subject_mask_shard_runs/{terminal_run_id}"]
    editable = root[f"refined_subject_masks_runs/{editable_refined_run_id}"]
    raw_labels = tuple(str(value) for value in terminal.attrs.get("mask_labels", ()))
    refined_labels = tuple(
        str(value) for value in editable.attrs.get("mask_labels", ())
    )
    if not raw_labels or not refined_labels:
        raise ValueError("Subject-mask terminal/refined component labels are absent.")
    raw_components = SubjectMaskComponentRegistry(raw_labels)
    refined_components = SubjectMaskComponentRegistry(refined_labels)

    source_frames = np.asarray(
        crop["source_acquisition_frame_index"][:], dtype=np.int64
    )
    source_frame_count = int(root["raw_video"].attrs.get("source_frame_count") or 0)
    if source_frame_count <= 0 or (
        source_frames.size and int(source_frames.max()) >= source_frame_count
    ):
        raise ValueError(
            "Training artifact lacks a valid source-video frame domain for masks."
        )
    source_crop_row_ids = np.asarray(terminal["source_crop_row_ids"][:], dtype=np.int64)
    instance_keys = np.asarray(terminal["instance_key"][:], dtype=np.uint64)
    crop_instance_keys = np.asarray(crop["instance_key"][:], dtype=np.uint64)
    source_crop_xywh = np.asarray(crop["source_crop_xywh"][:], dtype=np.float32)
    if not np.array_equal(instance_keys, crop_instance_keys):
        raise ValueError(
            "Terminal masks and crop geometry have different instance keys."
        )
    if not np.array_equal(
        source_frames,
        np.asarray(terminal["source_acquisition_frame_index"][:], dtype=np.int64),
    ):
        raise ValueError(
            "Terminal masks and crop geometry have different source frames."
        )
    if not np.array_equal(
        source_crop_row_ids,
        np.arange(instance_keys.shape[0], dtype=np.int64),
    ):
        raise ValueError(
            "Training review mask publication requires terminal rows to cover the "
            "complete crop rowset in exact order."
        )
    frame_row_offsets = derive_subject_mask_frame_row_offsets(
        source_frames,
        n_frames=source_frame_count,
    )
    crop_arrays = {
        "instance_key": instance_keys,
        "source_acquisition_frame_index": source_frames,
        "source_crop_xywh": source_crop_xywh,
    }
    common_arrays = {
        "source_crop_row_ids": source_crop_row_ids,
        "instance_key": instance_keys,
        "source_acquisition_frame_index": source_frames,
        "frame_row_offsets": frame_row_offsets,
        "source_crop_xywh": source_crop_xywh,
    }

    probabilities = np.asarray(terminal["mask_probs_roi"][:], dtype=np.uint8)
    raw_available = np.asarray(terminal["available_channels"][:], dtype=bool)
    decoded = probabilities.astype(np.float32) / np.float32(255.0)
    thresholded = (decoded >= np.float32(0.5)).astype(np.uint8)
    raw_metrics = derive_subject_mask_metrics(thresholded)
    raw_arrays = {
        **common_arrays,
        "mask_probs_roi": probabilities,
        "available_channels": raw_available,
        "metrics/prob_max": np.max(probabilities, axis=(2, 3)).astype(
            np.float32,
            copy=False,
        )
        / np.float32(255.0),
        **{f"metrics/{name}": values for name, values in raw_metrics.items()},
    }
    terminal_source_manifest = {
        "schema_id": "palette.training_terminal_subject_mask_source",
        "schema_version": 1,
        "run_path": f"subject_mask_shard_runs/{terminal_run_id}",
        "metadata_sha256": terminal_metadata_sha256,
        "recording_identity": recording_identity,
        "crop_run_path": f"crop_runs/{crop_run_id}",
        "row_count": int(instance_keys.shape[0]),
    }
    raw_snapshot_root = working_root / "subject_mask_raw_snapshot.zarr"
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw_arrays,
        source_crop_arrays=crop_arrays,
        source_manifest=terminal_source_manifest,
        n_frames=source_frame_count,
        components=raw_components,
        destination=raw_snapshot_root,
        run_id=raw_run_id,
        kind="raw_probability_uint8",
        source_run_path=f"subject_mask_shard_runs/{terminal_run_id}",
        source_attributes={
            "source_crop_run": crop_run_id,
            "mask_labels": list(raw_labels),
            "label_schema_id": str(
                terminal.attrs.get("label_schema_id") or "subject_v1_union"
            ),
            "method": str(terminal.attrs.get("method") or "terminal_inference"),
            "probabilities_encoding": PROBABILITY_ENCODING_LINEAR_UINT8,
        },
        threshold=0.5,
        include_threshold_cache=False,
        created_by=created_by,
    )

    dense_masks = np.asarray(editable["masks_roi"][:], dtype=np.uint8)
    refined_available = np.asarray(editable["available_channels"][:], dtype=bool)
    refined_metrics = derive_subject_mask_metrics(dense_masks)
    refined_arrays = {
        **common_arrays,
        "masks_roi": dense_masks,
        "available_channels": refined_available,
        **{f"metrics/{name}": values for name, values in refined_metrics.items()},
    }
    refined_snapshot_root = working_root / "subject_mask_refined_snapshot.zarr"
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        refined_arrays,
        source_crop_arrays=crop_arrays,
        source_manifest=raw_publication.manifest,
        n_frames=source_frame_count,
        components=refined_components,
        destination=refined_snapshot_root,
        run_id=canonical_refined_run_id,
        kind="refined_dense_core",
        source_run_path=f"subject_mask_runs/{raw_run_id}",
        source_attributes={
            "source_crop_run": crop_run_id,
            "source_subject_mask_run": raw_run_id,
            "mask_labels": list(refined_labels),
            "label_schema_id": "subject_v1_lr",
            "method": "smart_finalize_subject_masks_v1",
        },
        created_by=created_by,
    )

    quality_source_paths = (
        "masks_roi",
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "available_channels",
    )
    quality_source = SubjectMaskQualitySourceReference(
        run_name=canonical_refined_run_id,
        manifest_digest=canonical_json_sha256(refined_publication.manifest),
        dense_array_values_sha256=sha256_array(dense_masks),
        component_registry_digest=canonical_json_sha256(
            refined_components.as_manifest()
        ),
        source_array_values_sha256={
            path: sha256_array(refined_arrays[path]) for path in quality_source_paths
        },
    )
    quality_shadow_root = working_root / "subject_mask_quality_snapshots"
    quality_publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        refined_arrays,
        n_frames=source_frame_count,
        components=refined_components,
        source=quality_source,
        source_manifest=refined_publication.manifest,
        destination=quality_shadow_root / "quality.zarr",
        run_id=quality_run_id,
        shadow_root=quality_shadow_root,
        scratch_root=working_root,
        created_by=created_by,
    )
    bundle_receipt = publish_subject_mask_bundle_candidate(
        analysis_zarr=archive,
        recording_identity=recording_identity,
        raw_snapshot_root=raw_publication.output_path,
        raw_run_id=raw_run_id,
        refined_snapshot_root=refined_publication.output_path,
        refined_run_id=canonical_refined_run_id,
        quality_snapshot_root=quality_publication.output_path,
        quality_run_id=quality_run_id,
        bundle_id=bundle_id,
    )

    writable = open_zarr_group_direct(archive, mode="a")
    editable = writable[f"refined_subject_masks_runs/{editable_refined_run_id}"]
    component_sources = editable.attrs.get("source_component_sources")
    normalized_sources: dict[str, dict[str, object]] = {}
    for label in refined_labels:
        prior = (
            component_sources.get(label)
            if isinstance(component_sources, Mapping)
            else None
        )
        normalized_sources[label] = {
            **(dict(prior) if isinstance(prior, Mapping) else {}),
            "source_stage": "subject_mask_runs",
            "source_run": raw_run_id,
        }
        provenance_path = f"components/{label}/provenance"
        if provenance_path in editable:
            provenance = editable[provenance_path]
            provenance.attrs.update(
                {
                    "source_stage": "subject_mask_runs",
                    "source_run": raw_run_id,
                    "source_surface_path": (
                        f"subject_mask_runs/{raw_run_id}/mask_probs_roi"
                    ),
                    "source_probability_path": (
                        f"subject_mask_runs/{raw_run_id}/mask_probs_roi"
                    ),
                }
            )
    editable.attrs.update(
        {
            "source_subject_mask_run": raw_run_id,
            "source_component_sources": normalized_sources,
            "source_canonical_refined_subject_mask_run": canonical_refined_run_id,
            "source_subject_mask_bundle_id": bundle_id,
        }
    )
    return {
        "raw_manifest_digest": canonical_json_sha256(raw_publication.manifest),
        "refined_manifest_digest": canonical_json_sha256(refined_publication.manifest),
        "quality_manifest_digest": canonical_json_sha256(quality_publication.manifest),
        "bundle_manifest_digest": bundle_receipt["bundle_manifest_digest"],
        "source_frame_count": source_frame_count,
        "raw_components": list(raw_labels),
        "refined_components": list(refined_labels),
    }


def _validate_keypoint_review_state(
    archive: Path,
    *,
    crop_run_id: str,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    delta_run_id: str,
    delta_generation: str,
) -> dict[str, Any]:
    """Validate the task-complete keypoint-only review surface."""

    root = open_zarr_group_direct(archive, mode="r")
    if root.attrs.get("stage_selector_eligible") is not False:
        raise RuntimeError("Training review artifact became selector eligible.")
    if root.attrs.get("training_artifact_status") != "review_active":
        raise RuntimeError("Training review artifact is not review_active.")
    run_paths = {
        "raw_keypoints": f"keypoints_runs/{raw_run_id}",
        "keypoint_quality": f"keypoint_quality_runs/{quality_run_id}",
        "refined_keypoints": f"refined_keypoints_runs/{refined_run_id}",
        "body_frame": f"analysis/body_frame_runs/{body_frame_run_id}",
    }
    for label, path in run_paths.items():
        run = root[path]
        if (
            not is_run_complete(run)
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError(
                f"{label} base is not complete selector-ineligible evidence."
            )
    refined = root[run_paths["refined_keypoints"]]
    if refined.attrs.get("artifact_mutability") != "immutable_snapshot":
        raise RuntimeError("Refined keypoint base is not an immutable snapshot.")
    crop_run = root[f"crop_runs/{crop_run_id}"]
    crop_manifest = crop_run.attrs.get("run_manifest")
    refined_manifest = refined.attrs.get("run_manifest")
    if not isinstance(crop_manifest, Mapping) or not isinstance(
        refined_manifest, Mapping
    ):
        raise RuntimeError(
            "Review artifact is missing a persisted crop/refined manifest."
        )
    expected_crop_digest = (
        refined_manifest.get("payload", {})
        .get("source_bindings", {})
        .get("crop_snapshot", {})
        .get("manifest_digest")
    )
    if canonical_json_sha256(crop_manifest) != expected_crop_digest:
        raise RuntimeError(
            "Persisted crop manifest differs from refined source binding."
        )
    delta = root[f"edit_delta_runs/{delta_run_id}"]
    generation = delta[f"generations/{delta_generation}"]
    if (
        generation.attrs.get("status") != "open"
        or generation.attrs.get("target_kind") != "keypoints"
        or generation.attrs.get("base_run_path")
        != f"refined_keypoints_runs/{refined_run_id}"
    ):
        raise RuntimeError("Keypoint edit generation lost its immutable-base binding.")
    review_policy = manual_keypoint_qc_policy_from_manifest(
        generation.attrs.get("review_qc_policy")
    )
    source_bindings = refined_manifest.get("payload", {}).get("source_bindings", {})
    skeleton = source_bindings.get("skeleton", {})
    skeleton_semantics = skeleton.get("semantics", {})
    if (
        generation.attrs.get("review_qc_policy_digest") != review_policy.policy_digest
        or review_policy.skeleton_id != skeleton.get("skeleton_id")
        or review_policy.skeleton_digest != skeleton.get("skeleton_digest")
        or list(review_policy.keypoint_labels)
        != skeleton_semantics.get("keypoint_labels")
    ):
        raise RuntimeError(
            "Keypoint edit generation QC policy differs from the refined skeleton."
        )
    refined_keys = np.asarray(refined["instance_key"][:], dtype=np.uint64)
    base_key_digest = instance_key_digest(refined_keys)
    if (
        delta.attrs.get("base_instance_key_count") != int(refined_keys.shape[0])
        or delta.attrs.get("base_instance_key_sha256") != base_key_digest
        or generation.attrs.get("base_instance_key_sha256") != base_key_digest
    ):
        raise RuntimeError(
            "Keypoint edit generation instance-key binding differs from its "
            "immutable base."
        )
    return {
        "run_paths": run_paths,
        "keypoint_delta_path": (
            f"edit_delta_runs/{delta_run_id}/generations/{delta_generation}"
        ),
        "manual_keypoint_qc_policy_digest": review_policy.policy_digest,
        "row_count": int(refined_keys.shape[0]),
        "metadata_read_mode": "direct_unconsolidated_while_review_mutable",
    }


def _validate_review_state(
    archive: Path,
    *,
    crop_run_id: str,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    delta_run_id: str,
    delta_generation: str,
    raw_mask_run_id: str,
    canonical_refined_mask_run_id: str,
    mask_quality_run_id: str,
    mask_bundle_id: str,
    editable_refined_mask_run_id: str,
) -> dict[str, Any]:
    keypoint_state = _validate_keypoint_review_state(
        archive,
        crop_run_id=crop_run_id,
        raw_run_id=raw_run_id,
        quality_run_id=quality_run_id,
        refined_run_id=refined_run_id,
        body_frame_run_id=body_frame_run_id,
        delta_run_id=delta_run_id,
        delta_generation=delta_generation,
    )
    root = open_zarr_group_direct(archive, mode="r")
    refined = root[keypoint_state["run_paths"]["refined_keypoints"]]
    for family, run_id in (
        ("subject_mask_runs", raw_mask_run_id),
        ("refined_subject_masks_runs", canonical_refined_mask_run_id),
    ):
        core_errors = validate_persisted_subject_mask_core_publication(
            archive,
            family=family,
            run_id=run_id,
        )
        if core_errors:
            raise RuntimeError(
                f"Persisted {family}/{run_id} failed validation: "
                + "; ".join(core_errors)
            )
    raw_masks = root[f"subject_mask_runs/{raw_mask_run_id}"]
    raw_encoding = probabilities_encoding_from_attrs(
        raw_masks.attrs,
        source_path=f"subject_mask_runs/{raw_mask_run_id}/mask_probs_roi",
        observed_dtype=raw_masks["mask_probs_roi"].dtype,
    )
    if raw_encoding != PROBABILITY_ENCODING_LINEAR_UINT8:
        raise RuntimeError(
            "Training review raw subject masks require the exact linear uint8 "
            "probability encoding."
        )
    bundle_validation = validate_subject_mask_bundle_candidate(
        analysis_zarr=archive,
        bundle_id=mask_bundle_id,
    )
    bundle = root[f"{SUBJECT_MASK_BUNDLE_FAMILY}/{mask_bundle_id}"]
    bundle_manifest = bundle.attrs.get(SUBJECT_MASK_BUNDLE_MANIFEST_ATTRIBUTE)
    if not isinstance(bundle_manifest, Mapping):
        raise RuntimeError("Training review artifact lacks its subject-mask bundle.")
    bundle_members = bundle_manifest["payload"]["members"]
    if (
        bundle_members["raw"]["run_id"] != raw_mask_run_id
        or bundle_members["refined"]["run_id"] != canonical_refined_mask_run_id
        or bundle_members["quality"]["run_id"] != mask_quality_run_id
    ):
        raise RuntimeError(
            "Subject-mask bundle members differ from the review receipt."
        )

    masks = root[f"refined_subject_masks_runs/{editable_refined_mask_run_id}"]
    if (
        refined_subject_mask_lifecycle_state(masks)
        != REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    ):
        raise RuntimeError("Refined subject-mask review surface is not editable_draft.")
    if (
        masks.attrs.get("stage_selector_eligible") is not False
        or "masks_roi" not in masks
    ):
        raise RuntimeError(
            "Editable subject masks require dense, selector-ineligible masks_roi."
        )
    if (
        masks.attrs.get("source_subject_mask_run") != raw_mask_run_id
        or masks.attrs.get("source_canonical_refined_subject_mask_run")
        != canonical_refined_mask_run_id
        or masks.attrs.get("source_subject_mask_bundle_id") != mask_bundle_id
    ):
        raise RuntimeError(
            "Editable subject-mask draft is not bound to its canonical bundle."
        )
    masks_roi = masks["masks_roi"]
    refined_keys = np.asarray(refined["instance_key"][:], dtype=np.uint64)
    mask_keys = np.asarray(masks["instance_key"][:], dtype=np.uint64)
    expected_mask_labels = [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    if (
        np.dtype(masks_roi.dtype) != np.dtype(np.uint8)
        or len(masks_roi.shape) != 4
        or int(masks_roi.shape[0]) != int(refined_keys.shape[0])
        or int(masks_roi.shape[1]) != len(expected_mask_labels)
        or masks.attrs.get("mask_labels") != expected_mask_labels
        or not np.array_equal(mask_keys, refined_keys)
    ):
        raise RuntimeError(
            "Editable subject masks differ from the strict refined-keypoint row identity."
        )
    return {
        **keypoint_state,
        "refined_subject_mask_path": (
            f"refined_subject_masks_runs/{editable_refined_mask_run_id}"
        ),
        "subject_mask_contract_chain": {
            "raw": f"subject_mask_runs/{raw_mask_run_id}",
            "refined": (f"refined_subject_masks_runs/{canonical_refined_mask_run_id}"),
            "quality": f"subject_mask_quality_runs/{mask_quality_run_id}",
            "bundle": f"{SUBJECT_MASK_BUNDLE_FAMILY}/{mask_bundle_id}",
            "bundle_manifest_digest": bundle_validation["bundle_manifest_digest"],
        },
        "dense_masks_roi_shape": [int(value) for value in masks_roi.shape],
        "dense_masks_roi_dtype": str(np.dtype(masks_roi.dtype)),
        "mask_labels": expected_mask_labels,
    }


def publish_training_keypoint_review_artifact(
    *,
    source_archive: Path,
    destination: Path,
    scratch_root: Path,
    crop_run_id: str,
    terminal_keypoint_run_id: str,
    raw_keypoint_run_id: str,
    quality_run_id: str,
    refined_keypoint_run_id: str,
    body_frame_run_id: str,
    keypoint_delta_run_id: str,
    keypoint_delta_generation: str,
    created_by: str,
    copy_backend: str = "python",
) -> Mapping[str, Any]:
    """Publish one atomic keypoint-only training-review artifact.

    This is deliberately separate from :func:`publish_training_review_artifact`.
    It closes the complete keypoint review lifecycle without inventing subject-mask
    terminals for a pose-only task.  The prediction/refinement bases are immutable;
    review writes are confined to the bound instance-key delta generation.
    """

    source = source_archive.expanduser().resolve()
    target = destination.expanduser().resolve()
    scratch = _bounded_node_local_scratch(scratch_root)
    if copy_backend != "python":
        raise ValueError(
            "Training keypoint review currently supports copy_backend='python' only."
        )
    if not source.is_dir() or source.suffix != ".zarr":
        raise FileNotFoundError(f"Source training Zarr not found: {source}")
    if target.suffix != ".zarr":
        raise ValueError("Training review artifact destination must end in .zarr.")
    if target.exists():
        raise FileExistsError(f"Training review artifact already exists: {target}")
    run_ids = {
        "crop": _safe_run_id(crop_run_id, name="crop_run_id"),
        "terminal_keypoints": _safe_run_id(
            terminal_keypoint_run_id, name="terminal_keypoint_run_id"
        ),
        "raw_keypoints": _safe_run_id(raw_keypoint_run_id, name="raw_keypoint_run_id"),
        "keypoint_quality": _safe_run_id(quality_run_id, name="quality_run_id"),
        "refined_keypoints": _safe_run_id(
            refined_keypoint_run_id, name="refined_keypoint_run_id"
        ),
        "body_frame": _safe_run_id(body_frame_run_id, name="body_frame_run_id"),
        "keypoint_delta": _safe_run_id(
            keypoint_delta_run_id, name="keypoint_delta_run_id"
        ),
        "keypoint_delta_generation": _safe_run_id(
            keypoint_delta_generation, name="keypoint_delta_generation"
        ),
    }
    recording_identity = _recording_identity(source)
    terminal_path = f"keypoint_shard_runs/{run_ids['terminal_keypoints']}"
    terminal_metadata_sha256 = _metadata_sha256(source, terminal_path)
    identity = _initial_refined_identity(
        recording_identity=recording_identity,
        terminal_metadata_sha256=terminal_metadata_sha256,
        refined_run_id=run_ids["refined_keypoints"],
    )

    with tempfile.TemporaryDirectory(
        prefix="palette-training-keypoint-review-", dir=str(scratch)
    ) as temporary:
        working_root = Path(temporary)
        bundle_root = working_root / "keypoint_bundle"
        local_archive = working_root / target.name
        chain = publish_training_keypoint_review_candidate_chain(
            source_archive=source,
            crop_run_id=run_ids["crop"],
            terminal_run_id=run_ids["terminal_keypoints"],
            bundle_root=bundle_root,
            raw_run_id=run_ids["raw_keypoints"],
            quality_run_id=run_ids["keypoint_quality"],
            refined_run_id=run_ids["refined_keypoints"],
            body_frame_run_id=run_ids["body_frame"],
            refined_identity=identity,
            created_by=created_by,
        )
        shutil.copytree(source, local_archive)
        keypoint_import = publish_keypoint_v2_production_candidate_chain(
            analysis_zarr=local_archive,
            chain=chain,
            copy_backend=copy_backend,
        )
        local_root = open_zarr_group_direct(local_archive, mode="a")
        local_root[f"crop_runs/{run_ids['crop']}"].attrs["run_manifest"] = dict(
            chain.crop.manifest
        )
        local_root.attrs.update(
            {
                "training_artifact_status": "building_review_surfaces",
                "training_task": "keypoints",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "metadata_read_mode": "direct_unconsolidated_while_review_mutable",
            }
        )
        manual_qc_policy = build_default_manual_keypoint_qc_policy(
            skeleton_id=chain.refined.source.skeleton_id,
            skeleton_digest=chain.refined.source.skeleton_digest,
            keypoint_labels=chain.refined.source.skeleton_semantics[
                "keypoint_labels"
            ],
        )
        delta_binding = create_delta_generation(
            local_root,
            delta_run=run_ids["keypoint_delta"],
            generation=run_ids["keypoint_delta_generation"],
            generation_ordinal=1,
            target_kind="keypoints",
            base_run_path=f"refined_keypoints_runs/{run_ids['refined_keypoints']}",
            created_by=created_by,
            review_qc_policy=manual_qc_policy.as_manifest(),
        )
        payload = json_attr_safe(
            {
                "status": "review_active",
                "training_task": "keypoints",
                "created_at_utc": utc_now(),
                "created_by": str(created_by),
                "policy": TRAINING_KEYPOINT_REVIEW_ARTIFACT_POLICY,
                "source_archive": str(source),
                "destination": str(target),
                "recording_identity": recording_identity,
                "source_evidence": {
                    "crop_run": run_ids["crop"],
                    "terminal_keypoint_run": run_ids["terminal_keypoints"],
                    "terminal_keypoint_metadata_sha256": terminal_metadata_sha256,
                },
                "runs": run_ids,
                "keypoint_chain_receipt_digest": chain.receipt["payload_digest"],
                "keypoint_import_schema": keypoint_import["schema_id"],
                "keypoint_delta_binding": dict(delta_binding),
                "keypoint_review_authority": (
                    "immutable_refined_snapshot_plus_instance_key_delta_generation"
                ),
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "metadata_read_mode": (
                    "direct_unconsolidated_while_review_mutable"
                ),
                "published_consolidated_base_generation": keypoint_import[
                    "consolidation"
                ],
                "run_provenance": build_writer_run_provenance(
                    command=(
                        "fisheye.utils.publish_training_keypoint_review_artifact"
                    ),
                    params={
                        "policy": TRAINING_KEYPOINT_REVIEW_ARTIFACT_POLICY,
                        "created_by": str(created_by),
                        "stage_selector_eligible": False,
                        "registry_activation": "deferred",
                    },
                    input_run_ids={
                        "crop_run": run_ids["crop"],
                        "terminal_keypoint_run": run_ids["terminal_keypoints"],
                    },
                ),
            }
        )
        receipt = {
            "schema_id": TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID,
            "schema_version": TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        local_root.attrs.update(
            {
                "training_artifact_status": "review_active",
                "training_review_artifact": receipt,
            }
        )
        write_json_atomic(
            local_archive / TRAINING_KEYPOINT_REVIEW_ARTIFACT_RECEIPT,
            receipt,
        )
        _validate_keypoint_review_state(
            local_archive,
            crop_run_id=run_ids["crop"],
            raw_run_id=run_ids["raw_keypoints"],
            quality_run_id=run_ids["keypoint_quality"],
            refined_run_id=run_ids["refined_keypoints"],
            body_frame_run_id=run_ids["body_frame"],
            delta_run_id=run_ids["keypoint_delta"],
            delta_generation=run_ids["keypoint_delta_generation"],
        )
        local_inventory = tree_inventory(local_archive, hash_content=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        hidden = target.with_name(
            f".{target.name}.publish_tmp.{os.getpid()}.{uuid4().hex}"
        )
        with archive_metadata_publication_lock(target):
            if target.exists() or hidden.exists():
                raise FileExistsError(
                    f"Training keypoint review target became occupied: {target}"
                )
            try:
                shutil.copytree(local_archive, hidden)
                if tree_inventory(hidden, hash_content=True) != local_inventory:
                    raise RuntimeError(
                        "Published keypoint review copy differs from node-local source."
                    )
                _validate_keypoint_review_state(
                    hidden,
                    crop_run_id=run_ids["crop"],
                    raw_run_id=run_ids["raw_keypoints"],
                    quality_run_id=run_ids["keypoint_quality"],
                    refined_run_id=run_ids["refined_keypoints"],
                    body_frame_run_id=run_ids["body_frame"],
                    delta_run_id=run_ids["keypoint_delta"],
                    delta_generation=run_ids["keypoint_delta_generation"],
                )
                os.replace(hidden, target)
            except BaseException:
                if hidden.exists():
                    shutil.rmtree(hidden)
                raise

    final_state = _validate_keypoint_review_state(
        target,
        crop_run_id=run_ids["crop"],
        raw_run_id=run_ids["raw_keypoints"],
        quality_run_id=run_ids["keypoint_quality"],
        refined_run_id=run_ids["refined_keypoints"],
        body_frame_run_id=run_ids["body_frame"],
        delta_run_id=run_ids["keypoint_delta"],
        delta_generation=run_ids["keypoint_delta_generation"],
    )
    return json_attr_safe(
        {
            "schema_id": TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID,
            "schema_version": TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "status": "review_active",
            "destination": str(target),
            "receipt": str(target / TRAINING_KEYPOINT_REVIEW_ARTIFACT_RECEIPT),
            "receipt_digest": receipt["payload_digest"],
            "runs": run_ids,
            "review_state": final_state,
            "published_consolidated_base_generation": keypoint_import[
                "consolidation"
            ],
            "physical_inventory": local_inventory.to_json(),
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
        }
    )


def publish_training_review_artifact(
    *,
    source_archive: Path,
    destination: Path,
    scratch_root: Path,
    crop_run_id: str,
    terminal_keypoint_run_id: str,
    terminal_subject_mask_run_id: str,
    raw_keypoint_run_id: str,
    quality_run_id: str,
    refined_keypoint_run_id: str,
    body_frame_run_id: str,
    keypoint_delta_run_id: str,
    keypoint_delta_generation: str,
    raw_subject_mask_run_id: str,
    canonical_refined_subject_mask_run_id: str,
    subject_mask_quality_run_id: str,
    subject_mask_bundle_id: str,
    editable_refined_subject_mask_run_id: str,
    created_by: str,
    copy_backend: str = "python",
) -> Mapping[str, Any]:
    """Create and atomically publish one selector-ineligible review artifact."""

    source = source_archive.expanduser().resolve()
    target = destination.expanduser().resolve()
    scratch = _bounded_node_local_scratch(scratch_root)
    if copy_backend != "python":
        raise ValueError(
            "Training review artifact currently supports copy_backend='python' only."
        )
    if not source.is_dir() or source.suffix != ".zarr":
        raise FileNotFoundError(f"Source training Zarr not found: {source}")
    if target.suffix != ".zarr":
        raise ValueError("Training review artifact destination must end in .zarr.")
    if target.exists():
        raise FileExistsError(f"Training review artifact already exists: {target}")
    run_ids = {
        "crop": _safe_run_id(crop_run_id, name="crop_run_id"),
        "terminal_keypoints": _safe_run_id(
            terminal_keypoint_run_id, name="terminal_keypoint_run_id"
        ),
        "terminal_subject_masks": _safe_run_id(
            terminal_subject_mask_run_id, name="terminal_subject_mask_run_id"
        ),
        "raw_keypoints": _safe_run_id(raw_keypoint_run_id, name="raw_keypoint_run_id"),
        "keypoint_quality": _safe_run_id(quality_run_id, name="quality_run_id"),
        "refined_keypoints": _safe_run_id(
            refined_keypoint_run_id, name="refined_keypoint_run_id"
        ),
        "body_frame": _safe_run_id(body_frame_run_id, name="body_frame_run_id"),
        "keypoint_delta": _safe_run_id(
            keypoint_delta_run_id, name="keypoint_delta_run_id"
        ),
        "keypoint_delta_generation": _safe_run_id(
            keypoint_delta_generation, name="keypoint_delta_generation"
        ),
        "raw_subject_masks": _safe_run_id(
            raw_subject_mask_run_id, name="raw_subject_mask_run_id"
        ),
        "canonical_refined_subject_masks": _safe_run_id(
            canonical_refined_subject_mask_run_id,
            name="canonical_refined_subject_mask_run_id",
        ),
        "subject_mask_quality": _safe_run_id(
            subject_mask_quality_run_id, name="subject_mask_quality_run_id"
        ),
        "subject_mask_bundle": _safe_run_id(
            subject_mask_bundle_id, name="subject_mask_bundle_id"
        ),
        "editable_refined_subject_masks": _safe_run_id(
            editable_refined_subject_mask_run_id,
            name="editable_refined_subject_mask_run_id",
        ),
    }
    recording_identity = _recording_identity(source)
    terminal_keypoint_path = f"keypoint_shard_runs/{run_ids['terminal_keypoints']}"
    terminal_mask_path = f"subject_mask_shard_runs/{run_ids['terminal_subject_masks']}"
    terminal_keypoint_metadata_sha256 = _metadata_sha256(source, terminal_keypoint_path)
    terminal_mask_metadata_sha256 = _metadata_sha256(source, terminal_mask_path)
    identity = _initial_refined_identity(
        recording_identity=recording_identity,
        terminal_metadata_sha256=terminal_keypoint_metadata_sha256,
        refined_run_id=run_ids["refined_keypoints"],
    )

    with tempfile.TemporaryDirectory(
        prefix="palette-training-review-artifact-", dir=str(scratch)
    ) as temporary:
        working_root = Path(temporary)
        bundle_root = working_root / "keypoint_bundle"
        local_archive = working_root / target.name
        chain = publish_training_keypoint_review_candidate_chain(
            source_archive=source,
            crop_run_id=run_ids["crop"],
            terminal_run_id=run_ids["terminal_keypoints"],
            bundle_root=bundle_root,
            raw_run_id=run_ids["raw_keypoints"],
            quality_run_id=run_ids["keypoint_quality"],
            refined_run_id=run_ids["refined_keypoints"],
            body_frame_run_id=run_ids["body_frame"],
            refined_identity=identity,
            created_by=created_by,
        )
        shutil.copytree(source, local_archive)
        # The immutable keypoint importer first validates the copied artifact's
        # published consolidated generation.  Only after that import succeeds
        # do we transition the artifact into its direct-metadata mutable review
        # lifecycle and create editable children.
        keypoint_import = publish_keypoint_v2_production_candidate_chain(
            analysis_zarr=local_archive,
            chain=chain,
            copy_backend=copy_backend,
        )
        local_root = open_zarr_group_direct(local_archive, mode="a")
        local_crop = local_root[f"crop_runs/{run_ids['crop']}"]
        local_crop.attrs["run_manifest"] = dict(chain.crop.manifest)
        local_root.attrs.update(
            {
                "training_artifact_status": "building_review_surfaces",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "metadata_read_mode": "direct_unconsolidated_while_review_mutable",
            }
        )
        manual_qc_policy = build_default_manual_keypoint_qc_policy(
            skeleton_id=chain.refined.source.skeleton_id,
            skeleton_digest=chain.refined.source.skeleton_digest,
            keypoint_labels=chain.refined.source.skeleton_semantics["keypoint_labels"],
        )
        delta_binding = create_delta_generation(
            local_root,
            delta_run=run_ids["keypoint_delta"],
            generation=run_ids["keypoint_delta_generation"],
            generation_ordinal=1,
            target_kind="keypoints",
            base_run_path=f"refined_keypoints_runs/{run_ids['refined_keypoints']}",
            created_by=created_by,
            review_qc_policy=manual_qc_policy.as_manifest(),
        )
        mask_summary = finalize_subject_masks(
            local_archive,
            subject_shard_runs=(run_ids["terminal_subject_masks"],),
            target_crop_run=run_ids["crop"],
            refined_run=run_ids["editable_refined_subject_masks"],
            components=("subject_body", "eyes_union", "swim_bladder"),
            metric_level="cheap",
            write_eye_geometry=False,
            write_component_contours=False,
            write_sampled_component_contours=False,
            retain_source_seeds=False,
            mask_storage="dense_uint8",
            mask_rle_validation_mode="none",
            dense_mask_row_chunk=128,
            assignment_keypoint_group="refined_keypoints_runs",
            assignment_keypoints_run=run_ids["refined_keypoints"],
            require_production_proof=False,
            review_draft=True,
            defer_registry_status=True,
            overwrite=False,
        )
        subject_mask_contract_chain = _publish_subject_mask_contract_chain(
            archive=local_archive,
            working_root=working_root,
            recording_identity=recording_identity,
            crop_run_id=run_ids["crop"],
            terminal_run_id=run_ids["terminal_subject_masks"],
            terminal_metadata_sha256=terminal_mask_metadata_sha256,
            raw_run_id=run_ids["raw_subject_masks"],
            canonical_refined_run_id=run_ids["canonical_refined_subject_masks"],
            quality_run_id=run_ids["subject_mask_quality"],
            bundle_id=run_ids["subject_mask_bundle"],
            editable_refined_run_id=run_ids["editable_refined_subject_masks"],
            created_by=created_by,
        )
        local_root = open_zarr_group_direct(local_archive, mode="a")
        payload = json_attr_safe(
            {
                "status": "review_active",
                "created_at_utc": utc_now(),
                "created_by": str(created_by),
                "policy": TRAINING_REVIEW_ARTIFACT_POLICY,
                "source_archive": str(source),
                "destination": str(target),
                "recording_identity": recording_identity,
                "source_evidence": {
                    "crop_run": run_ids["crop"],
                    "terminal_keypoint_run": run_ids["terminal_keypoints"],
                    "terminal_keypoint_metadata_sha256": (
                        terminal_keypoint_metadata_sha256
                    ),
                    "terminal_subject_mask_run": run_ids["terminal_subject_masks"],
                    "terminal_subject_mask_metadata_sha256": (
                        terminal_mask_metadata_sha256
                    ),
                },
                "runs": run_ids,
                "keypoint_chain_receipt_digest": chain.receipt["payload_digest"],
                "keypoint_import_schema": keypoint_import["schema_id"],
                "keypoint_delta_binding": dict(delta_binding),
                "subject_mask_finalization": {
                    "status": mask_summary.get("status"),
                    "refined_run": mask_summary.get("refined_run"),
                    "rows": mask_summary.get("roi_count"),
                    "mask_storage": mask_summary.get("mask_storage"),
                },
                "subject_mask_contract_chain": subject_mask_contract_chain,
                "keypoint_review_authority": (
                    "immutable_refined_snapshot_plus_instance_key_delta_generation"
                ),
                "subject_mask_review_authority": "dense_editable_masks_roi",
                "negative_frame_policy": "rowless_frame_decision_preserved",
                "stage_selector_eligible": False,
                "registry_activation": "deferred",
                "metadata_read_mode": ("direct_unconsolidated_while_review_mutable"),
                "run_provenance": build_writer_run_provenance(
                    command=("fisheye.utils.publish_training_review_artifact"),
                    params={
                        "policy": TRAINING_REVIEW_ARTIFACT_POLICY,
                        "created_by": str(created_by),
                        "stage_selector_eligible": False,
                        "registry_activation": "deferred",
                    },
                    input_run_ids={
                        "crop_run": run_ids["crop"],
                        "terminal_keypoint_run": run_ids["terminal_keypoints"],
                        "terminal_subject_mask_run": run_ids["terminal_subject_masks"],
                    },
                ),
            }
        )
        receipt = {
            "schema_id": TRAINING_REVIEW_ARTIFACT_SCHEMA_ID,
            "schema_version": TRAINING_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "payload_digest": canonical_json_sha256(payload),
            "payload": payload,
        }
        local_root.attrs.update(
            {
                "training_artifact_status": "review_active",
                "training_review_artifact": receipt,
            }
        )
        write_json_atomic(
            local_archive / TRAINING_REVIEW_ARTIFACT_RECEIPT,
            receipt,
        )
        initial_consolidation = consolidate_metadata_capture_expected_warnings(
            local_archive
        )
        _validate_review_state(
            local_archive,
            crop_run_id=run_ids["crop"],
            raw_run_id=run_ids["raw_keypoints"],
            quality_run_id=run_ids["keypoint_quality"],
            refined_run_id=run_ids["refined_keypoints"],
            body_frame_run_id=run_ids["body_frame"],
            delta_run_id=run_ids["keypoint_delta"],
            delta_generation=run_ids["keypoint_delta_generation"],
            raw_mask_run_id=run_ids["raw_subject_masks"],
            canonical_refined_mask_run_id=run_ids["canonical_refined_subject_masks"],
            mask_quality_run_id=run_ids["subject_mask_quality"],
            mask_bundle_id=run_ids["subject_mask_bundle"],
            editable_refined_mask_run_id=run_ids["editable_refined_subject_masks"],
        )
        local_inventory = tree_inventory(local_archive, hash_content=True)

        target.parent.mkdir(parents=True, exist_ok=True)
        hidden = target.with_name(
            f".{target.name}.publish_tmp.{os.getpid()}.{uuid4().hex}"
        )
        with archive_metadata_publication_lock(target):
            if target.exists() or hidden.exists():
                raise FileExistsError(
                    f"Training review publication target became occupied: {target}"
                )
            try:
                shutil.copytree(local_archive, hidden)
                hidden_inventory = tree_inventory(hidden, hash_content=True)
                if hidden_inventory != local_inventory:
                    raise RuntimeError(
                        "Published artifact copy differs from node-local source."
                    )
                _validate_review_state(
                    hidden,
                    crop_run_id=run_ids["crop"],
                    raw_run_id=run_ids["raw_keypoints"],
                    quality_run_id=run_ids["keypoint_quality"],
                    refined_run_id=run_ids["refined_keypoints"],
                    body_frame_run_id=run_ids["body_frame"],
                    delta_run_id=run_ids["keypoint_delta"],
                    delta_generation=run_ids["keypoint_delta_generation"],
                    raw_mask_run_id=run_ids["raw_subject_masks"],
                    canonical_refined_mask_run_id=run_ids[
                        "canonical_refined_subject_masks"
                    ],
                    mask_quality_run_id=run_ids["subject_mask_quality"],
                    mask_bundle_id=run_ids["subject_mask_bundle"],
                    editable_refined_mask_run_id=run_ids[
                        "editable_refined_subject_masks"
                    ],
                )
                if target.exists():
                    raise FileExistsError(
                        f"Training review target appeared during publication: {target}"
                    )
                os.replace(hidden, target)
            except Exception:
                if hidden.exists():
                    shutil.rmtree(hidden)
                raise

    final_state = _validate_review_state(
        target,
        crop_run_id=run_ids["crop"],
        raw_run_id=run_ids["raw_keypoints"],
        quality_run_id=run_ids["keypoint_quality"],
        refined_run_id=run_ids["refined_keypoints"],
        body_frame_run_id=run_ids["body_frame"],
        delta_run_id=run_ids["keypoint_delta"],
        delta_generation=run_ids["keypoint_delta_generation"],
        raw_mask_run_id=run_ids["raw_subject_masks"],
        canonical_refined_mask_run_id=run_ids["canonical_refined_subject_masks"],
        mask_quality_run_id=run_ids["subject_mask_quality"],
        mask_bundle_id=run_ids["subject_mask_bundle"],
        editable_refined_mask_run_id=run_ids["editable_refined_subject_masks"],
    )
    return json_attr_safe(
        {
            "schema_id": TRAINING_REVIEW_ARTIFACT_SCHEMA_ID,
            "schema_version": TRAINING_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "status": "review_active",
            "destination": str(target),
            "receipt": str(target / TRAINING_REVIEW_ARTIFACT_RECEIPT),
            "receipt_digest": receipt["payload_digest"],
            "runs": run_ids,
            "review_state": final_state,
            "initial_consolidation": initial_consolidation,
            "physical_inventory": local_inventory.to_json(),
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
        }
    )


__all__ = [
    "TRAINING_KEYPOINT_REVIEW_ARTIFACT_POLICY",
    "TRAINING_KEYPOINT_REVIEW_ARTIFACT_RECEIPT",
    "TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID",
    "TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_VERSION",
    "TRAINING_REVIEW_ARTIFACT_POLICY",
    "TRAINING_REVIEW_ARTIFACT_RECEIPT",
    "TRAINING_REVIEW_ARTIFACT_SCHEMA_ID",
    "TRAINING_REVIEW_ARTIFACT_SCHEMA_VERSION",
    "publish_training_keypoint_review_artifact",
    "publish_training_review_artifact",
]

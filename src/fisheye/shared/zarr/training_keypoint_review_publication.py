"""Finalize one sampled-training terminal pose run into the strict v2 chain.

This adapter is intentionally narrow.  It does not rerun inference, does not
activate selectors, and does not pretend a sampled local frame axis is a full
recording crop-v2 axis.  The terminal arrays are normalized at the existing
float32 keypoint-v2 boundary, then the shared raw/quality/refined/body-frame
publishers own physical storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np
import zarr

from fisheye.shared.pose_model_schema_binding import (
    build_explicit_pose_model_schema_binding,
)
from fisheye.shared.run_provenance import build_run_provenance
from fisheye.shared.zarr.benchmark_runtime import sha256_array, sha256_file, utc_now
from fisheye.shared.zarr.body_frame_producer import (
    BodyFrameSourceReference,
    build_keypoint_body_frame_recipe,
    prepare_keypoint_body_frame,
)
from fisheye.shared.zarr.body_frame_publication import (
    BodyFrameShadowPublication,
    publish_selector_ineligible_body_frame_snapshot,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication import (
    KeypointShadowPublication,
    prepare_raw_keypoint_v2_from_yolo_arrays,
    publish_selector_ineligible_keypoint_snapshot,
)
from fisheye.shared.zarr.keypoint_publication_mode import (
    KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
    KeypointChainPublicationDispositions,
    KeypointPublicationDisposition,
)
from fisheye.shared.zarr.keypoint_quality_producer import (
    ObservationLocalKeypointQualityPolicy,
    prepare_observation_local_keypoint_quality,
)
from fisheye.shared.zarr.keypoint_quality_publication import (
    KeypointQualityShadowPublication,
    publish_selector_ineligible_keypoint_quality_snapshot,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KeypointQualitySourceReference,
)
from fisheye.shared.zarr.keypoint_schema import KeypointDimensions
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_keypoint_manifest import (
    RefinedKeypointSnapshotIdentity,
    build_refined_keypoint_source_bindings,
)
from fisheye.shared.zarr.refined_keypoint_producer import (
    prepare_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.refined_keypoint_publication import (
    RefinedKeypointShadowPublication,
    publish_selector_ineligible_refined_keypoint_snapshot,
)
from fisheye.shared.zarr.storage_profiles import (
    PUBLISHED_HTTP_V1,
    StorageProfile,
    storage_profile_from_manifest,
)
from fisheye.shared.zarr.training_crop_materialization import (
    bind_training_crop_materialization,
)
from fisheye.shared.zarr.training_keypoint_crop_source import (
    build_training_keypoint_crop_source_manifest,
)
from fisheye.shared.zarr_run_completion import is_run_complete


TRAINING_KEYPOINT_REVIEW_CHAIN_SCHEMA_ID = (
    "palette.keypoint.training_review_candidate_chain"
)
TRAINING_KEYPOINT_REVIEW_CHAIN_SCHEMA_VERSION = 1
TRAINING_KEYPOINT_REVIEW_CHAIN_RECEIPT = "training_review_chain_receipt.json"

_TERMINAL_ARRAYS = (
    "instance_key",
    "source_crop_row_ids",
    "source_acquisition_frame_index",
    "frame_indices",
    "keypoints_roi",
    "keypoint_confidences",
    "confidence",
    "detection_success",
    "pose_bbox_xyxy_roi",
)


@dataclass(frozen=True)
class TrainingKeypointCropPublication:
    output_path: Path
    run_id: str
    dimensions: CropDimensions
    plans: Any
    manifest: Mapping[str, Any]
    arrays: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingKeypointReviewChain:
    crop: TrainingKeypointCropPublication
    raw: KeypointShadowPublication
    quality: KeypointQualityShadowPublication
    refined: RefinedKeypointShadowPublication
    body_frame: BodyFrameShadowPublication
    receipt_path: Path
    receipt: Mapping[str, Any]


def _values(value: Any) -> np.ndarray:
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def open_training_keypoint_crop_publication(
    archive_path: Path,
    *,
    run_id: str,
    require_consolidated: bool = True,
) -> TrainingKeypointCropPublication:
    bound = bind_training_crop_materialization(
        archive_path,
        run_id=run_id,
        require_consolidated=require_consolidated,
    )
    binding_payload = bound.binding["payload"]
    dimensions_value = binding_payload["dimensions"]
    offsets_shape = binding_payload["array_declarations"]["frame_row_offsets"]["shape"]
    dimensions = CropDimensions(
        n_frames=int(offsets_shape[0]) - 1,
        n_instances=int(dimensions_value["row_count"]),
        source_width=int(dimensions_value["source_width"]),
        source_height=int(dimensions_value["source_height"]),
    )
    storage_value = bound.run_group.attrs.get("storage_plan")
    if not isinstance(storage_value, Mapping) or not isinstance(
        storage_value.get("storage_profile"), Mapping
    ):
        raise ValueError("Training crop run lacks its byte-planned storage profile.")
    plans = plan_crop_geometry_storage(
        dimensions,
        profile=storage_profile_from_manifest(storage_value["storage_profile"]),
    )
    arrays = {
        path: bound.run_group[path] for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    }
    root = zarr.open_group(str(bound.archive_path), mode="r", use_consolidated=False)
    recording_identity = str(root.attrs.get("recording_id") or "").strip()
    if not recording_identity:
        raise ValueError("Training artifact lacks recording_id.")
    manifest = build_training_keypoint_crop_source_manifest(
        run_id=bound.run_id,
        recording_identity=recording_identity,
        training_crop_binding=bound.binding,
    )
    return TrainingKeypointCropPublication(
        output_path=bound.archive_path,
        run_id=bound.run_id,
        dimensions=dimensions,
        plans=plans,
        manifest=manifest,
        arrays=arrays,
        receipt={
            "logical_content_digest": manifest["payload"]["logical_content"]["digest"],
            "training_crop_binding_digest": bound.binding["payload_digest"],
        },
    )


def _terminal_model_sha256(attrs: Mapping[str, Any]) -> str:
    provenance = attrs.get("run_provenance")
    artifacts = (
        provenance.get("input_artifacts") if isinstance(provenance, Mapping) else None
    )
    values = {
        str(item.get("sha256") or "").strip().lower()
        for item in artifacts or ()
        if isinstance(item, Mapping) and item.get("role") == "keypoint_model"
    }
    values = {
        value
        for value in values
        if len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    }
    if len(values) != 1:
        raise ValueError("Terminal keypoints require exactly one model SHA-256.")
    return next(iter(values))


def _pose_binding(attrs: Mapping[str, Any]) -> dict[str, Any]:
    pose_schema = attrs.get("pose_schema")
    labels = attrs.get("keypoint_labels")
    model_shape = attrs.get("model_kpt_shape")
    if (
        not isinstance(pose_schema, Mapping)
        or not isinstance(labels, list)
        or not isinstance(model_shape, list)
        or pose_schema.get("keypoint_labels") != labels
        or not isinstance(pose_schema.get("edges"), list)
    ):
        raise ValueError("Terminal keypoints lack exact ordered skeleton semantics.")
    return build_explicit_pose_model_schema_binding(
        model_sha256=_terminal_model_sha256(attrs),
        assertion_id="sampled_training_terminal_republication_v1",
        skeleton_id=pose_schema.get("skeleton_id"),
        model_kpt_shape=model_shape,
        keypoint_labels=labels,
        edges=pose_schema["edges"],
    )


def _preprocessing(
    attrs: Mapping[str, Any],
    *,
    terminal_path: str,
    crop: TrainingKeypointCropPublication,
) -> KeypointPreprocessingReference:
    transform = attrs.get("model_input_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Terminal keypoints lack model_input_transform.")
    input_mode = str(attrs.get("input_mode_effective") or "").strip()
    if not input_mode:
        raise ValueError("Terminal keypoints lack input_mode_effective.")
    return KeypointPreprocessingReference(
        profile_id="sampled_training_terminal_v1",
        profile_version=1,
        input_mode="terminal_candidate_republication",
        document={
            "source_terminal_path": terminal_path,
            "source_input_mode_effective": input_mode,
            "source_model_input_transform": dict(transform),
            "source_model_input_stride": attrs.get("model_input_stride"),
            "source_native_roi_shape_hw": attrs.get("native_roi_shape_hw"),
            "source_model_input_shape_hw": attrs.get("model_input_shape_hw"),
            "source_network_input_shape_hw": attrs.get("model_network_input_shape_hw"),
            "training_crop_manifest_digest": canonical_json_sha256(crop.manifest),
            "training_crop_binding_digest": crop.receipt[
                "training_crop_binding_digest"
            ],
            "coordinate_mapping": (
                "sampled_local_frame_plus_acquisition_frame_lineage_v1"
            ),
            "inference_reexecuted": False,
        },
    )


def _dispositions(
    *,
    source_archive: Path,
    terminal_path: str,
    terminal_metadata_sha256: str,
    run_ids: Mapping[str, str],
) -> KeypointChainPublicationDispositions:
    def one(stage: str) -> KeypointPublicationDisposition:
        provenance = build_run_provenance(
            command=("fisheye.shared.zarr.training_keypoint_review_publication"),
            params={
                "stage": stage,
                "source_archive": str(source_archive),
                "terminal_path": terminal_path,
                "terminal_metadata_sha256": terminal_metadata_sha256,
                "run_id": run_ids[stage],
                "selector_activation": "deferred",
            },
            input_run_ids={"terminal_keypoints": terminal_path},
            cwd=Path.cwd(),
        )
        return KeypointPublicationDisposition(
            mode=KEYPOINT_PUBLICATION_MODE_PRODUCTION_CANDIDATE,
            publication_owner_uuid=uuid.uuid4().hex,
            run_provenance=provenance,
        )

    return KeypointChainPublicationDispositions(
        raw=one("raw"),
        quality=one("quality"),
        refined=one("refined"),
        body_frame=one("body_frame"),
    )


def publish_training_keypoint_review_candidate_chain(
    *,
    source_archive: Path,
    crop_run_id: str,
    terminal_run_id: str,
    bundle_root: Path,
    raw_run_id: str,
    quality_run_id: str,
    refined_run_id: str,
    body_frame_run_id: str,
    refined_identity: RefinedKeypointSnapshotIdentity,
    storage_profile: StorageProfile = PUBLISHED_HTTP_V1,
    quality_policy: ObservationLocalKeypointQualityPolicy = (
        ObservationLocalKeypointQualityPolicy()
    ),
    created_by: str = "training_keypoint_review_candidate",
) -> TrainingKeypointReviewChain:
    """Create a strict immutable base plus quality/refined/body-frame surfaces."""

    archive = source_archive.expanduser().resolve()
    root_path = bundle_root.expanduser().resolve()
    if root_path.exists():
        raise FileExistsError(f"Training keypoint bundle already exists: {root_path}")
    root_path.mkdir(parents=True)
    crop = open_training_keypoint_crop_publication(
        archive,
        run_id=crop_run_id,
        require_consolidated=True,
    )
    source_root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    terminal_path = f"keypoint_shard_runs/{terminal_run_id}"
    terminal = source_root[terminal_path]
    if (
        not is_run_complete(terminal)
        or terminal.attrs.get("stage_selector_eligible") is not False
        or terminal.attrs.get("artifact_mutability") != "raw_immutable"
    ):
        raise ValueError("Terminal keypoint run is not complete immutable evidence.")
    if terminal.attrs.get("source_crop_run") != crop.run_id:
        raise ValueError("Terminal keypoint run binds a different training crop.")
    missing = sorted(set(_TERMINAL_ARRAYS) - set(terminal))
    if missing:
        raise ValueError(f"Terminal keypoint run lacks arrays: {missing!r}.")
    pose_binding = _pose_binding(dict(terminal.attrs))
    preprocessing = _preprocessing(
        dict(terminal.attrs), terminal_path=terminal_path, crop=crop
    )
    n_keypoints = int(terminal["keypoints_roi"].shape[1])
    dimensions = KeypointDimensions(
        n_frames=crop.dimensions.n_frames,
        n_instances=crop.dimensions.n_instances,
        n_keypoints=n_keypoints,
        source_width=crop.dimensions.source_width,
        source_height=crop.dimensions.source_height,
    )
    yolo_arrays = {path: terminal[path] for path in _TERMINAL_ARRAYS}
    if "pose_failure_codes" in terminal:
        yolo_arrays["pose_failure_codes"] = terminal["pose_failure_codes"]
    conversion = prepare_raw_keypoint_v2_from_yolo_arrays(
        yolo_arrays,
        dimensions=dimensions,
        source_crop_arrays=crop.arrays,
        source_crop_manifest=crop.manifest,
        pose_model_schema_binding=pose_binding,
        preprocessing=preprocessing,
    )
    terminal_metadata = sha256_file(archive / terminal_path / "zarr.json")
    run_ids = {
        "raw": raw_run_id,
        "quality": quality_run_id,
        "refined": refined_run_id,
        "body_frame": body_frame_run_id,
    }
    dispositions = _dispositions(
        source_archive=archive,
        terminal_path=terminal_path,
        terminal_metadata_sha256=terminal_metadata,
        run_ids=run_ids,
    )
    raw = publish_selector_ineligible_keypoint_snapshot(
        conversion.prepared,
        destination=root_path / "raw_keypoints.zarr",
        run_id=raw_run_id,
        shadow_root=root_path,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.raw,
    )
    skeleton_digest = keypoint_skeleton_digest(pose_binding)
    pose_schema = pose_binding["pose_schema"]
    quality_source = KeypointQualitySourceReference(
        run_name=raw.run_id,
        manifest_digest=canonical_json_sha256(raw.manifest),
        skeleton_id=pose_schema["skeleton_id"],
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(
            np.asarray(raw.prepared.arrays["keypoint_row_signature"])
        ),
    )
    quality_prepared = prepare_observation_local_keypoint_quality(
        raw.prepared.arrays,
        source_dimensions=raw.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        source=quality_source,
        skeleton_digest=skeleton_digest,
        policy=quality_policy,
    )
    quality = publish_selector_ineligible_keypoint_quality_snapshot(
        quality_prepared,
        source_manifest=raw.manifest,
        destination=root_path / "keypoint_quality.zarr",
        run_id=quality_run_id,
        shadow_root=root_path,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.quality,
    )
    review_state_map = {0: "unreviewed"}
    reason_code_map = {0: "none"}
    refined_prepared = prepare_refined_keypoint_snapshot(
        raw.prepared.arrays,
        dimensions=raw.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        skeleton_digest=skeleton_digest,
        keypoint_quality_arrays=quality.prepared.arrays,
        quality_dimensions=quality.prepared.dimensions,
        quality_profile=quality.prepared.profile,
        decisions=(),
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    refined_source = build_refined_keypoint_source_bindings(
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop.manifest,
    )
    if refined_identity.recording_identity != refined_source.recording_identity:
        raise ValueError("Refined identity binds a different training recording.")
    refined = publish_selector_ineligible_refined_keypoint_snapshot(
        refined_prepared,
        source=refined_source,
        raw_manifest=raw.manifest,
        quality_manifest=quality.manifest,
        crop_manifest=crop.manifest,
        raw_arrays=raw.prepared.arrays,
        quality_arrays=quality.prepared.arrays,
        source_crop_arrays=crop.arrays,
        identity=refined_identity,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
        destination=root_path / "refined_keypoints.zarr",
        run_id=refined_run_id,
        shadow_root=root_path,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.refined,
    )
    recipe = build_keypoint_body_frame_recipe(
        pose_schema=pose_schema,
        skeleton_digest=skeleton_digest,
        keypoint_count=n_keypoints,
    )
    body_source = BodyFrameSourceReference(
        stage="refined_keypoints",
        run_name=refined.run_id,
        manifest_digest=canonical_json_sha256(refined.manifest),
        skeleton_id=pose_schema["skeleton_id"],
        skeleton_digest=skeleton_digest,
        keypoint_row_signatures_digest=sha256_array(
            np.asarray(refined.prepared.arrays["keypoint_row_signature"])
        ),
    )
    body_prepared = prepare_keypoint_body_frame(
        refined.prepared.arrays,
        source_dimensions=refined.prepared.dimensions,
        source_crop_arrays=crop.arrays,
        source=body_source,
        source_manifest=refined.manifest,
        recipe=recipe,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    body_frame = publish_selector_ineligible_body_frame_snapshot(
        body_prepared,
        source_manifest=refined.manifest,
        destination=root_path / "body_frame.zarr",
        run_id=body_frame_run_id,
        shadow_root=root_path,
        storage_profile=storage_profile,
        created_by=created_by,
        disposition=dispositions.body_frame,
    )
    payload = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "source_archive": str(archive),
        "source_crop_run": crop.run_id,
        "source_crop_manifest_digest": canonical_json_sha256(crop.manifest),
        "terminal_run_path": terminal_path,
        "terminal_metadata_sha256": terminal_metadata,
        "conversion_receipt": dict(conversion.conversion_receipt),
        "runs": run_ids,
        "selector_eligible": False,
        "registry_registered": False,
        "review_edits": "separate_instance_key_delta_generations",
    }
    receipt = {
        "schema_id": TRAINING_KEYPOINT_REVIEW_CHAIN_SCHEMA_ID,
        "schema_version": TRAINING_KEYPOINT_REVIEW_CHAIN_SCHEMA_VERSION,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    receipt_path = root_path / TRAINING_KEYPOINT_REVIEW_CHAIN_RECEIPT
    receipt_path.write_text(
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return TrainingKeypointReviewChain(
        crop=crop,
        raw=raw,
        quality=quality,
        refined=refined,
        body_frame=body_frame,
        receipt_path=receipt_path,
        receipt=receipt,
    )


__all__ = [
    "TRAINING_KEYPOINT_REVIEW_CHAIN_RECEIPT",
    "TRAINING_KEYPOINT_REVIEW_CHAIN_SCHEMA_ID",
    "TrainingKeypointCropPublication",
    "TrainingKeypointReviewChain",
    "open_training_keypoint_crop_publication",
    "publish_training_keypoint_review_candidate_chain",
]

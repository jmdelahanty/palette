"""Bind sampled-frame detector artifacts into immutable review seeds.

The ordinary canonical detection schema uses the complete acquisition frame
axis and therefore requires ``frame_indices == source_acquisition_frame_index``.
A sampled training Zarr has a different, equally explicit axis: local review
rows ``0..F-1`` map through ``raw_video/original_frame_indices`` to acquisition
frames.  This module owns that distinction without weakening the recording-
level schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.detection_producer_lifecycle import (
    ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR,
    DETECTION_ARTIFACT_RUN_FAMILY,
    require_detection_artifact_parent,
    validate_artifact_payload_inventory_seal,
)
from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.array_factory import (
    create_array_from_plan,
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array, utc_now
from fisheye.shared.zarr.canonical_detection_manifest import (
    normalize_canonical_detection_metadata_declarations,
)
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.detection_storage import (
    CanonicalDetectionStoragePlanSet,
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    StorageProfile,
)
from fisheye.shared.zarr.training_image_storage import (
    SAMPLED_TRAINING_DOWNSAMPLE_TRANSFORM_SCHEMA_ID,
    SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    COMPLETION_EPOCH_STRICT,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)

SAMPLED_TRAINING_DETECTION_SCHEMA_ID = "palette.stage.sampled_training_detection"
SAMPLED_TRAINING_DETECTION_SCHEMA_VERSION = 1
SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_ID = (
    "palette.sampled_training_detection.run_manifest"
)
SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_VERSION = 1
SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_ID = (
    "palette.sampled_training_detection.publication"
)
SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_VERSION = 1
SAMPLED_TRAINING_DETECTION_PUBLICATION_POLICY = (
    "node_local_access_aware_candidate_checked_atomic_run_copy_v1"
)
SAMPLED_TRAINING_DETECTION_ROLLBACK_POLICY = (
    "retain_owner_bound_selector_ineligible_tombstone_v1"
)
SAMPLED_TRAINING_DETECTION_RUN_MANIFEST_ATTR = "run_manifest"
SAMPLED_TRAINING_DETECTION_MAPPING_PATH = "raw_video/original_frame_indices"
_SELECTOR_ATTRS = ("authoritative_run", "latest_complete", "latest")
_ARTIFACT_FRAME_SOURCE_LINEAGE_SCHEMA_ID = (
    "palette.training_detection_artifact_frame_source_lineage.v1"
)
_FRAME_SOURCE_BINDING_SCHEMA_ID = (
    "palette.sampled_training_detection.frame_source_binding.v1"
)


@dataclass(frozen=True)
class SampledTrainingDetectionDimensions:
    """Concrete sampled review axis plus its source-camera extent."""

    n_frames: int
    n_instances: int
    source_total_frames: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        for name in (
            "n_frames",
            "n_instances",
            "source_total_frames",
            "source_width",
            "source_height",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer.")
        if self.n_frames <= 0:
            raise ValueError("Sampled detection n_frames must be positive.")
        if self.n_instances < 0:
            raise ValueError("Sampled detection n_instances cannot be negative.")
        if self.source_total_frames <= 0:
            raise ValueError("source_total_frames must be positive.")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Source-camera dimensions must be positive.")
        if self.n_frames > int(np.iinfo(np.int32).max):
            raise ValueError("Sampled frame axis exceeds int32.")

    @property
    def contract_dimensions(self) -> dict[str, int]:
        return {
            "n_frames": self.n_frames,
            "n_instances": self.n_instances,
            "n_frame_boundaries": self.n_frames + 1,
        }

    def as_manifest(self) -> dict[str, int]:
        return {
            **self.contract_dimensions,
            "source_total_frames": self.source_total_frames,
            "source_width": self.source_width,
            "source_height": self.source_height,
        }

    def planner_dimensions(self) -> CanonicalDetectionDimensions:
        return CanonicalDetectionDimensions(
            n_frames=self.n_frames,
            n_instances=self.n_instances,
            source_width=self.source_width,
            source_height=self.source_height,
        )


@dataclass(frozen=True)
class SampledTrainingDetectionCandidate:
    output_path: Path
    run_id: str
    dimensions: SampledTrainingDetectionDimensions
    mapping: np.ndarray
    arrays: Mapping[str, Any]
    plans: CanonicalDetectionStoragePlanSet
    manifest: Mapping[str, object]
    statistics: Mapping[str, object]

    @property
    def run_path(self) -> Path:
        return self.output_path / "detect_runs" / self.run_id


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _direct_declarations(run_path: Path) -> dict[str, dict[str, Any]]:
    relative_paths = (
        "",
        "instances",
        *CANONICAL_DETECTION_SCHEMA_V1.binding_paths,
    )
    declarations: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        node = run_path if not relative else run_path / relative
        declarations[relative] = _strict_json(node / "zarr.json")
    return declarations


def _sampled_metadata_document(
    declarations: Mapping[str, Mapping[str, Any]],
    *,
    dimensions: SampledTrainingDetectionDimensions,
) -> dict[str, object]:
    normalized = normalize_canonical_detection_metadata_declarations(
        declarations,
        consolidated_metadata_by_path=declarations,
        dimensions=dimensions.planner_dimensions(),
    )
    return {
        "schema_id": "palette.sampled_training_detection.metadata_declarations",
        "schema_version": 1,
        "scope": "run_groups_and_arrays_direct_metadata_excluding_attributes",
        "dimensions": dimensions.as_manifest(),
        "declarations": normalized["declarations"],
    }


def _mapping_from_archive(
    archive: Path,
) -> tuple[Any, Any, np.ndarray, SampledTrainingDetectionDimensions]:
    root = open_zarr_group_direct(archive, mode="r")
    if str(root.attrs.get("zarr_purpose") or "").strip().lower() != "training":
        raise ValueError("Sampled detection binding requires a training-purpose Zarr.")
    raw = root.get("raw_video")
    if raw is None:
        raise ValueError("Training Zarr lacks raw_video.")
    for name in ("images_full", "original_frame_indices"):
        if name not in raw:
            raise ValueError(f"Training Zarr lacks raw_video/{name}.")
    full = raw["images_full"]
    if len(full.shape) not in {3, 4} or np.dtype(full.dtype) != np.dtype("uint8"):
        raise ValueError("raw_video/images_full must be rank-3/4 uint8.")
    mapping_node = raw["original_frame_indices"]
    mapping_raw = np.asarray(mapping_node[:])
    n_frames = int(full.shape[0])
    if mapping_raw.shape != (n_frames,) or not np.issubdtype(
        mapping_raw.dtype, np.integer
    ):
        raise ValueError(
            "raw_video/original_frame_indices must be one integer per sampled frame."
        )
    mapping = np.asarray(mapping_raw, dtype=np.int64)
    if np.any(mapping < 0) or (
        mapping.size > 1 and np.any(mapping[1:] <= mapping[:-1])
    ):
        raise ValueError(
            "raw_video/original_frame_indices must be nonnegative and strictly increasing."
        )
    source_total = int(
        raw.attrs.get("source_frame_count")
        or raw.attrs.get("original_video_length")
        or 0
    )
    if source_total <= 0 or (mapping.size and int(mapping[-1]) >= source_total):
        raise ValueError(
            "Sampled frame mapping exceeds the declared source frame axis."
        )
    source_height = int(
        root.attrs.get("source_video_height")
        or raw.attrs.get("source_video_height")
        or full.shape[1]
    )
    source_width = int(
        root.attrs.get("source_video_width")
        or raw.attrs.get("source_video_width")
        or full.shape[2]
    )
    if (int(full.shape[1]), int(full.shape[2])) != (source_height, source_width):
        raise ValueError(
            "raw_video/images_full extent differs from the source-camera authority."
        )
    dimensions = SampledTrainingDetectionDimensions(
        n_frames=n_frames,
        n_instances=0,
        source_total_frames=source_total,
        source_width=source_width,
        source_height=source_height,
    )
    return root, raw, mapping, dimensions


def _artifact_path(archive: Path, run_id: str) -> Path:
    name = str(run_id).strip()
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("artifact_run_id must be one path-safe group name.")
    return archive / DETECTION_ARTIFACT_RUN_FAMILY / name


def _artifact_evidence(
    archive: Path,
    *,
    artifact_run_id: str,
    raw: Any,
    dimensions: SampledTrainingDetectionDimensions,
) -> tuple[Any, dict[str, Any]]:
    path = _artifact_path(archive, artifact_run_id)
    if not path.is_dir():
        raise FileNotFoundError(f"Detection artifact run not found: {path}")
    artifact = zarr.open_group(str(path), mode="r", use_consolidated=False)
    seal = validate_artifact_payload_inventory_seal(artifact)
    if artifact.attrs.get("schema_id") != "palette.training_detection_artifact.v1":
        raise ValueError("Artifact is not a sampled-training detector output.")
    if artifact.attrs.get("stage_selector_eligible") is not False:
        raise ValueError("Detection artifact must remain selector-ineligible.")
    if artifact.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise ValueError("Detection artifact is not complete.")
    if "source_frame_indices" not in artifact:
        raise ValueError(
            "Detection artifact lacks sampled-to-acquisition frame lineage."
        )
    lineage = artifact.attrs.get("artifact_frame_source_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("Detection artifact lacks frame-source lineage.")
    lineage_digest = str(
        artifact.attrs.get("artifact_frame_source_lineage_sha256") or ""
    )
    if lineage_digest != canonical_json_sha256(lineage):
        raise ValueError("Detection artifact frame-source lineage digest is stale.")
    frame_source_binding = _validate_artifact_frame_source_binding(
        raw=raw,
        artifact=artifact,
        lineage=lineage,
        dimensions=dimensions,
    )
    return artifact, {
        "artifact_path": f"{DETECTION_ARTIFACT_RUN_FAMILY}/{artifact_run_id}",
        "artifact_payload_inventory_seal_sha256": str(
            artifact.attrs.get(f"{ARTIFACT_PAYLOAD_INVENTORY_SEAL_ATTR}_sha256")
        ),
        "artifact_payload_inventory": seal,
        "artifact_frame_source_lineage_sha256": lineage_digest,
        "frame_source_binding": frame_source_binding,
        "model_registry_run_id": str(artifact.attrs.get("model_registry_run_id")),
        "model_registry_set_id": artifact.attrs.get("model_registry_set_id"),
        "model_path": str(artifact.attrs.get("model_path")),
    }


def _exact_int_list(value: Any, *, field: str, length: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must be an exact {length}-element integer list.")
    if any(type(item) is not int for item in value):
        raise ValueError(f"{field} must contain exact integers.")
    return [int(item) for item in value]


def _validate_artifact_frame_source_binding(
    *,
    raw: Any,
    artifact: Any,
    lineage: Mapping[str, Any],
    dimensions: SampledTrainingDetectionDimensions,
) -> dict[str, Any]:
    """Prove that normalized artifact boxes use the source-camera domain.

    Full-camera inference is an identity binding.  Downsampled inference is
    accepted only when the selected array carries the exact sampled-training
    image contract and a zero-padding source-to-stored transform.  With no
    padding, normalized coordinates are invariant under the resize; arbitrary
    letterboxing remains unsupported because it requires an explicit inverse
    box transform.
    """

    frame_source = str(artifact.attrs.get("frame_source_path") or "")
    if frame_source not in {"raw_video/images_full", "raw_video/images_ds"}:
        raise ValueError(
            "Canonical sampled binding requires raw_video/images_full or "
            "contract-bound raw_video/images_ds inference."
        )
    array_name = frame_source.split("/", 1)[1]
    if array_name not in raw:
        raise ValueError(
            f"Artifact frame source is absent from the archive: {frame_source}"
        )
    frame_array = raw[array_name]
    shape = [int(value) for value in frame_array.shape]
    if len(shape) not in {3, 4} or shape[0] != dimensions.n_frames:
        raise ValueError(
            "Artifact frame-source array shape is incompatible with the sampled axis."
        )
    channels = 1 if len(shape) == 3 else int(shape[3])
    height, width = int(shape[1]), int(shape[2])
    chunks = getattr(frame_array, "chunks", None)
    chunk_shape = [int(value) for value in chunks] if chunks is not None else None

    expected_lineage_fields = {
        "schema_id",
        "status",
        "selected_array_path",
        "selected_array_shape",
        "selected_array_dtype",
        "selected_array_chunks",
        "frame_row_count",
        "frame_source_extent",
        "selection",
        "model_request",
        "prediction_result_basis",
        "pixel_content_binding_status",
        "original_frame_mapping",
        "source_camera_overlay_suitability",
        "unbound_numeric_binding",
    }
    if set(lineage) != expected_lineage_fields:
        raise ValueError(
            "Detection artifact frame-source lineage fields are not exact."
        )
    if (
        lineage.get("schema_id") != _ARTIFACT_FRAME_SOURCE_LINEAGE_SCHEMA_ID
        or lineage.get("status") != "unbound_artifact_provenance_only"
        or lineage.get("selected_array_path") != frame_source
        or lineage.get("selected_array_shape") != shape
        or lineage.get("selected_array_dtype") != np.dtype(frame_array.dtype).str
        or lineage.get("selected_array_chunks") != chunk_shape
        or lineage.get("frame_row_count") != dimensions.n_frames
        or lineage.get("prediction_result_basis")
        != "ultralytics_xyxy_rescaled_to_selected_training_frame_array_extent"
    ):
        raise ValueError(
            "Detection artifact frame-source lineage differs from the live array."
        )
    if lineage.get("frame_source_extent") != {
        "width": width,
        "height": height,
        "channels": channels,
        "units": "pixels",
        "extent_basis": "selected_training_frame_array_shape",
    }:
        raise ValueError("Detection artifact frame-source extent is stale.")
    selection = lineage.get("selection")
    if not isinstance(selection, Mapping) or (
        selection.get("path") != frame_source
        or selection.get("shape") != shape
        or selection.get("n_frames") != dimensions.n_frames
        or selection.get("height") != height
        or selection.get("width") != width
        or selection.get("channels") != channels
    ):
        raise ValueError("Detection artifact frame-source selection is stale.")
    unbound = lineage.get("unbound_numeric_binding")
    reference = unbound.get("reference") if isinstance(unbound, Mapping) else None
    temporal = unbound.get("temporal") if isinstance(unbound, Mapping) else None
    if (
        not isinstance(reference, Mapping)
        or reference.get("node_path") != frame_source
        or reference.get("width") != width
        or reference.get("height") != height
        or not isinstance(temporal, Mapping)
        or temporal.get("source_frame_count") != dimensions.n_frames
    ):
        raise ValueError("Detection artifact numeric binding is stale.")

    if frame_source == "raw_video/images_full":
        if (height, width) != (dimensions.source_height, dimensions.source_width):
            raise ValueError(
                "Full-camera artifact extent differs from source-camera authority."
            )
        return {
            "schema_id": _FRAME_SOURCE_BINDING_SCHEMA_ID,
            "artifact_frame_source_path": frame_source,
            "coordinate_mapping": "source_camera_normalized_identity",
            "source_shape_hw": [dimensions.source_height, dimensions.source_width],
            "stored_shape_hw": [height, width],
            "downsample_transform": None,
            "downsample_transform_sha256": None,
        }

    if len(shape) != 3 or np.dtype(frame_array.dtype) != np.dtype("uint8"):
        raise ValueError("Contract-bound raw_video/images_ds must be rank-3 uint8.")
    if (
        frame_array.attrs.get("storage_contract_schema_id")
        != (SAMPLED_TRAINING_IMAGE_STORAGE_SCHEMA_ID)
        or frame_array.attrs.get("access_unit") != "complete_sampled_frame"
    ):
        raise ValueError(
            "raw_video/images_ds lacks the exact sampled-image storage contract."
        )
    transform = frame_array.attrs.get("source_to_stored_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("raw_video/images_ds lacks source_to_stored_transform.")
    transform = dict(transform)
    if set(transform) != {
        "schema_id",
        "source_shape_hw",
        "stored_shape_hw",
        "resized_shape_hw",
        "padding_tblr",
        "mode",
        "interpolation",
        "padding_value_uint8",
    }:
        raise ValueError("raw_video/images_ds transform fields are not exact.")
    source_hw = _exact_int_list(
        transform.get("source_shape_hw"), field="source_shape_hw", length=2
    )
    stored_hw = _exact_int_list(
        transform.get("stored_shape_hw"), field="stored_shape_hw", length=2
    )
    resized_hw = _exact_int_list(
        transform.get("resized_shape_hw"), field="resized_shape_hw", length=2
    )
    padding = _exact_int_list(
        transform.get("padding_tblr"), field="padding_tblr", length=4
    )
    if (
        transform.get("schema_id") != SAMPLED_TRAINING_DOWNSAMPLE_TRANSFORM_SCHEMA_ID
        or source_hw != [dimensions.source_height, dimensions.source_width]
        or stored_hw != [height, width]
        or resized_hw != stored_hw
        or padding != [0, 0, 0, 0]
        or transform.get("mode") not in {"direct_resize", "aspect_preserving_letterbox"}
        or type(transform.get("interpolation")) is not str
        or not str(transform.get("interpolation"))
        or transform.get("padding_value_uint8") != 0
    ):
        raise ValueError(
            "Canonical images_ds binding requires an exact zero-padding resize "
            "from the source-camera extent."
        )
    parent_transform = raw.attrs.get("downsample_transform")
    if not isinstance(parent_transform, Mapping) or dict(parent_transform) != transform:
        raise ValueError("raw_video and images_ds downsample transforms disagree.")
    return {
        "schema_id": _FRAME_SOURCE_BINDING_SCHEMA_ID,
        "artifact_frame_source_path": frame_source,
        "coordinate_mapping": "source_camera_normalized_identity_via_zero_padding_resize",
        "source_shape_hw": source_hw,
        "stored_shape_hw": stored_hw,
        "downsample_transform": transform,
        "downsample_transform_sha256": canonical_json_sha256(transform),
    }


def _expected_arrays(
    archive: Path,
    *,
    artifact_run_id: str,
) -> tuple[
    SampledTrainingDetectionDimensions,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, object],
]:
    root, raw, mapping, base_dimensions = _mapping_from_archive(archive)
    artifact, artifact_evidence = _artifact_evidence(
        archive,
        artifact_run_id=artifact_run_id,
        raw=raw,
        dimensions=base_dimensions,
    )
    local_frames = np.asarray(artifact["frame_indices"][:], dtype=np.int32)
    bbox_norm = np.asarray(artifact["bbox_norm_coords"][:], dtype=np.float32)
    scores = np.asarray(artifact["scores"][:], dtype=np.float32)
    class_ids = np.asarray(artifact["class_ids"][:], dtype=np.int32)
    source_frames = np.asarray(artifact["source_frame_indices"][:], dtype=np.int64)
    row_count = int(local_frames.shape[0])
    if not (
        bbox_norm.shape == (row_count, 4)
        and scores.shape == (row_count,)
        and class_ids.shape == (row_count,)
        and source_frames.shape == (row_count,)
    ):
        raise ValueError("Detection artifact arrays do not share one row cardinality.")
    if local_frames.size and (
        np.any(local_frames < 0) or np.any(local_frames >= base_dimensions.n_frames)
    ):
        raise ValueError(
            "Detection artifact frame_indices exceed the sampled frame axis."
        )
    if row_count and not np.array_equal(
        source_frames,
        mapping[local_frames.astype(np.int64, copy=False)],
    ):
        raise ValueError(
            "Artifact source_frame_indices differ from raw_video/original_frame_indices."
        )
    instance_keys = mint_detection_instance_keys(
        recording_identity=str(root.attrs.get("recording_id") or ""),
        frame_indices=source_frames,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=base_dimensions.source_width,
        source_height=base_dimensions.source_height,
    )
    counts = np.bincount(
        local_frames.astype(np.int64, copy=False),
        minlength=base_dimensions.n_frames,
    )
    offsets = np.zeros(base_dimensions.n_frames + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    dimensions = SampledTrainingDetectionDimensions(
        n_frames=base_dimensions.n_frames,
        n_instances=row_count,
        source_total_frames=base_dimensions.source_total_frames,
        source_width=base_dimensions.source_width,
        source_height=base_dimensions.source_height,
    )
    arrays = {
        "instances/frame_indices": local_frames,
        "instances/source_acquisition_frame_index": source_frames,
        "instances/instance_key": instance_keys,
        "instances/bbox_norm_coords": bbox_norm,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": scores,
        "instances/class_ids": class_ids,
        "instances/frame_row_offsets": offsets,
    }
    errors = validate_sampled_training_detection_arrays(
        arrays,
        dimensions=dimensions,
        sampled_frame_mapping=mapping,
        recording_identity=str(root.attrs.get("recording_id") or ""),
    )
    if errors:
        raise ValueError("Sampled detection binding failed: " + "; ".join(errors))
    statistics: dict[str, object] = {
        "sampled_frames": dimensions.n_frames,
        "detection_rows": row_count,
        "frames_with_zero_detections": int(np.count_nonzero(counts == 0)),
        "frames_with_one_detection": int(np.count_nonzero(counts == 1)),
        "frames_with_multiple_detections": int(np.count_nonzero(counts > 1)),
        "one_fish_per_frame_expectation_pass": bool(np.all(counts == 1)),
        "cardinality_policy": (
            "report_exact_detector_evidence_do_not_force_top_one_at_inference"
        ),
    }
    return dimensions, mapping, arrays, artifact_evidence, statistics


def validate_sampled_training_detection_arrays(
    arrays: Mapping[str, Any],
    *,
    dimensions: SampledTrainingDetectionDimensions,
    sampled_frame_mapping: Any,
    recording_identity: str,
) -> tuple[str, ...]:
    """Validate local review frames and their exact acquisition-frame join."""

    mapping = np.asarray(sampled_frame_mapping, dtype=np.int64).reshape(-1)
    errors: list[str] = []
    if mapping.shape != (dimensions.n_frames,):
        errors.append("sampled frame mapping shape mismatch")
        return tuple(errors)
    if np.any(mapping < 0) or (
        mapping.size > 1 and np.any(mapping[1:] <= mapping[:-1])
    ):
        errors.append(
            "sampled frame mapping must be nonnegative and strictly increasing"
        )
    if mapping.size and int(mapping[-1]) >= dimensions.source_total_frames:
        errors.append("sampled frame mapping exceeds source_total_frames")

    canonical_issues = CANONICAL_DETECTION_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions.planner_dimensions(),
    )
    errors.extend(
        f"{issue.code} at {issue.path}: {issue.message}"
        for issue in canonical_issues
        if issue.code != "source_frame_identity_mismatch"
    )
    try:
        local = np.asarray(arrays["instances/frame_indices"][:], dtype=np.int64)
        source = np.asarray(
            arrays["instances/source_acquisition_frame_index"][:], dtype=np.int64
        )
        expected_source = (
            mapping[local] if local.size else np.empty((0,), dtype=np.int64)
        )
        if not np.array_equal(source, expected_source):
            errors.append(
                "source_acquisition_frame_index must equal original_frame_indices[frame_indices]"
            )
        expected_keys = mint_detection_instance_keys(
            recording_identity=str(recording_identity),
            frame_indices=source,
            bbox_norm_coords=np.asarray(arrays["instances/bbox_norm_coords"][:]),
            class_ids=np.asarray(arrays["instances/class_ids"][:]),
        )
        if not np.array_equal(
            np.asarray(arrays["instances/instance_key"][:]), expected_keys
        ):
            errors.append("instance_key is not minted from acquisition-frame identity")
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"sampled detection lineage validation failed: {exc}")
    return tuple(dict.fromkeys(errors))


def _storage_manifest(
    plans: CanonicalDetectionStoragePlanSet,
    *,
    dimensions: SampledTrainingDetectionDimensions,
) -> dict[str, object]:
    value = plans.as_manifest()
    value["schema_id"] = "palette.stage_storage.sampled_training_detection"
    value["schema_version"] = 1
    value["logical_stage_schema"] = {
        "id": SAMPLED_TRAINING_DETECTION_SCHEMA_ID,
        "version": SAMPLED_TRAINING_DETECTION_SCHEMA_VERSION,
    }
    value["dimensions"] = dimensions.as_manifest()
    return value


def _logical_schema_manifest(
    dimensions: SampledTrainingDetectionDimensions,
    *,
    mapping: np.ndarray,
) -> dict[str, object]:
    return {
        "schema_id": SAMPLED_TRAINING_DETECTION_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_DETECTION_SCHEMA_VERSION,
        "stage": "detect",
        "layout": "sampled_training_sparse_instances_with_frame_row_offsets_v1",
        "base_path": "detect_runs/<run>",
        "instance_group": "instances",
        "dimensions": dimensions.as_manifest(),
        "bindings": [
            binding.as_manifest() for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
        ],
        "frame_axis": {
            "local_domain": "raw_video_sampled_frame_row",
            "source_domain": "source_acquisition_frame_index",
            "mapping_path": SAMPLED_TRAINING_DETECTION_MAPPING_PATH,
            "mapping_dtype": str(mapping.dtype),
            "mapping_shape": list(mapping.shape),
            "mapping_sha256": sha256_array(mapping),
            "join": (
                "source_acquisition_frame_index="
                "original_frame_indices[frame_indices]"
            ),
        },
        "invariants": {
            "row_order": "frame_indices_nondecreasing",
            "row_identity": "instance_key",
            "instance_key_derivation": (
                "recording_id_source_acquisition_frame_bbox_class_v1"
            ),
            "frame_lookup": "frame_row_offsets_on_sampled_local_frame_axis",
            "instances_per_frame": "zero_one_or_many",
            "geometry_authority": "bbox_norm_coords_source_camera_normalized",
            "source_image_extent": "raw_video/images_full",
            "counts_aliases": "forbidden_from_logical_binding",
        },
    }


def _build_manifest(
    *,
    run_id: str,
    dimensions: SampledTrainingDetectionDimensions,
    mapping: np.ndarray,
    arrays: Mapping[str, Any],
    plans: CanonicalDetectionStoragePlanSet,
    artifact_evidence: Mapping[str, Any],
    declarations: Mapping[str, Mapping[str, Any]],
    statistics: Mapping[str, object],
) -> dict[str, object]:
    metadata_document = _sampled_metadata_document(
        declarations,
        dimensions=dimensions,
    )
    logical_content = {
        path: {
            "shape": list(np.asarray(value[:]).shape),
            "dtype": str(np.asarray(value[:]).dtype),
            "sha256": sha256_array(np.asarray(value[:])),
        }
        for path, value in arrays.items()
    }
    payload = {
        "run_id": str(run_id),
        "stage": "detect",
        "publication": {
            "status": "complete",
            "stage_selector_eligible": False,
            "metadata_mode": "direct_mutable_archive",
            "selector_activation": "deferred_until_review",
        },
        "logical_schema": _logical_schema_manifest(dimensions, mapping=mapping),
        "storage_plan": _storage_manifest(plans, dimensions=dimensions),
        "coordinate_contract": {
            "bbox_norm_coords": "source_camera_normalized_cxcywh_float32",
            "bbox_img_xyxy": "source_camera_pixel_edges_float32",
            "centers_img_xy": "source_camera_pixel_centers_float32",
            "source_image_path": "raw_video/images_full",
            "source_width": dimensions.source_width,
            "source_height": dimensions.source_height,
        },
        "source_artifact": dict(artifact_evidence),
        "sampled_frame_mapping": {
            "path": SAMPLED_TRAINING_DETECTION_MAPPING_PATH,
            "dtype": str(mapping.dtype),
            "shape": list(mapping.shape),
            "sha256": sha256_array(mapping),
            "source_total_frames": dimensions.source_total_frames,
        },
        "logical_content": {
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "arrays": logical_content,
        },
        "metadata_declarations": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(metadata_document),
            "document": metadata_document,
        },
        "cardinality_statistics": dict(statistics),
    }
    envelope = {
        "schema_id": SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": SAMPLED_TRAINING_DETECTION_RUN_MANIFEST_ATTR,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _write_by_physical_units(
    destination: Any,
    values: np.ndarray,
    *,
    plan: Any,
) -> None:
    chunk = plan.chunk_shape
    if chunk is None:
        raise ValueError("Detection arrays cannot be scalars.")
    rows = int(plan.shard_shape[0] if plan.shard_shape is not None else chunk[0])
    trailing = (slice(None),) * (values.ndim - 1)
    for start in range(0, int(values.shape[0]), rows):
        stop = min(start + rows, int(values.shape[0]))
        selection = (slice(start, stop), *trailing)
        destination[selection] = values[selection]


def build_sampled_training_detection_candidate(
    *,
    archive: str | Path,
    artifact_run_id: str,
    destination: str | Path,
    run_id: str,
    profile: StorageProfile = DETECTION_PUBLISHED_ACCESS_AWARE_V1,
) -> SampledTrainingDetectionCandidate:
    """Build and deeply validate one node-local immutable review seed."""

    if profile.profile_id != DETECTION_PUBLISHED_ACCESS_AWARE_V1.profile_id:
        raise ValueError(
            "Sampled training detection v1 freezes the access-aware detection "
            f"profile {DETECTION_PUBLISHED_ACCESS_AWARE_V1.profile_id!r}."
        )
    source = Path(archive).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Sampled detection candidate exists: {output}")
    if not (
        output.is_relative_to(Path("/tmp").resolve())
        or ".palette_benchmarks" in output.parts
    ):
        raise ValueError(
            "Candidate destination must be below /tmp or .palette_benchmarks."
        )
    name = str(run_id).strip()
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("run_id must be one path-safe group name.")
    dimensions, mapping, expected, artifact_evidence, statistics = _expected_arrays(
        source,
        artifact_run_id=artifact_run_id,
    )
    planner_dimensions = dimensions.planner_dimensions()
    plans = plan_canonical_detection_storage(planner_dimensions, profile=profile)
    root = zarr.open_group(str(output), mode="w-", zarr_format=3)
    parent = require_runs_parent(
        root, "detect_runs", completion_epoch=COMPLETION_EPOCH_STRICT
    )
    run = parent.create_group(name)
    mark_run_started(run, run_name=name, stage="detect")
    run.attrs.update(
        {
            "status": "running",
            "stage_selector_eligible": False,
            "immutable_snapshot": True,
            "production_candidate": True,
            "production_selector_activation": "deferred",
            "logical_schema": _logical_schema_manifest(dimensions, mapping=mapping),
            "source_artifact_path": artifact_evidence["artifact_path"],
        }
    )
    instances = run.create_group("instances")
    arrays: dict[str, Any] = {}
    binding_by_path = {
        binding.path: binding for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
    }
    for entry in plans.entries:
        path = entry.rule.path
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
            attributes={"selector_eligible": False},
        )
        _write_by_physical_units(array, expected[path], plan=entry.plan)
        arrays[path] = array
    declarations = _direct_declarations(output / "detect_runs" / name)
    manifest = _build_manifest(
        run_id=name,
        dimensions=dimensions,
        mapping=mapping,
        arrays=arrays,
        plans=plans,
        artifact_evidence=artifact_evidence,
        declarations=declarations,
        statistics=statistics,
    )
    run.attrs[SAMPLED_TRAINING_DETECTION_RUN_MANIFEST_ATTR] = manifest
    run.attrs["status"] = "complete"
    mark_run_complete(
        run,
        run_name=name,
        run_provenance=build_writer_run_provenance(
            command="fisheye.shared.zarr.sampled_training_detection_publication",
            params={
                "run_id": name,
                "artifact_run_id": str(artifact_run_id),
                "storage_profile": profile.profile_id,
            },
            input_artifacts=[
                {
                    "role": "unbound_detection_artifact",
                    "path": str(_artifact_path(source, artifact_run_id)),
                },
                {
                    "role": "sampled_frame_mapping",
                    "path": f"{source}/{SAMPLED_TRAINING_DETECTION_MAPPING_PATH}",
                },
            ],
        ),
    )
    candidate = SampledTrainingDetectionCandidate(
        output_path=output,
        run_id=name,
        dimensions=dimensions,
        mapping=mapping,
        arrays=arrays,
        plans=plans,
        manifest=manifest,
        statistics=statistics,
    )
    validation = validate_sampled_training_detection_run(
        candidate.run_path,
        archive=source,
    )
    if not validation["valid"]:
        raise RuntimeError(f"Sampled detection candidate is invalid: {validation}")
    return candidate


def validate_sampled_training_detection_run(
    run_path: str | Path,
    *,
    archive: str | Path,
) -> dict[str, Any]:
    """Recompute the complete contract from direct metadata and live sources."""

    path = Path(run_path).expanduser().resolve()
    source = Path(archive).expanduser().resolve()
    errors: list[str] = []
    try:
        run = zarr.open_group(str(path), mode="r", use_consolidated=False)
        manifest = run.attrs.get(SAMPLED_TRAINING_DETECTION_RUN_MANIFEST_ATTR)
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "schema_id",
            "schema_version",
            "persisted_attribute",
            "digest_algorithm",
            "payload_digest",
            "payload",
        }:
            raise ValueError("Sampled detection manifest envelope is not exact.")
        payload = manifest.get("payload")
        if (
            manifest.get("schema_id") != SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_ID
            or manifest.get("schema_version")
            != SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_VERSION
            or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
            or not isinstance(payload, Mapping)
            or manifest.get("payload_digest") != canonical_json_sha256(payload)
        ):
            raise ValueError("Sampled detection manifest header or digest is invalid.")
        if set(payload) != {
            "run_id",
            "stage",
            "publication",
            "logical_schema",
            "storage_plan",
            "coordinate_contract",
            "source_artifact",
            "sampled_frame_mapping",
            "logical_content",
            "metadata_declarations",
            "cardinality_statistics",
        }:
            raise ValueError("Sampled detection manifest payload fields are not exact.")
        artifact_info = payload["source_artifact"]
        if not isinstance(artifact_info, Mapping):
            raise ValueError("Sampled detection source_artifact is invalid.")
        artifact_relative = str(artifact_info.get("artifact_path") or "")
        parts = artifact_relative.split("/")
        if len(parts) != 2 or parts[0] != DETECTION_ARTIFACT_RUN_FAMILY:
            raise ValueError("Sampled detection artifact path is invalid.")
        dimensions, mapping, expected, expected_artifact, statistics = _expected_arrays(
            source,
            artifact_run_id=parts[1],
        )
        if dict(artifact_info) != expected_artifact:
            errors.append("source artifact evidence differs from the live artifact")
        if payload["logical_schema"] != _logical_schema_manifest(
            dimensions, mapping=mapping
        ):
            errors.append("logical schema differs from the frozen sampled contract")
        if payload["sampled_frame_mapping"] != {
            "path": SAMPLED_TRAINING_DETECTION_MAPPING_PATH,
            "dtype": str(mapping.dtype),
            "shape": list(mapping.shape),
            "sha256": sha256_array(mapping),
            "source_total_frames": dimensions.source_total_frames,
        }:
            errors.append("sampled frame mapping evidence is stale")
        arrays = {
            path_name: run[path_name]
            for path_name in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        errors.extend(
            validate_sampled_training_detection_arrays(
                arrays,
                dimensions=dimensions,
                sampled_frame_mapping=mapping,
                recording_identity=str(
                    open_zarr_group_direct(source, mode="r").attrs.get("recording_id")
                    or ""
                ),
            )
        )
        for path_name, expected_values in expected.items():
            observed = np.asarray(arrays[path_name][:])
            if not np.array_equal(observed, expected_values):
                errors.append(
                    f"decoded values differ from artifact binding at {path_name}"
                )
        logical_content = {
            path_name: {
                "shape": list(np.asarray(arrays[path_name][:]).shape),
                "dtype": str(np.asarray(arrays[path_name][:]).dtype),
                "sha256": sha256_array(np.asarray(arrays[path_name][:])),
            }
            for path_name in CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        }
        if payload["logical_content"] != {
            "digest_algorithm": "sha256_c_contiguous_bytes_v1",
            "arrays": logical_content,
        }:
            errors.append("logical content hashes are stale")
        declarations = _direct_declarations(path)
        metadata_document = _sampled_metadata_document(
            declarations,
            dimensions=dimensions,
        )
        if payload["metadata_declarations"] != {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(metadata_document),
            "document": metadata_document,
        }:
            errors.append("direct metadata declaration digest is stale")
        plans = plan_canonical_detection_storage(dimensions.planner_dimensions())
        if payload["storage_plan"] != _storage_manifest(
            plans,
            dimensions=dimensions,
        ):
            errors.append("storage plan differs from the frozen access-aware profile")
        binding_by_path = {
            binding.path: binding for binding in CANONICAL_DETECTION_SCHEMA_V1.bindings
        }
        entry_by_path = {entry.rule.path: entry for entry in plans.entries}
        for path_name in CANONICAL_DETECTION_SCHEMA_V1.binding_paths:
            binding = binding_by_path[path_name]
            contract = CANONICAL_DETECTION_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            errors.extend(
                f"{path_name}: {message}"
                for message in validate_array_metadata_declaration_from_plan(
                    declarations[path_name],
                    contract=contract,
                    plan=entry_by_path[path_name].plan,
                    fill_value=0,
                )
            )
        if payload["cardinality_statistics"] != statistics:
            errors.append("cardinality statistics differ from live offsets")
        if run.attrs.get("status") != "complete":
            errors.append("run status is not complete")
        if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            errors.append("run completion marker is not complete")
        if run.attrs.get("stage_selector_eligible") is not False:
            errors.append("run is not selector-ineligible")
        canonical_json_bytes(manifest)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "run_path": str(path),
    }


def _require_unselected(root: Any, *, run_id: str) -> None:
    parent = root["detect_runs"]
    collisions = [name for name in _SELECTOR_ATTRS if parent.attrs.get(name) == run_id]
    if collisions:
        raise RuntimeError(
            f"Sampled training candidate leaked into selectors: {collisions}."
        )


def publish_detection_artifact_run(
    *,
    local_archive: str | Path,
    target_archive: str | Path,
    artifact_run_id: str,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Import one completed node-local artifact without making it authoritative."""

    local = Path(local_archive).expanduser().resolve()
    target = Path(target_archive).expanduser().resolve()
    local_run = _artifact_path(local, artifact_run_id)
    target_run = _artifact_path(target, artifact_run_id)

    def validator(path: Path) -> Mapping[str, Any]:
        try:
            run = zarr.open_group(str(path), mode="r", use_consolidated=False)
            seal = validate_artifact_payload_inventory_seal(run)
            errors: list[str] = []
            if run.attrs.get("stage_selector_eligible") is not False:
                errors.append("artifact is not selector-ineligible")
            if run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
                errors.append("artifact is not complete")
            return {"valid": not errors, "errors": errors, "seal": seal}
        except Exception as exc:
            return {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}

    def complete(_root: Any, _parent: Any, run: Any) -> None:
        if (
            run.attrs.get("stage_selector_eligible") is not False
            or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        ):
            raise RuntimeError("Imported detection artifact lost completion state.")

    def verify(root: Any) -> None:
        parent = root[DETECTION_ARTIFACT_RUN_FAMILY]
        forbidden = [name for name in _SELECTOR_ATTRS if name in parent.attrs]
        if forbidden:
            raise RuntimeError(
                f"Detection artifact parent contains forbidden selectors: {forbidden}."
            )

    return atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=target,
            local_run_path=local_run,
            target_run_path=target_run,
            run_name=str(artifact_run_id),
            lock_suffix="sampled_training_detection_artifact",
            publish_schema_id=SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_ID,
            policy="immutable_selector_free_artifact_import_v1",
            rollback_policy=SAMPLED_TRAINING_DETECTION_ROLLBACK_POLICY,
            content_checksum=True,
            persist_run_receipt=False,
        ),
        copy_backend=copy_backend,
        validate_run=validator,
        prepare_parents=lambda root: (require_detection_artifact_parent(root),),
        complete_run=complete,
        verify_pointers=verify,
        payload_metadata={
            "snapshot_role": "unbound_sampled_training_detection_artifact",
            "selector_activation": "forbidden",
        },
    )


def publish_sampled_training_detection_candidate(
    candidate: SampledTrainingDetectionCandidate,
    *,
    archive: str | Path,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Atomically import one bound sampled review seed without selectors."""

    target = Path(archive).expanduser().resolve()
    target_run = target / "detect_runs" / candidate.run_id

    def validator(path: Path) -> Mapping[str, Any]:
        return validate_sampled_training_detection_run(path, archive=target)

    def prepare(root: Any) -> tuple[Any, ...]:
        return (
            require_runs_parent(
                root,
                "detect_runs",
                completion_epoch=COMPLETION_EPOCH_STRICT,
            ),
        )

    def complete(_root: Any, _parent: Any, run: Any) -> None:
        if (
            run.attrs.get("status") != "complete"
            or run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
            or run.attrs.get("stage_selector_eligible") is not False
        ):
            raise RuntimeError("Sampled detection candidate lost completion state.")

    publication = atomic_publish_run_group(
        AtomicRunPublishSpec(
            source_zarr=target,
            local_run_path=candidate.run_path,
            target_run_path=target_run,
            run_name=candidate.run_id,
            lock_suffix="sampled_training_detection_candidate",
            publish_schema_id=SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_ID,
            policy=SAMPLED_TRAINING_DETECTION_PUBLICATION_POLICY,
            rollback_policy=SAMPLED_TRAINING_DETECTION_ROLLBACK_POLICY,
            content_checksum=True,
            persist_run_receipt=False,
        ),
        copy_backend=copy_backend,
        validate_run=validator,
        prepare_parents=prepare,
        complete_run=complete,
        verify_pointers=lambda root: _require_unselected(root, run_id=candidate.run_id),
        payload_metadata={
            "snapshot_role": "bound_sampled_training_detection_review_seed_v1",
            "source_artifact_path": candidate.manifest["payload"]["source_artifact"][
                "artifact_path"
            ],
            "selector_activation": "deferred",
            "metadata_mode": "direct_mutable_archive",
            "cardinality_statistics": dict(candidate.statistics),
        },
    )
    final = validate_sampled_training_detection_run(target_run, archive=target)
    if not final["valid"]:
        raise RuntimeError(f"Published sampled detection is invalid: {final}")
    return {
        "schema_id": SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_ID,
        "schema_version": SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "published_at_utc": utc_now(),
        "archive": str(target),
        "run_id": candidate.run_id,
        "run_path": f"detect_runs/{candidate.run_id}",
        "source_artifact_path": candidate.manifest["payload"]["source_artifact"][
            "artifact_path"
        ],
        "stage_selector_eligible": False,
        "metadata_mode": "direct_mutable_archive",
        "root_consolidation": "deferred_until_training_publication_is_immutable",
        "cardinality_statistics": dict(candidate.statistics),
        "atomic_publication": publication,
        "validation": final,
    }


def build_and_publish_sampled_training_detection(
    *,
    archive: str | Path,
    artifact_run_id: str,
    scratch_root: str | Path,
    run_id: str,
    copy_backend: str = "python",
) -> dict[str, Any]:
    """Materialize locally, publish atomically, and remove transient storage."""

    scratch = Path(scratch_root).expanduser().resolve()
    if not scratch.is_dir() or str(scratch).startswith(("/groups/", "/nrs/")):
        raise ValueError("scratch_root must be an existing node-local directory.")
    with tempfile.TemporaryDirectory(
        prefix="palette-sampled-detection-binding-",
        dir=str(scratch),
    ) as temporary:
        destination = Path(temporary) / "candidate.zarr"
        candidate = build_sampled_training_detection_candidate(
            archive=archive,
            artifact_run_id=artifact_run_id,
            destination=destination,
            run_id=run_id,
        )
        return publish_sampled_training_detection_candidate(
            candidate,
            archive=archive,
            copy_backend=copy_backend,
        )


__all__ = [
    "SAMPLED_TRAINING_DETECTION_MANIFEST_SCHEMA_ID",
    "SAMPLED_TRAINING_DETECTION_PUBLICATION_SCHEMA_ID",
    "SAMPLED_TRAINING_DETECTION_SCHEMA_ID",
    "SampledTrainingDetectionCandidate",
    "SampledTrainingDetectionDimensions",
    "build_and_publish_sampled_training_detection",
    "build_sampled_training_detection_candidate",
    "publish_detection_artifact_run",
    "publish_sampled_training_detection_candidate",
    "validate_sampled_training_detection_arrays",
    "validate_sampled_training_detection_run",
]

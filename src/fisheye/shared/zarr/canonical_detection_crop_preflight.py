"""Read-only crop-geometry preflight for one canonical-v3 detection run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.shared.observation_coordinate_publication import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.zarr.canonical_detection_manifest import (
    CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    canonical_detection_dimensions_from_manifest,
    require_active_coordinate_canonical_detection,
    validate_canonical_detection_publication,
)
from fisheye.shared.zarr.canonical_detection_shadow import (
    canonical_detection_metadata_declaration_maps,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
    derive_crop_placement_geometry,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.detection_storage import (
    plan_canonical_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr_io import open_zarr_root


CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_ID = (
    "palette.canonical_detection.crop_geometry_preflight"
)
CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_VERSION = 1


def _values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value[...])


def _padding_summary(
    source_crop_xywh: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    instance_key: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, object]:
    crop = np.asarray(source_crop_xywh)
    left = np.maximum(0, -crop[:, 0].astype(np.int64))
    top = np.maximum(0, -crop[:, 1].astype(np.int64))
    right = np.maximum(
        0,
        crop[:, 0].astype(np.int64) + crop[:, 2].astype(np.int64) - int(source_width),
    )
    bottom = np.maximum(
        0,
        crop[:, 1].astype(np.int64) + crop[:, 3].astype(np.int64) - int(source_height),
    )
    padded = np.flatnonzero((left | top | right | bottom) != 0)
    examples = [
        {
            "row_index": int(index),
            "instance_key": int(instance_key[index]),
            "frame_index": int(frame_indices[index]),
            "source_crop_xywh": [int(value) for value in crop[index]],
            "padding_ltrb": [
                int(left[index]),
                int(top[index]),
                int(right[index]),
                int(bottom[index]),
            ],
        }
        for index in padded[:8]
    ]
    return {
        "padded_row_count": int(padded.size),
        "fully_contained_row_count": int(crop.shape[0] - padded.size),
        "max_padding_ltrb": [
            int(left.max(initial=0)),
            int(top.max(initial=0)),
            int(right.max(initial=0)),
            int(bottom.max(initial=0)),
        ],
        "examples": examples,
    }


def inspect_canonical_detection_crop_preflight(
    *,
    analysis_zarr: Path,
    detection_run_id: str,
    policy: CropGeometryPolicy,
    allow_selector_ineligible_candidate: bool = False,
    expected_manifest_digest: str | None = None,
) -> dict[str, object]:
    """Validate one exact canonical-v3 run and derive its crop placement."""

    if type(allow_selector_ineligible_candidate) is not bool:
        raise TypeError("allow_selector_ineligible_candidate must be an exact bool.")
    if policy.size_mode is not CropSizeMode.FIXED_PER_RUN:
        raise ValueError("Canonical crop preflight currently requires fixed_per_run.")
    run_id = str(detection_run_id).strip()
    if not run_id or "/" in run_id:
        raise ValueError("detection_run_id must be one exact child-group name.")
    archive = analysis_zarr.expanduser().resolve()
    root = open_zarr_root(archive, mode="r")
    run_path = f"detect_runs/{run_id}"
    if run_path not in root:
        raise FileNotFoundError(f"Canonical detection run not found: {run_path}")
    run = root[run_path]
    manifest_value = run.attrs.get("run_manifest")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("Canonical detection run lacks run_manifest.")
    manifest = dict(manifest_value)
    if manifest.get("schema_version") != (
        CANONICAL_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("Crop-v2 preflight requires canonical detection manifest v3.")
    dimensions = canonical_detection_dimensions_from_manifest(manifest)
    arrays = {path: run[path] for path in CANONICAL_DETECTION_SCHEMA_V1.binding_paths}
    profile = storage_profile_from_manifest(
        manifest["payload"]["storage_plan"]["storage_profile"]
    )
    plans = plan_canonical_detection_storage(dimensions, profile=profile)
    direct, consolidated = canonical_detection_metadata_declaration_maps(
        archive,
        run_id=run_id,
        plans=plans,
    )
    errors = validate_canonical_detection_publication(
        manifest,
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        arrays=arrays,
    )
    if errors:
        raise ValueError(
            "Canonical detection publication is invalid: " + "; ".join(errors)
        )
    if run.attrs.get("status") != "complete":
        raise ValueError("Canonical detection run is not scientifically complete.")
    if run.attrs.get("palette_run_completion_status") != "complete":
        raise ValueError("Canonical detection lifecycle completion is not complete.")
    eligible = run.attrs.get("stage_selector_eligible")
    if type(eligible) is not bool:
        raise ValueError("Canonical detection selector eligibility is not exact bool.")
    publication = manifest["payload"]["publication"]
    if publication.get("stage_selector_eligible") is not eligible:
        raise ValueError("Canonical manifest and run selector eligibility differ.")
    if not eligible and not allow_selector_ineligible_candidate:
        raise ValueError(
            "Selector-ineligible canonical detection requires an explicit "
            "read-only candidate-preflight allowance."
        )
    if not allow_selector_ineligible_candidate:
        require_active_coordinate_canonical_detection(
            root,
            group_path=run_path,
            expected_manifest_digest=expected_manifest_digest,
        )
    elif expected_manifest_digest is not None and (
        manifest.get("payload_digest") != expected_manifest_digest
    ):
        raise ValueError(
            "Canonical candidate manifest digest differs from the expected digest."
        )

    recording_identity = str(root.attrs.get("recording_id") or "").strip()
    source_evidence = manifest["payload"]["source_evidence"]
    if source_evidence.get("recording_identity") != recording_identity:
        raise ValueError("Canonical detection and archive recording identities differ.")
    _ownership, acquisition = load_persisted_acquisition_camera_authority(root)
    expected_dimensions = {
        "n_frames": int(acquisition.record.source_total_frames),
        "source_width": int(acquisition.record.width_px),
        "source_height": int(acquisition.record.height_px),
    }
    observed_dimensions = {
        "n_frames": dimensions.n_frames,
        "source_width": dimensions.source_width,
        "source_height": dimensions.source_height,
    }
    if observed_dimensions != expected_dimensions:
        raise ValueError(
            "Canonical detection dimensions differ from active acquisition authority."
        )

    centers = _values(arrays["instances/centers_img_xy"])
    bbox_img = _values(arrays["instances/bbox_img_xyxy"])
    instance_key = _values(arrays["instances/instance_key"])
    frame_indices = _values(arrays["instances/frame_indices"])
    roi_width, roi_height = policy.fixed_size_wh or (0, 0)
    roi_sizes = np.repeat(
        np.asarray([roi_width, roi_height], dtype=np.int32).reshape(1, 2),
        dimensions.n_instances,
        axis=0,
    )
    _coordinates, source_crop, _bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        roi_sizes,
    )
    padding = _padding_summary(
        source_crop,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
        instance_key=instance_key,
        frame_indices=frame_indices,
    )
    if (
        padding["padded_row_count"]
        and policy.padding_mode is CropPaddingMode.REQUIRE_FULLY_CONTAINED
    ):
        raise ValueError(
            "Crop policy requires full containment but canonical detections "
            f"produce {padding['padded_row_count']} padded rows."
        )
    return {
        "schema_id": CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_ID,
        "schema_version": CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_VERSION,
        "status": "ready",
        "mode": "read_only",
        "analysis_zarr": str(archive),
        "recording_identity": recording_identity,
        "detection_run_id": run_id,
        "detection_group_path": run_path,
        "detection_manifest_digest": manifest["payload_digest"],
        "detection_logical_content_digest": manifest["payload"]["logical_content"][
            "digest"
        ],
        "detection_selector_eligible": eligible,
        "candidate_preflight_allowance": bool(allow_selector_ineligible_candidate),
        "dimensions": dimensions.as_manifest(),
        "crop_policy": policy.as_manifest(),
        "padding": padding,
        "crop_zarr_writes": False,
        "registry_updated": False,
        "selector_updated": False,
    }


__all__ = [
    "CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_ID",
    "CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_VERSION",
    "inspect_canonical_detection_crop_preflight",
]

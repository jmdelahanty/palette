"""Keypoint-family LSF planners.

Whole-recording and clipped-collection planners remain separate modules.  This
package contains only keypoint-specific bindings and builders shared by those
planners; scheduler mechanics live in :mod:`fisheye.cluster.lsf`.
"""

from fisheye.cluster.keypoints.common import (
    DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    DEFAULT_ZEBRAFISH_MIN_ROI_SIZE,
    FlatRoiCacheBinding,
    KeypointInputCapability,
    KeypointRunNames,
    PoseModelBinding,
    build_keypoint_run_names,
    build_prediction_job,
    build_refinement_job,
    resolve_pose_model_binding,
    resolve_keypoint_storage,
    resolve_keypoint_v2_publication_storage,
    validate_flat_roi_cache_binding,
    validate_keypoint_input_dag,
    validate_registered_geometry_crop_authority,
)

__all__ = [
    "DEFAULT_KEYPOINT_FRAME_SHARD_ROWS",
    "DEFAULT_KEYPOINT_ROI_SHARD_ROWS",
    "DEFAULT_ZEBRAFISH_MIN_ROI_SIZE",
    "FlatRoiCacheBinding",
    "KeypointInputCapability",
    "KeypointRunNames",
    "PoseModelBinding",
    "build_keypoint_run_names",
    "build_prediction_job",
    "build_refinement_job",
    "resolve_pose_model_binding",
    "resolve_keypoint_storage",
    "resolve_keypoint_v2_publication_storage",
    "validate_flat_roi_cache_binding",
    "validate_keypoint_input_dag",
    "validate_registered_geometry_crop_authority",
]

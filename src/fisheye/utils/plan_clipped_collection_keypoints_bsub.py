#!/usr/bin/env python3
"""Compatibility entry point for the family-owned clipped keypoint planner."""

from fisheye.cluster.keypoints.clipped_collection import (
    DEFAULT_GROUPS_REPO,
    DEFAULT_REGISTRY,
    ClipPlan,
    WorkflowPlan,
    _bsub_prefix,
    _collection_clip_ids,
    _default_alias_manifest,
    _discover_cache_manifests,
    _load_manifest,
    _open_root,
    _parse_bsub_job_id,
    _recording_dir_from_zarr,
    _replace_job_placeholders,
    _run_command,
    _sanitize_component,
    _shell_join,
    _utc_run_id,
    apply_plan,
    build_arg_parser,
    build_plan,
    main,
)

__all__ = [
    "DEFAULT_GROUPS_REPO",
    "DEFAULT_REGISTRY",
    "ClipPlan",
    "WorkflowPlan",
    "apply_plan",
    "build_arg_parser",
    "build_plan",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Metadata-only discovery for exact body-frame gaze successors."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    STORAGE_SCHEMA_ID,
    STORAGE_SCHEMA_VERSION,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

from .chaser_exact_gaze_contract import (
    FORBIDDEN_SELECTORS,
    GAZE_TRACKING_PARENT,
    ExactGazeTrackingContractError,
    validate_gaze_scientific_manifest,
)


def _group_names(group: Any) -> tuple[str, ...]:
    try:
        return tuple(str(value) for value in group.group_keys())
    except Exception:
        return tuple(
            str(key)
            for key, value in getattr(group, "items", lambda: ())()
            if hasattr(value, "attrs")
        )


def _valid_manifest(run: Any, *, expected_run_path: str) -> Mapping[str, Any] | None:
    attrs = dict(getattr(run, "attrs", {}))
    manifest = attrs.get(MANIFEST_ATTR)
    digest = str(attrs.get(MANIFEST_DIGEST_ATTR) or "")
    if not isinstance(manifest, Mapping) or len(digest) != 64:
        return None
    try:
        current_digest = canonical_json_sha256(dict(manifest))
    except Exception:
        return None
    if (
        current_digest != digest
        or attrs.get("schema_id") != STORAGE_SCHEMA_ID
        or attrs.get("schema_version") != STORAGE_SCHEMA_VERSION
        or attrs.get("successor_kind") != "chaser_gaze_tracking"
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("successor_kind") != "chaser_gaze_tracking"
        or manifest.get("run_path") != expected_run_path
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def compatible_gaze_tracking_binding(
    root: Any,
    *,
    recording_id: str,
    spatial_sources: Mapping[str, Any],
    keypoint_relative_manifest: Mapping[str, Any],
    keypoint_radial_manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve exactly one gaze child by all source identities and policy."""

    providers = spatial_sources.get("position_providers")
    semantic = spatial_sources.get("protocol_semantic_selection")
    if (
        not isinstance(providers, list)
        or len(providers) != 2
        or not isinstance(providers[0], Mapping)
        or providers[0].get("provider_role") != "keypoint"
        or not isinstance(semantic, Mapping)
    ):
        return None
    expected_relative = providers[0].get("relative_frame")
    radial_scientific = keypoint_radial_manifest.get("scientific_manifest")
    if not isinstance(radial_scientific, Mapping):
        return None
    radial_sources = radial_scientific.get("sources")
    if not isinstance(radial_sources, Mapping):
        return None
    expected_radial = {
        **dict(providers[0].get("radial_near_field", {})),
        "scientific_payload_sha256": keypoint_radial_manifest.get(
            "scientific_payload_sha256"
        ),
        "arena_geometry_and_scale": radial_sources.get("arena_geometry_and_scale"),
        "arena": radial_scientific.get("arena"),
    }
    semantic_digest = semantic.get("manifest_sha256")
    dimensions = keypoint_relative_manifest.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return None
    try:
        n_frames = int(dimensions.get("n_frames", 0))
        n_chasers = int(dimensions.get("n_chasers", 0))
        n_rows = int(dimensions.get("n_rows", 0))
    except (TypeError, ValueError):
        return None
    if n_frames <= 0 or n_chasers <= 0 or n_rows != n_frames * n_chasers:
        return None
    try:
        parent = root[GAZE_TRACKING_PARENT]
    except Exception:
        return None
    if FORBIDDEN_SELECTORS.intersection(getattr(parent, "attrs", {})):
        return None
    matches: list[Mapping[str, Any]] = []
    for run_name in _group_names(parent):
        if run_name in {".", ".."} or run_name.casefold() in FORBIDDEN_SELECTORS:
            continue
        run_path = f"{GAZE_TRACKING_PARENT}/{run_name}"
        manifest = _valid_manifest(parent[run_name], expected_run_path=run_path)
        if manifest is None or manifest.get("recording_id") != recording_id:
            continue
        payload = manifest.get("scientific_payload_sha256")
        try:
            verified = validate_gaze_scientific_manifest(
                manifest.get("scientific_manifest"),
                expected_scientific_payload_sha256=payload,
                expected_n_frames=n_frames,
                expected_n_chasers=n_chasers,
                expected_relative_binding=expected_relative,
                expected_semantic_manifest_sha256=semantic_digest,
                expected_radial_binding=expected_radial,
            )
        except ExactGazeTrackingContractError:
            continue
        matches.append(
            {
                "run_path": run_path,
                "manifest_sha256": canonical_json_sha256(dict(manifest)),
                "scientific_payload_sha256": payload,
                "source_relative_frame": verified["source_relative_frame"],
                "source_eye_orientation": verified["source_eye_orientation"],
                "source_radial_geometry": verified["source_radial_geometry"],
                "semantic_selection_manifest_sha256": verified[
                    "semantic_selection_manifest_sha256"
                ],
                "parameters": verified["parameters"],
            }
        )
    return matches[0] if len(matches) == 1 else None


__all__ = ["compatible_gaze_tracking_binding"]

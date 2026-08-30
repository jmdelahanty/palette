"""Metadata-only discovery for exact body-alignment-by-distance successors."""

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

from .chaser_exact_body_alignment_contract import (
    BODY_ALIGNMENT_PARENT,
    FORBIDDEN_SELECTORS,
    ExactBodyAlignmentContractError,
    validate_body_alignment_scientific_manifest,
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
        or attrs.get("successor_kind") != "chaser_body_alignment_by_distance"
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("successor_kind") != "chaser_body_alignment_by_distance"
        or manifest.get("run_path") != expected_run_path
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def compatible_body_alignment_binding(
    root: Any,
    *,
    recording_id: str,
    spatial_sources: Mapping[str, Any],
    spatial_epoch_records: Any,
    keypoint_relative_manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve exactly one alignment child by immutable source identity."""

    providers = spatial_sources.get("position_providers")
    semantic = spatial_sources.get("protocol_semantic_selection")
    if (
        not isinstance(providers, (tuple, list))
        or len(providers) != 2
        or not isinstance(providers[0], Mapping)
        or providers[0].get("provider_role") != "keypoint"
        or not isinstance(semantic, Mapping)
    ):
        return None
    expected_relative = providers[0].get("relative_frame")
    dimensions = keypoint_relative_manifest.get("dimensions")
    authorities = keypoint_relative_manifest.get("source_authorities")
    scale = keypoint_relative_manifest.get("scale_policy")
    if (
        not isinstance(dimensions, Mapping)
        or not isinstance(authorities, Mapping)
        or not isinstance(authorities.get("fish_position"), Mapping)
        or not isinstance(authorities.get("body_frame"), Mapping)
        or not isinstance(scale, Mapping)
    ):
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
        parent = root[BODY_ALIGNMENT_PARENT]
    except Exception:
        return None
    if FORBIDDEN_SELECTORS.intersection(getattr(parent, "attrs", {})):
        return None

    matches: list[Mapping[str, Any]] = []
    for run_name in _group_names(parent):
        if run_name in {".", ".."} or run_name.casefold() in FORBIDDEN_SELECTORS:
            continue
        run_path = f"{BODY_ALIGNMENT_PARENT}/{run_name}"
        manifest = _valid_manifest(parent[run_name], expected_run_path=run_path)
        if manifest is None or manifest.get("recording_id") != recording_id:
            continue
        payload = manifest.get("scientific_payload_sha256")
        try:
            verified = validate_body_alignment_scientific_manifest(
                manifest.get("scientific_manifest"),
                expected_scientific_payload_sha256=payload,
                expected_n_frames=n_frames,
                expected_n_chasers=n_chasers,
                expected_relative_binding=expected_relative,
                expected_semantic_binding=semantic,
                expected_fish_position_authority=authorities["fish_position"],
                expected_body_frame_authority=authorities["body_frame"],
                expected_scale_policy=scale,
                expected_epoch_records=spatial_epoch_records,
            )
        except ExactBodyAlignmentContractError:
            continue
        matches.append(
            {
                "run_path": run_path,
                "manifest_sha256": canonical_json_sha256(dict(manifest)),
                "scientific_payload_sha256": payload,
                **verified,
            }
        )
    return matches[0] if len(matches) == 1 else None


__all__ = ["compatible_body_alignment_binding"]

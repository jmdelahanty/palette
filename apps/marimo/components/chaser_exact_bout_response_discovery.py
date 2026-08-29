"""Metadata-only discovery for exact generalized bout-response successors."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.analysis_workflows.composable_chaser_successor_publication import (
    MANIFEST_ATTR,
    MANIFEST_DIGEST_ATTR,
    STORAGE_SCHEMA_ID,
    STORAGE_SCHEMA_VERSION,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    validate_exact_relative_frame_binding,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

from .chaser_exact_bout_response_contract import (
    BOUT_RESPONSE_PARENT,
    ExactBoutResponseContractError,
    validate_scientific_manifest,
)

FORBIDDEN_SELECTORS = frozenset(
    {
        "latest",
        "latest_complete",
        "latest_pending",
        "current",
        "current_run",
        "selected",
        "authoritative",
        "authoritative_run",
        "default",
    }
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


def _valid_manifest(
    run: Any,
    *,
    expected_run_path: str,
) -> Mapping[str, Any] | None:
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
        or attrs.get("successor_kind") != "generalized_chaser_bout_response"
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("successor_kind") != "generalized_chaser_bout_response"
        or manifest.get("run_path") != expected_run_path
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def compatible_generalized_bout_response_binding(
    root: Any,
    *,
    recording_id: str,
    spatial_sources: Mapping[str, Any],
    keypoint_relative_manifest: Mapping[str, Any],
    controller_trial_binding: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Resolve one persisted bout-response child by all sealed source identities."""

    if controller_trial_binding is None:
        return None
    providers = spatial_sources.get("position_providers")
    if not isinstance(providers, list) or len(providers) != 2:
        return None
    keypoint = providers[0]
    if not isinstance(keypoint, Mapping) or keypoint.get("provider_role") != "keypoint":
        return None
    try:
        expected_relative = validate_exact_relative_frame_binding(
            keypoint.get("relative_frame"),
            label="spatial keypoint relative-frame binding",
        )
    except ExactRelativeFrameBindingError:
        return None
    semantic = spatial_sources.get("protocol_semantic_selection")
    if not isinstance(semantic, Mapping):
        return None
    semantic_digest = semantic.get("manifest_sha256")
    controller_payload = controller_trial_binding.get("scientific_payload_sha256")
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
        parent = root[BOUT_RESPONSE_PARENT]
    except Exception:
        return None
    if FORBIDDEN_SELECTORS.intersection(getattr(parent, "attrs", {})):
        return None

    matches: list[Mapping[str, Any]] = []
    for run_name in _group_names(parent):
        if run_name in {".", ".."} or run_name.casefold() in FORBIDDEN_SELECTORS:
            continue
        run_path = f"{BOUT_RESPONSE_PARENT}/{run_name}"
        manifest = _valid_manifest(parent[run_name], expected_run_path=run_path)
        if manifest is None or manifest.get("recording_id") != recording_id:
            continue
        scientific_payload = manifest.get("scientific_payload_sha256")
        try:
            verified = validate_scientific_manifest(
                manifest.get("scientific_manifest"),
                expected_scientific_payload_sha256=scientific_payload,
                expected_n_frames=n_frames,
                expected_n_chasers=n_chasers,
                expected_relative_binding=expected_relative.normalized_identity,
                expected_semantic_manifest_sha256=semantic_digest,
                expected_controller_payload_sha256=controller_payload,
            )
        except ExactBoutResponseContractError:
            continue
        matches.append(
            {
                "run_path": run_path,
                "manifest_sha256": canonical_json_sha256(dict(manifest)),
                "scientific_payload_sha256": scientific_payload,
                "source_relative_frame": verified["source_relative_frame"],
                "source_motion": verified["source_motion"],
                "source_swim_bouts": verified["source_swim_bouts"],
                "semantic_selection_manifest_sha256": verified[
                    "semantic_selection_manifest_sha256"
                ],
                "controller_trial_payload_sha256": verified[
                    "controller_trial_payload_sha256"
                ],
                "body_extension_present": verified["body_extension_present"],
            }
        )
    return matches[0] if len(matches) == 1 else None


__all__ = ["compatible_generalized_bout_response_binding"]

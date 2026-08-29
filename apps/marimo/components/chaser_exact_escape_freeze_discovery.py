"""Metadata-only discovery for exact escape/freeze successors."""

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

from .chaser_exact_escape_freeze_contract import (
    ESCAPE_FREEZE_PARENT,
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
        or attrs.get("successor_kind") != "chaser_escape_freeze"
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get("stage_selector_eligible") is not False
        or manifest.get("successor_kind") != "chaser_escape_freeze"
        or manifest.get("run_path") != expected_run_path
        or manifest.get("selector_eligible") is not False
        or manifest.get("selection") != "none"
    ):
        return None
    return manifest


def compatible_escape_freeze_binding(
    root: Any,
    *,
    recording_id: str,
    controller_trial_binding: Mapping[str, Any] | None,
    bout_response_binding: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Resolve one escape/freeze child by exact dependency payloads and motion."""

    if controller_trial_binding is None or bout_response_binding is None:
        return None
    controller_payload = controller_trial_binding.get("scientific_payload_sha256")
    bout_payload = bout_response_binding.get("scientific_payload_sha256")
    bout_motion = bout_response_binding.get("source_motion")
    if not isinstance(bout_motion, Mapping):
        return None
    try:
        parent = root[ESCAPE_FREEZE_PARENT]
    except Exception:
        return None
    if FORBIDDEN_SELECTORS.intersection(getattr(parent, "attrs", {})):
        return None

    matches: list[Mapping[str, Any]] = []
    for run_name in _group_names(parent):
        if run_name in {".", ".."} or run_name.casefold() in FORBIDDEN_SELECTORS:
            continue
        run_path = f"{ESCAPE_FREEZE_PARENT}/{run_name}"
        manifest = _valid_manifest(parent[run_name], expected_run_path=run_path)
        if manifest is None or manifest.get("recording_id") != recording_id:
            continue
        scientific_payload = manifest.get("scientific_payload_sha256")
        try:
            verified = validate_scientific_manifest(
                manifest.get("scientific_manifest"),
                expected_scientific_payload_sha256=scientific_payload,
                expected_controller_payload_sha256=controller_payload,
                expected_bout_response_payload_sha256=bout_payload,
                expected_bout_motion=bout_motion,
            )
        except (TypeError, ValueError):
            continue
        matches.append(
            {
                "run_path": run_path,
                "manifest_sha256": canonical_json_sha256(dict(manifest)),
                "scientific_payload_sha256": scientific_payload,
                "source_motion": verified["source_motion"],
                "controller_trial_payload_sha256": verified[
                    "controller_trial_payload_sha256"
                ],
                "bout_response_payload_sha256": verified[
                    "bout_response_payload_sha256"
                ],
                "classifier_parameters": verified["classifier_parameters"],
                "n_trials": verified["n_trials"],
                "n_events": verified["n_events"],
                "n_sweep_rows": verified["n_sweep_rows"],
            }
        )
    return matches[0] if len(matches) == 1 else None


__all__ = ["compatible_escape_freeze_binding"]

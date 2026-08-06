"""Fail-closed lifecycle and mutation guard for refined subject-mask runs.

The canonical coordinate schema is independent of publication state. Explicit
editable drafts may use that schema while review mutates their dense mask
authority. Only approved sealed snapshots are immutable and selector-visible.
Mutation callers must resolve the target from the archive root immediately
before their first write because another publisher may have sealed it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from fisheye.shared.coordinate_reference import canonical_node_path


REFINED_SUBJECT_MASK_CANONICAL_CONTRACT = "canonical_v2"
REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR = (
    "refined_subject_mask_publication_owner"
)
REFINED_SUBJECT_MASK_LIFECYCLE_ATTR = "refined_subject_mask_lifecycle"
REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_ID = (
    "palette.refined_subject_mask_lifecycle"
)
REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_VERSION = 1
REFINED_SUBJECT_MASK_EDITABLE_DRAFT = "editable_draft"
REFINED_SUBJECT_MASK_SEALED_SNAPSHOT = "sealed_snapshot"
_LIFECYCLE_FIELDS = {"schema_id", "schema_version", "state"}


class RefinedSubjectMaskMutationError(RuntimeError):
    """Raised when a caller attempts to mutate an immutable refined run."""


def _lifecycle_record(state: str) -> dict[str, object]:
    if state not in {
        REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
        REFINED_SUBJECT_MASK_SEALED_SNAPSHOT,
    }:
        raise RefinedSubjectMaskMutationError(
            f"Unsupported refined subject-mask lifecycle state {state!r}."
        )
    return {
        "schema_id": REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_VERSION,
        "state": state,
    }


def refined_subject_mask_lifecycle_state(run_group: Any) -> str:
    """Return the exact lifecycle, preserving fail-closed legacy behavior."""

    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask lifecycle target has no attrs mapping."
        )
    value = attrs.get(REFINED_SUBJECT_MASK_LIFECYCLE_ATTR)
    if value is None:
        if (
            attrs.get("coordinate_contract")
            == REFINED_SUBJECT_MASK_CANONICAL_CONTRACT
            or REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR in attrs
        ):
            return REFINED_SUBJECT_MASK_SEALED_SNAPSHOT
        return REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    if not isinstance(value, Mapping) or set(value) != _LIFECYCLE_FIELDS:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask lifecycle must be one exact versioned mapping."
        )
    record = copy.deepcopy(dict(value))
    state = record.get("state")
    if record != _lifecycle_record(str(state)):
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask lifecycle differs from its canonical form."
        )
    return str(state)


def stamp_refined_subject_mask_editable_draft(run_group: Any) -> Any:
    """Declare one canonical-schema run editable and selector-ineligible."""

    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask editable draft has no attrs mapping."
        )
    if attrs.get("stage_selector_eligible") is not False:
        raise RefinedSubjectMaskMutationError(
            "Editable refined subject-mask drafts must be selector-ineligible."
        )
    attrs[REFINED_SUBJECT_MASK_LIFECYCLE_ATTR] = _lifecycle_record(
        REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    )
    if (
        attrs.get(REFINED_SUBJECT_MASK_LIFECYCLE_ATTR)
        != _lifecycle_record(REFINED_SUBJECT_MASK_EDITABLE_DRAFT)
    ):
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask editable lifecycle did not persist exactly."
        )
    return run_group


def require_approved_refined_subject_mask_review(run_group: Any) -> None:
    """Require exact run-level and per-component approval before sealing."""

    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask approval target has no attrs mapping."
        )
    labels = attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)) or not labels:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing requires nonempty mask_labels."
        )
    component_names = tuple(str(item) for item in labels)
    if len(component_names) != len(set(component_names)):
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing requires unique component labels."
        )
    statuses = attrs.get("component_review_statuses")
    if not isinstance(statuses, Mapping) or set(statuses) != set(component_names):
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing requires one review status per component."
        )
    for component_name in component_names:
        status = statuses.get(component_name)
        if not isinstance(status, Mapping) or status.get("state") != "approved":
            raise RefinedSubjectMaskMutationError(
                "Refined subject-mask sealing requires every component review to "
                f"be approved; {component_name!r} is not approved."
            )
    run_status = attrs.get("refined_subject_mask_review_status")
    if not isinstance(run_status, Mapping) or run_status.get("state") != "approved":
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing requires approved aggregate review state."
        )


def stamp_refined_subject_mask_sealed_snapshot(run_group: Any) -> Any:
    """Seal one approved, selector-ineligible refined snapshot."""

    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing target has no attrs mapping."
        )
    if attrs.get("stage_selector_eligible") is not False:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealing must precede selector activation."
        )
    require_approved_refined_subject_mask_review(run_group)
    attrs[REFINED_SUBJECT_MASK_LIFECYCLE_ATTR] = _lifecycle_record(
        REFINED_SUBJECT_MASK_SEALED_SNAPSHOT
    )
    if (
        attrs.get(REFINED_SUBJECT_MASK_LIFECYCLE_ATTR)
        != _lifecycle_record(REFINED_SUBJECT_MASK_SEALED_SNAPSHOT)
    ):
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask sealed lifecycle did not persist exactly."
        )
    return run_group


def migrate_unapproved_refined_subject_mask_to_editable_draft(
    root: Any,
    run_name: str,
) -> dict[str, object]:
    """Demote one mistakenly activated, unapproved canonical run in place.

    The selector-eligibility bit is cleared before any other mutation, so an
    interrupted migration fails safe. Scientific arrays are never rewritten.
    """

    name = str(run_name).strip()
    if not name or "/" in name or name in {".", ".."}:
        raise RefinedSubjectMaskMutationError(
            f"Invalid refined subject-mask run name {run_name!r}."
        )
    try:
        parent = root["refined_subject_masks_runs"]
        run_group = parent[name]
    except Exception as exc:
        raise RefinedSubjectMaskMutationError(
            f"Refined subject-mask migration target {name!r} is unavailable: {exc}."
        ) from exc
    attrs = getattr(run_group, "attrs", None)
    parent_attrs = getattr(parent, "attrs", None)
    if attrs is None or parent_attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask migration target lacks attrs mappings."
        )
    if attrs.get("coordinate_contract") != REFINED_SUBJECT_MASK_CANONICAL_CONTRACT:
        raise RefinedSubjectMaskMutationError(
            "Only canonical_v2 refined subject-mask runs may use this migration."
        )
    if REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR not in attrs:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask migration requires the original publication owner."
        )
    if type(attrs.get("stage_selector_eligible")) is not bool:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask migration requires exact boolean selector eligibility."
        )
    run_review = attrs.get("refined_subject_mask_review_status")
    if isinstance(run_review, Mapping) and run_review.get("state") == "approved":
        raise RefinedSubjectMaskMutationError(
            "Approved refined subject-mask runs cannot be demoted to editable drafts."
        )
    for selector_name in (
        "authoritative_run",
        "approved_run",
        "selected_run",
    ):
        if parent_attrs.get(selector_name) == name:
            raise RefinedSubjectMaskMutationError(
                f"Authoritative selector {selector_name!r} prevents draft migration."
            )
    pending = parent_attrs.get("latest_pending")
    if pending is not None and str(pending) != name:
        raise RefinedSubjectMaskMutationError(
            "Another refined subject-mask run already owns latest_pending."
        )

    already_editable = (
        REFINED_SUBJECT_MASK_LIFECYCLE_ATTR in attrs
        and refined_subject_mask_lifecycle_state(run_group)
        == REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    )
    attrs["stage_selector_eligible"] = False
    stamp_refined_subject_mask_editable_draft(run_group)
    removed_selectors: list[str] = []
    for selector_name in ("latest", "latest_complete"):
        if parent_attrs.get(selector_name) == name:
            del parent_attrs[selector_name]
            removed_selectors.append(selector_name)
    parent_attrs["latest_pending"] = name
    parent_attrs["refined_subject_mask_review_status_latest"] = name
    return {
        "run": name,
        "changed": not already_editable or bool(removed_selectors),
        "lifecycle_state": REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
        "stage_selector_eligible": False,
        "removed_selectors": removed_selectors,
        "latest_pending": name,
        "scientific_arrays_rewritten": False,
    }


def require_mutable_refined_subject_mask_group(
    run_group: Any,
    *,
    run_name: str | None = None,
) -> Any:
    """Return ``run_group`` only when it is not a canonical publication."""

    attrs = getattr(run_group, "attrs", None)
    if attrs is None:
        raise RefinedSubjectMaskMutationError(
            "Refined subject-mask mutation target has no attrs mapping."
        )
    path = canonical_node_path(run_group)
    label = str(run_name or path or "<unknown>")
    state = refined_subject_mask_lifecycle_state(run_group)
    if state == REFINED_SUBJECT_MASK_SEALED_SNAPSHOT:
        raise RefinedSubjectMaskMutationError(
            f"Refined subject-mask run {label!r} is an immutable canonical "
            "publication; create a new derived run instead of mutating it."
        )
    if (
        REFINED_SUBJECT_MASK_LIFECYCLE_ATTR in attrs
        and attrs.get("stage_selector_eligible") is not False
    ):
        raise RefinedSubjectMaskMutationError(
            f"Editable refined subject-mask run {label!r} must remain "
            "selector-ineligible."
        )
    return run_group


def resolve_mutable_refined_subject_mask_run(root: Any, run_name: str) -> Any:
    """Freshly resolve and authorize one mutable refined run.

    This function deliberately ignores any previously cached child handle.
    """

    name = str(run_name).strip()
    if not name or "/" in name or name in {".", ".."}:
        raise RefinedSubjectMaskMutationError(
            f"Invalid refined subject-mask run name {run_name!r}."
        )
    path = f"refined_subject_masks_runs/{name}"
    try:
        run_group = root[path]
    except Exception as exc:
        raise RefinedSubjectMaskMutationError(
            f"Refined subject-mask run {name!r} is unavailable: {exc}."
        ) from exc
    if canonical_node_path(run_group) != path:
        raise RefinedSubjectMaskMutationError(
            f"Refined subject-mask run {name!r} resolved to an unexpected path."
        )
    return require_mutable_refined_subject_mask_group(run_group, run_name=name)


__all__ = [
    "REFINED_SUBJECT_MASK_CANONICAL_CONTRACT",
    "REFINED_SUBJECT_MASK_EDITABLE_DRAFT",
    "REFINED_SUBJECT_MASK_LIFECYCLE_ATTR",
    "REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_ID",
    "REFINED_SUBJECT_MASK_LIFECYCLE_SCHEMA_VERSION",
    "REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR",
    "REFINED_SUBJECT_MASK_SEALED_SNAPSHOT",
    "RefinedSubjectMaskMutationError",
    "refined_subject_mask_lifecycle_state",
    "migrate_unapproved_refined_subject_mask_to_editable_draft",
    "require_approved_refined_subject_mask_review",
    "require_mutable_refined_subject_mask_group",
    "resolve_mutable_refined_subject_mask_run",
    "stamp_refined_subject_mask_editable_draft",
    "stamp_refined_subject_mask_sealed_snapshot",
]

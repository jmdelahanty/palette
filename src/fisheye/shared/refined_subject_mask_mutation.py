"""Fail-closed mutation guard for refined subject-mask run groups.

Canonical refined publications are immutable scientific artifacts.  Mutation
callers must resolve the target from the archive root immediately before their
first write; accepting a cached group handle is insufficient because another
publisher may have sealed the run since that handle was obtained.
"""

from __future__ import annotations

from typing import Any

from fisheye.shared.coordinate_reference import canonical_node_path


REFINED_SUBJECT_MASK_CANONICAL_CONTRACT = "canonical_v2"
REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR = (
    "refined_subject_mask_publication_owner"
)


class RefinedSubjectMaskMutationError(RuntimeError):
    """Raised when a caller attempts to mutate an immutable refined run."""


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
    if (
        attrs.get("coordinate_contract") == REFINED_SUBJECT_MASK_CANONICAL_CONTRACT
        or REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR in attrs
    ):
        raise RefinedSubjectMaskMutationError(
            f"Refined subject-mask run {label!r} is an immutable canonical "
            "publication; create a new derived run instead of mutating it."
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
    "REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR",
    "RefinedSubjectMaskMutationError",
    "require_mutable_refined_subject_mask_group",
    "resolve_mutable_refined_subject_mask_run",
]

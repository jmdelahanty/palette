"""Derived effects owed by a committed subject-mask Apply.

After the pixel write and SQLite finalization, a mask Apply owes three
derived effects: the run-wide QC refresh, the tail successor publication, and
the registry refresh.  Its receipt stays ``secondary_effects_state='pending'``
until all three succeed.  This module is the single implementation of that
sequence; the idempotent same-``apply_id`` retry, the fresh Apply, and the
background worker all call it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping


def registry_scope_from_row(row: Mapping[str, object]) -> dict[str, object]:
    """Registry scope from a session or task row (both join the task columns)."""

    scope = row.get("scope")
    return {
        "scope": scope if isinstance(scope, Mapping) else {},
        "dataset_id": str(row.get("dataset_id") or "") or None,
        "zarr_use": str(row.get("zarr_use") or "") or None,
    }


def reopen_mask_run(runtime):
    """Reopen the archive root and refined run; call under the run write lock."""

    from fisheye.tune import refined_subject_mask_review as review_mod

    fresh_root = review_mod.open_zarr_root(runtime.zarr_path, mode="a")
    runtime.root = fresh_root
    runtime.refined = review_mod._open_existing_refined_subject_run(
        fresh_root, runtime.refined.run_name,
    )
    return fresh_root


def run_apply_effects_locked(
    *,
    store,
    runtime,
    root,
    apply_id: str,
    expected_revision: int,
    refresh_registry: Callable[..., bool],
    registry_scope: Mapping[str, object],
    user: str,
    after_derived: Callable[[dict[str, object]], None] | None = None,
    before_complete: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run QC, tail successor, and registry effects, then mark them complete.

    The caller holds the refined-run write lock and has verified run
    ownership.  ``root`` is the archive root opened under that lock.  The
    optional hooks let a caller record its own audit event at the same point
    in the sequence as before this extraction.
    """

    from fisheye.tune import refined_subject_mask_review as review_mod
    from .web_mask_tail_refresh import refresh_training_tail_after_mask_apply
    from .web_subject_mask_apply_qc import refresh_subject_mask_apply_qc_locked

    revision = int(expected_revision)
    derived: dict[str, object] = dict(refresh_subject_mask_apply_qc_locked(
        root=root,
        refined_run=runtime.refined.run_name,
        expected_edit_revision=revision,
    ))
    runtime.refined = review_mod._open_existing_refined_subject_run(root, runtime.refined.run_name)
    derived.update(refresh_training_tail_after_mask_apply(
        store=store, runtime=runtime, apply_id=apply_id,
        expected_mask_revision=revision,
    ))
    if after_derived is not None:
        after_derived(derived)
    if not refresh_registry(
        store=store,
        task_id=runtime.task_id,
        recording_id=runtime.recording_id,
        user=user,
        workflow_kind="subject_mask_component",
        scope=registry_scope.get("scope") or {},
        zarr_path=runtime.zarr_path,
        dataset_id=registry_scope.get("dataset_id"),
        zarr_use=registry_scope.get("zarr_use"),
    ):
        raise RuntimeError("Subject-mask registry refresh remains pending.")
    if before_complete is not None:
        before_complete(derived)
    effects_complete = store.mark_session_checkpoint_apply_effects_complete(
        task_id=runtime.task_id,
        component_name=runtime.component_name,
        apply_id=apply_id,
    )
    if not effects_complete:
        raise RuntimeError("Subject-mask Apply effects receipt remains pending.")
    return derived

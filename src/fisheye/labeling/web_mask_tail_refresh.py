"""Offer mask-derived training successors without replacing active review tasks."""

from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.labeling.assignment_store import TASK_SUPERSEDED_STATE
from fisheye.labeling.tail_successor_lineage import (
    KEYPOINT_FAMILY,
    checkpoint_edit_times,
    MASK_FAMILY,
    lineage_tasks,
    read_family_lineage,
    stranded_keypoint_rows,
    task_run,
)
from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.shared.recovered_training_review_contract import (
    NATIVE_REVIEW_SCHEMA,
    REVIEW_SCHEMA,
)
from fisheye.labeling.web_subject_mask_apply_state import (
    TAIL_SUCCESSOR_EVENT,
    tail_successor_event,
    tail_successor_offer,
)


def refresh_training_tail_after_mask_apply(
    *,
    store,
    runtime,
    apply_id: str,
    expected_mask_revision: int,
) -> dict[str, object]:
    """Refresh an exact paired tail dataset after applied body/swim mask edits.

    The caller holds the mask run lock. The original tasks and sessions retain
    their identity; the response offers separately versioned successor tasks.
    Unapplied pose checkpoints must finish before their manual labels are copied.
    """
    mask = runtime.refined.group
    if mask.attrs.get("schema_id") not in (REVIEW_SCHEMA, NATIVE_REVIEW_SCHEMA):
        return {"tail_refresh_status": "not_applicable"}
    if runtime.component_name not in {"subject_body", "swim_bladder"}:
        return {"tail_refresh_status": "not_applicable"}
    from fisheye.training.mask_tail_apply_refresh import (
        regenerate_training_tail_version,
        validate_completed_tail_version,
    )

    completed = tail_successor_event(
        store,
        runtime,
        apply_id=apply_id,
        expected_mask_revision=expected_mask_revision,
    )
    if completed is not None:
        after = completed["after"]
        paths = validate_completed_tail_version(
            archive=runtime.zarr_path,
            version=after["version"],
            source_bindings=after["source_bindings"],
        )
        for offered in after["tasks"]:
            task = store.get_task(offered["task_id"])
            key = (
                "pose_edit" if offered["workflow_kind"] == "keypoints" else "mask_edit"
            )
            if (
                task is None
                or task.get("run_name") != paths[key].split("/")[1]
                or task.get("scope", {}).get("source_mask_apply_id") != str(apply_id)
            ):
                raise RuntimeError(
                    "Completed tail successor task has a conflicting binding"
                )
        return tail_successor_offer(
            store,
            runtime,
            apply_id=apply_id,
            expected_mask_revision=expected_mask_revision,
        )
    parent = runtime.root.get("refined_keypoints_runs")
    candidates = []
    if parent is not None:
        for name, pose in parent.groups():
            if (
                pose.attrs.get("schema_id") == mask.attrs.get("schema_id")
                and keypoint_source_crop_run_from_attributes(pose.attrs)
                == keypoint_source_crop_run_from_attributes(mask.attrs)
                and pose.attrs.get("source_bindings")
                == mask.attrs.get("source_bindings")
                and pose.attrs.get("source_subject_mask_run")
                == mask.attrs.get("source_subject_mask_run")
            ):
                candidates.append(name)
    if len(candidates) != 1:
        raise RuntimeError(
            "Tail refresh requires exactly one bound keypoint review run; source selection is ambiguous or missing."
        )
    pose_name = candidates[0]
    archive = Path(runtime.zarr_path).resolve()
    replaced, source_tasks = _replaced_tasks(
        store, runtime, archive=archive, pose_name=pose_name
    )
    _refuse_unfinished_work(store, replaced)
    _refuse_stranded_keypoints(store, runtime, archive=archive, pose_name=pose_name)
    # Close the older versions before the snapshot so nothing new can be
    # applied to them afterwards; claims made earlier are visible as applying.
    previous = store.supersede_tasks(
        task_ids=[str(task["task_id"]) for task in replaced],
        user=runtime.user,
        reason="replaced_by_tail_successor",
        details={"source_mask_apply_id": str(apply_id), "source_task_id": runtime.task_id},
    )
    try:
        _refuse_unfinished_work(store, replaced)
        result = regenerate_training_tail_version(
            archive=archive,
            refined_mask_run=runtime.refined.run_name,
            refined_keypoint_run=pose_name,
            apply_id=apply_id,
            expected_mask_revision=expected_mask_revision,
        )
    except BaseException:
        store.restore_superseded_tasks(
            previous_states=previous,
            user=runtime.user,
            reason="tail_successor_not_published",
        )
        raise
    tasks = []
    for task in result["tasks"]:
        task = dict(task)
        task["scope"] = {
            **task["scope"],
            "source_mask_apply_id": str(apply_id),
            "source_keypoint_task_ids": source_tasks,
            "source_keypoint_run": pose_name,
        }
        if task["workflow_kind"] == "keypoints":
            task["title"] = "Review refreshed tail labels; saved manual points retained"
            task["notes"] = (
                "Automatic mask-derived points regenerated in a new version. Saved manual landmarks, including manual tail points and clears, are retained. Earlier tasks and browser sessions are unchanged."
            )
        # A retry must not reset a successor which the reviewer already opened.
        existing = store.get_task(task["task_id"])
        if existing is None:
            store.upsert_task(**task, actor_user=runtime.user)
        elif (
            existing.get("recording_id") != task["recording_id"]
            or existing.get("workflow_kind") != task["workflow_kind"]
            or existing.get("run_name") != task["run_name"]
            or existing.get("scope") != task["scope"]
        ):
            raise RuntimeError(
                "Existing tail successor task has a conflicting source binding"
            )
        tasks.append(
            {
                "task_id": task["task_id"],
                "workflow_kind": task["workflow_kind"],
                "title": task["title"],
            }
        )
    offer = {
        "tail_refresh_status": "complete",
        "tail_refresh_version": result["version"],
        "tail_refresh_tasks": tasks,
        "tail_refresh_failures": result["failures"],
        "tail_refresh_valid_rows": result["tail_valid_count"],
        "tail_refresh_training_eligible_rows": result["training_eligible_count"],
        "tail_refresh_manual_point_count": result["manual_point_count"],
        "tail_refresh_mask_revision": int(expected_mask_revision),
    }
    if "visible_endpoint_rows" in result:
        offer["tail_refresh_visible_endpoint_rows"] = result["visible_endpoint_rows"]
    store.record_event(
        task_id=runtime.task_id,
        recording_id=runtime.recording_id,
        user=runtime.user,
        event_type=TAIL_SUCCESSOR_EVENT,
        target={"apply_id": str(apply_id)},
        after={
            "version": result["version"],
            "paths": result["paths"],
            "source_bindings": result["source_bindings"],
            "tasks": tasks,
            "manual_point_count": result["manual_point_count"],
            "training_eligible_count": result["training_eligible_count"],
            "failures": result["failures"],
            "tail_refresh": offer,
            "superseded_task_ids": sorted(previous),
        },
    )
    # A review completed while this effect was pending retires its successors now.
    store.retire_untouched_review_successors(
        source_task_id=runtime.task_id, user=runtime.user
    )
    return offer


def _replaced_tasks(store, runtime, *, archive: Path, pose_name: str):
    """Open tasks on older versions in this lineage, and the paired keypoint tasks.

    Mask tasks on the applied mask run itself (the task being applied and any
    other component's task on that run) stay open: the run is still editable,
    and its next Apply snapshots every component again, so nothing there can
    be stranded. Its review can also still be completed.
    """

    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    tasks = store.list_tasks(recording_id=runtime.recording_id)
    keypoints = read_family_lineage(root, KEYPOINT_FAMILY)
    masks = read_family_lineage(root, MASK_FAMILY)
    keypoint_tasks = lineage_tasks(
        tasks, archive=archive, family=KEYPOINT_FAMILY,
        component=keypoints.component(pose_name),
    )
    mask_tasks = lineage_tasks(
        tasks, archive=archive, family=MASK_FAMILY,
        component=masks.component(runtime.refined.run_name),
    )
    source_tasks = [
        str(task["task_id"]) for task in keypoint_tasks if task_run(task) == pose_name
    ]
    replaced = [
        task
        for task in (
            *keypoint_tasks,
            *(t for t in mask_tasks if task_run(t) != runtime.refined.run_name),
        )
        if str(task["task_id"]) != str(runtime.task_id)
        and str(task.get("state") or "") not in ("complete", TASK_SUPERSEDED_STATE)
    ]
    return replaced, source_tasks


def _refuse_unfinished_work(store, tasks) -> None:
    for task in tasks:
        task_id = str(task["task_id"])
        if store.count_unapplied_session_checkpoints(
            task_id=task_id
        ) or store.count_pending_session_checkpoint_apply_effects(task_id=task_id):
            kind = "keypoint" if task.get("workflow_kind") == "keypoints" else "mask"
            raise RuntimeError(
                f"Apply the saved {kind} edits in task {task_id} before retrying mask Apply; "
                "that task is replaced by the new version and its edits are waiting to be copied."
            )


def _refuse_stranded_keypoints(store, runtime, *, archive: Path, pose_name: str) -> None:
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    checkpoints = store.list_recording_applied_checkpoints(
        recording_id=runtime.recording_id, workflow_kind="keypoints"
    )
    stranded = [
        row
        for row in stranded_keypoint_rows(
            root=root,
            checkpoints=checkpoints,
            lineage=read_family_lineage(root, KEYPOINT_FAMILY),
            target_run=pose_name,
            edit_times=checkpoint_edit_times(
                checkpoints,
                store.list_recording_events(
                    recording_id=runtime.recording_id, event_type="checkpoint_keypoints"
                ),
            ),
        )
        if row.disposition in ("carry", "not_carryable")
    ]
    if stranded:
        runs = sorted({row.source_run for row in stranded})
        raise RuntimeError(
            f"{len(stranded)} keypoint rows labeled in an older version ({', '.join(runs)}) "
            "are not in the current one. Carry them forward first "
            "(python -m fisheye.labeling.carry_forward_tail_keypoints) so this Apply does not drop them."
        )

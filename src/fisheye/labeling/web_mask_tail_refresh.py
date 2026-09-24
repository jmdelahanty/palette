"""Offer mask-derived training successors without replacing active review tasks."""

from __future__ import annotations

from pathlib import Path

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
    source_tasks = []
    for task in store.list_tasks(recording_id=runtime.recording_id):
        scope = task.get("scope") or {}
        if (
            task.get("workflow_kind") != "keypoints"
            or str(scope.get("refined_run") or task.get("run_name") or "") != pose_name
            or Path(str(scope.get("zarr_path") or "")).resolve() != archive
        ):
            continue
        task_id = str(task["task_id"])
        if store.count_unapplied_session_checkpoints(
            task_id=task_id, component_name="keypoints"
        ) or store.count_pending_session_checkpoint_apply_effects(
            task_id=task_id, component_name="keypoints"
        ):
            raise RuntimeError(
                "Apply the paired keypoint task's saved checkpoints before retrying mask Apply; its manual labels are waiting to be copied."
            )
        source_tasks.append(task_id)
    result = regenerate_training_tail_version(
        archive=archive,
        refined_mask_run=runtime.refined.run_name,
        refined_keypoint_run=pose_name,
        apply_id=apply_id,
        expected_mask_revision=expected_mask_revision,
    )
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
        },
    )
    return offer

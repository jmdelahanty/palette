"""Browser checkpoint adapter for the crop-border training recipe exception."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fisheye.shared.keypoint_motion_authority import (
    keypoint_source_crop_run_from_attributes,
)
from fisheye.shared.recovered_training_review_contract import (
    REVIEW_SCHEMA,
    NATIVE_REVIEW_SCHEMA,
)
from fisheye.training.mask_tail_border_acceptance import (
    ACTION,
    acceptance_action,
    apply_acceptance_actions,
    bound_acceptances,
    body_digest,
)
from .web_runtimes import (
    _subject_mask_checkpoint_mask,
    _subject_mask_row_identity,
    _subject_mask_tail_border_status,
)


def _supported(runtime) -> bool:
    return runtime.component_name == "subject_body" and runtime.refined.group.attrs.get(
        "schema_id"
    ) in (REVIEW_SCHEMA, NATIVE_REVIEW_SCHEMA)


def save_action(
    store, runtime, *, roi_idx: int, edited_mask: np.ndarray, requested: object
):
    action = acceptance_action(requested)
    if action is not None and not _supported(runtime):
        raise ValueError(
            "Tail crop-border acceptance requires a training subject-body task"
        )
    if (
        action is not None
        and action["action"] == "accept"
        and not (np.any(edited_mask[(0, -1), :]) or np.any(edited_mask[:, (0, -1)]))
    ):
        raise ValueError("The body mask must touch the crop border for this acceptance")
    if action is None:
        prior = store.get_session_checkpoint(
            task_id=runtime.task_id,
            roi_idx=roi_idx,
            component_name=runtime.component_name,
            state="active",
        )
        if prior is not None and isinstance(prior.get("payload"), Mapping):
            prior_action = prior["payload"].get(ACTION)
            if prior_action is not None and np.array_equal(
                _subject_mask_checkpoint_mask(prior), edited_mask
            ):
                action = acceptance_action(prior_action)
    return action


def checkpoint_fields(action, edited_mask: np.ndarray) -> dict[str, object]:
    return (
        {ACTION: action, "tail_body_mask_sha256": body_digest(edited_mask)}
        if action is not None
        else {}
    )


def preflight_run(runtime) -> None:
    if _supported(runtime):
        bound_acceptances(
            runtime.refined.group, mask_labels=tuple(runtime.refined.component_names)
        )


def apply_row_action(
    runtime,
    checkpoint,
    *,
    roi_idx: int,
    before_mask: np.ndarray,
    edited_mask: np.ndarray,
):
    payload = checkpoint.get("payload")
    action = (
        acceptance_action(payload.get(ACTION)) if isinstance(payload, Mapping) else None
    )
    if action is not None and not _supported(runtime):
        raise ValueError(
            "Tail crop-border acceptance requires a training subject-body task"
        )
    if action is not None and payload.get("tail_body_mask_sha256") != body_digest(
        edited_mask
    ):
        raise ValueError("Tail crop-border checkpoint mask digest mismatch")
    if (
        action is not None
        and action["action"] == "accept"
        and not (np.any(edited_mask[(0, -1), :]) or np.any(edited_mask[:, (0, -1)]))
    ):
        raise ValueError("Accepted body mask no longer touches the crop border")
    if action is not None:
        identity = _subject_mask_row_identity(runtime, roi_idx)
        if (
            "source_crop_row_ids" not in identity
            or "frame_indices" not in identity
            or not keypoint_source_crop_run_from_attributes(runtime.refined.group.attrs)
        ):
            raise ValueError(
                "Tail crop-border acceptance requires crop and frame identity"
            )
    if not _supported(runtime):
        return None
    return {
        "roi_idx": int(roi_idx),
        "action": action,
        "before_body_sha256": body_digest(before_mask),
        "row_identity": _subject_mask_row_identity(runtime, roi_idx),
        "user": str(checkpoint.get("user") or ""),
        "timestamp": str(checkpoint.get("updated_at_utc") or ""),
    }


def prepare_apply_row(
    runtime,
    checkpoint,
    *,
    roi_idx: int,
    current_stack: np.ndarray,
    before_mask: np.ndarray,
    edited_mask: np.ndarray,
):
    if tuple(edited_mask.shape) != tuple(before_mask.shape):
        raise ValueError(
            f"checkpoint mask shape mismatch for row {roi_idx}: expected {tuple(before_mask.shape)}, got {tuple(edited_mask.shape)}"
        )
    action = apply_row_action(
        runtime,
        checkpoint,
        roi_idx=roi_idx,
        before_mask=before_mask,
        edited_mask=edited_mask,
    )
    edited_stack = current_stack.copy()
    edited_stack[runtime.comp_idx] = edited_mask
    return edited_stack, action


def commit_actions(runtime, actions: list[dict[str, object]], *, revision: int) -> None:
    if actions:
        apply_acceptance_actions(
            runtime.refined.group,
            actions=actions,
            mask_labels=tuple(runtime.refined.component_names),
            revision=revision,
        )


def row_status_payload(runtime, store) -> dict[str, object]:
    roi_idx = int(runtime.roi_indices[runtime.position])
    mask = np.asarray(
        runtime.refined.group["masks_roi"][roi_idx, runtime.comp_idx], dtype=np.uint8
    )
    checkpoint = store.get_session_checkpoint(
        task_id=runtime.task_id,
        roi_idx=roi_idx,
        component_name=runtime.component_name,
        state="active",
    )
    if checkpoint is not None:
        mask = _subject_mask_checkpoint_mask(checkpoint)
    return {
        "ok": True,
        "roi_idx": roi_idx,
        "tail_crop_border": _subject_mask_tail_border_status(
            runtime, store=store, roi_idx=roi_idx, mask=mask
        ),
    }

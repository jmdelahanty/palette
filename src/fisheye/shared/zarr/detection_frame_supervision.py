"""Shared bridge from editable detection review to immutable frame-level supervision.

Detection instances are sparse positive observations.  Detection training,
however, samples images: one image may contain zero, one, or many instances.
This module owns the conversion between those two axes.  It never invents a
placeholder instance for an explicitly reviewed negative frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Optional

import numpy as np
import zarr

from fisheye.shared.zarr.detect_frame_decisions import (
    DETECT_FRAME_DECISION_FAMILY,
    FRAME_DECISION_NEGATIVE,
    FRAME_DECISION_UNREVIEWED,
    FRAME_REASON_NONE,
    FRAME_REVIEW_CONTRACT_ATTR,
    FRAME_REVIEW_CONTRACT_ID,
    load_detect_frame_decisions,
)
from fisheye.shared.zarr.array_contracts import ArrayContract, UINT8, UINT16
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import TRAINING_IMMUTABLE_V1


DETECTION_TRAINING_SUPERVISION_GROUP = "detection_training_supervision"
DETECTION_TRAINING_SUPERVISION_SCHEMA_ID = (
    "palette.detection_training_frame_supervision"
)
DETECTION_TRAINING_SUPERVISION_SCHEMA_VERSION = 1

LABEL_STATE_POSITIVE = np.uint8(1)
LABEL_STATE_NEGATIVE = np.uint8(2)
LABEL_STATE_CODE_MAP = {1: "positive", 2: "negative"}

_SUPERVISION_ARRAY_CONTRACTS = {
    "label_state_codes": ArrayContract(
        schema_id=f"{DETECTION_TRAINING_SUPERVISION_SCHEMA_ID}.label_state_codes",
        schema_version=DETECTION_TRAINING_SUPERVISION_SCHEMA_VERSION,
        dtype=UINT8,
        shape_template=("n_frames",),
        axis_names=("frame",),
        description="Positive or explicitly reviewed negative training state.",
    ),
    "reason_codes": ArrayContract(
        schema_id=f"{DETECTION_TRAINING_SUPERVISION_SCHEMA_ID}.reason_codes",
        schema_version=DETECTION_TRAINING_SUPERVISION_SCHEMA_VERSION,
        dtype=UINT16,
        shape_template=("n_frames",),
        axis_names=("frame",),
        description="Source review reason for each supervised frame.",
    ),
}


class DetectionFrameSupervisionError(ValueError):
    """Raised when sparse instances and frame decisions cannot be exported."""


@dataclass(frozen=True)
class DetectionFrameSupervisionPlan:
    """Exact mapping from one source artifact into training sample/instance axes."""

    source_frame_indices: np.ndarray
    label_state_codes: np.ndarray
    reason_codes: np.ndarray
    source_instance_row_indices: np.ndarray
    instance_output_frame_indices: np.ndarray
    frame_counts: np.ndarray
    frame_row_offsets: np.ndarray
    source_decision_run_path: Optional[str]
    source_decision_digest: Optional[str]

    @property
    def frame_count(self) -> int:
        return int(self.source_frame_indices.shape[0])

    @property
    def instance_count(self) -> int:
        return int(self.source_instance_row_indices.shape[0])

    @property
    def positive_frame_count(self) -> int:
        return int(np.count_nonzero(self.label_state_codes == LABEL_STATE_POSITIVE))

    @property
    def negative_frame_count(self) -> int:
        return int(np.count_nonzero(self.label_state_codes == LABEL_STATE_NEGATIVE))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _refined_run_from_bbox_path(bbox_path: str) -> Optional[str]:
    parts = str(bbox_path).strip("/").split("/")
    if len(parts) >= 4 and parts[0] == "refined_detect_runs":
        return parts[1]
    return None


def _decision_digest(
    *,
    run_name: str,
    decision_codes: np.ndarray,
    reason_codes: np.ndarray,
    source_acquisition_frame_index: np.ndarray,
) -> str:
    return _canonical_sha256(
        {
            "schema_id": "palette.detect_frame_decisions",
            "schema_version": 1,
            "source_refined_detect_run": str(run_name),
            "decision_codes_sha256": sha256(
                np.asarray(decision_codes, dtype=np.uint8).tobytes(order="C")
            ).hexdigest(),
            "reason_codes_sha256": sha256(
                np.asarray(reason_codes, dtype=np.uint16).tobytes(order="C")
            ).hexdigest(),
            "source_acquisition_frame_index_sha256": sha256(
                np.asarray(source_acquisition_frame_index, dtype=np.int64).tobytes(
                    order="C"
                )
            ).hexdigest(),
        }
    )


def build_detection_frame_supervision_plan(
    root: zarr.Group,
    *,
    bbox_path: str,
    frame_indices_path: Optional[str],
    n_frames: int,
) -> DetectionFrameSupervisionPlan:
    """Build a fail-closed frame-axis plan for one detection-training source.

    Sources with a bound frame-decision run are review-complete only when every
    source frame is either positive or explicitly negative.  Historical sources
    without that surface preserve their positive-only behavior.
    """

    if int(n_frames) <= 0:
        raise DetectionFrameSupervisionError("Detection image axis must be positive.")
    bbox = np.asarray(root[bbox_path][:])
    if bbox.ndim != 2 or tuple(bbox.shape[1:]) != (4,):
        raise DetectionFrameSupervisionError(
            f"{bbox_path} must have shape (N, 4); got {bbox.shape}."
        )
    if bbox.size and not np.isfinite(bbox).all():
        raise DetectionFrameSupervisionError(
            f"{bbox_path} contains non-finite rows; export requires canonical positive instances."
        )
    instance_count = int(bbox.shape[0])
    if frame_indices_path is None:
        instance_frames = np.arange(instance_count, dtype=np.int64)
    else:
        instance_frames = np.asarray(
            root[frame_indices_path][:], dtype=np.int64
        ).reshape(-1)
    if instance_frames.shape != (instance_count,):
        raise DetectionFrameSupervisionError(
            "Detection frame_indices length does not match the instance axis."
        )
    if instance_frames.size:
        if int(instance_frames.min()) < 0 or int(instance_frames.max()) >= int(n_frames):
            raise DetectionFrameSupervisionError(
                "Detection frame_indices are outside the source image axis."
            )
        if np.any(np.diff(instance_frames) < 0):
            raise DetectionFrameSupervisionError(
                "Detection instances must be ordered by nondecreasing frame_indices."
            )

    positive_frames = np.unique(instance_frames)
    run_name = _refined_run_from_bbox_path(bbox_path)
    decision_path: Optional[str] = None
    decision_digest: Optional[str] = None
    selected_frames = positive_frames
    selected_states = np.full(
        positive_frames.shape, LABEL_STATE_POSITIVE, dtype=np.uint8
    )
    selected_reasons = np.full(
        positive_frames.shape, FRAME_REASON_NONE, dtype=np.uint16
    )

    family = root.get(DETECT_FRAME_DECISION_FAMILY)
    has_bound_decisions = bool(
        run_name and family is not None and str(run_name) in family
    )
    refined_run = (
        root.get(f"refined_detect_runs/{run_name}") if run_name is not None else None
    )
    strict_review_declared = bool(
        refined_run is not None
        and refined_run.attrs.get(FRAME_REVIEW_CONTRACT_ATTR)
        == FRAME_REVIEW_CONTRACT_ID
    )
    if has_bound_decisions:
        assert run_name is not None
        decisions = load_detect_frame_decisions(
            root,
            source_refined_detect_run=run_name,
            n_frames=int(n_frames),
        )
        positive_mask = np.zeros(int(n_frames), dtype=bool)
        positive_mask[positive_frames] = True
        negative_mask = decisions.decision_codes == FRAME_DECISION_NEGATIVE
        collision = np.flatnonzero(positive_mask & negative_mask)
        if collision.size:
            raise DetectionFrameSupervisionError(
                "Frames cannot be both positive and explicitly negative: "
                f"{collision[:10].tolist()}."
            )
        unresolved = np.flatnonzero(
            (~positive_mask)
            & (decisions.decision_codes == FRAME_DECISION_UNREVIEWED)
        )
        if unresolved.size:
            raise DetectionFrameSupervisionError(
                "Frame-decision export is incomplete; unresolved frames include "
                f"{unresolved[:10].tolist()} ({unresolved.size} total)."
            )
        selected_frames = np.arange(int(n_frames), dtype=np.int64)
        selected_states = np.where(
            positive_mask, LABEL_STATE_POSITIVE, LABEL_STATE_NEGATIVE
        ).astype(np.uint8, copy=False)
        selected_reasons = np.where(
            positive_mask, FRAME_REASON_NONE, decisions.reason_codes
        ).astype(np.uint16, copy=False)
        decision_path = f"{DETECT_FRAME_DECISION_FAMILY}/{run_name}"
        decision_group = family[run_name]
        source_acquisition = np.asarray(
            decision_group["source_acquisition_frame_index"][:], dtype=np.int64
        ).reshape(-1)
        decision_digest = _decision_digest(
            run_name=run_name,
            decision_codes=decisions.decision_codes,
            reason_codes=decisions.reason_codes,
            source_acquisition_frame_index=source_acquisition,
        )
    elif strict_review_declared:
        unresolved = np.setdiff1d(
            np.arange(int(n_frames), dtype=np.int64),
            positive_frames,
            assume_unique=True,
        )
        if unresolved.size:
            raise DetectionFrameSupervisionError(
                "Frame-decision export is incomplete; unresolved frames include "
                f"{unresolved[:10].tolist()} ({unresolved.size} total)."
            )
        selected_frames = np.arange(int(n_frames), dtype=np.int64)
        selected_states = np.full(
            int(n_frames), LABEL_STATE_POSITIVE, dtype=np.uint8
        )
        selected_reasons = np.full(
            int(n_frames), FRAME_REASON_NONE, dtype=np.uint16
        )

    if selected_frames.size == 0:
        raise DetectionFrameSupervisionError(
            "Source has neither positive instances nor explicit negative frames."
        )

    output_frame_for_source = np.full(int(n_frames), -1, dtype=np.int64)
    output_frame_for_source[selected_frames] = np.arange(
        selected_frames.shape[0], dtype=np.int64
    )
    instance_output_frames = output_frame_for_source[instance_frames]
    if np.any(instance_output_frames < 0):
        raise DetectionFrameSupervisionError(
            "A positive instance was excluded from the supervised frame axis."
        )
    frame_counts = np.bincount(
        instance_output_frames,
        minlength=int(selected_frames.shape[0]),
    ).astype(np.int32, copy=False)
    frame_offsets = np.zeros(int(selected_frames.shape[0]) + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)

    observed_positive = frame_counts > 0
    if not np.array_equal(
        observed_positive, selected_states == LABEL_STATE_POSITIVE
    ):
        raise DetectionFrameSupervisionError(
            "Positive/negative label states do not agree with instance ranges."
        )

    return DetectionFrameSupervisionPlan(
        source_frame_indices=selected_frames.astype(np.int64, copy=False),
        label_state_codes=selected_states,
        reason_codes=selected_reasons,
        source_instance_row_indices=np.arange(instance_count, dtype=np.int64),
        instance_output_frame_indices=instance_output_frames.astype(
            np.int32, copy=False
        ),
        frame_counts=frame_counts,
        frame_row_offsets=frame_offsets,
        source_decision_run_path=decision_path,
        source_decision_digest=decision_digest,
    )


def validate_exported_frame_supervision(
    group: zarr.Group,
    *,
    frame_counts: np.ndarray,
) -> None:
    """Deeply validate the immutable frame-level supervision declaration."""

    attrs = dict(group.attrs)
    if attrs.get("schema_id") != DETECTION_TRAINING_SUPERVISION_SCHEMA_ID:
        raise DetectionFrameSupervisionError("Unexpected supervision schema_id.")
    if attrs.get("schema_version") != DETECTION_TRAINING_SUPERVISION_SCHEMA_VERSION:
        raise DetectionFrameSupervisionError("Unexpected supervision schema_version.")
    if attrs.get("label_state_code_map") != {
        str(code): label for code, label in LABEL_STATE_CODE_MAP.items()
    }:
        raise DetectionFrameSupervisionError("Unexpected label_state_code_map.")
    expected_names = {"label_state_codes", "reason_codes"}
    if set(group.array_keys()) != expected_names:
        raise DetectionFrameSupervisionError(
            "Frame supervision must contain exactly label_state_codes and reason_codes."
        )
    states = np.asarray(group["label_state_codes"][:], dtype=np.uint8).reshape(-1)
    reasons = np.asarray(group["reason_codes"][:], dtype=np.uint16).reshape(-1)
    counts = np.asarray(frame_counts, dtype=np.int64).reshape(-1)
    if states.shape != counts.shape or reasons.shape != counts.shape:
        raise DetectionFrameSupervisionError(
            "Frame supervision arrays must match the frame-count axis."
        )
    unknown = sorted(set(int(v) for v in states.tolist()) - set(LABEL_STATE_CODE_MAP))
    if unknown:
        raise DetectionFrameSupervisionError(
            f"Unknown frame supervision states: {unknown}."
        )
    if np.any((states == LABEL_STATE_POSITIVE) != (counts > 0)):
        raise DetectionFrameSupervisionError(
            "Positive states must have instances and negative states must have none."
        )
    if np.any((states == LABEL_STATE_POSITIVE) & (reasons != FRAME_REASON_NONE)):
        raise DetectionFrameSupervisionError(
            "Positive frame supervision must use reason code zero."
        )


def write_exported_frame_supervision(
    root: zarr.Group,
    *,
    label_state_codes: np.ndarray,
    reason_codes: np.ndarray,
) -> zarr.Group:
    """Write the immutable training frame-axis declaration via shared planning."""

    states = np.asarray(label_state_codes, dtype=np.uint8).reshape(-1)
    reasons = np.asarray(reason_codes, dtype=np.uint16).reshape(-1)
    if states.shape != reasons.shape or states.size == 0:
        raise DetectionFrameSupervisionError(
            "Exported supervision arrays must be nonempty and share one frame axis."
        )
    group = root.require_group(DETECTION_TRAINING_SUPERVISION_GROUP)
    for name, values in (
        ("label_state_codes", states),
        ("reason_codes", reasons),
    ):
        contract = _SUPERVISION_ARRAY_CONTRACTS[name]
        intent = contract.storage_intent(
            shape=values.shape,
            access=AccessPattern.EAGER,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=(int(values.shape[0]),),
            shard_axes=(0,),
            name=name,
            dimensions={"n_frames": int(values.shape[0])},
        )
        array = create_array_from_plan(
            group,
            name=name,
            contract=contract,
            plan=plan_storage(intent, TRAINING_IMMUTABLE_V1),
            fill_value=0,
        )
        array[:] = values
    group.attrs.update(
        {
            "schema_id": DETECTION_TRAINING_SUPERVISION_SCHEMA_ID,
            "schema_version": DETECTION_TRAINING_SUPERVISION_SCHEMA_VERSION,
            "sample_axis": "frame",
            "zero_instance_frame_semantics": "explicit_reviewed_negative",
            "label_state_code_map": {
                str(code): label for code, label in LABEL_STATE_CODE_MAP.items()
            },
            "n_frames": int(states.shape[0]),
            "n_positive_frames": int(
                np.count_nonzero(states == LABEL_STATE_POSITIVE)
            ),
            "n_negative_frames": int(
                np.count_nonzero(states == LABEL_STATE_NEGATIVE)
            ),
        }
    )
    return group

"""Versioned frame-level decisions for sparse refined-detection review.

Refined detection instances represent positive observations.  An explicitly
reviewed empty frame therefore cannot be encoded as an instance without
inventing a fake detection.  This module owns the small mutable sibling
surface that records that frame-level decision while binding it to one exact
refined-detection run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

import numpy as np
import zarr

from fisheye.shared.zarr.array_contracts import (
    ArrayContract,
    INT32,
    INT64,
    UINT8,
    UINT16,
)
from fisheye.shared.zarr.array_factory import create_array_from_plan
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import EDITABLE_LOCAL_V1

DETECT_FRAME_DECISION_FAMILY = "detect_frame_decision_runs"
DETECT_FRAME_DECISION_SCHEMA_ID = "palette.detect_frame_decisions"
DETECT_FRAME_DECISION_SCHEMA_VERSION = 1
FRAME_REVIEW_CONTRACT_ATTR = "detect_frame_review_contract"
FRAME_REVIEW_CONTRACT_ID = "palette.detect_frame_review.v1"

FRAME_DECISION_UNREVIEWED = np.uint8(0)
FRAME_DECISION_NEGATIVE = np.uint8(1)
FRAME_REASON_NONE = np.uint16(0)
FRAME_REASON_SUBJECT_OUTSIDE_DISH = np.uint16(1)

DECISION_CODE_MAP: Mapping[int, str] = MappingProxyType(
    {
        int(FRAME_DECISION_UNREVIEWED): "unreviewed",
        int(FRAME_DECISION_NEGATIVE): "negative",
    }
)
REASON_CODE_MAP: Mapping[int, str] = MappingProxyType(
    {
        int(FRAME_REASON_NONE): "none",
        int(FRAME_REASON_SUBJECT_OUTSIDE_DISH): "subject_outside_dish",
    }
)
REASON_CODE_BY_LABEL: Mapping[str, int] = MappingProxyType(
    {label: code for code, label in REASON_CODE_MAP.items()}
)


class DetectFrameDecisionError(ValueError):
    """Raised when a frame-decision surface violates the v1 contract."""


@dataclass(frozen=True)
class DetectFrameDecisions:
    decision_codes: np.ndarray
    reason_codes: np.ndarray

    def decision_label(self, frame_index: int) -> str:
        return DECISION_CODE_MAP[int(self.decision_codes[int(frame_index)])]

    def reason_label(self, frame_index: int) -> str:
        return REASON_CODE_MAP[int(self.reason_codes[int(frame_index)])]


_ARRAY_CONTRACTS: Mapping[str, ArrayContract] = MappingProxyType(
    {
        "frame_indices": ArrayContract(
            schema_id=f"{DETECT_FRAME_DECISION_SCHEMA_ID}.frame_indices",
            schema_version=DETECT_FRAME_DECISION_SCHEMA_VERSION,
            dtype=INT32,
            shape_template=("n_frames",),
            axis_names=("frame",),
            description="Contiguous local frame identity for every reviewable frame.",
            units="frame_index",
        ),
        "source_acquisition_frame_index": ArrayContract(
            schema_id=(
                f"{DETECT_FRAME_DECISION_SCHEMA_ID}.source_acquisition_frame_index"
            ),
            schema_version=DETECT_FRAME_DECISION_SCHEMA_VERSION,
            dtype=INT64,
            shape_template=("n_frames",),
            axis_names=("frame",),
            description="Acquisition-frame identity associated with each local frame.",
            units="acquisition_frame_index",
        ),
        "decision_codes": ArrayContract(
            schema_id=f"{DETECT_FRAME_DECISION_SCHEMA_ID}.decision_codes",
            schema_version=DETECT_FRAME_DECISION_SCHEMA_VERSION,
            dtype=UINT8,
            shape_template=("n_frames",),
            axis_names=("frame",),
            description="Current explicit frame-review decision code.",
        ),
        "reason_codes": ArrayContract(
            schema_id=f"{DETECT_FRAME_DECISION_SCHEMA_ID}.reason_codes",
            schema_version=DETECT_FRAME_DECISION_SCHEMA_VERSION,
            dtype=UINT16,
            shape_template=("n_frames",),
            axis_names=("frame",),
            description="Reason code for the current frame-review decision.",
        ),
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_run_component(value: str) -> str:
    name = str(value).strip()
    if not name or "/" in name or name in {".", ".."}:
        raise DetectFrameDecisionError(
            "source_refined_detect_run must be one nonempty Zarr path component."
        )
    return name


def _empty(n_frames: int) -> DetectFrameDecisions:
    return DetectFrameDecisions(
        decision_codes=np.zeros(int(n_frames), dtype=np.uint8),
        reason_codes=np.zeros(int(n_frames), dtype=np.uint16),
    )


def _source_acquisition_indices(root: zarr.Group, *, n_frames: int) -> np.ndarray:
    raw_video = root.get("raw_video")
    if raw_video is not None and "original_frame_indices" in raw_video:
        values = np.asarray(
            raw_video["original_frame_indices"][:], dtype=np.int64
        ).reshape(-1)
        if values.shape != (int(n_frames),):
            raise DetectFrameDecisionError(
                "raw_video/original_frame_indices length does not match the frame-decision axis."
            )
        if np.any(values < 0) or (values.size > 1 and np.any(np.diff(values) <= 0)):
            raise DetectFrameDecisionError(
                "raw_video/original_frame_indices must be nonnegative and strictly increasing."
            )
        return values
    return np.arange(int(n_frames), dtype=np.int64)


def _array_plan(name: str, *, n_frames: int):
    contract = _ARRAY_CONTRACTS[name]
    intent = contract.storage_intent(
        shape=(int(n_frames),),
        access=AccessPattern.EAGER,
        write_mode=WriteMode.RANDOM_UPDATE,
        access_unit_shape=(1,),
        growth_axis=0,
        shard_axes=(0,),
        name=name,
        dimensions={"n_frames": int(n_frames)},
    )
    return plan_storage(intent, EDITABLE_LOCAL_V1)


def _validate_arrays(
    run: zarr.Group,
    *,
    source_refined_detect_run: str,
    n_frames: int,
    source_acquisition_frame_index: np.ndarray,
) -> DetectFrameDecisions:
    attrs = dict(run.attrs)
    expected_attrs = {
        "schema_id": DETECT_FRAME_DECISION_SCHEMA_ID,
        "schema_version": DETECT_FRAME_DECISION_SCHEMA_VERSION,
        "source_refined_detect_run": source_refined_detect_run,
        "n_frames": int(n_frames),
    }
    for key, expected in expected_attrs.items():
        if attrs.get(key) != expected:
            raise DetectFrameDecisionError(
                f"Frame-decision attribute {key!r} must equal {expected!r}; "
                f"got {attrs.get(key)!r}."
            )
    expected_decision_map = {
        str(code): label for code, label in DECISION_CODE_MAP.items()
    }
    expected_reason_map = {str(code): label for code, label in REASON_CODE_MAP.items()}
    if attrs.get("decision_code_map") != expected_decision_map:
        raise DetectFrameDecisionError(
            "decision_code_map is not the exact v1 registry."
        )
    if attrs.get("reason_code_map") != expected_reason_map:
        raise DetectFrameDecisionError("reason_code_map is not the exact v1 registry.")
    if (
        attrs.get("selector_eligible") is not False
        or attrs.get("lifecycle_state") != "editable"
    ):
        raise DetectFrameDecisionError(
            "Frame-decision review state must remain editable and selector-ineligible."
        )
    expected_names = set(_ARRAY_CONTRACTS)
    observed_names = set(run.array_keys())
    if observed_names != expected_names:
        raise DetectFrameDecisionError(
            "Frame-decision arrays must be exact; "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}."
        )
    dimensions = {"n_frames": int(n_frames)}
    for name, contract in _ARRAY_CONTRACTS.items():
        errors = contract.validate_observation(run[name], dimensions=dimensions)
        if errors:
            raise DetectFrameDecisionError(
                f"Frame-decision array {name!r} violates its contract: {'; '.join(errors)}."
            )
    frame_indices = np.asarray(run["frame_indices"][:], dtype=np.int32).reshape(-1)
    if not np.array_equal(frame_indices, np.arange(int(n_frames), dtype=np.int32)):
        raise DetectFrameDecisionError(
            "frame_indices must equal arange(n_frames) exactly."
        )
    observed_source = np.asarray(
        run["source_acquisition_frame_index"][:], dtype=np.int64
    ).reshape(-1)
    if not np.array_equal(observed_source, source_acquisition_frame_index):
        raise DetectFrameDecisionError(
            "source_acquisition_frame_index does not match raw_video frame lineage."
        )
    decisions = np.asarray(run["decision_codes"][:], dtype=np.uint8).reshape(-1)
    reasons = np.asarray(run["reason_codes"][:], dtype=np.uint16).reshape(-1)
    unknown_decisions = sorted(
        set(int(v) for v in decisions.tolist()) - set(DECISION_CODE_MAP)
    )
    unknown_reasons = sorted(
        set(int(v) for v in reasons.tolist()) - set(REASON_CODE_MAP)
    )
    if unknown_decisions:
        raise DetectFrameDecisionError(
            f"Unknown frame decision codes: {unknown_decisions}."
        )
    if unknown_reasons:
        raise DetectFrameDecisionError(
            f"Unknown frame reason codes: {unknown_reasons}."
        )
    invalid_pairs = np.flatnonzero(
        ((decisions == FRAME_DECISION_UNREVIEWED) & (reasons != FRAME_REASON_NONE))
        | ((decisions == FRAME_DECISION_NEGATIVE) & (reasons == FRAME_REASON_NONE))
    )
    if invalid_pairs.size:
        raise DetectFrameDecisionError(
            "Frame decision/reason pairs are inconsistent at frames "
            f"{invalid_pairs[:10].tolist()}."
        )
    return DetectFrameDecisions(decision_codes=decisions, reason_codes=reasons)


def load_detect_frame_decisions(
    root: zarr.Group,
    *,
    source_refined_detect_run: str,
    n_frames: int,
) -> DetectFrameDecisions:
    """Load and deeply validate one bound decision run, or return all-unreviewed."""

    run_name = _require_run_component(source_refined_detect_run)
    if int(n_frames) <= 0:
        raise DetectFrameDecisionError("n_frames must be positive.")
    family = root.get(DETECT_FRAME_DECISION_FAMILY)
    if family is None or run_name not in family:
        return _empty(int(n_frames))
    return _validate_arrays(
        family[run_name],
        source_refined_detect_run=run_name,
        n_frames=int(n_frames),
        source_acquisition_frame_index=_source_acquisition_indices(
            root, n_frames=int(n_frames)
        ),
    )


def _ensure_run(
    root: zarr.Group,
    *,
    source_refined_detect_run: str,
    n_frames: int,
) -> zarr.Group:
    run_name = _require_run_component(source_refined_detect_run)
    source_indices = _source_acquisition_indices(root, n_frames=int(n_frames))
    family = root.get(DETECT_FRAME_DECISION_FAMILY)
    if family is None:
        family = root.create_group(DETECT_FRAME_DECISION_FAMILY)
        family.attrs.update(
            {
                "schema_family": DETECT_FRAME_DECISION_SCHEMA_ID,
                "selector_policy": "none_editable_training_review_surface",
            }
        )
    if run_name in family:
        run = family[run_name]
        _validate_arrays(
            run,
            source_refined_detect_run=run_name,
            n_frames=int(n_frames),
            source_acquisition_frame_index=source_indices,
        )
        return run

    run = family.create_group(run_name)
    created_at = _utc_now()
    run.attrs.update(
        {
            "schema_id": DETECT_FRAME_DECISION_SCHEMA_ID,
            "schema_version": DETECT_FRAME_DECISION_SCHEMA_VERSION,
            "source_refined_detect_run": run_name,
            "n_frames": int(n_frames),
            "decision_code_map": {
                str(code): label for code, label in DECISION_CODE_MAP.items()
            },
            "reason_code_map": {
                str(code): label for code, label in REASON_CODE_MAP.items()
            },
            "created_at": created_at,
            "updated_at": created_at,
            "selector_eligible": False,
            "lifecycle_state": "editable",
        }
    )
    values = {
        "frame_indices": np.arange(int(n_frames), dtype=np.int32),
        "source_acquisition_frame_index": source_indices,
        "decision_codes": np.zeros(int(n_frames), dtype=np.uint8),
        "reason_codes": np.zeros(int(n_frames), dtype=np.uint16),
    }
    for name, value in values.items():
        array = create_array_from_plan(
            run,
            name=name,
            contract=_ARRAY_CONTRACTS[name],
            plan=_array_plan(name, n_frames=int(n_frames)),
            fill_value=0,
        )
        array[:] = value
    return run


def set_detect_frame_negative(
    root: zarr.Group,
    *,
    source_refined_detect_run: str,
    n_frames: int,
    frame_index: int,
    reason: str = "subject_outside_dish",
) -> DetectFrameDecisions:
    """Persist one explicit reviewed-negative frame decision."""

    frame = int(frame_index)
    if not 0 <= frame < int(n_frames):
        raise DetectFrameDecisionError(
            f"frame_index must be in [0, {int(n_frames)}); got {frame}."
        )
    reason_label = str(reason).strip().lower()
    reason_code = REASON_CODE_BY_LABEL.get(reason_label)
    if reason_code is None or reason_code == int(FRAME_REASON_NONE):
        raise DetectFrameDecisionError(
            f"Negative frame reason must be one of {sorted(set(REASON_CODE_BY_LABEL) - {'none'})}."
        )
    run = _ensure_run(
        root,
        source_refined_detect_run=source_refined_detect_run,
        n_frames=int(n_frames),
    )
    run["decision_codes"][frame] = FRAME_DECISION_NEGATIVE
    run["reason_codes"][frame] = np.uint16(reason_code)
    run.attrs["updated_at"] = _utc_now()
    return load_detect_frame_decisions(
        root,
        source_refined_detect_run=source_refined_detect_run,
        n_frames=int(n_frames),
    )


def clear_detect_frame_decision(
    root: zarr.Group,
    *,
    source_refined_detect_run: str,
    n_frames: int,
    frame_index: int,
) -> DetectFrameDecisions:
    """Return an existing frame decision to unreviewed without creating a run."""

    frame = int(frame_index)
    if not 0 <= frame < int(n_frames):
        raise DetectFrameDecisionError(
            f"frame_index must be in [0, {int(n_frames)}); got {frame}."
        )
    family = root.get(DETECT_FRAME_DECISION_FAMILY)
    run_name = _require_run_component(source_refined_detect_run)
    if family is None or run_name not in family:
        return _empty(int(n_frames))
    run = family[run_name]
    _validate_arrays(
        run,
        source_refined_detect_run=run_name,
        n_frames=int(n_frames),
        source_acquisition_frame_index=_source_acquisition_indices(
            root, n_frames=int(n_frames)
        ),
    )
    run["reason_codes"][frame] = FRAME_REASON_NONE
    run["decision_codes"][frame] = FRAME_DECISION_UNREVIEWED
    run.attrs["updated_at"] = _utc_now()
    return load_detect_frame_decisions(
        root,
        source_refined_detect_run=run_name,
        n_frames=int(n_frames),
    )


__all__ = [
    "DECISION_CODE_MAP",
    "DETECT_FRAME_DECISION_FAMILY",
    "DETECT_FRAME_DECISION_SCHEMA_ID",
    "DETECT_FRAME_DECISION_SCHEMA_VERSION",
    "DetectFrameDecisionError",
    "DetectFrameDecisions",
    "FRAME_DECISION_NEGATIVE",
    "FRAME_DECISION_UNREVIEWED",
    "REASON_CODE_MAP",
    "clear_detect_frame_decision",
    "load_detect_frame_decisions",
    "set_detect_frame_negative",
]

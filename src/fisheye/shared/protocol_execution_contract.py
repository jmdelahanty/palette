"""Strict consumer for Citrus finalized protocol-execution evidence.

The producer interval authority is ``stimulus_frame_num``.  Camera-frame IDs
are retained only as correspondence evidence and are never converted into
Palette acquisition-array indices by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, Mapping

import h5py
import numpy as np

from fisheye.shared.coordinate_identity import identity_array_content_sha256
from fisheye.shared.protocol_semantic_contract import (
    ProtocolSemanticContractError,
    ProtocolSemanticSnapshot,
)


EXECUTION_GROUP_H5_PATH = "/protocol_execution"
EXECUTION_JSON_H5_PATH = "/protocol_execution/execution_index_json"
EXECUTION_HASH_H5_PATH = "/protocol_execution/execution_index_hash"
EXECUTION_SCHEMA_ID = "citrus.protocol.execution_index"
EXECUTION_SCHEMA_VERSION = 1
EXECUTION_POLICY_ID = (
    "citrus.protocol.execution_index.half_open_stimulus_frames.v1"
)
EXECUTION_INTERVAL_AXIS = "stimulus_frame_num"
EXECUTION_CAMERA_FRAME_ROLE = "correspondence_only"
CHASER_REPOSITIONING_OWNERSHIP = (
    "before_chaser_post_start_belongs_to_training;at_or_after_belongs_to_post"
)
CHASER_PHASE_NAMES = ("chaser_pre", "chaser_training", "chaser_post")
PALETTE_PROTOCOL_EXECUTION_SCHEMA_ID = "palette.stimulus.protocol_execution.v1"
PALETTE_PROTOCOL_EXECUTION_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ProtocolExecutionContractError(ProtocolSemanticContractError):
    """Raised when finalized producer execution evidence is incomplete or stale."""


def _decode_text(value: Any, *, name: str) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8")
    elif isinstance(value, str):
        text = value
    elif hasattr(value, "item"):
        return _decode_text(value.item(), name=name)
    else:
        raise ProtocolExecutionContractError(f"{name} is not UTF-8 scalar text.")
    if not text:
        raise ProtocolExecutionContractError(f"{name} is empty.")
    return text


def _parse_object(text: str, *, name: str) -> Mapping[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolExecutionContractError(
                    f"{name} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as exc:
        raise ProtocolExecutionContractError(f"{name} is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise ProtocolExecutionContractError(f"{name} must contain one object.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    observed = set(value)
    missing = sorted(required - observed)
    unknown = sorted(observed - required - optional)
    if missing or unknown:
        raise ProtocolExecutionContractError(
            f"{name} has a non-canonical key set; missing={missing}, unknown={unknown}."
        )


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolExecutionContractError(
            f"{name} must be one exact integer >= {minimum}."
        )
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ProtocolStimulusFrameInterval:
    """One producer-authored half-open stimulus-frame interval."""

    start_stimulus_frame_inclusive: int
    end_stimulus_frame_exclusive: int
    first_camera_frame_id: int
    last_camera_frame_id: int

    @property
    def is_empty(self) -> bool:
        return self.start_stimulus_frame_inclusive == self.end_stimulus_frame_exclusive

    def to_record(self) -> dict[str, int]:
        return {
            "start_stimulus_frame_inclusive": self.start_stimulus_frame_inclusive,
            "end_stimulus_frame_exclusive": self.end_stimulus_frame_exclusive,
            "first_camera_frame_id": self.first_camera_frame_id,
            "last_camera_frame_id": self.last_camera_frame_id,
        }


@dataclass(frozen=True)
class ProtocolExecutionStep:
    step_index: int
    stimulus_mode_id: int
    completion_status: str
    end_reason: str
    interval: ProtocolStimulusFrameInterval
    chaser_phases: Mapping[str, ProtocolStimulusFrameInterval] | None


@dataclass(frozen=True)
class ProtocolExecutionIndex:
    """Verified exact producer execution index bound to one trial index."""

    execution_json: str
    execution_hash: str
    protocol_trial_index_hash: str
    status: str
    steps: tuple[ProtocolExecutionStep, ...]
    payload: Mapping[str, Any]


def _interval(value: object, *, name: str) -> ProtocolStimulusFrameInterval:
    if not isinstance(value, Mapping):
        raise ProtocolExecutionContractError(f"{name} must be one object.")
    _exact_keys(
        value,
        required={
            "start_stimulus_frame_inclusive",
            "end_stimulus_frame_exclusive",
            "first_camera_frame_id",
            "last_camera_frame_id",
        },
        name=name,
    )
    start = _exact_int(
        value.get("start_stimulus_frame_inclusive"),
        name=f"{name}.start_stimulus_frame_inclusive",
    )
    end = _exact_int(
        value.get("end_stimulus_frame_exclusive"),
        name=f"{name}.end_stimulus_frame_exclusive",
    )
    if end < start:
        raise ProtocolExecutionContractError(f"{name} has reversed bounds.")
    return ProtocolStimulusFrameInterval(
        start_stimulus_frame_inclusive=start,
        end_stimulus_frame_exclusive=end,
        first_camera_frame_id=_exact_int(
            value.get("first_camera_frame_id"),
            name=f"{name}.first_camera_frame_id",
        ),
        last_camera_frame_id=_exact_int(
            value.get("last_camera_frame_id"),
            name=f"{name}.last_camera_frame_id",
        ),
    )


def validate_protocol_execution_index(
    *,
    execution_json: str,
    execution_hash: str,
    snapshot: ProtocolSemanticSnapshot,
) -> ProtocolExecutionIndex:
    """Validate exact bytes, recipe binding, and realized phase partitions."""

    if snapshot.snapshot_schema_version != 2:
        raise ProtocolExecutionContractError(
            "Finalized execution-index v1 requires Citrus protocol snapshot v2."
        )
    if _SHA256_RE.fullmatch(execution_hash) is None:
        raise ProtocolExecutionContractError("execution_index_hash has bad format.")
    observed_hash = "sha256:" + sha256(execution_json.encode("utf-8")).hexdigest()
    if observed_hash != execution_hash:
        raise ProtocolExecutionContractError(
            "execution_index_json bytes do not match execution_index_hash."
        )
    payload = _parse_object(execution_json, name="execution_index_json")
    _exact_keys(
        payload,
        required={
            "schema_id",
            "schema_version",
            "policy_id",
            "status",
            "authoritative_interval_axis",
            "camera_frame_role",
            "chaser_repositioning_ownership",
            "protocol_trial_index_hash",
            "steps",
        },
        optional={"issues"},
        name="execution_index_json",
    )
    expected_scalars = {
        "schema_id": EXECUTION_SCHEMA_ID,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "policy_id": EXECUTION_POLICY_ID,
        "authoritative_interval_axis": EXECUTION_INTERVAL_AXIS,
        "camera_frame_role": EXECUTION_CAMERA_FRAME_ROLE,
        "chaser_repositioning_ownership": CHASER_REPOSITIONING_OWNERSHIP,
        "protocol_trial_index_hash": snapshot.trial_index_sha256,
    }
    for key, expected in expected_scalars.items():
        if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
            raise ProtocolExecutionContractError(
                f"execution_index_json.{key} must be {expected!r}."
            )
    status = payload.get("status")
    if status not in {"complete", "interrupted", "invalid"}:
        raise ProtocolExecutionContractError("execution_index_json.status is unsupported.")
    if status == "invalid":
        raise ProtocolExecutionContractError(
            "Citrus marked the finalized protocol execution index invalid."
        )
    if "issues" in payload:
        issues = payload["issues"]
        if (
            not isinstance(issues, list)
            or any(type(issue) is not str or not issue for issue in issues)
        ):
            raise ProtocolExecutionContractError(
                "execution_index_json.issues must contain nonempty strings."
            )
        if issues:
            raise ProtocolExecutionContractError(
                "Non-invalid protocol execution index unexpectedly contains issues."
            )

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ProtocolExecutionContractError("execution_index_json.steps must be an array.")
    if len(raw_steps) > snapshot.step_count:
        raise ProtocolExecutionContractError("Execution is longer than its recipe.")
    if status == "complete" and len(raw_steps) != snapshot.step_count:
        raise ProtocolExecutionContractError(
            "Complete execution does not cover the exact recipe."
        )

    steps: list[ProtocolExecutionStep] = []
    any_interrupted = False
    for expected_index, raw_step in enumerate(raw_steps):
        name = f"execution_index_json.steps[{expected_index}]"
        if not isinstance(raw_step, Mapping):
            raise ProtocolExecutionContractError(f"{name} must be one object.")
        mode_id = snapshot.steps[expected_index].stimulus_mode_id
        required = {
            "step_index",
            "stimulus_mode_id",
            "completion_status",
            "end_reason",
            "interval",
        }
        optional = {"chaser_phases"} if mode_id == 12 else set()
        _exact_keys(raw_step, required=required, optional=optional, name=name)
        if _exact_int(raw_step.get("step_index"), name=f"{name}.step_index") != expected_index:
            raise ProtocolExecutionContractError(f"{name} is not the next recipe step.")
        if type(raw_step.get("stimulus_mode_id")) is not int or raw_step.get(
            "stimulus_mode_id"
        ) != mode_id:
            raise ProtocolExecutionContractError(f"{name} mode differs from its recipe.")
        completion_status = raw_step.get("completion_status")
        if completion_status not in {"completed", "interrupted"}:
            raise ProtocolExecutionContractError(f"{name}.completion_status is invalid.")
        any_interrupted = any_interrupted or completion_status == "interrupted"
        end_reason = raw_step.get("end_reason")
        if type(end_reason) is not str or not end_reason:
            raise ProtocolExecutionContractError(f"{name}.end_reason is empty.")
        step_interval = _interval(raw_step.get("interval"), name=f"{name}.interval")
        phases: Mapping[str, ProtocolStimulusFrameInterval] | None = None
        if mode_id == 12:
            raw_phases = raw_step.get("chaser_phases")
            if not isinstance(raw_phases, Mapping):
                raise ProtocolExecutionContractError(f"{name}.chaser_phases is missing.")
            _exact_keys(
                raw_phases,
                required=set(CHASER_PHASE_NAMES),
                name=f"{name}.chaser_phases",
            )
            parsed = {
                phase_name: _interval(
                    raw_phases[phase_name],
                    name=f"{name}.chaser_phases.{phase_name}",
                )
                for phase_name in CHASER_PHASE_NAMES
            }
            pre, training, post = (parsed[phase] for phase in CHASER_PHASE_NAMES)
            if not (
                pre.start_stimulus_frame_inclusive
                == step_interval.start_stimulus_frame_inclusive
                and pre.end_stimulus_frame_exclusive
                == training.start_stimulus_frame_inclusive
                and training.end_stimulus_frame_exclusive
                == post.start_stimulus_frame_inclusive
                and post.end_stimulus_frame_exclusive
                == step_interval.end_stimulus_frame_exclusive
            ):
                raise ProtocolExecutionContractError(
                    f"{name}.chaser_phases do not partition the exact step interval."
                )
            phases = MappingProxyType(parsed)
        steps.append(
            ProtocolExecutionStep(
                step_index=expected_index,
                stimulus_mode_id=mode_id,
                completion_status=completion_status,
                end_reason=end_reason,
                interval=step_interval,
                chaser_phases=phases,
            )
        )

    for previous, current in zip(steps, steps[1:]):
        if (
            current.interval.start_stimulus_frame_inclusive
            < previous.interval.end_stimulus_frame_exclusive
        ):
            raise ProtocolExecutionContractError(
                "Realized protocol step intervals overlap or are out of order."
            )
    if status == "complete" and any_interrupted:
        raise ProtocolExecutionContractError(
            "Complete execution contains an interrupted step."
        )
    if status == "interrupted" and not (
        any_interrupted or len(steps) < snapshot.step_count
    ):
        raise ProtocolExecutionContractError(
            "Interrupted execution has neither an interrupted step nor a recipe prefix."
        )
    return ProtocolExecutionIndex(
        execution_json=execution_json,
        execution_hash=execution_hash,
        protocol_trial_index_hash=snapshot.trial_index_sha256,
        status=status,
        steps=tuple(steps),
        payload=_freeze(payload),
    )


def read_protocol_execution_index(
    h5: h5py.File,
    *,
    snapshot: ProtocolSemanticSnapshot,
) -> ProtocolExecutionIndex:
    """Read one finalized v2 execution index; absence is a contract failure."""

    for path in (EXECUTION_GROUP_H5_PATH, EXECUTION_JSON_H5_PATH, EXECUTION_HASH_H5_PATH):
        if path not in h5:
            raise ProtocolExecutionContractError(
                f"Citrus snapshot v2 is missing finalized execution evidence {path}."
            )
    group = h5[EXECUTION_GROUP_H5_PATH]
    attrs = {
        "schema_id": _decode_text(
            group.attrs.get("schema_id"), name=f"{EXECUTION_GROUP_H5_PATH}@schema_id"
        ),
        "policy_id": _decode_text(
            group.attrs.get("policy_id"), name=f"{EXECUTION_GROUP_H5_PATH}@policy_id"
        ),
        "status": _decode_text(
            group.attrs.get("status"), name=f"{EXECUTION_GROUP_H5_PATH}@status"
        ),
    }
    raw_version = group.attrs.get("schema_version")
    if hasattr(raw_version, "item"):
        raw_version = raw_version.item()
    if type(raw_version) is not int:
        raise ProtocolExecutionContractError(
            "/protocol_execution@schema_version must be one exact integer."
        )
    result = validate_protocol_execution_index(
        execution_json=_decode_text(
            h5[EXECUTION_JSON_H5_PATH][()], name=EXECUTION_JSON_H5_PATH
        ),
        execution_hash=_decode_text(
            h5[EXECUTION_HASH_H5_PATH][()], name=EXECUTION_HASH_H5_PATH
        ),
        snapshot=snapshot,
    )
    if (
        attrs["schema_id"] != EXECUTION_SCHEMA_ID
        or raw_version != EXECUTION_SCHEMA_VERSION
        or attrs["policy_id"] != EXECUTION_POLICY_ID
        or attrs["status"] != result.status
    ):
        raise ProtocolExecutionContractError(
            "/protocol_execution attrs disagree with execution_index_json."
        )
    return result


def read_materialized_protocol_execution_index(
    run_group: Any,
    *,
    snapshot: ProtocolSemanticSnapshot,
) -> ProtocolExecutionIndex:
    """Reload exact execution bytes from one immutable Palette stimulus run."""

    group = run_group.get("protocol_execution")
    if group is None:
        raise ProtocolExecutionContractError(
            "Materialized snapshot v2 lacks protocol_execution."
        )
    attrs = group.attrs
    expected = {
        "schema_id": PALETTE_PROTOCOL_EXECUTION_SCHEMA_ID,
        "schema_version": PALETTE_PROTOCOL_EXECUTION_SCHEMA_VERSION,
        "source": "citrus_h5_protocol_execution",
        "source_schema_id": EXECUTION_SCHEMA_ID,
        "source_schema_version": EXECUTION_SCHEMA_VERSION,
        "source_policy_id": EXECUTION_POLICY_ID,
        "authoritative_interval_axis": EXECUTION_INTERVAL_AXIS,
        "camera_frame_role": EXECUTION_CAMERA_FRAME_ROLE,
        "acquisition_containment_status": (
            "unavailable_without_sealed_stimulus_to_acquisition_mapping"
        ),
    }
    for name, value in expected.items():
        if attrs.get(name) != value:
            raise ProtocolExecutionContractError(
                f"Materialized protocol_execution attr {name!r} is stale."
            )
    node = group.get("execution_index_json_utf8")
    if node is None:
        raise ProtocolExecutionContractError(
            "Materialized protocol_execution lacks exact UTF-8 bytes."
        )
    values = np.asarray(node[:])
    if values.ndim != 1 or values.dtype != np.dtype("u1") or values.size == 0:
        raise ProtocolExecutionContractError(
            "Materialized protocol execution bytes have an unsupported shape or dtype."
        )
    try:
        execution_json = values.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolExecutionContractError(
            "Materialized protocol execution bytes are not UTF-8."
        ) from exc
    execution_hash = attrs.get("execution_index_hash")
    if type(execution_hash) is not str:
        raise ProtocolExecutionContractError(
            "Materialized protocol execution hash is absent."
        )
    result = validate_protocol_execution_index(
        execution_json=execution_json,
        execution_hash=execution_hash,
        snapshot=snapshot,
    )
    if (
        attrs.get("status") != result.status
        or attrs.get("protocol_trial_index_hash")
        != result.protocol_trial_index_hash
    ):
        raise ProtocolExecutionContractError(
            "Materialized protocol_execution attrs disagree with its exact bytes."
        )
    return result


def build_protocol_frame_correspondence_proxy_payload(
    execution: ProtocolExecutionIndex,
    *,
    raw_stimulus: np.ndarray,
    raw_camera: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one deterministic sealed correspondence proxy payload.

    The returned row arrays preserve source-row order. Their step and phase
    membership is derived only from the producer's authoritative half-open
    ``stimulus_frame_num`` execution intervals. Camera IDs remain
    correspondence-only values.
    """

    if type(execution) is not ProtocolExecutionIndex:
        raise ProtocolExecutionContractError(
            "Protocol correspondence proxy requires a validated execution index."
        )
    raw_stimulus = np.asarray(raw_stimulus)
    raw_camera = np.asarray(raw_camera)
    if (
        raw_stimulus.ndim != 1
        or raw_camera.ndim != 1
        or raw_stimulus.shape != raw_camera.shape
        or raw_stimulus.size == 0
    ):
        raise ProtocolExecutionContractError(
            "Protocol frame correspondence requires nonempty aligned frame columns."
        )
    if (
        np.any(raw_stimulus < 0)
        or np.any(raw_camera < 0)
        or np.any(raw_stimulus > np.iinfo(np.int64).max)
        or np.any(raw_camera > np.iinfo(np.int64).max)
    ):
        raise ProtocolExecutionContractError(
            "Protocol frame correspondence values exceed int64."
        )
    stimulus = np.asarray(raw_stimulus, dtype=np.int64)
    camera = np.asarray(raw_camera, dtype=np.int64)
    step_index = np.full(stimulus.shape, -1, dtype=np.int32)
    phase_id = np.full(stimulus.shape, -1, dtype=np.int8)
    phase_ids = {name: index for index, name in enumerate(CHASER_PHASE_NAMES)}
    expected_frames = 0
    missing_frames = 0
    unique_stimulus = np.unique(stimulus)

    for realized in execution.steps:
        interval = realized.interval
        start = interval.start_stimulus_frame_inclusive
        end = interval.end_stimulus_frame_exclusive
        mask = (stimulus >= start) & (stimulus < end)
        if np.any(step_index[mask] != -1):
            raise ProtocolExecutionContractError(
                "Protocol execution intervals overlap frame metadata."
            )
        step_index[mask] = realized.step_index
        expected_frames += end - start
        observed_unique = int(
            np.count_nonzero((unique_stimulus >= start) & (unique_stimulus < end))
        )
        missing_frames += max(0, (end - start) - observed_unique)
        if realized.chaser_phases is not None:
            for phase_name in CHASER_PHASE_NAMES:
                phase = realized.chaser_phases[phase_name]
                phase_mask = (
                    (stimulus >= phase.start_stimulus_frame_inclusive)
                    & (stimulus < phase.end_stimulus_frame_exclusive)
                )
                if np.any(phase_id[phase_mask] != -1):
                    raise ProtocolExecutionContractError(
                        "Protocol chaser phase intervals overlap frame metadata."
                    )
                phase_id[phase_mask] = phase_ids[phase_name]

    arrays = {
        "stimulus_frame_num": stimulus,
        "camera_frame_id_correspondence": camera,
        "protocol_step_index": step_index,
        "chaser_phase_id": phase_id,
        "in_realized_protocol": step_index >= 0,
    }
    manifest = {
        "schema_id": "palette.protocol_frame_correspondence_proxy.v1",
        "schema_version": 1,
        "mapping_class": "sealed_derived_correspondence_proxy",
        "selector_eligible": False,
        "scientific_use_class": "visualization_and_exploratory_alignment_only",
        "authoritative_source_axis": "stimulus_frame_num",
        "camera_frame_role": "correspondence_only",
        "acquisition_join_status": (
            "unavailable_without_sealed_stimulus_to_acquisition_mapping"
        ),
        "source_frame_metadata_path": "video_metadata/frame_metadata",
        "protocol_execution_hash": execution.execution_hash,
        "phase_id_mapping": phase_ids,
        "row_count": int(stimulus.size),
        "expected_realized_stimulus_frame_count": int(expected_frames),
        "missing_realized_stimulus_frame_count": int(missing_frames),
        "duplicate_stimulus_frame_row_count": int(
            stimulus.size - unique_stimulus.size
        ),
        "arrays": {
            name: {
                "dtype": np.asarray(values).dtype.str,
                "shape": [int(item) for item in np.asarray(values).shape],
                "content_sha256": identity_array_content_sha256(values),
            }
            for name, values in arrays.items()
        },
    }
    return arrays, manifest


__all__ = [
    "CHASER_PHASE_NAMES",
    "EXECUTION_CAMERA_FRAME_ROLE",
    "EXECUTION_HASH_H5_PATH",
    "EXECUTION_INTERVAL_AXIS",
    "EXECUTION_JSON_H5_PATH",
    "EXECUTION_POLICY_ID",
    "PALETTE_PROTOCOL_EXECUTION_SCHEMA_ID",
    "PALETTE_PROTOCOL_EXECUTION_SCHEMA_VERSION",
    "ProtocolExecutionContractError",
    "ProtocolExecutionIndex",
    "ProtocolExecutionStep",
    "ProtocolStimulusFrameInterval",
    "build_protocol_frame_correspondence_proxy_payload",
    "read_protocol_execution_index",
    "read_materialized_protocol_execution_index",
    "validate_protocol_execution_index",
]

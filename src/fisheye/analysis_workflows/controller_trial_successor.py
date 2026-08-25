"""Exact controller-trial successor over the common chaser-relative row axis.

The legacy escape/freeze analyses reconstructed trials independently and could
fall back to contiguous active-state segments.  This successor deliberately
does neither.  It accepts only explicitly valid logged trial IDs, preserves
the exact active row membership, represents holes inside a trial envelope as
gap evidence, and assigns a deterministic ordinal independently for each
chaser.

The result is an in-memory, selector-ineligible candidate.  Publication is a
separate boundary; no selector, registry row, or production authority is
changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCHEMA_ID = "palette.analysis.controller_chase_trials"
SCHEMA_VERSION = 1
PREPARED_SCHEMA_ID = "palette.analysis.controller_chase_trials.prepared_successor"
PREPARED_SCHEMA_VERSION = 1
METHOD_ID = "exact_logged_trial_id_active_membership_v1"

SEMANTIC_ROLE_CODES = MappingProxyType(
    {"chaser_pre": 1, "chaser_training": 2, "chaser_post": 3}
)

TRIGGER_SOURCE_FIRST_LOGGED_ACTIVE_ROW = 1

TRIAL_GAP_REASON_NOT_GAP = 0
TRIAL_GAP_REASON_SELECTION_NONMEMBER = 1
TRIAL_GAP_REASON_CHASER_OCCURRENCE_UNAVAILABLE = 2
TRIAL_GAP_REASON_ACTIVE_STATE_UNAVAILABLE = 3
TRIAL_GAP_REASON_EXPLICITLY_INACTIVE = 4
TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE = 5
TRIAL_GAP_REASON_TRIAL_ID_MISMATCH = 6


class ControllerTrialSuccessorError(ValueError):
    """Raised when logged controller trials cannot be materialized exactly."""


def _fail(message: str) -> None:
    raise ControllerTrialSuccessorError(message)


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array(value: Any, *, name: str, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype) or result.shape != shape:
        _fail(
            f"{name} must have exact dtype {np.dtype(dtype).str!r} and "
            f"shape {shape!r}; got {result.dtype.str!r} and {result.shape!r}."
        )
    return result


@dataclass(frozen=True, slots=True)
class ControllerTrialInput:
    """Typed arrays copied from one exact chaser-relative-frame source."""

    recording_id: str
    source_run_path: str
    source_manifest_sha256: str
    n_frames: int
    n_chasers: int
    acquisition_frame_id: np.ndarray
    timestamp_ns: np.ndarray
    timestamp_valid: np.ndarray
    chaser_identity_code: np.ndarray
    selection_member: np.ndarray
    chaser_occurrence_member: np.ndarray
    trial_id: np.ndarray
    trial_valid: np.ndarray
    active_state_code: np.ndarray
    active_state_valid: np.ndarray
    semantic_selection_binding: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PreparedControllerTrials:
    """Exact trial table plus dense membership evidence."""

    recording_id: str
    n_frames: int
    n_chasers: int
    n_trials: int
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as exc:
            raise KeyError(f"Unknown controller-trial array {name!r}.") from exc

    @property
    def payload_digest(self) -> str:
        return str(self.manifest["payload_digest"])


def _validate_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{name} must be one non-empty exact string.")
    return value


def _validate_digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _declarations(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "dtype": np.asarray(array).dtype.str,
            "shape": list(np.asarray(array).shape),
            "content_sha256": array_values_sha256(np.asarray(array)),
        }
        for name, array in sorted(arrays.items())
    ]


def _semantic_binding(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    plain = _plain(value)
    if not isinstance(plain, dict):  # pragma: no cover - guarded by typing
        _fail("semantic_selection_binding must be one JSON object.")
    required = {
        "run_name",
        "run_path",
        "manifest_sha256",
        "selection_identity_sha256",
        "protocol_semantic_hash",
        "trial_index_integrity_status",
        "roles",
        "semantic_role_bindings",
    }
    if not required.issubset(plain):
        _fail("semantic_selection_binding lacks exact semantic identity fields.")
    if plain.get("selector_eligible") is not False or plain.get(
        "production_authority"
    ) is not False:
        _fail("Semantic source must remain selector-ineligible and non-authoritative.")
    if list(plain.get("roles", ())) != [
        "chaser_pre",
        "chaser_training",
        "chaser_post",
    ]:
        _fail("Semantic source roles are missing, duplicated, or reordered.")
    return plain


def prepare_controller_trial_successor(
    source: ControllerTrialInput,
) -> PreparedControllerTrials:
    """Materialize exact logged chase trials without fallback segmentation."""

    if type(source) is not ControllerTrialInput:
        raise TypeError("source must be one ControllerTrialInput.")
    recording_id = _validate_text(source.recording_id, name="recording_id")
    source_run_path = _validate_text(source.source_run_path, name="source_run_path")
    source_digest = _validate_digest(
        source.source_manifest_sha256,
        name="source_manifest_sha256",
    )
    if type(source.n_frames) is not int or source.n_frames < 0:
        _fail("n_frames must be one non-negative exact integer.")
    if type(source.n_chasers) is not int or source.n_chasers <= 0:
        _fail("n_chasers must be one positive exact integer.")
    n_rows = source.n_frames * source.n_chasers
    shape = (n_rows,)
    acquisition = _array(
        source.acquisition_frame_id,
        name="acquisition_frame_id",
        dtype=np.int64,
        shape=shape,
    )
    timestamp = _array(
        source.timestamp_ns,
        name="timestamp_ns",
        dtype=np.int64,
        shape=shape,
    )
    timestamp_valid = _array(
        source.timestamp_valid,
        name="timestamp_valid",
        dtype=bool,
        shape=shape,
    )
    chaser_code = _array(
        source.chaser_identity_code,
        name="chaser_identity_code",
        dtype=np.uint16,
        shape=shape,
    )
    selection = _array(
        source.selection_member,
        name="selection_member",
        dtype=bool,
        shape=shape,
    )
    occurrence = _array(
        source.chaser_occurrence_member,
        name="chaser_occurrence_member",
        dtype=bool,
        shape=shape,
    )
    trial_id = _array(
        source.trial_id,
        name="trial_id",
        dtype=np.int64,
        shape=shape,
    )
    trial_valid = _array(
        source.trial_valid,
        name="trial_valid",
        dtype=bool,
        shape=shape,
    )
    active_code = _array(
        source.active_state_code,
        name="active_state_code",
        dtype=np.uint8,
        shape=shape,
    )
    active_valid = _array(
        source.active_state_valid,
        name="active_state_valid",
        dtype=bool,
        shape=shape,
    )
    if np.any(active_code > 1):
        _fail("active_state_code must use the exact binary registry 0/1.")
    if np.any(trial_valid & (trial_id <= 0)):
        _fail("A valid logged trial ID must be strictly positive.")

    eligible = selection & occurrence
    if np.any(eligible & ~active_valid):
        _fail(
            "An eligible controller row lacks an explicit logged active-state "
            "value; legacy active-interval reconstruction is prohibited."
        )
    active = eligible & active_valid & (active_code == 1)
    unresolved_active_trial_id = active & ~trial_valid
    if np.any(unresolved_active_trial_id):
        _fail(
            "An explicitly active controller row lacks a strictly positive "
            "producer-logged trial ID; legacy contiguous-interval trial "
            "reconstruction is prohibited."
        )
    member = active & trial_valid
    semantic_binding = _semantic_binding(source.semantic_selection_binding)

    # One source frame is repeated for every chaser.  The flattened source row
    # is still the primary mapping authority because membership is per chaser.
    expected_codes = chaser_code.reshape(source.n_frames, source.n_chasers)
    if source.n_frames and np.any(expected_codes != expected_codes[:1, :]):
        _fail("Chaser identity codes changed along the declared chaser axis.")
    frames = acquisition.reshape(source.n_frames, source.n_chasers)
    if source.n_frames and np.any(frames != frames[:, :1]):
        _fail("Acquisition-frame identity is not repeated across the chaser axis.")

    keys = sorted(
        {
            (int(chaser_code[row]), int(trial_id[row]))
            for row in np.flatnonzero(member)
        },
        key=lambda key: (
            int(np.flatnonzero(member & (chaser_code == key[0]) & (trial_id == key[1]))[0]),
            key[0],
            key[1],
        ),
    )
    trial_count = len(keys)
    table: dict[str, np.ndarray] = {
        "trial_row_id": np.arange(trial_count, dtype=np.int64),
        "chaser_identity_code": np.zeros(trial_count, dtype=np.uint16),
        "logged_trial_id": np.zeros(trial_count, dtype=np.int64),
        "trial_ordinal": np.zeros(trial_count, dtype=np.int32),
        "start_source_frame_row": np.zeros(trial_count, dtype=np.int64),
        "end_source_frame_row_exclusive": np.zeros(trial_count, dtype=np.int64),
        "start_acquisition_frame_id": np.zeros(trial_count, dtype=np.int64),
        "end_acquisition_frame_id_inclusive": np.zeros(trial_count, dtype=np.int64),
        "trigger_acquisition_frame_id": np.zeros(trial_count, dtype=np.int64),
        "trigger_timestamp_ns": np.zeros(trial_count, dtype=np.int64),
        "trigger_timestamp_valid": np.zeros(trial_count, dtype=bool),
        "active_member_count": np.zeros(trial_count, dtype=np.int64),
        "envelope_frame_count": np.zeros(trial_count, dtype=np.int64),
        "gap_frame_count": np.zeros(trial_count, dtype=np.int64),
        "gap_fraction": np.full(trial_count, np.nan, dtype=np.float64),
        "trigger_source_code": np.full(
            trial_count,
            TRIGGER_SOURCE_FIRST_LOGGED_ACTIVE_ROW,
            dtype=np.uint8,
        ),
        "fallback_used": np.zeros(trial_count, dtype=bool),
    }
    dense_trial_row = np.full(n_rows, -1, dtype=np.int64)
    envelope_trial_row = np.full(n_rows, -1, dtype=np.int64)
    envelope_member = np.zeros(n_rows, dtype=bool)
    gap_member = np.zeros(n_rows, dtype=bool)
    gap_reason = np.full(n_rows, TRIAL_GAP_REASON_NOT_GAP, dtype=np.uint8)
    ordinal_by_chaser: dict[int, int] = {}

    for out_row, (code, logged_id) in enumerate(keys):
        rows = np.flatnonzero(member & (chaser_code == code) & (trial_id == logged_id))
        if rows.size == 0:  # pragma: no cover - constructed from rows above
            _fail("Internal controller-trial grouping lost a logged member.")
        frame_rows = rows // source.n_chasers
        start_frame_row = int(frame_rows.min())
        end_frame_row = int(frame_rows.max()) + 1
        chaser_axis = int(rows[0] % source.n_chasers)
        envelope_rows = (
            np.arange(start_frame_row, end_frame_row, dtype=np.int64)
            * source.n_chasers
            + chaser_axis
        )
        if np.any(chaser_code[envelope_rows] != code):
            _fail("Trial envelope crosses a changed chaser identity.")
        if np.any(envelope_trial_row[envelope_rows] >= 0):
            _fail(
                "Logged trial envelopes overlap for one chaser; a source row "
                "cannot have two visualization/censoring trial identities."
            )
        ordinal_by_chaser[code] = ordinal_by_chaser.get(code, 0) + 1
        table["chaser_identity_code"][out_row] = code
        table["logged_trial_id"][out_row] = logged_id
        table["trial_ordinal"][out_row] = ordinal_by_chaser[code]
        table["start_source_frame_row"][out_row] = start_frame_row
        table["end_source_frame_row_exclusive"][out_row] = end_frame_row
        table["start_acquisition_frame_id"][out_row] = acquisition[rows[0]]
        table["end_acquisition_frame_id_inclusive"][out_row] = acquisition[rows[-1]]
        table["trigger_acquisition_frame_id"][out_row] = acquisition[rows[0]]
        table["trigger_timestamp_ns"][out_row] = timestamp[rows[0]]
        table["trigger_timestamp_valid"][out_row] = timestamp_valid[rows[0]]
        table["active_member_count"][out_row] = rows.size
        table["envelope_frame_count"][out_row] = envelope_rows.size
        table["gap_frame_count"][out_row] = envelope_rows.size - rows.size
        table["gap_fraction"][out_row] = (
            float(envelope_rows.size - rows.size) / float(envelope_rows.size)
        )
        dense_trial_row[rows] = out_row
        envelope_trial_row[envelope_rows] = out_row
        envelope_member[envelope_rows] = True
        exact_current_member = (
            member[envelope_rows]
            & (chaser_code[envelope_rows] == code)
            & (trial_id[envelope_rows] == logged_id)
        )
        gaps = envelope_rows[~exact_current_member]
        gap_member[gaps] = True
        for gap in gaps.tolist():
            if not selection[gap]:
                reason = TRIAL_GAP_REASON_SELECTION_NONMEMBER
            elif not occurrence[gap]:
                reason = TRIAL_GAP_REASON_CHASER_OCCURRENCE_UNAVAILABLE
            elif not active_valid[gap]:
                reason = TRIAL_GAP_REASON_ACTIVE_STATE_UNAVAILABLE
            elif active_code[gap] == 0:
                reason = TRIAL_GAP_REASON_EXPLICITLY_INACTIVE
            elif not trial_valid[gap]:
                reason = TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE
            elif trial_id[gap] != logged_id:
                reason = TRIAL_GAP_REASON_TRIAL_ID_MISMATCH
            else:  # pragma: no cover - exact membership handled above
                _fail("A trial-envelope gap has no auditable exclusion reason.")
            gap_reason[gap] = reason

    if np.any(member & (dense_trial_row < 0)):
        _fail("A logged active member did not attach to exactly one trial.")
    if np.any(gap_member & member):
        _fail("A trial row cannot be both an active member and a preserved gap.")
    if np.any(envelope_member != (envelope_trial_row >= 0)):
        _fail("Trial-envelope membership and row identity disagree.")
    if np.any(gap_member != (gap_reason != TRIAL_GAP_REASON_NOT_GAP)):
        _fail("Trial-gap membership and reason evidence disagree.")

    arrays: dict[str, np.ndarray] = {
        **table,
        "source_relative_row_id": np.arange(n_rows, dtype=np.int64),
        "trial_row_id_by_source_row": dense_trial_row,
        "trial_envelope_row_id_by_source_row": envelope_trial_row,
        "logged_active_trial_member": member,
        "trial_envelope_member": envelope_member,
        "trial_gap_member": gap_member,
        "trial_gap_reason_code_by_source_row": gap_reason,
        "logged_active_trial_id_unavailable": unresolved_active_trial_id,
    }
    readonly = {name: _readonly(value) for name, value in arrays.items()}
    declarations = _declarations(readonly)
    manifest_body: dict[str, Any] = {
        "schema_id": PREPARED_SCHEMA_ID,
        "schema_version": PREPARED_SCHEMA_VERSION,
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "trial_unit": "exact_logged_controller_trial",
            "membership_unit": "relative_frame_row_x_chaser",
        },
        "recording_id": recording_id,
        "source_relative_frame": {
            "run_path": source_run_path,
            "manifest_sha256": source_digest,
        },
        "semantic_selection": semantic_binding,
        "dimensions": {
            "n_frames": source.n_frames,
            "n_chasers": source.n_chasers,
            "n_source_rows": n_rows,
            "n_trials": trial_count,
        },
        "policy": {
            "trial_identity": "exact_strictly_positive_logged_trial_id_per_chaser",
            "active_membership": "explicit_valid_active_state_code_equals_1",
            "fallback": "prohibited_fail_closed",
            "legacy_contiguous_interval_reconstruction": "rejected",
            "gap_policy": "preserve_nonmember_rows_inside_first_last_active_envelope",
            "gap_reason_precedence": [
                "semantic_selection_nonmember",
                "chaser_occurrence_unavailable",
                "controller_active_state_unavailable",
                "explicit_controller_inactive",
                "logged_trial_id_unavailable",
                "logged_trial_id_mismatch",
            ],
            "unresolved_active_rows": "hard_error",
            "trigger_selection": "first_logged_active_member",
            "trial_ordinal": "one_based_onset_order_independent_per_chaser",
        },
        "identity_registries": {
            "trigger_source": {
                str(TRIGGER_SOURCE_FIRST_LOGGED_ACTIVE_ROW): (
                    "first_logged_active_member"
                )
            },
            "trial_gap_reason": {
                str(TRIAL_GAP_REASON_NOT_GAP): "not_a_trial_gap",
                str(TRIAL_GAP_REASON_SELECTION_NONMEMBER): (
                    "semantic_selection_nonmember"
                ),
                str(TRIAL_GAP_REASON_CHASER_OCCURRENCE_UNAVAILABLE): (
                    "chaser_occurrence_unavailable"
                ),
                str(TRIAL_GAP_REASON_ACTIVE_STATE_UNAVAILABLE): (
                    "controller_active_state_unavailable"
                ),
                str(TRIAL_GAP_REASON_EXPLICITLY_INACTIVE): (
                    "explicit_controller_inactive"
                ),
                str(TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE): (
                    "logged_trial_id_unavailable"
                ),
                str(TRIAL_GAP_REASON_TRIAL_ID_MISMATCH): (
                    "logged_trial_id_mismatch"
                ),
            },
        },
        "array_declarations": declarations,
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    payload = canonical_json_sha256(manifest_body)
    manifest = _freeze({**manifest_body, "payload_digest": payload})
    return PreparedControllerTrials(
        recording_id=recording_id,
        n_frames=source.n_frames,
        n_chasers=source.n_chasers,
        n_trials=trial_count,
        arrays=MappingProxyType(readonly),
        manifest=manifest,
    )


def controller_trial_input_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
) -> ControllerTrialInput:
    """Bind the strict relative-frame and semantic-selection handles.

    The imports are local to keep this pure module inexpensive to import and to
    make the fail-closed type boundary explicit.
    """

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be a strict loader-minted handle.")
    relative_frame.assert_current()
    semantic_selection.assert_current()
    if relative_frame.recording_id != semantic_selection.recording_id:
        _fail("Relative-frame and semantic-selection recordings differ.")
    required = {
        "trial_id",
        "trial_valid",
        "active_state_code",
        "active_state_valid",
        "timestamp_ns",
        "timestamp_valid",
    }
    missing = sorted(required - set(relative_frame.base_arrays))
    if missing:
        _fail(f"Relative-frame source lacks controller trial arrays: {missing!r}.")
    semantic_binding = semantic_selection.source_binding()
    role_codes = semantic_role_codes_from_handles(relative_frame, semantic_selection)
    selection_by_frame = role_codes != 0
    selection_member = np.broadcast_to(
        selection_by_frame[:, np.newaxis],
        (relative_frame.n_frames, relative_frame.n_chasers),
    ).reshape(-1)
    return ControllerTrialInput(
        recording_id=relative_frame.recording_id,
        source_run_path=relative_frame.run_path,
        source_manifest_sha256=relative_frame.manifest_sha256,
        n_frames=relative_frame.n_frames,
        n_chasers=relative_frame.n_chasers,
        acquisition_frame_id=relative_frame.base_array("acquisition_frame_id"),
        timestamp_ns=relative_frame.base_array("timestamp_ns"),
        timestamp_valid=relative_frame.base_array("timestamp_valid"),
        chaser_identity_code=relative_frame.base_array("chaser_identity_code"),
        selection_member=selection_member,
        chaser_occurrence_member=relative_frame.base_array(
            "chaser_occurrence_member"
        ),
        trial_id=relative_frame.base_array("trial_id"),
        trial_valid=relative_frame.base_array("trial_valid"),
        active_state_code=relative_frame.base_array("active_state_code"),
        active_state_valid=relative_frame.base_array("active_state_valid"),
        semantic_selection_binding=semantic_binding,
    )


def semantic_role_codes_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
) -> np.ndarray:
    """Project exact semantic intervals onto one acquisition-frame axis.

    The relative-frame publication represents all acquisition rows available
    from its exact provider source.  It is not expected to have been produced
    from the later protocol-semantic epoch publication.  Membership is
    therefore an exact join by acquisition frame ID, not equality between two
    unrelated source-context records.
    """

    from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
        ChaserRelativeFrameSourceHandle,
    )
    from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
        ProtocolSemanticChaserSelectionSourceHandle,
    )

    if type(relative_frame) is not ChaserRelativeFrameSourceHandle:
        raise TypeError("relative_frame must be a strict loader-minted handle.")
    if type(semantic_selection) is not ProtocolSemanticChaserSelectionSourceHandle:
        raise TypeError("semantic_selection must be a strict loader-minted handle.")
    relative_frame.assert_current()
    semantic_selection.assert_current()
    if relative_frame.recording_id != semantic_selection.recording_id:
        _fail("Relative-frame and semantic-selection recordings differ.")

    acquisition = relative_frame.base_frame_chaser("acquisition_frame_id")
    if relative_frame.n_frames and not np.all(acquisition == acquisition[:, :1]):
        _fail("Relative-frame acquisition identity is not constant across chasers.")
    frame_ids = (
        np.asarray(acquisition[:, 0], dtype=np.int64)
        if relative_frame.n_frames
        else np.asarray([], dtype=np.int64)
    )
    codes = np.zeros(relative_frame.n_frames, dtype=np.uint8)
    records = semantic_selection.role_records
    if tuple(records) != tuple(SEMANTIC_ROLE_CODES):
        _fail("Semantic selection does not have the exact chaser role order.")
    for role_name, role_code in SEMANTIC_ROLE_CODES.items():
        record = records[role_name]
        start = record.get("selected_start_frame")
        end = record.get("selected_end_frame_exclusive")
        if type(start) is not int or type(end) is not int or end <= start:
            _fail(f"Semantic role {role_name!r} has invalid exact frame bounds.")
        membership = (frame_ids >= start) & (frame_ids < end)
        if np.any(membership & (codes != 0)):
            _fail("Semantic role bounds overlap on the acquisition frame axis.")
        codes[membership] = role_code
    codes.setflags(write=False)
    return codes


def prepare_controller_trial_successor_from_handles(
    relative_frame: Any,
    semantic_selection: Any,
) -> PreparedControllerTrials:
    """Prepare a controller-trial successor from current exact source handles."""

    return prepare_controller_trial_successor(
        controller_trial_input_from_handles(relative_frame, semantic_selection)
    )


__all__ = [
    "METHOD_ID",
    "PREPARED_SCHEMA_ID",
    "PREPARED_SCHEMA_VERSION",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SEMANTIC_ROLE_CODES",
    "TRIAL_GAP_REASON_ACTIVE_STATE_UNAVAILABLE",
    "TRIAL_GAP_REASON_CHASER_OCCURRENCE_UNAVAILABLE",
    "TRIAL_GAP_REASON_EXPLICITLY_INACTIVE",
    "TRIAL_GAP_REASON_NOT_GAP",
    "TRIAL_GAP_REASON_SELECTION_NONMEMBER",
    "TRIAL_GAP_REASON_TRIAL_ID_MISMATCH",
    "TRIAL_GAP_REASON_TRIAL_ID_UNAVAILABLE",
    "ControllerTrialInput",
    "ControllerTrialSuccessorError",
    "PreparedControllerTrials",
    "controller_trial_input_from_handles",
    "prepare_controller_trial_successor",
    "prepare_controller_trial_successor_from_handles",
    "semantic_role_codes_from_handles",
]

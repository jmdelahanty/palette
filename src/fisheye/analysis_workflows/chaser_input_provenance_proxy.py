"""In-memory projection of native chaser states onto acquisition frames.

The Citrus native stimulus source is the immutable evidence surface.  This
module creates an explicitly exploratory proxy view of that evidence for
camera-frame analyses.  A native sample is eligible only when every declared
chaser is present, valid, and finite in that same sample row.  The selector
never combines chasers from different native rows and never carries a state
forward over an acquisition frame for which no native sample was recorded.

This is intentionally not a Zarr writer, selector resolver, or presentation
time reconstruction.  The result is an in-memory, read-only snapshot whose
provenance states that the selected controller state is input-acquisition
provenance, not verified displayed geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.chaser_input_provenance_proxy_contract import (
    BEHAVIORAL_DENOMINATOR,
    CAMERA_EXPOSURE_REFERENCE,
    CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
    ChaserInputProvenanceProxyResult,
    PHYSICAL_PRESENTATION_VERIFIED,
    PRESENTATION_TIMESTAMP_AVAILABLE,
    PROJECTION_RECORD_SCHEMA_ID,
    PROJECTION_RECORD_SCHEMA_VERSION,
    PROXY_POLICY_ID,
    SCIENTIFIC_USE_CLASS,
    TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT,
)

SELECTION_REASON_CODES = (
    "selected",
    "no_complete_chaser_sample",
)
CANDIDATE_REASON_CODES = (
    "complete",
    "incomplete_chaser_sample",
)


class ChaserInputProvenanceProxyError(ValueError):
    """Raised when native chaser evidence cannot be projected safely."""


def _error(message: str) -> None:
    raise ChaserInputProvenanceProxyError(message)


def _readonly_copy(value: object, *, label: str) -> np.ndarray:
    try:
        array = np.array(value, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        _error(f"{label} cannot be copied into a NumPy array: {exc}.")
    array.setflags(write=False)
    return array


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _error(f"{label} must be one non-empty exact string.")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _error(f"{label} must be one lowercase SHA-256 digest.")
    return text


def _require_int64(array: np.ndarray, *, label: str) -> None:
    if array.dtype != np.dtype("<i8") or array.ndim != 1:
        _error(f"{label} must be one-dimensional little-endian int64.")


def _require_int64_matrix(array: np.ndarray, *, label: str, shape: tuple[int, int]) -> None:
    if array.dtype != np.dtype("<i8") or array.shape != shape:
        _error(f"{label} must have shape {shape} and little-endian int64 dtype.")


def _require_unique(array: np.ndarray, *, label: str) -> None:
    if np.unique(array).size != array.size:
        _error(f"{label} contains duplicate values.")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _walk_mapping(value: Mapping[str, Any]):
    for key, item in value.items():
        yield str(key), item
        if isinstance(item, Mapping):
            yield from _walk_mapping(item)


def _mapping_attr(source: object, *names: str) -> Mapping[str, Any] | None:
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, Mapping):
            return value
    return None


def _source_array(source: object, *names: str, label: str) -> np.ndarray:
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if callable(value):
                continue
            return _readonly_copy(value, label=label)
    _error(f"Native source handle is missing {label}.")


def _authority_digest(authority: Mapping[str, Any], source: object) -> str:
    value = getattr(source, "source_authority_digest", None)
    if type(value) is str and value:
        return value
    # The native source handle's manifest and verification digests identify the
    # publication, not the nested authority record itself.  When the handle
    # does not expose a dedicated authority digest, bind this compact record to
    # the canonical readable authority mapping.
    try:
        encoded = json.dumps(
            _jsonable(authority),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _error(f"Native source authority is not canonical JSON: {exc}.")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _error(f"Projection record is not canonical JSON: {exc}.")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _NativeSnapshot:
    """Copied and validated native arrays used by the pure selector."""

    source_handle: object = field(repr=False, compare=False)
    recording_id: str
    source_authority_id: str
    source_authority_digest: str
    source_manifest_sha256: str
    source_verification_digest: str
    source_run_path: str
    total_frames: int
    stimulus_frame_num: np.ndarray
    timestamp_ns_session: np.ndarray
    source_acquisition_frame_index: np.ndarray
    source_sample_row_index: np.ndarray
    source_stimulus_run_row_index: np.ndarray
    source_stimulus_source_row_index: np.ndarray
    chaser_index: np.ndarray
    chaser_position_xy: np.ndarray
    chaser_valid: np.ndarray
    authority: Mapping[str, Any]


def _snapshot_source(source: object) -> _NativeSnapshot:
    if source is None:
        _error("A native stimulus-sample source handle is required.")

    verifier = getattr(source, "assert_verified", None)
    if not callable(verifier):
        _error("Native source handle does not expose assert_verified().")
    try:
        verifier()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        _error(f"Native source handle verification failed: {exc}.")

    recording_id = _text(getattr(source, "recording_id", None), label="recording_id")
    authority = _mapping_attr(source, "source_authority", "authorities")
    if authority is None:
        _error("Native source handle is missing its source authority mapping.")
    nested_recording_ids = [
        value for key, value in _walk_mapping(authority) if key == "recording_id"
    ]
    if any(value != recording_id for value in nested_recording_ids):
        _error("Native source contains mixed recording identities in its authority.")

    source_authority_id = getattr(source, "source_authority_id", None)
    if source_authority_id is None:
        source_authority_id = authority.get("schema_id")
    source_authority_id = _text(source_authority_id, label="source_authority_id")
    source_authority_digest = _digest(
        _authority_digest(authority, source), label="source_authority_digest"
    )
    source_manifest_sha256 = _digest(
        getattr(source, "manifest_sha256", None), label="source_manifest_sha256"
    )
    source_verification_digest = _digest(
        getattr(source, "verification_digest", None),
        label="source_verification_digest",
    )
    source_run_path = _text(
        getattr(source, "run_path", None), label="source_run_path"
    )

    dimensions = getattr(source, "dimensions", None)
    total_frames = getattr(dimensions, "total_frames", None)
    if type(total_frames) is not int or total_frames <= 0:
        _error("Native source handle must declare a positive total_frames value.")

    stimulus_frame_num = _source_array(
        source, "stimulus_frame_num", label="stimulus_frame_num"
    )
    timestamp_ns_session = _source_array(
        source,
        "timestamp_ns_session",
        "timestamp_ns",
        label="timestamp_ns_session",
    )
    acquisition = _source_array(
        source,
        "source_acquisition_frame_index",
        label="source_acquisition_frame_index",
    )
    sample_row = _source_array(
        source, "source_sample_row_index", "native_sample_row_index", label="source_sample_row_index"
    ) if hasattr(source, "source_sample_row_index") or hasattr(source, "native_sample_row_index") else np.arange(
        stimulus_frame_num.size, dtype=np.int64
    )
    source_run_row = _source_array(
        source,
        "source_stimulus_run_row_index",
        label="source_stimulus_run_row_index",
    )
    source_source_row = _source_array(
        source,
        "source_stimulus_source_row_index",
        label="source_stimulus_source_row_index",
    )
    chaser_index = _source_array(source, "chaser_index", label="chaser_index")
    chaser_position = _source_array(
        source,
        "chaser_position_xy",
        "chaser_position_arena_xy",
        label="chaser_position_xy",
    )
    chaser_valid = _source_array(source, "chaser_valid", label="chaser_valid")

    n_samples = int(stimulus_frame_num.size)
    if n_samples <= 0:
        _error("Native stimulus sample axis must not be empty.")
    _require_int64(stimulus_frame_num, label="stimulus_frame_num")
    _require_int64(timestamp_ns_session, label="timestamp_ns_session")
    _require_int64(acquisition, label="source_acquisition_frame_index")
    _require_int64(sample_row, label="source_sample_row_index")
    _require_unique(sample_row, label="source_sample_row_index")
    if np.any(sample_row < 0):
        _error("source_sample_row_index must be nonnegative.")
    if np.any(np.diff(sample_row) <= 0):
        _error("Native source sample rows are not strictly ordered.")
    if np.any(np.diff(stimulus_frame_num) <= 0):
        _error("Native stimulus_frame_num is not strictly ordered.")
    if np.any(timestamp_ns_session < 0):
        _error("timestamp_ns_session must be nonnegative session time.")
    if np.any(np.diff(timestamp_ns_session) < 0):
        _error("Native timestamp_ns_session ordering is malformed.")
    if np.any(np.diff(acquisition) < 0):
        _error("Native source_acquisition_frame_index ordering is malformed.")
    if np.any(acquisition < 0) or np.any(acquisition >= total_frames):
        _error("source_acquisition_frame_index leaves the declared recording domain.")

    if source_run_row.shape != (n_samples,):
        # The native handle stores one source row per sample and chaser.  The
        # flattened form is accepted only for a one-chaser source.
        if source_run_row.ndim != 2:
            _error("source_stimulus_run_row_index must preserve sample-by-chaser rows.")
    if source_run_row.ndim != 2 or source_source_row.ndim != 2:
        _error("Stimulus source-row lineage must be two-dimensional.")
    n_chasers = int(chaser_index.size)
    if (
        chaser_index.dtype != np.dtype("<i2")
        or chaser_index.ndim != 1
        or n_chasers <= 0
    ):
        _error("Declared chaser identity axis must be non-empty little-endian int16.")
    expected_pair = (n_samples, n_chasers)
    if source_run_row.shape != expected_pair or source_source_row.shape != expected_pair:
        _error("Stimulus source-row lineage is not aligned to all declared chasers.")
    _require_int64_matrix(source_run_row, label="source_stimulus_run_row_index", shape=expected_pair)
    _require_int64_matrix(source_source_row, label="source_stimulus_source_row_index", shape=expected_pair)
    if np.any(source_run_row < 0) or np.any(source_source_row < 0):
        _error("Stimulus source-row lineage must be nonnegative.")
    if np.unique(source_run_row).size != source_run_row.size:
        _error("Stimulus source run-row lineage contains mixed or duplicate authority rows.")
    if np.unique(chaser_index).size != n_chasers or np.any(chaser_index < 0):
        _error("Declared chaser identity axis is not unique and nonnegative.")
    if chaser_position.shape != expected_pair + (2,):
        _error("Chaser coordinates must have shape [sample, chaser, 2].")
    if not np.issubdtype(chaser_position.dtype, np.number):
        _error("Chaser coordinates must be numeric.")
    if chaser_valid.shape != expected_pair or chaser_valid.dtype != np.dtype(bool):
        _error("Chaser validity must be bool with shape [sample, chaser].")
    if any(
        not np.all(np.isfinite(chaser_position[row][chaser_valid[row]]))
        for row in range(n_samples)
    ):
        _error("A chaser coordinate marked valid is nonfinite.")
    for array, label in (
        (timestamp_ns_session, "timestamp_ns_session"),
        (acquisition, "source_acquisition_frame_index"),
        (sample_row, "source_sample_row_index"),
    ):
        if array.shape != (n_samples,):
            _error(f"{label} is not aligned to the native stimulus sample axis.")

    return _NativeSnapshot(
        source_handle=source,
        recording_id=recording_id,
        source_authority_id=source_authority_id,
        source_authority_digest=source_authority_digest,
        source_manifest_sha256=source_manifest_sha256,
        source_verification_digest=source_verification_digest,
        source_run_path=source_run_path,
        total_frames=total_frames,
        stimulus_frame_num=stimulus_frame_num,
        timestamp_ns_session=timestamp_ns_session,
        source_acquisition_frame_index=acquisition,
        source_sample_row_index=sample_row,
        source_stimulus_run_row_index=source_run_row,
        source_stimulus_source_row_index=source_source_row,
        chaser_index=chaser_index,
        chaser_position_xy=chaser_position,
        chaser_valid=chaser_valid,
        authority=_freeze(authority),
    )


def _readonly_result_array(value: object, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _sample_tie_key(snapshot: _NativeSnapshot, row: int) -> tuple[Any, ...]:
    return (
        int(snapshot.timestamp_ns_session[row]),
        int(snapshot.stimulus_frame_num[row]),
        tuple(int(value) for value in snapshot.source_stimulus_run_row_index[row]),
        tuple(int(value) for value in snapshot.source_stimulus_source_row_index[row]),
        int(snapshot.source_sample_row_index[row]),
    )


def select_chaser_input_provenance_proxy(
    source_handle: object,
) -> ChaserInputProvenanceProxyResult:
    """Select one complete logged state per represented acquisition frame.

    Selection is lexicographically maximal by timestamp, stimulus frame, and
    source sample lineage.  Incomplete samples are retained as candidates but
    cannot be selected.  A frame with only incomplete candidates therefore has
    no selected chaser state; no value is carried from an adjacent frame.
    """

    snapshot = _snapshot_source(source_handle)
    frame_values = np.unique(snapshot.source_acquisition_frame_index)
    n_frames = int(frame_values.size)
    n_chasers = int(snapshot.chaser_index.size)

    candidate_rows: list[int] = []
    candidate_complete: list[bool] = []
    candidate_reason: list[str] = []
    offsets = [0]
    selected_rows = np.full(n_frames, -1, dtype=np.int64)
    selected = np.zeros(n_frames, dtype=bool)
    selection_reason = np.full(n_frames, "no_complete_chaser_sample", dtype="<U32")

    for frame_offset, frame in enumerate(frame_values.tolist()):
        rows = np.flatnonzero(snapshot.source_acquisition_frame_index == frame)
        complete_rows: list[int] = []
        for row_value in rows.tolist():
            row = int(row_value)
            complete = bool(np.all(snapshot.chaser_valid[row]))
            candidate_rows.append(row)
            candidate_complete.append(complete)
            candidate_reason.append("complete" if complete else "incomplete_chaser_sample")
            if complete:
                complete_rows.append(row)
        offsets.append(len(candidate_rows))
        if not complete_rows:
            continue
        ranked = sorted(complete_rows, key=lambda row: _sample_tie_key(snapshot, row))
        best = ranked[-1]
        best_key = _sample_tie_key(snapshot, best)
        if sum(_sample_tie_key(snapshot, row) == best_key for row in complete_rows) != 1:
            _error(
                "Ambiguous exact tie among complete native stimulus samples for "
                f"source_acquisition_frame_index={int(frame)}."
            )
        selected_rows[frame_offset] = best
        selected[frame_offset] = True
        selection_reason[frame_offset] = "selected"

    candidate_rows_array = np.asarray(candidate_rows, dtype=np.int64)
    candidate_count = np.diff(np.asarray(offsets, dtype=np.int64))
    candidate_source_run = snapshot.source_stimulus_run_row_index[candidate_rows_array]
    candidate_source_source = snapshot.source_stimulus_source_row_index[candidate_rows_array]
    selected_count = int(np.count_nonzero(selected))
    projection_record = {
        "schema_id": PROJECTION_RECORD_SCHEMA_ID,
        "schema_version": PROJECTION_RECORD_SCHEMA_VERSION,
        "recording_id": snapshot.recording_id,
        "policy_id": PROXY_POLICY_ID,
        "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
        "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
        "physical_presentation_verified": PHYSICAL_PRESENTATION_VERIFIED,
        "presentation_timestamp_available": PRESENTATION_TIMESTAMP_AVAILABLE,
        "camera_presentation_clock_transform_available": CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
        "camera_exposure_reference": CAMERA_EXPOSURE_REFERENCE,
        "scientific_use_class": SCIENTIFIC_USE_CLASS,
        "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
        "native_sample_axis": "stimulus_samples",
        "native_sample_rows_preserved": True,
        "source_acquisition_frame_field": "source_acquisition_frame_index",
        "selection_order": [
            "timestamp_ns_session",
            "stimulus_frame_num",
            "source_stimulus_run_row_index",
            "source_stimulus_source_row_index",
            "source_sample_row_index",
        ],
        "complete_sample_rule": "all_declared_chasers_valid_and_finite_in_one_native_sample",
        "missing_frame_rule": "no_carry_forward",
        "native_sample_count": int(snapshot.stimulus_frame_num.size),
        "unique_acquisition_frame_count": n_frames,
        "selected_acquisition_frame_count": selected_count,
        "chaser_count": n_chasers,
        "candidate_sample_row_index_is_zero_based": True,
        "source_authority_id": snapshot.source_authority_id,
        "source_authority_digest": snapshot.source_authority_digest,
        "source_manifest_sha256": snapshot.source_manifest_sha256,
        "source_verification_digest": snapshot.source_verification_digest,
        "source_run_path": snapshot.source_run_path,
    }
    projection_record = _freeze(projection_record)
    projection_record_sha256 = _json_sha256(projection_record)

    def selected_values(array: np.ndarray, *, fill: Any) -> np.ndarray:
        shape = (n_frames,) + array.shape[1:]
        result = np.full(shape, fill, dtype=array.dtype)
        valid_rows = selected_rows >= 0
        if np.any(valid_rows):
            result[valid_rows] = array[selected_rows[valid_rows]]
        return result

    result = ChaserInputProvenanceProxyResult(
        source_handle=snapshot.source_handle,
        recording_id=snapshot.recording_id,
        source_authority_id=snapshot.source_authority_id,
        source_authority_digest=snapshot.source_authority_digest,
        source_manifest_sha256=snapshot.source_manifest_sha256,
        source_verification_digest=snapshot.source_verification_digest,
        source_run_path=snapshot.source_run_path,
        acquisition_frame_index=_readonly_result_array(frame_values, dtype=np.int64),
        candidate_offsets=_readonly_result_array(offsets, dtype=np.int64),
        candidate_sample_count=_readonly_result_array(candidate_count, dtype=np.int64),
        candidate_native_sample_row_index=_readonly_result_array(candidate_rows_array, dtype=np.int64),
        candidate_stimulus_frame_num=_readonly_result_array(
            snapshot.stimulus_frame_num[candidate_rows_array], dtype=np.int64
        ),
        candidate_timestamp_ns_session=_readonly_result_array(
            snapshot.timestamp_ns_session[candidate_rows_array], dtype=np.int64
        ),
        candidate_source_acquisition_frame_index=_readonly_result_array(
            snapshot.source_acquisition_frame_index[candidate_rows_array], dtype=np.int64
        ),
        candidate_complete=_readonly_result_array(candidate_complete, dtype=bool),
        candidate_reason_code=_readonly_result_array(candidate_reason, dtype="<U32"),
        candidate_source_stimulus_run_row_index=_readonly_result_array(
            candidate_source_run, dtype=np.int64
        ),
        candidate_source_stimulus_source_row_index=_readonly_result_array(
            candidate_source_source, dtype=np.int64
        ),
        selected=_readonly_result_array(selected, dtype=bool),
        selection_reason_code=_readonly_result_array(selection_reason, dtype="<U32"),
        selected_native_sample_row_index=_readonly_result_array(selected_rows, dtype=np.int64),
        selected_stimulus_frame_num=_readonly_result_array(
            selected_values(snapshot.stimulus_frame_num, fill=-1), dtype=np.int64
        ),
        selected_timestamp_ns_session=_readonly_result_array(
            selected_values(snapshot.timestamp_ns_session, fill=-1), dtype=np.int64
        ),
        selected_source_stimulus_run_row_index=_readonly_result_array(
            selected_values(snapshot.source_stimulus_run_row_index, fill=-1), dtype=np.int64
        ),
        selected_source_stimulus_source_row_index=_readonly_result_array(
            selected_values(snapshot.source_stimulus_source_row_index, fill=-1), dtype=np.int64
        ),
        selected_chaser_index=_readonly_result_array(
            np.broadcast_to(snapshot.chaser_index, (n_frames, n_chasers)), dtype=np.int16
        ),
        selected_chaser_position_xy=_readonly_result_array(
            selected_values(snapshot.chaser_position_xy, fill=np.nan), dtype=np.float64
        ),
        selected_chaser_valid=_readonly_result_array(
            selected_values(snapshot.chaser_valid, fill=False), dtype=bool
        ),
        acquisition_projection_record=projection_record,
        acquisition_projection_record_sha256=projection_record_sha256,
        provenance=_freeze(
            {
                "policy_id": PROXY_POLICY_ID,
                "temporal_alignment_requirement": TEMPORAL_ALIGNMENT_REQUIREMENT,
                "temporal_alignment_class": TEMPORAL_ALIGNMENT_CLASS,
                "physical_presentation_verified": PHYSICAL_PRESENTATION_VERIFIED,
                "presentation_timestamp_available": PRESENTATION_TIMESTAMP_AVAILABLE,
                "camera_presentation_clock_transform_available": CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
                "camera_exposure_reference": CAMERA_EXPOSURE_REFERENCE,
                "scientific_use_class": SCIENTIFIC_USE_CLASS,
                "behavioral_denominator": BEHAVIORAL_DENOMINATOR,
                "native_sample_rows_preserved": True,
                "native_sample_count": int(snapshot.stimulus_frame_num.size),
                "unique_input_acquisition_frame_count": n_frames,
                "chaser_count": n_chasers,
                "no_carry_forward_across_missing_acquisition_frames": True,
                "complete_sample_rule": "all_declared_chasers_valid_and_finite_in_one_native_sample",
                "source_recording_id": snapshot.recording_id,
                "source_authority_id": snapshot.source_authority_id,
                "source_authority_digest": snapshot.source_authority_digest,
                "source_manifest_sha256": snapshot.source_manifest_sha256,
                "source_verification_digest": snapshot.source_verification_digest,
                "source_run_path": snapshot.source_run_path,
                "acquisition_projection_record_sha256": projection_record_sha256,
            }
        ),
    )
    return result


select_latest_logged_cpu_state_per_input_acquisition_proxy = (
    select_chaser_input_provenance_proxy
)

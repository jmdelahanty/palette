from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.unified_h5.common import UnifiedH5ContractError
from fisheye.shared.unified_h5.display_submissions import (
    NOT_SUBMITTED,
    PARENT_SCHEMA,
    TABLE_SCHEMA,
    _scalar_number,
    build_stimulus_display_submissions,
)

TABLE_ATTRS = {**TABLE_SCHEMA, "timing_semantics": "software_observed_cpu_enqueue"}
PARENT_ATTRS = {**PARENT_SCHEMA, "display_submission_rate_target_hz": 120.0, "glfw_swap_interval": 0}
DTYPE = [
    ("display_frame_index", "<u8"),
    ("source_stimulus_frame_num", "<u8"),
    ("swap_call_end_ns_session", "<i8"),
    ("swap_call_end_ns_epoch", "<i8"),
    ("source_state_valid", "u1"),
]


def _rows(stims: list[int], valid: list[int] | None = None) -> np.ndarray:
    rows = np.zeros(len(stims), dtype=DTYPE)
    rows["display_frame_index"] = 160_072 + np.arange(len(stims))
    rows["source_stimulus_frame_num"] = stims
    rows["swap_call_end_ns_session"] = np.arange(len(stims)) * 8_333_333
    rows["swap_call_end_ns_epoch"] = 1_760_000_000_000_000_000 + rows["swap_call_end_ns_session"]
    rows["source_state_valid"] = [1] * len(stims) if valid is None else valid
    return rows


def _build(rows: np.ndarray, states: np.ndarray, **overrides):
    return build_stimulus_display_submissions(
        rows,
        states,
        table_attrs=overrides.get("table_attrs", TABLE_ATTRS),
        parent_attrs=overrides.get("parent_attrs", PARENT_ATTRS),
        source_reference_sha256="ref",
    )


def test_first_submission_holds_and_never_submitted_states() -> None:
    # Two invalid start rows, state 1 held for two frames, state 3 never shown.
    rows = _rows([0, 0, 0, 1, 1, 2, 4], valid=[0, 0, 1, 1, 1, 1, 1])
    result = _build(rows, np.arange(5))

    assert result.invalid_prefix_rows == 2
    np.testing.assert_array_equal(result.submitted, [True, True, True, False, True])
    np.testing.assert_array_equal(result.submission_count, [1, 2, 1, 0, 1])
    np.testing.assert_array_equal(
        result.first_display_frame_index, [160_074, 160_075, 160_077, -1, 160_078]
    )
    assert result.first_swap_end_ns_session[1] == 3 * 8_333_333
    assert result.first_swap_end_ns_epoch[3] == NOT_SUBMITTED
    assert (result.never_submitted_count, result.held_state_count) == (1, 1)
    assert (result.display_rate_target_hz, result.glfw_swap_interval) == (120.0, 0)


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda r: r.__setitem__("source_state_valid", [1, 0, 1]), "invalid_row_after_first_state"),
        (lambda r: r.__setitem__("display_frame_index", [5, 5, 6]), "display_frame_index_not_strictly"),
        (lambda r: r.__setitem__("source_stimulus_frame_num", [1, 0, 2]), "states_out_of_order"),
        (lambda r: r.__setitem__("swap_call_end_ns_epoch", [3, 2, 4]), "epoch_time_decreases"),
        (lambda r: r.__setitem__("source_stimulus_frame_num", [0, 1, 9]), "state_not_in_frames"),
    ],
)
def test_malformed_rows_are_refused(mutate, reason: str) -> None:
    rows = _rows([0, 1, 2])
    mutate(rows)
    with pytest.raises(UnifiedH5ContractError, match=reason):
        _build(rows, np.arange(3))


def test_unpinned_schema_is_refused() -> None:
    with pytest.raises(UnifiedH5ContractError, match="display_submission_schema:schema_version"):
        _build(_rows([0]), np.arange(1), table_attrs={**TABLE_ATTRS, "schema_version": 2})
    with pytest.raises(UnifiedH5ContractError, match="display_timing_schema:schema_id"):
        _build(_rows([0]), np.arange(1), parent_attrs={**PARENT_ATTRS, "schema_id": "other"})


def test_scalar_number_decodes_typed_descriptors() -> None:
    u64 = {"shape": [], "type": {"class": "integer", "signed": False, "size_bytes": 8}, "payload_hex": "0100000000000000"}
    f64 = {"shape": [], "type": {"class": "ieee_float", "size_bytes": 8}, "payload_hex": "0000000000005e40"}
    assert _scalar_number(u64, "schema_version") == 1
    assert _scalar_number(f64, "rate") == 120.0

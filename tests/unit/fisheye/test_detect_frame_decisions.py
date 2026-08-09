from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.detect_frame_decisions import (
    DETECT_FRAME_DECISION_FAMILY,
    DetectFrameDecisionError,
    clear_detect_frame_decision,
    load_detect_frame_decisions,
    materialize_detect_frame_decision_run,
    set_detect_frame_negative,
)


def _root(tmp_path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "review.zarr"), mode="w", zarr_format=3)
    raw = root.create_group("raw_video")
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([10, 20, 30], dtype=np.int64),
    )
    return root


def test_absent_decision_run_is_all_unreviewed_without_writing(tmp_path) -> None:
    root = _root(tmp_path)

    decisions = load_detect_frame_decisions(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
    )

    assert decisions.decision_codes.tolist() == [0, 0, 0]
    assert decisions.reason_codes.tolist() == [0, 0, 0]
    assert DETECT_FRAME_DECISION_FAMILY not in root


def test_materialize_all_positive_decision_receipt_without_inventing_labels(
    tmp_path,
) -> None:
    root = _root(tmp_path)

    decisions = materialize_detect_frame_decision_run(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
    )

    assert decisions.decision_codes.tolist() == [0, 0, 0]
    assert decisions.reason_codes.tolist() == [0, 0, 0]
    run = root[f"{DETECT_FRAME_DECISION_FAMILY}/refined-a"]
    assert run.attrs["source_refined_detect_run"] == "refined-a"
    assert run.attrs["selector_eligible"] is False
    np.testing.assert_array_equal(
        run["source_acquisition_frame_index"][:],
        [10, 20, 30],
    )


def test_negative_decision_is_bound_typed_and_clearable(tmp_path) -> None:
    root = _root(tmp_path)

    decisions = set_detect_frame_negative(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
        frame_index=1,
    )

    run = root[f"{DETECT_FRAME_DECISION_FAMILY}/refined-a"]
    assert decisions.decision_label(1) == "negative"
    assert decisions.reason_label(1) == "subject_outside_dish"
    assert run["frame_indices"].dtype == np.dtype("int32")
    assert run["source_acquisition_frame_index"].dtype == np.dtype("int64")
    assert run["decision_codes"].dtype == np.dtype("uint8")
    assert run["reason_codes"].dtype == np.dtype("uint16")
    assert run["decision_codes"].attrs["write_mode"] == "random_update"
    assert run["decision_codes"].attrs["storage_profile_id"] == "editable_local_v1"
    assert run.attrs["selector_eligible"] is False
    assert set(root[DETECT_FRAME_DECISION_FAMILY].attrs) == {
        "schema_family",
        "selector_policy",
    }

    cleared = clear_detect_frame_decision(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
        frame_index=1,
    )
    assert cleared.decision_codes.tolist() == [0, 0, 0]
    assert cleared.reason_codes.tolist() == [0, 0, 0]


def test_validation_rejects_tampered_decision_reason_pair(tmp_path) -> None:
    root = _root(tmp_path)
    set_detect_frame_negative(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
        frame_index=1,
    )
    run = root[f"{DETECT_FRAME_DECISION_FAMILY}/refined-a"]
    run["reason_codes"][1] = np.uint16(0)

    with pytest.raises(DetectFrameDecisionError, match="inconsistent"):
        load_detect_frame_decisions(
            root,
            source_refined_detect_run="refined-a",
            n_frames=3,
        )


def test_validation_rejects_changed_acquisition_lineage(tmp_path) -> None:
    root = _root(tmp_path)
    set_detect_frame_negative(
        root,
        source_refined_detect_run="refined-a",
        n_frames=3,
        frame_index=1,
    )
    root["raw_video/original_frame_indices"][1] = np.int64(21)

    with pytest.raises(DetectFrameDecisionError, match="frame lineage"):
        load_detect_frame_decisions(
            root,
            source_refined_detect_run="refined-a",
            n_frames=3,
        )

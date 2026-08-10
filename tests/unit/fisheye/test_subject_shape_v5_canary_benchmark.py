from __future__ import annotations

import numpy as np
import pytest

from fisheye.diagnostics.benchmark_subject_shape_v5_canary import (
    RESULT_SCHEMA_ID,
    RESULT_SCHEMA_VERSION,
    require_acceptance_result,
    require_v5_identity,
    validate_frame_cardinality_arrays,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
    CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
    CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
    SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR,
    SUBJECT_SHAPE_BUNDLE_ID_ATTR,
    SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
    SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def test_frame_cardinality_accepts_empty_and_multi_row_frames() -> None:
    result = validate_frame_cardinality_arrays(
        np.asarray([0, 2, 2, 3, 6], dtype=np.int64),
        np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int64),
        require_empty_frame=True,
        require_multi_row_frame=True,
    )

    assert result["valid"] is True
    assert result["empty_frame_count"] == 1
    assert result["multi_row_frame_count"] == 2
    assert result["maximum_rows_per_frame"] == 3
    assert result["empty_frame_example"] == {
        "frame_index": 1,
        "row_start": 2,
        "row_stop": 2,
        "row_count": 0,
    }
    assert result["multi_row_frame_example"] == {
        "frame_index": 0,
        "row_start": 0,
        "row_stop": 2,
        "row_count": 2,
    }


def test_frame_cardinality_rejects_offset_frame_mismatch() -> None:
    with pytest.raises(ValueError, match="does not exactly index"):
        validate_frame_cardinality_arrays(
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
            require_empty_frame=False,
            require_multi_row_frame=False,
        )


def test_frame_cardinality_requirement_fails_closed() -> None:
    with pytest.raises(ValueError, match="no multi-row frame"):
        validate_frame_cardinality_arrays(
            np.asarray([0, 1, 1, 2], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            require_empty_frame=True,
            require_multi_row_frame=True,
        )


def _v5_attrs(bundle_id: str) -> dict[str, object]:
    return {
        "schema_id": CANONICAL_SUBJECT_SHAPE_RUN_SCHEMA_ID,
        "schema_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_RUN_SCHEMA_VERSION,
        "method": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD,
        "method_version": CANONICAL_SUBJECT_SHAPE_BUNDLE_METHOD_VERSION,
        "row_axis": "recording_subject_mask_bundle_rows",
        "subject_shape_source_kind": SUBJECT_SHAPE_BUNDLE_SOURCE_KIND,
        SUBJECT_SHAPE_BUNDLE_ID_ATTR: bundle_id,
        SUBJECT_SHAPE_BUNDLE_ACTIVE_AT_DERIVATION_ATTR: False,
        SUBJECT_SHAPE_SOURCE_BINDING_DIGEST_ATTR: "a" * 64,
        "palette_run_completion_status": "complete",
        "stage_selector_eligible": False,
    }


def test_v5_identity_accepts_exact_ineligible_bundle_publication() -> None:
    require_v5_identity(_v5_attrs("bundle_v3"), bundle_id="bundle_v3")


def test_v5_identity_does_not_relabel_historical_v4() -> None:
    attrs = _v5_attrs("bundle_v3")
    attrs["schema_version"] = 4
    attrs["method"] = "subject_shape_from_refined_masks_v11"

    with pytest.raises(ValueError, match="v5 identity differs"):
        require_v5_identity(attrs, bundle_id="bundle_v3")


def test_acceptance_result_requires_exact_digest_and_pass() -> None:
    payload = {"status": "passed", "run_name": "shape_v5"}
    result = {
        "schema_id": RESULT_SCHEMA_ID,
        "schema_version": RESULT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_acceptance_result(result)

    result["payload_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity/digest"):
        require_acceptance_result(result)

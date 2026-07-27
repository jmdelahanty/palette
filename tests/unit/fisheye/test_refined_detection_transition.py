from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.instance_keys import (
    mint_detection_instance_keys,
    mint_manual_curation_instance_keys,
)
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
)
from fisheye.shared.zarr.refined_detection_transition import (
    RefinedDetectionTransitionError,
    build_accept_all_refined_detection_root,
    build_refined_detection_transition,
)


RECORDING_IDENTITY = "sleepyfish_cam2010095"


def _current_run() -> dict[str, dict[str, np.ndarray]]:
    source_bbox = np.asarray(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.25, 0.30, 0.10, 0.20],
            [0.75, 0.60, 0.20, 0.10],
        ],
        dtype=np.float64,
    )
    manual_bbox = np.asarray([[0.35, 0.40, 0.12, 0.16]], dtype=np.float32)
    manual_key = mint_manual_curation_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        refined_row_ids=np.asarray([12], dtype=np.int64),
        frame_indices=np.asarray([2], dtype=np.int32),
        bbox_norm_coords=manual_bbox,
        class_ids=np.asarray([2], dtype=np.int32),
    )[0]
    return {
        "instances": {
            # Deliberately not in canonical frame/refined-row order.
            "frame_indices": np.asarray([3, 0, 2], dtype=np.int32),
            "refined_row_ids": np.asarray([11, 10, 12], dtype=np.int64),
            "bbox_norm_coords": np.asarray(
                [source_bbox[2], source_bbox[0], manual_bbox[0]],
                dtype=np.float64,
            ),
            "source_kind_codes": np.asarray(
                [
                    SOURCE_KIND_CODE_MAP["raw_detect"],
                    SOURCE_KIND_CODE_MAP["raw_detect"],
                    SOURCE_KIND_CODE_MAP["manual"],
                ],
                dtype=np.int8,
            ),
            "manual_edit_flags": np.asarray([False, False, True], dtype=bool),
            "source_detect_row_index": np.asarray([2, 0, -1], dtype=np.int32),
            "confidence_scores": np.asarray([0.7, 0.9, 0.0], dtype=np.float32),
            "class_ids": np.asarray([1, 1, 2], dtype=np.int32),
            "instance_key": np.asarray([102, 100, manual_key], dtype=np.uint64),
            "reason": np.asarray(
                ["clean", "clean", "manual_addition"],
                dtype=object,
            ),
            "frame_counts": np.asarray([1, 0, 1, 1], dtype=np.int32),
            "frame_offsets": np.asarray([0, 1, 1, 2, 3], dtype=np.int64),
        },
        "source_detections": {
            "source_detect_row_index": np.arange(3, dtype=np.int32),
            "frame_indices": np.asarray([0, 1, 3], dtype=np.int32),
            "bbox_norm_coords": source_bbox,
            "decision_codes": np.asarray(
                [
                    SOURCE_DECISION_CODE_MAP["accepted"],
                    SOURCE_DECISION_CODE_MAP["filtered"],
                    SOURCE_DECISION_CODE_MAP["accepted"],
                ],
                dtype=np.int8,
            ),
            "resolved_refined_row_id": np.asarray([10, -1, 11], dtype=np.int64),
            "confidence_scores": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            "class_ids": np.asarray([1, 3, 1], dtype=np.int32),
            "instance_key": np.asarray([100, 101, 102], dtype=np.uint64),
            "reason": np.asarray(["clean", "filtered_blip", "clean"], dtype=object),
        },
    }


def _transition(run=None, **kwargs):
    return build_refined_detection_transition(
        _current_run() if run is None else run,
        n_frames=4,
        source_width=640,
        source_height=480,
        recording_identity=RECORDING_IDENTITY,
        **kwargs,
    )


def test_current_sparse_run_transitions_to_exact_full_v1() -> None:
    result = _transition()

    assert result.report["status"] == "contract_ready"
    assert result.report["selector_eligible"] is False
    assert result.report["lossy_conversions"] == []
    assert (
        REFINED_DETECTION_SCHEMA_V1.validate(
            result.arrays,
            dimensions=result.dimensions,
        )
        == ()
    )
    assert result.arrays["instances/frame_indices"].tolist() == [0, 2, 3]
    assert result.arrays["instances/refined_row_ids"].tolist() == [10, 12, 11]
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 1, 1, 2, 3]
    assert result.arrays["source_detections/frame_row_offsets"].tolist() == [
        0,
        1,
        2,
        2,
        3,
    ]
    assert result.arrays["instances/bbox_norm_coords"].dtype == np.float32
    assert result.arrays["instances/source_kind_codes"].dtype == np.uint8
    assert result.arrays["source_detections/decision_codes"].dtype == np.uint8
    assert result.arrays["instances/score_valid"].tolist() == [True, False, True]
    assert result.instance_reason_codes == {0: "none", 1: "manual_addition"}
    assert result.source_reason_codes == {0: "none", 1: "filtered_blip"}


def test_accept_all_root_preserves_multi_instance_canonical_rows() -> None:
    frames = np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int32)
    bbox_norm = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.80, 0.20, 0.10, 0.10],
            [0.50, 0.50, 0.20, 0.20],
            [0.20, 0.80, 0.10, 0.10],
            [0.50, 0.80, 0.10, 0.10],
            [0.80, 0.80, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    class_ids = np.asarray([1, 1, 2, 1, 2, 3], dtype=np.int32)
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=640,
        source_height=480,
    )
    keys = mint_detection_instance_keys(
        recording_identity=RECORDING_IDENTITY,
        frame_indices=frames,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    offsets = np.asarray([0, 2, 2, 3, 6], dtype=np.int64)
    canonical = {
        "instances/frame_indices": frames,
        "instances/source_acquisition_frame_index": frames.astype(np.int64),
        "instances/instance_key": keys,
        "instances/bbox_norm_coords": bbox_norm,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": np.linspace(0.5, 1.0, 6, dtype=np.float32),
        "instances/class_ids": class_ids,
        "instances/frame_row_offsets": offsets,
    }
    result = build_accept_all_refined_detection_root(
        canonical,
        dimensions=CanonicalDetectionDimensions(
            n_frames=4,
            n_instances=6,
            source_width=640,
            source_height=480,
        ),
        recording_identity=RECORDING_IDENTITY,
    )

    assert result.report["source_profile"] == ("canonical_detection_v1_accept_all_root")
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 2, 2, 3, 6]
    assert result.arrays["source_detections/frame_row_offsets"].tolist() == [
        0,
        2,
        2,
        3,
        6,
    ]
    assert np.array_equal(
        result.arrays["instances/instance_key"],
        result.arrays["source_detections/instance_key"],
    )
    assert result.arrays["instances/refined_row_ids"].tolist() == list(range(6))
    assert result.arrays["instances/score_valid"].all()
    assert not result.arrays["instances/manual_edit_flags"].any()
    assert set(result.arrays["source_detections/decision_codes"].tolist()) == {
        SOURCE_DECISION_CODE_MAP["accepted"]
    }
    assert (
        REFINED_DETECTION_SCHEMA_V1.validate(
            result.arrays,
            dimensions=result.dimensions,
        )
        == ()
    )


def test_missing_keys_use_bound_raw_source_and_manual_allocator() -> None:
    run = _current_run()
    del run["instances"]["instance_key"]
    del run["source_detections"]["instance_key"]
    raw = {
        "instance_key": np.asarray([100, 101, 102], dtype=np.uint64),
    }

    result = _transition(run, source_detect_group=raw)

    assert result.arrays["source_detections/instance_key"].tolist() == [100, 101, 102]
    assert result.arrays["instances/instance_key"][0] == np.uint64(100)
    assert result.arrays["instances/instance_key"][2] == np.uint64(102)
    assert int(np.unique(result.arrays["instances/instance_key"]).size) == 3


def test_interpolated_rows_fail_closed() -> None:
    run = _current_run()
    run["instances"]["source_kind_codes"][0] = np.int8(2)

    with pytest.raises(RefinedDetectionTransitionError) as captured:
        _transition(run)

    assert captured.value.report["status"] == "blocked"
    assert "interpolation is outside refined v1" in captured.value.report["blockers"][0]


def test_manual_nonzero_score_requires_explicit_lossy_opt_in() -> None:
    run = _current_run()
    run["instances"]["confidence_scores"][2] = np.float32(0.5)

    with pytest.raises(RefinedDetectionTransitionError, match="manual_score_reset"):
        _transition(copy.deepcopy(run))

    result = _transition(run, allow_manual_score_reset=True)
    manual = (
        result.arrays["instances/source_kind_codes"] == SOURCE_KIND_CODE_MAP["manual"]
    )
    assert result.arrays["instances/scores"][manual].tolist() == [0.0]
    assert result.report["lossy_conversions"] == [
        {
            "field": "instances/scores",
            "operation": "reset_manual_score_to_zero",
            "row_count": 1,
        }
    ]


def test_missing_source_keys_without_bound_fallback_fails_closed() -> None:
    run = _current_run()
    del run["source_detections"]["instance_key"]

    with pytest.raises(RefinedDetectionTransitionError) as captured:
        _transition(run)

    assert "lacks durable keys" in captured.value.report["blockers"][0]


def test_historical_source_keys_require_explicit_reported_initialization() -> None:
    run = _current_run()
    del run["instances"]["instance_key"]
    del run["source_detections"]["instance_key"]

    result = _transition(
        run,
        allow_initialize_missing_source_keys=True,
    )

    assert np.unique(result.arrays["source_detections/instance_key"]).size == 3
    assert result.report["identity_initializations"] == [
        {
            "field": "source_detections/instance_key",
            "operation": "mint_recording_frame_bbox_class_v1",
            "row_count": 3,
            "recording_identity": RECORDING_IDENTITY,
        }
    ]
    assert result.report["lossy_conversions"] == []


def test_full_transition_refuses_clipped_lineage_without_reading_it() -> None:
    run = _current_run()
    run["instances"]["source_clip_indices"] = np.asarray(
        [0, 0, 1],
        dtype=np.int64,
    )

    with pytest.raises(RefinedDetectionTransitionError) as captured:
        _transition(run)

    assert (
        "full-acquisition transition refuses clipped lineage arrays"
        in (captured.value.report["blockers"][0])
    )


def test_existing_manual_key_requires_allocator_or_parent_evidence() -> None:
    run = _current_run()
    run["instances"]["instance_key"][2] += np.uint64(1)

    with pytest.raises(RefinedDetectionTransitionError) as captured:
        _transition(run)

    assert "explicit parent identity evidence" in captured.value.report["blockers"][0]

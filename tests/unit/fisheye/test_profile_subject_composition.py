"""B4: dataset-profile composition reads the canonical subject record.

Design: ``docs/design/2026-10-07-intake-single-writer/README.md`` (B4). The
detection, keypoint and subject-mask profiles must take genotype and age from
the canonical subject record (``analysis/subject_metadata_runs``) when one is
published, and must keep producing today's values for historical archives
that have no record. The historical goldens below were captured from the
pre-B4 readers on ``main`` @ c9eee0ff and must not change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import zarr

from fisheye.shared.subject_mask_profile import build_subject_mask_profile_summary
from fisheye.shared.subject_metadata import publish_subject_metadata
from fisheye.utils.detection_profile import build_detection_profile_summary
from fisheye.utils.keypoint_profile import build_keypoint_profile_summary


def _add_detections(root: zarr.Group) -> None:
    parent = root.require_group("detect_runs")
    run = parent.require_group("detect_001")
    run.create_array("frame_indices", data=np.asarray([0, 0, 1, 3], dtype=np.int32))
    run.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.2, 0.25, 0.1, 0.12], [0.65, 0.72, 0.2, 0.18], [0.52, 0.3, 0.14, 0.1], [0.08, 0.12, 0.08, 0.09]],
            dtype=np.float64,
        ),
    )
    run.create_array("frame_counts", data=np.asarray([2, 1, 0, 1], dtype=np.int32))
    parent.attrs["latest"] = "detect_001"


def _add_keypoints(root: zarr.Group) -> None:
    parent = root.require_group("keypoints_runs")
    run = parent.require_group("keypoints_001")
    run.attrs.update(
        {
            "method": "traditional_pose",
            "skeleton_id": "fish_v1",
            "kpt_shape": [3, 2],
            "pose_schema": {
                "name": "traditional_v1",
                "skeleton_id": "fish_v1",
                "kpt_shape": [3, 2],
                "nodes": [
                    {"id": 0, "name": "swim_bladder"},
                    {"id": 1, "name": "eye_left"},
                    {"id": 2, "name": "eye_right"},
                ],
                "edges": [[0, 1], [0, 2], [1, 2]],
            },
        }
    )
    run.create_array("keypoints_roi", data=np.arange(24, dtype=np.float64).reshape((4, 3, 2)))
    run.create_array("usable_keypoints", data=np.asarray([True, True, False, True]))
    parent.attrs["latest"] = "keypoints_001"


def _add_masks(root: zarr.Group) -> None:
    parent = root.require_group("refined_subject_masks_runs")
    run = parent.require_group("masks_v1")
    run.attrs.update(
        {
            "created_at_utc": "2026-02-12T03:00:00+00:00",
            "label_schema_id": "subject_v1_union",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "run_semantics": "full",
            "total_rois": 2,
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True, True]))
    masks = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    masks[:, 0, :, :2] = 1
    run.create_array("masks_roi", data=masks)
    parent.attrs["latest"] = "masks_v1"


def _build_root(tmp_path: Path, setup: Callable[[zarr.Group], None]) -> tuple[zarr.Group, Path]:
    zarr_path = tmp_path / "rec_b4_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "rec_b4"
    root.attrs["zarr_use"] = "analysis"
    _add_detections(root)
    _add_keypoints(root)
    _add_masks(root)
    setup(root)
    return zarr.open_group(str(zarr_path), mode="r", use_consolidated=False), zarr_path


def _compositions(root: zarr.Group, zarr_path: Path) -> dict[str, Any]:
    stamp = "2026-10-07T00:00:00+00:00"
    summaries = {
        "detection": build_detection_profile_summary(root, zarr_path=zarr_path, created_at_utc=stamp),
        "keypoint": build_keypoint_profile_summary(root, zarr_path=zarr_path, created_at_utc=stamp),
        "subject_mask": build_subject_mask_profile_summary(root, zarr_path=zarr_path, created_at_utc=stamp),
    }
    return {name: summary.get("composition") for name, summary in summaries.items()}


def _assert_all(compositions: dict[str, Any], expected: Any) -> None:
    for name, composition in compositions.items():
        assert composition == expected, name
        if expected is not None:
            # Field order is part of the stored profile bytes.
            assert list(composition) == list(expected), name


def _session_context(root: zarr.Group, payload: dict[str, Any]) -> None:
    root.require_group("analysis_metadata").attrs["session_context"] = json.dumps(payload)


# ---------------------------------------------------------------- new records


def _unified_h5_record(root: zarr.Group) -> None:
    root.attrs["rig_id"] = "omnifin3"
    publish_subject_metadata(
        root,
        {
            "subject_ids": ["fish-1"],
            "subject_type": "individual",
            "species": "Danio rerio",
            "genotype": "nacre",
            "date_of_fertilization": "20261001",
            "queried_at_utc": "2026-10-07T14:00:00Z",
        },
        translator="h5_attributes",
    )


def _orange_record(root: zarr.Group) -> None:
    publish_subject_metadata(
        root,
        {
            "genotype": "Tg(elavl3:H2B-GCaMP6s)",
            "species": "Danio rerio",
            "subject_count": 1,
            "dof": "20260930",
            "recorded_at_utc": "2026-10-06T22:30:00Z",
            "dish_revision_at_recording": 3,
        },
        translator="orange_reference",
    )


def _declared_dpf_record(root: zarr.Group) -> None:
    publish_subject_metadata(
        root,
        {"genotype": "casper", "dpf_at_acquisition": 8},
        translator="h5_attributes",
    )


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (_unified_h5_record, {"rig_id": "omnifin3", "genotype": "nacre", "dpf_at_acquisition": 6}),
        # 22:30Z on 10-06 is 18:30 lab time: still 10-06, so 6 days.
        (_orange_record, {"genotype": "Tg(elavl3:H2B-GCaMP6s)", "dpf_at_acquisition": 6}),
        (_declared_dpf_record, {"genotype": "casper", "dpf_at_acquisition": 8}),
    ],
    ids=["unified_h5", "orange_reference", "declared_dpf"],
)
def test_subject_record_only_archive_reports_genotype_and_age(tmp_path, setup, expected) -> None:
    _assert_all(_compositions(*_build_root(tmp_path, setup)), expected)


def test_subject_record_is_authoritative_over_stale_root_copies(tmp_path) -> None:
    def setup(root: zarr.Group) -> None:
        root.attrs["genotype"] = "stale-root"
        root.attrs["dpf_at_acquisition"] = 99
        _session_context(root, {"genotype": "stale-context", "protocol_name": "Screen"})
        _unified_h5_record(root)

    _assert_all(
        _compositions(*_build_root(tmp_path, setup)),
        {"rig_id": "omnifin3", "protocol_name": "Screen", "genotype": "nacre", "dpf_at_acquisition": 6},
    )


def test_record_without_a_field_keeps_the_historical_value(tmp_path) -> None:
    def setup(root: zarr.Group) -> None:
        root.attrs["dpf_at_acquisition"] = 7
        publish_subject_metadata(root, {"genotype": "nacre"}, translator="h5_attributes")

    _assert_all(
        _compositions(*_build_root(tmp_path, setup)),
        {"genotype": "nacre", "dpf_at_acquisition": 7},
    )


def test_declared_absence_record_reports_no_subject_fields(tmp_path) -> None:
    def setup(root: zarr.Group) -> None:
        root.attrs["camera_id"] = "cam7"
        publish_subject_metadata(
            root,
            {"subject_lookup_status": "not_collected", "subject_lookup_reason": "no dish"},
            translator="declared_absence",
        )

    _assert_all(_compositions(*_build_root(tmp_path, setup)), {"camera_id": "cam7"})


# --------------------------------------------- historical goldens (no record)


def _root_attrs(root: zarr.Group) -> None:
    root.attrs.update({"rig_id": "omnifin0", "dish_design": "cedar", "genotype": "WT", "dpf_at_acquisition": 6})
    _session_context(
        root,
        {"camera_id": "2010094", "arena_id": "arena_2", "canvas_name": "shadow", "protocol_name_from_definition": "Default"},
    )


def _session_context_only(root: zarr.Group) -> None:
    _session_context(root, {"genotype": "  nacre  ", "days_post_fertilization": "5", "protocol_name": "Screen"})


def _subject_metadata_nested_dish(root: zarr.Group) -> None:
    root.require_group("analysis_metadata").attrs["subject_metadata"] = json.dumps(
        {"days_post_fertilization": 7, "dish": {"genotype": "Tg(elavl3:gcamp7f)"}}
    )


def _zebrobot_snapshot_only(root: zarr.Group) -> None:
    meta = root.require_group("analysis_metadata")
    meta.attrs["subject_metadata"] = "{}"
    meta.attrs["zebrobot_snapshot"] = {"genotype": "casper", "dpf_at_acquisition": 9}


def _precedence_and_bad_values(root: zarr.Group) -> None:
    root.attrs.update({"genotype": "root-wins", "dpf_at_acquisition": "not-a-number", "protocol_name": " "})
    _session_context(root, {"genotype": "ctx", "dpf_at_acquisition": None, "days_post_fertilization": 4, "protocol_name": "Ctx"})
    root.require_group("analysis_metadata").attrs["subject_metadata"] = {"genotype": "snap", "dpf_at_acquisition": 11}


def _legacy_singleton_is_not_read(root: zarr.Group) -> None:
    root.require_group("analysis/subject_metadata").attrs["subject_metadata"] = {
        "genotype": "singleton",
        "dpf_at_acquisition": 3,
    }


def _neither(root: zarr.Group) -> None:
    del root  # nothing beyond the profiled arrays


HISTORICAL_GOLDENS = [
    (
        _root_attrs,
        {
            "rig_id": "omnifin0",
            "camera_id": "2010094",
            "arena_id": "arena_2",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "Default",
            "genotype": "WT",
            "dpf_at_acquisition": 6,
        },
    ),
    (_session_context_only, {"protocol_name": "Screen", "genotype": "nacre", "dpf_at_acquisition": 5}),
    (_subject_metadata_nested_dish, {"genotype": "Tg(elavl3:gcamp7f)", "dpf_at_acquisition": 7}),
    (_zebrobot_snapshot_only, {"genotype": "casper", "dpf_at_acquisition": 9}),
    (_precedence_and_bad_values, {"protocol_name": "Ctx", "genotype": "root-wins", "dpf_at_acquisition": 4}),
    (_legacy_singleton_is_not_read, None),
    (_neither, None),
]


@pytest.mark.parametrize(
    ("setup", "expected"),
    HISTORICAL_GOLDENS,
    ids=[
        "root_attrs",
        "session_context_only",
        "subject_metadata_nested_dish",
        "zebrobot_snapshot_only",
        "precedence_and_bad_values",
        "legacy_singleton_is_not_read",
        "neither",
    ],
)
def test_historical_archive_without_record_keeps_golden_composition(tmp_path, setup, expected) -> None:
    _assert_all(_compositions(*_build_root(tmp_path, setup)), expected)

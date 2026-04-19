from __future__ import annotations

from pathlib import Path

from fisheye.detection import detect_keypoints_traditional as trad_mod
from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.pose.schema import (
    resolve_required_keypoint_indices_from_attrs,
    resolve_skeleton_identity_from_attrs,
    schema_payload_from_package,
)
from fisheye.utils import export_keypoint_training_zarr as export_mod


class _FakeExportGroup:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs


def test_schema_payload_from_package_exposes_canonical_identity_attrs() -> None:
    schema, payload = schema_payload_from_package("traditional_v1")

    assert schema.name == "traditional_v1"
    assert payload["skeleton_id"] == "pose_schema:traditional_v1"
    assert payload["kpt_shape"] == [3, 2]
    assert payload["keypoint_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert payload["source"] == "configs/fisheye/pose_schemas/traditional_v1.json"


def test_resolve_skeleton_identity_prefers_explicit_attrs_over_pose_schema() -> None:
    resolved = resolve_skeleton_identity_from_attrs(
        {
            "skeleton_id": "explicit_skeleton",
            "kpt_shape": [5, 2],
            "pose_schema": {
                "name": "traditional_v2",
                "skeleton_id": "pose_schema:traditional_v2",
                "kpt_shape": [99, 99],
            },
        }
    )

    assert resolved.pose_schema_name == "traditional_v2"
    assert resolved.skeleton_id == "explicit_skeleton"
    assert resolved.kpt_shape == (5, 2)


def test_resolve_skeleton_identity_falls_back_to_pose_schema_name_and_runtime_shape() -> None:
    resolved = resolve_skeleton_identity_from_attrs(
        {"pose_schema": {"name": "traditional_v2"}},
        keypoint_count=5,
    )

    assert resolved.pose_schema_name == "traditional_v2"
    assert resolved.skeleton_id == "pose_schema:traditional_v2"
    assert resolved.kpt_shape == (5, 2)


def test_resolve_required_keypoint_indices_uses_canonical_labels_from_run_attrs() -> None:
    resolved = resolve_required_keypoint_indices_from_attrs(
        {
            "keypoint_labels": ["eye_right", "tail_tip", "bladder", "left_eye"],
            "pose_schema": {
                "name": "traditional_v2",
                "nodes": [
                    {"id": 0, "name": "swim_bladder"},
                    {"id": 1, "name": "eye_left"},
                    {"id": 2, "name": "eye_right"},
                ],
            },
        },
        ("swim_bladder", "eye_left", "eye_right"),
        keypoint_count=4,
    )

    assert resolved == {
        "swim_bladder": 2,
        "eye_left": 3,
        "eye_right": 0,
    }


def test_raw_keypoint_writers_use_shared_pose_payload() -> None:
    _schema, payload = schema_payload_from_package("traditional_v1")

    assert trad_mod.TRADITIONAL_POSE_ATTR_PAYLOAD == payload
    assert yolo_mod.TRADITIONAL_POSE_ATTR_PAYLOAD == payload


def test_export_runtime_identity_resolution_uses_shared_schema_precedence() -> None:
    kp_group = _FakeExportGroup(
        {
            "pose_schema": {"name": "traditional_v2"},
        }
    )

    skeleton_id, kpt_shape, signature = export_mod._resolve_dataset_skeleton_identity(
        dataset_payload={
            "pose_schema": {
                "name": "traditional_v2",
                "skeleton_id": "dataset_skeleton_v2",
                "kpt_shape": [5, 3],
            }
        },
        kp_group=kp_group,
        source_zarr=Path("/tmp/source_pose.zarr"),
        keypoint_run="kp_pose_001",
        keypoint_count=5,
        manifest_skeleton_id="manifest_skeleton_v2",
        manifest_kpt_shape=(5, 3),
    )

    assert skeleton_id == "pose_schema:traditional_v2"
    assert kpt_shape == (5, 3)
    assert signature == "skeleton_id=pose_schema:traditional_v2, kpt_shape=[5,3]"


def test_export_keypoint_label_resolution_prefers_runtime_labels() -> None:
    kp_group = _FakeExportGroup(
        {
            "keypoint_labels": ["left_eye", "tail_tip", "bladder"],
        }
    )

    labels = export_mod._resolve_dataset_keypoint_labels(
        manifest_payload={"keypoint_labels": ["eye_left", "tail_tip", "swim_bladder"]},
        dataset_payload={},
        annotation_group=kp_group,
        source_zarr=Path("/tmp/source_pose.zarr"),
        keypoint_run="kp_pose_001",
        keypoint_count=3,
    )

    assert labels == ["eye_left", "tail_tip", "swim_bladder"]


def test_export_keypoint_label_resolution_rejects_dataset_mismatch() -> None:
    kp_group = _FakeExportGroup(
        {
            "keypoint_labels": ["eye_left", "tail_tip", "swim_bladder"],
        }
    )

    try:
        export_mod._resolve_dataset_keypoint_labels(
            manifest_payload={},
            dataset_payload={"keypoint_labels": ["tail_tip", "eye_left", "swim_bladder"]},
            annotation_group=kp_group,
            source_zarr=Path("/tmp/source_pose.zarr"),
            keypoint_run="kp_pose_001",
            keypoint_count=3,
        )
    except ValueError as exc:
        assert "dataset keypoint_labels" in str(exc)
    else:
        raise AssertionError("Expected dataset keypoint_labels mismatch to raise ValueError")

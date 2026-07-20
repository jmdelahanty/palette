from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from fisheye.training.train_pose import (
    _infer_pose_schema,
    _load_pose_manifest_authority,
    _record_registry_training_run,
)


def _pose_schema() -> dict:
    return {
        "skeleton_id": "pose_skel_exact",
        "kpt_shape": [3, 3],
        "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
        "skeleton": [[0, 1], [0, 2], [1, 2]],
    }


def _write_manifest(path: Path, *, marker: str = "initial") -> None:
    path.write_text(
        json.dumps(
            {
                "task": "pose",
                "set_id": "pose_set",
                "marker": marker,
                "pose_schema": _pose_schema(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _matching_live_metadata() -> dict:
    return {
        "training.zarr": {
            "tracking_info": {
                "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
                "keypoint_skeleton": [[0, 1], [0, 2], [1, 2]],
                "skeleton_id": "pose_skel_exact",
                "keypoint_count": 3,
                "model_kpt_shape": [3, 3],
            }
        }
    }


def test_hash_verified_manifest_schema_is_primary_pose_training_authority(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pose.manifest.json"
    _write_manifest(manifest_path)
    authority = _load_pose_manifest_authority(manifest_path)
    assert authority is not None

    resolved = _infer_pose_schema(
        (3, 3),
        _matching_live_metadata(),
        manifest_pose_schema=authority["pose_schema"],
    )

    assert resolved == _pose_schema()
    assert len(authority["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "hostile_value", "message"),
    [
        (
            "keypoint_labels",
            ["eye_left", "swim_bladder", "eye_right"],
            "ordered keypoint labels disagree",
        ),
        ("keypoint_skeleton", [[0, 2], [0, 1], [1, 2]], "skeleton disagrees"),
        ("skeleton_id", "pose_skel_other", "skeleton_id disagrees"),
        ("keypoint_count", 4, "cardinality disagrees"),
        ("model_kpt_shape", [3, 2], "model_kpt_shape disagrees"),
    ],
)
def test_manifest_primary_schema_rejects_populated_live_source_disagreement(
    field: str,
    hostile_value: object,
    message: str,
) -> None:
    live = _matching_live_metadata()
    live["training.zarr"]["tracking_info"][field] = hostile_value

    with pytest.raises(ValueError, match=message):
        _infer_pose_schema(
            (3, 3),
            live,
            manifest_pose_schema=_pose_schema(),
        )


def test_registry_registration_rejects_manifest_content_drift(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pose.manifest.json"
    _write_manifest(manifest_path)
    authority = _load_pose_manifest_authority(manifest_path)
    assert authority is not None
    _write_manifest(manifest_path, marker="changed-after-schema-resolution")

    registry_path = tmp_path / "registry.sqlite"
    with pytest.raises(ValueError, match="changed after its schema authority was resolved"):
        _record_registry_training_run(
            args=SimpleNamespace(registry=registry_path),
            console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
            invocation_payload={"tool": "test"},
            run_id="pose_run",
            set_id="pose_set",
            config_path=None,
            manifest_path=manifest_path,
            model_path=None,
            metrics_path=None,
            status="in_progress",
            final_metrics={"stage": "preflight"},
            pose_schema=authority["pose_schema"],
            expected_manifest_sha256=authority["manifest_sha256"],
        )

    assert not registry_path.exists()

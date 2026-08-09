from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.keypoint_manual_review_qc import (
    build_default_manual_keypoint_qc_policy,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.tabular_deltas import instance_key_digest
from fisheye.shared.zarr.training_review_artifact_publication import (
    TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID,
    _validate_keypoint_review_state,
)


def _review_fixture(path: Path) -> dict[str, str]:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "training_artifact_status": "review_active",
            "training_task": "keypoints",
            "stage_selector_eligible": False,
        }
    )
    run_ids = {
        "crop": "crop_v2",
        "raw": "raw_v2",
        "quality": "quality_v1",
        "refined": "refined_v2",
        "body": "body_v1",
        "delta": "edits_v1",
        "generation": "generation_000001",
    }
    crop_manifest = {"schema_id": "test.crop", "schema_version": 1}
    crop = root.require_group(f"crop_runs/{run_ids['crop']}")
    crop.attrs["run_manifest"] = crop_manifest
    for path_value in (
        f"keypoints_runs/{run_ids['raw']}",
        f"keypoint_quality_runs/{run_ids['quality']}",
        f"analysis/body_frame_runs/{run_ids['body']}",
    ):
        root.require_group(path_value).attrs["stage_selector_eligible"] = False

    labels = [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    ]
    skeleton_digest = "a" * 64
    refined = root.require_group(f"refined_keypoints_runs/{run_ids['refined']}")
    refined.attrs.update(
        {
            "stage_selector_eligible": False,
            "artifact_mutability": "immutable_snapshot",
            "run_manifest": {
                "schema_id": "test.refined",
                "schema_version": 1,
                "payload": {
                    "source_bindings": {
                        "crop_snapshot": {
                            "manifest_digest": canonical_json_sha256(crop_manifest)
                        },
                        "skeleton": {
                            "skeleton_id": "five_point_v1",
                            "skeleton_digest": skeleton_digest,
                            "semantics": {"keypoint_labels": labels},
                        },
                    }
                },
            },
        }
    )
    refined.create_array(
        "instance_key", data=np.asarray([11, 12, 13], dtype=np.uint64)
    )
    policy = build_default_manual_keypoint_qc_policy(
        skeleton_id="five_point_v1",
        skeleton_digest=skeleton_digest,
        keypoint_labels=labels,
    )
    instance_keys = np.asarray([11, 12, 13], dtype=np.uint64)
    delta = root.require_group(f"edit_delta_runs/{run_ids['delta']}")
    delta.attrs.update(
        {
            "base_instance_key_count": 3,
            "base_instance_key_sha256": instance_key_digest(instance_keys),
        }
    )
    generation = root.require_group(
        "edit_delta_runs/"
        f"{run_ids['delta']}/generations/{run_ids['generation']}"
    )
    generation.attrs.update(
        {
            "status": "open",
            "target_kind": "keypoints",
            "base_run_path": f"refined_keypoints_runs/{run_ids['refined']}",
            "review_qc_policy": policy.as_manifest(),
            "review_qc_policy_digest": policy.policy_digest,
            "base_instance_key_sha256": instance_key_digest(instance_keys),
        }
    )
    root.attrs["training_review_artifact"] = {
        "schema_id": TRAINING_KEYPOINT_REVIEW_ARTIFACT_SCHEMA_ID,
        "schema_version": 1,
    }
    return run_ids


def test_keypoint_only_review_state_binds_immutable_base_and_delta(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "review.zarr"
    run_ids = _review_fixture(archive)

    result = _validate_keypoint_review_state(
        archive,
        crop_run_id=run_ids["crop"],
        raw_run_id=run_ids["raw"],
        quality_run_id=run_ids["quality"],
        refined_run_id=run_ids["refined"],
        body_frame_run_id=run_ids["body"],
        delta_run_id=run_ids["delta"],
        delta_generation=run_ids["generation"],
    )

    assert result["row_count"] == 3
    assert result["metadata_read_mode"] == (
        "direct_unconsolidated_while_review_mutable"
    )

    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root[f"refined_keypoints_runs/{run_ids['refined']}"]["instance_key"][0] = 99
    with pytest.raises(RuntimeError, match="instance-key binding"):
        _validate_keypoint_review_state(
            archive,
            crop_run_id=run_ids["crop"],
            raw_run_id=run_ids["raw"],
            quality_run_id=run_ids["quality"],
            refined_run_id=run_ids["refined"],
            body_frame_run_id=run_ids["body"],
            delta_run_id=run_ids["delta"],
            delta_generation=run_ids["generation"],
        )

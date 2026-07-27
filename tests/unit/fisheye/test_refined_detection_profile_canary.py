from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.zarr.benchmark_fixture import (
    inventory_tree,
    thaw_tree_for_cleanup,
)
from fisheye.shared.zarr.refined_detection_profile_canary import (
    publish_refined_detection_profile_canary,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
)


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    fixture_root = tmp_path / "source_fixture"
    source_path = fixture_root / "source.zarr"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs.update(
        {
            "source_video_width": 640,
            "source_video_height": 480,
        }
    )
    frames = np.asarray([0, 0, 2, 3, 3, 3], dtype=np.int32)
    source.create_array("frame_indices", data=frames)
    source.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.20, 0.20, 0.10, 0.10],
                [0.80, 0.20, 0.10, 0.10],
                [0.50, 0.50, 0.20, 0.20],
                [0.20, 0.80, 0.10, 0.10],
                [0.50, 0.80, 0.10, 0.10],
                [0.80, 0.80, 0.10, 0.10],
            ],
            dtype=np.float32,
        ),
    )
    source.create_array(
        "scores",
        data=np.linspace(0.5, 1.0, frames.size, dtype=np.float32),
    )
    source.create_array(
        "class_ids",
        data=np.asarray([1, 1, 2, 1, 2, 3], dtype=np.int32),
    )
    source.create_array(
        "frame_counts",
        data=np.asarray([2, 0, 1, 3], dtype=np.int32),
    )
    inventory = inventory_tree(source_path)
    fixture_manifest = {
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "copied_zarr_relative_path": "source.zarr",
        "copied_inventory": inventory.as_manifest(),
    }
    manifest_path = fixture_root / "fixture_manifest.json"
    manifest_path.write_text(
        json.dumps(fixture_manifest, sort_keys=True),
        encoding="utf-8",
    )
    return source_path, manifest_path


def test_profile_canary_is_paired_immutable_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    source_path, fixture_manifest_path = _source_fixture(tmp_path)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    destination = (
        tmp_path
        / "shared/.palette_benchmarks/refined_detection_storage/profile_canary/test_v1"
    )

    result = publish_refined_detection_profile_canary(
        source_group_path=source_path,
        source_fixture_manifest_path=fixture_manifest_path,
        source_run_id="legacy_detect_1",
        recording_identity="multi_subject_recording",
        destination=destination,
        scratch_root=scratch_root,
        canary_id="test_v1",
        crimson_implementation_commit="a" * 40,
        crimson_evidence_commit="b" * 40,
        crimson_evidence_sha256="c" * 64,
        require_object_gate=False,
    )

    assert result["payload"]["status"] == "complete"
    assert result["payload"]["profile_promoted"] is False
    validation = result["payload"]["pair_validation"]
    assert validation["exact_decoded_logical_hash_equality"] is True
    assert validation["canonical_source_audit_equality"] is True
    assert validation["selector_attributes_absent"] is True
    assert validation["production_state_changes"] == []
    assert result["payload"]["logical_snapshot"]["dimensions"]["n_instances"] == 6
    assert (destination / "canary_manifest.json").is_file()
    assert not any(scratch_root.iterdir())

    regular_path = destination / "candidates/regular.zarr"
    candidate_path = destination / "candidates/access_aware.zarr"
    regular = zarr.open_group(str(regular_path), mode="r", use_consolidated=True)
    candidate = zarr.open_group(str(candidate_path), mode="r", use_consolidated=True)
    regular_run = regular["refined_detect_runs/refined_detect_regular_test_v1"]
    candidate_run = candidate["refined_detect_runs/refined_detect_access_aware_test_v1"]
    assert np.asarray(regular_run["instances/frame_row_offsets"][:]).tolist() == [
        0,
        2,
        2,
        3,
        6,
    ]
    assert np.array_equal(
        np.asarray(regular_run["instances/instance_key"][:]),
        np.asarray(candidate_run["instances/instance_key"][:]),
    )
    regular_metadata = json.loads(
        (
            regular_path / "refined_detect_runs/refined_detect_regular_test_v1/"
            "instances/frame_indices/zarr.json"
        ).read_text(encoding="utf-8")
    )
    candidate_metadata = json.loads(
        (
            candidate_path / "refined_detect_runs/"
            "refined_detect_access_aware_test_v1/instances/frame_indices/"
            "zarr.json"
        ).read_text(encoding="utf-8")
    )
    assert regular_metadata["codecs"][0]["name"] == "bytes"
    assert candidate_metadata["codecs"][0]["name"] == "bytes"
    assert regular_run.attrs["storage_plan"]["storage_profile_role"] == (
        "paired_unsharded_control"
    )
    assert candidate_run.attrs["storage_plan"]["storage_profile_role"] == (
        "unpromoted_access_aware_candidate"
    )

    thaw_tree_for_cleanup(destination)


def test_profile_canary_can_verify_the_promoted_default(tmp_path: Path) -> None:
    source_path, fixture_manifest_path = _source_fixture(tmp_path)
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    destination = (
        tmp_path
        / "shared/.palette_benchmarks/refined_detection_storage/profile_canary/"
        "promoted_v1"
    )

    result = publish_refined_detection_profile_canary(
        source_group_path=source_path,
        source_fixture_manifest_path=fixture_manifest_path,
        source_run_id="legacy_detect_1",
        recording_identity="multi_subject_recording",
        destination=destination,
        scratch_root=scratch_root,
        canary_id="promoted_v1",
        crimson_implementation_commit="a" * 40,
        crimson_evidence_commit="b" * 40,
        crimson_evidence_sha256="c" * 64,
        access_aware_profile=DETECTION_PUBLISHED_ACCESS_AWARE_V1,
        require_object_gate=False,
    )

    assert result["payload"]["profile_promoted"] is True
    assert result["payload"]["artifacts"]["access_aware"][
        "storage_profile_id"
    ] == "detection_published_access_aware_v1"
    assert result["payload"]["gate"]["promotion_decision"] == (
        "promoted_profile_verification"
    )
    thaw_tree_for_cleanup(destination)

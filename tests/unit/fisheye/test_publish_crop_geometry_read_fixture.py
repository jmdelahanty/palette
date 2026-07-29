from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr

from fisheye.diagnostics.publish_crop_geometry_read_fixture import (
    ARCHIVE_NAME,
    CROP_READ_FIXTURE_SCHEMA_ID,
    HANDOFF_NAME,
    publish_crop_geometry_read_fixture,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_crop_shadow import REFINED_RUN_ID, _refined_source


def _policy() -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(8, 8),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )


def _video(tmp_path: Path) -> tuple[Path, Path]:
    recording = tmp_path / "recording"
    video = recording / "cams" / "camera.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"benchmark source video")
    return recording, video


def test_fixture_exercises_real_candidate_publisher_without_source_mutation(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    source_archive = source.archive_path
    source_run = zarr.open_group(
        str(source_archive / "refined_detect_runs" / REFINED_RUN_ID),
        mode="r",
        use_consolidated=False,
    )
    source_manifest_digest = source_run.attrs["run_manifest"]["payload_digest"]
    recording, video = _video(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    destination = (
        tmp_path / ".palette_benchmarks" / "crop_geometry" / "integration_fixture_v1"
    )

    handoff = publish_crop_geometry_read_fixture(
        source_refined_zarr=source_archive,
        source_refined_run_id=REFINED_RUN_ID,
        source_video=video,
        recording_path=recording,
        camera_identity="cam2010095",
        fps=100.0,
        codec="hevc",
        pixel_format="yuv420p",
        destination=destination,
        work_root=scratch,
        crop_run_id="crop_read_fixture_v2",
        policy=_policy(),
    )

    assert handoff["schema_id"] == CROP_READ_FIXTURE_SCHEMA_ID
    assert handoff["payload_digest"] == canonical_json_sha256(handoff["payload"])
    payload = handoff["payload"]
    assert payload["status"] == "complete"
    assert payload["benchmark_only"] is True
    assert payload["selector_eligible"] is False
    assert payload["registry_updated"] is False
    assert payload["production_state_changes"] == []
    assert payload["validation"] == {
        "crop_run_manifest_digest": payload["publication"]["run_manifest_digest"],
        "crop_logical_content_digest": payload["publication"]["logical_content_digest"],
        "storage_profile_id": "published_http_v1",
        "array_count": 13,
        "direct_consolidated_metadata_equal": True,
        "source_refined_authority_valid": True,
        "source_pixel_authority_valid": True,
        "frame_count": 4,
        "row_count": 4,
        "geometry_only": True,
        "selector_eligible": False,
        "completion_contract_valid": True,
    }
    assert payload["handoff_policy"]["profile_promotion_evidence"] is False
    assert payload["publication"]["node_local_materialization"]["writer_receipt"]
    for copy_receipt in (
        payload["seed"]["source_copy"],
        payload["seed_copy_to_shared_staging"],
    ):
        assert set(copy_receipt["timing_seconds"]) == {
            "source_inventory",
            "copy",
            "destination_inventory",
            "total",
        }
        assert all(value >= 0 for value in copy_receipt["timing_seconds"].values())
    assert (
        len(
            payload["publication"]["node_local_materialization"]["writer_receipt"][
                "writes"
            ]
        )
        == 13
    )
    assert list(scratch.iterdir()) == []
    assert destination.is_dir()
    assert not list(destination.parent.glob(f".{destination.name}.partial.*"))

    persisted = json.loads((destination / HANDOFF_NAME).read_text(encoding="utf-8"))
    assert persisted == handoff
    archive = destination / ARCHIVE_NAME
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    for root in (direct, consolidated):
        assert root.attrs["benchmark_only"] is True
        assert root.attrs["selector_eligible"] is False
        refined = root["refined_detect_runs"]
        assert refined.attrs["authoritative_run"] == REFINED_RUN_ID
        crop = root["crop_runs/crop_read_fixture_v2"]
        assert crop.attrs["stage_selector_eligible"] is False
        assert crop.attrs["production_candidate"] is True
        assert "roi_images" not in crop
        assert len(list(crop.arrays())) == 13

    original = zarr.open_group(str(source_archive), mode="r", use_consolidated=False)
    original_parent = original["refined_detect_runs"]
    assert "authoritative_run" not in original_parent.attrs
    assert original_parent[REFINED_RUN_ID].attrs["stage_selector_eligible"] is False
    assert (
        original_parent[REFINED_RUN_ID].attrs["run_manifest"]["payload_digest"]
        == source_manifest_digest
    )


def test_fixture_rejects_nonbenchmark_destination_before_copy(tmp_path: Path) -> None:
    source = _refined_source(tmp_path)
    recording, video = _video(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(ValueError, match=".palette_benchmarks"):
        publish_crop_geometry_read_fixture(
            source_refined_zarr=source.archive_path,
            source_refined_run_id=REFINED_RUN_ID,
            source_video=video,
            recording_path=recording,
            camera_identity="cam2010095",
            fps=100.0,
            codec="hevc",
            pixel_format="yuv420p",
            destination=tmp_path / "unsafe" / "fixture",
            work_root=scratch,
            crop_run_id="crop_read_fixture_v2",
            policy=_policy(),
        )

    assert not (tmp_path / "unsafe" / "fixture").exists()

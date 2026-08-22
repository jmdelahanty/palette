from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.cluster.keypoints.common import validate_crop_run_provider_record
from fisheye.utils import build_hybrid_acquisition_offline_crop_run as mod
from fisheye.utils.build_hybrid_acquisition_offline_crop_run import (
    ROUTING_REASON_CODE_MAP,
    _prepare_canonical_ledger_hybrid_payload,
    _prepare_hybrid_payload,
    build_hybrid_acquisition_offline_crop_run,
)
from fisheye.shared.acquisition_crop_stream_ledger import (
    publish_acquisition_crop_stream_ledger,
)
from fisheye.shared.crop_pixel_work_package import (
    SIGNED_SOURCE_BINDING_PROFILE,
    _source_binding,
)
from fisheye.shared.refined_detect_curation import extract_present_curated_rows
from fisheye.shared.row_source_signature import load_row_source_signature_spec
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    build_crop_run_reference,
)


def _create_array(group, name: str, data) -> None:
    group.create_array(name, data=np.asarray(data), overwrite=True)


def _make_hybrid_source_archive(tmp_path: Path) -> tuple[Path, Path]:
    recording_dir = tmp_path / "recording"
    cams_dir = recording_dir / "cams"
    cams_dir.mkdir(parents=True)
    source_video = cams_dir / "Cam123_recording.mp4"
    source_video.write_bytes(b"fake")
    crop_video = (
        recording_dir / "derived" / "external_crop_recorder" / "Cam123_crop.mp4"
    )
    crop_video.parent.mkdir(parents=True)
    crop_video.write_bytes(b"fake-crop")

    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 10
    root.attrs["height"] = 10
    root.attrs["total_frames"] = 4
    root.attrs["source_video_path"] = str(source_video)

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_acquisition"
    crop = crop_parent.create_group("crop_acquisition")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["source_pixels"] = "acquisition_crop_video"
    crop.attrs["roi_pixel_provider"] = "acquisition_crop_video"
    crop.attrs["source_crop_video_path"] = str(crop_video)
    crop.attrs["roi_size"] = [4, 4]
    _create_array(crop, "frame_indices", np.array([0, 2], dtype=np.int64))
    _create_array(crop, "instance_key", np.array([101, 103], dtype=np.uint64))
    _create_array(
        crop, "source_crop_video_frame_indices", np.array([0, 1], dtype=np.int64)
    )
    _create_array(crop, "source_crop_local_frame_ids", np.array([0, 2], dtype=np.int64))
    _create_array(
        crop, "source_crop_meta_row_indices", np.array([0, 2], dtype=np.int64)
    )
    _create_array(
        crop, "roi_coordinates_full", np.array([[1, 1], [3, 3]], dtype=np.int32)
    )
    _create_array(crop, "roi_sizes_full", np.array([[4, 4], [4, 4]], dtype=np.int32))
    _create_array(
        crop,
        "source_crop_xywh",
        np.array([[1, 1, 4, 4], [3, 3, 4, 4]], dtype=np.float32),
    )
    _create_array(
        crop, "bbox_img_xyxy", np.array([[2, 2, 4, 4], [4, 4, 6, 6]], dtype=np.float64)
    )
    _create_array(
        crop,
        "bbox_norm_coords",
        np.array([[0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
    )
    _create_array(
        crop, "bbox_roi_xyxy", np.array([[1, 1, 3, 3], [1, 1, 3, 3]], dtype=np.float64)
    )
    _create_array(
        crop,
        "bbox_crop_norm_coords",
        np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], dtype=np.float64),
    )

    refined_parent = root.create_group("refined_detect_runs")
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "intended_use": "analysis_and_training",
    }
    instances = refined.create_group("instances")
    _create_array(instances, "refined_row_ids", np.array([0, 1, 2], dtype=np.int64))
    _create_array(instances, "frame_indices", np.array([0, 1, 2], dtype=np.int32))
    _create_array(instances, "frame_offsets", np.array([0, 1, 2, 3, 3], dtype=np.int64))
    _create_array(instances, "instance_key", np.array([101, 102, 103], dtype=np.uint64))
    _create_array(
        instances,
        "bbox_img_xyxy",
        np.array([[2, 2, 4, 4], [6, 6, 8, 8], [4, 4, 6, 6]], dtype=np.float64),
    )
    _create_array(
        instances,
        "bbox_norm_coords",
        np.array(
            [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
            dtype=np.float64,
        ),
    )
    _create_array(instances, "source_kind_codes", np.array([0, 0, 0], dtype=np.int8))
    _create_array(
        instances, "manual_edit_flags", np.array([False, False, False], dtype=bool)
    )
    _create_array(
        instances, "source_detect_row_index", np.array([0, 1, 2], dtype=np.int64)
    )
    _create_array(instances, "frame_counts", np.array([1, 1, 1, 0], dtype=np.int32))
    return zarr_path, source_video


def _make_canonical_ledger_source_archive(tmp_path: Path) -> tuple[Path, Path, str]:
    recording_dir = tmp_path / "recording-ledger"
    cams = recording_dir / "cams"
    derived = recording_dir / "derived" / "external_crop_recorder"
    cams.mkdir(parents=True)
    derived.mkdir(parents=True)
    source_video = cams / "Cam123_recording.mp4"
    crop_video = derived / "Cam123_crop.mp4"
    source_video.write_bytes(b"full-video")
    crop_video.write_bytes(b"crop-video")
    crop_csv = derived / "Cam123_crop_meta.csv"
    columns = (
        "recording_frame_id,has_detection,blank_frame,crop_x,crop_y,crop_w,crop_h,"
        "detection_x,detection_y,detection_w,detection_h,detection_confidence,"
        "crop_video_frame_index,local_frame_id\n"
    )
    crop_csv.write_text(
        columns
        + "1,true,false,100,100,384,384,180,180,40,40,0.9,0,10\n"
        + "2,false,true,,,,,,,,,,1,11\n"
        + "3,true,false,600,600,384,384,650,650,40,40,0.8,2,12\n"
        + "4,true,false,700,700,384,384,740,740,40,40,0.7,3,13\n",
        encoding="utf-8",
    )
    manifest = {
        "camera_id": "Cam123",
        "video_streams": {
            "schema_id": "orange_runtime_video_streams_v1",
            "frame_clock": "recording_frame_id",
            "streams": {
                "crop": {
                    "stream_id": "crop",
                    "output_kind": "crop",
                    "camera_id": "Cam123",
                    "frame_clock": "recording_frame_id",
                    "video_pixel_coordinate_space": "crop_frame_pixels",
                    "source_geometry_coordinate_space": "full_frame_pixels",
                    "width": 384,
                    "height": 384,
                    "frame_count": 4,
                    "video": str(crop_video.relative_to(recording_dir)),
                    "metadata": str(crop_csv.relative_to(recording_dir)),
                }
            },
        },
    }
    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "width": 1000,
            "height": 1000,
            "total_frames": 4,
            "source_video_path": str(source_video),
            "source_video_fingerprint_strategy": "stat_v1",
            "source_video_fingerprint": "c" * 64,
            "source_video_size_bytes": source_video.stat().st_size,
            "source_video_mtime_ns": source_video.stat().st_mtime_ns,
        }
    )
    stream = (
        root.create_group("analysis")
        .create_group("acquisition_video_streams")
        .create_group("streams")
        .create_group("crop")
    )
    publication = publish_acquisition_crop_stream_ledger(
        stream,
        recording_dir,
        manifest,
        imported_at_utc="2026-08-15T12:00:00+00:00",
    )
    refined = root.create_group("refined_detect_runs").create_group(
        "refined_detect_ledger"
    )
    refined.attrs["detect_review_status"] = {
        "state": "approved",
        "intended_use": "analysis_and_training",
    }
    instances = refined.create_group("instances")
    _create_array(instances, "refined_row_ids", np.arange(4, dtype=np.int64))
    _create_array(instances, "frame_indices", np.arange(4, dtype=np.int32))
    _create_array(instances, "frame_offsets", np.arange(5, dtype=np.int64))
    _create_array(instances, "instance_key", np.arange(501, 505, dtype=np.uint64))
    _create_array(
        instances,
        "bbox_norm_coords",
        np.array(
            [
                [0.2, 0.2, 0.04, 0.04],
                [0.3, 0.3, 0.04, 0.04],
                [0.2, 0.2, 0.04, 0.04],
                [0.75, 0.75, 0.04, 0.04],
            ],
            dtype=np.float32,
        ),
    )
    _create_array(
        instances,
        "bbox_img_xyxy",
        np.array(
            [
                [180, 180, 220, 220],
                [280, 280, 320, 320],
                [180, 180, 220, 220],
                [730, 730, 770, 770],
            ],
            dtype=np.float32,
        ),
    )
    _create_array(instances, "source_kind_codes", np.zeros(4, dtype=np.int8))
    _create_array(instances, "manual_edit_flags", np.zeros(4, dtype=bool))
    _create_array(instances, "source_detect_row_index", np.arange(4, dtype=np.int64))
    _create_array(instances, "frame_counts", np.ones(4, dtype=np.int32))
    return zarr_path, source_video, publication.record_sha256


def test_build_hybrid_acquisition_offline_crop_run_dry_run_selects_offline_recovered_rows(
    tmp_path: Path,
) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)

    report = build_hybrid_acquisition_offline_crop_run(
        zarr_path,
        acquisition_source_mode="legacy_crop_run",
        acquisition_crop_run="crop_acquisition",
        refined_detect_run="refined_detect_001",
        run_name="crop_hybrid_test",
        source_video_path=source_video,
        apply=False,
    )

    assert report["status"] == "dry_run"
    assert report["summary"]["online_rows_available"] == 2
    assert report["summary"]["reviewed_refined_rows"] == 3
    assert report["summary"]["acquisition_video_rows_reused"] == 2
    assert report["summary"]["supplemental_rows_materialized"] == 1
    assert report["summary"]["supplemental_unmatched_instance_key"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "crop_hybrid_test" not in root["crop_runs"]


def _prepare_fixture_payload(zarr_path: Path) -> dict[str, np.ndarray | dict[str, int]]:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    acquisition = root["crop_runs/crop_acquisition"]
    refined = root["refined_detect_runs/refined_detect_001"]
    return _prepare_hybrid_payload(
        root=root,
        acquisition_group=acquisition,
        refined_payload=extract_present_curated_rows(refined),
        frame_width=10,
        frame_height=10,
        roi_shape=(4, 4),
    )


def test_reviewed_hybrid_payload_preserves_refined_authority_and_identity(
    tmp_path: Path,
) -> None:
    zarr_path, _source_video = _make_hybrid_source_archive(tmp_path)

    payload = _prepare_fixture_payload(zarr_path)

    np.testing.assert_array_equal(payload["instance_key"], [101, 102, 103])
    np.testing.assert_array_equal(payload["source_refined_row_ids"], [0, 1, 2])
    np.testing.assert_array_equal(payload["source_pixel_kind_codes"], [0, 1, 0])
    np.testing.assert_array_equal(
        payload["supplemental_cache_row_indices"], [-1, 0, -1]
    )
    np.testing.assert_array_equal(payload["frame_counts"], [1, 1, 1, 0])
    np.testing.assert_array_equal(payload["frame_row_offsets"], [0, 1, 2, 3, 3])
    np.testing.assert_allclose(
        payload["bbox_img_xyxy"],
        [[2, 2, 4, 4], [6, 6, 8, 8], [4, 4, 6, 6]],
        atol=1e-6,
    )


def test_reviewed_hybrid_payload_reuses_matching_crop_for_corrected_contained_box(
    tmp_path: Path,
) -> None:
    zarr_path, _source_video = _make_hybrid_source_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    instances = root["refined_detect_runs/refined_detect_001/instances"]
    boxes = np.asarray(instances["bbox_img_xyxy"][:], dtype=np.float64)
    boxes[0] = [2.5, 2.5, 4.5, 4.5]
    instances["bbox_img_xyxy"][:] = boxes
    norms = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64)
    norms[0] = [0.35, 0.35, 0.2, 0.2]
    instances["bbox_norm_coords"][:] = norms

    payload = _prepare_fixture_payload(zarr_path)

    assert int(np.asarray(payload["source_pixel_kind_codes"])[0]) == 0
    np.testing.assert_allclose(
        np.asarray(payload["bbox_img_xyxy"])[0],
        [2.5, 2.5, 4.5, 4.5],
    )
    np.testing.assert_allclose(
        np.asarray(payload["bbox_roi_xyxy"])[0],
        [1.5, 1.5, 3.5, 3.5],
    )


def test_reviewed_hybrid_payload_routes_incompatible_and_manual_rows_to_supplement(
    tmp_path: Path,
) -> None:
    zarr_path, _source_video = _make_hybrid_source_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    instances = root["refined_detect_runs/refined_detect_001/instances"]
    arrays = {
        "refined_row_ids": np.array([0, 3, 1, 2], dtype=np.int64),
        "frame_indices": np.array([0, 0, 1, 2], dtype=np.int32),
        "instance_key": np.array([101, 104, 102, 103], dtype=np.uint64),
        "bbox_img_xyxy": np.array(
            [[0, 0, 6, 6], [5, 5, 7, 7], [6, 6, 8, 8], [4, 4, 6, 6]],
            dtype=np.float64,
        ),
        "bbox_norm_coords": np.array(
            [
                [0.3, 0.3, 0.6, 0.6],
                [0.6, 0.6, 0.2, 0.2],
                [0.7, 0.7, 0.2, 0.2],
                [0.5, 0.5, 0.2, 0.2],
            ],
            dtype=np.float64,
        ),
        "source_kind_codes": np.array([0, 3, 0, 0], dtype=np.int8),
        "manual_edit_flags": np.array([True, True, False, False], dtype=bool),
        "source_detect_row_index": np.array([0, -1, 1, 2], dtype=np.int64),
        "frame_counts": np.array([2, 1, 1, 0], dtype=np.int32),
        "frame_offsets": np.array([0, 2, 3, 4, 4], dtype=np.int64),
    }
    for name, values in arrays.items():
        del instances[name]
        _create_array(instances, name, values)

    payload = _prepare_fixture_payload(zarr_path)

    np.testing.assert_array_equal(payload["instance_key"], [101, 104, 102, 103])
    np.testing.assert_array_equal(payload["source_pixel_kind_codes"], [1, 1, 1, 0])
    np.testing.assert_array_equal(
        payload["supplemental_cache_row_indices"], [0, 1, 2, -1]
    )
    np.testing.assert_array_equal(payload["frame_counts"], [2, 1, 1, 0])
    np.testing.assert_array_equal(payload["frame_row_offsets"], [0, 2, 3, 4, 4])
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["supplemental_unmatched_instance_key"] == 2
    assert summary["supplemental_reviewed_bbox_outside_acquisition_roi"] == 1
    assert summary["acquisition_video_rows_reused"] == 1


def test_build_hybrid_refuses_unapproved_refined_input(tmp_path: Path) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    del root["refined_detect_runs/refined_detect_001"].attrs["detect_review_status"]

    with pytest.raises(ValueError, match="requires an approved refined-detection"):
        build_hybrid_acquisition_offline_crop_run(
            zarr_path,
            acquisition_source_mode="legacy_crop_run",
            acquisition_crop_run="crop_acquisition",
            refined_detect_run="refined_detect_001",
            run_name="crop_hybrid_unapproved",
            source_video_path=source_video,
            apply=False,
        )


def test_build_hybrid_apply_keeps_reviewed_candidate_selector_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)

    def fake_write_supplemental_cache(**_kwargs):
        return {"schema": "fake.test.flat_roi_cache", "cache_complete": True}

    monkeypatch.setattr(mod, "_write_supplemental_cache", fake_write_supplemental_cache)

    report = build_hybrid_acquisition_offline_crop_run(
        zarr_path,
        acquisition_source_mode="legacy_crop_run",
        acquisition_crop_run="crop_acquisition",
        refined_detect_run="refined_detect_001",
        run_name="crop_hybrid_reviewed_v2",
        source_video_path=source_video,
        apply=True,
    )

    assert report["status"] == "ok"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["crop_runs"]
    run = parent["crop_hybrid_reviewed_v2"]
    assert run.attrs["schema_id"] == "palette.hybrid_acquisition_offline_crop_run.v3"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["bbox_authority"] == "reviewed_refined_detection"
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert parent.attrs["latest_any"] == "crop_acquisition"
    np.testing.assert_array_equal(run["instance_key"][:], [101, 102, 103])
    np.testing.assert_array_equal(
        run["source_acquisition_crop_row_indices"][:], [0, -1, 1]
    )
    assert "selected_live_detection_bbox_img_xyxy" not in run
    consolidated = zarr.open_group(str(zarr_path), mode="r", use_consolidated=True)
    consolidated_run = consolidated["crop_runs/crop_hybrid_reviewed_v2"]
    assert (
        consolidated_run.attrs["provider_record_sha256"]
        == report["provider_record_sha256"]
    )
    assert "unexpected_warning_count" in report["metadata_consolidation"]


def test_build_hybrid_consolidation_failure_marks_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)

    monkeypatch.setattr(
        mod,
        "_write_supplemental_cache",
        lambda **_kwargs: {
            "schema": "fake.test.flat_roi_cache",
            "cache_complete": True,
        },
    )
    calls = 0

    def fail_consolidation(_archive_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic consolidation failure")

    monkeypatch.setattr(
        mod, "consolidate_metadata_capture_expected_warnings", fail_consolidation
    )

    with pytest.raises(RuntimeError, match="synthetic consolidation failure"):
        build_hybrid_acquisition_offline_crop_run(
            zarr_path,
            acquisition_source_mode="legacy_crop_run",
            acquisition_crop_run="crop_acquisition",
            refined_detect_run="refined_detect_001",
            run_name="crop_hybrid_consolidation_failure",
            source_video_path=source_video,
            apply=True,
        )

    assert calls == 2
    direct = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    failed = direct["crop_runs/crop_hybrid_consolidation_failure"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert "palette_run_completed_at_utc" not in failed.attrs
    assert "synthetic consolidation failure" in failed.attrs["palette_run_error"]


def test_canonical_ledger_routing_uses_frame_identity_and_preserves_refined_keys(
    tmp_path: Path,
) -> None:
    zarr_path, _source_video, _ledger_digest = _make_canonical_ledger_source_archive(
        tmp_path
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    stream = root["analysis/acquisition_video_streams/streams/crop"]
    ledger = stream["ledger_runs"][stream.attrs["canonical_ledger_run"]]
    refined = root["refined_detect_runs/refined_detect_ledger"]

    payload = _prepare_canonical_ledger_hybrid_payload(
        root=root,
        ledger_group=ledger,
        refined_payload=extract_present_curated_rows(refined),
        frame_width=1000,
        frame_height=1000,
        roi_shape=(384, 384),
    )

    np.testing.assert_array_equal(payload["instance_key"], [501, 502, 503, 504])
    np.testing.assert_array_equal(
        payload["source_acquisition_crop_row_indices"], [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(payload["source_pixel_kind_codes"], [0, 1, 1, 1])
    np.testing.assert_array_equal(
        payload["routing_reason_codes"],
        [
            ROUTING_REASON_CODE_MAP["acquisition_crop_selected"],
            ROUTING_REASON_CODE_MAP["blank_acquisition_crop"],
            ROUTING_REASON_CODE_MAP["canonical_roi_not_contained"],
            ROUTING_REASON_CODE_MAP["coordinate_or_extent_mismatch"],
        ],
    )


def test_canonical_collection_ledger_propagates_exact_media_members(
    tmp_path: Path,
) -> None:
    zarr_path, _source_video, _ledger_digest = _make_canonical_ledger_source_archive(
        tmp_path
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    stream = root["analysis/acquisition_video_streams/streams/crop"]
    ledger = stream["ledger_runs"][stream.attrs["canonical_ledger_run"]]
    _create_array(
        ledger,
        "source_crop_video_member_indices",
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    _create_array(
        ledger,
        "source_full_video_member_indices",
        np.array([0, 0, 1, 1], dtype=np.int32),
    )
    refined = root["refined_detect_runs/refined_detect_ledger"]

    payload = _prepare_canonical_ledger_hybrid_payload(
        root=root,
        ledger_group=ledger,
        refined_payload=extract_present_curated_rows(refined),
        frame_width=1000,
        frame_height=1000,
        roi_shape=(384, 384),
    )

    np.testing.assert_array_equal(
        payload["source_crop_video_member_indices"], [0, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        payload["source_full_video_member_indices"], [0, 0, 1, 1]
    )
    np.testing.assert_array_equal(
        payload["supplemental_cache_row_indices"], [-1, 0, 1, 2]
    )
    np.testing.assert_allclose(
        np.asarray(payload["source_acquisition_crop_xywh"])[0],
        [100, 100, 384, 384],
    )
    np.testing.assert_allclose(
        np.asarray(payload["source_crop_xywh"])[0], [100, 100, 384, 384]
    )


def test_canonical_ledger_build_binds_provider_record_and_stays_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, source_video, ledger_digest = _make_canonical_ledger_source_archive(
        tmp_path
    )

    monkeypatch.setattr(
        mod,
        "_write_supplemental_cache",
        lambda **_kwargs: {
            "schema": "fake.test.flat_roi_cache",
            "cache_complete": True,
        },
    )
    report = build_hybrid_acquisition_offline_crop_run(
        zarr_path,
        acquisition_ledger_record_sha256=ledger_digest,
        refined_detect_run="refined_detect_ledger",
        run_name="crop_hybrid_ledger_v1",
        source_video_path=source_video,
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["acquisition_source_mode"] == "canonical_ledger"
    assert len(report["provider_record_sha256"]) == 64
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["crop_runs/crop_hybrid_ledger_v1"]
    assert run.attrs["schema_id"] == "palette.hybrid_acquisition_offline_crop_run.v3"
    assert run.attrs["crop_policy_id"] == "zebrafish_crop_384_v1"
    assert run.attrs["source_pixel_routing_policy"] == (
        "goodbatbadbat_crop_pixel_routing_v1"
    )
    assert run.attrs["source_acquisition_crop_ledger_record_sha256"] == ledger_digest
    assert run.attrs["provider_record_sha256"] == report["provider_record_sha256"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run["source_row_signature"].shape == (4, 32)
    assert run["source_row_signature"].dtype == np.dtype(np.uint8)
    assert run.attrs["source_rowset_fingerprint"] == report["source_rowset_fingerprint"]
    signature_spec = load_row_source_signature_spec(run.attrs)
    assert signature_spec.spec_digest == report["source_row_signature_spec_digest"]
    reference = build_crop_run_reference(
        run,
        run_id="crop_hybrid_ledger_v1",
    )
    assert reference["profile"] == CROP_RUN_REFERENCE_SIGNED_PROFILE
    package_binding = _source_binding(run, run_id="crop_hybrid_ledger_v1")
    assert package_binding["source_binding_profile"] == (SIGNED_SOURCE_BINDING_PROFILE)
    np.testing.assert_array_equal(run["instance_key"][:], [501, 502, 503, 504])
    np.testing.assert_array_equal(run["routing_reason_codes"][:], [0, 1, 4, 6])
    binding = validate_crop_run_provider_record(
        analysis_zarr=zarr_path,
        crop_run="crop_hybrid_ledger_v1",
        expected_record_sha256=report["provider_record_sha256"],
    )
    assert binding is not None
    assert binding["row_count"] == 4
    assert binding["acquisition_ledger_record_sha256"] == ledger_digest
    assert binding["source_rowset_fingerprint"] == report["source_rowset_fingerprint"]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_crop_run_provider_record(
            analysis_zarr=zarr_path,
            crop_run="crop_hybrid_ledger_v1",
            expected_record_sha256="f" * 64,
        )

    writable = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    signature_array = writable["crop_runs/crop_hybrid_ledger_v1/source_row_signature"]
    signature_array[0, 0] = np.uint8(int(signature_array[0, 0]) ^ 1)
    with pytest.raises(ValueError, match="source_row_signature payload is stale"):
        _source_binding(
            writable["crop_runs/crop_hybrid_ledger_v1"],
            run_id="crop_hybrid_ledger_v1",
        )
    with pytest.raises(ValueError, match="source_row_signature payload is stale"):
        validate_crop_run_provider_record(
            analysis_zarr=zarr_path,
            crop_run="crop_hybrid_ledger_v1",
            expected_record_sha256=report["provider_record_sha256"],
        )


def test_canonical_ledger_dry_run_does_not_create_crop_parent(
    tmp_path: Path,
) -> None:
    zarr_path, source_video, ledger_digest = _make_canonical_ledger_source_archive(
        tmp_path
    )
    before = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "crop_runs" not in before

    report = build_hybrid_acquisition_offline_crop_run(
        zarr_path,
        acquisition_ledger_record_sha256=ledger_digest,
        refined_detect_run="refined_detect_ledger",
        run_name="crop_hybrid_ledger_dry_run",
        source_video_path=source_video,
        apply=False,
    )

    assert report["status"] == "dry_run"
    after = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "crop_runs" not in after


def test_production_route_refuses_legacy_crop_run_without_explicit_mode(
    tmp_path: Path,
) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)

    with pytest.raises(ValueError, match="legacy_crop_run"):
        build_hybrid_acquisition_offline_crop_run(
            zarr_path,
            acquisition_crop_run="crop_acquisition",
            refined_detect_run="refined_detect_001",
            source_video_path=source_video,
            apply=False,
        )

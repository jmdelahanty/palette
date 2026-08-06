from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.tune import detect_training_promotion_backend as promotion_backend
from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
)
from fisheye.tune.detect_training_promotion_backend import (
    ClippedPromotionFrame,
    PromotionOptions,
    promote_detection_frames,
    promote_clipped_detection_frames,
)
from fisheye.utils import promote_analysis_detect_to_training as promote_cli


def _write_analysis_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"width": 4, "height": 4, "source_video_fps": 60})
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_ds",
        data=np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4),
        chunks=(1, 4, 4),
    )
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_a"
    refined = refined_parent.create_group("refined_a")
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([10, 11], dtype=np.int64), chunks=(2,))
    instances.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), chunks=(2,))
    instances.create_array("frame_offsets", data=np.array([0, 1, 2], dtype=np.int64), chunks=(3,))
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32),
        chunks=(2, 4),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.25, 0.25], [0.625, 0.625, 0.25, 0.25]], dtype=np.float32),
        chunks=(2, 4),
    )
    instances.create_array(
        "source_kind_codes",
        data=np.array([REFINED_SOURCE_KIND_CODE_MAP["manual"], REFINED_SOURCE_KIND_CODE_MAP["raw_detect"]], dtype=np.int8),
        chunks=(2,),
    )
    instances.create_array("manual_edit_flags", data=np.array([True, False], dtype=bool), chunks=(2,))
    instances.create_array("source_detect_row_index", data=np.array([100, 101], dtype=np.int32), chunks=(2,))
    instances.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32), chunks=(3,))
    instances.create_array("confidence_scores", data=np.array([1.0, 0.8], dtype=np.float32), chunks=(2,))
    instances.create_array("class_ids", data=np.array([0, 0], dtype=np.int32), chunks=(2,))

    source = refined.create_group("source_detections")
    source.create_array("source_detect_row_index", data=np.array([200], dtype=np.int32), chunks=(1,))
    source.create_array("frame_indices", data=np.array([2], dtype=np.int32), chunks=(1,))
    source.create_array("bbox_img_xyxy", data=np.full((1, 4), np.nan, dtype=np.float32), chunks=(1, 4))
    source.create_array("bbox_norm_coords", data=np.full((1, 4), np.nan, dtype=np.float32), chunks=(1, 4))
    source.create_array(
        "decision_codes",
        data=np.array([REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["manual_clear"]], dtype=np.int8),
        chunks=(1,),
    )
    source.create_array("resolved_refined_row_id", data=np.array([-1], dtype=np.int64), chunks=(1,))


def _write_empty_training_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", shape=(0, 4, 4), chunks=(1, 4, 4), dtype=np.uint8, fill_value=0)


def _write_empty_clipped_training_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", shape=(0, 4, 4), chunks=(1, 4, 4), dtype=np.uint8, fill_value=0)
    raw.create_array("images_full", shape=(0, 8, 8), chunks=(1, 8, 8), dtype=np.uint8, fill_value=0)


def _write_clipped_analysis_zarr(path: Path, *, extra_positive: bool = False) -> str:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({"width": 8, "height": 8, "source_video_fps": 30})
    refined_group_path = "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_clip"
    refined = root.create_group(refined_group_path)
    instances = refined.create_group("instances")
    refined_row_ids = np.array([55, 56], dtype=np.int64) if extra_positive else np.array([55], dtype=np.int64)
    frame_indices = np.array([5, 7], dtype=np.int32) if extra_positive else np.array([5], dtype=np.int32)
    frame_offsets = (
        np.array([0, 0, 0, 0, 0, 0, 1, 1, 2], dtype=np.int64)
        if extra_positive
        else np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    )
    bbox_img = (
        np.array([[2, 2, 6, 6], [1, 1, 5, 5]], dtype=np.float32)
        if extra_positive
        else np.array([[2, 2, 6, 6]], dtype=np.float32)
    )
    bbox_norm = (
        np.array([[0.5, 0.5, 0.5, 0.5], [0.375, 0.375, 0.5, 0.5]], dtype=np.float32)
        if extra_positive
        else np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    )
    row_count = int(refined_row_ids.shape[0])
    instances.create_array("refined_row_ids", data=refined_row_ids, chunks=(row_count,))
    instances.create_array("frame_indices", data=frame_indices, chunks=(row_count,))
    instances.create_array("frame_offsets", data=frame_offsets, chunks=(9,))
    instances.create_array(
        "bbox_img_xyxy",
        data=bbox_img,
        chunks=(row_count, 4),
    )
    instances.create_array(
        "bbox_norm_coords",
        data=bbox_norm,
        chunks=(row_count, 4),
    )
    instances.create_array(
        "source_kind_codes",
        data=np.full(row_count, REFINED_SOURCE_KIND_CODE_MAP["manual"], dtype=np.int8),
        chunks=(row_count,),
    )
    instances.create_array("manual_edit_flags", data=np.full(row_count, True, dtype=bool), chunks=(row_count,))
    instances.create_array(
        "source_detect_row_index",
        data=np.array([500, 502], dtype=np.int32) if extra_positive else np.array([500], dtype=np.int32),
        chunks=(row_count,),
    )
    instances.create_array(
        "frame_counts",
        data=np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int32) if extra_positive else np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int32),
        chunks=(8,),
    )
    instances.create_array("confidence_scores", data=np.full(row_count, 1.0, dtype=np.float32), chunks=(row_count,))
    instances.create_array("class_ids", data=np.zeros(row_count, dtype=np.int32), chunks=(row_count,))

    source = refined.create_group("source_detections")
    source.create_array("source_detect_row_index", data=np.array([501], dtype=np.int32), chunks=(1,))
    source.create_array("frame_indices", data=np.array([6], dtype=np.int32), chunks=(1,))
    source.create_array("bbox_img_xyxy", data=np.full((1, 4), np.nan, dtype=np.float32), chunks=(1, 4))
    source.create_array("bbox_norm_coords", data=np.full((1, 4), np.nan, dtype=np.float32), chunks=(1, 4))
    source.create_array(
        "decision_codes",
        data=np.array([REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["manual_clear"]], dtype=np.int8),
        chunks=(1,),
    )
    source.create_array("resolved_refined_row_id", data=np.array([-1], dtype=np.int64), chunks=(1,))
    return refined_group_path


def _write_clipped_collection(
    analysis_path: Path,
    *,
    refined_group_path: str,
    source_video: Path,
    frame_index_path: Path,
) -> None:
    root = zarr.open_group(str(analysis_path), mode="a")
    collection = root.create_group("experiment_index/finalized_runs/collection_a")
    collection.attrs["selected_runs"] = [
        {
            "work_unit_id": "clip_000000_cam2010093",
            "camera_serial": "2010093",
            "clip_id": "clip_000000",
            "clip_index": 0,
            "detect_run": "detect_clip",
            "detect_quality_run": "quality_clip",
            "refined_detect_run": "refined_clip",
            "detect_group_path": "clips/clip_000000/cameras/2010093/detect_runs/detect_clip",
            "refined_group_path": refined_group_path,
            "source": {"video_path": str(source_video)},
        }
    ]
    refined_parent = root.require_group("refined_detect_runs")
    refined_parent.attrs["latest_collection"] = "collection_a"
    table = pa.Table.from_pylist(
        [
            {
                "parent_frame_index": 105,
                "camera_serial": "2010093",
                "clip_id": "clip_000000",
                "clip_local_frame_index": 5,
                "recording_frame_id": 106,
                "video_path": str(source_video),
            },
            {
                "parent_frame_index": 106,
                "camera_serial": "2010093",
                "clip_id": "clip_000000",
                "clip_local_frame_index": 6,
                "recording_frame_id": 107,
                "video_path": str(source_video),
            },
        ]
    )
    pq.write_table(table, frame_index_path)


def _write_minimal_registry(path: Path, *, analysis: Path, training: Path, recording_id: str = "recording") -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE datasets (
                dataset_id TEXT PRIMARY KEY,
                recording_id TEXT,
                zarr_path TEXT NOT NULL,
                zarr_use TEXT,
                status TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO datasets (dataset_id, recording_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?, ?);",
            ("analysis_ds", recording_id, str(analysis.resolve()), "analysis", "active"),
        )
        conn.execute(
            "INSERT INTO datasets (dataset_id, recording_id, zarr_path, zarr_use, status) VALUES (?, ?, ?, ?, ?);",
            ("training_ds", recording_id, str(training.resolve()), "training", "active"),
        )
        conn.commit()
    finally:
        conn.close()


def test_promote_positive_frame_appends_then_no_changes(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)

    result = promote_detection_frames(analysis, training, [0], apply=True)

    assert result["status"] == "ok"
    assert result["action_counts"] == {"append": 1}
    root = zarr.open_group(str(training), mode="r")
    assert root["raw_video/images_ds"].shape == (1, 4, 4)
    np.testing.assert_array_equal(root["raw_video/images_ds"][0], np.arange(16, dtype=np.uint8).reshape(4, 4))
    crop_run = root["crop_runs"].attrs["latest"]
    crop = root[f"crop_runs/{crop_run}"]
    np.testing.assert_allclose(crop["bbox_norm_coords"][:], [[0.5, 0.5, 0.25, 0.25]])
    assert crop["label_state"][:].tolist() == ["positive"]
    assert crop["label_origin"][:].tolist() == ["manual_review"]
    assert crop.attrs["canonical_refined_detect_path"] == "refined_detect_runs/refined_detect_promoted_manual/instances"
    assert root["source_index/source_frame_idx"][:].tolist() == [0]
    assert root["source_index/source_refined_row_ids"][:].tolist() == [10]
    instances = root["refined_detect_runs/refined_detect_promoted_manual/instances"]
    assert instances["frame_indices"][:].tolist() == [0]
    np.testing.assert_allclose(instances["bbox_norm_coords"][:], [[0.5, 0.5, 0.25, 0.25]])
    assert instances["manual_edit_flags"][:].tolist() == [True]
    assert instances["source_detect_row_index"][:].tolist() == [-1]
    assert root["refined_detect_runs"].attrs["latest"] == "refined_detect_promoted_manual"

    dry_run = promote_detection_frames(analysis, training, [0], apply=False)

    assert dry_run["action_counts"] == {"no_change": 1}
    assert dry_run["items"][0]["target_row"] == 0


def test_frame_axis_promotion_rejects_duplicate_frames_before_identity_collapse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")
    refined = root.create_group("refined")
    payload = {
        "review_axis": np.asarray(["frame"], dtype=object),
        "frame_indices": np.asarray([4, 4], dtype=np.int32),
    }
    monkeypatch.setattr(
        promotion_backend.detect_review_mod,
        "_load_dense_curated_edit_payload",
        lambda _group, total_frames: payload,
    )

    with pytest.raises(RuntimeError, match="stable instance-key identity"):
        promotion_backend._load_refined_payload(
            root,
            refined_group=refined,
            frames=[4],
        )


def test_promotion_completion_routes_authority_through_approve(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)

    result = promote_detection_frames(analysis, training, [0], apply=True)

    sync = result["refined_instances_sync"]
    assert sync["authoritative_approval"]["status"] == "ok"
    assert sync["authoritative_approval"]["run"] == "refined_detect_promoted_manual"
    root = zarr.open_group(str(training), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    assert "detect_review_status_latest" not in parent.attrs
    assert parent.attrs["authoritative_run"] == "refined_detect_promoted_manual"
    provenance = dict(parent.attrs["authoritative_run_provenance"])
    assert provenance["approved_by"]
    assert provenance["note"] == "detect training promotion"
    status = dict(parent["refined_detect_promoted_manual"].attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "promotion"


def test_promoted_run_resolves_via_authoritative_run_resolution(tmp_path: Path) -> None:
    from fisheye.shared.run_resolution import RunResolution, resolve_run

    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)

    promote_detection_frames(analysis, training, [0], apply=True)

    root = zarr.open_group(str(training), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    resolved = resolve_run(parent, RunResolution.AUTHORITATIVE, parent_path="refined_detect_runs")
    assert resolved.run_name == "refined_detect_promoted_manual"
    assert resolved.resolution_source == "authoritative"
    assert resolved.fallback_used is False


def test_promotion_fails_loudly_when_authority_approval_blocked(tmp_path: Path, monkeypatch) -> None:
    import fisheye.cli.palette as palette_cli

    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)
    monkeypatch.setattr(
        palette_cli,
        "approve",
        lambda _request: {
            "status": "blocked",
            "reason_code": "RUN_NOT_COMPLETE",
            "next_hints": ["complete the run first"],
        },
    )

    with pytest.raises(RuntimeError, match="RUN_NOT_COMPLETE"):
        promote_detection_frames(analysis, training, [0], apply=True)

    root = zarr.open_group(str(training), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    assert "authoritative_run" not in parent.attrs
    assert "detect_review_status_latest" not in parent.attrs


def test_promote_manual_clear_frame_as_negative(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)

    result = promote_detection_frames(analysis, training, [2], apply=True)

    assert result["action_counts"] == {"append": 1}
    root = zarr.open_group(str(training), mode="r")
    crop = root[f"crop_runs/{root['crop_runs'].attrs['latest']}"]
    assert crop["label_state"][:].tolist() == ["negative"]
    assert np.isnan(crop["bbox_norm_coords"][:]).all()
    assert root["source_index/source_frame_idx"][:].tolist() == [2]
    assert root["source_index/source_detect_row_index"][:].tolist() == [200]
    instances = root["refined_detect_runs/refined_detect_promoted_manual/instances"]
    assert instances["frame_indices"][:].tolist() == []
    assert instances["frame_counts"][:].tolist() == [0]


def test_negative_frame_skips_when_negative_promotion_disabled(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    _write_analysis_zarr(analysis)
    _write_empty_training_zarr(training)

    result = promote_detection_frames(analysis, training, [2], options=PromotionOptions(include_negative=False), apply=False)

    assert result["action_counts"] == {"skip": 1}
    assert result["items"][0]["reason"] == "negative_promotion_disabled"


def test_dry_run_reports_append_when_training_zarr_missing(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "missing_training.zarr"
    _write_analysis_zarr(analysis)

    result = promote_detection_frames(analysis, training, [0], apply=False)

    assert result["training_zarr_exists"] is False
    assert result["action_counts"] == {"append": 1}
    assert not training.exists()


def test_promote_clipped_positive_frame_appends_with_clip_identity(tmp_path: Path, monkeypatch) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis)
    _write_empty_clipped_training_zarr(training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")

    monkeypatch.setattr(
        promotion_backend,
        "_decode_video_frame_gray_native",
        lambda _path, frame_idx: np.full((8, 8), int(frame_idx), dtype=np.uint8),
    )

    frame = ClippedPromotionFrame(
        parent_frame_index=105,
        clip_local_frame_index=5,
        refined_group_path=refined_group_path,
        refined_run="refined_clip",
        collection_id="collection_a",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010093",
        recording_frame_id=106,
        source_video_path=source_video,
    )
    result = promote_clipped_detection_frames(analysis, training, [frame], apply=True)

    assert result["status"] == "ok"
    assert result["mode"] == "clipped"
    assert result["action_counts"] == {"append": 1}
    root = zarr.open_group(str(training), mode="r")
    assert root["raw_video/images_ds"].shape == (1, 4, 4)
    assert root["raw_video/images_full"].shape == (1, 8, 8)
    np.testing.assert_array_equal(root["raw_video/images_ds"][0], np.full((4, 4), 5, dtype=np.uint8))
    np.testing.assert_array_equal(root["raw_video/images_full"][0], np.full((8, 8), 5, dtype=np.uint8))
    assert root["raw_video/original_frame_indices"][:].tolist() == [105]
    crop = root[f"crop_runs/{root['crop_runs'].attrs['latest']}"]
    np.testing.assert_allclose(crop["bbox_norm_coords"][:], [[0.5, 0.5, 0.5, 0.5]])
    assert crop["label_state"][:].tolist() == ["positive"]
    assert crop.attrs["canonical_refined_detect_path"] == "refined_detect_runs/refined_detect_promoted_manual/instances"
    source = root["source_index"]
    assert source["source_collection_id"][:].tolist() == ["collection_a"]
    assert source["source_refined_group_path"][:].tolist() == [refined_group_path]
    assert source["source_clip_id"][:].tolist() == ["clip_000000"]
    assert source["source_camera_serial"][:].tolist() == ["2010093"]
    assert source["source_frame_idx"][:].tolist() == [105]
    assert source["source_parent_frame_index"][:].tolist() == [105]
    assert source["source_clip_local_frame_index"][:].tolist() == [5]
    assert source["source_recording_frame_id"][:].tolist() == [106]
    assert source["source_refined_row_ids"][:].tolist() == [55]
    assert source["source_detect_row_index"][:].tolist() == [500]
    instances = root["refined_detect_runs/refined_detect_promoted_manual/instances"]
    assert instances["frame_indices"][:].tolist() == [0]
    np.testing.assert_allclose(instances["bbox_norm_coords"][:], [[0.5, 0.5, 0.5, 0.5]])
    assert instances["manual_edit_flags"][:].tolist() == [True]
    assert instances["source_detect_row_index"][:].tolist() == [-1]
    refined_parent = root["refined_detect_runs"]
    assert "detect_review_status_latest" not in refined_parent.attrs
    assert refined_parent.attrs["authoritative_run"] == "refined_detect_promoted_manual"

    dry_run = promote_clipped_detection_frames(analysis, training, [frame], apply=False)

    assert dry_run["action_counts"] == {"no_change": 1}
    assert dry_run["items"][0]["target_row"] == 0


def test_promote_clipped_appends_decode_same_clip_in_one_batch(tmp_path: Path, monkeypatch) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis, extra_positive=True)
    _write_empty_clipped_training_zarr(training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")
    batch_calls: list[tuple[Path, list[int]]] = []

    def fake_batch_decode(path: Path, frame_indices: list[int]) -> tuple[dict[int, np.ndarray], str]:
        batch_calls.append((path, list(frame_indices)))
        return {
            int(frame_idx): np.full((8, 8), int(frame_idx), dtype=np.uint8)
            for frame_idx in frame_indices
        }, "test_batch_decode"

    monkeypatch.setattr(promotion_backend, "_decode_video_frames_gray_native_batch", fake_batch_decode)

    frames = [
        ClippedPromotionFrame(
            parent_frame_index=105,
            clip_local_frame_index=5,
            refined_group_path=refined_group_path,
            refined_run="refined_clip",
            collection_id="collection_a",
            clip_id="clip_000000",
            clip_index=0,
            camera_serial="2010093",
            recording_frame_id=106,
            source_video_path=source_video,
        ),
        ClippedPromotionFrame(
            parent_frame_index=107,
            clip_local_frame_index=7,
            refined_group_path=refined_group_path,
            refined_run="refined_clip",
            collection_id="collection_a",
            clip_id="clip_000000",
            clip_index=0,
            camera_serial="2010093",
            recording_frame_id=108,
            source_video_path=source_video,
        ),
    ]

    result = promote_clipped_detection_frames(analysis, training, frames, apply=True)

    assert result["action_counts"] == {"append": 2}
    assert batch_calls == [(source_video.resolve(), [5, 7])]
    assert result["decode_groups"][0]["decode_method"] == "test_batch_decode"
    assert result["decode_groups"][0]["requested_frame_count"] == 2
    assert [source["decode_method"] for source in result["image_sources"]] == [
        "test_batch_decode",
        "test_batch_decode",
    ]
    root = zarr.open_group(str(training), mode="r")
    np.testing.assert_array_equal(root["raw_video/images_ds"][0], np.full((4, 4), 5, dtype=np.uint8))
    np.testing.assert_array_equal(root["raw_video/images_ds"][1], np.full((4, 4), 7, dtype=np.uint8))
    np.testing.assert_array_equal(root["raw_video/images_full"][0], np.full((8, 8), 5, dtype=np.uint8))
    np.testing.assert_array_equal(root["raw_video/images_full"][1], np.full((8, 8), 7, dtype=np.uint8))
    assert root["source_index/source_clip_local_frame_index"][:].tolist() == [5, 7]
    instances = root["refined_detect_runs/refined_detect_promoted_manual/instances"]
    assert instances["frame_indices"][:].tolist() == [0, 1]
    assert instances["frame_counts"][:].tolist() == [1, 1]


def test_promote_clipped_creates_images_full_when_missing(tmp_path: Path, monkeypatch) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis)
    _write_empty_training_zarr(training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")

    monkeypatch.setattr(
        promotion_backend,
        "_decode_video_frames_gray_native_batch",
        lambda _path, frame_indices: (
            {int(frame_idx): np.full((8, 8), int(frame_idx), dtype=np.uint8) for frame_idx in frame_indices},
            "test_batch_decode",
        ),
    )

    frame = ClippedPromotionFrame(
        parent_frame_index=105,
        clip_local_frame_index=5,
        refined_group_path=refined_group_path,
        refined_run="refined_clip",
        collection_id="collection_a",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010093",
        recording_frame_id=106,
        source_video_path=source_video,
    )

    result = promote_clipped_detection_frames(analysis, training, [frame], apply=True)

    assert result["action_counts"] == {"append": 1}
    root = zarr.open_group(str(training), mode="r")
    assert root["raw_video/images_full"].shape == (1, 8, 8)
    assert root["raw_video/images_full"].attrs["source"] == "clipped_promotion_source_video_luma"
    np.testing.assert_array_equal(root["raw_video/images_full"][0], np.full((8, 8), 5, dtype=np.uint8))


def test_promote_clipped_manual_clear_as_negative(tmp_path: Path, monkeypatch) -> None:
    analysis = tmp_path / "analysis.zarr"
    training = tmp_path / "training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis)
    _write_empty_clipped_training_zarr(training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")

    monkeypatch.setattr(
        promotion_backend,
        "_decode_video_frame_gray_native",
        lambda _path, frame_idx: np.full((8, 8), int(frame_idx), dtype=np.uint8),
    )

    frame = ClippedPromotionFrame(
        parent_frame_index=106,
        clip_local_frame_index=6,
        refined_group_path=refined_group_path,
        refined_run="refined_clip",
        collection_id="collection_a",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010093",
        recording_frame_id=107,
        source_video_path=source_video,
    )
    result = promote_clipped_detection_frames(analysis, training, [frame], apply=True)

    assert result["action_counts"] == {"append": 1}
    root = zarr.open_group(str(training), mode="r")
    crop = root[f"crop_runs/{root['crop_runs'].attrs['latest']}"]
    assert crop["label_state"][:].tolist() == ["negative"]
    assert np.isnan(crop["bbox_norm_coords"][:]).all()
    assert root["source_index/source_parent_frame_index"][:].tolist() == [106]
    assert root["source_index/source_clip_local_frame_index"][:].tolist() == [6]
    assert root["source_index/source_detect_row_index"][:].tolist() == [501]
    instances = root["refined_detect_runs/refined_detect_promoted_manual/instances"]
    assert instances["frame_indices"][:].tolist() == []
    assert instances["frame_counts"][:].tolist() == [0]


def test_cli_clipped_backfill_defaults_to_manual_positive_and_negative(tmp_path: Path, capsys) -> None:
    analysis = tmp_path / "recording_analysis.zarr"
    training = tmp_path / "recording_training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis)
    _write_empty_clipped_training_zarr(training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")
    frame_index_path = tmp_path / "recording_frame_index.parquet"
    _write_clipped_collection(
        analysis,
        refined_group_path=refined_group_path,
        source_video=source_video,
        frame_index_path=frame_index_path,
    )

    exit_code = promote_cli.main(
        [
            str(analysis),
            "--collection-id",
            "collection_a",
            "--recording-frame-index",
            str(frame_index_path),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["mode"] == "clipped"
    assert payload["apply"] is False
    assert payload["training_zarr"] == str(training.resolve())
    assert payload["training_zarr_inferred"] is True
    assert payload["discovery"]["manual_only"] is True
    assert payload["discovery"]["discovered_frame_count"] == 2
    assert payload["action_counts"] == {"append": 2}
    assert [item["parent_frame"] for item in payload["items"]] == [105, 106]
    assert [item["label_state"] for item in payload["items"]] == ["positive", "negative"]


def test_cli_uses_registry_training_target_for_clipped_backfill(tmp_path: Path, capsys) -> None:
    analysis = tmp_path / "recording_analysis.zarr"
    inferred_training = tmp_path / "recording_training.zarr"
    registry_training = tmp_path / "recording_clipped_training.zarr"
    refined_group_path = _write_clipped_analysis_zarr(analysis)
    _write_empty_clipped_training_zarr(registry_training)
    source_video = tmp_path / "clip.mp4"
    source_video.write_bytes(b"fake")
    frame_index_path = tmp_path / "recording_frame_index.parquet"
    registry_path = tmp_path / "registry.sqlite"
    _write_clipped_collection(
        analysis,
        refined_group_path=refined_group_path,
        source_video=source_video,
        frame_index_path=frame_index_path,
    )
    _write_minimal_registry(registry_path, analysis=analysis, training=registry_training)

    exit_code = promote_cli.main(
        [
            str(analysis),
            "--use-registry-target",
            "--registry",
            str(registry_path),
            "--collection-id",
            "collection_a",
            "--recording-frame-index",
            str(frame_index_path),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["training_zarr"] == str(registry_training.resolve())
    assert payload["training_zarr"] != str(inferred_training.resolve())
    assert payload["training_zarr_source"] == "registry"
    assert payload["training_zarr_inferred"] is False
    assert payload["registry_target"]["analysis_dataset_id"] == "analysis_ds"
    assert payload["registry_target"]["training_dataset_id"] == "training_ds"

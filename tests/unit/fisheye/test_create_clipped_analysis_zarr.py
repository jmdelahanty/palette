from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import zarr

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.zarr.crop_pixel_authority import bind_crop_pixel_authority
from fisheye.cluster.native_detection_authority import load_native_archive_authority
from fisheye.utils.build_recording_frame_index import build_recording_frame_index
from fisheye.utils.create_clipped_analysis_zarr import create_clipped_analysis_zarr
from fisheye.utils.repair_clipped_analysis_acquisition_authority import (
    repair_clipped_analysis_acquisition_authority,
)


@pytest.fixture(autouse=True)
def _stub_video_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(path: Path) -> dict[str, object]:
        keyframe = path.with_name(f"{path.stem}_keyframe.json")
        payload = json.loads(keyframe.read_text(encoding="utf-8"))
        return {
            "width": 640,
            "height": 640,
            "total_frames": int(payload["total_frames"]),
            "fps": float(payload["fps"]),
            "codec": str(payload.get("codec") or "hevc"),
            "pix_fmt": "yuv420p",
        }

    monkeypatch.setattr(
        "fisheye.shared.import_video_metadata.probe_ffprobe_video_metadata",
        _probe,
    )


def _write_metadata(path: Path, frame_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "timestamp", "timestamp_sys"]
        )
        writer.writeheader()
        for idx, frame_id in enumerate(frame_ids):
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": 1000 + idx,
                    "timestamp_sys": 2000 + idx,
                }
            )


def _write_clip_bundle(
    root: Path, clip_id: str, frame_ids: list[int]
) -> dict[str, object]:
    clip_index = int(clip_id.rsplit("_", 1)[1])
    clip_dir = root / "clips" / clip_id
    video = clip_dir / "Cam2010093_example.mp4"
    metadata = clip_dir / "Cam2010093_example_meta.csv"
    keyframe = clip_dir / "Cam2010093_example_keyframe.json"
    manifest = clip_dir / "clip_manifest.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    _write_metadata(metadata, frame_ids)
    keyframe.write_text(
        json.dumps({"total_frames": len(frame_ids), "fps": 2, "keyframe_frames": [0]}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"clip_id": clip_id}), encoding="utf-8")
    return {
        "recording_id": "rec_a",
        "session_id": "rec_a",
        "producer": "test",
        "recording_backend_mode": "materialized_stream_copy",
        "camera_serial": "2010093",
        "clip_index": clip_index,
        "clip_id": clip_id,
        "clip_directory": f"clips/{clip_id}",
        "video_path": f"clips/{clip_id}/{video.name}",
        "metadata_path": f"clips/{clip_id}/{metadata.name}",
        "keyframe_path": f"clips/{clip_id}/{keyframe.name}",
        "clip_manifest_path": f"clips/{clip_id}/clip_manifest.json",
        "frame_count": len(frame_ids),
        "first_recording_frame_id": frame_ids[0],
        "last_recording_frame_id": frame_ids[-1],
        "first_clip_local_frame_index": 0,
        "last_clip_local_frame_index": len(frame_ids) - 1,
    }


def _write_clipped_recording(root: Path) -> None:
    first = _write_clip_bundle(root, "clip_000000", [1, 2, 3])
    second = _write_clip_bundle(root, "clip_000001", [4, 5])
    (root / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "recording_id": "rec_a",
                "session_id": "rec_a",
                "producer": "test_index",
                "recording_backend_mode": "materialized_stream_copy",
                "clips": [first, second],
            }
        ),
        encoding="utf-8",
    )
    build_recording_frame_index(root)


def _write_analysis_context_source(
    path: Path,
    *,
    dish_mask: dict[str, object] | None,
    experiment_setup: dict[str, object] | None = None,
) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    analysis = root.create_group("analysis_metadata")
    if dish_mask is not None:
        analysis.attrs["dish_mask"] = dish_mask
    if experiment_setup is not None:
        root.attrs["experiment_setup"] = experiment_setup


def test_create_clipped_analysis_zarr_writes_shell_layout(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is True
    assert result["clip_count"] == 2
    assert result["clip_camera_row_count"] == 2
    assert result["recording_frame_index_row_count"] == 5
    assert Path(result["shell_manifest_path"]).exists()

    root = zarr.open_group(str(output_zarr), mode="r")
    assert root.attrs["analysis_layout"] == "clipped_recording_shell"
    assert root.attrs["source_layout"] == "rolling_clips"
    assert root.attrs["recording_frame_index_row_count"] == 5
    assert root.attrs["recording_frame_id_min"] == 1
    assert root.attrs["recording_frame_id_max"] == 5
    assert root["raw_video"].attrs["storage_mode"] == "external_clips"
    status = load_acquisition_authority_publication_status(root)
    assert status.status == ACQUISITION_AUTHORITY_PUBLISHED
    assert status.authority_mode == CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
    ownership, acquisition = load_persisted_acquisition_camera_authority(
        root, expected_camera_id="2010093"
    )
    assert ownership.record.mode == CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
    assert acquisition.record.source_total_frames == 5
    assert acquisition.record.width_px == 640
    assert acquisition.record.height_px == 640
    assert acquisition.record.frame_domain["mode"] == (
        "external_clipped_recording_frame_index_v1"
    )
    native = load_native_archive_authority(
        SimpleNamespace(recording_id="rec_a", analysis_zarr=output_zarr)
    )
    assert native.camera_serial == "2010093"
    assert native.n_frames == 5
    assert native.source_width == 640
    assert native.source_height == 640
    assert root["detect_runs"].attrs["scope"] == "parent_finalized_or_aggregated"
    assert root["analysis_metadata"].attrs["dish_mask_scope"] == "recording_camera"
    assert root["analysis_metadata"].attrs["dish_mask_required_per_clip"] is False
    assert (
        root["analysis_metadata"].attrs["orange_fixed_dish_location_invariant"] is True
    )

    clip_group = root["clips"]["clip_000000"]
    assert clip_group.attrs["granularity"] == "clip"
    camera_group = clip_group["cameras"]["2010093"]
    assert camera_group.attrs["granularity"] == "clip_camera"
    assert camera_group.attrs["frame_count"] == 3
    assert camera_group.attrs["dish_mask_scope"] == "recording_camera"
    assert (
        camera_group.attrs["dish_mask_clip_policy"]
        == "single_camera_mask_applies_to_all_clips"
    )
    assert (
        camera_group["source"]
        .attrs["video_path"]
        .endswith("clips/clip_000000/Cam2010093_example.mp4")
    )
    assert camera_group["source"]["frame_map"].attrs[
        "recording_frame_id_semantics"
    ] == ("session_continuous_recording_frame_id")
    assert camera_group["detect_runs"].attrs["latest"] is None
    assert camera_group["detect_runs"].attrs["scope"] == "clip_camera"
    assert root["experiment_index"]["clip_table"].attrs["row_count"] == 2


def test_created_clipped_archive_binds_as_crop_pixel_authority(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    bound = bind_crop_pixel_authority(
        output_zarr,
        expected_recording_identity="rec_a",
        expected_camera_identity="2010093",
        expected_n_frames=5,
        expected_source_width=640,
        expected_source_height=640,
    )

    assert bound.source_video_path is None
    assert len(bound.source_video_paths) == 2
    assert len(bound.source_index_paths) == 3
    assert bound.binding_document["provider_profile"] == (
        "published_external_clipped_video_collection_v1"
    )
    bound.assert_verified()


def test_repair_clipped_analysis_authority_dry_run_apply_and_idempotence(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    root = zarr.open_group(output_zarr, mode="a", use_consolidated=False)
    del root["analysis/acquisition_camera_frames"]
    del root.attrs["acquisition_authority_publication_status"]
    del root["raw_video"].attrs["acquisition_authority_publication_status"]
    del root.attrs["camera_id"]
    del root.attrs["source_video_metadata"]

    dry_run = repair_clipped_analysis_acquisition_authority(output_zarr)
    assert dry_run["status"] == "ok"
    assert dry_run["action"] == "would_publish"
    assert dry_run["acquisition_source"]["member_count"] == 2

    applied = repair_clipped_analysis_acquisition_authority(output_zarr, apply=True)
    assert applied["status"] == "ok"
    assert applied["action"] == "publish"

    second = repair_clipped_analysis_acquisition_authority(output_zarr, apply=True)
    assert second["status"] == "ok"
    assert second["action"] == "already_complete"


def test_repair_clipped_analysis_authority_rejects_changed_clip(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)
    changed = root_dir / "clips" / "clip_000001" / "Cam2010093_example.mp4"
    changed.write_bytes(b"changed-video")

    report = repair_clipped_analysis_acquisition_authority(output_zarr, apply=True)

    assert report["status"] == "blocked"
    assert "source_video_metadata conflicts" in report["error"]


def test_create_clipped_analysis_zarr_rejects_member_geometry_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"

    def _probe(path: Path) -> dict[str, object]:
        return {
            "width": 800 if "clip_000001" in str(path) else 640,
            "height": 640,
            "total_frames": 2 if "clip_000001" in str(path) else 3,
            "fps": 2.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        }

    monkeypatch.setattr(
        "fisheye.shared.import_video_metadata.probe_ffprobe_video_metadata",
        _probe,
    )

    with pytest.raises(ValueError, match="geometry changes"):
        create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)
    assert not output_zarr.exists()


def test_create_clipped_analysis_zarr_dry_run_does_not_write(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(
        root_dir, output_zarr=output_zarr, dry_run=True
    )

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is False
    assert result["wrote_manifest"] is False
    assert not output_zarr.exists()


def test_create_clipped_analysis_zarr_refuses_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    with pytest.raises(FileExistsError):
        create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    result = create_clipped_analysis_zarr(
        root_dir, output_zarr=output_zarr, overwrite=True
    )

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is True


def test_create_clipped_analysis_zarr_discovers_and_copies_recording_context(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    source_zarr = root_dir / "zarr" / "rec_a_training.zarr"
    dish_mask = {
        "shape": "circle",
        "detected_circle": {"center": [332, 326], "radius": 300},
        "metrics": {
            "image_shape": [640, 640],
            "center_norm": [0.51875, 0.509375],
            "radius_norm": 0.46875,
        },
    }
    experiment_setup = {
        "num_dishes": 1,
        "fish_per_dish": 1,
        "total_expected_fish": 1,
        "setup_type": "single_dish",
        "source": "test",
    }
    _write_analysis_context_source(
        source_zarr,
        dish_mask=dish_mask,
        experiment_setup=experiment_setup,
    )
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(
        root_dir,
        output_zarr=output_zarr,
        require_dish_mask=True,
    )

    assert result["analysis_context_source"] == {
        "source_zarr": str(source_zarr),
        "has_dish_mask": True,
        "has_experiment_setup": True,
        "experiment_setup_source_zarr": None,
    }
    root = zarr.open_group(str(output_zarr), mode="r")
    analysis = root["analysis_metadata"]
    assert analysis.attrs["dish_mask"] == dish_mask
    assert analysis.attrs["dish_mask_source_zarr"] == str(source_zarr)
    assert analysis.attrs["dish_mask_source_key"] == "analysis_metadata.attrs.dish_mask"
    assert (
        analysis.attrs["dish_mask_copy_tool"]
        == "fisheye.utils.create_clipped_analysis_zarr"
    )
    assert root.attrs["experiment_setup"] == experiment_setup
    assert root.attrs["experiment_setup_source_zarr"] == str(source_zarr)


def test_create_clipped_analysis_zarr_require_dish_mask_fails_before_write(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"

    with pytest.raises(ValueError, match="no sibling Zarr supplies"):
        create_clipped_analysis_zarr(
            root_dir,
            output_zarr=output_zarr,
            require_dish_mask=True,
            dry_run=True,
        )

    assert not output_zarr.exists()


def test_create_clipped_analysis_zarr_rejects_conflicting_discovered_masks(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    zarr_dir = root_dir / "zarr"
    _write_analysis_context_source(
        zarr_dir / "rec_a_training.zarr",
        dish_mask={
            "shape": "circle",
            "detected_circle": {"center": [1, 2], "radius": 3},
        },
    )
    _write_analysis_context_source(
        zarr_dir / "rec_a_clipped_training.zarr",
        dish_mask={
            "shape": "circle",
            "detected_circle": {"center": [4, 5], "radius": 6},
        },
    )

    with pytest.raises(ValueError, match="Conflicting sibling dish masks"):
        create_clipped_analysis_zarr(
            root_dir,
            output_zarr=zarr_dir / "rec_a_analysis.zarr",
            dry_run=True,
        )


def test_create_clipped_analysis_zarr_explicit_source_can_disable_discovery(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    source_zarr = tmp_path / "chosen_source.zarr"
    chosen_mask = {
        "shape": "circle",
        "detected_circle": {"center": [10, 11], "radius": 12},
    }
    _write_analysis_context_source(source_zarr, dish_mask=chosen_mask)
    output_zarr = root_dir / "zarr" / "rec_a_analysis.zarr"

    create_clipped_analysis_zarr(
        root_dir,
        output_zarr=output_zarr,
        copy_analysis_metadata_from=source_zarr,
        require_dish_mask=True,
        auto_discover_analysis_metadata=False,
    )

    root = zarr.open_group(str(output_zarr), mode="r")
    assert root["analysis_metadata"].attrs["dish_mask"] == chosen_mask
    assert root["analysis_metadata"].attrs["dish_mask_source_zarr"] == str(source_zarr)

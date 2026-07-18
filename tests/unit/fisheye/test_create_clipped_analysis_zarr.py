from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import zarr

from fisheye.utils.build_recording_frame_index import build_recording_frame_index
from fisheye.utils.create_clipped_analysis_zarr import create_clipped_analysis_zarr


def _write_metadata(path: Path, frame_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx, frame_id in enumerate(frame_ids):
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": 1000 + idx,
                    "timestamp_sys": 2000 + idx,
                }
            )


def _write_clip_bundle(root: Path, clip_id: str, frame_ids: list[int]) -> dict[str, object]:
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
    assert root["detect_runs"].attrs["scope"] == "parent_finalized_or_aggregated"
    assert root["analysis_metadata"].attrs["dish_mask_scope"] == "recording_camera"
    assert root["analysis_metadata"].attrs["dish_mask_required_per_clip"] is False
    assert root["analysis_metadata"].attrs["orange_fixed_dish_location_invariant"] is True

    clip_group = root["clips"]["clip_000000"]
    assert clip_group.attrs["granularity"] == "clip"
    camera_group = clip_group["cameras"]["2010093"]
    assert camera_group.attrs["granularity"] == "clip_camera"
    assert camera_group.attrs["frame_count"] == 3
    assert camera_group.attrs["dish_mask_scope"] == "recording_camera"
    assert camera_group.attrs["dish_mask_clip_policy"] == "single_camera_mask_applies_to_all_clips"
    assert camera_group["source"].attrs["video_path"].endswith("clips/clip_000000/Cam2010093_example.mp4")
    assert camera_group["source"]["frame_map"].attrs["recording_frame_id_semantics"] == (
        "session_continuous_recording_frame_id"
    )
    assert camera_group["detect_runs"].attrs["latest"] is None
    assert camera_group["detect_runs"].attrs["scope"] == "clip_camera"
    assert root["experiment_index"]["clip_table"].attrs["row_count"] == 2


def test_create_clipped_analysis_zarr_dry_run_does_not_write(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr, dry_run=True)

    assert result["status"] == "ok"
    assert result["wrote_zarr"] is False
    assert result["wrote_manifest"] is False
    assert not output_zarr.exists()


def test_create_clipped_analysis_zarr_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    root_dir = tmp_path / "rec_a"
    _write_clipped_recording(root_dir)
    output_zarr = tmp_path / "rec_a_analysis.zarr"
    create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    with pytest.raises(FileExistsError):
        create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr)

    result = create_clipped_analysis_zarr(root_dir, output_zarr=output_zarr, overwrite=True)

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
    assert analysis.attrs["dish_mask_copy_tool"] == "fisheye.utils.create_clipped_analysis_zarr"
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
        dish_mask={"shape": "circle", "detected_circle": {"center": [1, 2], "radius": 3}},
    )
    _write_analysis_context_source(
        zarr_dir / "rec_a_clipped_training.zarr",
        dish_mask={"shape": "circle", "detected_circle": {"center": [4, 5], "radius": 6}},
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

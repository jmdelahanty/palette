from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr import training_base_publication as publication_mod
from fisheye.utils.import_sampled_training_pynvvc import (
    SampledTrainingImportResult,
)


def _fake_import(*, zarr_path: Path, **kwargs) -> SampledTrainingImportResult:
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update({"zarr_purpose": "training", "zarr_use": "training"})
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5),
        chunks=(1, 4, 5),
        compressors=[],
    )
    raw.create_array(
        "images_ds",
        data=np.arange(3 * 2 * 3, dtype=np.uint8).reshape(3, 2, 3),
        chunks=(1, 2, 3),
        compressors=[],
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 3, 6], dtype=np.int32),
        chunks=(3,),
    )
    return SampledTrainingImportResult(
        zarr_path=zarr_path,
        source_video_path=Path(kwargs["video_path"]),
        imported_frame_count=3,
        source_frame_count=10,
        frame_step=3,
        skip_tail_frames=1,
        original_resolution=(4, 5),
        downsampled_resolution=(2, 3),
        decode_backend="test",
        duration_s=0.1,
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    scratch = tmp_path / "node" / "job"
    scratch.mkdir(parents=True)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    recording = tmp_path / "recording"
    recording.mkdir()
    h5 = recording / "raw.h5"
    h5.write_bytes(b"h5")
    config = tmp_path / "config.yaml"
    config.write_text("import: {}\n", encoding="utf-8")
    return {
        "destination": tmp_path / "published" / "training.zarr",
        "scratch_root": scratch,
        "video_path": video,
        "source_frame_count": 10,
        "frame_step": 3,
        "skip_tail_frames": 1,
        "config_path": config,
        "camera_id": "2010094",
        "recording_dir": recording,
        "h5_path": h5,
        "require_cuda": False,
    }


def test_publish_sampled_training_base_is_atomic_and_ineligible(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        publication_mod, "import_sampled_training_pynvvc", _fake_import
    )
    inputs = _inputs(tmp_path)
    result = publication_mod.publish_sampled_training_base(**inputs)

    destination = Path(inputs["destination"])
    root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    assert result["status"] == "complete"
    assert result["validation"]["full_chunks"] == [1, 4, 5]
    assert result["validation"]["downsampled_chunks"] == [1, 2, 3]
    assert result["physical_inventory"]["content_sha256"]
    assert root.attrs["training_artifact_status"] == "awaiting_detection_review"
    assert root.attrs["stage_selector_eligible"] is False
    assert root.attrs["registry_activation"] == "deferred"
    assert (
        root.attrs["sampled_training_base_publication"]["metadata_mode"]
        == "direct_mutable"
    )
    assert not list(destination.parent.glob(".*.publish_tmp.*"))


def test_validate_sampled_training_base_rejects_multi_frame_chunks(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "bad.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.create_array(
        "images_full",
        data=np.zeros((3, 4, 5), dtype=np.uint8),
        chunks=(3, 4, 5),
    )
    raw.create_array(
        "images_ds",
        data=np.zeros((3, 2, 3), dtype=np.uint8),
        chunks=(1, 2, 3),
    )
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 3, 6], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="one-frame physical chunks"):
        publication_mod.validate_sampled_training_base(
            archive,
            source_frame_count=10,
            frame_step=3,
            skip_tail_frames=1,
        )


def test_publish_sampled_training_base_refuses_existing_destination(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        publication_mod, "import_sampled_training_pynvvc", _fake_import
    )
    inputs = _inputs(tmp_path)
    destination = Path(inputs["destination"])
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="destination exists"):
        publication_mod.publish_sampled_training_base(**inputs)

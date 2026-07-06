from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import zarr

from fisheye.shared.crop_geometry import resolve_full_frame_shape
from fisheye.shared.roi_pixel_contract import ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
from fisheye.utils import import_sampled_training_pynvvc as import_mod
from fisheye.utils.import_sampled_training_pynvvc import import_sampled_training_pynvvc


class _FakePynvvcReader:
    def __init__(self, video_path: Path, *, start_frame: int = 0, gpu_id: int = 0) -> None:
        assert start_frame == 0
        self.source_height = 4
        self.source_width = 5
        self.closed = False

    def iter_frames(self):
        for idx in range(10):
            yield torch.full((4, 5), idx, dtype=torch.uint8)

    def close(self) -> None:
        self.closed = True


def test_import_sampled_training_pynvvc_writes_luma_training_zarr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "Cam2010093_demo.mp4"
    video.write_bytes(b"placeholder")
    h5_path = tmp_path / "raw" / "demo.h5"
    h5_path.parent.mkdir(parents=True)
    h5_path.write_bytes(b"h5-placeholder")
    (tmp_path / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_id": "2026-06-23T16-01-09Z_arena_1",
                "session_uuid": "2026-06-23T16-01-09Z_arena_1",
                "recording_name": "2026-06-23T16-01-09Z_arena_1_RedScare",
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "protocol_name": "RedScare",
                "arena_id": "1",
                "camera_id": "2010093",
                "dish_design": "palm1",
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "import.yaml"
    config.write_text(
        """
import:
  resolutions: both
  chunk_size: 2
  downsampled:
    size: [2, 3]
    method: nearest
    preserve_aspect: false
    chunk_size: 2
""",
        encoding="utf-8",
    )
    out = tmp_path / "demo_training.zarr"
    monkeypatch.setattr(
        import_mod,
        "probe_video_colorimetry_attrs",
        lambda path: {
            "video_color_range": "pc",
            "video_color_space": "bt709",
            "video_color_transfer": "bt709",
            "video_color_primaries": "bt709",
            "source_video_colorimetry_source": "ffprobe_stream",
        },
    )

    result = import_sampled_training_pynvvc(
        video_path=video,
        zarr_path=out,
        source_frame_count=10,
        frame_step=3,
        skip_tail_frames=1,
        config_path=config,
        camera_id="2010093",
        recording_dir=tmp_path,
        h5_path=h5_path,
        require_cuda=False,
        reader_factory=_FakePynvvcReader,
    )

    assert result.imported_frame_count == 3
    assert result.decode_backend == "pynvvc_luma"
    root = zarr.open_group(str(out), mode="r")
    raw = root["raw_video"]
    assert root.attrs["zarr_purpose"] == "training"
    assert root.attrs["zarr_use"] == "training"
    assert root.attrs["recording_id"] == "2026-06-23T16-01-09Z_arena_1"
    assert root.attrs["recording_name"] == "2026-06-23T16-01-09Z_arena_1_RedScare"
    assert root.attrs["protocol_name"] == "RedScare"
    assert root.attrs["source_video_width"] == 5
    assert root.attrs["source_video_height"] == 4
    assert root.attrs["video_width"] == 5
    assert root.attrs["video_height"] == 4
    assert raw.attrs["decode_backend"] == "pynvvc_luma"
    assert raw.attrs["decode_contract_status"] == "canonical_orange_mono_pynvvc_luma"
    assert raw.attrs["source_pixels"] == "raw_camera_video"
    assert raw.attrs["source_pixel_contract"] == "orange.camera.mono8.full_frame.v1"
    assert raw.attrs["source_pixel_range"] == "0_255"
    assert raw.attrs["applied_range_semantics"] == "orange_mono8_full_range_0_255"
    assert raw.attrs["container_color_range_observed"] == "pc"
    assert raw.attrs["video_color_range"] == "pc"
    assert raw.attrs["video_color_space"] == "bt709"
    assert raw.attrs["video_color_transfer"] == "bt709"
    assert raw.attrs["video_color_primaries"] == "bt709"
    assert raw.attrs["source_video_colorimetry_source"] == "ffprobe_stream"
    assert root.attrs["video_color_range"] == "pc"
    assert root.attrs["video_color_space"] == "bt709"
    assert raw.attrs["pixel_contract_name"] == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    assert raw.attrs["color_range"] == "source_full_range_0_255_container_observed_pc"
    assert raw.attrs["stored_luma_color_range"] == "source_full_range_0_255_y_plane"
    assert raw.attrs["frame_step"] == 3
    assert raw.attrs["source_frame_count"] == 10
    assert raw.attrs["source_video_width"] == 5
    assert raw.attrs["source_video_height"] == 4
    assert raw.attrs["video_width"] == 5
    assert raw.attrs["video_height"] == 4
    assert raw.attrs["import_profile"] == "sampled_training_pynvvc_luma"
    assert raw.attrs["import_profile_schema_id"] == "palette.import_profile_contract.v1"
    assert raw.attrs["source_video_fingerprint_strategy"] == "stat_v1"
    assert raw.attrs["source_video_fingerprint"]
    assert raw.attrs["source_video_fingerprint_payload"]["frame_count"] == 10
    assert raw.attrs["source_video_size_bytes"] == len(b"placeholder")
    assert raw.attrs["source_h5"] == "demo.h5"
    assert raw.attrs["source_h5_path"] == str(h5_path)
    assert raw.attrs["source_h5_fingerprint_strategy"] == "stat_v1"
    assert raw.attrs["source_h5_fingerprint"]
    assert root.attrs["source_video_fingerprint"] == raw.attrs["source_video_fingerprint"]
    assert root.attrs["source_h5_fingerprint"] == raw.attrs["source_h5_fingerprint"]
    assert raw.attrs["original_resolution"] == [4, 5]
    assert resolve_full_frame_shape(root) == (4, 5)
    assert raw["original_frame_indices"][:].tolist() == [0, 3, 6]
    assert raw["images_full"].shape == (3, 4, 5)
    assert raw["images_ds"].shape == (3, 2, 3)
    np.testing.assert_array_equal(raw["images_full"][:, 0, 0], np.array([0, 3, 6], dtype=np.uint8))
    np.testing.assert_array_equal(raw["images_ds"][:, 0, 0], np.array([0, 3, 6], dtype=np.uint8))

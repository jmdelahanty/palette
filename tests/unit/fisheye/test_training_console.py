from __future__ import annotations

import io

from rich.console import Console

from fisheye.training.config import DatasetConfig, SourceType
from fisheye.training.training_console import print_dataset_details


def test_dataset_config_defaults_to_refined_source_type(tmp_path) -> None:
    zarr_path = tmp_path / "dataset.zarr"
    zarr_path.mkdir()
    (zarr_path / "zarr.json").write_text("{}", encoding="utf-8")

    cfg = DatasetConfig(zarr_path=zarr_path)

    assert cfg.source_type == SourceType.REFINED


def test_print_dataset_details_shows_manual_edited_detections() -> None:
    console = Console(file=io.StringIO(), force_terminal=False, color_system=None)

    print_dataset_details(
        console,
        {
            "example.zarr": {
                "video_frames": 120,
                "fps": 60.0,
                "crop_info": {
                    "source_type": "refined",
                    "total_rois": 18,
                },
                "detection_info": {
                    "total_detections": 18,
                    "detection_rate": 15.0,
                },
                "data_quality": {
                    "has_refinement": True,
                    "manual_edited_detections": 3,
                },
            }
        },
        task="detect",
    )

    rendered = console.file.getvalue()

    assert "Manual-Edited Detections" in rendered
    assert "3" in rendered

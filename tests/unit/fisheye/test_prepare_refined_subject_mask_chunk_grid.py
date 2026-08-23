from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import prepare_refined_subject_mask_chunk_grid as cli


def _crop(zarr_path: Path, sizes: np.ndarray) -> None:
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    crop = root.require_group("crop_runs").create_group("crop_v2")
    crop.create_array("roi_sizes_full", data=sizes, overwrite=True)
    crop.create_array(
        "source_crop_row_ids",
        data=np.arange(sizes.shape[0], dtype=np.int64),
        overwrite=True,
    )


def test_grid_cli_derives_mask_shape_from_fixed_crop_authority(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "analysis.zarr"
    _crop(archive, np.repeat(np.asarray([[384, 320]], dtype=np.int32), 3, axis=0))
    output = tmp_path / "grid.json"

    assert (
        cli.main(
            [
                "--zarr",
                str(archive),
                "--crop-run",
                "crop_v2",
                "--output-manifest",
                str(output),
                "--mask-label",
                "subject_body",
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["mask_shape"] == [3, 1, 320, 384]


def test_grid_cli_rejects_variable_crop_size(tmp_path: Path) -> None:
    archive = tmp_path / "analysis.zarr"
    _crop(
        archive,
        np.asarray([[384, 384], [320, 384]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="one fixed positive int32 ROI size"):
        cli.main(
            [
                "--zarr",
                str(archive),
                "--crop-run",
                "crop_v2",
                "--output-manifest",
                str(tmp_path / "grid.json"),
                "--mask-label",
                "subject_body",
            ]
        )

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis_workflows.materializers.collection_proxy_coordinates import (
    build_collection_proxy_coordinate_materialization_plan,
    derive_current_geometry,
)


def test_derive_current_geometry_preserves_dtype_and_semantics() -> None:
    normalized = np.asarray(
        [[0.5, 0.25, 0.2, 0.1], [0.75, 0.75, 0.5, 0.5]],
        dtype=np.float32,
    )

    bbox, centers = derive_current_geometry(
        normalized,
        width_px=100,
        height_px=200,
    )

    assert bbox.dtype == normalized.dtype
    assert centers.dtype == normalized.dtype
    np.testing.assert_allclose(
        bbox,
        np.asarray(
            [[40.0, 40.0, 60.0, 60.0], [50.0, 100.0, 100.0, 200.0]], dtype=np.float32
        ),
    )
    np.testing.assert_allclose(
        centers,
        np.asarray([[50.0, 50.0], [75.0, 150.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "normalized",
    (
        np.asarray([[0.1, 0.1, 0.4, 0.1]], dtype=np.float64),
        np.asarray([[0.5, 0.5, 0.0, 0.1]], dtype=np.float64),
        np.asarray([[np.nan, 0.5, 0.1, 0.1]], dtype=np.float64),
    ),
)
def test_derive_current_geometry_rejects_invalid_boxes(
    normalized: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        derive_current_geometry(normalized, width_px=100, height_px=100)


def test_plan_is_read_only_and_rejects_scratch_inside_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.zarr"
    source.mkdir()

    plan = build_collection_proxy_coordinate_materialization_plan(
        source,
        historical_rowset="/crop_runs/historical/",
        scratch_root=tmp_path / "scratch",
        run_name="successor",
    )

    assert plan.historical_rowset == "crop_runs/historical"
    assert plan.target_run_path == source / "crop_runs" / "successor"
    assert not plan.scratch_root.exists()
    with pytest.raises(ValueError, match="inside the authoritative Zarr"):
        build_collection_proxy_coordinate_materialization_plan(
            source,
            historical_rowset="crop_runs/historical",
            scratch_root=source / "scratch",
            run_name="successor",
        )

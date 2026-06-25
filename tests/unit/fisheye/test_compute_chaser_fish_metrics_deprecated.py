from __future__ import annotations

import pytest

from fisheye.analysis.compute_chaser_fish_metrics import DEPRECATED_MESSAGE, compute_metrics, main


def test_compute_chaser_fish_metrics_main_is_deprecated() -> None:
    with pytest.raises(SystemExit, match="deprecated and must not be used"):
        main(["dummy.zarr"])


def test_compute_chaser_fish_metrics_programmatic_entry_is_deprecated() -> None:
    with pytest.raises(RuntimeError, match="deprecated and must not be used"):
        compute_metrics(
            zarr_root=None,  # type: ignore[arg-type]
            stimulus_run="stimulus_1",
            keypoints_path="keypoints_runs/kp_1",
            output_run="legacy",
            chaser_index=0,
            overwrite=False,
            console=None,  # type: ignore[arg-type]
        )
    assert "chaser_distance_runs" in DEPRECATED_MESSAGE

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fisheye.core import pipeline as pipeline_mod
from fisheye.core.pipeline import Pipeline, PipelineConfig


def _make_pipeline(tmp_path: Path, **kwargs) -> Pipeline:
    config_path = tmp_path / "missing_pipeline_config.yaml"
    cfg = PipelineConfig(
        zarr_path=str(tmp_path / "archive.zarr"),
        config_path=str(config_path),
        **kwargs,
    )
    return Pipeline(cfg)


def test_default_refine_params_are_sparse_first(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)

    refine_params = pipeline.pipeline_params["refine_detect"]

    assert refine_params == {"filters": {"remove_jumps": True, "remove_blips": False}}


@pytest.mark.parametrize(
    ("kwargs", "expected_flag"),
    [
        ({"refine_max_gap": 5}, "--refine-max-gap"),
        ({"refine_method": "linear"}, "--refine-method"),
    ],
)
def test_run_refine_rejects_deprecated_interpolation_overrides(
    tmp_path: Path,
    kwargs: dict[str, object],
    expected_flag: str,
) -> None:
    pipeline = _make_pipeline(tmp_path, **kwargs)
    pipeline.zarr_root = object()

    with pytest.raises(
        ValueError,
        match="Interpolation overrides are deprecated and unsupported for refine_detect",
    ) as exc_info:
        pipeline._run_refine()

    assert expected_flag in str(exc_info.value)


@pytest.mark.parametrize(
    "argv",
    [
        ["archive.zarr", "--refine-max-gap", "5"],
        ["archive.zarr", "--refine-method", "linear"],
    ],
)
def test_main_rejects_deprecated_refine_interpolation_flags(argv: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        with pytest.raises(
            SystemExit,
            match="Interpolation overrides are deprecated and unsupported for refine_detect",
        ):
            sys.argv = ["fisheye", *argv]
            pipeline_mod.main()
    finally:
        sys.argv = old_argv

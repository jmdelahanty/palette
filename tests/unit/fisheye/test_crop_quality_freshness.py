from __future__ import annotations

from pathlib import Path

from fisheye.utils.crop_quality_freshness import is_crop_quality_row_fresh


def test_is_crop_quality_row_fresh_true_when_mtime_matches(tmp_path: Path) -> None:
    zarr_path = tmp_path / "example.zarr"
    zarr_path.mkdir()
    fs_mtime_ns = zarr_path.stat().st_mtime_ns

    is_fresh, reason = is_crop_quality_row_fresh(
        zarr_path=zarr_path,
        zarr_mtime_ns=fs_mtime_ns,
    )

    assert is_fresh is True
    assert reason is None


def test_is_crop_quality_row_fresh_false_when_registry_mtime_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "example.zarr"
    zarr_path.mkdir()

    is_fresh, reason = is_crop_quality_row_fresh(
        zarr_path=zarr_path,
        zarr_mtime_ns=None,
    )

    assert is_fresh is False
    assert reason == "missing_registry_zarr_mtime_ns"


def test_is_crop_quality_row_fresh_false_when_mtime_differs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "example.zarr"
    zarr_path.mkdir()
    fs_mtime_ns = zarr_path.stat().st_mtime_ns

    is_fresh, reason = is_crop_quality_row_fresh(
        zarr_path=zarr_path,
        zarr_mtime_ns=fs_mtime_ns + 1,
    )

    assert is_fresh is False
    assert reason is not None
    assert reason.startswith("zarr_mtime_mismatch(")

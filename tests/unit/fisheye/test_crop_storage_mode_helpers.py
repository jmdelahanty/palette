from __future__ import annotations

import sys
from unittest.mock import MagicMock

import zarr

if "decord" not in sys.modules:
    sys.modules["decord"] = MagicMock()

from fisheye.tracking.crop import (  # noqa: E402
    _enforce_training_materialized_crop_contract,
    _finalize_crop_parent_pointers,
    _infer_archive_use,
)


def test_finalize_crop_parent_pointers_promotes_geometry_only_to_latest_any_only() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")

    materialized = parent.create_group("crop_materialized")
    materialized.attrs["crop_storage_mode"] = "materialized"

    geometry = parent.create_group("crop_geometry")
    geometry.attrs["crop_storage_mode"] = "geometry_only"

    parent.attrs["latest"] = "crop_materialized"
    parent.attrs["latest_materialized"] = "crop_materialized"
    parent.attrs["latest_any"] = "crop_materialized"

    _finalize_crop_parent_pointers(
        parent,
        run_name="crop_geometry",
        crop_storage_mode="geometry_only",
        success=True,
        previous_latest="crop_materialized",
        previous_latest_materialized="crop_materialized",
        previous_latest_any="crop_materialized",
    )

    assert parent.attrs["latest"] == "crop_materialized"
    assert parent.attrs["latest_materialized"] == "crop_materialized"
    assert parent.attrs["latest_any"] == "crop_geometry"


def test_finalize_crop_parent_pointers_promotes_materialized_to_all_latest_pointers() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")

    previous = parent.create_group("crop_previous")
    previous.attrs["crop_storage_mode"] = "materialized"

    current = parent.create_group("crop_current")
    current.attrs["crop_storage_mode"] = "materialized"

    parent.attrs["latest"] = "crop_previous"
    parent.attrs["latest_materialized"] = "crop_previous"
    parent.attrs["latest_any"] = "crop_previous"

    _finalize_crop_parent_pointers(
        parent,
        run_name="crop_current",
        crop_storage_mode="materialized",
        success=True,
        previous_latest="crop_previous",
        previous_latest_materialized="crop_previous",
        previous_latest_any="crop_previous",
    )

    assert parent.attrs["latest"] == "crop_current"
    assert parent.attrs["latest_materialized"] == "crop_current"
    assert parent.attrs["latest_any"] == "crop_current"


def test_infer_archive_use_prefers_root_attrs() -> None:
    root = zarr.group()
    root.attrs["zarr_purpose"] = "training"

    assert _infer_archive_use(root, "/tmp/example_analysis.zarr") == "training"


def test_enforce_training_materialized_crop_contract_rejects_geometry_only() -> None:
    root = zarr.group()
    root.attrs["zarr_purpose"] = "training"

    try:
        _enforce_training_materialized_crop_contract(
            root,
            zarr_path="/tmp/example_training.zarr",
            crop_storage_mode="geometry_only",
        )
    except ValueError as exc:
        assert "Training zarrs require materialized crop runs" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected training geometry-only crop enforcement to fail")

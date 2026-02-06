"""Unit tests for registry model-input filtering."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


def _register_dataset(
    registry: Registry,
    *,
    dataset_id: str,
    has_gray: bool | None,
    has_rgb: bool | None,
    formats: list[str] | None,
) -> None:
    registry.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=Path(f"/tmp/{dataset_id}.zarr"),
    )
    acquisition = {
        "has_images_ds": has_gray,
        "has_images_ds_rgb": has_rgb,
        "downsample_formats_json": json.dumps(formats) if formats is not None else None,
    }
    registry.upsert_provenance(
        dataset_id,
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition=acquisition,
        zarr_purpose="training",
    )


def test_query_datasets_model_input_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite"
    registry = Registry(db_path)
    _register_dataset(
        registry,
        dataset_id="gray_only",
        has_gray=True,
        has_rgb=False,
        formats=["gray"],
    )
    _register_dataset(
        registry,
        dataset_id="rgb_only",
        has_gray=False,
        has_rgb=True,
        formats=["rgb"],
    )
    _register_dataset(
        registry,
        dataset_id="both_modalities",
        has_gray=True,
        has_rgb=True,
        formats=["gray", "rgb"],
    )
    # Legacy-like row: availability captured only via downsample_formats_json.
    _register_dataset(
        registry,
        dataset_id="legacy_gray",
        has_gray=None,
        has_rgb=None,
        formats=["gray"],
    )

    gray_ids = {row["dataset_id"] for row in registry.query_datasets(model_input="gray")}
    rgb_ids = {row["dataset_id"] for row in registry.query_datasets(model_input="rgb")}
    all_ids = {row["dataset_id"] for row in registry.query_datasets()}
    registry.close()

    assert all_ids == {"gray_only", "rgb_only", "both_modalities", "legacy_gray"}
    assert gray_ids == {"gray_only", "both_modalities", "legacy_gray"}
    assert rgb_ids == {"rgb_only", "both_modalities"}


def test_query_datasets_model_input_rejects_invalid(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    with pytest.raises(ValueError, match="Unsupported model_input"):
        registry.query_datasets(model_input="foo")
    registry.close()

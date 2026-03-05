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


# ---------------------------------------------------------------------------
# exclude_step_ok tests
# ---------------------------------------------------------------------------


def _register_simple_dataset(registry: Registry, dataset_id: str) -> None:
    """Register a minimal dataset for step-status filtering tests."""
    registry.upsert_dataset(
        dataset_id,
        session_uuid=dataset_id,
        zarr_path=Path(f"/tmp/{dataset_id}.zarr"),
    )


def test_query_datasets_exclude_step_ok_filters_completed(tmp_path: Path) -> None:
    """exclude_step_ok omits datasets whose step status is 'ok'."""
    from fisheye.registry.status_ledger import upsert_recording_step_status

    registry = Registry(tmp_path / "registry.sqlite")

    # dataset_a: detect status = 'ok' -> should be excluded
    _register_simple_dataset(registry, "dataset_a")
    upsert_recording_step_status(
        registry, dataset_id="dataset_a", step_name="detect", status="ok",
    )

    # dataset_b: detect status = 'missing' -> should be included
    _register_simple_dataset(registry, "dataset_b")
    upsert_recording_step_status(
        registry, dataset_id="dataset_b", step_name="detect", status="missing",
    )

    # dataset_c: no detect step row at all -> should be included
    _register_simple_dataset(registry, "dataset_c")

    # Without filter: all 3 returned.
    all_ids = {row["dataset_id"] for row in registry.query_datasets()}
    assert all_ids == {"dataset_a", "dataset_b", "dataset_c"}

    # With filter: dataset_a excluded.
    filtered_ids = {
        row["dataset_id"]
        for row in registry.query_datasets(exclude_step_ok="detect")
    }
    assert filtered_ids == {"dataset_b", "dataset_c"}

    registry.close()


def test_query_datasets_exclude_step_ok_includes_non_ok_statuses(tmp_path: Path) -> None:
    """Datasets with error/absent/na statuses are included by exclude_step_ok."""
    from fisheye.registry.status_ledger import upsert_recording_step_status

    registry = Registry(tmp_path / "registry.sqlite")

    for status_val in ("error", "absent", "na"):
        did = f"ds_{status_val}"
        _register_simple_dataset(registry, did)
        upsert_recording_step_status(
            registry, dataset_id=did, step_name="detect", status=status_val,
        )

    filtered_ids = {
        row["dataset_id"]
        for row in registry.query_datasets(exclude_step_ok="detect")
    }
    assert filtered_ids == {"ds_error", "ds_absent", "ds_na"}

    registry.close()


def test_query_datasets_exclude_step_ok_none_is_noop(tmp_path: Path) -> None:
    """exclude_step_ok=None (default) has no filtering effect."""
    from fisheye.registry.status_ledger import upsert_recording_step_status

    registry = Registry(tmp_path / "registry.sqlite")

    _register_simple_dataset(registry, "ds_ok")
    upsert_recording_step_status(
        registry, dataset_id="ds_ok", step_name="detect", status="ok",
    )
    _register_simple_dataset(registry, "ds_missing")

    all_ids = {row["dataset_id"] for row in registry.query_datasets()}
    default_ids = {row["dataset_id"] for row in registry.query_datasets(exclude_step_ok=None)}
    assert all_ids == default_ids == {"ds_ok", "ds_missing"}

    registry.close()


# ---------------------------------------------------------------------------
# require_steps_ok tests
# ---------------------------------------------------------------------------


def test_query_datasets_require_steps_ok_filters_missing_prereqs(tmp_path: Path) -> None:
    """require_steps_ok returns only datasets where all specified steps are 'ok'."""
    from fisheye.registry.status_ledger import upsert_recording_step_status

    registry = Registry(tmp_path / "registry.sqlite")

    # ds_both_ok: detect=ok, crop=ok -> included
    _register_simple_dataset(registry, "ds_both_ok")
    upsert_recording_step_status(registry, dataset_id="ds_both_ok", step_name="detect", status="ok")
    upsert_recording_step_status(registry, dataset_id="ds_both_ok", step_name="crop", status="ok")

    # ds_detect_only: detect=ok, crop missing -> excluded
    _register_simple_dataset(registry, "ds_detect_only")
    upsert_recording_step_status(registry, dataset_id="ds_detect_only", step_name="detect", status="ok")

    # ds_crop_error: detect=ok, crop=error -> excluded
    _register_simple_dataset(registry, "ds_crop_error")
    upsert_recording_step_status(registry, dataset_id="ds_crop_error", step_name="detect", status="ok")
    upsert_recording_step_status(registry, dataset_id="ds_crop_error", step_name="crop", status="error")

    # ds_none: no step status rows -> excluded
    _register_simple_dataset(registry, "ds_none")

    filtered_ids = {
        row["dataset_id"]
        for row in registry.query_datasets(require_steps_ok=["detect", "crop"])
    }
    assert filtered_ids == {"ds_both_ok"}

    registry.close()


def test_query_datasets_require_steps_ok_none_is_noop(tmp_path: Path) -> None:
    """require_steps_ok=None (default) returns all datasets regardless of step status."""
    from fisheye.registry.status_ledger import upsert_recording_step_status

    registry = Registry(tmp_path / "registry.sqlite")

    _register_simple_dataset(registry, "ds_a")
    upsert_recording_step_status(registry, dataset_id="ds_a", step_name="detect", status="ok")
    _register_simple_dataset(registry, "ds_b")

    all_ids = {row["dataset_id"] for row in registry.query_datasets()}
    default_ids = {row["dataset_id"] for row in registry.query_datasets(require_steps_ok=None)}
    assert all_ids == default_ids == {"ds_a", "ds_b"}

    registry.close()

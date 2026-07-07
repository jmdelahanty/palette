from pathlib import Path

import pytest
import zarr

from fisheye.registry import temp_store_guard
from fisheye.registry.db import Registry
from fisheye.registry.temp_store_guard import ALLOW_TEMP_STORES_ENV


def _minimal_zarr(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["session_uuid"] = "tmp_ds"
    root.attrs["zarr_use"] = "analysis"
    return root


def test_tmp_registry_with_tmp_store_is_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ALLOW_TEMP_STORES_ENV, raising=False)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        store_path = tmp_path / "store.zarr"
        root = _minimal_zarr(store_path)
        dataset_id = registry.register_from_root(root, store_path)

        row = registry.conn.execute(
            "SELECT zarr_path FROM datasets WHERE dataset_id = ?;",
            (dataset_id,),
        ).fetchone()
        assert row["zarr_path"] == str(store_path)
    finally:
        registry.close()


def test_non_tmp_registry_with_tmp_store_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(ALLOW_TEMP_STORES_ENV, raising=False)
    temp_root = tmp_path / "simulated-temp-root"
    temp_root.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    store_path = temp_root / "store.zarr"
    monkeypatch.setattr(temp_store_guard, "_resolved_temp_roots", lambda: (temp_root.resolve(),))

    registry = Registry(registry_path)
    try:
        with pytest.raises(ValueError) as excinfo:
            registry.upsert_dataset("blocked_ds", session_uuid="blocked_ds", zarr_path=store_path)
    finally:
        registry.close()

    message = str(excinfo.value)
    assert str(store_path.resolve()) in message
    assert str(registry_path.resolve()) in message
    assert f"{ALLOW_TEMP_STORES_ENV}=1" in message


def test_override_allows_tmp_store_with_non_tmp_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "simulated-temp-root"
    temp_root.mkdir()
    registry_path = tmp_path / "registry.sqlite"
    store_path = temp_root / "store.zarr"
    monkeypatch.setattr(temp_store_guard, "_resolved_temp_roots", lambda: (temp_root.resolve(),))
    monkeypatch.setenv(ALLOW_TEMP_STORES_ENV, "1")

    registry = Registry(registry_path)
    try:
        registry.upsert_dataset("override_ds", session_uuid="override_ds", zarr_path=store_path)
        row = registry.conn.execute(
            "SELECT zarr_path FROM datasets WHERE dataset_id = ?;",
            ("override_ds",),
        ).fetchone()
        assert row["zarr_path"] == str(store_path)
    finally:
        registry.close()

from pathlib import Path

import pytest
import zarr

from fisheye.registry import temp_store_guard
from fisheye.registry.db import Registry
from fisheye.registry.temp_store_guard import (
    ALLOW_SCRATCH_STORES_ENV,
    ALLOW_TEMP_STORES_ENV,
    ALLOW_UNOWNED_ANALYSIS_ENV,
)


def _minimal_zarr(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["session_uuid"] = "tmp_ds"
    root.attrs["zarr_use"] = "analysis"
    return root


def test_tmp_registry_with_tmp_store_is_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ALLOW_TEMP_STORES_ENV, raising=False)
    monkeypatch.delenv(ALLOW_SCRATCH_STORES_ENV, raising=False)
    monkeypatch.delenv(ALLOW_UNOWNED_ANALYSIS_ENV, raising=False)
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
    monkeypatch.delenv(ALLOW_SCRATCH_STORES_ENV, raising=False)
    monkeypatch.delenv(ALLOW_UNOWNED_ANALYSIS_ENV, raising=False)
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
        registry.upsert_dataset(
            "override_ds",
            session_uuid="override_ds",
            zarr_path=store_path,
            recording_id="override_ds",
        )
        row = registry.conn.execute(
            "SELECT zarr_path FROM datasets WHERE dataset_id = ?;",
            ("override_ds",),
        ).fetchone()
        assert row["zarr_path"] == str(store_path)
    finally:
        registry.close()


def test_durable_registry_refuses_in_memory_store_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        temp_store_guard,
        "_resolved_temp_roots",
        lambda: (Path("/simulated-temp-root"),),
    )
    monkeypatch.delenv(ALLOW_SCRATCH_STORES_ENV, raising=False)
    registry_path = tmp_path / "registry.sqlite"
    store_path = Path("/home/researcher/gitrepos/palette/in-memory.zarr")
    registry = Registry(registry_path)
    try:
        with pytest.raises(ValueError, match="in_memory_store_name"):
            registry.upsert_dataset(
                "scratch",
                session_uuid="recording_1",
                zarr_path=store_path,
                recording_id="recording_1",
                zarr_use="analysis",
            )
    finally:
        registry.close()


def test_durable_registry_refuses_agent_worktree_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        temp_store_guard,
        "_resolved_temp_roots",
        lambda: (Path("/simulated-temp-root"),),
    )
    monkeypatch.delenv(ALLOW_SCRATCH_STORES_ENV, raising=False)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(ValueError, match="agent_worktree"):
            registry.upsert_dataset(
                "scratch",
                session_uuid="recording_1",
                zarr_path=Path(
                    "/home/researcher/gitrepos/palette/.claude/worktrees/agent-1/analysis.zarr"
                ),
                recording_id="recording_1",
                zarr_use="analysis",
            )
    finally:
        registry.close()


def test_durable_registry_refuses_unowned_analysis_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        temp_store_guard,
        "_resolved_temp_roots",
        lambda: (Path("/simulated-temp-root"),),
    )
    monkeypatch.delenv(ALLOW_UNOWNED_ANALYSIS_ENV, raising=False)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(ValueError, match="without a normalized recording_id"):
            registry.upsert_dataset(
                "unowned_analysis",
                session_uuid=None,
                zarr_path=Path("/groups/lab/recordings/unowned_analysis.zarr"),
                zarr_use="analysis",
            )
    finally:
        registry.close()


def test_durable_registry_allows_unowned_training_merge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        temp_store_guard,
        "_resolved_temp_roots",
        lambda: (Path("/simulated-temp-root"),),
    )
    monkeypatch.delenv(ALLOW_UNOWNED_ANALYSIS_ENV, raising=False)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            "training_merge",
            session_uuid=None,
            zarr_path=Path("/nvme1/training/datasets/training_merge.zarr"),
            artifact_kind="derived_training_merge",
            zarr_use="training",
        )
        row = registry.conn.execute(
            "SELECT zarr_use FROM datasets WHERE dataset_id = 'training_merge';"
        ).fetchone()
        assert row["zarr_use"] == "training"
    finally:
        registry.close()


def test_scratch_override_does_not_bypass_recording_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        temp_store_guard,
        "_resolved_temp_roots",
        lambda: (Path("/simulated-temp-root"),),
    )
    monkeypatch.setenv(ALLOW_SCRATCH_STORES_ENV, "1")
    monkeypatch.delenv(ALLOW_UNOWNED_ANALYSIS_ENV, raising=False)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(ValueError, match="without a normalized recording_id"):
            registry.upsert_dataset(
                "scratch",
                session_uuid=None,
                zarr_path=Path("/home/researcher/gitrepos/palette/in-memory.zarr"),
                zarr_use="analysis",
            )
    finally:
        registry.close()

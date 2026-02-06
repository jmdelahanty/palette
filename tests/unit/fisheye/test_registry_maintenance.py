"""Unit tests for registry maintenance helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.maintenance import (
    _collect_invalid_dataset_candidates,
    _delete_dataset_ids,
    _is_nested_zarr_subpath,
)


def test_is_nested_zarr_subpath() -> None:
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs")
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs/run_01")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr/subset.zarr")


def test_collect_and_delete_invalid_candidates(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    good = tmp_path / "good.zarr"
    nested = tmp_path / "good.zarr" / "detect_runs"
    nested_zarr = tmp_path / "good.zarr" / "subset.zarr"
    stale = tmp_path / "missing.zarr"

    registry.upsert_dataset("good", session_uuid="good", zarr_path=good)
    registry.upsert_dataset("nested", session_uuid=None, zarr_path=nested)
    registry.upsert_dataset("nested_zarr", session_uuid=None, zarr_path=nested_zarr)
    registry.upsert_dataset("stale", session_uuid=None, zarr_path=stale)
    registry.conn.execute("UPDATE datasets SET status = 'missing' WHERE dataset_id = 'stale';")
    registry.conn.commit()

    candidates = _collect_invalid_dataset_candidates(registry)
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"nested", "stale"}
    assert by_id["nested"].reasons == ("nested_zarr_subpath",)
    assert by_id["stale"].reasons == ("status_missing",)

    # Dry run must not delete.
    assert _delete_dataset_ids(registry, ["nested", "stale"], dry_run=True) == 0
    still_present = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert still_present == {"good", "nested", "nested_zarr", "stale"}

    deleted = _delete_dataset_ids(registry, ["nested", "stale"], dry_run=False)
    assert deleted == 2
    remaining = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert remaining == {"good", "nested_zarr"}
    registry.close()


def test_collect_candidates_can_infer_missing_without_status(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("missing_active", session_uuid=None, zarr_path=tmp_path / "not_there.zarr")
    nested_path = tmp_path / "recording.zarr" / "detect_runs"
    nested_path.mkdir(parents=True)
    (nested_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")
    registry.upsert_dataset(
        "nested_active",
        session_uuid=None,
        zarr_path=nested_path,
    )

    candidates = _collect_invalid_dataset_candidates(
        registry,
        include_missing_scan=True,
    )
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"missing_active", "nested_active"}
    assert by_id["missing_active"].reasons == ("status_missing",)
    assert by_id["nested_active"].reasons == ("nested_zarr_subpath",)
    registry.close()

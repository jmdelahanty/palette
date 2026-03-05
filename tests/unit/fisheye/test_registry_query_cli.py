"""Tests for the registry query CLI module."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.query import _build_query, _parse_args


def _register_dataset(registry: Registry, *, dataset_id: str, root: Path) -> None:
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid=dataset_id,
        zarr_path=root / f"{dataset_id}.zarr",
    )


def test_registry_query_since_filters_by_created_utc(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_dataset(registry, dataset_id="dataset_a_old", root=tmp_path)
        _register_dataset(registry, dataset_id="dataset_b_edge", root=tmp_path)
        _register_dataset(registry, dataset_id="dataset_c_new", root=tmp_path)

        registry.conn.execute(
            "UPDATE datasets SET created_utc = ? WHERE dataset_id = ?;",
            ("2026-02-01T00:00:00+00:00", "dataset_a_old"),
        )
        registry.conn.execute(
            "UPDATE datasets SET created_utc = ? WHERE dataset_id = ?;",
            ("2026-02-15T00:00:00+00:00", "dataset_b_edge"),
        )
        registry.conn.execute(
            "UPDATE datasets SET created_utc = ? WHERE dataset_id = ?;",
            ("2026-02-20T00:00:00+00:00", "dataset_c_new"),
        )
        registry.conn.commit()

        args = _parse_args(["--since", "2026-02-15", "--limit", "0"])
        query, params = _build_query(args)
        rows = registry.conn.execute(query, params).fetchall()
    finally:
        registry.close()

    dataset_ids = [str(row["dataset_id"]) for row in rows]
    assert dataset_ids == ["dataset_b_edge", "dataset_c_new"]

"""Unit tests for registry maintenance helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.maintenance import (
    _collect_empty_training_set_candidates,
    _collect_failed_run_candidates,
    _collect_invalid_dataset_candidates,
    _delete_training_set_ids,
    _delete_training_run_ids,
    _delete_dataset_ids,
    _is_nested_zarr_subpath,
    _normalize_status_values,
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


def test_collect_and_delete_failed_training_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="run_failed",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="failed",
    )
    registry.record_training_run(
        run_id="run_failed_caps",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="FAILED",
    )
    registry.record_training_run(
        run_id="run_success",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_in_progress",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
    )
    registry.record_model_export(run_id="run_failed", export_type="onnx", path=tmp_path / "run_failed.onnx")
    registry.record_model_export(run_id="run_success", export_type="onnx", path=tmp_path / "run_success.onnx")

    status_values = _normalize_status_values(["failed"])
    candidates = _collect_failed_run_candidates(registry, status_values=status_values)
    candidate_ids = {candidate.run_id for candidate in candidates}
    assert candidate_ids == {"run_failed", "run_failed_caps"}

    # Dry run must not delete.
    assert _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert still_present == {"run_failed", "run_failed_caps", "run_success", "run_in_progress"}

    deleted = _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert remaining == {"run_success", "run_in_progress"}

    export_rows = registry.conn.execute(
        "SELECT run_id FROM model_exports ORDER BY run_id;"
    ).fetchall()
    assert [row["run_id"] for row in export_rows] == ["run_success"]
    registry.close()


def test_collect_and_delete_empty_training_sets(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="set_empty_1",
        name="empty one",
        query_filter=None,
        dataset_ids=["dataset_a"],
    )
    registry.upsert_training_set(
        set_id="set_linked",
        name="linked",
        query_filter=None,
        dataset_ids=["dataset_b"],
    )
    registry.upsert_training_set(
        set_id="set_empty_2",
        name="empty two",
        query_filter=None,
        dataset_ids=[],
    )
    registry.record_training_run(
        run_id="run_linked",
        set_id="set_linked",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_unlinked",
        set_id=None,
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )

    candidates = _collect_empty_training_set_candidates(registry)
    candidate_ids = {candidate.set_id for candidate in candidates}
    assert candidate_ids == {"set_empty_1", "set_empty_2"}

    assert _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert still_present == {"set_empty_1", "set_empty_2", "set_linked"}

    deleted = _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert remaining == {"set_linked"}
    registry.close()

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


def test_registry_query_context_filters_use_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            dataset_id="dataset_ctx",
            session_uuid="session_ctx",
            zarr_path=tmp_path / "dataset_ctx.zarr",
            recording_id="recording_ctx",
            artifact_kind="source_recording",
        )
        registry.upsert_provenance(
            "dataset_ctx",
            provenance={"dpf_at_acquisition": 7, "snapshot_status": "complete"},
            context={},
            protocol_name=None,
            protocol_hash="hash_ctx",
            acquisition={},
            zarr_purpose="analysis",
        )
        registry.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, session_uuid, recording_name, recording_path,
                recording_type, recording_subtype, behavior_mode, artifact_schema_id,
                rig_id, arena_id, camera_id, canvas_name, protocol_name, dish_design,
                created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            (
                "recording_ctx",
                "session_ctx",
                "recording_ctx",
                str(tmp_path / "recordings" / "recording_ctx"),
                "behavior",
                "free",
                "free",
                "behavior_v1",
                "rig_recording",
                "arena_recording",
                "camera_recording",
                "canvas_recording",
                "protocol_recording",
                "dish_design_recording",
            ),
        )
        registry.conn.execute(
            """
            UPDATE provenance
            SET rig_id = ?, arena_id = ?, camera_id = ?, canvas_name = ?, protocol_name = ?
            WHERE dataset_id = ?;
            """,
            (
                "rig_legacy",
                "arena_legacy",
                "camera_legacy",
                "canvas_legacy",
                "protocol_legacy",
                "dataset_ctx",
            ),
        )
        registry.conn.commit()

        args = _parse_args(
            [
                "--protocol",
                "protocol_recording",
                "--camera-id",
                "camera_recording",
                "--require-context",
                "--limit",
                "0",
            ]
        )
        query, params = _build_query(args)
        rows = registry.conn.execute(query, params).fetchall()

        args_missing = _parse_args(["--missing-context", "--limit", "0"])
        query_missing, params_missing = _build_query(args_missing)
        missing_rows = registry.conn.execute(query_missing, params_missing).fetchall()

        args_legacy = _parse_args(["--protocol", "protocol_legacy", "--limit", "0"])
        query_legacy, params_legacy = _build_query(args_legacy)
        legacy_rows = registry.conn.execute(query_legacy, params_legacy).fetchall()
    finally:
        registry.close()

    assert [str(row["dataset_id"]) for row in rows] == ["dataset_ctx"]
    assert rows[0]["protocol_name"] == "protocol_recording"
    assert rows[0]["camera_id"] == "camera_recording"
    assert missing_rows == []
    assert legacy_rows == []


def test_registry_query_fish_id_uses_legacy_provenance_alias_when_normalized_subject_missing(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            dataset_id="dataset_legacy",
            session_uuid="session_legacy",
            zarr_path=tmp_path / "dataset_legacy.zarr",
            recording_id="recording_legacy",
            artifact_kind="source_recording",
        )
        registry.upsert_provenance(
            "dataset_legacy",
            provenance={
                "fish_id": "legacy_subject",
                "cross_id": "legacy_cross",
                "genotype": "legacy_genotype",
                "dpf_at_acquisition": 6,
                "subject_count": 1,
            },
            context={},
            protocol_name=None,
            protocol_hash=None,
            acquisition={},
            zarr_purpose="analysis",
        )
        registry.conn.commit()

        args = _parse_args(["--fish-id", "legacy_subject", "--limit", "0"])
        query, params = _build_query(args)
        rows = registry.conn.execute(query, params).fetchall()
    finally:
        registry.close()

    assert [str(row["dataset_id"]) for row in rows] == ["dataset_legacy"]
    assert rows[0]["fish_id"] == "legacy_subject"
    assert rows[0]["cross_id"] is None
    assert rows[0]["genotype"] is None
    assert rows[0]["dpf_at_acquisition"] is None

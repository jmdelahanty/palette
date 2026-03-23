"""Tests for registry status reporting helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.status import main as registry_status_main


def test_registry_status_missing_dpf_uses_dataset_context_current(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
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
            provenance={"snapshot_status": "complete"},
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
                created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
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
            ),
        )
        registry.conn.execute(
            """
            INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
            VALUES (?, ?, datetime('now'), datetime('now'));
            """,
            ("cross_ctx", "genotype_ctx"),
        )
        registry.conn.execute(
            """
            INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
            VALUES (?, ?, datetime('now'), datetime('now'));
            """,
            ("dish_ctx", "cross_ctx"),
        )
        registry.conn.execute(
            """
            INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
            VALUES (?, ?, datetime('now'), datetime('now'));
            """,
            ("subject_ctx", "dish_ctx"),
        )
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition,
                created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            ("recording_ctx", "subject_ctx", "dataset_ctx", "dish_ctx", "cross_ctx", 8),
        )
        registry.conn.commit()
    finally:
        registry.close()

    registry_status_main(["--registry", str(registry_path)])
    output = capsys.readouterr().out

    assert "Datasets: 1" in output
    assert "DPF missing: 0" in output

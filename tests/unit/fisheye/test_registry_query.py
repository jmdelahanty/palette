"""Tests for registry_query subject-lineage filters."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.registry_query import main as registry_query_main


def _seed_registry_for_subject_filters(registry_path: Path) -> None:
    registry = Registry(registry_path)
    # Minimal dataset rows.
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=registry_path.parent / "a.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=registry_path.parent / "b.zarr",
        recording_id="recording_b",
        artifact_kind="source_recording",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    registry.upsert_provenance(
        "dataset_b",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    # Recording context rows for view joins.
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_a",
            "session_a",
            "recording_a",
            str(registry_path.parent / "recording_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_b",
            "session_b",
            "recording_b",
            str(registry_path.parent / "recording_b"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    # Lineage entities.
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_a', 'genotype_x', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_b', 'genotype_y', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_a', 'cross_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_b', 'cross_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_a', 'dish_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_b', 'dish_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_a", "subject_a", "dataset_a", "dish_a", "cross_a", 8),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_b", "subject_b", "dataset_b", "dish_b", "cross_b", 12),
    )
    registry.conn.commit()
    registry.close()


def test_registry_query_filters_by_cross_id(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--cross-id",
            "cross_a",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}


def test_registry_query_filters_by_genotype_and_dpf(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--genotype",
            "genotype_y",
            "--dpf",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_filters_by_dpf_range(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--dpf-min",
            "9",
            "--dpf-max",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_rejects_invalid_dpf_range(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--dpf-min",
                "13",
                "--dpf-max",
                "12",
                "--json",
            ]
        )
    except SystemExit as exc:
        assert "--dpf-min must be <= --dpf-max." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for invalid DPF range.")

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.experimental_sessions import (
    EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID,
    EXPERIMENTAL_SESSION_SCHEMA_ID,
    ExperimentalSessionAssignmentConflictError,
    ExperimentalSessionIdentityError,
    MissingExperimentalSessionIdentityError,
    UnknownDatasetIdentityError,
    UnknownExperimentalSessionError,
    UnknownRecordingIdentityError,
)
from fisheye.registry.query import _build_query, _parse_args


def _register_recording(
    registry: Registry,
    *,
    recording_id: str,
    session_uuid: str,
    started_utc: str,
    arena_id: str,
) -> str:
    dataset_id = f"dataset_{arena_id}"
    registry.upsert_recording(
        recording_id=recording_id,
        session_uuid=session_uuid,
        recording_name=recording_id,
        started_utc=started_utc,
        arena_id=arena_id,
    )
    registry.upsert_dataset(
        dataset_id,
        session_uuid=session_uuid,
        zarr_path=Path(f"/recordings/{recording_id}/{dataset_id}.zarr"),
        recording_id=recording_id,
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    return dataset_id


def _create_session(
    registry: Registry, session_id: str = "experiment_20260810_001"
) -> None:
    registry.create_experimental_session(
        experimental_session_id=session_id,
        creation_method="operator_manifest",
        created_by="pytest_operator",
        created_at_utc="2026-08-10T12:00:00Z",
        session_snapshot_id="11111111-1111-4111-8111-111111111111",
        evidence={"manifest_sha256": "a" * 64},
    )


def test_multiple_arenas_share_explicit_session_despite_start_skew(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        first_dataset = _register_recording(
            registry,
            recording_id="recording_arena_1",
            session_uuid="acquisition_arena_1",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_1",
        )
        second_dataset = _register_recording(
            registry,
            recording_id="recording_arena_2",
            session_uuid="acquisition_arena_2",
            started_utc="2026-08-10T12:00:01+00:00",
            arena_id="arena_2",
        )

        before = registry.query_datasets(require_experimental_session=False)
        assert {row["dataset_id"] for row in before} == {first_dataset, second_dataset}
        assert {row["experimental_session_id"] for row in before} == {None}

        _create_session(registry)
        assignments = registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_20260810_001",
            recording_ids=("recording_arena_1", "recording_arena_2"),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
            assigned_at_utc="2026-08-10T12:01:00+00:00",
            assignment_batch_id="22222222-2222-4222-8222-222222222222",
            evidence={"source": "four_arena_manifest.json"},
        )

        assert len(assignments) == 2
        assert {item.experimental_session_id for item in assignments} == {
            "experiment_20260810_001"
        }
        assert {item.assignment_batch_id for item in assignments} == {
            "22222222-2222-4222-8222-222222222222"
        }
        assert len({item.assignment_snapshot_id for item in assignments}) == 2
        assert {item.assignment_revision for item in assignments} == {1}
        assert {item.supersedes_assignment_snapshot_id for item in assignments} == {
            None
        }

        rows = registry.query_datasets(
            experimental_session_id="experiment_20260810_001",
            require_experimental_session=True,
        )
        assert {row["dataset_id"] for row in rows} == {first_dataset, second_dataset}
        assert {row["session_uuid"] for row in rows} == {
            "acquisition_arena_1",
            "acquisition_arena_2",
        }
        assert {row["recording_started_utc"] for row in rows} == {
            "2026-08-10T12:00:00+00:00",
            "2026-08-10T12:00:01+00:00",
        }
        assert {row["experimental_session_identity_status"] for row in rows} == {
            "explicit"
        }
        assert {
            registry.resolve_dataset_experimental_session_assignment(
                first_dataset
            ).recording_id,
            registry.resolve_dataset_experimental_session_assignment(
                second_dataset
            ).recording_id,
        } == {"recording_arena_1", "recording_arena_2"}
        second_assignment = next(
            item for item in assignments if item.recording_id == "recording_arena_2"
        )
        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                """
                UPDATE recording_experimental_session_current
                SET assignment_snapshot_id = ?
                WHERE recording_id = 'recording_arena_1';
                """,
                (second_assignment.assignment_snapshot_id,),
            )
        registry.conn.rollback()
    finally:
        registry.close()


def test_timestamps_and_names_never_infer_session_identity(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="matching_name_arena_1",
            session_uuid="matching_name_arena_1",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_1",
        )
        _register_recording(
            registry,
            recording_id="matching_name_arena_2",
            session_uuid="matching_name_arena_2",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_2",
        )
        _register_recording(
            registry,
            recording_id="matching_name_arena_3",
            session_uuid="matching_name_arena_3",
            started_utc="2026-08-10T12:00:01+00:00",
            arena_id="arena_3",
        )

        rows = registry.query_datasets(require_experimental_session=False)
        assert len(rows) == 3
        assert {row["experimental_session_identity_status"] for row in rows} == {
            "missing"
        }
        assert {row["experimental_session_id"] for row in rows} == {None}
    finally:
        registry.close()


def test_missing_and_recording_only_contexts_fail_closed(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = _register_recording(
            registry,
            recording_id="recording_unassigned",
            session_uuid="acquisition_unassigned",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_1",
        )
        registry.upsert_dataset(
            "dataset_recording_only",
            session_uuid="acquisition_recording_only",
            zarr_path=Path("/recordings/recording_only/analysis.zarr"),
            recording_id="recording_context_not_registered",
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
        registry.upsert_dataset(
            "dataset_without_recording",
            session_uuid="acquisition_without_recording",
            zarr_path=Path("/datasets/no_recording.zarr"),
            artifact_kind="training_artifact",
            zarr_use="training",
        )

        assert (
            registry.get_recording_experimental_session_assignment(
                "recording_unassigned", require_assigned=False
            )
            is None
        )
        with pytest.raises(MissingExperimentalSessionIdentityError):
            registry.resolve_dataset_experimental_session_assignment(dataset_id)
        with pytest.raises(UnknownRecordingIdentityError):
            registry.resolve_dataset_experimental_session_assignment(
                "dataset_recording_only"
            )
        with pytest.raises(MissingExperimentalSessionIdentityError):
            registry.resolve_dataset_experimental_session_assignment(
                "dataset_without_recording"
            )
        with pytest.raises(UnknownDatasetIdentityError):
            registry.resolve_dataset_experimental_session_assignment("unknown_dataset")

        rows = registry.query_datasets(require_experimental_session=False)
        assert {row["dataset_id"] for row in rows} == {
            dataset_id,
            "dataset_recording_only",
            "dataset_without_recording",
        }
    finally:
        registry.close()


def test_initial_assignments_are_atomic_unique_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        _register_recording(
            registry,
            recording_id="recording_b",
            session_uuid="acquisition_b",
            started_utc="2026-08-10T12:00:01+00:00",
            arena_id="arena_b",
        )
        _create_session(registry, "experiment_a")
        registry.create_experimental_session(
            experimental_session_id="experiment_b",
            creation_method="operator_manifest",
            created_by="pytest_operator",
            evidence={},
        )

        with pytest.raises(ExperimentalSessionIdentityError, match="unique"):
            registry.assign_recordings_to_experimental_session(
                experimental_session_id="experiment_a",
                recording_ids=("recording_a", "recording_a"),
                assignment_method="operator_manifest",
                assigned_by="pytest_operator",
            )
        with pytest.raises(UnknownRecordingIdentityError, match="recording_missing"):
            registry.assign_recordings_to_experimental_session(
                experimental_session_id="experiment_a",
                recording_ids=("recording_a", "recording_missing"),
                assignment_method="operator_manifest",
                assigned_by="pytest_operator",
            )
        assert (
            registry.get_recording_experimental_session_assignment(
                "recording_a", require_assigned=False
            )
            is None
        )

        registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_a",
            recording_ids=("recording_a",),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
        )
        for target in ("experiment_a", "experiment_b"):
            with pytest.raises(
                ExperimentalSessionAssignmentConflictError, match="correction API"
            ):
                registry.assign_recordings_to_experimental_session(
                    experimental_session_id=target,
                    recording_ids=("recording_a",),
                    assignment_method="operator_manifest",
                    assigned_by="pytest_operator",
                )
        assert (
            registry.get_recording_experimental_session_assignment(
                "recording_a"
            ).experimental_session_id
            == "experiment_a"
        )
        assert (
            registry.get_recording_experimental_session_assignment(
                "recording_b", require_assigned=False
            )
            is None
        )
    finally:
        registry.close()


def test_correction_is_append_only_audited_and_compare_and_swap_guarded(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        _create_session(registry, "experiment_a")
        registry.create_experimental_session(
            experimental_session_id="experiment_b",
            creation_method="operator_manifest",
            created_by="pytest_operator",
            evidence={"manifest_sha256": "b" * 64},
        )
        initial = registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_a",
            recording_ids=("recording_a",),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
            evidence={"reason": "initial manifest"},
        )[0]

        with pytest.raises(
            ExperimentalSessionAssignmentConflictError,
            match="compare-and-swap failed",
        ):
            registry.correct_recording_experimental_session_assignment(
                recording_id="recording_a",
                experimental_session_id="experiment_b",
                expected_current_assignment_snapshot_id=(
                    "33333333-3333-4333-8333-333333333333"
                ),
                assignment_method="operator_correction",
                assigned_by="pytest_operator",
                evidence={"reason": "stale caller"},
            )
        assert registry.list_recording_experimental_session_assignment_history(
            "recording_a"
        ) == (initial,)

        corrected = registry.correct_recording_experimental_session_assignment(
            recording_id="recording_a",
            experimental_session_id="experiment_b",
            expected_current_assignment_snapshot_id=initial.assignment_snapshot_id,
            assignment_method="operator_correction",
            assigned_by="pytest_operator",
            assigned_at_utc="2026-08-10T13:00:00Z",
            assignment_batch_id="44444444-4444-4444-8444-444444444444",
            assignment_snapshot_id="55555555-5555-4555-8555-555555555555",
            evidence={"reason": "source manifest corrected", "ticket": "LAB-17"},
        )

        assert corrected.experimental_session_id == "experiment_b"
        assert corrected.assignment_revision == 2
        assert corrected.supersedes_assignment_snapshot_id == (
            initial.assignment_snapshot_id
        )
        assert corrected.assignment_snapshot_id == (
            "55555555-5555-4555-8555-555555555555"
        )
        history = registry.list_recording_experimental_session_assignment_history(
            "recording_a"
        )
        assert history == (initial, corrected)
        assert history[0].evidence == {"reason": "initial manifest"}
        assert history[1].evidence == {
            "reason": "source manifest corrected",
            "ticket": "LAB-17",
        }
        assert (
            registry.get_recording_experimental_session_assignment(
                "recording_a"
            ).assignment_snapshot_id
            == corrected.assignment_snapshot_id
        )

        with pytest.raises(
            ExperimentalSessionAssignmentConflictError,
            match="compare-and-swap failed",
        ):
            registry.correct_recording_experimental_session_assignment(
                recording_id="recording_a",
                experimental_session_id="experiment_a",
                expected_current_assignment_snapshot_id=initial.assignment_snapshot_id,
                assignment_method="operator_correction",
                assigned_by="pytest_operator",
            )
        assert registry.list_recording_experimental_session_assignment_history(
            "recording_a"
        ) == (initial, corrected)
    finally:
        registry.close()


def test_recording_and_session_deletes_cannot_erase_assignment_history(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        _create_session(registry, "experiment_a")
        initial = registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_a",
            recording_ids=("recording_a",),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
        )[0]

        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                "DELETE FROM recordings WHERE recording_id = 'recording_a';"
            )
        registry.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            registry.conn.execute(
                "DELETE FROM experimental_sessions "
                "WHERE experimental_session_id = 'experiment_a';"
            )
        registry.conn.rollback()
        assert registry.list_recording_experimental_session_assignment_history(
            "recording_a"
        ) == (initial,)
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("session_id", "method", "actor", "evidence"),
    [
        (" leading_space", "operator_manifest", "operator", {}),
        ("valid_id", "Operator Manifest", "operator", {}),
        ("valid_id", "operator_manifest", "", {}),
        ("valid_id", "operator_manifest", "operator", {"bad": float("nan")}),
    ],
)
def test_session_creation_rejects_invalid_contract_fields(
    tmp_path: Path,
    session_id: str,
    method: str,
    actor: str,
    evidence: dict[str, object],
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(ExperimentalSessionIdentityError):
            registry.create_experimental_session(
                experimental_session_id=session_id,
                creation_method=method,
                created_by=actor,
                evidence=evidence,
            )
        assert (
            registry.conn.execute(
                "SELECT COUNT(*) FROM experimental_sessions;"
            ).fetchone()[0]
            == 0
        )
    finally:
        registry.close()


def test_session_entity_and_assignment_provenance_is_exact(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        _create_session(registry)
        assignment = registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_20260810_001",
            recording_ids=("recording_a",),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
            assigned_at_utc="2026-08-10T12:01:00Z",
            assignment_batch_id="22222222-2222-4222-8222-222222222222",
            evidence={"manifest_sha256": "b" * 64},
        )[0]
        session = registry.get_experimental_session("experiment_20260810_001")
        row = registry.query_datasets(
            experimental_session_id="experiment_20260810_001"
        )[0]

        assert session.schema_id == EXPERIMENTAL_SESSION_SCHEMA_ID
        assert session.session_snapshot_id == "11111111-1111-4111-8111-111111111111"
        assert session.created_at_utc == "2026-08-10T12:00:00+00:00"
        assert session.registry_schema_version == 67
        assert assignment.schema_id == EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID
        assert assignment.assignment_revision == 1
        assert assignment.supersedes_assignment_snapshot_id is None
        assert assignment.assigned_at_utc == "2026-08-10T12:01:00+00:00"
        assert assignment.evidence == {"manifest_sha256": "b" * 64}
        assert assignment.registry_schema_version == 67
        assert row["dataset_id"] == dataset_id
        assert row["experimental_session_assignment_snapshot_id"] == (
            assignment.assignment_snapshot_id
        )
        assert row["experimental_session_assignment_batch_id"] == (
            assignment.assignment_batch_id
        )
        assert row["experimental_session_assignment_schema_id"] == (
            EXPERIMENTAL_SESSION_ASSIGNMENT_SCHEMA_ID
        )
        assert row["experimental_session_assignment_revision"] == 1
        assert row["experimental_session_supersedes_assignment_snapshot_id"] is None
        assert row["experimental_session_snapshot_id"] == session.session_snapshot_id
        assert row["experimental_session_schema_id"] == EXPERIMENTAL_SESSION_SCHEMA_ID
        assert row["experimental_session_creation_registry_schema_version"] == 67
        assert row["experimental_session_assignment_registry_schema_version"] == 67
    finally:
        registry.close()


def test_migration_preserves_existing_rows_as_unassigned(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    registry = Registry(path)
    dataset_id = _register_recording(
        registry,
        recording_id="legacy_recording",
        session_uuid="legacy_acquisition_session",
        started_utc="2026-08-10T12:00:00+00:00",
        arena_id="legacy_arena",
    )
    registry.close()

    with sqlite3.connect(path) as conn:
        conn.execute("DROP VIEW recording_step_status_latest;")
        conn.execute("DROP VIEW dataset_context_current;")
        conn.execute("DROP TABLE recording_experimental_session_current;")
        conn.execute("DROP TABLE recording_experimental_session_assignments;")
        conn.execute("DROP TABLE experimental_sessions;")
        conn.execute("DELETE FROM schema_version WHERE version = 67;")
        conn.execute("PRAGMA user_version = 66;")
        conn.commit()

    reopened = Registry(path)
    try:
        row = reopened.query_datasets()[0]
        assert row["dataset_id"] == dataset_id
        assert row["session_uuid"] == "legacy_acquisition_session"
        assert row["experimental_session_id"] is None
        assert row["experimental_session_identity_status"] == "missing"
        assert reopened._current_schema_version() == 67
        assert (
            reopened.conn.execute(
                "SELECT COUNT(*) FROM recording_experimental_session_assignments;"
            ).fetchone()[0]
            == 0
        )
    finally:
        reopened.close()


def test_registry_query_cli_exposes_exact_session_filters(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        _create_session(registry)
        registry.assign_recordings_to_experimental_session(
            experimental_session_id="experiment_20260810_001",
            recording_ids=("recording_a",),
            assignment_method="operator_manifest",
            assigned_by="pytest_operator",
        )
        args = _parse_args(
            [
                "--experimental-session-id",
                "experiment_20260810_001",
                "--experimental-session-status",
                "explicit",
            ]
        )
        sql, params = _build_query(args)
        row = registry.conn.execute(sql, params).fetchone()
        assert row is not None
        assert row["experimental_session_id"] == "experiment_20260810_001"
        assert row["experimental_session_identity_status"] == "explicit"
    finally:
        registry.close()


def test_unknown_session_and_duplicate_entity_fail_closed(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _register_recording(
            registry,
            recording_id="recording_a",
            session_uuid="acquisition_a",
            started_utc="2026-08-10T12:00:00+00:00",
            arena_id="arena_a",
        )
        with pytest.raises(UnknownExperimentalSessionError):
            registry.assign_recordings_to_experimental_session(
                experimental_session_id="unknown_session",
                recording_ids=("recording_a",),
                assignment_method="operator_manifest",
                assigned_by="pytest_operator",
            )
        _create_session(registry)
        with pytest.raises(
            ExperimentalSessionAssignmentConflictError, match="already exists"
        ):
            _create_session(registry)
    finally:
        registry.close()

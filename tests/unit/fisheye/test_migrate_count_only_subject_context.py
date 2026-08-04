from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.experiment_setup import resolve_experiment_setup
from fisheye.shared.subject_metadata import resolve_subject_metadata
from fisheye.shared.subject_metadata import publish_subject_metadata
from fisheye.utils.migrate_count_only_subject_context import (
    LEGACY_IDENTITY_SCOPE,
    LEGACY_SOURCE,
    apply_plan,
    build_plan,
    select_targets,
)

RECORDING_ID = "2026-07-01T14-32-13Z_arena_1_DefaultScreen"
REVIEWER = "test-reviewer"
REASON = "Replace synthetic recording-local IDs with anonymous count context."


def _legacy_metadata(*, subject_id: str, dpf: int = 5, count: int = 1) -> dict:
    return {
        "source": LEGACY_SOURCE,
        "status": "manual_backfill",
        "recording_id": RECORDING_ID,
        "species": "Danionella cerebrum",
        "subject_count": count,
        "subject_type": "individual" if count == 1 else "group",
        "subject_id": subject_id,
        "subject_ids": [subject_id],
        "identity_scope": LEGACY_IDENTITY_SCOPE,
        "dpf_at_acquisition": dpf,
        "days_post_fertilization": dpf,
        "date_of_fertilization": "2026-06-26",
        "backfilled_at_utc": "2026-07-06T00:00:00+00:00",
    }


def _archive(
    tmp_path: Path,
    *,
    dataset_id: str,
    zarr_use: str,
    subject_id: str,
    dpf: int = 5,
) -> Path:
    path = tmp_path / "recordings" / RECORDING_ID / "zarr" / f"{dataset_id}.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "session_uuid": dataset_id,
            "recording_id": RECORDING_ID,
            "recording_name": RECORDING_ID,
            "recording_path": str(path.parent.parent),
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "zarr_purpose": zarr_use,
            "zarr_use": zarr_use,
            "subject_count": 1,
        }
    )
    analysis_metadata = root.require_group("analysis_metadata")
    analysis_metadata.attrs["subject_metadata"] = json.dumps(
        _legacy_metadata(subject_id=subject_id, dpf=dpf), sort_keys=True
    )
    return path


def _registry_fixture(
    tmp_path: Path, *, training_dpf: int = 5
) -> tuple[Path, Path, Path]:
    analysis_path = _archive(
        tmp_path,
        dataset_id="analysis-dataset",
        zarr_use="analysis",
        subject_id=f"{RECORDING_ID}:subject_0",
    )
    training_path = _archive(
        tmp_path,
        dataset_id="training-dataset",
        zarr_use="training",
        subject_id="2026-07-01T14-32-13Z_arena_1:subject_0",
        dpf=training_dpf,
    )
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        registry.upsert_recording(
            recording_id=RECORDING_ID,
            recording_name=RECORDING_ID,
            recording_path=str(analysis_path.parent.parent),
            recording_type="behavior",
            recording_subtype="free",
            behavior_mode="free",
        )
        for dataset_id, path, zarr_use in (
            ("analysis-dataset", analysis_path, "analysis"),
            ("training-dataset", training_path, "training"),
        ):
            registry.upsert_dataset(
                dataset_id,
                session_uuid=dataset_id,
                recording_id=RECORDING_ID,
                zarr_path=path,
                artifact_kind="source_recording",
                zarr_purpose=zarr_use,
                zarr_use=zarr_use,
            )
        now = "2026-07-06T00:00:00+00:00"
        for dataset_id, subject_id in (
            ("analysis-dataset", f"{RECORDING_ID}:subject_0"),
            ("training-dataset", "2026-07-01T14-32-13Z_arena_1:subject_0"),
        ):
            registry.conn.execute(
                """
                INSERT INTO subjects (
                    subject_id, species, metadata_json, created_utc, updated_utc
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (
                    subject_id,
                    "Danionella cerebrum",
                    json.dumps({"source": LEGACY_SOURCE, "recording_id": RECORDING_ID}),
                    now,
                    now,
                ),
            )
            registry.conn.execute(
                """
                INSERT INTO recording_subjects (
                    recording_id, subject_id, dataset_id, species,
                    dpf_at_acquisition, metadata_json, created_utc, updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    RECORDING_ID,
                    subject_id,
                    dataset_id,
                    "Danionella cerebrum",
                    5,
                    json.dumps(
                        {
                            "source": LEGACY_SOURCE,
                            "identity_scope": LEGACY_IDENTITY_SCOPE,
                            "dataset_id": dataset_id,
                        },
                        sort_keys=True,
                    ),
                    now,
                    now,
                ),
            )
        registry.conn.commit()
    finally:
        registry.close()
    return registry_path, analysis_path, training_path


def test_migration_publishes_count_only_authority_and_removes_placeholders(
    tmp_path: Path,
) -> None:
    registry_path, analysis_path, _ = _registry_fixture(tmp_path)
    targets = select_targets(registry_path, all_placeholders=True)
    assert [target.dataset_id for target in targets] == ["analysis-dataset"]

    plan = build_plan(
        registry_path,
        targets,
        reviewer=REVIEWER,
        reason=REASON,
    )
    assert plan["disposition_counts"] == {"eligible": 1}
    assert plan["action_counts"] == {"publish": 1}
    assert len(plan["recordings"][0]["cleanup"]["recording_subjects"]) == 2

    result = apply_plan(registry_path, plan)
    assert result["disposition_counts"] == {"applied": 1}
    assert result["recordings"][0]["deleted_recording_subjects"] == 2
    assert result["recordings"][0]["deleted_orphan_subjects"] == 2

    root = zarr.open_group(str(analysis_path), mode="r", use_consolidated=False)
    subject = resolve_subject_metadata(root, allow_legacy=False)
    setup = resolve_experiment_setup(root, allow_legacy=False)
    assert subject.subject_ids == ()
    assert subject.subject_identity_kind == "none"
    assert subject.metadata["species"] == "Danionella cerebrum"
    assert subject.metadata["subject_count"] == 1
    assert subject.metadata["manual_assertion"]["reviewer"] == REVIEWER
    assert setup.expected_subject_count == 1
    assert setup.assigned_subject_count is None
    assert setup.subject_assignment_status == "count_only"
    assert setup.source["kind"] == "manual_operator_assertion"

    registry = Registry(registry_path)
    try:
        assert (
            registry.conn.execute(
                "SELECT COUNT(*) FROM recording_subjects WHERE recording_id = ?",
                (RECORDING_ID,),
            ).fetchone()[0]
            == 0
        )
        assert registry.conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 0
        rows = registry.conn.execute(
            """
            SELECT subject_count_snapshot, subject_count_effective, subject_count_recorded,
                   subject_identity_status, subject_context_source,
                   species_effective, dpf_at_acquisition_effective,
                   subject_id, subject_ids_json
            FROM dataset_context_current
            WHERE recording_id = ?
            ORDER BY dataset_id;
            """,
            (RECORDING_ID,),
        ).fetchall()
    finally:
        registry.close()
    assert len(rows) == 2
    for row in rows:
        assert int(row["subject_count_effective"]) == 1
        assert row["subject_count_recorded"] is None
        assert row["subject_identity_status"] == "count_only"
        assert row["subject_context_source"] == "count_only"
        assert row["species_effective"] == "Danionella cerebrum"
        assert int(row["dpf_at_acquisition_effective"]) == 5
        assert row["subject_id"] is None
        assert row["subject_ids_json"] is None


def test_migration_is_idempotent_after_placeholder_cleanup(tmp_path: Path) -> None:
    registry_path, analysis_path, _ = _registry_fixture(tmp_path)
    first = build_plan(
        registry_path,
        select_targets(registry_path, all_placeholders=True),
        reviewer=REVIEWER,
        reason=REASON,
    )
    assert apply_plan(registry_path, first)["disposition_counts"] == {"applied": 1}
    before = zarr.open_group(str(analysis_path), mode="r", use_consolidated=False)
    subject_runs = list(before["analysis/subject_metadata_runs"].group_keys())
    setup_runs = list(before["analysis/experiment_setup_runs"].group_keys())

    second = build_plan(
        registry_path,
        select_targets(registry_path, recording_ids=[RECORDING_ID]),
        reviewer=REVIEWER,
        reason=REASON,
    )
    assert second["action_counts"] == {"verify_existing": 1}
    assert second["recordings"][0]["cleanup"]["recording_subjects"] == []
    result = apply_plan(registry_path, second)
    assert result["action_counts"] == {"verified_existing": 1}
    assert result["recordings"][0]["deleted_recording_subjects"] == 0

    after = zarr.open_group(str(analysis_path), mode="r", use_consolidated=False)
    assert list(after["analysis/subject_metadata_runs"].group_keys()) == subject_runs
    assert list(after["analysis/experiment_setup_runs"].group_keys()) == setup_runs


def test_migration_recovers_when_subject_published_before_setup(tmp_path: Path) -> None:
    registry_path, analysis_path, _ = _registry_fixture(tmp_path)
    plan = build_plan(
        registry_path,
        select_targets(registry_path, all_placeholders=True),
        reviewer=REVIEWER,
        reason=REASON,
    )
    desired = plan["recordings"][0]["desired"]
    root = zarr.open_group(str(analysis_path), mode="r+", use_consolidated=False)
    publish_subject_metadata(
        root,
        desired["subject_metadata_record"]["subject_metadata"],
    )
    assert "analysis/experiment_setup_runs" not in root

    result = apply_plan(registry_path, plan)

    assert result["disposition_counts"] == {"applied": 1}
    assert result["recordings"][0]["deleted_recording_subjects"] == 2
    reopened = zarr.open_group(str(analysis_path), mode="r", use_consolidated=False)
    assert resolve_experiment_setup(
        reopened, allow_legacy=False
    ).subject_assignment_status == ("count_only")


def test_migration_blocks_an_explicit_or_unrelated_subject_membership(
    tmp_path: Path,
) -> None:
    registry_path, _, _ = _registry_fixture(tmp_path)
    registry = Registry(registry_path)
    try:
        now = "2026-07-06T00:00:00+00:00"
        registry.conn.execute(
            """
            INSERT INTO subjects (
                subject_id, species, metadata_json, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (
                "verified-subject",
                "Danionella cerebrum",
                json.dumps({"source": "verified_biological_identity"}),
                now,
                now,
            ),
        )
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, species,
                metadata_json, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                RECORDING_ID,
                "verified-subject",
                "analysis-dataset",
                "Danionella cerebrum",
                json.dumps({"source": "verified_biological_identity"}),
                now,
                now,
            ),
        )
        registry.conn.commit()
    finally:
        registry.close()

    plan = build_plan(
        registry_path,
        select_targets(registry_path, all_placeholders=True),
        reviewer=REVIEWER,
        reason=REASON,
    )

    assert plan["disposition_counts"] == {"blocked": 1}
    assert plan["reason_counts"] == {
        "explicit_or_unrelated_subject_membership_present": 1
    }


def test_apply_rejects_a_modified_review_plan(tmp_path: Path) -> None:
    registry_path, _, _ = _registry_fixture(tmp_path)
    plan = build_plan(
        registry_path,
        select_targets(registry_path, all_placeholders=True),
        reviewer=REVIEWER,
        reason=REASON,
    )
    modified = deepcopy(plan)
    modified["reason"] = "changed after review"

    with pytest.raises(ValueError, match="digest mismatch"):
        apply_plan(registry_path, modified)


def test_migration_blocks_disagreement_between_analysis_and_training_context(
    tmp_path: Path,
) -> None:
    registry_path, _, _ = _registry_fixture(tmp_path, training_dpf=6)
    plan = build_plan(
        registry_path,
        select_targets(registry_path, all_placeholders=True),
        reviewer=REVIEWER,
        reason=REASON,
    )

    assert plan["disposition_counts"] == {"blocked": 1}
    assert plan["reason_counts"] == {"legacy_placeholder_context_disagreement": 1}


def test_registry_scan_does_not_project_recording_local_placeholder_identity(
    tmp_path: Path,
) -> None:
    path = _archive(
        tmp_path,
        dataset_id="legacy-only",
        zarr_use="analysis",
        subject_id=f"{RECORDING_ID}:subject_0",
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        root = zarr.open_group(str(path), mode="r", use_consolidated=False)
        dataset_id = registry.register_from_root(root, path)
        provenance = registry.conn.execute(
            "SELECT fish_id, subject_count, species, dpf_at_acquisition "
            "FROM provenance WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        context = registry.conn.execute(
            "SELECT subject_identity_status, subject_count_effective "
            "FROM dataset_context_current WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        memberships = registry.conn.execute(
            "SELECT COUNT(*) FROM recording_subjects"
        ).fetchone()[0]
    finally:
        registry.close()

    assert dict(provenance) == {
        "fish_id": None,
        "subject_count": 1,
        "species": "Danionella cerebrum",
        "dpf_at_acquisition": 5,
    }
    assert dict(context) == {
        "subject_identity_status": "count_only",
        "subject_count_effective": 1,
    }
    assert memberships == 0


def test_multi_subject_count_only_metadata_does_not_create_memberships(
    tmp_path: Path,
) -> None:
    path = _archive(
        tmp_path,
        dataset_id="anonymous-pair",
        zarr_use="analysis",
        subject_id=f"{RECORDING_ID}:subject_0",
    )
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    metadata = _legacy_metadata(subject_id=f"{RECORDING_ID}:subject_0", count=2)
    metadata["subject_ids"] = [
        f"{RECORDING_ID}:subject_0",
        f"{RECORDING_ID}:subject_1",
    ]
    root["analysis_metadata"].attrs["subject_metadata"] = json.dumps(metadata)

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.register_from_root(root, path)
        context = registry.conn.execute(
            "SELECT subject_count_effective, subject_identity_status "
            "FROM dataset_context_current WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        memberships = registry.conn.execute(
            "SELECT COUNT(*) FROM recording_subjects"
        ).fetchone()[0]
    finally:
        registry.close()
    assert dict(context) == {
        "subject_count_effective": 2,
        "subject_identity_status": "count_only",
    }
    assert memberships == 0


def test_registry_labels_partial_identity_without_inventing_remaining_subjects(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_recording(recording_id="recording_partial")
        registry.upsert_dataset(
            "dataset_partial",
            session_uuid="dataset_partial",
            recording_id="recording_partial",
            zarr_path=tmp_path / "partial.zarr",
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
        registry.upsert_provenance(
            "dataset_partial",
            provenance={"subject_count": 2},
            context={},
            protocol_name=None,
            protocol_hash=None,
            acquisition={},
            zarr_purpose="analysis",
        )
        now = "2026-07-06T00:00:00+00:00"
        registry.conn.execute(
            """
            INSERT INTO subjects (subject_id, created_utc, updated_utc)
            VALUES (?, ?, ?);
            """,
            ("known-subject", now, now),
        )
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (
                "recording_partial",
                "known-subject",
                "dataset_partial",
                now,
                now,
            ),
        )
        registry.conn.commit()
        row = registry.conn.execute(
            """
            SELECT subject_count_snapshot, subject_count_effective,
                   subject_count_recorded,
                   subject_identity_status, subject_ids_json
            FROM dataset_context_current
            WHERE dataset_id = 'dataset_partial';
            """
        ).fetchone()
    finally:
        registry.close()

    assert int(row["subject_count_snapshot"]) == 2
    assert int(row["subject_count_effective"]) == 1
    assert int(row["subject_count_recorded"]) == 1
    assert row["subject_identity_status"] == "partial"
    assert json.loads(row["subject_ids_json"]) == ["known-subject"]

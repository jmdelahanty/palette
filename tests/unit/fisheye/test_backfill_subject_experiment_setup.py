from __future__ import annotations

from pathlib import Path

import h5py
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.experiment_setup import resolve_experiment_setup
from fisheye.shared.subject_metadata import (
    publish_subject_metadata,
    resolve_subject_metadata,
)
from fisheye.utils.backfill_subject_experiment_setup import (
    apply_backfill_plan,
    build_backfill_plan,
    select_backfill_targets,
)


SUBJECT_ID = "40d99fea-846b-4890-bad2-b4e152dfdde0"


def _archive(tmp_path: Path, recording_id: str) -> tuple[Path, zarr.Group]:
    recording_dir = tmp_path / "recordings" / recording_id
    zarr_path = recording_dir / "zarr" / f"{recording_id}_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "session_uuid": recording_id,
            "recording_id": recording_id,
            "recording_name": recording_id,
            "recording_path": str(recording_dir),
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
            "zarr_purpose": "analysis",
            "zarr_use": "analysis",
        }
    )
    return zarr_path, root


def _write_subject_h5(
    zarr_path: Path,
    *,
    subject_id: str | None = SUBJECT_ID,
    subject_count: int | None = 1,
) -> Path:
    recording_dir = zarr_path.parent.parent
    h5_path = recording_dir / "raw" / "session.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as h5:
        subject = h5.create_group("subject_metadata")
        if subject_id is not None:
            subject.attrs["fish_id"] = subject_id
        if subject_count is not None:
            subject.attrs["subject_count"] = subject_count
        subject.attrs["fish_count"] = 35
        subject.attrs["species"] = "Danio rerio"
    return h5_path


def _register(registry_path: Path, zarr_path: Path) -> str:
    registry = Registry(registry_path)
    try:
        return str(registry.scan_zarr(zarr_path))
    finally:
        registry.close()


def test_backfill_dry_run_apply_refresh_and_idempotency(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path, root = _archive(tmp_path, "batman_001")
    _write_subject_h5(zarr_path)
    dataset_id = _register(registry_path, zarr_path)

    targets = select_backfill_targets(registry_path, all_recordings=True)
    assert [target.dataset_id for target in targets] == [dataset_id]

    dry_run = build_backfill_plan(targets)
    assert dry_run["action_counts"] == {"publish": 1}
    assert "analysis/subject_metadata_runs" not in root
    assert "analysis/experiment_setup_runs" not in root

    applied = apply_backfill_plan(registry_path, dry_run)
    assert applied["disposition_counts"] == {"applied": 1}
    assert applied["action_counts"] == {"published": 1}

    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    subject = resolve_subject_metadata(reopened, allow_legacy=False)
    setup = resolve_experiment_setup(reopened, allow_legacy=False)
    assert subject.subject_ids == (SUBJECT_ID,)
    assert setup.expected_subject_count == 1
    assert setup.subject_metadata_ref == subject.group_path
    subject_runs = list(reopened["analysis/subject_metadata_runs"].group_keys())
    setup_runs = list(reopened["analysis/experiment_setup_runs"].group_keys())

    registry = Registry(registry_path)
    try:
        provenance = registry.conn.execute(
            "SELECT fish_id, subject_count FROM provenance WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        membership = registry.conn.execute(
            "SELECT recording_id, subject_id FROM recording_subjects WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    finally:
        registry.close()
    assert dict(provenance) == {"fish_id": SUBJECT_ID, "subject_count": 1}
    assert dict(membership) == {
        "recording_id": "batman_001",
        "subject_id": SUBJECT_ID,
    }

    second_plan = build_backfill_plan(targets)
    assert second_plan["action_counts"] == {"verify_existing": 1}
    second_apply = apply_backfill_plan(registry_path, second_plan)
    assert second_apply["action_counts"] == {"verified_existing": 1}
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert list(reopened["analysis/subject_metadata_runs"].group_keys()) == subject_runs
    assert list(reopened["analysis/experiment_setup_runs"].group_keys()) == setup_runs


def test_backfill_skips_missing_source_identity_and_excludes_training(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    missing_h5_path, _ = _archive(tmp_path, "missing_h5")
    _register(registry_path, missing_h5_path)
    missing_id_path, _ = _archive(tmp_path, "missing_identity")
    _write_subject_h5(missing_id_path, subject_id=None)
    _register(registry_path, missing_id_path)

    training_path = (
        tmp_path
        / "recordings"
        / "training_only"
        / "zarr"
        / "training_only_training.zarr"
    )
    training = zarr.open_group(str(training_path), mode="w", zarr_format=3)
    training.attrs.update(
        {
            "session_uuid": "training_only",
            "recording_id": "training_only",
            "zarr_purpose": "training",
            "zarr_use": "training",
        }
    )
    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            "training-only",
            session_uuid="training_only",
            recording_id="training_only",
            zarr_path=training_path,
            artifact_kind="source_recording",
            zarr_purpose="training",
            zarr_use="training",
        )
    finally:
        registry.close()

    targets = select_backfill_targets(registry_path, all_recordings=True)
    assert {target.recording_id for target in targets} == {
        "missing_h5",
        "missing_identity",
    }
    plan = build_backfill_plan(targets)
    assert plan["disposition_counts"] == {"skipped": 2}
    assert plan["reason_counts"] == {
        "explicit_subject_identity_missing": 1,
        "source_h5_missing": 1,
    }


def test_backfill_blocks_existing_authority_that_conflicts_with_h5(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path, root = _archive(tmp_path, "conflict")
    _write_subject_h5(zarr_path)
    publish_subject_metadata(
        root,
        {
            "subject_count": 1,
            "fish_id": "3fc97b1d-5f17-4b9f-90f4-33c690553f1f",
        },
    )
    _register(registry_path, zarr_path)

    plan = build_backfill_plan(
        select_backfill_targets(registry_path, all_recordings=True)
    )

    assert plan["disposition_counts"] == {"blocked": 1}
    assert plan["reason_counts"] == {"subject_metadata_conflicts_with_h5": 1}

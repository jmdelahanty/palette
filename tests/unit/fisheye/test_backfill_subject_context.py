import json
from pathlib import Path

import h5py
import pytest

from fisheye.registry.db import Registry
from fisheye.utils import backfill_subject_context as backfill


def test_legacy_placeholder_cli_refuses_new_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        backfill.main(
            [
                "--registry",
                str(tmp_path / "registry.sqlite"),
                "--species",
                "Danionella cerebrum",
                "--apply",
            ]
        )
    assert exc_info.value.code == 2
    assert "--apply is retired" in capsys.readouterr().err


def _write_group(path: Path, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs or {},
            }
        ),
        encoding="utf-8",
    )


def _read_attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text(encoding="utf-8"))["attributes"]


def _make_registry_with_dataset(tmp_path: Path) -> tuple[Path, Path, str, str]:
    recording_id = "2026-07-01T14-32-13Z_arena_1_DefaultScreen"
    dataset_id = "dataset-defaultscreen-a1"
    recording_dir = tmp_path / "recordings" / recording_id
    zarr_path = recording_dir / "zarr" / f"{recording_id}_analysis.zarr"
    _write_group(zarr_path, {"zarr_purpose": "analysis"})
    _write_group(zarr_path / "analysis_metadata", {})

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        now = "2026-07-06T00:00:00+00:00"
        registry.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, recording_name, recording_path, protocol_name, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (recording_id, recording_id, str(recording_dir), "DefaultScreen", now, now),
        )
        registry.conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, recording_id, zarr_path, artifact_kind, zarr_use, status, created_utc, last_seen_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (dataset_id, recording_id, str(zarr_path), "source_recording", "analysis", "active", now, now),
        )
        registry.conn.commit()
    finally:
        registry.close()
    return registry_path, zarr_path, dataset_id, recording_id


def _load_single_target(registry_path: Path) -> backfill.DatasetTarget:
    targets = backfill.load_targets_from_registry(
        registry_path,
        recording_id_contains="DefaultScreen",
        zarr_use="analysis",
    )
    assert len(targets) == 1
    return targets[0]


def test_dry_run_plans_zarr_and_registry_without_mutating(tmp_path: Path) -> None:
    registry_path, zarr_path, _, _ = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)

    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        apply=False,
        overwrite=False,
    )

    assert rows[0]["status"] == "planned"
    assert rows[0]["zarr"]["status"] == "planned"
    assert rows[0]["registry"]["status"] == "planned"
    assert "subject_metadata" not in _read_attrs(zarr_path / "analysis_metadata")
    registry = Registry(registry_path)
    try:
        count = registry.conn.execute("SELECT COUNT(*) FROM recording_subjects;").fetchone()[0]
    finally:
        registry.close()
    assert count == 0


def test_apply_writes_zarr_subject_metadata_and_registry_context(tmp_path: Path) -> None:
    registry_path, zarr_path, _, recording_id = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)

    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        apply=True,
        overwrite=False,
    )

    assert rows[0]["status"] == "updated"
    analysis_attrs = _read_attrs(zarr_path / "analysis_metadata")
    subject_meta = json.loads(analysis_attrs["subject_metadata"])
    assert subject_meta["species"] == "Danionella cerebrum"
    assert subject_meta["subject_count"] == 1
    assert subject_meta["subject_id"] == f"{recording_id}:subject_0"

    root_attrs = _read_attrs(zarr_path)
    assert root_attrs["subject_count"] == 1
    assert root_attrs["experiment_setup"]["total_expected_fish"] == 1

    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            """
            SELECT subject_count_effective, subject_context_source, species, subject_id
            FROM dataset_context_current
            WHERE dataset_id = ?;
            """,
            (target.dataset_id,),
        ).fetchone()
    finally:
        registry.close()
    assert int(row["subject_count_effective"]) == 1
    assert row["subject_context_source"] == "normalized"
    assert row["species"] == "Danionella cerebrum"
    assert row["subject_id"] == f"{recording_id}:subject_0"


def test_apply_is_idempotent_for_matching_subject_context(tmp_path: Path) -> None:
    registry_path, _, _, _ = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)
    kwargs = dict(
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        overwrite=False,
    )

    first = backfill.backfill_subject_context_for_targets(registry_path, [target], apply=True, **kwargs)
    second = backfill.backfill_subject_context_for_targets(registry_path, [target], apply=True, **kwargs)

    assert first[0]["status"] == "updated"
    assert second[0]["status"] == "skipped"


def test_second_pass_adds_dpf_and_fertilization_date_to_existing_subject_context(tmp_path: Path) -> None:
    registry_path, zarr_path, _, _ = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)
    common = dict(
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        overwrite=False,
    )

    backfill.backfill_subject_context_for_targets(registry_path, [target], apply=True, **common)
    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        apply=True,
        dpf_at_acquisition=9,
        date_of_fertilization="2026-06-22",
        **common,
    )

    assert rows[0]["status"] == "updated"
    analysis_attrs = _read_attrs(zarr_path / "analysis_metadata")
    subject_meta = json.loads(analysis_attrs["subject_metadata"])
    assert subject_meta["dpf_at_acquisition"] == 9
    assert subject_meta["days_post_fertilization"] == 9
    assert subject_meta["date_of_fertilization"] == "2026-06-22"

    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            """
            SELECT dpf_at_acquisition
            FROM dataset_context_current
            WHERE dataset_id = ?;
            """,
            (target.dataset_id,),
        ).fetchone()
        subject_row = registry.conn.execute(
            """
            SELECT metadata_json
            FROM recording_subjects
            WHERE recording_id = ?;
            """,
            (target.recording_id,),
        ).fetchone()
    finally:
        registry.close()
    assert int(row["dpf_at_acquisition"]) == 9
    assert json.loads(subject_row["metadata_json"])["date_of_fertilization"] == "2026-06-22"


def test_dry_run_reports_registry_plan_when_existing_rows_lack_dpf(tmp_path: Path) -> None:
    registry_path, _, _, _ = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)
    common = dict(
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        overwrite=False,
    )

    backfill.backfill_subject_context_for_targets(registry_path, [target], apply=True, **common)
    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        apply=False,
        dpf_at_acquisition=9,
        date_of_fertilization="2026-06-22",
        **common,
    )

    assert rows[0]["status"] == "planned"
    assert rows[0]["registry"]["status"] == "planned"
    assert rows[0]["registry"]["reason"] == "dry_run"


def test_derives_dpf_from_fertilization_date_and_recording_date(tmp_path: Path) -> None:
    registry_path, zarr_path, _, _ = _make_registry_with_dataset(tmp_path)
    target = _load_single_target(registry_path)

    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        date_of_fertilization="2026-06-26",
        apply=True,
        overwrite=False,
    )

    assert rows[0]["dpf_at_acquisition"] == 5
    assert rows[0]["derived_context"]["dpf_source"] == "date_of_fertilization_and_recording_date"
    assert rows[0]["derived_context"]["dpf_from_date"]["recording_date"] == "2026-07-01"
    subject_meta = json.loads(_read_attrs(zarr_path / "analysis_metadata")["subject_metadata"])
    assert subject_meta["days_post_fertilization"] == 5
    assert subject_meta["date_of_fertilization"] == "2026-06-26"

    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            """
            SELECT dpf_at_acquisition
            FROM dataset_context_current
            WHERE dataset_id = ?;
            """,
            (target.dataset_id,),
        ).fetchone()
    finally:
        registry.close()
    assert int(row["dpf_at_acquisition"]) == 5


def test_can_derive_dpf_from_raw_h5_subject_metadata(tmp_path: Path) -> None:
    registry_path, zarr_path, _, _ = _make_registry_with_dataset(tmp_path)
    raw_dir = zarr_path.parent.parent / "raw"
    raw_dir.mkdir(parents=True)
    h5_path = raw_dir / "recording.h5"
    with h5py.File(h5_path, "w") as handle:
        subject = handle.create_group("subject_metadata")
        subject.attrs["days_post_fertilization"] = 11
        subject.attrs["date_of_fertilization"] = "2026-06-20"
    target = _load_single_target(registry_path)

    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        derive_dpf_from_metadata=True,
        apply=True,
        overwrite=False,
    )

    assert rows[0]["dpf_at_acquisition"] == 11
    assert rows[0]["date_of_fertilization"] == "2026-06-20"
    assert rows[0]["derived_context"]["dpf_source"] == "raw_h5.subject_metadata"
    subject_meta = json.loads(_read_attrs(zarr_path / "analysis_metadata")["subject_metadata"])
    assert subject_meta["days_post_fertilization"] == 11


def test_conflicting_registry_subject_rows_fail_without_partial_commit(tmp_path: Path) -> None:
    registry_path, zarr_path, _, recording_id = _make_registry_with_dataset(tmp_path)
    registry = Registry(registry_path)
    try:
        now = "2026-07-06T00:00:00+00:00"
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, species, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?);
            """,
            (recording_id, f"{recording_id}:other_subject", "Danio rerio", now, now),
        )
        registry.conn.commit()
    finally:
        registry.close()
    target = _load_single_target(registry_path)

    rows = backfill.backfill_subject_context_for_targets(
        registry_path,
        [target],
        species="Danionella cerebrum",
        subject_count=1,
        subject_id_template="{recording_id}:subject_{index}",
        apply=True,
        overwrite=False,
    )

    assert rows[0]["status"] == "conflict"
    assert rows[0]["registry"]["reason"] == "existing_recording_subjects_conflict"
    assert "subject_metadata" not in _read_attrs(zarr_path / "analysis_metadata")

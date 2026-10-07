from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.refinement.refine_detect import (
    _validate_quality_experiment_setup_binding,
)
from fisheye.shared.experiment_setup import (
    EXPERIMENT_SETUP_RECORD_ATTR,
    build_experiment_setup_record,
    publish_experiment_setup,
    resolve_experiment_setup,
    resolve_expected_subject_count,
)
from fisheye.shared.subject_metadata import publish_subject_metadata
from fisheye.utils.import_recording_analysis import (
    RecordingAnalysisPlan,
    import_experiment_setup,
)


def _subject_metadata() -> dict[str, object]:
    return {
        "subject_count": "1",
        "subject_type": "individual",
        "fish_id": "40d99fea-846b-4890-bad2-b4e152dfdde0",
        "fish_count": "35",
        "dish_id": "dish-001",
        "cross_id": "cross-001",
        "genotype": "wildtype",
        "species": "Danio rerio",
        "sex": "unknown",
        "days_post_fertilization": "7",
    }


def _build_setup_record(metadata: dict[str, object] | None = None) -> dict[str, object]:
    return build_experiment_setup_record(
        metadata or _subject_metadata(),
        subject_metadata_ref=(
            "analysis/subject_metadata_runs/subject_metadata_test"
        ),
        subject_metadata_sha256="a" * 64,
    )


def _publish_setup(root: zarr.Group):
    subject = publish_subject_metadata(root, _subject_metadata())
    return publish_experiment_setup(
        root,
        build_experiment_setup_record(
            _subject_metadata(),
            subject_metadata_ref=subject.group_path,
            subject_metadata_sha256=subject.record_sha256,
        ),
    )


def test_setup_separates_recording_count_from_source_dish_population() -> None:
    record = _build_setup_record()

    assert record["expected_subject_count"] == 1
    assert record["assigned_subject_count"] == 1
    assert record["source_dish_population_count"] == 35
    assert record["setup_type"] == "single_subject_single_arena"


def test_setup_models_multiple_anonymous_subjects_as_count_only() -> None:
    record = build_experiment_setup_record(
        {"subject_count": 3, "species": "Danionella cerebrum"},
        source={
            "kind": "manual_operator_assertion",
            "count_field": "subject_count",
        },
        subject_metadata_ref="analysis/subject_metadata_runs/subject_metadata_test",
        subject_metadata_sha256="a" * 64,
    )

    assert record["expected_subject_count"] == 3
    assert record["assigned_subject_count"] is None
    assert record["subject_assignment_status"] == "count_only"
    assert record["setup_type"] == "multi_subject_single_arena"
    assert record["source"]["kind"] == "manual_operator_assertion"


def test_setup_counts_a_singular_subject_id_like_the_subject_record() -> None:
    # Citrus subject identity v3 writes one ``subject_id``; the setup must count
    # ids with the subject-metadata rule, not its own (B3).
    from fisheye.shared.subject_metadata import build_subject_metadata_record

    metadata = {
        "subject_count": "1",
        "subject_type": "individual",
        "subject_id": "4512e0a1-2d7e-496a-8eb0-8d25f7aa93d9",
    }
    record = _build_setup_record(metadata)

    assert build_subject_metadata_record(metadata)["subject_ids"] == [
        "4512e0a1-2d7e-496a-8eb0-8d25f7aa93d9"
    ]
    assert record["assigned_subject_count"] == 1
    assert record["subject_assignment_status"] == "explicit"


def test_setup_does_not_count_a_dish_group_id_as_an_individual() -> None:
    # A Citrus v3 dish_group subject_id names the group, not a fish.
    record = _build_setup_record(
        {"subject_count": "3", "subject_type": "dish_group", "subject_id": "g-1"}
    )

    assert record["assigned_subject_count"] is None
    assert record["subject_assignment_status"] == "count_only"


def test_setup_counts_an_untyped_single_id_for_one_subject() -> None:
    record = _build_setup_record({"subject_count": "1", "subject_id": "s-1"})

    assert record["assigned_subject_count"] == 1
    assert record["subject_assignment_status"] == "explicit"


@pytest.mark.parametrize("field", ["subject_ids", "fish_ids"])
def test_setup_counts_id_lists_unchanged(field: str) -> None:
    record = _build_setup_record(
        {"subject_count": "3", field: ["a", " b ", "a", ""]}
    )

    assert record["assigned_subject_count"] == 2
    assert record["subject_assignment_status"] == "partial"


def test_setup_refuses_subject_id_with_legacy_fish_id() -> None:
    from fisheye.shared.subject_metadata import SubjectMetadataError

    with pytest.raises(SubjectMetadataError, match="both subject_id and legacy fish_id"):
        _build_setup_record({"subject_count": "1", "subject_id": "s", "fish_id": "f"})


def test_publish_resolve_is_idempotent_and_rejects_explicit_contradiction(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    subject = publish_subject_metadata(root, _subject_metadata())
    record = build_experiment_setup_record(
        _subject_metadata(),
        subject_metadata_ref=subject.group_path,
        subject_metadata_sha256=subject.record_sha256,
    )

    first = publish_experiment_setup(root, record)
    second = publish_experiment_setup(root, record)

    assert second == first
    assert first.expected_subject_count == 1
    assert first.legacy is False
    assert root["analysis/experiment_setup_runs"].attrs["latest"] == first.run_name
    with pytest.raises(ValueError, match="contradicts experiment setup"):
        resolve_expected_subject_count(root, 2)


def test_resolver_fails_closed_on_tampered_record(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    resolved = _publish_setup(root)
    run = root[resolved.group_path]
    tampered = dict(run.attrs[EXPERIMENT_SETUP_RECORD_ATTR])
    tampered["expected_subject_count"] = 2
    run.attrs[EXPERIMENT_SETUP_RECORD_ATTR] = tampered

    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_experiment_setup(root)


def test_refinement_rejects_unbound_or_stale_quality_setup(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w", zarr_format=3)
    setup = _publish_setup(root)

    with pytest.raises(ValueError, match="not bound"):
        _validate_quality_experiment_setup_binding(
            setup,
            {"expected_subject_count": 1},
        )
    with pytest.raises(ValueError, match="stale or contradictory"):
        _validate_quality_experiment_setup_binding(
            setup,
            {
                "expected_subject_count": 1,
                "experiment_setup_sha256": "0" * 64,
            },
        )
    _validate_quality_experiment_setup_binding(
        setup,
        {
            "expected_subject_count": 1,
            "experiment_setup_sha256": setup.record_sha256,
        },
    )


def test_h5_import_publishes_setup_subject_snapshot_and_registry_membership(
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "batman_recording"
    h5_path = recording_dir / "raw" / "session.h5"
    zarr_path = recording_dir / "zarr" / "batman_recording_analysis.zarr"
    h5_path.parent.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    with h5py.File(h5_path, "w") as h5:
        subject = h5.create_group("subject_metadata")
        for key, value in _subject_metadata().items():
            subject.attrs[key] = value

    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "analysis",
            "session_uuid": "batman-session",
            "recording_id": "batman_recording",
            "recording_name": "batman_recording",
            "recording_path": str(recording_dir),
            "recording_type": "behavior",
            "recording_subtype": "free",
            "behavior_mode": "free",
        }
    )
    plan = RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=h5_path,
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=zarr_path,
    )

    summary = import_experiment_setup(plan)
    assert summary is not None
    assert summary["expected_subject_count"] == 1
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    subject_path = str(summary["subject_metadata_path"])
    subject_run = root[subject_path]
    assert subject_run.attrs["subject_metadata_record"]["subject_metadata"]["fish_id"] == (
        "40d99fea-846b-4890-bad2-b4e152dfdde0"
    )
    assert subject_run.attrs["subject_ids"] == [
        "40d99fea-846b-4890-bad2-b4e152dfdde0"
    ]
    assert subject_run.attrs["subject_identity_kind"] == "uuid"
    assert resolve_experiment_setup(root).expected_subject_count == 1

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.register_from_root(root, zarr_path)
        provenance = registry.conn.execute(
            "SELECT subject_count, fish_id FROM provenance WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        membership = registry.conn.execute(
            "SELECT subject_id, dpf_at_acquisition FROM recording_subjects "
            "WHERE recording_id = ?",
            ("batman_recording",),
        ).fetchone()
        context = registry.conn.execute(
            "SELECT subject_count_snapshot, subject_count_recorded, "
            "subject_count_effective FROM dataset_context_current "
            "WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        assert tuple(provenance) == (1, "40d99fea-846b-4890-bad2-b4e152dfdde0")
        assert tuple(membership) == (
            "40d99fea-846b-4890-bad2-b4e152dfdde0",
            7,
        )
        assert tuple(context) == (1, 1, 1)
    finally:
        registry.close()

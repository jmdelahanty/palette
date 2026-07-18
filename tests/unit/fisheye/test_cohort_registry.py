from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from fisheye.cohorts.registry import (
    CohortSelectionError,
    build_cohort_plan,
    freeze_cohort,
    validate_frozen_cohort,
)
from fisheye.cohorts.release import main as cohort_release_main
from fisheye.cohorts.spec import CohortSpec, CohortSpecError
from fisheye.registry.db import Registry


HASH_A = "a" * 64
HASH_B = "b" * 64


def _spec(**overrides: object) -> CohortSpec:
    payload: dict[str, object] = {
        "schema_id": "palette.cohort_query",
        "schema_version": 1,
        "cohort_id": "test_cohort_v1",
        "cohort_name": "Test cohort",
        "dataset": {
            "statuses": ["active"],
            "zarr_uses": ["analysis"],
            "zarr_origins": ["source"],
        },
        "protocol": {
            "stimulus_modes_any": ["CHASER"],
            "protocol_hashes_any": [HASH_A],
        },
        "subjects": {"match_policy": "unambiguous_recording"},
        "prerequisites": {"required_steps_ok": []},
        "missing_selected_metadata": "error",
    }
    payload.update(overrides)
    return CohortSpec.from_mapping(payload)


def _seed_dataset(
    registry: Registry,
    tmp_path: Path,
    *,
    dataset_id: str,
    recording_id: str,
    protocol_hash: str,
    stimulus_mode: str = "CHASER",
    dpf_values: tuple[int | None, ...] = (),
    line_strains: tuple[str, ...] = (),
    genotype_values: tuple[str, ...] = (),
    step_statuses: dict[str, str] | None = None,
) -> None:
    registry.upsert_dataset(
        dataset_id,
        session_uuid=f"session_{dataset_id}",
        zarr_path=tmp_path / f"{dataset_id}.zarr",
        recording_id=recording_id,
        artifact_kind="source_recording",
        zarr_origin="source",
        zarr_use="analysis",
    )
    registry.upsert_recording(
        recording_id=recording_id,
        session_uuid=f"session_{dataset_id}",
        recording_name=recording_id,
        recording_path=str(tmp_path / recording_id),
        started_utc=f"2026-01-{len(dataset_id):02d}T00:00:00Z",
        protocol_name="ProtocolA" if protocol_hash == HASH_A else "ProtocolB",
    )
    registry.upsert_provenance(
        dataset_id,
        provenance={},
        context={},
        protocol_name="ProtocolA" if protocol_hash == HASH_A else "ProtocolB",
        protocol_hash=protocol_hash,
        acquisition={},
        zarr_purpose="analysis",
    )
    registry.conn.execute(
        """
        INSERT OR IGNORE INTO stimulus_protocols (
            protocol_hash, protocol_name, step_count, protocol_json,
            definition_source, extracted_utc
        ) VALUES (?, ?, 1, '{}', 'test', '2026-01-01T00:00:00Z')
        """,
        (protocol_hash, "ProtocolA" if protocol_hash == HASH_A else "ProtocolB"),
    )
    run_id = f"stimulus_{dataset_id}"
    registry.conn.execute(
        """
        INSERT INTO recording_stimulus_runs (
            dataset_id, recording_id, stimulus_run_id, protocol_hash,
            protocol_name, is_latest, step_count, source_path,
            source_metadata_sha256, source_zarr_path, extracted_utc
        ) VALUES (?, ?, ?, ?, ?, 1, 1, 'stimulus', ?, ?, '2026-01-01T00:00:00Z')
        """,
        (
            dataset_id,
            recording_id,
            run_id,
            protocol_hash,
            "ProtocolA" if protocol_hash == HASH_A else "ProtocolB",
            "c" * 64,
            str(tmp_path / f"{dataset_id}.zarr"),
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_stimulus_modes (
            dataset_id, stimulus_run_id, stimulus_mode, step_count, total_duration_s
        ) VALUES (?, ?, ?, 1, 10.0)
        """,
        (dataset_id, run_id, stimulus_mode),
    )
    subject_count = max(len(dpf_values), len(line_strains), len(genotype_values), 0)
    for index in range(subject_count):
        subject_id = f"{recording_id}_subject_{index}"
        cross_id = f"{recording_id}_cross_{index}"
        strain = line_strains[index] if index < len(line_strains) else None
        genotype = genotype_values[index] if index < len(genotype_values) else None
        dpf = dpf_values[index] if index < len(dpf_values) else None
        registry.conn.execute(
            """
            INSERT INTO crosses (
                cross_id, line_strain, genotype, created_utc, updated_utc
            ) VALUES (?, ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """,
            (cross_id, strain, genotype),
        )
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, cross_id,
                dpf_at_acquisition, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
            """,
            (recording_id, subject_id, dataset_id, cross_id, dpf),
        )
    for step, status in (step_statuses or {}).items():
        registry.conn.execute(
            """
            INSERT INTO recording_step_status (
                dataset_id, recording_id, step_name, status, source, updated_utc
            ) VALUES (?, ?, ?, ?, 'test', '2026-01-01T00:00:00Z')
            """,
            (dataset_id, recording_id, step, status),
        )
    registry.conn.commit()


def test_exact_protocol_freeze_includes_every_matching_dataset(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="matching_a",
        recording_id="recording_a",
        protocol_hash=HASH_A,
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="matching_b",
        recording_id="recording_b",
        protocol_hash=HASH_A,
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="other",
        recording_id="recording_c",
        protocol_hash=HASH_B,
    )
    registry.close()

    plan = build_cohort_plan(registry_path, _spec())
    assert plan["summary"]["included_count"] == 2
    assert plan["summary"]["excluded_count"] == 1
    frozen = freeze_cohort(plan)
    assert frozen["member_count"] == 2
    assert [row["dataset_id"] for row in frozen["members"]] == [
        "matching_a",
        "matching_b",
    ]
    assert frozen["selection_policy"]["limit"] is None
    assert frozen["manifest_sha256"]
    assert validate_frozen_cohort(frozen) == []

    frozen["members"][1]["recording_id"] = frozen["members"][0]["recording_id"]
    errors = validate_frozen_cohort(frozen)
    assert any("recording_id is duplicated" in error for error in errors)
    assert "manifest_sha256 mismatch" in errors


def test_selected_biology_is_and_across_fields_and_or_within_field(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="match",
        recording_id="recording_match",
        protocol_hash=HASH_A,
        dpf_values=(8,),
        line_strains=("strain_b",),
        genotype_values=("wildtype",),
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="wrong_dpf",
        recording_id="recording_wrong",
        protocol_hash=HASH_A,
        dpf_values=(12,),
        line_strains=("strain_b",),
        genotype_values=("wildtype",),
    )
    registry.close()
    spec = _spec(
        subjects={
            "dpf": {"values": [7, 8]},
            "line_strains_any": ["strain_a", "strain_b"],
            "match_policy": "unambiguous_recording",
        }
    )

    plan = build_cohort_plan(registry_path, spec)
    assert plan["summary"]["included_count"] == 1
    assert plan["summary"]["excluded_count"] == 1
    assert next(row for row in plan["records"] if row["dataset_id"] == "wrong_dpf")[
        "exclusions"
    ] == ["dpf_mismatch"]


def test_missing_and_ambiguous_selected_biology_block_freeze(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="missing",
        recording_id="recording_missing",
        protocol_hash=HASH_A,
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="ambiguous",
        recording_id="recording_ambiguous",
        protocol_hash=HASH_A,
        dpf_values=(7, 8),
    )
    registry.close()
    spec = _spec(
        subjects={
            "dpf": {"min": 7, "max": 9},
            "match_policy": "unambiguous_recording",
        }
    )

    plan = build_cohort_plan(registry_path, spec)
    assert plan["summary"]["blocked_count"] == 2
    assert plan["summary"]["blocker_reasons"] == {
        "ambiguous_dpf_metadata": 1,
        "missing_dpf_metadata": 1,
    }
    with pytest.raises(CohortSelectionError, match="2 otherwise-matching"):
        freeze_cohort(plan)


def test_required_step_and_duplicate_recording_dataset_are_blockers(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="duplicate_a",
        recording_id="same_recording",
        protocol_hash=HASH_A,
        step_statuses={"tracks": "ok"},
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="duplicate_b",
        recording_id="same_recording",
        protocol_hash=HASH_A,
        step_statuses={"tracks": "missing"},
    )
    registry.close()
    spec = _spec(prerequisites={"required_steps_ok": ["tracks"]})

    plan = build_cohort_plan(registry_path, spec)
    assert plan["summary"]["blocked_count"] == 2
    assert (
        plan["summary"]["blocker_reasons"]["multiple_candidate_datasets_for_recording"]
        == 2
    )
    assert (
        plan["summary"]["blocker_reasons"]["required_step_not_ok:tracks:missing"] == 1
    )


def test_multi_subject_completeness_is_explicit_for_any_and_all_subjects(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="partial_dpf",
        recording_id="partial_recording",
        protocol_hash=HASH_A,
        dpf_values=(7, None),
    )
    registry.close()

    selected_dpf = {"dpf": {"values": [7]}}
    unambiguous = build_cohort_plan(
        registry_path,
        _spec(
            subjects={
                **selected_dpf,
                "match_policy": "unambiguous_recording",
            }
        ),
    )
    assert unambiguous["summary"]["blocker_reasons"] == {"incomplete_dpf_metadata": 1}

    any_subject = build_cohort_plan(
        registry_path,
        _spec(subjects={**selected_dpf, "match_policy": "any_subject"}),
    )
    assert any_subject["summary"]["included_count"] == 1

    all_subjects = build_cohort_plan(
        registry_path,
        _spec(subjects={**selected_dpf, "match_policy": "all_subjects"}),
    )
    assert all_subjects["summary"]["blocker_reasons"] == {"incomplete_dpf_metadata": 1}


def test_multiple_latest_stimulus_runs_block_an_otherwise_matching_recording(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="ambiguous_stimulus",
        recording_id="ambiguous_stimulus_recording",
        protocol_hash=HASH_A,
    )
    registry.conn.execute(
        """
        INSERT INTO recording_stimulus_runs (
            dataset_id, recording_id, stimulus_run_id, protocol_hash,
            protocol_name, is_latest, step_count, source_path,
            source_metadata_sha256, source_zarr_path, extracted_utc
        ) VALUES (?, ?, 'second_latest', ?, 'ProtocolA', 1, 1, 'stimulus', ?, ?,
                  '2026-01-01T00:00:00Z')
        """,
        (
            "ambiguous_stimulus",
            "ambiguous_stimulus_recording",
            HASH_A,
            "d" * 64,
            str(tmp_path / "ambiguous_stimulus.zarr"),
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_stimulus_modes (
            dataset_id, stimulus_run_id, stimulus_mode, step_count, total_duration_s
        ) VALUES ('ambiguous_stimulus', 'second_latest', 'CHASER', 1, 10.0)
        """
    )
    registry.conn.commit()
    registry.close()

    plan = build_cohort_plan(registry_path, _spec())
    assert plan["summary"]["blocker_reasons"] == {"multiple_latest_stimulus_runs": 1}


def test_spec_rejects_untyped_protocol_hash_and_unknown_match_policy() -> None:
    with pytest.raises(CohortSpecError, match="64-character"):
        _spec(protocol={"protocol_hashes_any": ["not-a-hash"]})
    with pytest.raises(CohortSpecError, match="match_policy"):
        _spec(subjects={"match_policy": "guess"})
    with pytest.raises(CohortSpecError, match="unknown field"):
        _spec(protocol={"protocol_hash_any": [HASH_A]})


def _build_clean_palette_checkout(path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    scripts_dir = path / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(repo / "scripts" / "py", scripts_dir / "py")
    (path / "src").symlink_to(repo / "src", target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Palette Tests",
            "-c",
            "user.email=palette-tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )


def test_release_render_freezes_membership_and_wires_dependency_dag(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="release_a",
        recording_id="release_recording_a",
        protocol_hash=HASH_A,
    )
    _seed_dataset(
        registry,
        tmp_path,
        dataset_id="release_b",
        recording_id="release_recording_b",
        protocol_hash=HASH_A,
    )
    registry.close()
    spec_path = tmp_path / "cohort.json"
    spec_path.write_text(json.dumps(_spec().to_mapping()), encoding="utf-8")
    cluster_repo = tmp_path / "cluster-palette"
    _build_clean_palette_checkout(cluster_repo)
    output_root = tmp_path / "analytics"

    result = cohort_release_main(
        [
            "--spec",
            str(spec_path),
            "--release-id",
            "release_render_v1",
            "--registry",
            str(registry_path),
            "--output-root",
            str(output_root),
            "--palette-repo",
            str(cluster_repo),
            "--skip-report",
        ]
    )

    assert result == 0
    release_dir = output_root / "logs" / "releases" / "release_render_v1"
    frozen = json.loads(
        (release_dir / "frozen_cohort_manifest.json").read_text(encoding="utf-8")
    )
    assert frozen["member_count"] == 2
    submission = json.loads(
        (release_dir / "release_submission.json").read_text(encoding="utf-8")
    )
    assert submission["status"] == "rendered"
    assert [stage["name"] for stage in submission["stages"]] == [
        "recording_analytics",
        "collection_binding",
        "analytics_export_and_statistics",
    ]
    assert [stage["job_id"] for stage in submission["stages"]] == [
        "900001",
        "900002",
        "900003",
    ]
    collection_job = (
        output_root
        / "logs"
        / "lsf"
        / "collection_manifest_release_render_v1"
        / "run_collection_manifest.sh"
    )
    assert collection_job.is_file()
    export_submission = next(
        stage
        for stage in submission["stages"]
        if stage["name"] == "analytics_export_and_statistics"
    )
    assert export_submission["command"][-2:] == [
        "--dependency-done",
        "900002",
    ]

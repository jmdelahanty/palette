from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.utils import apply_recording_subject_trait_allocation as apply_traits


def test_batman_keypoint_allocation_is_recording_grouped_and_camera_complete() -> None:
    repository = Path(__file__).resolve().parents[3]
    path = (
        repository
        / "docs"
        / "diagnostics"
        / "batman_keypoint_recording_allocation_20260807.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    groups = payload["recording_groups"]
    assert len(groups) == 9
    assert payload["role_counts"] == {
        "train": 28,
        "development": 4,
        "sealed_test": 4,
    }

    expected_cameras = {"1", "2", "3", "4"}
    recording_ids: list[str] = []
    counts = {"train": 0, "development": 0, "sealed_test": 0}
    for group in groups:
        group_recordings = group["recording_ids"]
        assert len(group_recordings) == 4
        observed_cameras = {
            recording_id.split("_arena_", 1)[1].split("_", 1)[0]
            for recording_id in group_recordings
        }
        assert observed_cameras == expected_cameras
        counts[group["role"]] += len(group_recordings)
        recording_ids.extend(group_recordings)

    assert counts == payload["role_counts"]
    assert len(recording_ids) == 36
    assert len(set(recording_ids)) == 36
    assert (
        "2026-07-21T19-38-32Z_arena_2_Batman"
        in payload["existing_reviewed_training_recordings"]
    )
    assert payload["population_snapshot"]["pigmentation_phenotype_counts"] == {
        "wild_type_pigmented": 36
    }
    assert payload["population_snapshot"]["canonical_strain_counts"] == {"AB": 36}
    assert payload["population_snapshot"]["phenotype_coverage_complete"] is True
    assert payload["population_snapshot"]["dpf_counts"] == {"7": 28, "8": 8}

    table_path = path.with_name(payload["recording_table"])
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    assert {row["recording_id"] for row in rows} == set(recording_ids)
    assert len({row["dataset_id"] for row in rows}) == 36
    assert len({row["subject_id"] for row in rows}) == 36
    assert Counter(row["role"] for row in rows) == payload["role_counts"]
    assert Counter(row["camera_id"] for row in rows) == {
        "2010093": 9,
        "2010094": 9,
        "2010095": 9,
        "2010096": 9,
    }
    assert {row["species"] for row in rows} == {"Danio rerio"}
    assert {row["canonical_strain"] for row in rows} == {"AB"}
    assert {row["pigmentation_phenotype"] for row in rows} == {
        "wild_type_pigmented"
    }
    assert {row["melanophore_status"] for row in rows} == {"normal"}
    assert {row["xanthophore_status"] for row in rows} == {"normal"}
    assert {row["iridophore_status"] for row in rows} == {"normal"}
    assert {row["pigment_pattern_status"] for row in rows} == {"wild_type"}
    assert {row["optical_transparency"] for row in rows} == {"normal"}
    assert {row["pigmentation_value_origin"] for row in rows} == {
        "subject_observed"
    }
    assert Counter(row["dpf"] for row in rows) == {"7": 28, "8": 8}


def test_batman_trait_allocation_applies_atomically_and_idempotently(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    allocation_path = (
        repository
        / "docs"
        / "diagnostics"
        / "batman_keypoint_recording_allocation_20260807.json"
    )
    payload = json.loads(allocation_path.read_text(encoding="utf-8"))
    table_path = allocation_path.with_name(payload["recording_table"])
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, line_strain, genotype)
        VALUES ('batman_cross', 'AB [AB IC] SEPT25', 'AB [AB IC] SEPT25');
        """
    )
    for row in rows:
        registry.conn.execute(
            "INSERT INTO datasets (dataset_id, zarr_path) VALUES (?, ?);",
            (row["dataset_id"], f"/tmp/{row['dataset_id']}.zarr"),
        )
        registry.conn.execute(
            "INSERT INTO recordings (recording_id) VALUES (?);",
            (row["recording_id"],),
        )
        registry.conn.execute(
            "INSERT INTO subjects (subject_id, species) VALUES (?, ?);",
            (row["subject_id"], row["species"]),
        )
        registry.conn.execute(
            """
            INSERT INTO recording_subjects (
                recording_id, subject_id, dataset_id, cross_id,
                dpf_at_acquisition, species, genotype, line_strain
            ) VALUES (?, ?, ?, 'batman_cross', ?, ?, ?, ?);
            """,
            (
                row["recording_id"],
                row["subject_id"],
                row["dataset_id"],
                int(row["dpf"]),
                row["species"],
                row["genotype"],
                row["line_strain"],
            ),
        )
    registry.conn.commit()
    registry.close()

    assert (
        apply_traits.main(
            [
                "--registry",
                str(registry_path),
                "--allocation",
                str(allocation_path),
            ]
        )
        == 0
    )
    for _ in range(2):
        assert (
            apply_traits.main(
                [
                    "--registry",
                    str(registry_path),
                    "--allocation",
                    str(allocation_path),
                    "--apply",
                ]
            )
            == 0
        )

    registry = Registry(registry_path)
    assert registry.conn.execute(
        "SELECT COUNT(*) AS n FROM strain_label_mappings;"
    ).fetchone()["n"] == 1
    assert registry.conn.execute(
        "SELECT COUNT(*) AS n FROM strain_trait_expectations;"
    ).fetchone()["n"] == 6
    assert registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_subject_traits;"
    ).fetchone()["n"] == 216
    assert registry.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM recording_subject_trait_resolved
        WHERE value_origin = 'subject_observed';
        """
    ).fetchone()["n"] == 216
    registry.close()

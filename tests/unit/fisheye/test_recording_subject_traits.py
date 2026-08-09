from __future__ import annotations

import json

import pytest

from fisheye.registry.db import (
    PIGMENTATION_PHENOTYPE_VOCABULARY_ID,
    Registry,
)


def _seed_recording_subject(registry: Registry) -> None:
    registry.conn.execute(
        """
        INSERT INTO datasets (dataset_id, zarr_path)
        VALUES ('dataset_1', '/tmp/dataset_1.zarr');
        """
    )
    registry.conn.execute(
        "INSERT INTO recordings (recording_id) VALUES ('recording_1');"
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, line_strain, genotype)
        VALUES ('cross_1', 'AB [AB IC] SEPT25', 'AB [AB IC] SEPT25');
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, species)
        VALUES ('dish_1', 'cross_1', 'Danio rerio');
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, species)
        VALUES ('subject_1', 'dish_1', 'Danio rerio');
        """
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id,
            species, genotype, line_strain
        ) VALUES (
            'recording_1', 'subject_1', 'dataset_1', 'dish_1', 'cross_1',
            'Danio rerio', 'AB [AB IC] SEPT25', 'AB [AB IC] SEPT25'
        );
        """
    )
    registry.conn.commit()


def test_recording_subject_trait_migration_and_upsert(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    _seed_recording_subject(registry)

    registry.upsert_recording_subject_trait(
        recording_id="recording_1",
        subject_id="subject_1",
        trait_name="pigmentation_phenotype",
        trait_value="hypopigmented",
        assignment_method="manual_review",
        assigned_by="reviewer",
        assigned_at_utc="2026-08-07T12:00:00+00:00",
        evidence={"frame_index": 100},
    )

    row = registry.conn.execute(
        "SELECT * FROM recording_subject_trait_overview;"
    ).fetchone()
    assert row is not None
    assert row["dataset_id"] == "dataset_1"
    assert row["trait_value"] == "hypopigmented"
    assert row["vocabulary_id"] == PIGMENTATION_PHENOTYPE_VOCABULARY_ID
    assert json.loads(row["evidence_json"]) == {"frame_index": 100}

    registry.upsert_recording_subject_trait(
        recording_id="recording_1",
        subject_id="subject_1",
        trait_name="pigmentation_phenotype",
        trait_value="wild_type_pigmented",
        assignment_method="manual_review",
        assigned_by="reviewer",
    )
    rows = registry.conn.execute(
        "SELECT trait_value FROM recording_subject_traits;"
    ).fetchall()
    assert [row["trait_value"] for row in rows] == ["wild_type_pigmented"]
    registry.close()


def test_existing_version_63_registry_upgrades_to_strain_traits(tmp_path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.conn.execute("DROP VIEW recording_subject_trait_resolved;")
    registry.conn.execute("DROP TABLE strain_trait_expectations;")
    registry.conn.execute("DROP TABLE strain_label_mappings;")
    registry.conn.execute("DELETE FROM schema_version WHERE version >= 64;")
    registry.conn.execute("PRAGMA user_version = 63;")
    registry.conn.commit()
    registry.close()

    upgraded = Registry(registry_path)
    version = upgraded.conn.execute(
        "SELECT MAX(version) AS version FROM schema_version;"
    ).fetchone()["version"]
    assert version == 66
    objects = {
        row["name"]
        for row in upgraded.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE name IN (
                'strain_label_mappings',
                'strain_trait_expectations',
                'recording_subject_trait_resolved'
            );
            """
        ).fetchall()
    }
    assert objects == {
        "strain_label_mappings",
        "strain_trait_expectations",
        "recording_subject_trait_resolved",
    }
    upgraded.close()


def test_legacy_bootstrap_version_64_reconciles_missing_subject_trait_table(
    tmp_path,
) -> None:
    registry_path = tmp_path / "legacy_bootstrap.sqlite"
    registry = Registry(registry_path)
    registry.conn.execute("DROP VIEW recording_subject_trait_resolved;")
    registry.conn.execute("DROP VIEW recording_subject_trait_overview;")
    registry.conn.execute("DROP TABLE recording_subject_traits;")
    registry.conn.execute("DELETE FROM schema_version WHERE version >= 65;")
    registry.conn.execute("PRAGMA user_version = 64;")
    registry.conn.commit()
    registry.close()

    reconciled = Registry(registry_path)
    version = reconciled.conn.execute(
        "SELECT MAX(version) AS version FROM schema_version;"
    ).fetchone()["version"]
    assert version == 66
    objects = {
        row["name"]
        for row in reconciled.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE name IN (
                'recording_subject_traits',
                'recording_subject_trait_overview',
                'recording_subject_trait_resolved'
            );
            """
        ).fetchall()
    }
    assert objects == {
        "recording_subject_traits",
        "recording_subject_trait_overview",
        "recording_subject_trait_resolved",
    }
    reconciled.close()


def test_strain_expectations_resolve_and_subject_observations_override(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    _seed_recording_subject(registry)
    registry.upsert_strain_label_mapping(
        species="Danio rerio",
        source_label="AB [AB IC] SEPT25",
        canonical_strain="AB",
        assignment_method="operator_confirmed_mapping",
        evidence={"source_label_semantics": "suffix_uninterpreted"},
    )
    registry.upsert_strain_trait_expectation(
        species="Danio rerio",
        canonical_strain="AB",
        trait_name="pigmentation_phenotype",
        trait_value="wild_type_pigmented",
        assignment_method="curated_strain_reference",
    )
    registry.upsert_strain_trait_expectation(
        species="Danio rerio",
        canonical_strain="AB",
        trait_name="melanophore_status",
        trait_value="normal",
        assignment_method="curated_strain_reference",
    )

    rows = registry.conn.execute(
        """
        SELECT trait_name, trait_value, value_origin, canonical_strain
        FROM recording_subject_trait_resolved
        ORDER BY trait_name;
        """
    ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "trait_name": "melanophore_status",
            "trait_value": "normal",
            "value_origin": "strain_expected",
            "canonical_strain": "AB",
        },
        {
            "trait_name": "pigmentation_phenotype",
            "trait_value": "wild_type_pigmented",
            "value_origin": "strain_expected",
            "canonical_strain": "AB",
        },
    ]

    registry.upsert_recording_subject_trait(
        recording_id="recording_1",
        subject_id="subject_1",
        trait_name="pigmentation_phenotype",
        trait_value="hypopigmented",
        assignment_method="manual_visual_review",
    )
    resolved = registry.conn.execute(
        """
        SELECT trait_value, value_origin
        FROM recording_subject_trait_resolved
        WHERE trait_name = 'pigmentation_phenotype';
        """
    ).fetchone()
    assert dict(resolved) == {
        "trait_value": "hypopigmented",
        "value_origin": "subject_observed",
    }
    registry.close()


@pytest.mark.parametrize(
    ("trait_name", "trait_value"),
    [
        ("melanophore_status", "normal"),
        ("xanthophore_status", "normal"),
        ("iridophore_status", "normal"),
        ("pigment_pattern_status", "wild_type"),
        ("optical_transparency", "normal"),
    ],
)
def test_independent_pigmentation_traits_use_controlled_vocabularies(
    tmp_path,
    trait_name: str,
    trait_value: str,
) -> None:
    registry = Registry(tmp_path / f"{trait_name}.sqlite")
    _seed_recording_subject(registry)
    registry.upsert_recording_subject_trait(
        recording_id="recording_1",
        subject_id="subject_1",
        trait_name=trait_name,
        trait_value=trait_value,
        assignment_method="manual_visual_review",
    )
    row = registry.conn.execute(
        "SELECT vocabulary_id FROM recording_subject_traits;"
    ).fetchone()
    assert row["vocabulary_id"].startswith("palette.")
    registry.close()


def test_recording_subject_trait_rejects_invalid_controlled_value(tmp_path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    _seed_recording_subject(registry)

    with pytest.raises(ValueError, match="Unsupported pigmentation_phenotype"):
        registry.upsert_recording_subject_trait(
            recording_id="recording_1",
            subject_id="subject_1",
            trait_name="pigmentation_phenotype",
            trait_value="probably_clear",
            assignment_method="manual_review",
        )
    registry.close()

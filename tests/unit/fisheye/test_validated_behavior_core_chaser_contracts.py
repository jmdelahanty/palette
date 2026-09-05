from __future__ import annotations

from pathlib import Path

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_core_chaser_contracts import (
    CHASER_EXTENSION_OMITTED_TABLES,
    CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
    CORE_CHASER_EXTENSION_TABLE_SPECS,
    CORE_CHASER_EXPORT_PROFILE_ID,
    CORE_CHASER_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_contracts import CORE_TABLE_NAMES
from fisheye.analytics_exports.validated_behavior_cohort import (
    validated_behavior_manifest_path,
)
import fisheye.analytics_exports.validated_behavior_core_chaser_adapters as adapters
from fisheye.analytics_exports.validated_behavior_profiles import (
    resolve_validated_behavior_profile,
)
from fisheye.analytics_exports.validated_behavior_phase_c_contracts import (
    PHASE_C_TABLE_SPECS,
)


def _foreign_key_targets(table_name: str) -> set[str]:
    return {
        target
        for _local, target, _target_fields in CORE_CHASER_TABLE_SPECS[
            table_name
        ].foreign_keys
    }


def test_composite_roster_subtracts_competing_core_like_tables_before_composition() -> (
    None
):
    assert set(CORE_BEHAVIOR_TABLE_SPECS).issubset(CORE_CHASER_TABLE_SPECS)
    assert set(CORE_CHASER_EXTENSION_TABLE_SPECS) == (
        set(PHASE_C_TABLE_SPECS) - CHASER_EXTENSION_OMITTED_TABLES
    )
    assert "provider_motion_samples" not in CORE_CHASER_TABLE_SPECS
    assert "bout_detector_signal_samples" not in CORE_CHASER_TABLE_SPECS
    assert "body_frame_samples" not in CORE_CHASER_TABLE_SPECS
    assert list(CORE_CHASER_TABLE_SPECS).count("canonical_swim_bouts") == 1


def test_composite_is_an_installed_profile_on_the_existing_export_surface() -> None:
    profile = resolve_validated_behavior_profile(CORE_CHASER_EXPORT_PROFILE_ID)
    scientific_tables = set(CORE_CHASER_TABLE_SPECS).difference(CORE_TABLE_NAMES)

    assert profile.table_specs is CORE_CHASER_TABLE_SPECS
    assert len(profile.table_specs) == 30
    assert set(profile.row_extractors()) == scientific_tables
    assert len(scientific_tables) == 27
    assert validated_behavior_manifest_path(Path("/tmp/export"), "composite") == (
        Path("/tmp/export")
        / "validated_behavior"
        / "v1"
        / "manifests"
        / "export_run_id=composite.json"
    )


def test_paradigm_sample_relations_target_selected_core_rows() -> None:
    assert (
        CORE_CHASER_TABLE_SPECS["chaser_relative_samples"].required_capability
        == CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY
    )
    assert "kinematics_samples" in _foreign_key_targets("chaser_relative_samples")
    assert "subject_body_frame_samples" in _foreign_key_targets("body_relative_samples")
    assert "body_frame_samples" not in _foreign_key_targets("body_relative_samples")

    chaser_fk = next(
        foreign_key
        for foreign_key in CORE_CHASER_TABLE_SPECS[
            "chaser_relative_samples"
        ].foreign_keys
        if foreign_key[1] == "kinematics_samples"
    )
    assert chaser_fk == (
        ("export_run_id", "recording_id", "track_sample_id"),
        "kinematics_samples",
        ("export_run_id", "recording_id", "track_sample_index"),
    )

    body_fk = next(
        foreign_key
        for foreign_key in CORE_CHASER_TABLE_SPECS["body_relative_samples"].foreign_keys
        if foreign_key[1] == "subject_body_frame_samples"
    )
    assert body_fk == (
        (
            "export_run_id",
            "recording_id",
            "core_subject_shape_row_index",
        ),
        "subject_body_frame_samples",
        ("export_run_id", "recording_id", "subject_shape_row_index"),
    )
    fields = {
        field.name: field
        for field in CORE_CHASER_TABLE_SPECS["body_relative_samples"].contract.fields
    }
    assert fields["core_subject_shape_row_index"].nullable is True


def test_body_relation_projects_only_valid_rows_into_the_core_foreign_key() -> None:
    projected = adapters._core_body_join_columns(  # noqa: SLF001
        {
            "body_source_row_id": [7, -1, 9],
            "body_source_row_valid": [True, False, True],
        }
    )

    assert projected["core_subject_shape_row_index"] == [7, None, 9]


def test_body_relation_rejects_noncanonical_missing_row_sentinel() -> None:
    try:
        adapters._core_body_join_columns(  # noqa: SLF001
            {
                "body_source_row_id": [7],
                "body_source_row_valid": [False],
            }
        )
    except adapters.CoreChaserExportAdapterError as exc:
        assert "exact -1 sentinel" in str(exc)
    else:  # pragma: no cover - makes the hostile expectation explicit
        raise AssertionError("invalid body source-row evidence was accepted")

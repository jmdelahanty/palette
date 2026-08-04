from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.analysis import track_kinematics as writer
from fisheye.analysis.track_kinematics_schema import (
    TRACK_KINEMATICS_ARENA_BUNDLE,
    TRACK_KINEMATICS_CORE_BUNDLE,
    TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS,
    TRACK_KINEMATICS_LEGACY_EXCLUDED_RUN_ARRAYS,
    TRACK_KINEMATICS_PHYSICAL_BUNDLE,
    TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS,
    TRACK_KINEMATICS_RUN_DECLARATIONS,
    TrackKinematicsStructuredDTypeBlockedError,
    build_track_kinematics_track_declarations,
    declaration_paths,
)
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_INTERPOLATION_DTYPE,
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode


def _by_path(declarations):
    return {declaration.relative_path: declaration for declaration in declarations}


def test_closed_inventory_exactly_matches_maintained_writer_vocabulary() -> None:
    core = declaration_paths(TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS)
    physical = declaration_paths(TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS)

    assert len(core) == 69
    assert len(physical) == 35
    assert not (core & physical)
    assert core == writer._expected_motion_track_surface_paths(include_physical=False)
    assert core | physical == writer._expected_motion_track_surface_paths(
        include_physical=True
    )

    assert len(build_track_kinematics_track_declarations(include_physical=False)) == 69
    assert len(build_track_kinematics_track_declarations(include_physical=True)) == 104


def test_optional_bundles_are_closed_and_legacy_auxiliaries_are_excluded() -> None:
    assert {
        declaration.bundle for declaration in TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
    } == {TRACK_KINEMATICS_CORE_BUNDLE}
    assert all(
        declaration.required for declaration in TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
    )
    assert {
        declaration.bundle
        for declaration in TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
    } == {TRACK_KINEMATICS_PHYSICAL_BUNDLE}
    assert all(
        declaration.required is False
        for declaration in TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
    )

    run = _by_path(TRACK_KINEMATICS_RUN_DECLARATIONS)
    assert set(run) == {"track_ids", "track_arena_ids"}
    assert run["track_ids"].required is True
    assert run["track_arena_ids"].required is False
    assert run["track_arena_ids"].bundle == TRACK_KINEMATICS_ARENA_BUNDLE
    assert run["track_arena_ids"].fill_semantics == "minus_one"

    declared = declaration_paths(TRACK_KINEMATICS_RUN_DECLARATIONS)
    assert not declared.intersection(TRACK_KINEMATICS_LEGACY_EXCLUDED_RUN_ARRAYS)
    assert all(
        not declaration.relative_path.startswith("swim_bouts/")
        for declaration in TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
        + TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
    )


def test_exact_current_dtypes_shapes_access_and_fill_semantics() -> None:
    all_track = _by_path(
        TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
        + TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
    )

    assert all_track["positions_px"].dtype == np.dtype("float32")
    assert all_track["positions_mm"].dtype == np.dtype("float32")
    assert all_track["positions_px"].shape_template == ("n_track_samples", 2)
    assert all_track["positions_px"].coordinate_space == "source_camera_pixels"
    assert all_track["positions_px"].fill_semantics == "nan"

    assert all_track["track_sample_key"].dtype == np.dtype("int64")
    assert all_track["track_sample_key"].shape_template == (
        "n_track_samples",
        2,
    )
    assert all_track["track_sample_key"].authority_role is (
        AnalysisAuthorityRole.LINEAGE_INDEX
    )

    assert all_track["sample_valid"].dtype == np.dtype(bool)
    assert all_track["sample_valid"].fill_semantics == "false"
    assert all_track["sample_reason_code"].dtype == np.dtype("int16")
    assert all_track["delta_frames"].dtype == np.dtype("int32")
    assert all_track["detection_source"].dtype == np.dtype("int8")
    assert all_track["speed_smoothed_px"].dtype == np.dtype("float32")
    assert all_track["speed_smoothed_px"].fill_semantics == "nan"

    assert all_track["second_indices"].access_pattern is AccessPattern.EAGER
    assert all_track["speed_per_second_px"].shape_template == ("n_track_seconds",)
    assert all_track["speed_per_second_px"].access_pattern is AccessPattern.EAGER
    assert all_track["heading_per_second_resultant"].fill_semantics == "zero"
    assert all_track["heading_degrees"].access_pattern is AccessPattern.WINDOWED

    assert all_track["frame_indices"].authority_role is (
        AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    )
    assert all_track["speed_raw_px"].authority_role is (
        AnalysisAuthorityRole.COMPATIBILITY_ALIAS
    )
    assert all_track["movement/speed/raw/px"].authority_role is (
        AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY
    )


def test_structured_lineage_layout_is_exact_and_fails_shared_conversion() -> None:
    core = _by_path(TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS)
    interpolation = core["source_frame_interpolation"]
    instance = core["source_instance_key"]

    assert interpolation.dtype == TRACK_SAMPLE_INTERPOLATION_DTYPE
    assert interpolation.dtype.itemsize == 24
    assert interpolation.structured_fields == (
        ("left_source_frame_index", "int64", 0),
        ("right_source_frame_index", "int64", 8),
        ("right_weight", "float64", 16),
    )
    assert instance.dtype == TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
    assert instance.dtype.itemsize == 9
    assert instance.structured_fields == (
        ("valid", "bool", 0),
        ("instance_key", "uint64", 1),
    )
    assert instance.fill_semantics == (
        "structured_null_record_valid_false_instance_key_zero"
    )

    with pytest.raises(
        TrackKinematicsStructuredDTypeBlockedError,
        match="current shared DTypeContract/StoragePlan/array-factory boundary",
    ):
        interpolation.to_analysis_array_declaration()
    with pytest.raises(TrackKinematicsStructuredDTypeBlockedError):
        instance.to_analysis_array_declaration()


def test_every_simple_declaration_converts_without_claiming_adoption() -> None:
    declarations = (
        TRACK_KINEMATICS_RUN_DECLARATIONS
        + TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
        + TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
    )
    converted: list[AnalysisArrayDeclaration] = []
    blocked: list[str] = []
    for declaration in declarations:
        try:
            converted.append(declaration.to_analysis_array_declaration())
        except TrackKinematicsStructuredDTypeBlockedError:
            blocked.append(declaration.relative_path)

    assert blocked == ["source_frame_interpolation", "source_instance_key"]
    assert len(converted) == 104
    assert all(item.write_mode is WriteMode.IMMUTABLE for item in converted)
    assert all(item.byte_planner_adopted is False for item in converted)
    assert all(
        item.physical_policy_owner == "track_kinematics_rechunk_v3"
        for item in converted
    )


def test_bound_paths_and_manifests_are_exact_strict_json() -> None:
    bound = TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS[0].bind_track(-7)
    assert bound.scope == "run"
    assert bound.relative_path.startswith("tracks/id_-7/")

    manifests = [
        declaration.as_manifest()
        for declaration in (
            TRACK_KINEMATICS_RUN_DECLARATIONS
            + TRACK_KINEMATICS_CORE_TRACK_DECLARATIONS
            + TRACK_KINEMATICS_PHYSICAL_TRACK_DECLARATIONS
        )
    ]
    encoded = json.dumps(manifests, allow_nan=False, sort_keys=True)
    assert "numpy_structured_v1" in encoded
    assert '"byte_planner_adopted": false' in encoded

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fisheye.analysis_workflows.provider_spatial_grid_policy import (
    BIN_WIDTH_RULE_ID,
    EDGE_POLICY_ID,
    GEOMETRY_COORDINATE_SPACE_ID,
    GRID_COORDINATE_SPACE_ID,
    GOODBATBADBAT_BIN_WIDTH_MM,
    REJECTED_DETECTION_GATE_BOUNDARY_ROLE,
    REVIEWED_TOP_RIM_BOUNDARY_ROLE,
    CircularArenaGeometryAuthority,
    PhysicalScaleAuthority,
    SelectionAuthority,
    ArenaMMGridPolicyError,
    build_arena_mm_grid_policy,
    validate_source_binding_authority_record,
)


CANARY_RADIUS_PX = 2152.594087583115
CANARY_MM_PER_PIXEL = 0.019016605362130807
RECORDING_ID = "2026-08-10T17-20-55Z_arena_2_goodbatbadbat"


def _selection() -> SelectionAuthority:
    source = {
        "schema_id": "palette.arena_geometry_selection_record",
        "schema_version": 2,
        "selection_id": "arena_geometry_selection_06b5cd2c35c04917004e",
        "recording_id": RECORDING_ID,
        "selected_candidate_id": "palette_fit_reviewed_v1",
    }
    digest = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SelectionAuthority(
        selection_id=source["selection_id"],
        recording_id=RECORDING_ID,
        record_sha256=digest,
        record_ref="/analysis/arena_geometry_selection/arena_geometry_selection_06b5cd2c35c04917004e",
    )


def _geometry(**overrides: object) -> CircularArenaGeometryAuthority:
    values: dict[str, object] = {
        "geometry_id": "arena_geometry_selection_06b5cd2c35c04917004e",
        "coordinate_authority_id": "47758cca2a336a848300b92ebc77d953e74d417b0634915ada7421b63a401d69",
        "center_x_px": 2286.7729648010045,
        "center_y_px": 2307.6434917690376,
        "radius_px": CANARY_RADIUS_PX,
        "record_ref": "/analysis/arena_geometry_selection/selected_candidate",
        "boundary_role": REVIEWED_TOP_RIM_BOUNDARY_ROLE,
        "observed_feature": REVIEWED_TOP_RIM_BOUNDARY_ROLE,
    }
    values.update(overrides)
    return CircularArenaGeometryAuthority(**values)


def _scale(**overrides: object) -> PhysicalScaleAuthority:
    values: dict[str, object] = {
        "scale_id": "source_camera_physical_scale_camera_2_v1",
        "coordinate_authority_id": "47758cca2a336a848300b92ebc77d953e74d417b0634915ada7421b63a401d69",
        "mm_per_pixel": CANARY_MM_PER_PIXEL,
        "record_ref": "/analysis/calibration/coordinate_frames/source_camera_physical",
    }
    values.update(overrides)
    return PhysicalScaleAuthority(**values)


def _policy(**overrides: object):
    values: dict[str, object] = {
        "recording_id": RECORDING_ID,
        "geometry": _geometry(),
        "scale": _scale(),
        "selection": _selection(),
    }
    values.update(overrides)
    return build_arena_mm_grid_policy(**values)


def test_canary_grid_uses_declared_radius_and_covers_physical_rim() -> None:
    policy = _policy()

    assert policy.rim_radius_mm == pytest.approx(40.93503226842414)
    assert policy.extent_radius_mm == pytest.approx(41.0)
    assert policy.x_edges.dtype == np.float64
    assert policy.y_edges.dtype == np.float64
    assert policy.x_edges[0] == -41.0
    assert policy.x_edges[-1] == 41.0
    assert policy.x_edges.size == 83
    assert policy.x_edges[0] <= -policy.rim_radius_mm
    assert policy.x_edges[-1] >= policy.rim_radius_mm
    np.testing.assert_array_equal(policy.x_edges, policy.y_edges)
    np.testing.assert_array_equal(policy.x_edges, -policy.x_edges[::-1])
    np.testing.assert_allclose(np.diff(policy.x_edges), GOODBATBADBAT_BIN_WIDTH_MM)

    grid = policy.to_occupancy_grid()
    np.testing.assert_array_equal(grid.x_edges, policy.x_edges)
    np.testing.assert_array_equal(grid.y_edges, policy.y_edges)
    assert grid.edge_policy_id == EDGE_POLICY_ID


def test_grid_policy_is_deterministic_and_round_trips_its_digest() -> None:
    first = _policy()
    second = _policy()

    assert first.record_sha256 == second.record_sha256
    assert first.source_binding_authority_record() == second.source_binding_authority_record()
    restored = type(first).from_record(first.as_record())
    assert restored.record_sha256 == first.record_sha256
    np.testing.assert_array_equal(restored.x_edges, first.x_edges)
    assert restored.to_occupancy_grid().bin_shape == (82, 82)


def test_source_binding_is_exact_and_digest_bound() -> None:
    binding = _policy().source_binding_authority_record()
    validate_source_binding_authority_record(binding)
    assert binding["schema_version"] == 2
    assert binding["grid_coordinate_space"] == GRID_COORDINATE_SPACE_ID
    assert binding["bin_width_rule_id"] == BIN_WIDTH_RULE_ID
    assert binding["x_edges_sha256"] == binding["y_edges_sha256"]
    tampered = dict(binding)
    tampered["bin_width_mm"] = 2.0
    with pytest.raises(ArenaMMGridPolicyError, match="stale record_sha256"):
        validate_source_binding_authority_record(tampered)

    legacy = dict(binding)
    legacy.pop("x_edges_sha256")
    legacy.pop("y_edges_sha256")
    legacy["schema_version"] = 1
    legacy.pop("record_sha256")
    legacy["record_sha256"] = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    validate_source_binding_authority_record(legacy)


def test_reviewed_palette_top_rim_semantics_are_preserved() -> None:
    geometry = _geometry(
        boundary_role=REVIEWED_TOP_RIM_BOUNDARY_ROLE,
        observed_feature=REVIEWED_TOP_RIM_BOUNDARY_ROLE,
    )
    policy = _policy(geometry=geometry)

    assert policy.geometry.boundary_role == REVIEWED_TOP_RIM_BOUNDARY_ROLE
    assert policy.geometry.observed_feature == REVIEWED_TOP_RIM_BOUNDARY_ROLE
    assert policy.as_record()["geometry"]["boundary_role"] == REVIEWED_TOP_RIM_BOUNDARY_ROLE
    binding = policy.source_binding_authority_record()
    assert binding["geometry"]["observed_feature"] == REVIEWED_TOP_RIM_BOUNDARY_ROLE
    assert binding["geometry"]["boundary_role"] == REVIEWED_TOP_RIM_BOUNDARY_ROLE


def test_physical_inner_rim_remains_a_distinct_accepted_boundary_role() -> None:
    geometry = _geometry(
        boundary_role="physical_inner_rim",
        observed_feature="dish_inner_rim_water_side_edge",
    )
    assert _policy(geometry=geometry).geometry.boundary_role == "physical_inner_rim"


def test_outward_detection_gate_cannot_supply_grid_geometry() -> None:
    with pytest.raises(ArenaMMGridPolicyError, match="detection gate"):
        _geometry(
            boundary_role=REJECTED_DETECTION_GATE_BOUNDARY_ROLE,
            observed_feature=REJECTED_DETECTION_GATE_BOUNDARY_ROLE,
        )


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: _geometry(radius_px=0.0), "radius_px"),
        (lambda: _scale(mm_per_pixel=0.0), "mm_per_pixel"),
        (lambda: _geometry(coordinate_space="pixel_grid"), "native camera pixels"),
        (lambda: _geometry(geometry_id="latest"), "immutable identity"),
        (lambda: _scale(scale_id="default"), "immutable identity"),
    ],
)
def test_invalid_or_mutable_authorities_fail_closed(factory, match: str) -> None:
    with pytest.raises(ArenaMMGridPolicyError, match=match):
        factory()


def test_policy_rejects_observed_data_and_pixel_grid_policy() -> None:
    with pytest.raises(ArenaMMGridPolicyError, match="observed positions"):
        _policy(observed_positions=np.asarray([[0.0, 0.0]]))

    policy = _policy()
    with pytest.raises(ArenaMMGridPolicyError, match="arena-centred millimetres"):
        type(policy)(
            policy_id=policy.policy_id,
            recording_id=policy.recording_id,
            geometry=policy.geometry,
            scale=policy.scale,
            selection=policy.selection,
            bin_width_mm=policy.bin_width_mm,
            x_edges=policy.x_edges,
            y_edges=policy.y_edges,
            extent_radius_mm=policy.extent_radius_mm,
            rim_radius_mm=policy.rim_radius_mm,
            grid_coordinate_space=GEOMETRY_COORDINATE_SPACE_ID,
        )


def test_stale_geometry_and_policy_digests_fail_closed() -> None:
    geometry_record = _geometry().as_record()
    geometry_record["radius_px"] = CANARY_RADIUS_PX + 1.0
    with pytest.raises(ArenaMMGridPolicyError, match="stale record_sha256"):
        CircularArenaGeometryAuthority.from_record(geometry_record)

    policy_record = _policy().as_record()
    policy_record["extent_radius_mm"] = 40.0
    with pytest.raises(ArenaMMGridPolicyError, match="stale record_sha256"):
        type(_policy()).from_record(policy_record)


def test_stale_selection_payload_digest_fails_closed() -> None:
    payload = {"selection_id": "selection-v1", "recording_id": RECORDING_ID}
    selection = SelectionAuthority.from_payload(
        selection_id="selection-v1",
        recording_id=RECORDING_ID,
        payload=payload,
    )
    with pytest.raises(ArenaMMGridPolicyError, match="stale record_sha256"):
        SelectionAuthority(
            selection_id=selection.selection_id,
            recording_id=selection.recording_id,
            record_sha256="0" * 64,
            record_payload=payload,
        )


def test_recording_and_coordinate_authorities_must_agree() -> None:
    with pytest.raises(ArenaMMGridPolicyError, match="recording_id"):
        _policy(recording_id="another-recording")
    with pytest.raises(ArenaMMGridPolicyError, match="different coordinate authorities"):
        _policy(scale=_scale(coordinate_authority_id="other-coordinate-authority"))

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_gaze_tracking as gaze_module
from fisheye.analysis.chaser_component_publication import (
    ChaserComponentContract,
    build_chaser_component_handle,
    component_record_sha256,
    persist_chaser_component_manifest,
)
from fisheye.analysis.chaser_distance_io import load_chaser_distance_run
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_egocentric_bearing import (
    METHOD as EGOCENTRIC_METHOD,
    METHOD_VERSION as EGOCENTRIC_METHOD_VERSION,
    SCHEMA_ID as EGOCENTRIC_SCHEMA_ID,
    SCHEMA_VERSION as EGOCENTRIC_SCHEMA_VERSION,
)
from fisheye.analysis.chaser_gaze_tracking import (
    _dense_frame_row_lookup,
    _resolve_egocentric_component,
    _source_refs,
    _virtual_positions,
    fit_dynamic_tracking_gain,
    fit_linear_tracking_gain,
    sustained_true_runs,
)
from tests.unit.fisheye.test_chaser_radial_occupancy import _make_archive


pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


def _egocentric_fixture(
    tmp_path: Path,
    *,
    name: str,
) -> tuple[Path, dict[str, object]]:
    frame_count = 8
    fish = np.zeros((frame_count, 2), dtype=np.float32)
    chaser = np.ones((frame_count, 1, 2), dtype=np.float32)
    zarr_path = _make_archive(
        tmp_path,
        fish_xy_mm=fish,
        chaser_xy_mm=chaser,
        windows=(
            ChaserDistanceWindow(
                0,
                "all_frames",
                0,
                frame_count - 1,
                0.0,
                0.8,
                0.8,
            ),
        ),
        name=name,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root["analysis/chaser_distance_runs/chaser_distance_1"]
    parent = run.require_group("egocentric_bearing")
    component = parent.require_group("sealed_ego")
    frames = component.require_group("frames")
    frames.create_array(
        "camera_frame_id", data=np.arange(frame_count, dtype=np.int64)
    )
    frames.create_array(
        "fish_heading_deg", data=np.zeros(frame_count, dtype=np.float32)
    )
    frames.create_array(
        "fish_heading_valid", data=np.ones(frame_count, dtype=bool)
    )
    per_chaser = component.require_group("per_chaser")
    per_chaser.create_array(
        "bearing_deg", data=np.zeros((frame_count, 1), dtype=np.float32)
    )
    per_chaser.create_array(
        "distance_mm", data=np.ones((frame_count, 1), dtype=np.float32)
    )
    per_chaser.create_array(
        "valid", data=np.ones((frame_count, 1), dtype=bool)
    )
    per_chaser.attrs["angle_convention"] = (
        "fish_body_frame; zero=forward; positive=anatomical_left"
    )
    component.attrs.update(
        {
            "schema_id": EGOCENTRIC_SCHEMA_ID,
            "schema_version": EGOCENTRIC_SCHEMA_VERSION,
            "method": EGOCENTRIC_METHOD,
            "method_version": EGOCENTRIC_METHOD_VERSION,
        }
    )
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    persist_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path="egocentric_bearing/sealed_ego",
        contract=ChaserComponentContract(
            component_family="egocentric_bearing",
            component_name="sealed_ego",
            semantic_schema_id=EGOCENTRIC_SCHEMA_ID,
            semantic_schema_version=EGOCENTRIC_SCHEMA_VERSION,
            method_id=EGOCENTRIC_METHOD,
            method_version=EGOCENTRIC_METHOD_VERSION,
            parameters={"fixture": "exact_dependency"},
            source_authorities={"fixture": "sealed"},
        ),
    )
    component.attrs["palette_run_completion_status"] = "complete"
    component.attrs["stage_selector_eligible"] = False
    parent.attrs.update({"latest": "sealed_ego", "latest_complete": "sealed_ego"})
    return zarr_path, build_chaser_component_handle(
        component,
        snapshot=snapshot,
        relative_path="egocentric_bearing/sealed_ego",
    )


def _resolve_fixture(
    zarr_path: Path,
    *,
    handle: dict[str, object] | None,
    legacy_compatibility: bool,
):
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    snapshot = load_chaser_distance_run(root, run_name="chaser_distance_1")
    return _resolve_egocentric_component(
        root,
        snapshot=snapshot,
        run_group=root[snapshot.run_path],
        requested="latest",
        dependency_handle=handle,
        legacy_compatibility=legacy_compatibility,
    )


def test_dense_frame_row_lookup_marks_trailing_frames_unavailable() -> None:
    row_index, present = _dense_frame_row_lookup(
        5,
        np.asarray([0, 2, 4, 7, 8, 9], dtype=np.int64),
    )

    np.testing.assert_array_equal(present, [True, True, True, False, False, False])
    np.testing.assert_array_equal(row_index, [0, 2, 4, -1, -1, -1])


def test_dense_frame_row_lookup_refuses_negative_row_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _dense_frame_row_lookup(-1, np.asarray([0, 1, 2], dtype=np.int64))


def test_static_tracking_gain_distinguishes_eye_tracking_from_head_fixed_eye() -> None:
    bearing = np.linspace(50.0, 100.0, 500)
    valid = np.ones(bearing.shape, dtype=bool)
    tracked = 0.8 * bearing + 5.0
    fixed = np.full(bearing.shape, 72.0)
    tracked_fit = fit_linear_tracking_gain(bearing, tracked, valid)
    fixed_fit = fit_linear_tracking_gain(bearing, fixed, valid)
    assert np.isclose(tracked_fit.gain, 0.8)
    assert np.isclose(tracked_fit.intercept_deg, 5.0)
    assert np.isclose(fixed_fit.gain, 0.0)


def test_dynamic_tracking_gain_recovers_positive_eye_lag() -> None:
    rng = np.random.default_rng(3)
    increments = rng.normal(0.0, 1.0, 600)
    bearing = np.cumsum(increments)
    gaze = np.zeros_like(bearing)
    gaze[3:] = bearing[:-3]
    valid = np.ones(bearing.shape, dtype=bool)
    fit = fit_dynamic_tracking_gain(bearing, gaze, valid, fps=100.0, max_lag_s=0.1)
    assert fit.lag_frames == 3
    assert fit.lag_seconds == 0.03
    assert fit.correlation > 0.99
    assert np.isclose(fit.gain, 1.0, atol=0.02)


def test_sustained_true_runs_uses_inclusive_intervals_and_minimum_length() -> None:
    mask = np.asarray([False, True, True, False, True, True, True, False, True])
    assert sustained_true_runs(mask, min_frames=3) == ((4, 6),)


def test_virtual_reference_is_dropped_when_it_overlaps_a_real_object() -> None:
    chaser_xy = np.asarray(
        [
            [[1.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [9.0, 9.0]],
        ]
    )
    refs, _positions = _virtual_positions(
        chaser_xy=chaser_xy,
        chaser_indices=np.asarray([0, 1]),
        center_xy=(5.0, 5.0),
        rotations_deg=(60.0, 180.0),
        min_separation_mm=1.0,
        pixels_per_mm=1.0,
    )
    assert {ref.rotation_deg for ref in refs} == {60.0}
    assert len(refs) == 2


def test_egocentric_dependency_handle_opens_exact_sealed_component(
    tmp_path: Path,
) -> None:
    zarr_path, handle = _egocentric_fixture(tmp_path, name="exact.zarr")

    _component, name, path, manifest_sha256 = _resolve_fixture(
        zarr_path,
        handle=handle,
        legacy_compatibility=False,
    )

    assert name == "sealed_ego"
    assert path.endswith("/egocentric_bearing/sealed_ego")
    assert manifest_sha256 == handle["component_manifest_sha256"]

    source_refs = _source_refs(
        SimpleNamespace(
            chaser_distance_run_name="canonical",
            chaser_distance_run_path="analysis/chaser_distance_runs/canonical",
            egocentric_component_name=name,
            egocentric_component_path=path,
            egocentric_component_manifest_sha256=manifest_sha256,
            eye_angle_run_name="eye_angles",
            eye_angle_run_path="analysis/eye_angle_runs/eye_angles",
        )
    )
    assert source_refs["egocentric_component_manifest_sha256"] == (
        handle["component_manifest_sha256"]
    )


def test_egocentric_latest_discovery_requires_explicit_compatibility(
    tmp_path: Path,
) -> None:
    zarr_path, _handle = _egocentric_fixture(tmp_path, name="legacy.zarr")

    with pytest.raises(ValueError, match="explicit self-digested"):
        _resolve_fixture(
            zarr_path,
            handle=None,
            legacy_compatibility=False,
        )

    _component, name, _path, manifest_sha256 = _resolve_fixture(
        zarr_path,
        handle=None,
        legacy_compatibility=True,
    )
    assert name == "sealed_ego"
    assert manifest_sha256 is None


@pytest.mark.parametrize(
    "field",
    ["component_path", "component_manifest_sha256", "record_sha256"],
)
def test_invalid_explicit_egocentric_handle_never_falls_back_to_latest(
    tmp_path: Path,
    field: str,
) -> None:
    zarr_path, handle = _egocentric_fixture(tmp_path, name=f"invalid-{field}.zarr")
    invalid = copy.deepcopy(handle)
    if field != "record_sha256":
        invalid[field] = (
            f"{invalid[field]}-wrong"
            if field == "component_path"
            else "0" * 64
        )
        body = {key: value for key, value in invalid.items() if key != "record_sha256"}
        invalid["record_sha256"] = component_record_sha256(body)
    else:
        invalid[field] = "0" * 64

    with pytest.raises(ValueError):
        _resolve_fixture(
            zarr_path,
            handle=invalid,
            legacy_compatibility=True,
        )


def test_egocentric_handle_rejects_different_base_snapshot(tmp_path: Path) -> None:
    zarr_path, handle = _egocentric_fixture(tmp_path, name="source.zarr")
    handle["base_publication_seal_sha256"] = "f" * 64
    body = {key: value for key, value in handle.items() if key != "record_sha256"}
    handle["record_sha256"] = component_record_sha256(body)

    with pytest.raises(ValueError, match="different base authority"):
        _resolve_fixture(
            zarr_path,
            handle=handle,
            legacy_compatibility=True,
        )


def test_egocentric_handle_rejects_wrong_component_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gaze_module,
        "open_explicit_chaser_component_group",
        lambda *_args, **_kwargs: SimpleNamespace(
            component_family="wrong_family",
            component_name="sealed_ego",
            component_path="analysis/chaser_distance_runs/run/wrong_family/sealed_ego",
            manifest_sha256="a" * 64,
            group={},
        ),
    )

    with pytest.raises(ValueError, match="different component family"):
        _resolve_egocentric_component(
            {},
            snapshot=SimpleNamespace(run_path="analysis/chaser_distance_runs/run"),
            run_group={},
            requested="latest",
            dependency_handle={"explicit": True},
            legacy_compatibility=True,
        )

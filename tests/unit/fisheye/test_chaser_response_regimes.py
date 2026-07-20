from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.chaser_response_regimes as regimes_module
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_response_regimes import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    ESCAPE_GAIN_PNG_ARTIFACT_NAME,
    FREEZE_CURVE_PNG_ARTIFACT_NAME,
    INTERACTIVE_ARTIFACT_NAME,
    SCHEMA_ID,
    TRACKING_QC_PNG_ARTIFACT_NAME,
    build_chaser_response_regimes_result,
    write_chaser_response_regimes_component,
)
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID
from tests.unit.fisheye.test_chaser_radial_occupancy import ARENA_CENTER_MM, _make_archive


FPS = 10.0
CX = CY = ARENA_CENTER_MM  # 50 mm


def _install_verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load_verified_track(
        root: zarr.Group,
        *,
        run_name: str = "latest",
        scope: str = "offline",
        track_id: int = 0,
        required_speed_levels: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        assert run_name == "latest"
        assert scope == "offline"
        assert track_id == 0
        assert required_speed_levels == ("smoothed",)
        parent = root["analysis/chaser_distance_runs"]
        distance_run_name = str(
            parent.attrs.get("latest") or sorted(parent.group_keys())[-1]
        )
        distance_run = parent[distance_run_name]
        positions = distance_run["positions"]
        fish_px = np.asarray(
            positions["fish_centroid_arena_xy"][:], dtype=np.float64
        )
        sample_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
        pixels_per_mm = float(distance_run.attrs["pixels_per_mm_projector"])
        fps = float(distance_run.attrs["fps"])
        fish_mm = fish_px / pixels_per_mm
        speed = np.full(fish_mm.shape[0], np.nan, dtype=np.float64)
        transition_valid = np.zeros(fish_mm.shape[0], dtype=bool)
        if fish_mm.shape[0] > 1:
            speed[1:] = np.linalg.norm(np.diff(fish_mm, axis=0), axis=1) * fps
            transition_valid[1:] = (
                sample_valid[:-1]
                & sample_valid[1:]
                & np.isfinite(fish_mm[:-1]).all(axis=1)
                & np.isfinite(fish_mm[1:]).all(axis=1)
            )
        run_ref = "/analysis/track_kinematics_runs/offline/verified_test"
        track_ref = f"{run_ref}/tracks/id_0"
        authority = {
            "schema_id": "palette.track_motion_read_authority",
            "schema_version": 1,
            "run_ref": run_ref,
            "track_ref": track_ref,
            "track_id": 0,
            "motion_manifest_ref": (
                f"{run_ref}@track_motion_publication_manifest"
            ),
            "motion_manifest_sha256": "a" * 64,
            "positions_px_ref": f"{track_ref}/positions_px",
            "positions_px_coordinate_descriptor_sha256": "b" * 64,
            "positions_mm_ref": f"{track_ref}/positions_mm",
            "positions_mm_coordinate_descriptor_sha256": "c" * 64,
            "track_sample_key_ref": f"{track_ref}/track_sample_key",
            "source_acquisition_frame_index_ref": (
                f"{track_ref}/source_acquisition_frame_index"
            ),
        }
        return SimpleNamespace(
            track_path=track_ref.removeprefix("/"),
            source_acquisition_frame_index=np.arange(
                fish_mm.shape[0], dtype=np.int64
            ),
            speed_mm_by_level={"smoothed": speed},
            sample_valid=sample_valid,
            transition_valid=transition_valid,
            authority_record=lambda: dict(authority),
        )

    monkeypatch.setattr(
        regimes_module,
        "load_track_kinematics_track",
        _load_verified_track,
    )


@pytest.fixture(autouse=True)
def _verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_track_reader(monkeypatch)


def _set_protocol(zarr_path: Path, *, chaser_radius_mm: float = 2.0) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["protocol_json"] = json.dumps(
        {
            "steps": [
                {
                    "parameters": {
                        "position_transition_duration_s": 0.0,
                        "chasers": [{"radius_mm": chaser_radius_mm}],
                    }
                }
            ]
        }
    )


def _build(tmp_path: Path, fish_mm: np.ndarray, chaser_mm: np.ndarray, *, name: str,
           fish_valid: np.ndarray | None = None, chaser_radius_mm: float = 2.0):
    """All fixtures keep the two agents on the same side of each other: if a trajectory
    passed *through* the other agent the radial sign would flip mid-run and the expected
    values would be meaningless."""
    n = fish_mm.shape[0]
    windows = (ChaserDistanceWindow(0, "training_event", 0, n - 1, 0.0, n / FPS, n / FPS),)
    z = _make_archive(
        tmp_path,
        fish_xy_mm=fish_mm,
        chaser_xy_mm=chaser_mm.reshape(n, 1, 2),
        windows=windows,
        fps=FPS,
        fish_valid=fish_valid,
        name=name,
    )
    _set_protocol(z, chaser_radius_mm=chaser_radius_mm)
    return z


def _freeze_then_flee(n_near: int = 200, n_far: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fish frozen 3 mm from a static chaser, then swimming radially away at 5 mm/s far out.

    The two segments are disjoint in space, so the single frame that bridges them is marked
    invalid -- otherwise its step would read as one enormous escape bout inside the near bin.
    """

    near = np.tile(np.asarray([CX + 3.0, CY]), (n_near, 1))
    # 5 mm/s at 10 fps == 0.5 mm/frame, starting 25 mm out and staying inside the last bin.
    far = np.stack([CX + 25.0 + 0.5 * np.arange(n_far), np.full(n_far, CY)], axis=1)
    fish = np.concatenate([near, far], axis=0)
    chaser = np.tile(np.asarray([CX, CY]), (fish.shape[0], 1))
    valid = np.ones(fish.shape[0], dtype=bool)
    valid[n_near] = False
    return fish, chaser, valid


# --------------------------------------------------------------------------------------
# The freeze curve and the escape gain
# --------------------------------------------------------------------------------------


def test_freeze_curve_recovers_a_close_range_freeze(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee()
    z = _build(tmp_path, fish, chaser, fish_valid=valid, name="freeze.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    assert float(r.immobile_fraction_near[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(r.immobile_fraction_far[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(r.freeze_index[0, 0]) == pytest.approx(1.0, abs=1e-6)

    centers = r.distance_bin_centers_mm
    near_bin = int(np.argmin(np.abs(centers - 3.5)))
    assert float(r.immobile_fraction[0, 0, near_bin]) == pytest.approx(1.0, abs=1e-6)


def test_verified_speed_uses_destination_frame_transition_anchor(
    tmp_path: Path,
) -> None:
    fish = np.asarray(
        [[CX + 10.0, CY], [CX + 10.0, CY], [CX + 11.0, CY]],
        dtype=np.float64,
    )
    chaser = np.tile(np.asarray([CX, CY]), (3, 1))
    z = _build(tmp_path, fish, chaser, name="transition_anchor.zarr")

    result = build_chaser_response_regimes_result(
        z,
        chaser_distance_run="chaser_distance_1",
        min_band_frames=1,
        min_bin_frames=1,
    )

    distance_bin = int(
        np.flatnonzero(
            (result.distance_bin_edges_mm[:-1] <= 10.0)
            & (result.distance_bin_edges_mm[1:] > 10.0)
        )[0]
    )
    assert result.frame_count[0, 0, distance_bin] == 2
    assert result.immobile_fraction[0, 0, distance_bin] == pytest.approx(0.5)
    assert result.moving_fraction[0, 0, distance_bin] == pytest.approx(0.5)


def test_verified_track_failure_does_not_fall_back_to_raw_centroid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fish, chaser, valid = _freeze_then_flee(n_near=20, n_far=10)
    z = _build(
        tmp_path,
        fish,
        chaser,
        fish_valid=valid,
        name="tampered_track.zarr",
    )
    monkeypatch.setattr(
        regimes_module,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("track motion manifest mismatch")
        ),
    )

    with pytest.raises(ValueError, match="manifest mismatch"):
        build_chaser_response_regimes_result(
            z,
            chaser_distance_run="chaser_distance_1",
            min_bin_frames=1,
        )


def test_raw_centroid_requires_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fish, chaser, valid = _freeze_then_flee(n_near=20, n_far=10)
    z = _build(
        tmp_path,
        fish,
        chaser,
        fish_valid=valid,
        name="explicit_raw.zarr",
    )
    monkeypatch.setattr(
        regimes_module,
        "load_track_kinematics_track",
        lambda *_args, **_kwargs: pytest.fail("strict track reader must not run"),
    )

    result = build_chaser_response_regimes_result(
        z,
        chaser_distance_run="chaser_distance_1",
        min_bin_frames=1,
        immobility_signal_mode="raw_centroid_explicit",
    )

    assert result.source_track_motion_authority is None
    assert result.diagnostics["immobility_signal_source"] == (
        "raw_centroid_explicit"
    )
    assert "immobility_signal_explicit_raw_centroid" in result.qc_warnings


def test_escape_gain_is_the_fish_velocity_along_the_chaser_axis(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee()
    z = _build(tmp_path, fish, chaser, fish_valid=valid, name="gain.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    # The fish swims directly away at 5 mm/s, so v_r == -5 wherever it is moving.
    assert float(r.escape_gain_far_mm_s[0, 0]) == pytest.approx(-5.0, abs=0.05)
    # It never swims toward the dot.
    assert float(r.approach_fraction[0, 0]) == pytest.approx(0.0, abs=1e-6)

    centers = r.distance_bin_centers_mm
    far = centers >= 20.0
    away = r.fraction_moving_away[0, 0, far]
    assert np.allclose(away[np.isfinite(away)], 1.0)


def test_fish_swimming_toward_the_chaser_gives_positive_radial_velocity(tmp_path: Path) -> None:
    n = 60
    fish = np.stack([CX + 55.0 - 0.5 * np.arange(n), np.full(n, CY)], axis=1)  # closing at 5 mm/s
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="toward.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    assert float(r.approach_fraction[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(r.escape_gain_far_mm_s[0, 0]) == pytest.approx(+5.0, abs=0.05)


# --------------------------------------------------------------------------------------
# The decomposition: who actually closes the distance
# --------------------------------------------------------------------------------------


def test_chaser_only_motion_is_attributed_to_the_chaser(tmp_path: Path) -> None:
    """A stationary fish and an approaching dot: the fish contributes nothing to the
    closing rate. This is the confound the whole module exists to break."""

    n = 60
    fish = np.tile(np.asarray([CX, CY]), (n, 1))
    chaser = np.stack([CX + 55.0 - 0.5 * np.arange(n), np.full(n, CY)], axis=1)  # 5 mm/s toward
    z = _build(tmp_path, fish, chaser, name="chaser_only.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    centers = r.distance_bin_centers_mm
    counted = r.frame_count[0, 0, :] > 5
    fish_sep = r.fish_separation_rate_mm_s[0, 0, counted]
    chaser_sep = r.chaser_separation_rate_mm_s[0, 0, counted]
    net = r.net_separation_rate_mm_s[0, 0, counted]

    assert np.allclose(fish_sep[np.isfinite(fish_sep)], 0.0, atol=1e-4)
    assert np.allclose(chaser_sep[np.isfinite(chaser_sep)], -5.0, atol=0.05)
    assert np.allclose(net[np.isfinite(net)], -5.0, atol=0.05)
    # The fish is stationary, so it reads as frozen everywhere -- not as an escape.
    assert float(r.immobile_fraction_far[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert centers.shape[0] == r.frame_count.shape[2]


def test_fish_only_motion_is_attributed_to_the_fish(tmp_path: Path) -> None:
    n = 60
    fish = np.stack([CX + 25.0 + 0.5 * np.arange(n), np.full(n, CY)], axis=1)  # 5 mm/s away
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="fish_only.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    counted = r.frame_count[0, 0, :] > 5
    fish_sep = r.fish_separation_rate_mm_s[0, 0, counted]
    chaser_sep = r.chaser_separation_rate_mm_s[0, 0, counted]
    assert np.allclose(fish_sep[np.isfinite(fish_sep)], +5.0, atol=0.05)
    assert np.allclose(chaser_sep[np.isfinite(chaser_sep)], 0.0, atol=1e-4)


# --------------------------------------------------------------------------------------
# The clamp: the hole at the centre is the dot, not the fish
# --------------------------------------------------------------------------------------


def test_distance_floor_at_chaser_radius_is_flagged(tmp_path: Path) -> None:
    n = 200
    # The fish never gets closer than the 2 mm dot radius -- exactly the clamp signature.
    radii = 2.0 + np.abs(np.sin(np.linspace(0, 6.0, n))) * 8.0
    fish = np.stack([CX + radii, np.full(n, CY)], axis=1)
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="clamp.zarr", chaser_radius_mm=2.0)
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    assert float(r.chaser_radius_mm[0]) == pytest.approx(2.0)
    assert float(r.min_distance_mm[0, 0]) == pytest.approx(2.0, abs=0.05)
    assert bool(r.distance_floor_is_clamp[0, 0])
    assert any(w.startswith("distance_floor_at_chaser_radius") for w in r.qc_warnings)


def test_distance_floor_well_above_radius_is_not_flagged_as_clamp(tmp_path: Path) -> None:
    n = 200
    fish = np.stack([CX + 10.0 + np.abs(np.sin(np.linspace(0, 6.0, n))) * 8.0, np.full(n, CY)], axis=1)
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="noclamp.zarr", chaser_radius_mm=2.0)
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    assert float(r.min_distance_mm[0, 0]) > 5.0
    assert not bool(r.distance_floor_is_clamp[0, 0])
    assert not any(w.startswith("distance_floor_at_chaser_radius") for w in r.qc_warnings)


# --------------------------------------------------------------------------------------
# Tracking QC: the freeze/dropout coupling is a first-class output
# --------------------------------------------------------------------------------------


def test_tracking_dropout_and_gap_structure_are_reported(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee(n_near=200, n_far=60)
    n = fish.shape[0]
    valid[50:150] = False  # one contiguous 100-frame (10 s) hole
    z = _build(tmp_path, fish, chaser, name="dropout.zarr", fish_valid=valid)
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    qc = r.tracking_qc[0]
    # 100 injected + the 1-frame segment boundary the fixture already carries.
    assert qc.dropped_frames == 101
    assert qc.gap_count == 2
    assert qc.longest_gap_frames == 100
    assert qc.longest_gap_s == pytest.approx(10.0)
    assert qc.dropout_fraction == pytest.approx(101.0 / n, rel=1e-3)
    assert any(w.startswith("high_tracking_dropout") for w in r.qc_warnings)
    assert any(w.startswith("long_tracking_gap") for w in r.qc_warnings)


def test_clean_tracking_raises_no_dropout_warnings(tmp_path: Path) -> None:
    # Single-regime trajectory, so there is no segment boundary and nothing to drop.
    n = 120
    fish = np.stack([CX + 25.0 + 0.25 * np.arange(n), np.full(n, CY)], axis=1)
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="clean.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    qc = r.tracking_qc[0]
    assert qc.dropped_frames == 0
    assert qc.gap_count == 0
    assert qc.longest_gap_s == pytest.approx(0.0)
    assert qc.analyzable_pairs == n - 1
    assert not any(w.startswith("high_tracking_dropout") for w in r.qc_warnings)
    assert not any(w.startswith("long_tracking_gap") for w in r.qc_warnings)


def test_velocity_is_not_measured_across_a_tracking_gap(tmp_path: Path) -> None:
    """The fish is teleported across the gap. If the estimator differenced across it, the
    step would register as an enormous velocity; the pair mask must drop it."""

    n = 240
    fish = np.tile(np.asarray([CX + 3.0, CY]), (n, 1))
    fish[120:] = np.asarray([CX + 40.0, CY])  # 37 mm jump, hidden inside the gap
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    valid = np.ones(n, dtype=bool)
    valid[100:120] = False
    z = _build(tmp_path, fish, chaser, name="gapjump.zarr", fish_valid=valid)
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)

    # Every analyzable pair has a stationary fish, so no bin may show a large speed.
    speeds = r.mean_speed_mm_s[0, 0, :]
    assert np.all(speeds[np.isfinite(speeds)] < 1e-3)
    assert float(r.immobile_fraction_far[0, 0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Undersampling guard
# --------------------------------------------------------------------------------------


def test_undersampled_band_returns_nan_rather_than_a_spurious_effect(tmp_path: Path) -> None:
    n = 400
    fish = np.tile(np.asarray([CX + 30.0, CY]), (n, 1))
    fish[:5] = np.asarray([CX + 3.0, CY])  # only 5 frames anywhere near the chaser
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="undersampled.zarr")

    r = build_chaser_response_regimes_result(
        z, chaser_distance_run="chaser_distance_1", min_band_frames=50, min_bin_frames=1
    )
    assert math.isnan(float(r.immobile_fraction_near[0, 0]))
    assert math.isnan(float(r.freeze_index[0, 0]))
    assert math.isfinite(float(r.immobile_fraction_far[0, 0]))

    # Dropping the guard lets the 5-frame band through -- which is exactly the noise we hide.
    loose = build_chaser_response_regimes_result(
        z, chaser_distance_run="chaser_distance_1", min_band_frames=1, min_bin_frames=1
    )
    assert math.isfinite(float(loose.immobile_fraction_near[0, 0]))


def test_thin_bins_are_suppressed_in_the_curves(tmp_path: Path) -> None:
    """A bin holding a handful of frames swings wildly and reads as structure on a plot.
    The curve point must be NaN while frame_count still records the support."""

    n = 400
    fish = np.tile(np.asarray([CX + 30.0, CY]), (n, 1))
    fish[:8] = np.asarray([CX + 3.0, CY])  # 8 frames only, in the 3-4 mm bin
    chaser = np.tile(np.asarray([CX, CY]), (n, 1))
    z = _build(tmp_path, fish, chaser, name="thinbin.zarr")

    r = build_chaser_response_regimes_result(
        z, chaser_distance_run="chaser_distance_1", min_bin_frames=20
    )
    centers = r.distance_bin_centers_mm
    thin_bin = int(np.argmin(np.abs(centers - 3.5)))
    assert int(r.frame_count[0, 0, thin_bin]) > 0          # support is still recorded
    assert math.isnan(float(r.immobile_fraction[0, 0, thin_bin]))  # but the value is withheld

    fat_bin = int(np.argmin(np.abs(centers - 35.0)))
    assert math.isfinite(float(r.immobile_fraction[0, 0, fat_bin]))

    loose = build_chaser_response_regimes_result(
        z, chaser_distance_run="chaser_distance_1", min_bin_frames=1
    )
    assert math.isfinite(float(loose.immobile_fraction[0, 0, thin_bin]))


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_write_component_round_trips(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee()
    z = _build(tmp_path, fish, chaser, fish_valid=valid, name="persist.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)
    path = write_chaser_response_regimes_component(z, r, overwrite=True)

    assert path == (
        f"analysis/chaser_distance_runs/chaser_distance_1/{COMPONENT_PARENT_NAME}/{DEFAULT_COMPONENT_NAME}"
    )
    root = zarr.open_group(str(z), mode="r", use_consolidated=False)
    component = root[path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "computed"
    assert component.attrs["source_refs"]["source_track_motion_authority"] == (
        r.source_track_motion_authority
    )
    assert component.attrs["parameters"]["immobility_signal_mode"] == (
        "verified_track_motion"
    )
    assert component.attrs["parameters"]["immobility_signal_source"] == (
        "track_motion.movement/speed/smoothed/mm"
    )

    n_bins = int(r.distance_bin_centers_mm.shape[0])
    regimes = component["regimes"]
    for name in (
        "immobile_fraction",
        "fish_radial_velocity_moving_mm_s",
        "fish_separation_rate_mm_s",
        "chaser_separation_rate_mm_s",
        "net_separation_rate_mm_s",
    ):
        assert regimes[name].shape == (1, 1, n_bins), name
    assert regimes.attrs["axis_order"] == ["epoch", "chaser", "distance_bin"]

    per_epoch = component["per_epoch_chaser"]
    assert float(np.asarray(per_epoch["freeze_index"][:])[0, 0]) == pytest.approx(1.0, abs=1e-5)

    qc = component["tracking_qc"]
    assert qc["dropout_fraction"].shape == (1,)
    # The fixture's only gap is its 1-frame segment boundary.
    assert float(np.asarray(qc["longest_gap_s"][:])[0]) == pytest.approx(0.1)

    assert float(np.asarray(component["chasers"]["chaser_radius_mm"][:])[0]) == pytest.approx(2.0)

    visualizations = component["visualizations"]
    for artifact in (
        FREEZE_CURVE_PNG_ARTIFACT_NAME,
        ESCAPE_GAIN_PNG_ARTIFACT_NAME,
        TRACKING_QC_PNG_ARTIFACT_NAME,
    ):
        png = visualizations[artifact]
        assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
        assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG")
    spec = visualizations[INTERACTIVE_ARTIFACT_NAME]
    assert spec.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec.attrs["renderer"] == "palette-chaser-response-regimes-v1"

    parent = root[f"analysis/chaser_distance_runs/chaser_distance_1/{COMPONENT_PARENT_NAME}"]
    assert parent.attrs["latest_complete"] == DEFAULT_COMPONENT_NAME


def test_write_component_requires_overwrite(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee()
    z = _build(tmp_path, fish, chaser, fish_valid=valid, name="twice.zarr")
    r = build_chaser_response_regimes_result(z, chaser_distance_run="chaser_distance_1", min_bin_frames=1)
    write_chaser_response_regimes_component(z, r, overwrite=True, write_png=False)
    with pytest.raises(ValueError, match="already exists"):
        write_chaser_response_regimes_component(z, r, overwrite=False, write_png=False)


def test_rejects_moving_threshold_below_immobility_threshold(tmp_path: Path) -> None:
    fish, chaser, valid = _freeze_then_flee()
    z = _build(tmp_path, fish, chaser, fish_valid=valid, name="badthresh.zarr")
    with pytest.raises(ValueError, match="moving_speed_threshold_mm_s"):
        build_chaser_response_regimes_result(
            z,
            chaser_distance_run="chaser_distance_1",
            immobility_speed_threshold_mm_s=5.0,
            moving_speed_threshold_mm_s=1.0,
        )

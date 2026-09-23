from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.visualization.interactive_track_kinematics as mod

RUN_NAME = "motion_current_v1"
RUN_PATH = f"analysis/track_kinematics_runs/provider/{RUN_NAME}"


def _archive(tmp_path: Path, *, bout_runs: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "provider_interactive.zarr"
    root = zarr.open_group(
        str(path),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    provider = root.require_group("analysis/track_kinematics_runs/provider")
    provider.create_group(RUN_NAME)
    bouts = root.require_group("analysis/swim_bout_runs")
    for run_name in bout_runs:
        run = bouts.create_group(run_name)
        run.attrs.update(
            {
                "source_track_kinematics_scope": "provider",
                "source_track_kinematics_run": RUN_NAME,
                "track_id": 7,
            }
        )
    return path


def _provider() -> SimpleNamespace:
    arrays = {
        "track_ids": np.asarray([7, 9], dtype=np.int64),
        "track_row_offsets": np.asarray([0, 3, 5], dtype=np.int64),
        "time_seconds": np.asarray([0.0, 0.1, 0.2, 0.0, 0.1], dtype=np.float32),
        "source_acquisition_frame_index": np.asarray(
            [100, 101, 103, 200, 201],
            dtype=np.int64,
        ),
        "positions_px": np.asarray(
            [[10, 20], [11, 20], [12, 21], [30, 40], [31, 40]],
            dtype=np.float32,
        ),
        "positions_mm": np.asarray(
            [[1, 2], [1.1, 2], [1.2, 2.1], [3, 4], [3.1, 4]],
            dtype=np.float32,
        ),
        "linear_sample_valid": np.asarray(
            [True, False, True, True, True],
            dtype=bool,
        ),
        "angular_sample_valid": np.asarray(
            [True, True, False, True, True],
            dtype=bool,
        ),
        "transition_valid": np.asarray(
            [False, True, False, False, True],
            dtype=bool,
        ),
        "linear_sample_reason_code": np.asarray(
            [0, 3, 0, 0, 0],
            dtype=np.uint16,
        ),
        "angular_sample_reason_code": np.asarray(
            [0, 0, 2, 0, 0],
            dtype=np.uint16,
        ),
        "transition_reason_code": np.asarray(
            [1, 0, 2, 1, 0],
            dtype=np.int16,
        ),
        "speed_raw_mm": np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
        "speed_filtered_mm": np.asarray([1, 1.5, 2, 4, 4.5], dtype=np.float32),
        "speed_smoothed_mm": np.asarray([1, 1.4, 1.9, 4, 4.4], dtype=np.float32),
        "speed_averaged_mm": np.asarray([1, 1.3, 1.8, 4, 4.3], dtype=np.float32),
        "acceleration_mm": np.asarray([0, 5, 5, 0, 5], dtype=np.float32),
        "smoothed_acceleration_mm": np.asarray(
            [0, 4, 4, 0, 4],
            dtype=np.float32,
        ),
        "angular_velocity_raw_deg_s": np.asarray(
            [0, 10, 20, 0, 10],
            dtype=np.float32,
        ),
        "angular_speed_raw_deg_s": np.asarray(
            [0, 10, 20, 0, 10],
            dtype=np.float32,
        ),
        "angular_velocity_smoothed_deg_s": np.asarray(
            [0, 8, 16, 0, 8],
            dtype=np.float32,
        ),
        "angular_speed_smoothed_deg_s": np.asarray(
            [0, 8, 16, 0, 8],
            dtype=np.float32,
        ),
    }
    provider = SimpleNamespace(
        run_name=RUN_NAME,
        run_path=RUN_PATH,
        arrays=arrays,
        available_arrays=tuple(sorted(arrays)),
        track_ids=arrays["track_ids"],
        track_row_offsets=arrays["track_row_offsets"],
        row_count=5,
        time_seconds=arrays["time_seconds"],
        source_acquisition_frame_index=arrays["source_acquisition_frame_index"],
        positions_px=arrays["positions_px"],
        positions_mm=arrays["positions_mm"],
        linear_sample_valid=arrays["linear_sample_valid"],
        angular_sample_valid=arrays["angular_sample_valid"],
        transition_valid=arrays["transition_valid"],
        linear_sample_reason_code=arrays["linear_sample_reason_code"],
        angular_sample_reason_code=arrays["angular_sample_reason_code"],
        transition_reason_code=arrays["transition_reason_code"],
        computation_record={
            "linear_sample_reason_codes": {"0": "ok", "3": "position_invalid"},
            "angular_sample_reason_codes": {"0": "ok", "2": "heading_invalid"},
            "transition_reason_codes": {
                "0": "ok",
                "1": "first_sample",
                "2": "frame_gap",
            },
        },
        source_authority_record={
            "position_source": {"estimator_id": "component_mask_triad.v1"}
        },
        provider_manifest_sha256="a" * 64,
        verification_digest="b" * 64,
        timing_is_authoritative=True,
    )
    provider.has_array = lambda name: name in arrays
    provider.array_slice = lambda name, rows=None: (
        arrays[name] if rows is None else arrays[name][rows]
    )
    provider.assert_current = lambda: None
    return provider


def _signal(
    signal_id: int,
    speed_level: str,
    *,
    default: bool,
    n_bouts: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        signal_id=signal_id,
        speed_level=speed_level,
        signal_name=speed_level.removeprefix("speed_"),
        role=(
            "detector_response"
            if speed_level == "speed_exponential"
            else "physical_estimator"
        ),
        source_level="speed_filtered",
        is_default=default,
        n_bouts=n_bouts,
        attrs={},
    )


def _bout_tables(run_name: str = "bouts_current_v1") -> SimpleNamespace:
    signals = (
        _signal(0, "speed_filtered", default=False, n_bouts=1),
        _signal(1, "speed_exponential", default=True, n_bouts=2),
    )
    candidate = SimpleNamespace(
        run_name=run_name,
        candidate_id=0,
        candidate_name="default",
        run_path=f"analysis/swim_bout_runs/{run_name}",
        is_latest=False,
        source_track_kinematics_run=RUN_NAME,
        track_id=7,
        detection_method="peak_event",
        default_signal_id=1,
        default_speed_level="speed_exponential",
        signals=signals,
        attrs={
            "layout": "compact_tabular_v2",
            "threshold_mm": 1.25,
            "exponential_tau_s": 0.025,
            "exponential_source_level": "speed_filtered",
        },
    )
    bouts = np.asarray(
        [(20, 0.1, 0.2)],
        dtype=[
            ("bout_id", "i8"),
            ("start_time_s", "f8"),
            ("end_time_s", "f8"),
        ],
    )
    return SimpleNamespace(
        run_name=run_name,
        run_path=f"analysis/swim_bout_runs/{run_name}",
        level_path=(
            f"analysis/swim_bout_runs/{run_name}/tables/bouts"
            "?candidate_id=0&signal_id=1"
        ),
        candidate=candidate,
        signal=signals[1],
        bouts=bouts,
        peak_events=np.zeros(0, dtype=[]),
        inter_bout_intervals=np.zeros(0, dtype=[]),
        series={
            "detection_signal_mm_s": np.asarray([0.0, 2.0, 0.0]),
            "speed_exponential_mm": np.asarray([0.0, 2.0, 0.0]),
        },
        run_attrs={"fps": 10.0},
    )


def test_discovers_each_current_provider_track_with_estimator_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path)
    provider = _provider()
    calls: list[dict[str, object]] = []
    opened: list[Path] = []
    original_open = mod.open_zarr_root

    def _open(path: Path | str, *args: object, **kwargs: object) -> zarr.Group:
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    def _load(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return provider

    monkeypatch.setattr(
        mod,
        "load_receipt_bound_provider_track_motion_source_handle",
        _load,
    )
    monkeypatch.setattr(mod, "open_zarr_root", _open)

    options = mod.discover_track_kinematics_run_options(
        archive,
        requested_run_path=RUN_PATH,
    )

    assert [option.track_id for option in options] == [7, 9]
    assert all(option.run_path == RUN_PATH for option in options)
    assert all(option.run_scope == "provider" for option in options)
    assert all(option.source_kind == "verified_provider_motion" for option in options)
    assert all(
        option.provider_estimator_id == "component_mask_triad.v1" for option in options
    )
    assert all(option.provider_handle is provider for option in options)
    assert all("selector-ineligible canary" in option.label for option in options)
    assert calls == [{"require_authoritative_timing": True}]
    assert opened == [archive.resolve() / "analysis/track_kinematics_runs/provider"]


def test_provider_bout_discovery_requires_current_strict_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, bout_runs=("bouts_current_v1", "bouts_stale_v1"))
    provider = _provider()
    tables_by_name = {
        name: _bout_tables(name) for name in ("bouts_current_v1", "bouts_stale_v1")
    }
    validations: list[tuple[str, object, slice]] = []

    monkeypatch.setattr(
        mod,
        "load_receipt_bound_provider_track_motion_source_handle",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(
        mod,
        "load_exact_selector_ineligible_swim_bout_events",
        lambda _root, *, run_name, **_kwargs: tables_by_name[run_name],
    )

    def _validate(
        tables: object, **kwargs: object
    ) -> tuple[dict[str, object], str, str]:
        if getattr(tables, "run_name") == "bouts_stale_v1":
            raise ValueError("stale binding")
        validations.append(
            (
                getattr(tables, "run_name"),
                kwargs["validation_profile"],
                kwargs["rows"],
            )
        )
        return {}, "c" * 64, "d" * 64

    monkeypatch.setattr(mod, "validate_provider_swim_bout_binding", _validate)

    options = mod.discover_swim_bout_run_options(
        archive,
        track_run_path=RUN_PATH,
        track_id=7,
    )

    assert [option.speed_level for option in options] == [
        "exponential",
        "filtered",
    ]
    assert all(option.run_name == "bouts_current_v1" for option in options)
    assert all("selector-ineligible canary" in option.label for option in options)
    assert validations == [
        (
            "bouts_current_v1",
            mod.PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1,
            slice(0, 3),
        )
    ]


def test_loads_provider_track_and_one_exact_bout_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, bout_runs=("bouts_current_v1",))
    provider = _provider()
    tables = _bout_tables()
    provider_loads: list[dict[str, object]] = []
    selected_loads: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    opened: list[Path] = []
    original_open = mod.open_zarr_root

    def _open(path: Path | str, *args: object, **kwargs: object) -> zarr.Group:
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    def _load_provider(*_args: object, **kwargs: object) -> SimpleNamespace:
        provider_loads.append(kwargs)
        return provider

    def _load_selected(_root: object, **kwargs: object) -> SimpleNamespace:
        selected_loads.append(kwargs)
        return tables

    monkeypatch.setattr(
        mod,
        "load_receipt_bound_provider_track_motion_source_handle",
        _load_provider,
    )
    monkeypatch.setattr(mod, "open_zarr_root", _open)
    monkeypatch.setattr(
        mod,
        "load_exact_selector_ineligible_swim_bout_events",
        lambda _root, *, run_name, **_kwargs: tables,
    )
    monkeypatch.setattr(
        mod,
        "load_exact_selector_ineligible_swim_bout_overlay_tables",
        _load_selected,
    )
    monkeypatch.setattr(
        mod,
        "validate_provider_swim_bout_binding",
        lambda _tables, **kwargs: validations.append(kwargs)
        or ({}, "c" * 64, "d" * 64),
    )

    data = mod.load_track_kinematics_interactive_data(
        archive,
        run_path=RUN_PATH,
        track_id=7,
        swim_bout_run="bouts_current_v1",
        swim_bout_candidate_id=0,
        swim_bout_signal_id=1,
        speed_level="exponential",
    )

    assert data.spec["schema_id"] == mod.PROVIDER_TRACK_KINEMATICS_SPEC_SCHEMA_ID
    assert data.spec["position_estimator_id"] == "component_mask_triad.v1"
    assert data.artifact_name == mod.PROVIDER_INTERACTIVE_ARTIFACT
    assert data.position_unit == "mm"
    np.testing.assert_allclose(data.time_seconds, [0.0, 0.1, 0.2])
    np.testing.assert_array_equal(data.frame_indices, [100, 101, 103])
    assert np.isnan(data.positions[1]).all()
    np.testing.assert_allclose(data.series["speed_raw_mm"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(data.series["detection_signal_mm_s"], [0.0, 2.0, 0.0])
    assert data.swim_bout_label == ("bouts_current_v1 (speed_exponential) (peak_event)")
    np.testing.assert_allclose(data.swim_bouts, [[0.1, 0.2]])
    assert data.validity_source == "provider_track_independent_validity"
    assert data.validity_labels.tolist() == [
        "transition:frame_gap",
        "linear:position_invalid",
        "angular:heading_invalid",
    ]
    assert len(selected_loads) == 1
    assert {
        key: selected_loads[0][key]
        for key in ("run_name", "candidate_id", "signal_id", "speed_level")
    } == {
        "run_name": "bouts_current_v1",
        "candidate_id": 0,
        "signal_id": 1,
        "speed_level": "exponential",
    }
    assert selected_loads[0]["swim_bout_parent"] is not None
    np.testing.assert_array_equal(
        selected_loads[0]["authoritative_frame_indices"],
        [100, 101, 103],
    )
    assert validations[0]["track_id"] == 7
    assert validations[0]["rows"] == slice(0, 3)
    assert validations[0]["validation_profile"] == (
        mod.PROVIDER_SWIM_BOUT_VALIDATION_PROFILE_CURRENT_STRICT_V1
    )
    assert provider_loads == [{"require_authoritative_timing": True}]
    assert opened == [archive.resolve() / "analysis/swim_bout_runs"]


def test_provider_load_requires_explicit_track_without_selector_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path)
    monkeypatch.setattr(
        mod,
        "load_receipt_bound_provider_track_motion_source_handle",
        lambda *_args, **_kwargs: _provider(),
    )

    with pytest.raises(ValueError, match="requires one track_id"):
        mod.load_track_kinematics_interactive_data(
            archive,
            run_path=RUN_PATH,
            swim_bout_run="none",
        )


@pytest.mark.parametrize("run_path", (f"/{RUN_PATH}", f"{RUN_PATH}/", f" {RUN_PATH}"))
def test_provider_load_rejects_noncanonical_run_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_path: str,
) -> None:
    archive = _archive(tmp_path)
    monkeypatch.setattr(
        mod,
        "load_receipt_bound_provider_track_motion_source_handle",
        lambda *_args, **_kwargs: _provider(),
    )

    with pytest.raises(ValueError, match="run_path"):
        mod.load_track_kinematics_interactive_data(
            archive,
            run_path=run_path,
            track_id=7,
            swim_bout_run="none",
        )

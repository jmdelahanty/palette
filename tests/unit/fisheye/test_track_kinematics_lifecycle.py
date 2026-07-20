from __future__ import annotations

import copy

import numpy as np
import pytest
import zarr

from fisheye.analysis import track_kinematics as mod
from fisheye.analysis.chaser_metrics_loader import ChaserMetricsBundle


def _new_run(tmp_path, *, name: str = "candidate"):
    root = zarr.open_group(
        str(tmp_path / f"{name}.zarr"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    run_name, run = mod.ensure_track_kinematics_run_group(
        root,
        name,
        run_type="offline",
    )
    parent = root["analysis/track_kinematics_runs"]
    # This unit test exercises selection ordering, not provenance enforcement.
    parent.attrs["palette_completion_epoch"] = 1
    return root, parent, parent["offline"], run_name, run


def test_track_completion_validates_complete_while_ineligible_then_selects_last(
    tmp_path,
) -> None:
    root, parent, offline, run_name, run = _new_run(tmp_path)
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
        }
    )
    offline.attrs["latest"] = "previous"
    observed: dict[str, object] = {}

    def validate_complete():
        observed.update(
            {
                "status": run.attrs["palette_run_completion_status"],
                "eligible": run.attrs["stage_selector_eligible"],
                "latest": parent.attrs["latest"],
                "latest_complete": parent.attrs["latest_complete"],
                "latest_offline": parent.attrs["latest_offline"],
                "type_latest": offline.attrs["latest"],
            }
        )
        return {"valid": True}

    mod.mark_track_kinematics_run_complete(
        root,
        run,
        run_name=run_name,
        run_type="offline",
        validate_complete_run=validate_complete,
    )

    assert observed == {
        "status": "complete",
        "eligible": False,
        "latest": "offline/previous",
        "latest_complete": "offline/previous",
        "latest_offline": "previous",
        "type_latest": "previous",
    }
    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    fresh_run = fresh_offline[run_name]
    assert fresh_parent.attrs["latest"] == f"offline/{run_name}"
    assert fresh_parent.attrs["latest_complete"] == f"offline/{run_name}"
    assert fresh_parent.attrs["latest_offline"] == run_name
    assert fresh_offline.attrs["latest"] == run_name
    assert fresh_run.attrs["stage_selector_eligible"] is True


def test_track_completion_keyboard_interrupt_restores_exact_selectors(tmp_path) -> None:
    root, parent, offline, run_name, run = _new_run(tmp_path)
    parent.attrs.update(
        {
            "latest": "offline/previous",
            "latest_complete": "offline/previous",
            "latest_offline": "previous",
            "latest_pending": "unrelated_pending",
        }
    )
    offline.attrs["latest"] = "previous"
    parent_snapshot = copy.deepcopy(dict(parent.attrs))
    offline_snapshot = copy.deepcopy(dict(offline.attrs))

    def interrupt():
        assert run.attrs["palette_run_completion_status"] == "complete"
        assert run.attrs["stage_selector_eligible"] is False
        raise KeyboardInterrupt("injected complete-path interrupt")

    with pytest.raises(KeyboardInterrupt, match="injected complete-path interrupt"):
        mod.mark_track_kinematics_run_complete(
            root,
            run,
            run_name=run_name,
            run_type="offline",
            validate_complete_run=interrupt,
        )

    fresh_parent = root["analysis/track_kinematics_runs"]
    fresh_offline = fresh_parent["offline"]
    fresh_run = fresh_offline[run_name]
    assert dict(fresh_parent.attrs) == parent_snapshot
    assert dict(fresh_offline.attrs) == offline_snapshot
    assert fresh_run.attrs["palette_run_completion_status"] == "failed"
    assert fresh_run.attrs["stage_selector_eligible"] is False


def test_track_overwrite_rejects_complete_and_selected_runs(tmp_path) -> None:
    root, parent, _offline, run_name, run = _new_run(tmp_path, name="complete")
    run.attrs["palette_run_completion_status"] = "complete"
    with pytest.raises(ValueError, match="complete or selected"):
        mod.ensure_track_kinematics_run_group(
            root,
            run_name,
            run_type="offline",
            overwrite=True,
        )

    root2, parent2, _offline2, selected_name, selected = _new_run(
        tmp_path,
        name="selected",
    )
    selected.attrs["palette_run_completion_status"] = "failed"
    parent2.attrs["latest_offline"] = selected_name
    with pytest.raises(ValueError, match="complete or selected"):
        mod.ensure_track_kinematics_run_group(
            root2,
            selected_name,
            run_type="offline",
            overwrite=True,
        )


def test_legacy_chaser_geometry_is_omitted_from_track_run_root(tmp_path) -> None:
    root = zarr.open_group(
        str(tmp_path / "chaser-omission.zarr"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    run = root.create_group("run")
    bundle = ChaserMetricsBundle(
        camera_frame_ids=np.asarray([10, 11], dtype=np.int64),
        stimulus_frame_nums=np.asarray([0, 1], dtype=np.int64),
        timestamp_ns=np.asarray([100, 200], dtype=np.int64),
        trial_state=np.asarray([1, 1], dtype=np.int16),
        metadata_mask=None,
        online={},
        offline={
            "distance_px": np.asarray([2.0, 3.0], dtype=np.float64),
            "distance_mm": np.asarray([0.2, 0.3], dtype=np.float64),
            "fish_centroid_px": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "chaser_position_px": np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            "angle_unsigned_deg": np.asarray([20.0, 30.0]),
            "has_offline": np.asarray([True, True]),
        },
        provenance={"metrics_run": "legacy", "stimulus_run": "stim", "chaser_index": 0},
    )

    metadata = mod._persist_chaser_metrics_to_run(
        run,
        bundle,
        fps=100.0,
        smooth_seconds=0.05,
        distance_interp_seconds=0.2,
    )

    assert metadata["coordinate_geometry_status"] == (
        "omitted_untyped_legacy_chaser_metrics_v1"
    )
    assert metadata["omitted_coordinate_fields"] == [
        "chaser_position_px",
        "distance_mm",
        "distance_px",
        "fish_centroid_px",
    ]
    assert "angle_unsigned_deg" in run
    assert "has_offline" in run
    assert not any(
        name.endswith("_px") or name.endswith("_mm")
        for name in run.array_keys()
    )
    mod._validate_no_run_root_coordinate_arrays(run)

    run.create_array(
        "distance_to_target_mm",
        data=np.asarray([0.2, 0.3], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="unsupported untyped"):
        mod._validate_no_run_root_coordinate_arrays(run)

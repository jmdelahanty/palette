from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import write_columnar_dataset


def _add_goodcopbadcop_cra_protocol_metadata(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    stimulus = root["analysis/stimulus_runs/stimulus_1"]
    stimulus.attrs["protocol_json"] = json.dumps(
        {
            "steps": [
                {
                    "parameters": {
                        "position_transition_duration_s": 0.1,
                        "pre_period_duration_s": 0.3,
                        "training_period_duration_s": 0.3,
                        "post_period_duration_s": 0.3,
                        "pixels_per_mm": 2.0,
                        "chasers": [
                            {
                                "enable_chase": True,
                                "behavior_mode": 0,
                                "color_r": 1.0,
                                "color_g": 0.0,
                                "color_b": 0.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_left",
                                "end_position_preset": "bottom_right",
                            },
                            {
                                "enable_chase": False,
                                "behavior_mode": 1,
                                "color_r": 0.0,
                                "color_g": 0.0,
                                "color_b": 1.0,
                                "color_a": 1.0,
                                "start_position_preset": "top_right",
                                "end_position_preset": "bottom_left",
                            },
                        ],
                    }
                }
            ]
        }
    )
    coords = stimulus.require_group("stimulus_coordinates")
    if "arena_1" in coords:
        del coords["arena_1"]
    arena = coords.create_group("arena_1")
    arena.attrs.update(
        {
            "texture_width_px": 20.0,
            "texture_height_px": 20.0,
            "texture_origin": "top_left",
        }
    )


def _add_goodcopbadcop_swim_bout_run(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    track = root["analysis/track_kinematics_runs/offline/tk_1/tracks/id_0"]
    track.create_array(
        "speed_filtered_mm",
        data=np.asarray(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            dtype=np.float32,
        ),
        chunks=(8,),
        overwrite=True,
    )
    tk_run = root["analysis/track_kinematics_runs/offline/tk_1"]
    tk_run.attrs["fps"] = 10.0
    tk_run.attrs["pixel_to_mm"] = 0.02

    parent = root["analysis"].require_group("swim_bout_runs")
    parent.attrs["latest"] = "bouts_1"
    run = parent.create_group("bouts_1")
    run.attrs.update(
        {
            "default_level": "filtered",
            "source_track_kinematics_run": "tk_1",
            "track_id": 0,
            "detection_method": "threshold",
        }
    )
    level = run.create_group("speed_filtered")
    level.attrs["n_bouts"] = 4

    bouts = np.zeros(
        4,
        dtype=[
            ("bout_id", np.int32),
            ("peak_time_s", np.float64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("duration_s", np.float64),
            ("path_length_mm", np.float64),
        ],
    )
    bouts["bout_id"] = [0, 1, 2, 3]
    bouts["peak_time_s"] = [0.10, 0.20, 0.45, 0.70]
    bouts["start_time_s"] = [0.08, 0.18, 0.42, 0.68]
    bouts["end_time_s"] = [0.12, 0.24, 0.50, 0.76]
    bouts["start_frame"] = [0, 1, 4, 7]
    bouts["end_frame"] = [1, 2, 5, 8]
    bouts["duration_s"] = [0.04, 0.06, 0.08, 0.08]
    bouts["path_length_mm"] = [0.2, 0.3, 0.4, 0.5]
    write_columnar_dataset(level, "bouts", bouts, {"n_bouts": 4})

    intervals = np.zeros(
        2,
        dtype=[
            ("interval_id", np.int32),
            ("valid", bool),
            ("prev_end_time_s", np.float64),
            ("next_start_time_s", np.float64),
            ("interval_s", np.float64),
        ],
    )
    intervals["interval_id"] = [0, 1]
    intervals["valid"] = [True, True]
    intervals["prev_end_time_s"] = [0.12, 0.50]
    intervals["next_start_time_s"] = [0.18, 0.68]
    intervals["interval_s"] = [0.06, 0.18]
    write_columnar_dataset(
        level,
        "inter_bout_intervals",
        intervals,
        {"n_intervals": 2},
    )

from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.analytics_exports.validated_behavior_bout_kinematics import (
    BoutKinematicsExportSourceError,
    validate_bout_kinematics_metric_rows,
    validate_bout_kinematics_run_provenance,
    validate_bout_kinematics_source_refs,
)


_BOUT_DTYPE = np.dtype(
    [
        ("bout_id", "i4"),
        ("start_frame", "i8"),
        ("end_frame", "i8"),
        ("core_start_frame", "i8"),
        ("core_end_frame", "i8"),
    ]
)
_METRIC_DTYPE = np.dtype(
    [
        ("bout_id", "i4"),
        ("source_start_frame", "i8"),
        ("source_end_frame", "i8"),
        ("source_core_start_frame", "i8"),
        ("source_core_end_frame", "i8"),
    ]
)


def _rows() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bouts = np.array([(1, 10, 20, 12, 18), (2, 30, 40, 32, 38)], dtype=_BOUT_DTYPE)
    metric = np.array([(1, 10, 20, 12, 18), (2, 30, 40, 32, 38)], dtype=_METRIC_DTYPE)
    return bouts, {
        name: metric.copy()
        for name in ("movement", "heading_raw", "heading_smoothed", "eye_gaze")
    }


def _refs() -> dict[str, object]:
    return {
        "zarr_path": "/recording/analysis.zarr",
        "source_track_kinematics_run": "track-run",
        "source_track_kinematics_scope": "offline",
        "source_track_kinematics_path": "analysis/track_kinematics_runs/offline/track-run",
        "source_track_kinematics_track_path": (
            "analysis/track_kinematics_runs/offline/track-run/tracks/id_0"
        ),
        "source_track_motion_authority": {"motion_manifest_sha256": "a" * 64},
        "source_swim_bout_track_motion_authority": {"motion_manifest_sha256": "a" * 64},
        "source_swim_bout_run": "bout-run",
        "source_swim_bout_path": (
            "analysis/swim_bout_runs/bout-run/tables/bouts?candidate_id=0&signal_id=4"
        ),
        "source_swim_bout_speed_level": "speed_exponential",
        "source_swim_bout_candidate_id": 0,
        "source_swim_bout_signal_id": 4,
        "source_track_id": 0,
        "source_eye_angle_run": "eye-run",
        "source_eye_angle_path": "analysis/eye_angle_runs/eye-run",
    }


def _expected() -> dict[str, object]:
    return {
        "zarr_path": "/recording/analysis.zarr",
        "track_run": "track-run",
        "track_path": "analysis/track_kinematics_runs/offline/track-run",
        "track_id": 0,
        "track_manifest_sha256": "a" * 64,
        "swim_bout_run": "bout-run",
        "swim_bout_candidate_id": 0,
        "swim_bout_signal_id": 4,
        "swim_bout_speed_level": "speed_exponential",
        "eye_run": "eye-run",
        "eye_path": "analysis/eye_angle_runs/eye-run",
    }


def test_bout_metric_rows_match_each_selected_canonical_bout() -> None:
    bouts, levels = _rows()
    assert validate_bout_kinematics_metric_rows(levels, bouts) == {
        "movement": 2,
        "heading_raw": 2,
        "heading_smoothed": 2,
        "eye_gaze": 2,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda levels: levels.pop("eye_gaze"),
        lambda levels: levels["heading_raw"].__setitem__(
            "bout_id", np.array([1, 1], dtype=np.int32)
        ),
        lambda levels: levels["movement"].__setitem__(
            "source_end_frame", np.array([20, 41], dtype=np.int64)
        ),
        lambda levels: levels.__setitem__(
            "heading_smoothed", levels["heading_smoothed"][:1]
        ),
    ),
)
def test_bout_metric_rows_reject_missing_duplicate_tampered_or_partial_sources(
    mutation: object,
) -> None:
    bouts, levels = _rows()
    mutation(levels)
    with pytest.raises(BoutKinematicsExportSourceError):
        validate_bout_kinematics_metric_rows(levels, bouts)


def test_bout_source_refs_match_the_three_admitted_sources() -> None:
    validate_bout_kinematics_source_refs(_refs(), expected=_expected())


def test_bout_provenance_must_name_the_exact_input_refs_and_producing_code() -> None:
    refs = _refs()
    provenance = {
        "git_sha": "a" * 40,
        "config_hash": "b" * 64,
        "params": {},
        "input_run_ids": refs,
        "command": "produce bout metrics",
        "fisheye_version": "0.1.0",
    }
    validate_bout_kinematics_run_provenance(provenance, refs=refs, git_commit="a" * 40)
    with pytest.raises(BoutKinematicsExportSourceError, match="source inputs"):
        validate_bout_kinematics_run_provenance(
            {**provenance, "input_run_ids": {}}, refs=refs, git_commit="a" * 40
        )
    with pytest.raises(BoutKinematicsExportSourceError, match="producing code"):
        validate_bout_kinematics_run_provenance(
            provenance, refs=refs, git_commit="c" * 40
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("source_swim_bout_signal_id",), 3),
        (("source_swim_bout_path",), None),
        (("source_eye_angle_run",), "other-eye"),
        (("source_track_motion_authority", "motion_manifest_sha256"), "b" * 64),
        (
            ("source_swim_bout_track_motion_authority", "motion_manifest_sha256"),
            "b" * 64,
        ),
        (
            ("source_track_kinematics_path",),
            "analysis/track_kinematics_runs/offline/other",
        ),
    ),
)
def test_bout_source_refs_reject_wrong_upstream_authority(
    path: tuple[str, ...], value: object
) -> None:
    refs = copy.deepcopy(_refs())
    target = refs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(BoutKinematicsExportSourceError):
        validate_bout_kinematics_source_refs(refs, expected=_expected())

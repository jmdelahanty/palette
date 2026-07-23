from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.megabouts_classifier_inputs as classifier_inputs_module
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.analysis.megabouts_classifier_inputs import (
    build_megabouts_classifier_input_pack,
    diagnose_input_pack_invalid_windows,
    summarize_input_pack,
)


def _reason_bytes(values: list[str], width: int = 32) -> np.ndarray:
    out = np.zeros((len(values), width), dtype=np.uint8)
    for idx, value in enumerate(values):
        encoded = value.encode("utf-8")[:width]
        out[idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return out


def _build_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")

    posture_parent = analysis.create_group("tail_posture_view_runs")
    posture_parent.attrs["latest"] = "posture_001"
    posture_parent.attrs["latest_megabouts_compatible"] = "posture_001"
    posture = posture_parent.create_group("posture_001")
    posture.attrs.update(
        {
            "view_family": "megabouts_compatible",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "test_tail_manifest_sha256": "d" * 64,
            "test_source_manifest_sha256": "e" * 64,
        }
    )
    frames = np.arange(8, dtype=np.int64)
    posture.create_array(
        "source_acquisition_frame_index",
        data=frames,
        overwrite=True,
    )
    posture.create_array(
        "instance_key",
        data=np.arange(100, 108, dtype=np.uint64),
        overwrite=True,
    )
    posture.create_array(
        "valid",
        data=np.asarray([True, True, False, True, True, True, True, True], dtype=bool),
        overwrite=True,
    )
    posture.create_array(
        "failure_reason_bytes",
        data=_reason_bytes(["ok", "ok", "source_body_mask_qc_failed", "ok", "ok", "ok", "ok", "ok"]),
        overwrite=True,
    )
    tail_angle = np.arange(8 * 10, dtype=np.float32).reshape(8, 10) / 100.0
    posture.create_array("tail_angle_rad", data=tail_angle, overwrite=True)

    track_parent = analysis.create_group("track_kinematics_runs")
    offline = track_parent.create_group("offline")
    offline.attrs["latest"] = "tk_001"
    track_run = offline.create_group("tk_001")
    track_run.attrs["fps"] = 10.0
    track = track_run.create_group("tracks").create_group("id_0")
    track.create_array("frame_indices", data=frames, overwrite=True)
    track.create_array("source_acquisition_frame_index", data=frames, overwrite=True)
    source_instance_key = np.zeros(
        frames.shape,
        dtype=[("valid", "?"), ("instance_key", "<u8")],
    )
    source_instance_key["valid"] = True
    source_instance_key["instance_key"] = np.arange(100, 108, dtype=np.uint64)
    track.create_array(
        "source_instance_key",
        data=source_instance_key,
        overwrite=True,
    )
    track.create_array(
        "positions_mm",
        data=np.stack([frames.astype(np.float32), frames.astype(np.float32) + 10.0], axis=1),
        overwrite=True,
    )
    track.create_array("smoothed_heading_radians", data=np.linspace(0.0, 0.7, 8, dtype=np.float32), overwrite=True)
    track.create_array(
        "sample_valid",
        data=np.asarray([True, True, True, True, True, False, True, True], dtype=bool),
        overwrite=True,
    )
    track.create_array(
        "sample_reason_code",
        data=np.asarray([0, 0, 0, 0, 0, 3, 0, 0], dtype=np.int16),
        overwrite=True,
    )
    track.create_array(
        "reason_bytes",
        data=_reason_bytes(["ok", "ok", "ok", "ok", "ok", "track_gap", "ok", "ok"]),
        overwrite=True,
    )

    swim_parent = analysis.create_group("swim_bout_runs")
    swim_parent.attrs["latest"] = "bouts_001"
    bout_run = swim_parent.create_group("bouts_001")
    bout_run.attrs.update(
        {
            "default_level": "speed_filtered",
            "source_track_kinematics_run": "tk_001",
            "track_id": 0,
            "source_track_motion_manifest_sha256": "a" * 64,
        }
    )
    level = bout_run.create_group("speed_filtered")
    level.attrs["fps"] = 10.0
    bouts = np.asarray(
        [(11, 1, 3), (12, 4, 6)],
        dtype=[("bout_id", "i4"), ("start_frame", "i8"), ("end_frame", "i8")],
    )
    write_columnar_dataset(level, "bouts", bouts)
    return root


def _install_verified_track_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give downstream tests a typed track reader without faking a full seal."""

    def _load_verified_track(
        root: zarr.Group,
        *,
        run_name: str = "latest",
        scope: str = "offline",
        track_id: int = 0,
        required_speed_levels: tuple[str, ...] = (),
    ) -> SimpleNamespace:
        del required_speed_levels
        offline = root["analysis/track_kinematics_runs"][scope]
        resolved_name = str(offline.attrs["latest"] if run_name == "latest" else run_name)
        run_path = f"analysis/track_kinematics_runs/{scope}/{resolved_name}"
        track_path = f"{run_path}/tracks/id_{int(track_id)}"
        run = root[run_path]
        track = root[track_path]
        return SimpleNamespace(
            run_name=resolved_name,
            scope=scope,
            run_path=run_path,
            track_id=int(track_id),
            track_path=track_path,
            run_attrs=dict(run.attrs),
            track_attrs={
                "sample_reason_codes": {
                    "0": "ok",
                    "1": "unassigned",
                    "2": "source_interpolated",
                    "3": "source_missing",
                    "4": "keypoint_failed",
                    "5": "heading_unusable",
                    "6": "position_nan",
                    "7": "manual_reject",
                }
            },
            authority_status="verified_canonical_track_motion_v1",
            motion_manifest_sha256="a" * 64,
            positions_px_descriptor_sha256="b" * 64,
            positions_mm_descriptor_sha256="c" * 64,
            source_acquisition_frame_index=np.asarray(
                track["source_acquisition_frame_index"][:], dtype=np.int64
            ),
            source_instance_key=(
                np.asarray(track["source_instance_key"][:])
                if "source_instance_key" in track
                else None
            ),
            positions_mm=np.asarray(track["positions_mm"][:], dtype=np.float32),
            heading_degrees=None,
            heading_radians=None,
            smoothed_heading_degrees=None,
            smoothed_heading_radians=np.asarray(
                track["smoothed_heading_radians"][:], dtype=np.float32
            ),
            sample_valid=np.asarray(track["sample_valid"][:], dtype=bool),
            sample_reason_code=np.asarray(
                track["sample_reason_code"][:], dtype=np.int16
            ),
        )

    monkeypatch.setattr(
        classifier_inputs_module,
        "load_track_kinematics_track",
        _load_verified_track,
    )


def _fake_tail_posture_publication(
    root: zarr.Group,
    run_path: str,
) -> SimpleNamespace:
    run = root[run_path]
    if (
        run.attrs.get("palette_run_completion_status") != "complete"
        or run.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError(f"{run_path} is not a complete eligible publication.")
    source_path = "analysis/subject_shape_runs/shape_001"
    return SimpleNamespace(
        run_path=run_path,
        kind="tail_posture_view",
        manifest=SimpleNamespace(
            record_ref=f"{run_path}@tail_coordinate_publication_manifest",
            record_sha256=str(run.attrs["test_tail_manifest_sha256"]),
        ),
        source=SimpleNamespace(
            run_path=source_path,
            manifest=SimpleNamespace(
                record_ref=f"{source_path}@subject_shape_publication_manifest",
                record_sha256=str(run.attrs["test_source_manifest_sha256"]),
            ),
        ),
    )


def _install_verified_source_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_track_reader(monkeypatch)
    monkeypatch.setattr(
        classifier_inputs_module,
        "load_tail_posture_coordinate_publication",
        _fake_tail_posture_publication,
    )


@pytest.fixture(autouse=True)
def _verified_source_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_verified_source_readers(monkeypatch)


def _replace_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    del group[name]
    group.create_array(name, data=np.asarray(values), overwrite=True)


def test_build_megabouts_classifier_input_pack_resolves_sources_and_shapes() -> None:
    root = _build_root()

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=4,
        min_tail_valid_fraction=0.75,
        min_traj_valid_fraction=0.75,
        max_consecutive_invalid_frames=1,
    )

    assert pack.tail_array.shape == (2, 10, 4)
    assert pack.traj_array.shape == (2, 3, 4)
    assert pack.source_bout_id.tolist() == [11, 12]
    assert pack.window_start_frame.tolist() == [1, 4]
    assert pack.window_end_frame.tolist() == [4, 7]
    np.testing.assert_allclose(pack.tail_array[0, :, 0], np.arange(10, 20, dtype=np.float32) / 100.0)
    assert pack.tail_valid.tolist() == [
        [True, False, True, True],
        [True, True, True, True],
    ]
    assert pack.traj_valid.tolist() == [
        [True, True, True, True],
        [True, False, True, True],
    ]
    assert pack.traj_reference_valid.tolist() == [True, True]
    theta0 = 0.1
    offsets = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(
        pack.traj_array[0, 0, :],
        np.cos(theta0) * offsets + np.sin(theta0) * offsets,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        pack.traj_array[0, 1, :],
        -np.sin(theta0) * offsets + np.cos(theta0) * offsets,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(pack.traj_array[0, 2, :], [0.0, 0.1, 0.2, 0.3], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pack.tail_valid_fraction, [0.75, 1.0])
    np.testing.assert_allclose(pack.traj_valid_fraction, [1.0, 0.75])
    assert pack.valid_bout.tolist() == [True, True]
    assert pack.failure_reason.tolist() == ["ok", "ok"]
    assert pack.parameters["window_policy"] == "start_frame_fixed_duration"
    assert pack.parameters["traj_alignment"] == "onset_translation_rotation"
    assert pack.parameters["traj_reference_index"] == 0
    assert pack.parameters["calls_megabouts"] is False
    assert pack.source_refs["tail_angle_rad"].endswith("/tail_angle_rad")
    assert pack.source_refs["tail_frame_indices"].endswith(
        "/source_acquisition_frame_index"
    )
    assert pack.source_refs["tail_instance_key"].endswith("/instance_key")
    assert pack.source_refs["track_source_instance_key"].endswith(
        "/source_instance_key"
    )
    assert pack.parameters["tail_track_join_policy"].startswith(
        "posture_instance_key_to_track_source_instance_key"
    )
    assert pack.source_refs["tail_posture_publication_manifest_sha256"] == "d" * 64
    assert pack.source_refs["tail_posture_publication_manifest_ref"] == (
        "analysis/tail_posture_view_runs/posture_001"
        "@tail_coordinate_publication_manifest"
    )
    assert pack.source_refs["tail_posture_source_subject_shape_run"] == (
        "analysis/subject_shape_runs/shape_001"
    )
    assert pack.source_refs[
        "tail_posture_source_subject_shape_publication_manifest_ref"
    ] == (
        "analysis/subject_shape_runs/shape_001"
        "@subject_shape_publication_manifest"
    )
    assert (
        pack.source_refs[
            "tail_posture_source_subject_shape_publication_manifest_sha256"
        ]
        == "e" * 64
    )


def test_build_joins_posture_by_instance_key_when_frame_has_multiple_instances() -> None:
    root = _build_root()
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    original_angles = np.asarray(posture["tail_angle_rad"][:])
    original_valid = np.asarray(posture["valid"][:])
    original_reasons = np.asarray(posture["failure_reason_bytes"][:])
    wrong_animal_angles = np.full((1, 10), 99.0, dtype=np.float32)

    _replace_array(
        posture,
        "source_acquisition_frame_index",
        np.asarray([1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
    )
    _replace_array(
        posture,
        "instance_key",
        np.asarray([999, 100, 101, 102, 103, 104, 105, 106, 107], dtype=np.uint64),
    )
    _replace_array(
        posture,
        "tail_angle_rad",
        np.concatenate([wrong_animal_angles, original_angles], axis=0),
    )
    _replace_array(
        posture,
        "valid",
        np.concatenate([np.asarray([True]), original_valid]),
    )
    _replace_array(
        posture,
        "failure_reason_bytes",
        np.concatenate([_reason_bytes(["wrong_animal"]), original_reasons], axis=0),
    )

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=1,
        min_tail_valid_fraction=0.0,
        min_traj_valid_fraction=0.0,
    )

    np.testing.assert_allclose(pack.tail_array[0, :, 0], original_angles[1])
    assert not np.all(pack.tail_array[0, :, 0] == 99.0)


def test_build_marks_unmatched_posture_instance_key_invalid_without_frame_fallback() -> None:
    root = _build_root()
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    keys = np.asarray(posture["instance_key"][:])
    keys[1] = np.uint64(901)
    _replace_array(posture, "instance_key", keys)

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=1,
        min_tail_valid_fraction=0.0,
        min_traj_valid_fraction=0.0,
    )

    assert pack.tail_valid[0].tolist() == [False]
    assert np.isnan(pack.tail_array[0, :, 0]).all()
    assert pack.traj_valid[0].tolist() == [True]


@pytest.mark.parametrize("missing_surface", ["posture", "track"])
def test_build_rejects_missing_instance_identity_surface(missing_surface: str) -> None:
    root = _build_root()
    if missing_surface == "posture":
        del root["analysis/tail_posture_view_runs/posture_001/instance_key"]
    else:
        del root[
            "analysis/track_kinematics_runs/offline/tk_001/tracks/id_0/"
            "source_instance_key"
        ]

    with pytest.raises(ValueError, match="instance_key"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=1)


def test_build_rejects_malformed_track_source_instance_identity() -> None:
    root = _build_root()
    track = root[
        "analysis/track_kinematics_runs/offline/tk_001/tracks/id_0"
    ]
    _replace_array(
        track,
        "source_instance_key",
        np.arange(100, 108, dtype=np.uint64),
    )

    with pytest.raises(ValueError, match="canonical nullable dtype"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=1)


def test_build_rejects_matching_instance_key_with_different_acquisition_frame() -> None:
    root = _build_root()
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    frames = np.asarray(posture["source_acquisition_frame_index"][:])
    frames[1] = 2
    _replace_array(posture, "source_acquisition_frame_index", frames)

    with pytest.raises(ValueError, match="acquisition-frame mismatch"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=1)


@pytest.mark.parametrize(
    ("lineage_attr", "replacement", "expected"),
    [
        ("source_track_kinematics_run", None, "lacks source_track_kinematics_run"),
        ("source_track_kinematics_run", "tk_other", "run mismatch"),
        ("track_id", None, "lacks track_id lineage"),
        ("track_id", 7, "track_id mismatch"),
        (
            "source_track_motion_manifest_sha256",
            None,
            "lacks source_track_motion_manifest_sha256",
        ),
        (
            "source_track_motion_manifest_sha256",
            "f" * 64,
            "manifest_sha256 mismatch",
        ),
    ],
)
def test_build_rejects_absent_or_mismatched_swim_bout_track_lineage(
    lineage_attr: str,
    replacement: object,
    expected: str,
) -> None:
    root = _build_root()
    bout_run = root["analysis/swim_bout_runs/bouts_001"]
    if replacement is None:
        del bout_run.attrs[lineage_attr]
    else:
        bout_run.attrs[lineage_attr] = replacement

    with pytest.raises(ValueError, match=expected):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=1)


def test_build_megabouts_classifier_input_pack_reports_invalid_coverage() -> None:
    root = _build_root()

    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)
    summary = summarize_input_pack(pack)

    assert pack.valid_bout.tolist() == [False, False]
    assert summary["valid_bout_count"] == 0
    assert summary["invalid_bout_count"] == 2
    assert summary["failure_reason_counts"] == {
        "tail_valid_fraction_below_threshold": 1,
        "traj_valid_fraction_below_threshold": 1,
    }


def test_diagnose_input_pack_invalid_windows_reports_source_causes() -> None:
    root = _build_root()
    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)

    report = diagnose_input_pack_invalid_windows(root, pack, max_examples=1)

    assert report["diagnostic"] == "megabouts_classifier_invalid_windows"
    assert report["mutates_archive"] is False
    assert report["calls_megabouts"] is False
    assert report["invalid_bout_count"] == 2
    assert report["tail_frame_issue_counts_across_invalid_windows"] == {"posture_valid_false": 1}
    assert report["traj_frame_issue_counts_across_invalid_windows"] == {"track_sample_valid_false": 1}
    assert report["posture_failure_reason_counts_across_invalid_frames"] == {"source_body_mask_qc_failed": 1}
    assert report["track_failure_reason_counts_across_invalid_frames"] == {"source_missing": 1}
    assert len(report["examples"]) == 1
    example = report["examples"][0]
    assert example["bout_index"] == 0
    assert example["invalid_posture_frames"] == [2]
    assert example["missing_track_frames"] == []


def test_build_megabouts_classifier_input_pack_can_resolve_time_only_bouts() -> None:
    root = _build_root()
    level = root["analysis/swim_bout_runs/bouts_001/speed_filtered"]
    bouts = np.asarray(
        [(21, 0.1, 0.3)],
        dtype=[("bout_id", "i4"), ("start_time_s", "f8"), ("end_time_s", "f8")],
    )
    write_columnar_dataset(level, "bouts", bouts)

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_s=0.2,
        min_tail_valid_fraction=0.5,
        min_traj_valid_fraction=0.5,
    )

    assert pack.tail_array.shape == (1, 10, 2)
    assert pack.source_start_frame.tolist() == [1]
    assert pack.source_end_frame.tolist() == [3]
    assert pack.window_end_frame.tolist() == [2]


def test_build_megabouts_classifier_input_pack_rejects_wrong_tail_segment_count() -> None:
    root = _build_root()
    posture = root["analysis/tail_posture_view_runs/posture_001"]
    del posture["tail_angle_rad"]
    posture.create_array("tail_angle_rad", data=np.zeros((8, 9), dtype=np.float32), overwrite=True)

    with pytest.raises(ValueError, match="10 tail-angle channels"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=4)


def test_build_megabouts_classifier_input_pack_rejects_windows_too_long_for_megabouts() -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="capped at 140 frames"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=141)


def test_build_megabouts_classifier_input_pack_rejects_invalid_trajectory_reference() -> None:
    root = _build_root()
    level = root["analysis/swim_bout_runs/bouts_001/speed_filtered"]
    bouts = np.asarray(
        [(31, 5, 7)],
        dtype=[("bout_id", "i4"), ("start_frame", "i8"), ("end_frame", "i8")],
    )
    write_columnar_dataset(level, "bouts", bouts)

    pack = build_megabouts_classifier_input_pack(
        root,
        bout_duration_frames=3,
        min_tail_valid_fraction=0.5,
        min_traj_valid_fraction=0.5,
        max_consecutive_invalid_frames=2,
    )

    assert pack.traj_reference_valid.tolist() == [False]
    assert pack.valid_bout.tolist() == [False]
    assert pack.failure_reason.tolist() == ["traj_reference_invalid"]


def test_latest_posture_fails_closed_on_ineligible_family_selector() -> None:
    root = _build_root()
    parent = root["analysis/tail_posture_view_runs"]
    staged = parent.create_group("posture_zzz_staged")
    staged.attrs.update(
        {
            "view_family": "megabouts_compatible",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "test_tail_manifest_sha256": "1" * 64,
            "test_source_manifest_sha256": "2" * 64,
        }
    )
    distractor = parent.create_group("zzzz_other_family")
    distractor.attrs.update(
        {
            "view_family": "other",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "test_tail_manifest_sha256": "3" * 64,
            "test_source_manifest_sha256": "4" * 64,
        }
    )
    parent.attrs["latest"] = "posture_zzz_staged"
    parent.attrs["latest_megabouts_compatible"] = "posture_zzz_staged"

    with pytest.raises(ValueError, match="do not guess a prior run"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=4)


@pytest.mark.parametrize(
    "selector",
    (" posture_001", "posture_001 ", "posture_001/"),
)
def test_latest_posture_rejects_noncanonical_family_selector(
    selector: str,
) -> None:
    root = _build_root()
    root["analysis/tail_posture_view_runs"].attrs[
        "latest_megabouts_compatible"
    ] = selector

    with pytest.raises(ValueError, match="family selector.*invalid"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=4)


@pytest.mark.parametrize(
    "selector",
    [
        "/analysis/tail_posture_view_runs/posture_001",
        "analysis/tail_posture_view_runs/posture_001/",
        "analysis//tail_posture_view_runs/posture_001",
    ],
)
def test_tail_posture_selector_rejects_nonexact_path(selector: str) -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="exact canonical path"):
        build_megabouts_classifier_input_pack(
            root,
            tail_posture_view_run=selector,
            bout_duration_frames=1,
        )


@pytest.mark.parametrize(
    "selector",
    [
        "/analysis/track_kinematics_runs/offline/tk_001",
        "analysis/track_kinematics_runs/offline/tk_001/",
        "analysis//track_kinematics_runs/offline/tk_001",
        "analysis/track_kinematics_runs/offline/tk_001/tracks/id_0",
        "offline/tk_001/extra",
        "wrong/prefix/offline/tk_001",
    ],
)
def test_track_selector_rejects_nonexact_or_nested_path(selector: str) -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="exact canonical path|must exactly equal|must be a bare"):
        build_megabouts_classifier_input_pack(
            root,
            track_kinematics_run=selector,
            bout_duration_frames=1,
        )


def test_track_selector_accepts_exact_canonical_path() -> None:
    root = _build_root()

    pack = build_megabouts_classifier_input_pack(
        root,
        track_kinematics_run=(
            "analysis/track_kinematics_runs/offline/tk_001"
        ),
        bout_duration_frames=1,
    )

    assert pack.source_refs["track_kinematics_run"] == (
        "analysis/track_kinematics_runs/offline/tk_001"
    )


def test_explicit_posture_must_be_eligible_and_in_megabouts_family() -> None:
    root = _build_root()
    parent = root["analysis/tail_posture_view_runs"]
    staged = parent.create_group("posture_staged")
    staged.attrs.update(
        {
            "view_family": "megabouts_compatible",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    wrong_family = parent.create_group("posture_other")
    wrong_family.attrs.update(
        {
            "view_family": "other",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    with pytest.raises(ValueError, match="complete and selector-eligible"):
        build_megabouts_classifier_input_pack(
            root,
            tail_posture_view_run="posture_staged",
            bout_duration_frames=4,
        )
    with pytest.raises(ValueError, match="run from family"):
        build_megabouts_classifier_input_pack(
            root,
            tail_posture_view_run=(
                "analysis/tail_posture_view_runs/posture_other"
            ),
            bout_duration_frames=4,
        )


@pytest.mark.parametrize("tamper_target", ["tail_manifest", "source_manifest"])
def test_build_rejects_posture_publication_changed_during_copy(
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    root = _build_root()
    call_count = 0

    def _changing_publication(
        source_root: zarr.Group,
        run_path: str,
    ) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        publication = _fake_tail_posture_publication(source_root, run_path)
        if call_count == 1:
            return publication
        manifest = publication.manifest
        source = publication.source
        if tamper_target == "tail_manifest":
            manifest = SimpleNamespace(
                record_ref=publication.manifest.record_ref,
                record_sha256="f" * 64,
            )
        else:
            source = SimpleNamespace(
                run_path=publication.source.run_path,
                manifest=SimpleNamespace(
                    record_ref=publication.source.manifest.record_ref,
                    record_sha256="f" * 64,
                ),
            )
        return SimpleNamespace(
            **{
                **publication.__dict__,
                "manifest": manifest,
                "source": source,
            }
        )

    monkeypatch.setattr(
        classifier_inputs_module,
        "load_tail_posture_coordinate_publication",
        _changing_publication,
    )

    with pytest.raises(ValueError, match="changed while Megabouts inputs were copied"):
        build_megabouts_classifier_input_pack(root, bout_duration_frames=4)


def test_diagnostic_rejects_posture_publication_changed_since_pack() -> None:
    root = _build_root()
    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)
    root["analysis/tail_posture_view_runs/posture_001"].attrs[
        "test_tail_manifest_sha256"
    ] = "f" * 64

    with pytest.raises(ValueError, match="changed since the input pack was built"):
        diagnose_input_pack_invalid_windows(root, pack)


def test_diagnostic_revalidates_posture_publication_after_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    pack = build_megabouts_classifier_input_pack(root, bout_duration_frames=4)
    call_count = 0

    def _changing_publication(
        source_root: zarr.Group,
        run_path: str,
    ) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        publication = _fake_tail_posture_publication(source_root, run_path)
        if call_count == 1:
            return publication
        return SimpleNamespace(
            **{
                **publication.__dict__,
                "manifest": SimpleNamespace(
                    record_ref=publication.manifest.record_ref,
                    record_sha256="f" * 64,
                ),
            }
        )

    monkeypatch.setattr(
        classifier_inputs_module,
        "load_tail_posture_coordinate_publication",
        _changing_publication,
    )

    with pytest.raises(ValueError, match="changed while Megabouts diagnostics"):
        diagnose_input_pack_invalid_windows(root, pack)

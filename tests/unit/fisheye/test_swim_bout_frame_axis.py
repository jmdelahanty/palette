from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_STORAGE_EMBEDDED,
    FRAME_AXIS_STORAGE_REFERENCE,
    SwimBoutFrameAxisError,
    build_frame_axis_contract,
    canonical_frame_axis_sha256,
    resolve_swim_bout_frame_axis,
)

SOURCE_PATH = (
    "analysis/track_kinematics_runs/offline/tk_exact/tracks/id_0/"
    "source_acquisition_frame_index"
)
LEGACY_SOURCE_PATH = (
    "analysis/track_kinematics_runs/offline/tk_exact/tracks/id_0/frame_indices"
)
MOTION_MANIFEST_SHA256 = "a" * 64


def _root_with_run(*, source_values: np.ndarray | None = None):
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    run = parent.create_group("bouts")
    run.attrs.update(
        {
            "source_track_kinematics_run": "tk_exact",
            "track_id": 0,
            "source_track_motion_manifest_sha256": MOTION_MANIFEST_SHA256,
        }
    )
    signals = run.create_group("signals")
    if source_values is not None:
        track = (
            analysis.create_group("track_kinematics_runs")
            .create_group("offline")
            .create_group("tk_exact")
            .create_group("tracks")
            .create_group("id_0")
        )
        track.create_array("source_acquisition_frame_index", data=source_values)
    return root, run, signals


def test_reference_contract_pins_exact_authoritative_path() -> None:
    frames = np.asarray([0, 1, 5, 6], dtype=np.int64)

    contract = build_frame_axis_contract(
        frames,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
    )

    assert contract["storage_mode"] == FRAME_AXIS_STORAGE_REFERENCE
    assert contract["authoritative_path"] == SOURCE_PATH
    assert contract["identity_array_role"] == "source_acquisition_frame_index"
    assert contract["source_track_motion_manifest_sha256"] == (
        MOTION_MANIFEST_SHA256
    )
    assert contract["embedded_path"] is None
    assert contract["shape"] == [4]
    assert contract["content_sha256"] == canonical_frame_axis_sha256(frames)


def test_reference_contract_rejects_latest_pointer() -> None:
    with pytest.raises(SwimBoutFrameAxisError, match="not 'latest'"):
        build_frame_axis_contract(
            np.arange(3, dtype=np.int64),
            authoritative_path=(
                "analysis/track_kinematics_runs/offline/latest/"
                "tracks/id_0/source_acquisition_frame_index"
            ),
            source_track_kinematics_run="latest",
            track_id=0,
            source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        )


def test_resolver_prefers_authority_over_embedded_fallback() -> None:
    authoritative = np.asarray([0, 2, 4], dtype=np.int64)
    root, run, signals = _root_with_run(source_values=authoritative)
    signals.create_array(
        "frame_indices",
        data=np.asarray([100, 101, 102], dtype=np.int64),
    )
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        authoritative,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        storage_mode=FRAME_AXIS_STORAGE_EMBEDDED,
    )

    resolved = resolve_swim_bout_frame_axis(root, run, expected_length=3)

    np.testing.assert_array_equal(resolved, authoritative)


def test_resolver_uses_declared_embedded_fallback_when_authority_is_absent() -> None:
    frames = np.asarray([0, 3, 4], dtype=np.int64)
    root, run, signals = _root_with_run()
    signals.create_array("frame_indices", data=frames)
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        frames,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        storage_mode=FRAME_AXIS_STORAGE_EMBEDDED,
    )

    resolved = resolve_swim_bout_frame_axis(root, run, expected_length=3)

    np.testing.assert_array_equal(resolved, frames)


def test_resolver_keeps_historical_embedded_compact_v2_readable() -> None:
    frames = np.asarray([7, 8, 11], dtype=np.int64)
    root, run, signals = _root_with_run()
    signals.create_array("frame_indices", data=frames)

    resolved = resolve_swim_bout_frame_axis(root, run, expected_length=3)

    np.testing.assert_array_equal(resolved, frames)


def test_resolver_accepts_explicit_historical_v1_frame_indices_rule() -> None:
    frames = np.asarray([2, 4, 8], dtype=np.int64)
    root, run, _signals = _root_with_run(source_values=frames)
    track = root[
        "analysis/track_kinematics_runs/offline/tk_exact/tracks/id_0"
    ]
    track.create_array("frame_indices", data=frames)
    contract = build_frame_axis_contract(
        frames,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
    )
    contract["schema_version"] = 1
    contract["authoritative_path"] = LEGACY_SOURCE_PATH
    del contract["identity_array_role"]
    del contract["source_track_motion_manifest_sha256"]
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = contract

    resolved = resolve_swim_bout_frame_axis(root, run, expected_length=3)

    np.testing.assert_array_equal(resolved, frames)


def test_resolver_rejects_changed_authoritative_payload() -> None:
    original = np.asarray([0, 1, 2], dtype=np.int64)
    root, run, _signals = _root_with_run(source_values=original)
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        original,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
    )
    root[SOURCE_PATH][:] = np.asarray([0, 1, 3], dtype=np.int64)

    with pytest.raises(SwimBoutFrameAxisError, match="payload digest mismatch"):
        resolve_swim_bout_frame_axis(root, run, expected_length=3)


def test_resolver_fails_closed_on_authoritative_shape_mismatch() -> None:
    source_values = np.asarray([0, 1], dtype=np.int64)
    declared_values = np.asarray([0, 1, 2], dtype=np.int64)
    root, run, signals = _root_with_run(source_values=source_values)
    signals.create_array("frame_indices", data=declared_values)
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        declared_values,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        storage_mode=FRAME_AXIS_STORAGE_EMBEDDED,
    )

    with pytest.raises(SwimBoutFrameAxisError, match="shape mismatch"):
        resolve_swim_bout_frame_axis(root, run, expected_length=3)


def test_reference_only_contract_fails_when_authority_is_missing() -> None:
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    root, run, _signals = _root_with_run()
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        frames,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        storage_mode=FRAME_AXIS_STORAGE_REFERENCE,
    )

    with pytest.raises(SwimBoutFrameAxisError, match="no embedded fallback"):
        resolve_swim_bout_frame_axis(root, run, expected_length=3)


def test_embedded_fallback_dtype_must_match_contract() -> None:
    frames = np.asarray([0, 1, 2], dtype=np.int64)
    root, run, signals = _root_with_run()
    signals.create_array("frame_indices", data=frames.astype(np.int32))
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = build_frame_axis_contract(
        frames,
        authoritative_path=SOURCE_PATH,
        source_track_kinematics_run="tk_exact",
        track_id=0,
        source_track_motion_manifest_sha256=MOTION_MANIFEST_SHA256,
        storage_mode=FRAME_AXIS_STORAGE_EMBEDDED,
    )

    with pytest.raises(SwimBoutFrameAxisError, match="dtype mismatch"):
        resolve_swim_bout_frame_axis(root, run, expected_length=3)

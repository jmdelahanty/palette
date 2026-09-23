"""Camera recording IDs come from the acquisition binding, never session_uuid.

A camera recording ID is hash(acquisition session, camera). In a unified H5 the
Orange acquisition session is ``/correspondence/acquisition/binding_json``
``recording_id``; ``/metadata/session@session_uuid`` names one Citrus Arena
session and must not be substituted for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from fisheye.shared.source_recording_identity import (
    SourceRecordingIdentityError,
    recording_id_from_session_camera,
)
from fisheye.shared.unified_h5.correspondence import BINDING
from fisheye.utils import organize_recordings
from tests.unit.fisheye.unified_h5_fixtures import emit_bound_h5

ACQUISITION_SESSION = "synthetic-paired-recording"
CAMERA = "CAM-42"


def _plan(h5_path: Path, tmp_path: Path):
    return organize_recordings._build_plan(
        h5_path,
        dest_root=tmp_path / "recordings",
        cam_root=None,
        rename_cams=True,
    )


def _arena_session(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as h5:
        return str(h5["/metadata/session"].attrs["session_uuid"])


def test_bound_h5_mints_from_acquisition_session_not_arena_session(
    tmp_path: Path,
) -> None:
    h5_path = emit_bound_h5(tmp_path / "arena_1.h5")
    arena_session = _arena_session(h5_path)
    assert arena_session.startswith("citsess_")

    plan = _plan(h5_path, tmp_path)

    assert plan.meta["session_uuid"] == ACQUISITION_SESSION
    assert plan.meta["recording_id"] == recording_id_from_session_camera(
        session_uuid=ACQUISITION_SESSION, camera_id=CAMERA
    )
    assert plan.meta["recording_id"] != recording_id_from_session_camera(
        session_uuid=arena_session, camera_id=CAMERA
    )


def test_one_camera_serving_two_arenas_is_one_camera_recording(
    tmp_path: Path,
) -> None:
    first = emit_bound_h5(tmp_path / "a" / "arena_1.h5")
    second = emit_bound_h5(tmp_path / "b" / "arena_2.h5")
    with h5py.File(second, "r+") as h5:
        h5["/metadata/session"].attrs["session_uuid"] = "citsess_second_arena"
    assert _arena_session(first) != _arena_session(second)

    first_plan = _plan(first, tmp_path / "a")
    second_plan = _plan(second, tmp_path / "b")

    assert first_plan.meta["recording_id"] == second_plan.meta["recording_id"]


def test_different_acquisition_sessions_are_different_recordings(
    tmp_path: Path,
) -> None:
    first = _plan(emit_bound_h5(tmp_path / "a" / "base.h5"), tmp_path / "a")
    other = _plan(
        emit_bound_h5(tmp_path / "b" / "other.h5", name="frames_only"),
        tmp_path / "b",
    )

    assert other.meta["session_uuid"] == "recording-42"
    assert first.meta["recording_id"] != other.meta["recording_id"]


def test_camera_conflicting_with_binding_is_refused(tmp_path: Path) -> None:
    h5_path = emit_bound_h5(
        tmp_path / "arena_1.h5", root_attrs={"camera_id": "2010093"}
    )

    with pytest.raises(SourceRecordingIdentityError, match="acquisition binding"):
        _plan(h5_path, tmp_path)


def test_tampered_binding_is_refused_without_fallback(tmp_path: Path) -> None:
    h5_path = emit_bound_h5(
        tmp_path / "arena_1.h5", root_attrs={"session_uuid": "session_arena_1"}
    )
    with h5py.File(h5_path, "r+") as h5:
        dataset = h5[BINDING]
        attrs, dtype = dict(dataset.attrs), dataset.dtype
        binding = json.loads(dataset[()])
        binding["recording_id"] = "tampered-session"
        del h5[BINDING]
        replacement = h5.create_dataset(
            BINDING,
            data=json.dumps(binding, sort_keys=True, separators=(",", ":")).encode(),
            dtype=dtype,
        )
        for key, value in attrs.items():
            replacement.attrs[key] = value

    with pytest.raises(SourceRecordingIdentityError, match="invalid acquisition"):
        _plan(h5_path, tmp_path)

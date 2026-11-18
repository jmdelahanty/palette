"""Derived per-recording orientation status (palette#377 §5, §10)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

from fisheye.shared.unified_h5 import orientation_status as mod
from fisheye.utils import report_orientation_status as cli
from tests.unit.fisheye.test_unified_h5_declared_orientation import (
    HOMOGRAPHY,
    _omnifin0,
    _write_h5,
)

PRE_CUTOFF = "2026-10-17T16-06-43Z"
SESSION = "citsess_" + "1" * 64


def _write(path: Path, *, session_uuid: str = SESSION, **kwargs) -> Path:
    kwargs.setdefault("record", None)
    kwargs.setdefault("session_start", PRE_CUTOFF)
    _write_h5(path, **kwargs)
    with h5py.File(path, "a") as h5:
        h5["metadata/session"].attrs["session_uuid"] = session_uuid
    return path


def _digest(matrix: np.ndarray) -> str:
    import hashlib

    return (
        "sha256:"
        + hashlib.sha256(
            np.ascontiguousarray(matrix.astype("<f8")).tobytes()
        ).hexdigest()
    )


@pytest.fixture
def listed(monkeypatch: pytest.MonkeyPatch):
    """Evidence that lists SESSION with the test homography."""

    real = json.loads(mod._evidence_bytes())
    record = real["evidence"]["omnifin0_20261115_static_dots_nir_filters_off"]
    record["recordings"] = [
        {
            "recording": "test_Cam2010093",
            "session_uuid": SESSION,
            "arena_id": "arena_1",
            "camera_serial": "2010093",
            "homography_matrix_sha256": _digest(HOMOGRAPHY),
        }
    ]
    payload = json.dumps(real).encode()
    monkeypatch.setattr(mod, "_evidence_bytes", lambda: payload)
    mod.evidence.cache_clear()
    yield
    mod.evidence.cache_clear()


def test_post_cutoff_record_that_passes_is_verified(tmp_path: Path) -> None:
    status = mod.resolve_orientation_status(
        _write(
            tmp_path / "a.h5", record=_omnifin0(), session_start="2026-11-16T10-00-00Z"
        )
    )
    assert status["orientation_status"] == mod.VERIFIED
    assert status["verified_link"] == "camera_to_canvas"
    assert status["basis"] == mod.BASIS_CITRUS_CHECK
    assert status["camera_to_dish_parity"] == "declared"
    assert status["recorded_space_parity_basis"]["viewing_side"] == "dorsal"
    assert status["session_uuid"] == SESSION


def test_pre_cutoff_record_is_still_checked_and_counts(tmp_path: Path) -> None:
    status = mod.resolve_orientation_status(
        _write(tmp_path / "a.h5", record=_omnifin0())
    )
    assert status["orientation_status"] == mod.VERIFIED
    assert status["basis"] == mod.BASIS_CITRUS_CHECK


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (
            {"record": None, "session_start": "2026-11-16T10-00-00Z"},
            "declared_orientation_missing",
        ),
        (
            {
                "record": _omnifin0(),
                "homography": np.array([[0.08, 0, 100], [0, 0.08, 10], [0, 0, 1.0]]),
            },
            "declared_orientation_determinant_mismatch",
        ),
        (
            {
                "record": None,
                "session_start": "2026-11-16T10-00-00Z",
                "hostname": "other.hhmi.org",
            },
            "declared_orientation_cutoff_host_mismatch",
        ),
    ],
)
def test_refused_citrus_record_is_unverified_not_raised(
    tmp_path: Path, kwargs: dict, reason: str
) -> None:
    status = mod.resolve_orientation_status(_write(tmp_path / "a.h5", **kwargs))
    assert status["orientation_status"] == mod.UNVERIFIED
    assert status["basis"] is None
    assert status["reason"] == f"declared_orientation_refused:{reason}"


def test_pre_cutoff_recording_without_evidence_is_unverified(tmp_path: Path) -> None:
    status = mod.resolve_orientation_status(_write(tmp_path / "a.h5"))
    assert status["orientation_status"] == mod.UNVERIFIED
    assert status["reason"] == "no_orientation_evidence"
    assert status["verified_link"] is None


def test_listed_recording_is_verified_by_the_measurement(
    tmp_path: Path, listed
) -> None:
    status = mod.resolve_orientation_status(_write(tmp_path / "a.h5"))
    assert status["orientation_status"] == mod.VERIFIED
    assert status["basis"] == "omnifin0_20261115_static_dots_nir_filters_off"
    assert status["evidence_sha256"].startswith("sha256:")
    assert status["camera_to_dish_parity"] == "declared"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda h5: h5["metadata/session"].attrs.__setitem__("arena_id", "arena_2"),
            "arena_id",
        ),
        (
            lambda h5: h5["metadata/session"].attrs.__setitem__("rig_id", "omnifin1"),
            "rig_id",
        ),
        (
            lambda h5: h5["geometry/correspondence/input"].attrs.__setitem__(
                "camera_serial", "2010094"
            ),
            "camera_serial",
        ),
        (
            lambda h5: h5[
                "geometry/calibration/2010093/homography_matrix"
            ].write_direct((HOMOGRAPHY + np.diag([0, 1e-12, 0])).astype("<f8")),
            "homography_matrix_sha256",
        ),
    ],
)
def test_listed_recording_must_match_its_evidence(
    tmp_path: Path, listed, mutate, reason: str
) -> None:
    path = _write(tmp_path / "a.h5")
    with h5py.File(path, "a") as h5:
        mutate(h5)
    status = mod.resolve_orientation_status(path)
    assert status["orientation_status"] == mod.UNVERIFIED
    assert status["reason"].startswith("orientation_evidence_mismatch:")
    assert reason in status["reason"]


def test_evidence_lists_exactly_the_10_17_18_phototaxis_sessions() -> None:
    mod.evidence.cache_clear()
    _sha, index = mod.evidence()
    assert len(index) == 24
    entries = list(index.values())
    assert {entry["evidence_id"] for entry in entries} == {
        "omnifin0_20261115_static_dots_nir_filters_off"
    }
    assert {entry["rig_id"] for entry in entries} == {"omnifin0"}
    sessions = Counter(entry["recording"].rsplit("_Cam", 1)[0] for entry in entries)
    assert sessions == {
        name: 4
        for name in (
            "2026_10_17_12_06_15",
            "2026_10_17_14_09_08",
            "2026_10_17_15_57_07",
            "2026_10_17_17_45_57",
            "2026_10_17_19_38_38",
            "2026_10_18_11_32_33",
        )
    }
    # One sealed matrix per arena across all six sessions, as measured.
    per_arena: dict[str, set[str]] = {}
    for entry in entries:
        assert entry["recording"].endswith("_Cam" + entry["camera_serial"])
        per_arena.setdefault(entry["arena_id"], set()).add(
            entry["homography_matrix_sha256"]
        )
    assert sorted(per_arena) == ["arena_1", "arena_2", "arena_3", "arena_4"]
    assert all(len(digests) == 1 for digests in per_arena.values())


def test_report_resolves_a_recording_folder(tmp_path: Path, listed) -> None:
    recording = tmp_path / "2026_10_17_12_06_15_Cam2010093"
    (recording / "raw/acquisition/citrus").mkdir(parents=True)
    _write(recording / "raw/acquisition/citrus/citsess_x.h5")
    row = cli.report(recording)
    assert row["orientation_status"] == mod.VERIFIED
    assert row["source"].endswith("citsess_x.h5")


def test_report_without_a_citrus_h5_is_unverified(tmp_path: Path, capsys) -> None:
    assert cli.main([str(tmp_path)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["orientation_status"] == mod.UNVERIFIED
    assert row["reason"] == "no_unified_citrus_h5"

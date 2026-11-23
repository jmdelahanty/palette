"""dish_design of unified recordings, from Orange's recording geometry contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.recording_dish_design import (
    GEOMETRY_CONTRACT_RELATIVE,
    recording_dish_design,
)
from fisheye.shared.recording_geometry import RecordingGeometryError

from tests.unit.fisheye import test_current_clipped_recording_import as clipped


def _camera(design: str | None = "palm1", *, rim: str | None = None) -> dict:
    rim = design if rim is None else rim
    camera: dict = {
        "daily_registration_geometry": {
            "recording_snapshot_entry": {
                "accepted_inner_rim_boundary": {"physical_geometry": {"dish_design_id": rim}},
                "accepted_experimental_area_boundary": {"physical_geometry": {"dish_design_id": rim}},
            }
        }
    }
    if design is not None:
        camera["tank_design"] = {
            "owner": "selected_canvas", "status": "referenced", "tank_design_id": design,
        }
    return camera


def _write(recording: Path, cameras: dict, *, version: int = 2) -> None:
    path = recording / GEOMETRY_CONTRACT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_id": "orange.recording.geometry_contract",
        "schema_version": version,
        "cameras": cameras,
    }))


@pytest.mark.parametrize("version", [1, 2, 3])
def test_each_camera_reads_its_own_design(tmp_path: Path, version: int) -> None:
    _write(tmp_path, {"2010093": _camera("palm1"), "2010095": _camera("palm2")}, version=version)

    assert recording_dish_design(tmp_path, "2010093") == "palm1"
    assert recording_dish_design(tmp_path, "2010095") == "palm2"
    assert recording_dish_design(tmp_path, "2010093", declared="palm1") == "palm1"


def test_absent_sources_keep_the_declared_value(tmp_path: Path) -> None:
    assert recording_dish_design(tmp_path, "2010093") is None
    assert recording_dish_design(tmp_path, "2010093", declared="legacy") == "legacy"
    assert recording_dish_design(None, "2010093", declared="legacy") == "legacy"
    _write(tmp_path, {"2010093": {}, "2010094": _camera(None, rim=None)})
    assert recording_dish_design(tmp_path, "2010093") is None
    assert recording_dish_design(tmp_path, "2010096") is None
    camera = _camera(None)
    camera["daily_registration_geometry"] = {}
    _write(tmp_path, {"2010093": camera})
    assert recording_dish_design(tmp_path, "2010093") is None


@pytest.mark.parametrize(
    ("cameras", "version", "declared", "match"),
    [
        ({"2010093": _camera("palm1")}, 4, None, "version"),
        ({"2010093": _camera("palm1", rim="palm2")}, 2, None, "disagrees with tank_design_id"),
        ({"2010093": _camera(None, rim="palm1")}, 2, None, "no tank_design"),
        ({"2010093": {**_camera("palm1"), "tank_design": {"status": "missing"}}}, 2, None, "referenced"),
        ({"2010093": {**_camera("palm1"), "tank_design": {"status": "referenced", "tank_design_id": ""}}},
         2, None, "referenced"),
        ({"2010093": _camera("palm1")}, 2, "palm2", "disagrees with"),
    ],
)
def test_contradictions_are_refused(tmp_path: Path, cameras, version, declared, match) -> None:
    _write(tmp_path, cameras, version=version)

    with pytest.raises(RecordingGeometryError, match=match):
        recording_dish_design(tmp_path, "2010093", declared=declared)


def test_unparseable_contract_is_refused(tmp_path: Path) -> None:
    path = tmp_path / GEOMETRY_CONTRACT_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text("{")

    with pytest.raises(RecordingGeometryError, match="not JSON"):
        recording_dish_design(tmp_path, "2010093")


def test_current_source_import_projects_dish_design_to_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = clipped._write_current_clipped_recording

    def with_contract(*args, **kwargs):
        recording = original(*args, **kwargs)
        _write(recording, {clipped.CAMERA: _camera("palm2")})
        return recording

    monkeypatch.setattr(clipped, "_write_current_clipped_recording", with_contract)
    registry = clipped._registered_crop_collection(tmp_path, monkeypatch)
    try:
        designs = [row[0] for row in registry.conn.execute("SELECT dish_design FROM recordings;")]
    finally:
        registry.close()

    assert designs == ["palm2"]

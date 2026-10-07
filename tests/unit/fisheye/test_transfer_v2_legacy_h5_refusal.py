"""New transfer-v2 deliveries carry a sealed unified H5 or no H5 at all.

Decision 1 of docs/design/2026-10-07-intake-single-writer: a transfer-v2
delivery whose H5 is a legacy (non-unified) stimulus H5 is refused. The
organizer refuses it while planning, before any destination or staging write,
and the importer refuses a sealed transfer-v2 parent manifest that binds one
before it creates the analysis Zarr. Unified H5s, recording-only deliveries,
and legacy H5s outside transfer-v2 intake keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py
import pytest

from fisheye.utils import import_recording_analysis as importer
from fisheye.utils import organize_transfer_recordings as organizer
from tests.unit.fisheye.test_current_clipped_recording_import import (
    _options,
    _stub_only_checkout_identity,
    _write_current_clipped_recording,
)
from tests.unit.fisheye.test_transfer_recording_organization import (
    _bytes,
    _plan,
    _resign,
    _source,
)
from tests.unit.fisheye.test_unified_transfer_receipts import (
    RECEIPT_RELATIVE,
    _transfer as _unified_transfer,
)
from tests.unit.fisheye.unified_h5_fixtures import emit_fixture, write_receipt

CAMERA = "02010093"
SESSION = "fixture-session"
SEALED_TRANSFER_BINDING = {
    "schema_id": "palette.organized_recording_transfer.v1",
    "source_to_parent_files": [],
}


def _legacy_h5(path: Path, *, camera: str = CAMERA, session: str = SESSION) -> Path:
    """A legacy stimulus H5 whose camera and session bind exactly."""
    with h5py.File(path, "w") as h5:
        h5.attrs["camera_id"] = camera
        h5.attrs["session_uuid"] = session
        h5.attrs["rig_id"] = "synthetic-rig"
    return path


# --- organization planning --------------------------------------------------


def test_plan_refuses_a_legacy_h5_before_any_write(tmp_path: Path) -> None:
    source = _source(tmp_path)
    _legacy_h5(source / "legacy_stimulus.h5")
    _resign(source)
    before = _bytes(source)
    destination = tmp_path / "recordings"

    with pytest.raises(ValueError, match="legacy .*H5.*refused"):
        _plan(source, destination)

    assert not destination.exists()
    assert _bytes(source) == before


def test_plan_refuses_a_legacy_h5_even_with_an_unhelpful_name(tmp_path: Path) -> None:
    source = _source(tmp_path)
    _legacy_h5(source / "renamed.hdf5")
    _resign(source)

    with pytest.raises(ValueError, match="legacy .*H5.*refused"):
        _plan(source, tmp_path / "recordings")


def test_camera_h5_context_admits_a_unified_h5_through_its_receipt(
    tmp_path: Path,
) -> None:
    source, inventory = _unified_transfer(tmp_path)
    contexts = {
        "CAM-42": {
            "recording_type": "behavior",
            "recording_subtype": "chaser",
            "behavior_mode": "free",
        }
    }
    h5_relative = next(path for path in inventory if path.endswith(".h5"))

    camera, metadata, receipt = organizer._camera_h5_context(
        source, h5_relative, inventory, contexts
    )

    assert camera == "CAM-42"
    assert receipt == RECEIPT_RELATIVE
    assert metadata["session_uuid"] == "synthetic-paired-recording"


def test_camera_h5_context_refuses_a_legacy_h5(tmp_path: Path) -> None:
    source = tmp_path / "transfer"
    source.mkdir()
    _legacy_h5(source / "legacy.h5")

    with pytest.raises(ValueError, match="legacy .*H5.*refused"):
        organizer._camera_h5_context(source, "legacy.h5", {"legacy.h5": {}}, {})


def test_plan_keeps_a_unified_h5_on_its_camera_with_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A full Orange-bound transfer fixture does not exist; the unified H5
    # admission itself is covered with real fixtures above and in
    # test_unified_transfer_receipts. This checks the plan keeps routing it.
    source = _source(tmp_path)
    with h5py.File(source / "unified.h5", "w") as h5:
        h5.create_group("metadata/session").attrs["recording_artifact_profile"] = (
            importer.UNIFIED_H5_PROFILE
        )
    receipt = source / "unified.receipt.json"
    receipt.write_text("{}")
    _resign(source)
    monkeypatch.setattr(
        organizer,
        "_unified_h5_context",
        lambda _source, relative, _inventory, _contexts: (
            CAMERA,
            {"session_uuid": SESSION},
            "unified.receipt.json",
        ),
    )

    plan = _plan(source, tmp_path / "recordings")

    parent = next(p for p in plan["parents"] if p["identity"]["camera_id"] == CAMERA)
    assert parent["h5_relative_path"] == "raw/acquisition/unified.h5"
    assert (
        parent["h5_finalization_receipt_relative_path"]
        == "raw/acquisition/unified.receipt.json"
    )


def test_recording_only_delivery_still_plans(tmp_path: Path) -> None:
    plan = _plan(_source(tmp_path), tmp_path / "recordings")

    assert len(plan["parents"]) == 2
    assert all(parent["h5_relative_path"] is None for parent in plan["parents"])
    assert all(item["role"] != "camera_h5" for item in plan["files"])


# --- import of an organized parent -----------------------------------------


def _seal_as_transfer_v2(recording: Path, **fields: object) -> None:
    path = recording / "recording_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_transfer"] = SEALED_TRANSFER_BINDING
    manifest.update(fields)
    path.write_text(json.dumps(manifest))


def _replace_h5_with_unified(recording: Path) -> None:
    h5_path = recording / "raw" / "acquisition" / "protocol.h5"
    h5_path.unlink()
    shutil.move(emit_fixture(recording / "raw" / "fixture", "base"), h5_path)
    receipt = write_receipt(recording / "raw" / "acquisition", "base")
    _seal_as_transfer_v2(
        recording,
        h5_finalization_receipt_relative_path=receipt.relative_to(recording).as_posix(),
    )


def test_transfer_v2_parent_with_a_legacy_h5_is_refused_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _write_current_clipped_recording(tmp_path, h5=True)
    _seal_as_transfer_v2(recording)
    _stub_only_checkout_identity(monkeypatch)
    plan = importer.RecordingAnalysisPlan(
        recording_dir=recording,
        h5_path=recording / "raw" / "acquisition" / "protocol.h5",
        cam_video=None,
        zarr_path=recording / "zarr" / "recording-clipped_analysis.zarr",
        recording_layout="clipped_video_collection",
    )
    before = _bytes(recording)

    result = importer.process_recording_import(plan, _options())

    assert not result.ok
    assert result.failed_step == "recording_import_preflight"
    assert "legacy_h5_refused_for_transfer_v2" in str(result.error)
    assert not plan.zarr_path.exists()
    assert _bytes(recording) == before
    with pytest.raises(ValueError, match="legacy_h5_refused_for_transfer_v2"):
        importer.resolve_single_recording_plan(recording_dir=recording, require_h5=False)


def test_transfer_v2_parent_with_a_unified_h5_but_no_receipt_is_refused(
    tmp_path: Path,
) -> None:
    recording = _write_current_clipped_recording(tmp_path, h5=True)
    _replace_h5_with_unified(recording)
    manifest_path = recording / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["h5_finalization_receipt_relative_path"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unified_h5_requires_finalization_receipt"):
        importer.resolve_single_recording_plan(recording_dir=recording, require_h5=False)
    assert not (recording / "zarr").exists()


def test_transfer_v2_parent_with_a_unified_h5_still_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _write_current_clipped_recording(tmp_path, h5=True)
    _replace_h5_with_unified(recording)
    _stub_only_checkout_identity(monkeypatch)

    plan = importer.resolve_single_recording_plan(recording_dir=recording, require_h5=False)
    result = importer.process_recording_import(plan, _options())

    assert result.ok, result
    assert result.receipt is not None


def test_transfer_v2_recording_only_parent_still_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _write_current_clipped_recording(tmp_path)
    _seal_as_transfer_v2(recording)
    _stub_only_checkout_identity(monkeypatch)

    plan = importer.resolve_single_recording_plan(recording_dir=recording, require_h5=False)
    assert plan.h5_path is None
    result = importer.process_recording_import(plan, _options())

    assert result.ok, result


def test_legacy_h5_outside_transfer_v2_intake_still_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A manifest without the transfer-v2 seal is not new transfer-v2 intake;
    # the legacy H5 route stays available to it.
    recording = _write_current_clipped_recording(tmp_path, h5=True)
    _stub_only_checkout_identity(monkeypatch)

    plan = importer.resolve_single_recording_plan(recording_dir=recording, require_h5=False)
    assert importer.stimulus_h5_unified_profile(plan.h5_path) is None
    result = importer.process_recording_import(plan, _options())

    assert result.ok, result

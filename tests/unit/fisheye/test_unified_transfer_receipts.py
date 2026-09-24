"""Transfer-v2 intake of unified H5s: identity from the binding, receipt from the H5.

The receipt path is derived from the H5's own observation_context_id
(``recording_observation_bindings/receipts/<id>.json``), must be part of the
transfer, and must name the same H5. The session importer then uses the
receipt the manifest declares.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from fisheye.utils import import_recording_analysis as importer
from fisheye.utils import organize_transfer_recordings as organizer
from tests.unit.fisheye.unified_h5_fixtures import FIXTURES, emit_fixture

RECEIPT = json.loads((FIXTURES / "base.receipt.json").read_text())
H5_RELATIVE = RECEIPT["contract"]["h5_artifact"]["relative_path"]
OBSERVATION = RECEIPT["contract"]["observation_context_id"]
RECEIPT_RELATIVE = f"recording_observation_bindings/receipts/{OBSERVATION}.json"


def _transfer(tmp_path: Path, *, receipt: dict | None = RECEIPT) -> tuple[Path, dict]:
    source = tmp_path / "transfer"
    h5 = source / H5_RELATIVE
    h5.parent.mkdir(parents=True)
    shutil.move(emit_fixture(tmp_path / "fixture", "base"), h5)
    inventory = {H5_RELATIVE: {}}
    if receipt is not None:
        path = source / RECEIPT_RELATIVE
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        inventory[RECEIPT_RELATIVE] = {}
    return source, inventory


def test_unified_h5_identity_context_and_receipt(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)

    camera, context, receipt = organizer._unified_h5_context(source, H5_RELATIVE, inventory)

    assert camera == "CAM-42"
    assert receipt == RECEIPT_RELATIVE
    # Acquisition session, not the Citrus per-Arena session.
    assert context["session_uuid"] == "synthetic-paired-recording"
    assert context["citrus_session_uuid"].startswith("citsess_")
    assert context["observation_context_id"] == OBSERVATION
    assert context["rig_id"] == "synthetic-preservation-rig"
    assert context["arena_id"] == "arena_1"


def test_missing_receipt_is_refused(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path, receipt=None)

    with pytest.raises(ValueError, match="receipt missing from transfer"):
        organizer._unified_h5_context(source, H5_RELATIVE, inventory)


def test_receipt_for_another_h5_is_refused(tmp_path: Path) -> None:
    other = json.loads(json.dumps(RECEIPT))
    other["contract"]["h5_artifact"]["relative_path"] = "citrus/other.h5"
    source, inventory = _transfer(tmp_path, receipt=other)

    with pytest.raises(ValueError, match="does not name this H5"):
        organizer._unified_h5_context(source, H5_RELATIVE, inventory)


def test_claims_that_disagree_with_the_binding_are_refused(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)
    with h5py.File(source / H5_RELATIVE, "r+") as h5:
        dataset = h5[organizer.UNIFIED_CLAIMS_PATH]
        attrs, dtype = dict(dataset.attrs), dataset.dtype
        claims = json.loads(dataset[()])
        claims["recording_id"] = "another-acquisition"
        del h5[organizer.UNIFIED_CLAIMS_PATH]
        replacement = h5.create_dataset(
            organizer.UNIFIED_CLAIMS_PATH,
            data=json.dumps(claims, sort_keys=True, separators=(",", ":")).encode(),
            dtype=dtype,
        )
        for key, value in attrs.items():
            replacement.attrs[key] = value

    with pytest.raises(ValueError, match="claims and acquisition binding disagree"):
        organizer._unified_h5_context(source, H5_RELATIVE, inventory)


def _recording_with_manifest(tmp_path: Path, receipt_relative: str) -> Path:
    recording = tmp_path / "recording"
    (recording / "raw").mkdir(parents=True)
    emit_fixture(recording / "raw", "base")
    (recording / "raw" / "receipt.json").write_text(json.dumps(RECEIPT))
    (recording / "recording_manifest.json").write_text(
        json.dumps({"h5_finalization_receipt_relative_path": receipt_relative})
    )
    return recording


def test_session_importer_uses_the_manifest_declared_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _recording_with_manifest(tmp_path, "raw/receipt.json")
    calls = []
    monkeypatch.setattr(
        importer.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    plan = importer.RecordingAnalysisPlan(
        recording_dir=recording,
        h5_path=recording / "raw" / "base.h5",
        cam_video=None,
        zarr_path=recording / "zarr" / "rec_analysis.zarr",
    )
    options = importer.RecordingImportOptions(
        import_video_metadata=False, video_metadata_overwrite=False, import_stimulus=True,
        stimulus_always=False, stimulus_run_name="candidate", stimulus_overwrite=False,
        stimulus_quiet=True,
    )

    ok, _code, command = importer.run_stimulus_import(plan, options)

    assert ok
    receipt_index = command.index("--finalization-receipt") + 1
    assert Path(command[receipt_index]) == (recording / "raw" / "receipt.json").resolve()


def test_manifest_receipt_outside_the_recording_is_refused(tmp_path: Path) -> None:
    recording = _recording_with_manifest(tmp_path, "../escape.json")
    (tmp_path / "escape.json").write_text("{}")

    with pytest.raises(ValueError):
        importer._manifest_finalization_receipt(recording)

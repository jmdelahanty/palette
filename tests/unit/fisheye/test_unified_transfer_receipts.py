"""Transfer-v2 intake of unified H5s: identity from the binding, receipt from the collection.

Orange's finalized observation collection names each H5 and its external
receipt (the contract's path authority). The H5's own observation context and
the receipt must agree with that entry, and its session context must equal the
producer's declared parent context exactly, including an omitted subtype.
The session importer then uses the receipt the manifest declares.
"""

from __future__ import annotations

import hashlib
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


CONTEXT = {"recording_type": "behavior", "recording_subtype": "chaser", "behavior_mode": "free"}
CONTEXTS = {"CAM-42": CONTEXT}


def _collection(receipt_bytes: bytes, *, h5_artifact=None) -> dict:
    contract = RECEIPT["contract"]
    return {
        "schema_id": "orange.recording.observation_binding_finalization",
        "schema_version": 1,
        "status": "finalized",
        "binding_status": "bound",
        "recording_id": "synthetic-paired-recording",
        "citrus_experiment_id": contract["citrus_experiment_id"],
        "context_count": 1,
        "observation_contexts": [
            {
                "status": "bound",
                "observation_context_id": OBSERVATION,
                "citrus_h5": h5_artifact or contract["h5_artifact"],
                "finalized_receipt": {
                    "relative_path": RECEIPT_RELATIVE,
                    "sha256": "sha256:" + hashlib.sha256(receipt_bytes).hexdigest(),
                    "contract_sha256": RECEIPT["contract_sha256"],
                    "receipt_id": RECEIPT["receipt_id"],
                },
            }
        ],
    }


def _transfer(
    tmp_path: Path, *, receipt: dict | None = RECEIPT, collection: dict | None = None
) -> tuple[Path, dict]:
    source = tmp_path / "transfer"
    h5 = source / H5_RELATIVE
    h5.parent.mkdir(parents=True)
    shutil.move(emit_fixture(tmp_path / "fixture", "base"), h5)
    inventory = {H5_RELATIVE: {}}
    receipt_bytes = json.dumps(receipt or RECEIPT).encode()
    if receipt is not None:
        path = source / RECEIPT_RELATIVE
        path.parent.mkdir(parents=True)
        path.write_bytes(receipt_bytes)
        inventory[RECEIPT_RELATIVE] = {}
    path = source / organizer.UNIFIED_COLLECTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection or _collection(receipt_bytes)))
    inventory[organizer.UNIFIED_COLLECTION_PATH] = {}
    return source, inventory


def _context(source, inventory, contexts=CONTEXTS):
    return organizer._unified_h5_context(source, H5_RELATIVE, inventory, contexts)


def test_unified_h5_identity_context_and_receipt(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)

    camera, context, receipt = _context(source, inventory)

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
        _context(source, inventory)


def test_receipt_for_another_h5_is_refused(tmp_path: Path) -> None:
    other = json.loads(json.dumps(RECEIPT))
    other["contract"]["h5_artifact"]["sha256"] = "sha256:" + "0" * 64
    source, inventory = _transfer(tmp_path, receipt=other)

    with pytest.raises(ValueError, match="does not name this H5"):
        _context(source, inventory)


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
        _context(source, inventory)


def test_h5_not_in_the_collection_is_refused(tmp_path: Path) -> None:
    unlisted = dict(RECEIPT["contract"]["h5_artifact"], relative_path="citrus/other.h5")
    source, inventory = _transfer(
        tmp_path, collection=_collection(json.dumps(RECEIPT).encode(), h5_artifact=unlisted)
    )

    with pytest.raises(ValueError, match="not listed exactly once"):
        _context(source, inventory)


def test_producer_context_must_match_the_h5(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)
    with h5py.File(source / H5_RELATIVE, "r+") as h5:
        h5["/metadata/session"].attrs["behavior_mode"] = "embedded"

    with pytest.raises(ValueError, match="behavior_mode"):
        _context(source, inventory)
    _context(source, inventory, {"CAM-42": dict(CONTEXT, behavior_mode="embedded")})


def test_omitted_subtype_refuses_an_h5_that_declares_one(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)
    subtype_free = {"CAM-42": {k: v for k, v in CONTEXT.items() if k != "recording_subtype"}}
    _context(source, inventory, subtype_free)
    with h5py.File(source / H5_RELATIVE, "r+") as h5:
        h5["/metadata/session"].attrs["recording_subtype"] = "chaser"

    with pytest.raises(ValueError, match="recording_subtype"):
        _context(source, inventory, subtype_free)
    _context(source, inventory)


def test_h5_camera_without_a_producer_context_is_refused(tmp_path: Path) -> None:
    source, inventory = _transfer(tmp_path)

    with pytest.raises(ValueError, match="no producer recording context"):
        _context(source, inventory, {"CAM-7": CONTEXT})


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

"""Citrus sealer 3.1.0: marker v4, snapshot v3, Citrus artifacts, revision chains.

The fixtures are the sealer's own deliveries at tag recording-transfer-v3.1.0
(tests/fixtures/recording_transfer_v3_1/README.md). Palette must rebuild each
sealed snapshot byte for byte. The refusal cases below are the sealer's
constructed cases that need only file edits. The ones that also rewrite
receipts are covered by the sealer's own suite, against the same ported code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from fisheye.shared import recording_transfer_snapshot as transfer
from fisheye.shared.recording_transfer_snapshot import (
    MARKER_NAME,
    SNAPSHOT_PATH,
    TransferSnapshotError,
    canonical_bytes,
    sealer_attested_inventory,
    verify_transfer_snapshot,
)
from fisheye.utils import citrus_transfer_v2_poller as poller
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v3_1"
BINDINGS = "recording_observation_bindings"
CAMERA_BY_SESSION = {"1": "02010093", "2": "02010094"}
EXPECTED = {
    "bound_v1_no_video": (3, 2, 0),
    "bound_v2_video": (4, 3, 8),
    "bound_v2_no_video": (4, 3, 4),
    "bound_v2_no_citrus_artifacts": (4, 3, 0),
    "bound_v1_upgraded_to_v2_pre310_diagnostic": (4, 3, 7),
}


def _copy(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / "staging" / name))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_sealer_fixture_is_admitted_byte_for_byte(tmp_path, name):
    root = _copy(tmp_path, name)
    marker_version, snapshot_version, artifacts = EXPECTED[name]
    verified = verify_transfer_snapshot(root)
    assert json.loads((root / MARKER_NAME).read_bytes())["schema_version"] == marker_version
    assert verified.snapshot["schema_version"] == snapshot_version
    assert len(verified.snapshot.get("citrus_artifacts", [])) == artifacts
    rebuilt = transfer.build_snapshot(root, MARKER_NAME, destination=True)
    assert canonical_bytes(rebuilt) == (root / SNAPSHOT_PATH).read_bytes()
    assert poller.check_marker(root / MARKER_NAME)["schema_version"] == marker_version


def test_v3_snapshot_carries_the_revision_chain_oldest_first(tmp_path):
    snapshot = verify_transfer_snapshot(
        _copy(tmp_path, "bound_v1_upgraded_to_v2_pre310_diagnostic")
    ).snapshot
    assert [row["path"] for row in snapshot["observation_binding_revisions"]] == [
        f"{BINDINGS}/finalized_collection.json",
        f"{BINDINGS}/finalized_collection.r2.json",
    ]
    diagnostics = [r for r in snapshot["citrus_artifacts"] if r["role"] == "process_diagnostic"]
    assert len(diagnostics) == 1  # the pre-3.1.0 name, once, in the lowest context


def _rewrite_marker(root: Path, **changes) -> None:
    path = root / MARKER_NAME
    marker = json.loads(path.read_bytes())
    marker.update(changes)
    path.write_bytes(canonical_bytes(marker))


def test_a_v3_marker_cannot_seal_a_v3_snapshot(tmp_path):
    root = _copy(tmp_path, "bound_v2_video")
    _rewrite_marker(root, schema_id=transfer.MARKER_SCHEMA_V3, schema_version=3)
    with pytest.raises(TransferSnapshotError, match="marker_v3 .*snapshot/schema_version"):
        verify_transfer_snapshot(root)


def test_a_v4_marker_cannot_seal_a_v2_snapshot(tmp_path):
    root = _copy(tmp_path, "bound_v1_no_video")
    _rewrite_marker(root, schema_id=transfer.MARKER_SCHEMA_V4, schema_version=4)
    with pytest.raises(TransferSnapshotError, match="marker_v4 .*snapshot/schema_version"):
        verify_transfer_snapshot(root)


def test_a_v4_marker_from_a_storage_reading_sealer_is_attested(tmp_path):
    root = _copy(tmp_path, "bound_v2_video")
    marker = json.loads((root / MARKER_NAME).read_bytes())
    snapshot = json.loads((root / SNAPSHOT_PATH).read_bytes())
    attested = sealer_attested_inventory(marker, snapshot, 1)
    assert attested is not None
    assert set(attested.items) == {item["path"] for item in snapshot["inventory"]}


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: (r / "citrus/extra_stimulus.mp4").write_bytes(b"NOT REAL MEDIA\n"),
         "undeclared Citrus artifact: citrus/extra_stimulus.mp4"),
        (lambda r: (r / "citrus/notes.txt").write_text("stray\n"),
         "undeclared Citrus artifact: citrus/notes.txt"),
        (lambda r: ((r / "citrus/sub").mkdir(), (r / "citrus/sub/x.json").write_text("{}\n")),
         "subdirectory under citrus/"),
        (lambda r: next((r / "citrus").glob("*_update_timing.csv")).write_text("changed\n"),
         "declared Citrus artifact size or SHA-256 mismatch"),
    ],
)
def test_citrus_folder_contradictions_are_refused(tmp_path, edit, message):
    root = _copy(tmp_path, "bound_v2_video")
    edit(root)
    with pytest.raises(TransferSnapshotError, match=message):
        verify_transfer_snapshot(root)


def _upgraded(tmp_path: Path) -> Path:
    return _copy(tmp_path, "bound_v1_upgraded_to_v2_pre310_diagnostic")


def _head(root: Path) -> tuple[Path, dict]:
    path = root / BINDINGS / "finalized_collection.r2.json"
    return path, json.loads(path.read_text())


def _broken_supersede(root: Path) -> None:
    path, head = _head(root)
    head["supersedes"]["sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(head, indent=2) + "\n")


def _two_heads(root: Path) -> None:
    _, r3 = _head(root)
    r1 = root / BINDINGS / "finalized_collection.json"
    r3["revision"] = 3
    r3["supersedes"] = {
        "relative_path": f"{BINDINGS}/finalized_collection.json",
        "sha256": "sha256:" + _sha(r1),
    }
    (root / BINDINGS / "finalized_collection.r3.json").write_text(json.dumps(r3, indent=2) + "\n")


def _gap(root: Path) -> None:
    path, _ = _head(root)
    path.rename(root / BINDINGS / "finalized_collection.r3.json")


def _beyond_receipt(root: Path) -> None:
    path, head = _head(root)
    head["observation_contexts"][0]["observation_identity_sha256"] = "sha256:" + "b" * 64
    path.write_text(json.dumps(head, indent=2) + "\n")


def _projection_without_chain(root: Path) -> None:
    path = root / "recording_session.json"
    manifest = json.loads(path.read_bytes())
    manifest["recording_observation_bindings"].pop("revision_chain")
    path.write_bytes(canonical_bytes(manifest))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_broken_supersede, "collection r2 does not supersede r1 exactly"),
        (_two_heads, "collection r3 does not supersede r2 exactly"),
        (lambda r: (r / BINDINGS / "finalized_collection.json").unlink(),
         "bound recording lacks finalized observation collection"),
        (_gap, "collection revision chain has a gap"),
        (_beyond_receipt, "collection r2 changes a context beyond its receipt"),
        (_projection_without_chain, "projection differs from finalized collection"),
    ],
)
def test_revision_chain_contradictions_are_refused(tmp_path, edit, message):
    root = _upgraded(tmp_path)
    edit(root)
    with pytest.raises(TransferSnapshotError, match=message):
        verify_transfer_snapshot(root)


def _stub_h5_context(monkeypatch) -> None:
    """Fixture H5s are placeholders; bind each to its camera by Citrus session."""

    def context(_source, relative, _inventory, _contexts):
        session = Path(relative).name.removeprefix("citsess_")[0]
        camera = CAMERA_BY_SESSION[session]
        return camera, {}, f"{BINDINGS}/receipts/obsctx_{session * 64}.json"

    monkeypatch.setattr(organizer, "_camera_h5_context", context)


@pytest.mark.parametrize("name", ["bound_v2_video", "bound_v1_upgraded_to_v2_pre310_diagnostic"])
def test_citrus_artifacts_go_only_to_their_h5_camera(tmp_path, monkeypatch, name):
    root = _copy(tmp_path, name)
    _stub_h5_context(monkeypatch)
    plan = organizer.build_transfer_organization_plan(root, destination_root=tmp_path / "out")
    snapshot = verify_transfer_snapshot(root).snapshot
    by_path = {item["source"]["path"]: item for item in plan["files"]}
    recording_ids = {p["identity"]["camera_id"]: p["identity"]["recording_id"] for p in plan["parents"]}
    for row in snapshot["citrus_artifacts"]:
        item = by_path[row["path"]]
        camera = CAMERA_BY_SESSION[row["citrus_session_uuid"].removeprefix("citsess_")[0]]
        assert (item["role"], item["camera_id"]) == ("camera_citrus_artifact", camera)
        assert item["destinations"] == [
            {"recording_id": recording_ids[camera], "relative_path": f"raw/acquisition/{row['path']}"}
        ]


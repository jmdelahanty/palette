"""Citrus sealer 3.2.0: receipt v3 (v2 plus contract.subject_reference).

The fixtures were made by the tag's own generator at recording-transfer-v3.2.0
(tests/fixtures/recording_transfer_v3_2/README.md). Palette must rebuild each
sealed snapshot byte for byte, and refuse the source trees the sealer refuses
with the sealer's messages. Receipt v3 changes nothing on disk: the snapshot is
v3 and the marker v4, exactly as for a receipt-v2 head.
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
    SUBJECT_REFERENCE_KEYS,
    SUBJECT_REFERENCE_OBJECT_KEYS,
    TransferSnapshotError,
    canonical_bytes,
    check_subject_reference,
    verify_transfer_snapshot,
)
from fisheye.utils import citrus_transfer_v2_poller as poller
from fisheye.utils import organize_transfer_recordings as organizer

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v3_2"
BINDINGS = "recording_observation_bindings"
CAMERA_BY_SESSION = {"1": "02010093", "2": "02010094"}
# agent-contracts citrus-recording-transfer-3.1/
# citrus_recording_observation_finalized_receipt.v3.schema.json (PR #77), the
# same bytes the sealer's tests pin.
RECEIPT_V3_SCHEMA = FIXTURES / "citrus_recording_observation_finalized_receipt.v3.schema.json"
RECEIPT_V3_SCHEMA_SHA256 = "c5a55e533d3cdb914392a5d0d4bf4e8bb83cd9e9b3a81e04428de0a20adaea27"
EXPECTED = {"bound_v3_video": 8, "bound_v3_null_subject_objects": 0}
REFUSED = {
    "mixed_v2_v3": "the head collection mixes receipt versions",
    "v3_without_subject_reference": "receipt v3 keys are not exact",
    "v3_extra_contract_key": "receipt v3 keys are not exact",
    "v2_with_subject_reference": "receipt v2 keys are not exact",
}


def _copy(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / "staging" / Path(name).name))


def _receipts(root: Path) -> list[dict]:
    return [
        json.loads(path.read_bytes())
        for path in sorted((root / BINDINGS / "receipts").glob("obsctx_*.json"))
    ]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_v3_sealer_fixture_is_admitted_byte_for_byte(tmp_path, name):
    root = _copy(tmp_path, name)
    verified = verify_transfer_snapshot(root)
    assert json.loads((root / MARKER_NAME).read_bytes())["schema_version"] == 4
    assert verified.snapshot["schema_version"] == 3
    assert len(verified.snapshot.get("citrus_artifacts", [])) == EXPECTED[name]
    assert {r["schema_version"] for r in _receipts(root)} == {3}
    rebuilt = transfer.build_snapshot(root, MARKER_NAME, destination=True)
    assert canonical_bytes(rebuilt) == (root / SNAPSHOT_PATH).read_bytes()
    assert poller.check_marker(root / MARKER_NAME)["schema_version"] == 4


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_sources_the_sealer_refuses_are_refused(tmp_path, name):
    root = _copy(tmp_path, f"refused_sources/{name}")
    with pytest.raises(TransferSnapshotError, match=REFUSED[name]):
        transfer.build_snapshot(root, MARKER_NAME)


def test_the_receipt_v3_schema_is_the_pinned_contract_and_the_fixtures_satisfy_it():
    assert hashlib.sha256(RECEIPT_V3_SCHEMA.read_bytes()).hexdigest() == RECEIPT_V3_SCHEMA_SHA256
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(RECEIPT_V3_SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    receipts = _receipts(FIXTURES / "bound_v3_video")
    assert len(receipts) == 2
    for receipt in receipts:
        validator.validate(receipt)
    # The sealer checks key sets only. A lookup_status that says "found" with
    # a null dish is structurally sealable but breaks the schema's conditionals,
    # which are the readers' contract: Palette must check them itself before it
    # relies on subject_reference semantics.
    null_objects = _receipts(FIXTURES / "bound_v3_null_subject_objects")[0]
    assert not validator.is_valid(null_objects)
    # The ported structural rule is the schema's key sets, nothing else.
    reference = schema["$defs"]["subjectReference"]
    assert set(reference["properties"]) == SUBJECT_REFERENCE_KEYS
    assert set(reference["required"]) == SUBJECT_REFERENCE_KEYS
    for key, keys in SUBJECT_REFERENCE_OBJECT_KEYS.items():
        assert set(reference["properties"][key]["properties"]) == keys, key
        assert set(reference["properties"][key]["required"]) == keys, key


def _subject() -> dict:
    receipt = _receipts(FIXTURES / "bound_v3_video")[0]
    return json.loads(json.dumps(receipt["contract"]["subject_reference"]))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(schema_id="citrus.subject_declaration"), "is not citrus.subject_reference v1"),
        (lambda s: s.update(schema_version=2), "is not citrus.subject_reference v1"),
        (lambda s: s.update(schema_version=True), "is not citrus.subject_reference v1"),
        (lambda s: s.pop("lookup_http_status"), "is not citrus.subject_reference v1"),
        (lambda s: s.update(fish_id=None), "is not citrus.subject_reference v1"),
        (lambda s: s["dish"].pop("cross_id"), r"subject_reference\.dish keys are not exact"),
        (
            lambda s: s.update(fish={"fish_id": "x", "revision": 1, "updated_at": "y", "label": "Z1"}),
            r"subject_reference\.fish keys are not exact",
        ),
        (lambda s: s["zebrobot_version"].pop("error"), r"subject_reference\.zebrobot_version keys are not exact"),
    ],
)
def test_subject_reference_structure_is_refused(edit, message):
    subject = _subject()
    edit(subject)
    with pytest.raises(TransferSnapshotError, match=message):
        check_subject_reference(subject)


def test_a_subject_reference_that_is_not_an_object_is_refused():
    with pytest.raises(TransferSnapshotError, match="is not citrus.subject_reference v1"):
        check_subject_reference("19234_2")


def test_null_dish_fish_and_version_pass_the_structural_check():
    subject = _subject()
    subject.update(dish=None, fish=None, zebrobot_version=None)
    check_subject_reference(subject)


def test_v3_citrus_artifacts_go_only_to_their_h5_camera(tmp_path, monkeypatch):
    root = _copy(tmp_path, "bound_v3_video")

    def context(_source, relative, _inventory, _contexts):
        session = Path(relative).name.removeprefix("citsess_")[0]
        return CAMERA_BY_SESSION[session], {}, f"{BINDINGS}/receipts/obsctx_{session * 64}.json"

    monkeypatch.setattr(organizer, "_camera_h5_context", context)
    plan = organizer.build_transfer_organization_plan(root, destination_root=tmp_path / "out")
    snapshot = verify_transfer_snapshot(root).snapshot
    by_path = {item["source"]["path"]: item for item in plan["files"]}
    recording_ids = {p["identity"]["camera_id"]: p["identity"]["recording_id"] for p in plan["parents"]}
    stimulus = [row for row in snapshot["citrus_artifacts"] if row["role"] == "stimulus_video"]
    assert len(stimulus) == 2
    for row in snapshot["citrus_artifacts"]:
        item = by_path[row["path"]]
        camera = CAMERA_BY_SESSION[row["citrus_session_uuid"].removeprefix("citsess_")[0]]
        assert (item["role"], item["camera_id"]) == ("camera_citrus_artifact", camera)
        assert item["destinations"] == [
            {"recording_id": recording_ids[camera], "relative_path": f"raw/acquisition/{row['path']}"}
        ]

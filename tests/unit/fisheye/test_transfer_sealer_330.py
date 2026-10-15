"""Citrus sealer 3.3.0: receipt v4 (contract.subject_reference v2).

The fixtures were made by the tag's own generator at recording-transfer-v3.3.0
(tests/fixtures/recording_transfer_v3_3/README.md). Palette must rebuild each
sealed snapshot byte for byte, and refuse the source trees the sealer refuses
with the sealer's messages. Receipt v4 changes nothing on disk: the snapshot is
v3 and the marker v4, as for receipt-v2 and v3 heads. Receipt v3 pairs only
with reference v1 and v4 only with v2; v3 and v4 never mix in a head or in the
revision chain, and a v4 head's earlier revisions are v4 too.
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

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "recording_transfer_v3_3"
BINDINGS = "recording_observation_bindings"
# agent-contracts #105 (merge 83f9ed5), the same bytes the sealer's tests pin.
RECEIPT_V4_SCHEMA = FIXTURES / "citrus_recording_observation_finalized_receipt.v4.schema.json"
RECEIPT_V4_SCHEMA_SHA256 = "47d2251244ed6bf18cf5e65c360da2b4a809afbdee1864a3cef75b0b44a76b83"
REFERENCE_V2_SCHEMA = FIXTURES / "citrus_subject_reference_v2.schema.json"
REFERENCE_V2_SCHEMA_SHA256 = "30f42c9d49d1ba040b509d6743144aefacda63365c5ca4261524506b35931eb9"
EXPECTED = {"bound_v4_video": 8, "bound_v4_dish_group": 0, "bound_v4_chain_v4_v4": 8}
REFUSED = {
    "mixed_v3_v4_head": "the head collection mixes receipt versions",
    "v4_with_reference_v1": "receipt v4 subject_reference is not citrus.subject_reference v2",
    "v3_with_reference_v2": "receipt v3 subject_reference is not citrus.subject_reference v1",
    "v4_without_subject_reference": "receipt v4 keys are not exact",
    "v4_extra_contract_key": "receipt v4 keys are not exact",
    "chain_v3_then_v4": "the collection revision chain mixes receipt v3 and v4",
    "chain_v4_then_v3": "the collection revision chain mixes receipt v3 and v4",
    "chain_v2_then_v4": "a receipt v4 head has an earlier revision with another receipt version",
    "chain_v1_then_v4": "a receipt v4 head has an earlier revision with another receipt version",
    "v4_unreadable_superseded": "superseded finalized receipt is not readable JSON",
}


def _copy(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / "staging" / Path(name).name))


def _receipts(root: Path) -> list[dict]:
    return [
        json.loads(path.read_bytes())
        for path in sorted((root / BINDINGS / "receipts").glob("obsctx_*.json"))
    ]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_v4_sealer_fixture_is_admitted_byte_for_byte(tmp_path, name):
    root = _copy(tmp_path, name)
    verified = verify_transfer_snapshot(root)
    assert json.loads((root / MARKER_NAME).read_bytes())["schema_version"] == 4
    assert verified.snapshot["schema_version"] == 3
    assert len(verified.snapshot.get("citrus_artifacts", [])) == EXPECTED[name]
    assert {r["schema_version"] for r in _receipts(root)} == {4}
    rebuilt = transfer.build_snapshot(root, MARKER_NAME, destination=True)
    assert canonical_bytes(rebuilt) == (root / SNAPSHOT_PATH).read_bytes()
    assert poller.check_marker(root / MARKER_NAME)["schema_version"] == 4


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_sources_the_sealer_refuses_are_refused(tmp_path, name):
    root = _copy(tmp_path, f"refused_sources/{name}")
    with pytest.raises(TransferSnapshotError, match=REFUSED[name]):
        transfer.build_snapshot(root, MARKER_NAME)


def test_the_v4_schemas_are_the_pinned_contract_and_the_fixtures_satisfy_them():
    assert hashlib.sha256(RECEIPT_V4_SCHEMA.read_bytes()).hexdigest() == RECEIPT_V4_SCHEMA_SHA256
    assert hashlib.sha256(REFERENCE_V2_SCHEMA.read_bytes()).hexdigest() == REFERENCE_V2_SCHEMA_SHA256
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(RECEIPT_V4_SCHEMA.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for name in ("bound_v4_video", "bound_v4_dish_group"):
        receipts = _receipts(FIXTURES / name)
        assert len(receipts) == 2
        for receipt in receipts:
            validator.validate(receipt)
    group = _receipts(FIXTURES / "bound_v4_dish_group")[0]["contract"]["subject_reference"]
    assert (group["subject_type"], group["subject_id"]) == ("dish_group", None)
    # The ported structural rule is the schema's key sets, nothing else.
    reference = schema["$defs"]["subjectReference"]
    assert set(reference["properties"]) == SUBJECT_REFERENCE_KEYS
    assert set(reference["required"]) == SUBJECT_REFERENCE_KEYS
    assert reference["properties"]["schema_version"] == {"const": 2}
    for key, keys in SUBJECT_REFERENCE_OBJECT_KEYS.items():
        assert set(reference["properties"][key]["properties"]) == keys, key
        assert set(reference["properties"][key]["required"]) == keys, key


def test_reference_versions_pair_strictly_with_receipt_versions():
    v2 = _receipts(FIXTURES / "bound_v4_video")[0]["contract"]["subject_reference"]
    check_subject_reference(v2, receipt_version=4)
    with pytest.raises(TransferSnapshotError, match="receipt v3 subject_reference is not citrus.subject_reference v1"):
        check_subject_reference(v2, receipt_version=3)
    v1 = json.loads(json.dumps(v2))
    v1["schema_version"] = 1
    check_subject_reference(v1, receipt_version=3)
    with pytest.raises(TransferSnapshotError, match="receipt v4 subject_reference is not citrus.subject_reference v2"):
        check_subject_reference(v1, receipt_version=4)

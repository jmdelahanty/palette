from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.recording_import_receipt import (
    CURRENT_RECORDING_IMPORT_PRODUCER_ID,
    MAX_RECEIPT_BYTES,
    RecordingImportReceipt,
    RecordingImportReceiptError,
    publish_recording_import_receipt,
    recording_import_receipt_path,
    recording_import_receipt_paths,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    SourceRecordingIdentity,
    SourceRecordingIdentityClaim,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "a" * 64


def _identity() -> SourceRecordingIdentityClaim:
    return SourceRecordingIdentityClaim.create(
        SourceRecordingIdentity.from_mapping(
            {
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: (
                    SOURCE_RECORDING_IDENTITY_PROFILE
                ),
                "recording_id": "recording-a",
                "session_uuid": "session-a",
                "camera_id": "2010093",
            }
        )
    )


def _receipt() -> RecordingImportReceipt:
    return RecordingImportReceipt.create(
        producer_id=CURRENT_RECORDING_IMPORT_PRODUCER_ID,
        producer_git_sha=GIT_SHA,
        config_sha256=DIGEST,
        target_relative_path="recordings/recording-a.zarr",
        identity_claim=_identity(),
        acquisition_ownership_ref="ownership/recording-a.json",
        acquisition_ownership_sha256=DIGEST,
        acquisition_frame_ref="frames/recording-a.parquet",
        acquisition_frame_sha256="b" * 64,
    )


def test_receipt_is_small_canonical_and_round_trips() -> None:
    receipt = _receipt()
    payload = receipt.to_json_bytes()

    assert len(payload) < MAX_RECEIPT_BYTES
    assert payload == json.dumps(
        receipt.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert receipt.receipt_sha256 == canonical_json_sha256(
        {key: value for key, value in receipt.as_dict().items() if key != "receipt_sha256"}
    )
    assert RecordingImportReceipt.from_json_bytes(payload) == receipt


def test_receipt_can_be_loaded_from_a_stable_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "recording_import_receipt.json"
    path.write_bytes(_receipt().to_json_bytes())

    assert RecordingImportReceipt.from_path(path) == _receipt()


def test_receipt_loader_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "outside.json"
    target.write_bytes(_receipt().to_json_bytes())
    link = tmp_path / "receipt.json"
    link.symlink_to(target)

    with pytest.raises(RecordingImportReceiptError, match="symbolic link"):
        RecordingImportReceipt.from_path(link)


def test_receipt_publication_is_atomic_digest_named_and_idempotent(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    zarr_path = tmp_path / "recording.zarr"

    path = publish_recording_import_receipt(zarr_path, receipt)

    assert path == recording_import_receipt_path(zarr_path, receipt.receipt_sha256)
    assert RecordingImportReceipt.from_path(path) == receipt
    assert publish_recording_import_receipt(zarr_path, receipt) == path
    assert recording_import_receipt_paths(zarr_path) == (path,)
    assert list(path.parent.glob("*.tmp")) == []


def test_receipt_publication_rejects_redirected_sidecar_directory(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    zarr_path = tmp_path / "recording.zarr"
    zarr_path.mkdir()
    (zarr_path / ".imports").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RecordingImportReceiptError, match="symbolic link"):
        publish_recording_import_receipt(zarr_path, _receipt())
    with pytest.raises(RecordingImportReceiptError, match="symbolic link"):
        recording_import_receipt_paths(zarr_path)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "payload_mutator",
    [
        lambda payload: payload.replace(
            b'"claim_sha256":"' + _identity().claim_sha256.encode() + b'"',
            b'"claim_sha256":"' + _identity().claim_sha256.encode() + b'",'
            b'"claim_sha256":"' + _identity().claim_sha256.encode() + b'"',
        ),
        lambda payload: payload.replace(
            b'"receipt_sha256":"' + _receipt().receipt_sha256.encode() + b'"',
            b'"receipt_sha256":NaN',
        ),
    ],
    ids=["duplicate-key", "non-finite-number"],
)
def test_json_parser_rejects_duplicate_keys_and_nonfinite_numbers(payload_mutator) -> None:
    with pytest.raises(RecordingImportReceiptError):
        RecordingImportReceipt.from_json_bytes(payload_mutator(_receipt().to_json_bytes()))


def test_receipt_rejects_unknown_payloads_and_bad_self_digest() -> None:
    raw = _receipt().as_dict()
    raw["manifest"] = {"all": "the payload"}
    with pytest.raises(RecordingImportReceiptError):
        RecordingImportReceipt.from_mapping(raw)

    raw = _receipt().as_dict()
    raw["producer"]["id"] = "retired.recording.importer"
    with pytest.raises(RecordingImportReceiptError, match="current recording importer"):
        RecordingImportReceipt.from_mapping(raw)

    raw = _receipt().as_dict()
    raw["receipt_sha256"] = "c" * 64
    with pytest.raises(RecordingImportReceiptError):
        RecordingImportReceipt.from_mapping(raw)


def test_identity_claim_is_reconstructed_from_canonical_identity() -> None:
    raw = _receipt().as_dict()
    raw["identity_claim"]["identity"]["recording_id"] = "different-recording"

    with pytest.raises(RecordingImportReceiptError, match="identity_claim"):
        RecordingImportReceipt.from_mapping(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity_profile", "wrong.profile"),
        ("producer_git_sha", GIT_SHA.upper()),
        ("producer_git_sha", "0" * 39),
        ("producer_git_dirty", True),
        ("config_sha256", "not-a-digest"),
        ("target_relative_path", "/absolute/archive.zarr"),
        ("target_relative_path", "../outside/archive.zarr"),
    ],
)
def test_receipt_rejects_invalid_contract_fields(field: str, value: str) -> None:
    kwargs = {
        "producer_id": "producer",
        "producer_git_sha": GIT_SHA,
        "config_sha256": DIGEST,
        "target_relative_path": "archive.zarr",
        "identity_claim": _identity(),
        "acquisition_ownership_ref": "ownership.json",
        "acquisition_ownership_sha256": DIGEST,
        "acquisition_frame_ref": "frames.parquet",
        "acquisition_frame_sha256": DIGEST,
    }
    kwargs[field] = value

    with pytest.raises(RecordingImportReceiptError):
        RecordingImportReceipt.create(**kwargs)


def test_receipt_parser_has_a_hard_size_bound() -> None:
    with pytest.raises(RecordingImportReceiptError, match="exceeds"):
        RecordingImportReceipt.from_json_bytes(b" " * (MAX_RECEIPT_BYTES + 1))

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fisheye.shared.zarr.coordinate_successor_files import (
    PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID,
    PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION,
    PayloadFileEquivalenceError,
    copy_metadata_and_link_payload,
    validate_payload_file_equivalence,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _copied_payload(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "array" / "c").mkdir(parents=True)
    (source / "zarr.json").write_text('{"node_type":"group"}\n')
    (source / "array" / "zarr.json").write_text('{"node_type":"array"}\n')
    (source / "array" / "c" / "0").write_bytes(b"compressed-payload")
    (source / "array" / "c" / "1").write_bytes(b"another-payload")
    copy_metadata_and_link_payload(source, target)
    return source, target


def test_equivalence_receipt_uses_inventory_and_samefile_without_reading_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = _copied_payload(tmp_path)

    def fail_read(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("payload bytes must not be read")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    receipt = validate_payload_file_equivalence(
        source,
        target,
        source_label="subject_mask_runs/source",
        target_label="subject_mask_runs/target",
    )

    assert receipt["schema_id"] == PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID
    assert receipt["schema_version"] == PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION
    assert receipt["payload_file_count"] == 2
    assert receipt["payload_files"] == [
        {"path": "array/c/0", "size_bytes": len(b"compressed-payload")},
        {"path": "array/c/1", "size_bytes": len(b"another-payload")},
    ]
    inventory = {
        "schema_id": PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID,
        "schema_version": PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_files": receipt["payload_files"],
    }
    assert receipt["inventory_digest"] == canonical_json_sha256(inventory)
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    assert receipt["receipt_digest"] == canonical_json_sha256(body)
    assert "source" not in receipt
    assert "target" not in receipt


def test_equivalence_rejects_missing_extra_replaced_and_size_changed_payloads(
    tmp_path: Path,
) -> None:
    source, target = _copied_payload(tmp_path)

    (target / "array" / "c" / "1").unlink()
    with pytest.raises(PayloadFileEquivalenceError, match="missing payload"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label="masks/source",
            target_label="masks/target",
        )

    (target / "array" / "c" / "1").write_bytes(b"replacement!!!!")
    with pytest.raises(PayloadFileEquivalenceError, match="not hard-linked"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label="masks/source",
            target_label="masks/target",
        )

    (target / "array" / "c" / "1").unlink()
    os.link(source / "array" / "c" / "0", target / "array" / "c" / "1")
    with pytest.raises(PayloadFileEquivalenceError, match="size differs"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label="masks/source",
            target_label="masks/target",
        )

    (target / "array" / "c" / "extra").write_bytes(b"extra")
    with pytest.raises(PayloadFileEquivalenceError, match="extra payload"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label="masks/source",
            target_label="masks/target",
        )


@pytest.mark.parametrize(
    "label",
    ["/absolute/path", "../outside", "masks/../source", "masks\\source", ""],
)
def test_equivalence_rejects_unsafe_archive_labels(
    tmp_path: Path, label: str
) -> None:
    source, target = _copied_payload(tmp_path)
    with pytest.raises(PayloadFileEquivalenceError, match="archive-relative"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label=label,
            target_label="masks/target",
        )


def test_equivalence_rejects_nonregular_payload(tmp_path: Path) -> None:
    source, target = _copied_payload(tmp_path)
    fifo = source / "array" / "c" / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(PayloadFileEquivalenceError, match="non-regular"):
        validate_payload_file_equivalence(
            source,
            target,
            source_label="masks/source",
            target_label="masks/target",
        )

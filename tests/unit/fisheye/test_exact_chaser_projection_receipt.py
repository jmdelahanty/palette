from __future__ import annotations

import json

import pytest

import fisheye.analysis_workflows.exact_chaser_projection_receipt as subject
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _child(path, *, relative: bool = False):
    key = path.name.split(".", 1)[0]
    parent = "chaser_relative_frame_runs" if relative else "test_runs"
    return {
        "record_sha256": canonical_json_sha256({"key": key, "relative": relative}),
        "run_path": f"analysis/{parent}/{key}",
        "recording_id": "recording-1",
        "manifest_sha256": canonical_json_sha256({"manifest": key}),
        "payload_digest": canonical_json_sha256({"payload": key}),
    }


def _paths(tmp_path, keys):
    values = {}
    for key in keys:
        path = tmp_path / f"{key}.json"
        path.write_text("{}", encoding="utf-8")
        values[key] = path
    return values


def test_projection_receipt_closes_child_roster_and_validates_digest(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    exact = _paths(tmp_path, subject.EXACT_CHILD_KEYS)
    relative = _paths(tmp_path, subject.RELATIVE_CHILD_KEYS)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda path, **_expected: _child(path),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda path, **_expected: _child(path, relative=True),
    )

    receipt = subject.build_exact_chaser_projection_receipt(
        archive,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit="a" * 40,
        expected_recording_id="recording-1",
    )
    validated = subject.validate_exact_chaser_projection_receipt(
        receipt,
        expected_analysis_zarr=archive,
        expected_recording_id="recording-1",
    )

    assert validated["record_sha256"] == receipt["record_sha256"]
    assert set(validated["exact_children"]) == set(subject.EXACT_CHILD_KEYS)
    assert set(validated["relative_frame_children"]) == set(subject.RELATIVE_CHILD_KEYS)
    tampered = json.loads(json.dumps(receipt))
    tampered["recording_id"] = "other-recording"
    with pytest.raises(
        subject.ExactChaserProjectionReceiptError, match="digest is stale"
    ):
        subject.validate_exact_chaser_projection_receipt(tampered)


def test_projection_identity_validation_does_not_open_child_receipts(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    exact = _paths(tmp_path, subject.EXACT_CHILD_KEYS)
    relative = _paths(tmp_path, subject.RELATIVE_CHILD_KEYS)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda path, **_expected: _child(path),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda path, **_expected: _child(path, relative=True),
    )
    receipt = subject.build_exact_chaser_projection_receipt(
        archive,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit="b" * 40,
        expected_recording_id="recording-1",
    )
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda *_args, **_kwargs: pytest.fail("identity check opened exact child"),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda *_args, **_kwargs: pytest.fail("identity check opened relative child"),
    )

    validated = subject.validate_exact_chaser_projection_receipt(
        receipt,
        expected_analysis_zarr=archive,
        validate_current_metadata=False,
        validate_child_receipts=False,
    )

    assert validated["record_sha256"] == receipt["record_sha256"]


def test_projection_receipt_v2_requires_and_binds_exact_gaze_child(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    exact = _paths(tmp_path, subject.EXACT_CHILD_KEYS_V2)
    relative = _paths(tmp_path, subject.RELATIVE_CHILD_KEYS)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda path, **_expected: _child(path),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda path, **_expected: _child(path, relative=True),
    )

    receipt = subject.build_exact_chaser_projection_receipt(
        archive,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit="c" * 40,
        expected_recording_id="recording-1",
    )
    validated = subject.validate_exact_chaser_projection_receipt(
        receipt,
        expected_analysis_zarr=archive,
        expected_recording_id="recording-1",
    )

    assert validated["schema_version"] == 2
    assert set(validated["exact_children"]) == set(subject.EXACT_CHILD_KEYS_V2)
    assert validated["exact_children"]["gaze"]["run_path"].endswith("/gaze")


def test_projection_receipt_v1_remains_closed_without_gaze(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    exact = _paths(tmp_path, subject.EXACT_CHILD_KEYS)
    relative = _paths(tmp_path, subject.RELATIVE_CHILD_KEYS)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda path, **_expected: _child(path),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda path, **_expected: _child(path, relative=True),
    )

    receipt = subject.build_exact_chaser_projection_receipt(
        archive,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit="d" * 40,
        expected_recording_id="recording-1",
    )

    assert receipt["schema_version"] == 1
    assert "gaze" not in receipt["exact_children"]


@pytest.mark.parametrize(
    ("keys", "schema_version", "expected_optional"),
    (
        (subject.EXACT_CHILD_KEYS_V3, 3, {"epoch_behavior"}),
        (subject.EXACT_CHILD_KEYS_V4, 4, {"gaze", "epoch_behavior"}),
        (
            subject.EXACT_CHILD_KEYS_V5,
            5,
            {"body_alignment_by_distance"},
        ),
        (
            subject.EXACT_CHILD_KEYS_V6,
            6,
            {"gaze", "body_alignment_by_distance"},
        ),
        (
            subject.EXACT_CHILD_KEYS_V7,
            7,
            {"epoch_behavior", "body_alignment_by_distance"},
        ),
        (
            subject.EXACT_CHILD_KEYS_V8,
            8,
            {"gaze", "epoch_behavior", "body_alignment_by_distance"},
        ),
    ),
)
def test_projection_receipt_closes_epoch_and_combined_rosters(
    tmp_path,
    monkeypatch,
    keys,
    schema_version,
    expected_optional,
) -> None:
    archive = tmp_path / "analysis.zarr"
    archive.mkdir()
    exact = _paths(tmp_path, keys)
    relative = _paths(tmp_path, subject.RELATIVE_CHILD_KEYS)
    monkeypatch.setattr(
        subject,
        "read_exact_immutable_child_validation_receipt",
        lambda path, **_expected: _child(path),
    )
    monkeypatch.setattr(
        subject,
        "read_chaser_relative_frame_validation_receipt",
        lambda path, **_expected: _child(path, relative=True),
    )

    receipt = subject.build_exact_chaser_projection_receipt(
        archive,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit="e" * 40,
        expected_recording_id="recording-1",
    )
    validated = subject.validate_exact_chaser_projection_receipt(
        receipt,
        expected_analysis_zarr=archive,
        expected_recording_id="recording-1",
    )

    assert validated["schema_version"] == schema_version
    assert set(validated["exact_children"]) == set(keys)
    assert expected_optional.issubset(validated["exact_children"])

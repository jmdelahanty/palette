from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.source_recording_identity import (
    MAX_RECORDING_MANIFEST_BYTES,
    SOURCE_ANALYSIS_CLASSIFICATION,
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    SourceRecordingIdentity,
    SourceRecordingIdentityClaim,
    SourceRecordingIdentityError,
    load_source_recording_identity_claim,
    load_source_recording_identity_profile,
)


def _write_root(tmp_path: Path, metadata: str | dict) -> Path:
    root = tmp_path / "recording.zarr"
    root.mkdir(parents=True)
    if isinstance(metadata, str):
        raw = metadata
    else:
        raw = json.dumps(metadata)
    (root / "zarr.json").write_text(raw, encoding="utf-8")
    return root


_UNSET = object()


def _metadata(attributes: object = _UNSET, **overrides: object) -> dict:
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {} if attributes is _UNSET else attributes,
    }
    document.update(overrides)
    return document


def _write_recording_pair(
    tmp_path: Path,
    *,
    manifest: dict[str, object],
    root_attributes: dict[str, object],
) -> tuple[Path, Path]:
    recording_dir = tmp_path / "recordings" / "recording-a"
    zarr_root = recording_dir / "zarr" / "recording-a.zarr"
    zarr_root.mkdir(parents=True)
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (zarr_root / "zarr.json").write_text(
        json.dumps(_metadata(root_attributes)),
        encoding="utf-8",
    )
    return manifest_path, zarr_root


def test_current_direct_root_profile_is_exact_value(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path,
        _metadata({SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE}),
    )

    assert load_source_recording_identity_profile(root) == SOURCE_RECORDING_IDENTITY_PROFILE


def test_unprofiled_v3_group_is_legacy(tmp_path: Path) -> None:
    root = _write_root(tmp_path, _metadata({"recording_id": "legacy"}))

    assert load_source_recording_identity_profile(root) is None


def test_v2_root_without_v3_metadata_remains_legacy(tmp_path: Path) -> None:
    root = tmp_path / "legacy-v2.zarr"
    root.mkdir()
    (root / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")

    assert load_source_recording_identity_profile(root) is None


@pytest.mark.parametrize(
    "attributes",
    [
        {SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: None},
        {SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: 7},
        {SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE + ".other"},
    ],
    ids=("null", "non_string", "unknown"),
)
def test_present_profile_must_be_exact_supported_string(
    tmp_path: Path, attributes: dict[str, object]
) -> None:
    root = _write_root(tmp_path, _metadata(attributes))

    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(root)


@pytest.mark.parametrize(
    "metadata",
    [
        _metadata(zarr_format=2),
        _metadata(zarr_format=3.0),
        _metadata(node_type="array"),
        _metadata(node_type=None),
        _metadata(attributes=None),
        _metadata(attributes=[]),
    ],
    ids=("v2", "non_integer_v3", "array", "null_node_type", "null_attrs", "list_attrs"),
)
def test_root_metadata_must_be_direct_v3_group_with_object_attributes(
    tmp_path: Path, metadata: dict
) -> None:
    root = _write_root(tmp_path, metadata)

    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(root)


def test_strict_metadata_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = _write_root(
        tmp_path / "duplicate",
        '{"zarr_format":3,"node_type":"group","attributes":{},'
        '"attributes":{}}',
    )
    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(duplicate)

    nonfinite = _write_root(
        tmp_path / "nonfinite",
        '{"zarr_format":3,"node_type":"group","attributes":{"x":NaN}}',
    )
    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(nonfinite)


def test_metadata_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "oversized"
    root.mkdir()
    (root / "zarr.json").write_bytes(
        b'{"zarr_format":3,"node_type":"group","attributes":{"x":"'
        + b"a" * MAX_RECORDING_MANIFEST_BYTES
        + b'"}}'
    )

    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(root)


def test_missing_direct_metadata_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "missing.zarr"
    root.mkdir()

    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(root)


def test_non_directory_root_is_rejected(tmp_path: Path) -> None:
    metadata = tmp_path / "zarr.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")

    with pytest.raises(SourceRecordingIdentityError):
        load_source_recording_identity_profile(metadata)


def test_current_profile_requires_exact_manifest_root_pair(tmp_path: Path) -> None:
    identity = SourceRecordingIdentity(
        recording_id="recording-a",
        session_uuid="session-a",
        camera_id="camera-a",
    )
    manifest_path, root = _write_recording_pair(
        tmp_path,
        manifest=identity.manifest_fields(),
        root_attributes=identity.analysis_root_fields(),
    )

    claim = load_source_recording_identity_claim(manifest_path, root)

    assert load_source_recording_identity_profile(root) == SOURCE_RECORDING_IDENTITY_PROFILE
    assert claim == SourceRecordingIdentityClaim.create(identity)
    assert SourceRecordingIdentityClaim.from_mapping(claim.as_dict()) == claim


def test_one_sided_current_profile_is_not_legacy(tmp_path: Path) -> None:
    identity = SourceRecordingIdentity(
        recording_id="recording-a",
        session_uuid="session-a",
        camera_id="camera-a",
    )
    _manifest_path, missing_root_profile = _write_recording_pair(
        tmp_path / "missing-root",
        manifest=identity.manifest_fields(),
        root_attributes={"recording_id": identity.recording_id},
    )
    with pytest.raises(SourceRecordingIdentityError, match="profiles disagree"):
        load_source_recording_identity_profile(missing_root_profile)

    unprofiled_manifest = {
        "recording_id": identity.recording_id,
        "session_uuid": identity.session_uuid,
        "camera_id": identity.camera_id,
    }
    _manifest_path, current_root = _write_recording_pair(
        tmp_path / "missing-manifest",
        manifest=unprofiled_manifest,
        root_attributes=identity.analysis_root_fields(),
    )
    with pytest.raises(SourceRecordingIdentityError, match="profiles disagree"):
        load_source_recording_identity_profile(current_root)


def test_current_manifest_allows_explicit_non_source_sibling(tmp_path: Path) -> None:
    identity = SourceRecordingIdentity(
        recording_id="recording-a",
        session_uuid="session-a",
        camera_id="camera-a",
    )
    _manifest_path, root = _write_recording_pair(
        tmp_path,
        manifest=identity.manifest_fields(),
        root_attributes={
            "artifact_kind": "derived_training_merge",
            "zarr_use": "training",
            "zarr_purpose": "training",
        },
    )

    assert load_source_recording_identity_profile(root) is None


def test_unprofiled_exact_source_classification_is_ambiguous(tmp_path: Path) -> None:
    _manifest_path, root = _write_recording_pair(
        tmp_path,
        manifest={"recording_id": "legacy"},
        root_attributes=dict(SOURCE_ANALYSIS_CLASSIFICATION),
    )

    with pytest.raises(SourceRecordingIdentityError, match="missing its current"):
        load_source_recording_identity_profile(root)


def test_current_pair_rejects_camera_divergence(tmp_path: Path) -> None:
    manifest_identity = SourceRecordingIdentity(
        recording_id="recording-a",
        session_uuid="session-a",
        camera_id="camera-a",
    )
    root_identity = SourceRecordingIdentity(
        recording_id="recording-a",
        session_uuid="session-a",
        camera_id="camera-b",
    )
    _manifest_path, root = _write_recording_pair(
        tmp_path,
        manifest=manifest_identity.manifest_fields(),
        root_attributes=root_identity.analysis_root_fields(),
    )

    with pytest.raises(SourceRecordingIdentityError, match="camera_id conflict"):
        load_source_recording_identity_profile(root)

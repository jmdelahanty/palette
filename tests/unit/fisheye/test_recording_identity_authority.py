from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from uuid import UUID

import pytest

from fisheye.registry.db import Registry
from fisheye.registry import recording_identity_authority as authority_module
from fisheye.registry.recording_identity_authority import (
    IDENTITY_REVISION_SCHEMA_ID,
    RecordingIdentityAuthorityError,
    RecordingIdentityProjectionConflict,
    canonical_dataset_path_hash,
    collect_regular_source_recording_identity,
)
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.shared.recording_import_receipt import (
    CURRENT_RECORDING_IMPORT_PRODUCER_ID,
    RecordingImportReceipt,
    RecordingImportReceiptError,
    publish_recording_import_receipt,
    recording_import_receipt_path,
)


DECIDED_AT = "2026-08-25T12:00:00+00:00"
_UNSET = object()


def _write_artifact(
    tmp_path: Path,
    *,
    name: str = "recording-a",
    recording_id: object = "recording-a",
    session_uuid: object = "session-a",
    manifest_recording_id: object = _UNSET,
    manifest_session_uuid: object = _UNSET,
    root_recording_id: object = _UNSET,
    root_session_uuid: object = _UNSET,
    manifest_camera_id: object = "2010093",
    root_camera_id: object = "2010093",
) -> Path:
    recording_dir = tmp_path / "recordings" / name
    zarr_path = recording_dir / "zarr" / "analysis.zarr"
    zarr_path.mkdir(parents=True)
    manifest = {
        SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
        "recording_id": (
            recording_id if manifest_recording_id is _UNSET else manifest_recording_id
        ),
        "session_uuid": (
            session_uuid if manifest_session_uuid is _UNSET else manifest_session_uuid
        ),
        "camera_id": manifest_camera_id,
    }
    root_attrs = {
        SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
        "recording_id": recording_id if root_recording_id is _UNSET else root_recording_id,
        "session_uuid": session_uuid if root_session_uuid is _UNSET else root_session_uuid,
        "camera_id": root_camera_id,
        "artifact_schema_id": "recording_analysis_v1",
        "artifact_kind": "source_recording",
        "zarr_origin": "source",
        "zarr_use": "analysis",
        "zarr_purpose": "analysis",
    }
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (zarr_path / "zarr.json").write_text(
        json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": root_attrs}
        ),
        encoding="utf-8",
    )
    return zarr_path


def _project(
    registry: Registry,
    zarr_path: Path,
    *,
    decided_at: str = DECIDED_AT,
    import_receipt: object = _UNSET,
):
    if import_receipt is _UNSET:
        import_receipt = _receipt_for(zarr_path)
    return registry.project_regular_source_recording_identity(
        zarr_path=zarr_path,
        decided_by="pytest",
        decided_at_utc=decided_at,
        import_receipt=import_receipt,
    )


def _receipt_for(
    zarr_path: Path,
    *,
    publish: bool = True,
    authority: bool = False,
) -> RecordingImportReceipt:
    """Create the compact receipt used by authority-binding tests."""

    claim = collect_regular_source_recording_identity(zarr_path)
    recording_dir = zarr_path.parent.parent
    ownership_ref = (
        "/analysis/acquisition_camera_frames/2010093@acquisition_import_ownership"
        if authority
        else "registry://acquisition/ownership/test"
    )
    frame_ref = (
        "/analysis/acquisition_camera_frames/2010093@acquisition_camera_frame"
        if authority
        else "registry://acquisition/frame/test"
    )
    ownership_sha256 = "2" * 64
    frame_sha256 = "3" * 64
    receipt = RecordingImportReceipt.create(
        producer_id=CURRENT_RECORDING_IMPORT_PRODUCER_ID,
        producer_git_sha="0" * 40,
        config_sha256="1" * 64,
        target_relative_path=zarr_path.relative_to(recording_dir).as_posix(),
        identity_claim=claim,
        acquisition_ownership_ref=ownership_ref,
        acquisition_ownership_sha256=ownership_sha256,
        acquisition_frame_ref=frame_ref,
        acquisition_frame_sha256=frame_sha256,
    )
    if publish:
        publish_recording_import_receipt(zarr_path, receipt)
    return receipt


def _receipt_authority_counts(registry: Registry) -> dict[str, int]:
    return {
        table: int(
            registry.conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        )
        for table in (
            "recordings",
            "datasets",
            "recording_identity_evidence",
            "recording_identity_revisions",
            "recording_identity_current",
            "dataset_recording_identity_current",
            "recording_import_receipt_bindings",
        )
    }


def _stub_live_receipt_verification(
    monkeypatch: pytest.MonkeyPatch,
    receipt: RecordingImportReceipt,
    *,
    error: Exception | None = None,
) -> None:
    ownership = SimpleNamespace(
        record_ref=receipt.acquisition_ownership_ref,
        record_sha256=receipt.acquisition_ownership_sha256,
    )
    frame = SimpleNamespace(
        record_ref=receipt.acquisition_frame_ref,
        record_sha256=receipt.acquisition_frame_sha256,
    )

    def verify(**_kwargs):
        if error is not None:
            raise error
        return (
            receipt.receipt_sha256,
            ownership,
            frame,
            "analysis/acquisition_camera_frames/2010093",
        )

    monkeypatch.setattr(authority_module, "_verify_live_import_receipt", verify)


@pytest.fixture(autouse=True)
def _default_live_receipt_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verify(*, receipt: RecordingImportReceipt, **_kwargs):
        return (
            receipt.receipt_sha256,
            SimpleNamespace(
                record_ref=receipt.acquisition_ownership_ref,
                record_sha256=receipt.acquisition_ownership_sha256,
            ),
            SimpleNamespace(
                record_ref=receipt.acquisition_frame_ref,
                record_sha256=receipt.acquisition_frame_sha256,
            ),
            "analysis/acquisition_camera_frames/2010093",
        )

    monkeypatch.setattr(authority_module, "_verify_live_import_receipt", verify)


def _insert_fixture_revision(
    registry: Registry,
    *,
    result: object,
    identity_snapshot_id: str,
    identity_revision: int,
    supersedes_identity_snapshot_id: str | None,
    revision_kind: str,
    correction_reason: str | None,
) -> None:
    registry.conn.execute(
        """
        INSERT INTO recording_identity_revisions(
            identity_snapshot_id, identity_scope_id, recording_id,
            session_uuid, identity_revision,
            supersedes_identity_snapshot_id, schema_id, revision_kind,
            decided_by, decided_at_utc, correction_reason,
            evidence_digest, initiating_dataset_id,
            registry_schema_version
        ) VALUES (?, ?, 'recording-a', 'session-a', ?, ?, ?, ?,
                  'fixture', '2026-08-25T13:00:00+00:00', ?, ?, ?, 72);
        """,
        (
            identity_snapshot_id,
            result.identity_scope_id,  # type: ignore[attr-defined]
            identity_revision,
            supersedes_identity_snapshot_id,
            IDENTITY_REVISION_SCHEMA_ID,
            revision_kind,
            correction_reason,
            result.evidence_digest,  # type: ignore[attr-defined]
            result.dataset_id,  # type: ignore[attr-defined]
        ),
    )


def test_collector_binds_manifest_and_direct_root_to_target(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)

    claim = collect_regular_source_recording_identity(zarr_path)

    assert claim.identity.recording_id == "recording-a"
    assert claim.identity.session_uuid == "session-a"
    assert claim.identity.camera_id == "2010093"
    assert claim.claim_sha256 == claim.as_dict()["claim_sha256"]
    assert claim.as_dict()["verified_source_roles"] == [
        "recording_manifest",
        "zarr_root_direct_metadata",
    ]


def test_v3_root_key_order_does_not_change_identity_evidence(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    first = collect_regular_source_recording_identity(zarr_path)
    root_path = zarr_path / "zarr.json"
    document = json.loads(root_path.read_text(encoding="utf-8"))
    reordered = {
        "attributes": document["attributes"],
        "node_type": document["node_type"],
        "zarr_format": document["zarr_format"],
    }
    root_path.write_text(json.dumps(reordered), encoding="utf-8")

    second = collect_regular_source_recording_identity(zarr_path)

    assert second.claim_sha256 == first.claim_sha256


def test_v2_root_is_deferred_from_current_authority(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    (zarr_path / "zarr.json").unlink()
    (zarr_path / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}),
        encoding="utf-8",
    )
    (zarr_path / ".zattrs").write_text(
        json.dumps(
            {
                "recording_id": "recording-a",
                "session_uuid": "session-a",
                "zarr_purpose": "analysis",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RecordingIdentityAuthorityError,
        match="requires a Zarr v3 root",
    ):
        collect_regular_source_recording_identity(zarr_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (SOURCE_RECORDING_IDENTITY_PROFILE_ATTR, None, SOURCE_RECORDING_IDENTITY_PROFILE_ATTR),
        ("artifact_schema_id", None, "artifact_schema_id"),
        ("artifact_kind", None, "artifact_kind"),
        ("zarr_origin", None, "zarr_origin"),
        ("zarr_use", None, "zarr_use"),
        ("zarr_purpose", None, "zarr_purpose"),
        ("artifact_kind", "derived", "artifact_kind"),
        ("artifact_kind", "SOURCE_RECORDING", "artifact_kind"),
        ("zarr_origin", "imported", "zarr_origin"),
        ("zarr_use", "training", "zarr_use"),
        ("zarr_purpose", "training", "zarr_purpose"),
        ("zarr_purpose", "production", "zarr_purpose"),
    ],
)
def test_current_authority_requires_supported_root_profile(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    root_path = zarr_path / "zarr.json"
    document = json.loads(root_path.read_text(encoding="utf-8"))
    if value is None:
        document["attributes"].pop(field)
    else:
        document["attributes"][field] = value
    root_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RecordingIdentityAuthorityError, match=message):
        collect_regular_source_recording_identity(zarr_path)


def test_current_authority_requires_marked_manifest_identity(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    manifest_path = zarr_path.parent.parent / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(SOURCE_RECORDING_IDENTITY_PROFILE_ATTR)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        RecordingIdentityAuthorityError,
        match=SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
    ):
        collect_regular_source_recording_identity(zarr_path)


def test_current_authority_rejects_manifest_root_camera_conflict(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path, root_camera_id="2010094")

    with pytest.raises(RecordingIdentityAuthorityError, match="camera_id conflict"):
        collect_regular_source_recording_identity(zarr_path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"zarr_format":3,"node_type":"group","attributes":{},"attributes":{}}',
        '{"zarr_format":3,"node_type":"group","attributes":{"value":NaN}}',
        '{"zarr_format":3,"node_type":"group","attributes":{}} trailing',
    ],
)
def test_strict_v3_root_rejects_invalid_json(
    tmp_path: Path,
    payload: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    (zarr_path / "zarr.json").write_text(payload, encoding="utf-8")

    with pytest.raises(RecordingIdentityAuthorityError, match="strict JSON"):
        collect_regular_source_recording_identity(zarr_path)


def test_unrelated_manifest_metadata_does_not_change_identity_evidence(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    first = collect_regular_source_recording_identity(zarr_path)
    manifest_path = zarr_path.parent.parent / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["operator_note"] = "unrelated to identity"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second = collect_regular_source_recording_identity(zarr_path)

    assert second.claim_sha256 == first.claim_sha256


@pytest.mark.parametrize("field", ["recording_id", "session_uuid"])
def test_source_conflict_cannot_mint_receipt_or_registry_state(
    tmp_path: Path,
    field: str,
) -> None:
    kwargs = {f"manifest_{field}": f"different-{field}"}
    zarr_path = _write_artifact(tmp_path, **kwargs)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="manifest/root .* conflict",
        ):
            _project(registry, zarr_path)

        assert registry.conn.execute("SELECT COUNT(*) FROM recordings;").fetchone()[0] == 0
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_evidence;"
        ).fetchone()[0] == 0
    finally:
        registry.close()


@pytest.mark.parametrize(
    "root_value",
    [None, " bad ", "bad\nid"],
)
def test_missing_or_malformed_root_identity_fails_closed(
    tmp_path: Path,
    root_value: object,
) -> None:
    zarr_path = _write_artifact(tmp_path, root_recording_id=root_value)
    with pytest.raises(RecordingIdentityAuthorityError, match="recording_id"):
        collect_regular_source_recording_identity(zarr_path)


def test_fresh_projection_mints_canonical_dataset_and_bound_revision(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)

        assert result.disposition == "authority_created"
        assert str(UUID(result.identity_scope_id)) == result.identity_scope_id
        assert str(UUID(result.identity_snapshot_id)) == result.identity_snapshot_id
        assert result.identity_revision == 1
        assert result.dataset_id.startswith("session-a:z")
        recording = registry.conn.execute(
            "SELECT * FROM recordings WHERE recording_id='recording-a';"
        ).fetchone()
        dataset = registry.conn.execute(
            "SELECT * FROM datasets WHERE dataset_id=?;", (result.dataset_id,)
        ).fetchone()
        revision = registry.conn.execute(
            "SELECT * FROM recording_identity_revisions;"
        ).fetchone()
        binding = registry.conn.execute(
            "SELECT * FROM dataset_recording_identity_current;"
        ).fetchone()
        evidence_row = registry.conn.execute(
            "SELECT * FROM recording_identity_evidence;"
        ).fetchone()

        assert recording["session_uuid"] == "session-a"
        assert dataset["recording_id"] == "recording-a"
        assert dataset["session_uuid"] == "session-a"
        # Identity projection does not own artifact classification or lifecycle.
        assert dataset["artifact_kind"] is None
        assert dataset["zarr_origin"] is None
        assert dataset["zarr_use"] is None
        assert dataset["status"] is None
        assert revision["schema_id"] == IDENTITY_REVISION_SCHEMA_ID
        assert revision["evidence_digest"] == result.evidence_digest
        assert binding["identity_snapshot_id"] == result.identity_snapshot_id
        stored = json.loads(evidence_row["evidence_json"])
        assert stored["claim_sha256"] == result.evidence_digest
        assert registry.conn.execute("PRAGMA foreign_key_check;").fetchall() == []
    finally:
        registry.close()


def test_receipt_projection_inserts_exact_binding_and_returns_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path)
    _stub_live_receipt_verification(monkeypatch, receipt)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path, import_receipt=receipt)

        assert result.receipt_sha256 == receipt.receipt_sha256
        binding = registry.conn.execute(
            """
            SELECT receipt_sha256, dataset_id, identity_scope_id,
                   identity_snapshot_id, bound_by, bound_at_utc,
                   registry_schema_version
            FROM recording_import_receipt_bindings;
            """
        ).fetchone()
        assert tuple(binding) == (
            receipt.receipt_sha256,
            result.dataset_id,
            result.identity_scope_id,
            result.identity_snapshot_id,
            "pytest",
            DECIDED_AT,
            72,
        )
        assert _receipt_authority_counts(registry)[
            "recording_import_receipt_bindings"
        ] == 1
    finally:
        registry.close()


def test_receipt_projection_exact_replay_is_binding_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path)
    _stub_live_receipt_verification(monkeypatch, receipt)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        first = _project(registry, zarr_path, import_receipt=receipt)
        second = _project(
            registry,
            zarr_path,
            decided_at="2026-08-25T13:00:00+00:00",
            import_receipt=receipt,
        )

        assert second.disposition == "exact_replay"
        assert second.receipt_sha256 == first.receipt_sha256
        assert second.identity_snapshot_id == first.identity_snapshot_id
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_import_receipt_bindings;"
        ).fetchone()[0] == 1
        binding = registry.conn.execute(
            "SELECT bound_by, bound_at_utc FROM recording_import_receipt_bindings;"
        ).fetchone()
        assert tuple(binding) == ("pytest", DECIDED_AT)
    finally:
        registry.close()


def test_copied_receipt_for_mismatched_target_fails_without_db_mutation(
    tmp_path: Path,
) -> None:
    source_path = _write_artifact(tmp_path, name="source")
    receipt = _receipt_for(source_path)
    target_path = _write_artifact(
        tmp_path,
        name="different-target",
        recording_id="recording-b",
        session_uuid="session-b",
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        before = _receipt_authority_counts(registry)
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="identity claim differs from the target",
        ):
            _project(registry, target_path, import_receipt=receipt)
        assert _receipt_authority_counts(registry) == before
    finally:
        registry.close()


def test_receipt_binds_camera_and_mapping_profile_claim(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path)
    manifest_path = zarr_path.parent.parent / "recording_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["camera_id"] = "2010094"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    root_path = zarr_path / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["attributes"]["camera_id"] = "2010094"
    root_path.write_text(json.dumps(root), encoding="utf-8")

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        before = _receipt_authority_counts(registry)
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="identity claim differs from the target",
        ):
            _project(registry, zarr_path, import_receipt=receipt)
        assert _receipt_authority_counts(registry) == before
    finally:
        registry.close()


@pytest.mark.parametrize("tamper", ["missing", "payload"])
def test_missing_or_tampered_receipt_sidecar_fails_without_db_mutation(
    tmp_path: Path,
    tamper: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path, publish=tamper != "missing")
    if tamper == "payload":
        sidecar = recording_import_receipt_path(zarr_path, receipt.receipt_sha256)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["identity_claim"]["claim_sha256"] = "f" * 64
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        before = _receipt_authority_counts(registry)
        with pytest.raises(RecordingImportReceiptError):
            _project(registry, zarr_path, import_receipt=receipt)
        assert _receipt_authority_counts(registry) == before
    finally:
        registry.close()


def test_identity_projection_always_binds_its_receipt(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)

        assert result.receipt_sha256 is not None
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_import_receipt_bindings;"
        ).fetchone()[0] == 1
    finally:
        registry.close()


def test_stale_acquisition_receipt_fails_before_any_registry_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path)
    _stub_live_receipt_verification(
        monkeypatch,
        receipt,
        error=RecordingIdentityProjectionConflict(
            "fixture stale acquisition receipt"
        ),
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        before = _receipt_authority_counts(registry)
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="stale acquisition receipt",
        ):
            _project(registry, zarr_path, import_receipt=receipt)
        assert _receipt_authority_counts(registry) == before
    finally:
        registry.close()


def test_receipt_binding_insert_failure_rolls_back_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path)
    _stub_live_receipt_verification(monkeypatch, receipt)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            """
            CREATE TRIGGER reject_receipt_binding_insert
            BEFORE INSERT ON recording_import_receipt_bindings
            BEGIN SELECT RAISE(ABORT, 'fixture receipt binding failure'); END;
            """
        )
        registry.conn.commit()
        before = _receipt_authority_counts(registry)

        with pytest.raises(sqlite3.IntegrityError, match="fixture receipt binding failure"):
            _project(registry, zarr_path, import_receipt=receipt)

        assert _receipt_authority_counts(registry) == before
    finally:
        registry.close()


def test_read_verified_recording_import_requires_live_published_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    receipt = _receipt_for(zarr_path, authority=True)
    _stub_live_receipt_verification(monkeypatch, receipt)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        projected = _project(registry, zarr_path, import_receipt=receipt)
        verified = registry.read_verified_recording_import(projected.dataset_id)

        assert verified.identity.dataset_id == projected.dataset_id
        assert verified.receipt == receipt
        assert verified.acquisition_ownership.record_sha256 == receipt.acquisition_ownership_sha256
        assert verified.acquisition_frame.record_sha256 == receipt.acquisition_frame_sha256
        assert verified.acquisition_authority_path == (
            "analysis/acquisition_camera_frames/2010093"
        )
    finally:
        registry.close()


@pytest.mark.parametrize(
    "case",
    ["pending", "tampered"],
)
def test_read_verified_recording_import_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        receipt = _receipt_for(zarr_path, authority=True)
        _stub_live_receipt_verification(monkeypatch, receipt)
        projected = _project(
            registry,
            zarr_path,
            import_receipt=receipt,
        )
        _stub_live_receipt_verification(
            monkeypatch,
            receipt,
            error=RecordingIdentityProjectionConflict(
                f"fixture {case} acquisition authority"
            ),
        )
        with pytest.raises(RecordingIdentityProjectionConflict):
            registry.read_verified_recording_import(projected.dataset_id)
    finally:
        registry.close()


def test_exact_replay_is_ledger_idempotent(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        first = _project(registry, zarr_path)
        second = _project(
            registry,
            zarr_path,
            decided_at="2026-08-25T13:00:00+00:00",
        )

        assert second.disposition == "exact_replay"
        assert second.identity_snapshot_id == first.identity_snapshot_id
        for table in (
            "recording_identity_evidence",
            "recording_identity_revisions",
            "recording_identity_current",
            "dataset_recording_identity_current",
        ):
            assert registry.conn.execute(
                f"SELECT COUNT(*) FROM {table};"
            ).fetchone()[0] == 1
        assert registry.conn.execute(
            "SELECT last_seen_utc FROM datasets WHERE dataset_id=?;",
            (first.dataset_id,),
        ).fetchone()[0] is None
    finally:
        registry.close()


def test_unrelated_source_telemetry_does_not_refresh_identity_binding(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    manifest_path = zarr_path.parent.parent / "recording_manifest.json"
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        first = _project(registry, zarr_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["operator_note"] = "new observation"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        second = _project(registry, zarr_path)

        assert second.disposition == "exact_replay"
        assert second.identity_snapshot_id == first.identity_snapshot_id
        assert second.evidence_digest == first.evidence_digest
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_evidence;"
        ).fetchone()[0] == 1
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_revisions;"
        ).fetchone()[0] == 1
        assert registry.read_verified_recording_identity(first.dataset_id)
    finally:
        registry.close()


def test_projection_fills_identity_without_changing_lifecycle(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        expected_id = registry.resolve_effective_dataset_id(
            "session-a", session_uuid="session-a", zarr_path=zarr_path.resolve()
        )
        registry.conn.execute(
            "INSERT INTO recordings(recording_id) VALUES ('recording-a');"
        )
        registry.conn.execute(
            """
            INSERT INTO datasets(
                dataset_id, zarr_path, path_hash, zarr_origin, status
            ) VALUES (?, ?, ?, 'imported', 'missing');
            """,
            (
                expected_id,
                str(zarr_path.resolve()),
                canonical_dataset_path_hash(zarr_path),
            ),
        )
        registry.conn.commit()

        result = _project(registry, zarr_path)

        assert result.dataset_id == expected_id
        row = registry.conn.execute(
            "SELECT * FROM datasets WHERE dataset_id=?;", (expected_id,)
        ).fetchone()
        assert row["recording_id"] == "recording-a"
        assert row["session_uuid"] == "session-a"
        assert row["path_hash"] == canonical_dataset_path_hash(zarr_path)
        assert row["artifact_kind"] is None
        assert row["zarr_origin"] == "imported"
        assert row["zarr_use"] is None
        assert row["status"] == "missing"
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("table", "field", "current"),
    [
        ("recordings", "session_uuid", "session-b"),
        ("datasets", "recording_id", "recording-b"),
        ("datasets", "session_uuid", "session-b"),
    ],
)
def test_existing_identity_conflict_rolls_back(
    tmp_path: Path,
    table: str,
    field: str,
    current: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        if table == "recordings":
            registry.conn.execute(
                "INSERT INTO recordings(recording_id, session_uuid) VALUES ('recording-a', ?);",
                (current,),
            )
        else:
            dataset_id = registry.resolve_effective_dataset_id(
                "session-a", session_uuid="session-a", zarr_path=zarr_path.resolve()
            )
            values = {
                "recording_id": "recording-a",
                "session_uuid": "session-a",
            }
            values[field] = current
            registry.conn.execute(
                """
                INSERT INTO datasets(
                    dataset_id, recording_id, session_uuid, zarr_path,
                    artifact_kind, status
                ) VALUES (?, ?, ?, ?, 'source_recording', 'active');
                """,
                (
                    dataset_id,
                    values["recording_id"],
                    values["session_uuid"],
                    str(zarr_path.resolve()),
                ),
            )
        registry.conn.commit()

        with pytest.raises(RecordingIdentityProjectionConflict, match=field):
            _project(registry, zarr_path)

        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_revisions;"
        ).fetchone()[0] == 0
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_kind", "derived_analysis"),
        ("zarr_use", "training"),
    ],
)
def test_non_regular_dataset_policy_is_not_relabelled(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.resolve_effective_dataset_id(
            "session-a", session_uuid="session-a", zarr_path=zarr_path.resolve()
        )
        registry.conn.execute(
            f"INSERT INTO datasets(dataset_id, zarr_path, path_hash, {field}) "
            "VALUES (?, ?, ?, ?);",
            (
                dataset_id,
                str(zarr_path.resolve()),
                canonical_dataset_path_hash(zarr_path),
                value,
            ),
        )
        registry.conn.commit()

        with pytest.raises(RecordingIdentityProjectionConflict, match=field):
            _project(registry, zarr_path)
    finally:
        registry.close()


def test_existing_exact_path_preserves_legacy_dataset_id(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            """
            INSERT INTO datasets(dataset_id, zarr_path, path_hash, status)
            VALUES ('other-id', ?, ?, 'active');
            """,
            (str(zarr_path.resolve()), canonical_dataset_path_hash(zarr_path)),
        )
        registry.conn.commit()

        result = _project(registry, zarr_path)

        assert result.dataset_id == "other-id"
        row = registry.conn.execute(
            "SELECT recording_id, session_uuid FROM datasets WHERE dataset_id='other-id';"
        ).fetchone()
        assert tuple(row) == ("recording-a", "session-a")
    finally:
        registry.close()


def test_duplicate_exact_path_rows_are_rejected(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        for dataset_id in ("first-id", "second-id"):
            registry.conn.execute(
                """
                INSERT INTO datasets(dataset_id, zarr_path, status)
                VALUES (?, ?, 'active');
                """,
                (dataset_id, str(zarr_path.resolve())),
            )
        registry.conn.commit()

        with pytest.raises(RecordingIdentityProjectionConflict, match="multiple dataset IDs"):
            _project(registry, zarr_path)
    finally:
        registry.close()


def test_current_pointer_rows_are_immutable(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _project(registry, zarr_path)
        for table in (
            "dataset_recording_identity_current",
            "recording_identity_current",
        ):
            for statement in (
                f"UPDATE {table} SET identity_revision = identity_revision;",
                f"DELETE FROM {table};",
                f"INSERT OR REPLACE INTO {table} SELECT * FROM {table};",
            ):
                with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                    registry.conn.execute(statement)
                registry.conn.rollback()
        assert registry.read_verified_recording_identity(
            registry.conn.execute("SELECT dataset_id FROM datasets;").fetchone()[0]
        )
    finally:
        registry.close()


def test_unadvanced_current_pointer_fails_closed(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)
        registry.conn.execute(
            """
            INSERT INTO recording_identity_revisions(
                identity_snapshot_id, identity_scope_id, recording_id,
                session_uuid, identity_revision,
                supersedes_identity_snapshot_id, schema_id, revision_kind,
                decided_by, decided_at_utc, correction_reason,
                evidence_digest, initiating_dataset_id,
                registry_schema_version
            ) VALUES (
                '22222222-2222-4222-8222-222222222222', ?,
                'recording-a', 'session-a', 2, ?, ?, 'correction',
                'fixture', '2026-08-25T13:00:00+00:00', 'fixture correction',
                ?, ?, 72
            );
            """,
            (
                result.identity_scope_id,
                result.identity_snapshot_id,
                IDENTITY_REVISION_SCHEMA_ID,
                result.evidence_digest,
                result.dataset_id,
            ),
        )
        registry.conn.commit()

        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="not the latest revision",
        ):
            _project(registry, zarr_path)
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="not the latest revision",
        ):
            registry.read_verified_recording_identity(result.dataset_id)
    finally:
        registry.close()


@pytest.mark.parametrize(
    (
        "identity_snapshot_id",
        "identity_revision",
        "predecessor_kind",
        "revision_kind",
        "correction_reason",
    ),
    [
        (
            "33333333-3333-4333-8333-333333333333",
            3,
            "current",
            "correction",
            "skips revision two",
        ),
        (
            "44444444-4444-4444-8444-444444444444",
            2,
            "wrong",
            "correction",
            "wrong predecessor",
        ),
        (
            "55555555-5555-4555-8555-555555555555",
            1,
            "none",
            "initial",
            None,
        ),
    ],
)
def test_revision_trigger_rejects_noncontiguous_or_duplicate_chain(
    tmp_path: Path,
    identity_snapshot_id: str,
    identity_revision: int,
    predecessor_kind: str,
    revision_kind: str,
    correction_reason: str | None,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)
        predecessor = {
            "current": result.identity_snapshot_id,
            "wrong": "66666666-6666-4666-8666-666666666666",
            "none": None,
        }[predecessor_kind]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            _insert_fixture_revision(
                registry,
                result=result,
                identity_snapshot_id=identity_snapshot_id,
                identity_revision=identity_revision,
                supersedes_identity_snapshot_id=predecessor,
                revision_kind=revision_kind,
                correction_reason=correction_reason,
            )
        registry.conn.rollback()
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_revisions;"
        ).fetchone()[0] == 1
    finally:
        registry.close()


def test_evidence_and_revisions_are_append_only(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _project(registry, zarr_path)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            registry.conn.execute(
                "UPDATE recording_identity_evidence SET schema_id='other';"
            )
        registry.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            registry.conn.execute("DELETE FROM recording_identity_revisions;")
        registry.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            registry.conn.execute(
                """
                INSERT OR REPLACE INTO recording_identity_evidence
                SELECT * FROM recording_identity_evidence;
                """
            )
        registry.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            registry.conn.execute(
                """
                INSERT OR REPLACE INTO recording_identity_revisions
                SELECT * FROM recording_identity_revisions;
                """
            )
        registry.conn.rollback()
    finally:
        registry.close()


def test_bound_rows_reject_legacy_identity_overwrite(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)

        with pytest.raises(sqlite3.IntegrityError):
            registry.upsert_recording(
                recording_id="recording-a", session_uuid="session-b"
            )
        registry.conn.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            registry.upsert_dataset(
                result.dataset_id,
                session_uuid="session-b",
                recording_id="recording-a",
                zarr_path=zarr_path,
                artifact_kind="source_recording",
            )
        registry.conn.rollback()
        with pytest.raises(
            sqlite3.IntegrityError,
            match="authority-bound recording context is immutable",
        ):
            registry.upsert_recording(
                recording_id="recording-a", camera_id="2010094"
            )
        registry.conn.rollback()
        with pytest.raises(
            sqlite3.IntegrityError,
            match="authority-bound recording context is immutable",
        ):
            registry.upsert_recording(
                recording_id="recording-a",
                recording_path=str(tmp_path / "recordings" / "other"),
            )
        registry.conn.rollback()
    finally:
        registry.close()


def test_one_session_can_bind_multiple_recordings(tmp_path: Path) -> None:
    first_path = _write_artifact(tmp_path, name="camera-a")
    second_path = _write_artifact(
        tmp_path,
        name="camera-b",
        recording_id="recording-b",
        session_uuid="session-a",
    )
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _project(registry, first_path)
        _project(registry, second_path)

        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recordings WHERE session_uuid='session-a';"
        ).fetchone()[0] == 2
    finally:
        registry.close()


def test_decision_timestamp_must_be_utc(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        with pytest.raises(RecordingIdentityAuthorityError, match="must use UTC"):
            _project(
                registry,
                zarr_path,
                decided_at="2026-08-25T08:00:00-04:00",
            )
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
    finally:
        registry.close()


def test_internal_savepoint_prevents_caught_outer_transaction_partial_commit(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            """
            CREATE TRIGGER reject_identity_dataset_insert
            BEFORE INSERT ON datasets
            BEGIN SELECT RAISE(ABORT, 'fixture late failure'); END;
            """
        )
        registry.conn.commit()

        with registry._transaction_context():
            with pytest.raises(sqlite3.IntegrityError, match="fixture late failure"):
                _project(registry, zarr_path)
            registry.conn.execute(
                "INSERT INTO recordings(recording_id) VALUES ('outer-survives');"
            )

        assert registry.conn.execute(
            "SELECT recording_id FROM recordings;"
        ).fetchall()[0][0] == "outer-survives"
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_evidence;"
        ).fetchone()[0] == 0
    finally:
        registry.close()


def test_source_change_before_locked_read_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    root_path = zarr_path / "zarr.json"
    registry = Registry(tmp_path / "registry.sqlite")
    original = authority_module._schema_ready

    def mutate_after_initial_read(conn: sqlite3.Connection) -> int:
        version = original(conn)
        document = json.loads(root_path.read_text(encoding="utf-8"))
        document["attributes"]["recording_id"] = "recording-b"
        root_path.write_text(json.dumps(document), encoding="utf-8")
        return version

    monkeypatch.setattr(authority_module, "_schema_ready", mutate_after_initial_read)
    try:
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="changed before projection",
        ):
            _project(registry, zarr_path)
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
        assert registry.conn.execute("SELECT COUNT(*) FROM recordings;").fetchone()[0] == 0
    finally:
        registry.close()


def test_source_change_before_final_read_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    root_path = zarr_path / "zarr.json"
    registry = Registry(tmp_path / "registry.sqlite")
    original = authority_module._preflight_evidence

    def mutate_after_preflight(
        conn: sqlite3.Connection,
        evidence: object,
        evidence_json: str,
    ) -> None:
        original(conn, evidence, evidence_json)  # type: ignore[arg-type]
        document = json.loads(root_path.read_text(encoding="utf-8"))
        document["attributes"]["session_uuid"] = "session-b"
        root_path.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(authority_module, "_preflight_evidence", mutate_after_preflight)
    try:
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="changed during projection",
        ):
            _project(registry, zarr_path)
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
        assert registry.conn.execute(
            "SELECT COUNT(*) FROM recording_identity_evidence;"
        ).fetchone()[0] == 0
    finally:
        registry.close()


def test_bound_locator_rejects_raw_mutation_until_dedicated_transition(
    tmp_path: Path,
) -> None:
    zarr_path = _write_artifact(tmp_path)
    moved_path = tmp_path / "recordings" / "recording-a" / "zarr" / "moved.zarr"
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = _project(registry, zarr_path)
        with pytest.raises(
            sqlite3.IntegrityError,
            match="identity and locator are immutable",
        ):
            registry.conn.execute(
                "UPDATE datasets SET zarr_path=?, path_hash=? WHERE dataset_id=?;",
                (
                    str(moved_path.resolve(strict=False)),
                    canonical_dataset_path_hash(moved_path),
                    result.dataset_id,
                ),
            )
        registry.conn.rollback()
        assert registry.read_verified_recording_identity(result.dataset_id).zarr_path == (
            zarr_path.resolve()
        )
    finally:
        registry.close()


def test_verified_reader_and_existing_view_survive_reopen(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    result = _project(registry, zarr_path)
    registry.close()

    reopened = Registry(registry_path)
    try:
        verified = reopened.read_verified_recording_identity(result.dataset_id)
        assert verified.recording_id == "recording-a"
        assert verified.session_uuid == "session-a"
        recording = reopened.conn.execute(
            "SELECT recording_path, camera_id FROM recordings WHERE recording_id=?;",
            (verified.recording_id,),
        ).fetchone()
        assert tuple(recording) == (
            str((tmp_path / "recordings" / "recording-a").resolve()),
            "2010093",
        )
        unpatched = reopened.conn.execute(
            "SELECT recording_id, session_uuid FROM dataset_context_current WHERE dataset_id=?;",
            (result.dataset_id,),
        ).fetchone()
        assert tuple(unpatched) == ("recording-a", "session-a")
    finally:
        reopened.close()


def test_verified_reader_rejects_missing_recording_parent(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    result = _project(registry, zarr_path)
    registry.close()

    with sqlite3.connect(registry_path) as conn:
        conn.execute("DELETE FROM recordings WHERE recording_id='recording-a';")

    reopened = Registry(registry_path)
    try:
        with pytest.raises(
            RecordingIdentityProjectionConflict,
            match="no complete recording identity binding",
        ):
            reopened.read_verified_recording_identity(result.dataset_id)
    finally:
        reopened.close()


def test_incomplete_schema_is_rejected_before_projection(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute("DROP TABLE dataset_recording_identity_current;")
        registry.conn.commit()

        with pytest.raises(RecordingIdentityAuthorityError, match="schema fingerprint"):
            _project(registry, zarr_path)
    finally:
        registry.close()


def test_foreign_keys_disabled_are_rejected_before_projection(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute("PRAGMA foreign_keys = OFF;")
        with pytest.raises(RecordingIdentityAuthorityError, match="foreign keys"):
            _project(registry, zarr_path)
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
    finally:
        registry.close()


def test_missing_immutability_trigger_is_rejected(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            "DROP TRIGGER recording_identity_evidence_reject_update;"
        )
        registry.conn.commit()
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="schema fingerprint",
        ):
            _project(registry, zarr_path)
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
    finally:
        registry.close()


def test_same_name_noop_trigger_is_rejected(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.conn.execute(
            "DROP TRIGGER recording_identity_evidence_reject_update;"
        )
        registry.conn.execute(
            """
            CREATE TRIGGER recording_identity_evidence_reject_update
            BEFORE UPDATE ON recording_identity_evidence
            BEGIN SELECT 1; END;
            """
        )
        registry.conn.commit()
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="schema fingerprint",
        ):
            _project(registry, zarr_path)
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 0
    finally:
        registry.close()


def test_unverified_legacy_bootstrap_is_rejected(tmp_path: Path) -> None:
    zarr_path = _write_artifact(tmp_path)
    registry_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(registry_path) as conn:
        conn.execute("CREATE TABLE datasets(dataset_id TEXT PRIMARY KEY);")
    registry = Registry(registry_path)
    try:
        with pytest.raises(
            RecordingIdentityAuthorityError,
            match="schema-managed registry",
        ):
            _project(registry, zarr_path)
    finally:
        registry.close()


def test_physical_v72_registry_upgrades_to_v73(tmp_path: Path) -> None:
    registry_path = tmp_path / "upgrade.sqlite"
    Registry(registry_path).close()
    with sqlite3.connect(registry_path) as conn:
        for table in (
            "recording_import_receipt_bindings",
            "dataset_recording_identity_current",
            "recording_identity_current",
            "recording_identity_revisions",
            "recording_identity_evidence",
        ):
            conn.execute(f"DROP TABLE {table};")
        conn.execute("DROP INDEX idx_recordings_exact_identity;")
        conn.execute("DROP INDEX idx_datasets_exact_identity;")
        conn.execute("DELETE FROM schema_version WHERE version = 73;")
        conn.execute("PRAGMA user_version = 72;")
        conn.commit()

    upgraded = Registry(registry_path)
    try:
        assert upgraded._current_schema_version() == 73
        tables = {
            str(row[0])
            for row in upgraded.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert {
            "recording_identity_evidence",
            "recording_identity_revisions",
            "recording_identity_current",
            "dataset_recording_identity_current",
            "recording_import_receipt_bindings",
        } <= tables
        assert upgraded.conn.execute("PRAGMA foreign_key_check;").fetchall() == []
    finally:
        upgraded.close()


def test_migration73_rejects_preexisting_authority_tables(tmp_path: Path) -> None:
    registry_path = tmp_path / "partial-upgrade.sqlite"
    Registry(registry_path).close()
    with sqlite3.connect(registry_path) as conn:
        for table in (
            "recording_import_receipt_bindings",
            "dataset_recording_identity_current",
            "recording_identity_current",
            "recording_identity_revisions",
            "recording_identity_evidence",
        ):
            conn.execute(f"DROP TABLE {table};")
        conn.execute("DROP INDEX idx_recordings_exact_identity;")
        conn.execute("DROP INDEX idx_datasets_exact_identity;")
        conn.execute("DELETE FROM schema_version WHERE version = 73;")
        conn.execute("PRAGMA user_version = 72;")
        conn.execute(
            "CREATE TABLE recording_identity_evidence(evidence_digest TEXT);"
        )
        conn.commit()

    with pytest.raises(RuntimeError, match="refuses pre-existing"):
        Registry(registry_path)

    with sqlite3.connect(registry_path) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version;").fetchone()[0] == 72
        assert conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='recording_identity_evidence';"
        ).fetchone()[0] == (
            "CREATE TABLE recording_identity_evidence(evidence_digest TEXT)"
        )

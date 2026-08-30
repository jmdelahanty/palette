"""Opt-in authority projection for regular source-recording identity.

This boundary owns its evidence collection: it reads the capture manifest and
direct root metadata from the exact target path before it mutates SQLite.  The
ordinary writer can create/fill projections or replay an exact decision; it
cannot correct an established identity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import unicodedata
from uuid import uuid4

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundAcquisitionImportOwnership,
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.source_recording_identity import (
    SourceRecordingIdentityClaim,
    SourceRecordingIdentityError,
    load_source_recording_identity_claim,
    SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)

from . import identity as registry_identity
from .temp_store_guard import assert_temp_store_registration_allowed


IDENTITY_REVISION_SCHEMA_ID = "palette.registry.recording_identity_revision.v1"
REQUIRED_SCHEMA_OBJECT_NAMES = frozenset(
    {
        "recording_identity_evidence",
        "recording_identity_revisions",
        "recording_identity_current",
        "dataset_recording_identity_current",
        "recording_import_receipt_bindings",
        "idx_recordings_exact_identity",
        "idx_datasets_exact_identity",
        "idx_recording_identity_revisions_evidence",
        "recording_identity_evidence_reject_duplicate_insert",
        "recording_identity_evidence_reject_update",
        "recording_identity_evidence_reject_delete",
        "recording_identity_revisions_reject_conflicting_insert",
        "recording_identity_revisions_reject_update",
        "recording_identity_revisions_reject_delete",
        "recording_import_receipt_bindings_reject_duplicate_insert",
        "recording_import_receipt_bindings_reject_update",
        "recording_import_receipt_bindings_reject_delete",
        "recording_identity_current_reject_duplicate_insert",
        "recording_identity_current_reject_update",
        "recording_identity_current_reject_delete",
        "dataset_recording_identity_current_reject_duplicate_insert",
        "dataset_recording_identity_current_reject_update",
        "dataset_recording_identity_current_reject_delete",
        "datasets_reject_bound_identity_locator_update",
        "recordings_reject_bound_identity_context_update",
    }
)
REQUIRED_SCHEMA_FINGERPRINT = (
    "c0cbe7e0506ff399de55e8b22d019428691ae3cae3cb06d065f3b28ce955d1ae"
)


class RecordingIdentityAuthorityError(RuntimeError):
    """The authority boundary could not safely apply a projection."""


class RecordingIdentityProjectionConflict(RecordingIdentityAuthorityError):
    """Existing non-null state conflicts with the resolved identity."""


class RecordingImportBindingNotFound(RecordingIdentityAuthorityError):
    """No receipt-bound current import exists at the requested locator."""


@dataclass(frozen=True, slots=True)
class RecordingIdentityProjectionResult:
    dataset_id: str
    identity_scope_id: str
    identity_snapshot_id: str
    identity_revision: int
    evidence_digest: str
    receipt_sha256: str
    disposition: str


@dataclass(frozen=True, slots=True)
class VerifiedRecordingIdentity:
    """A dataset identity whose live sources and registry binding agree."""

    dataset_id: str
    recording_id: str
    session_uuid: str
    zarr_path: Path
    identity_scope_id: str
    identity_snapshot_id: str
    identity_revision: int
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedRecordingImport:
    """A current importer receipt bound to live acquisition authority."""

    identity: VerifiedRecordingIdentity
    receipt: "RecordingImportReceipt"
    receipt_path: Path
    acquisition_ownership: BoundAcquisitionImportOwnership
    acquisition_frame: BoundAcquisitionCameraFrame
    acquisition_authority_path: str


def _exact_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RecordingIdentityAuthorityError(
            f"{field} must be a non-empty string without surrounding whitespace"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RecordingIdentityAuthorityError(
            f"{field} must contain valid Unicode text"
        ) from exc
    if any(
        unicodedata.category(character).startswith("C") for character in value
    ):
        raise RecordingIdentityAuthorityError(
            f"{field} must not contain control characters"
        )
    return value


def _exact_timestamp(value: str | None) -> str:
    raw = utc_now() if value is None else _exact_text(value, field="decided_at_utc")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordingIdentityAuthorityError(
            "decided_at_utc must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise RecordingIdentityAuthorityError(
            "decided_at_utc must include a UTC offset"
        )
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RecordingIdentityAuthorityError(
            "decided_at_utc must use UTC"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def canonical_dataset_path_hash(path: Path) -> str:
    """Return the registry's canonical resolved-locator hash."""

    return sha256(str(Path(path).expanduser().resolve()).encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def _resolve_regular_source_target(zarr_path: Path) -> Path:
    requested = Path(zarr_path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise RecordingIdentityAuthorityError(
            f"target Zarr path does not exist: {requested}"
        ) from exc
    if not resolved.is_dir():
        raise RecordingIdentityAuthorityError(
            f"target Zarr path is not a directory: {resolved}"
        )
    if "recordings" not in {part.casefold() for part in resolved.parts}:
        raise RecordingIdentityAuthorityError(
            f"regular source recording must be beneath a recordings directory: {resolved}"
        )
    return resolved


def _collect_regular_source_recording_claim(
    resolved: Path,
) -> SourceRecordingIdentityClaim:
    manifest_path = (
        recording_directory_for_source_target(resolved)
        / "recording_manifest.json"
    )
    try:
        return load_source_recording_identity_claim(manifest_path, resolved)
    except SourceRecordingIdentityError as exc:
        raise RecordingIdentityAuthorityError(str(exc)) from exc


def collect_regular_source_recording_identity(
    zarr_path: Path,
) -> SourceRecordingIdentityClaim:
    """Read exact manifest/root identity claims from one target artifact."""

    return _collect_regular_source_recording_claim(
        _resolve_regular_source_target(zarr_path)
    )


def _schema_ready(conn: sqlite3.Connection) -> int:
    foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()
    if foreign_keys is None or int(foreign_keys[0]) != 1:
        raise RecordingIdentityAuthorityError(
            "recording identity authority requires SQLite foreign keys"
        )
    identity = registry_identity.read_registry_identity(conn)
    if identity.identity_provenance != registry_identity.REGISTRY_IDENTITY_SCHEMA_MANAGED:
        raise RecordingIdentityAuthorityError(
            "recording identity authority requires a schema-managed registry"
        )
    version_row = conn.execute("SELECT MAX(version) FROM schema_version;").fetchone()
    schema_version = int(version_row[0]) if version_row and version_row[0] else 0
    if schema_version < 73:
        raise RecordingIdentityAuthorityError(
            "recording identity authority requires registry schema version 73 or later"
        )
    migration = conn.execute(
        "SELECT name FROM schema_version WHERE version = 73;"
    ).fetchone()
    if migration is None or str(migration[0]) != "recording_identity_authority":
        raise RecordingIdentityAuthorityError(
            "recording identity authority migration record is missing"
        )
    schema_rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL ORDER BY type, name;"
    ).fetchall()
    schema_payload = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "sql": " ".join(str(row[2]).split()),
        }
        for row in schema_rows
        if str(row[1]) in REQUIRED_SCHEMA_OBJECT_NAMES
    ]
    if canonical_json_sha256(schema_payload) != REQUIRED_SCHEMA_FINGERPRINT:
        raise RecordingIdentityAuthorityError(
            "recording identity authority schema fingerprint does not match migration 73"
        )
    return schema_version


@contextmanager
def _savepoint(conn: sqlite3.Connection) -> Iterator[None]:
    name = f"recording_identity_{uuid4().hex}"
    conn.execute(f"SAVEPOINT {name};")
    try:
        yield
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {name};")
        conn.execute(f"RELEASE SAVEPOINT {name};")
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {name};")


def _conflict(field: str, current: Any, expected: Any) -> None:
    raise RecordingIdentityProjectionConflict(
        f"{field} conflicts with resolved identity: current={current!r}, "
        f"resolved={expected!r}"
    )


def _preflight_recording(
    row: sqlite3.Row | None,
    *,
    session_uuid: str,
    camera_id: str,
    recording_path: Path,
) -> None:
    if row is None:
        return
    expectations = {
        "session_uuid": session_uuid,
        "camera_id": camera_id,
        "recording_path": str(recording_path),
    }
    for field, expected in expectations.items():
        if row[field] is not None and str(row[field]) != expected:
            _conflict(f"recordings.{field}", row[field], expected)


def _normalized_existing_path(raw: Any) -> Path | None:
    if type(raw) is not str or not raw.strip():
        return None
    return Path(raw).expanduser().resolve(strict=False)


def _preflight_dataset(
    row: sqlite3.Row | None,
    *,
    recording_id: str,
    session_uuid: str,
    zarr_path: Path,
    path_hash: str,
) -> None:
    if row is None:
        return
    expectations = {
        "recording_id": recording_id,
        "session_uuid": session_uuid,
    }
    for field, expected in expectations.items():
        if row[field] is not None and str(row[field]) != expected:
            _conflict(f"datasets.{field}", row[field], expected)
    current_path = _normalized_existing_path(row["zarr_path"])
    if current_path != zarr_path:
        _conflict("datasets.zarr_path", row["zarr_path"], str(zarr_path))
    if str(row["path_hash"] or "") != path_hash:
        _conflict("datasets.path_hash", row["path_hash"], path_hash)
    allowed = {
        "artifact_kind": {None, "source_recording"},
        "zarr_origin": {None, "source", "imported"},
        "zarr_use": {None, "analysis"},
    }
    for field, values in allowed.items():
        if row[field] not in values:
            _conflict(f"datasets.{field}", row[field], sorted(str(v) for v in values))


def _validate_current_revision(
    conn: sqlite3.Connection,
    current: sqlite3.Row,
) -> None:
    latest = conn.execute(
        """
        SELECT identity_snapshot_id, identity_revision
        FROM recording_identity_revisions
        WHERE identity_scope_id = ?
        ORDER BY identity_revision DESC
        LIMIT 1;
        """,
        (str(current["identity_scope_id"]),),
    ).fetchone()
    if latest is None or (
        int(latest["identity_revision"]) != int(current["identity_revision"])
        or str(latest["identity_snapshot_id"])
        != str(current["identity_snapshot_id"])
    ):
        raise RecordingIdentityProjectionConflict(
            "recording identity current pointer is not the latest revision"
        )


def _authority_state(
    conn: sqlite3.Connection,
    *,
    recording_id: str,
    session_uuid: str,
) -> sqlite3.Row | None:
    current = conn.execute(
        """
        SELECT c.*,
               r.recording_id AS revision_recording_id,
               r.session_uuid AS revision_session_uuid,
               r.revision_kind, r.supersedes_identity_snapshot_id,
               r.correction_reason, r.evidence_digest
        FROM recording_identity_current c
        LEFT JOIN recording_identity_revisions r
          ON r.identity_scope_id = c.identity_scope_id
         AND r.identity_snapshot_id = c.identity_snapshot_id
         AND r.identity_revision = c.identity_revision
        WHERE c.recording_id = ?;
        """,
        (recording_id,),
    ).fetchone()
    if current is None:
        stale_revision = conn.execute(
            """
            SELECT identity_scope_id FROM recording_identity_revisions
            WHERE recording_id = ? LIMIT 1;
            """,
            (recording_id,),
        ).fetchone()
        if stale_revision is not None:
            raise RecordingIdentityProjectionConflict(
                "recording identity has revisions but no current pointer"
            )
        return None

    if str(current["session_uuid"]) != session_uuid:
        _conflict("recording_identity_current.session_uuid", current["session_uuid"], session_uuid)
    if current["evidence_digest"] is None:
        raise RecordingIdentityProjectionConflict(
            "recording identity current pointer has no revision"
        )
    if (
        str(current["revision_recording_id"]) != recording_id
        or str(current["revision_session_uuid"]) != session_uuid
    ):
        raise RecordingIdentityProjectionConflict(
            "recording identity current pointer disagrees with its revision"
        )
    _validate_current_revision(conn, current)
    revision_evidence = _load_stored_evidence(
        conn,
        str(current["evidence_digest"]),
        required=True,
    )
    assert revision_evidence is not None
    if (
        revision_evidence.identity.recording_id != recording_id
        or revision_evidence.identity.session_uuid != session_uuid
    ):
        raise RecordingIdentityProjectionConflict(
            "recording identity revision evidence resolves to different identity"
        )
    return current


def _evidence_json(evidence: SourceRecordingIdentityClaim) -> str:
    return canonical_json_bytes(evidence.as_dict()).decode("utf-8")


def _load_stored_evidence(
    conn: sqlite3.Connection,
    evidence_digest: str,
    *,
    required: bool,
    expected_json: str | None = None,
) -> SourceRecordingIdentityClaim | None:
    row = conn.execute(
        "SELECT schema_id, evidence_json FROM recording_identity_evidence WHERE evidence_digest = ?;",
        (evidence_digest,),
    ).fetchone()
    if row is None:
        if required:
            raise RecordingIdentityProjectionConflict(
                "recording identity revision references missing evidence"
            )
        return None
    if str(row["schema_id"]) != SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID or (
        expected_json is not None and str(row["evidence_json"]) != expected_json
    ):
        raise RecordingIdentityProjectionConflict(
            "recording identity evidence digest is bound to different content"
        )
    try:
        stored = json.loads(
            str(row["evidence_json"]),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(stored, dict):
            raise TypeError("evidence document is not an object")
        rebuilt = SourceRecordingIdentityClaim.from_mapping(stored)
    except (
        SourceRecordingIdentityError,
        TypeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RecordingIdentityProjectionConflict(
            "stored recording identity evidence is invalid JSON"
        ) from exc
    if rebuilt.claim_sha256 != evidence_digest:
        raise RecordingIdentityProjectionConflict(
            "stored recording identity evidence does not match its digest"
        )
    return rebuilt


def _preflight_evidence(
    conn: sqlite3.Connection,
    evidence: SourceRecordingIdentityClaim,
    evidence_json: str,
) -> None:
    _load_stored_evidence(
        conn,
        evidence.claim_sha256,
        required=False,
        expected_json=evidence_json,
    )


def _preflight_binding(
    conn: sqlite3.Connection,
    row: sqlite3.Row | None,
    *,
    scope_id: str | None,
    recording_id: str,
    session_uuid: str,
    snapshot_id: str | None,
    revision: int | None,
) -> None:
    if row is None:
        return
    expected = {
        "identity_scope_id": scope_id,
        "recording_id": recording_id,
        "session_uuid": session_uuid,
        "identity_snapshot_id": snapshot_id,
        "identity_revision": revision,
    }
    for field, value in expected.items():
        current = int(row[field]) if field == "identity_revision" else str(row[field])
        if value is None or current != value:
            _conflict(f"dataset_recording_identity_current.{field}", row[field], value)


def recording_directory_for_source_target(resolved_path: Path) -> Path:
    """Resolve the unique recording directory that owns a source target."""

    candidates: list[Path] = []
    for parent in resolved_path.parents:
        if parent.name.casefold() == "recordings":
            break
        if (parent / "recording_manifest.json").is_file():
            candidates.append(parent)
    if len(candidates) != 1:
        raise RecordingIdentityAuthorityError(
            "source-recording target must have exactly one recording manifest ancestor"
        )
    return candidates[0]


def _verify_import_receipt(
    *,
    resolved_path: Path,
    evidence: SourceRecordingIdentityClaim,
    receipt: object,
) -> str:
    from fisheye.shared.recording_import_receipt import (
        RecordingImportReceipt,
        recording_import_receipt_path,
    )

    if type(receipt) is not RecordingImportReceipt:
        raise RecordingIdentityAuthorityError(
            "import receipt must be a validated RecordingImportReceipt"
        )
    if receipt.identity_claim != evidence:
        raise RecordingIdentityProjectionConflict(
            "import receipt identity claim differs from the target"
        )
    recording_dir = recording_directory_for_source_target(resolved_path)
    try:
        relative_target = resolved_path.relative_to(recording_dir).as_posix()
    except ValueError as exc:
        raise RecordingIdentityAuthorityError(
            "source-recording target is outside its recording directory"
        ) from exc
    if receipt.target_relative_path != relative_target:
        raise RecordingIdentityProjectionConflict(
            "import receipt target differs from the projected Zarr"
        )
    sidecar = recording_import_receipt_path(
        resolved_path,
        receipt.receipt_sha256,
    )
    if RecordingImportReceipt.from_path(sidecar) != receipt:
        raise RecordingIdentityProjectionConflict(
            "import receipt sidecar differs from the supplied receipt"
        )
    return receipt.receipt_sha256


def _authority_path_from_record_ref(record_ref: str, *, field: str) -> str:
    if not record_ref.startswith("/") or "@" not in record_ref:
        raise RecordingIdentityProjectionConflict(
            f"{field} reference is not a canonical acquisition reference"
        )
    path, _, attribute = record_ref[1:].partition("@")
    if not path or not attribute:
        raise RecordingIdentityProjectionConflict(
            f"{field} reference is not a canonical acquisition reference"
        )
    return path


def _verify_live_import_receipt(
    *,
    resolved_path: Path,
    evidence: SourceRecordingIdentityClaim,
    receipt: object,
    require_current_producer_code: bool = False,
) -> tuple[
    str,
    BoundAcquisitionImportOwnership,
    BoundAcquisitionCameraFrame,
    str,
]:
    """Bind a receipt to stable live acquisition evidence at one read boundary."""

    receipt_sha256 = _verify_import_receipt(
        resolved_path=resolved_path,
        evidence=evidence,
        receipt=receipt,
    )
    if require_current_producer_code:
        from fisheye.shared.run_provenance import git_identity

        code = git_identity(cwd=Path(__file__).resolve().parents[3])
        if (
            code.get("git_dirty") is not False
            or code.get("git_sha") != receipt.producer_git_sha
        ):
            raise RecordingIdentityProjectionConflict(
                "receipt producer commit does not match the clean binding checkout"
            )
    live_claim = _collect_regular_source_recording_claim(resolved_path)
    if live_claim != evidence:
        raise RecordingIdentityProjectionConflict(
            "live source-recording identity differs from the import receipt"
        )
    source_identity = live_claim.identity
    try:
        import zarr

        root = zarr.open_group(
            str(resolved_path),
            mode="r",
            zarr_format=3,
            use_consolidated=False,
        )
        publication = load_acquisition_authority_publication_status(root)
        if (
            publication.status != ACQUISITION_AUTHORITY_PUBLISHED
            or publication.authority_mode
            != EXTERNAL_ACQUISITION_AUTHORITY_MODE
        ):
            raise RecordingIdentityProjectionConflict(
                "acquisition authority is not a published external-video authority"
            )
        ownership, frame = load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=source_identity.camera_id,
        )
    except RecordingIdentityProjectionConflict:
        raise
    except Exception as exc:
        raise RecordingIdentityAuthorityError(
            "live acquisition authority could not be verified"
        ) from exc

    ownership_path = _authority_path_from_record_ref(
        ownership.record_ref,
        field="acquisition ownership",
    )
    frame_path = _authority_path_from_record_ref(
        frame.record_ref,
        field="acquisition frame",
    )
    if ownership_path != frame_path or publication.authority_path != frame_path:
        raise RecordingIdentityProjectionConflict(
            "acquisition publication authority path differs from live records"
        )
    if (
        frame.record.recording_id != evidence.identity.recording_id
        or frame.record.camera_id != source_identity.camera_id
    ):
        raise RecordingIdentityProjectionConflict(
            "live acquisition authority differs from source-recording identity"
        )
    if (
        receipt.acquisition_ownership_ref != ownership.record_ref
        or receipt.acquisition_ownership_sha256 != ownership.record_sha256
        or receipt.acquisition_frame_ref != frame.record_ref
        or receipt.acquisition_frame_sha256 != frame.record_sha256
    ):
        raise RecordingIdentityProjectionConflict(
            "receipt acquisition references differ from live authority"
        )
    ownership.assert_verified()
    frame.assert_verified()
    if load_acquisition_authority_publication_status(root) != publication:
        raise RecordingIdentityProjectionConflict(
            "acquisition publication status changed while it was verified"
        )
    if _verify_import_receipt(
        resolved_path=resolved_path,
        evidence=evidence,
        receipt=receipt,
    ) != receipt_sha256:
        raise RecordingIdentityProjectionConflict(
            "import receipt changed while live acquisition was verified"
        )
    return receipt_sha256, ownership, frame, frame_path


def _preflight_receipt_binding(
    conn: sqlite3.Connection,
    *,
    receipt_sha256: str,
    dataset_id: str,
    identity_scope_id: str,
    identity_snapshot_id: str,
) -> bool:
    rows = conn.execute(
        """
        SELECT receipt_sha256, dataset_id, identity_scope_id,
               identity_snapshot_id
        FROM recording_import_receipt_bindings
        WHERE receipt_sha256 = ?
           OR (dataset_id = ? AND identity_scope_id = ?
               AND identity_snapshot_id = ?)
        ORDER BY receipt_sha256;
        """,
        (
            receipt_sha256,
            dataset_id,
            identity_scope_id,
            identity_snapshot_id,
        ),
    ).fetchall()
    expected = (
        receipt_sha256,
        dataset_id,
        identity_scope_id,
        identity_snapshot_id,
    )
    if any(
        (
            str(row["receipt_sha256"]),
            str(row["dataset_id"]),
            str(row["identity_scope_id"]),
            str(row["identity_snapshot_id"]),
        )
        != expected
        for row in rows
    ):
        raise RecordingIdentityProjectionConflict(
            "import receipt is already bound to different authority state"
        )
    return bool(rows)


class RegistryRecordingIdentityMixin:
    """One supported writer and one verified reader for recording identity."""

    def project_regular_source_recording_identity(
        self,
        *,
        zarr_path: Path,
        decided_by: str,
        import_receipt: object,
        decided_at_utc: str | None = None,
    ) -> RecordingIdentityProjectionResult:
        """Project exact source agreement and bind its import receipt."""

        resolved_path = _resolve_regular_source_target(zarr_path)
        evidence = _collect_regular_source_recording_claim(resolved_path)
        source_identity = evidence.identity
        recording_id = source_identity.recording_id
        session_uuid = source_identity.session_uuid

        actor = _exact_text(decided_by, field="decided_by")
        decided_at = _exact_timestamp(decided_at_utc)
        path_hash = canonical_dataset_path_hash(resolved_path)
        recording_path = recording_directory_for_source_target(resolved_path).resolve()
        evidence_json = _evidence_json(evidence)
        receipt_sha256 = _verify_import_receipt(
            resolved_path=resolved_path,
            evidence=evidence,
            receipt=import_receipt,
        )

        registry_schema_version = _schema_ready(self.conn)
        assert_temp_store_registration_allowed(
            registry_path=self.path,
            store_path=resolved_path,
            recording_id=recording_id,
            zarr_use="analysis",
        )

        with self._transaction_context():
            with _savepoint(self.conn):
                try:
                    locked_evidence = _collect_regular_source_recording_claim(
                        resolved_path
                    )
                except RecordingIdentityAuthorityError as exc:
                    raise RecordingIdentityProjectionConflict(
                        "source-recording identity changed before projection"
                    ) from exc
                if locked_evidence != evidence:
                    raise RecordingIdentityProjectionConflict(
                        "source-recording identity changed before projection"
                    )
                _verify_live_import_receipt(
                    resolved_path=resolved_path,
                    evidence=locked_evidence,
                    receipt=import_receipt,
                    require_current_producer_code=True,
                )
                path_rows = self.conn.execute(
                    """
                    SELECT dataset_id FROM datasets
                    WHERE zarr_path = ? OR path_hash = ?
                    ORDER BY dataset_id;
                    """,
                    (str(resolved_path), path_hash),
                ).fetchall()
                if len(path_rows) > 1:
                    raise RecordingIdentityProjectionConflict(
                        "target path is already registered under multiple dataset IDs: "
                        + ", ".join(str(row["dataset_id"]) for row in path_rows)
                    )
                if path_rows:
                    # Dataset IDs are stable.  An exact existing locator wins only
                    # for preserving that row; its identity still goes through all
                    # conflict checks below.
                    dataset_id = str(path_rows[0]["dataset_id"])
                else:
                    dataset_id = self._resolve_effective_dataset_id(
                        base_dataset_id=session_uuid,
                        session_uuid=session_uuid,
                        zarr_path=resolved_path,
                    )
                dataset_id = _exact_text(dataset_id, field="effective dataset_id")

                recording = self.conn.execute(
                    "SELECT * FROM recordings WHERE recording_id = ?;",
                    (recording_id,),
                ).fetchone()
                dataset = self.conn.execute(
                    "SELECT * FROM datasets WHERE dataset_id = ?;",
                    (dataset_id,),
                ).fetchone()
                _preflight_recording(
                    recording,
                    session_uuid=session_uuid,
                    camera_id=source_identity.camera_id,
                    recording_path=recording_path,
                )
                _preflight_dataset(
                    dataset,
                    recording_id=recording_id,
                    session_uuid=session_uuid,
                    zarr_path=resolved_path,
                    path_hash=path_hash,
                )
                current = _authority_state(
                    self.conn,
                    recording_id=recording_id,
                    session_uuid=session_uuid,
                )
                binding = self.conn.execute(
                    "SELECT * FROM dataset_recording_identity_current WHERE dataset_id = ?;",
                    (dataset_id,),
                ).fetchone()
                current_scope = (
                    str(current["identity_scope_id"]) if current is not None else None
                )
                current_snapshot = (
                    str(current["identity_snapshot_id"]) if current is not None else None
                )
                current_revision = (
                    int(current["identity_revision"]) if current is not None else None
                )
                if current_scope is not None:
                    other_source_bindings = self.conn.execute(
                        """
                        SELECT d.dataset_id, d.zarr_path
                        FROM dataset_recording_identity_current b
                        INNER JOIN datasets d ON d.dataset_id = b.dataset_id
                        WHERE b.identity_scope_id = ? AND b.dataset_id != ?
                        ORDER BY d.dataset_id;
                        """,
                        (current_scope, dataset_id),
                    ).fetchall()
                    if other_source_bindings:
                        bound = ", ".join(
                            f"{row['dataset_id']}@{row['zarr_path']}"
                            for row in other_source_bindings
                        )
                        raise RecordingIdentityProjectionConflict(
                            "current source identity is already bound to an "
                            f"immutable locator: {bound}"
                        )
                _preflight_binding(
                    self.conn,
                    binding,
                    scope_id=current_scope,
                    recording_id=recording_id,
                    session_uuid=session_uuid,
                    snapshot_id=current_snapshot,
                    revision=current_revision,
                )
                _preflight_evidence(self.conn, evidence, evidence_json)

                scope_id = current_scope or str(uuid4())
                snapshot_id = current_snapshot or str(uuid4())
                revision = current_revision or 1
                receipt_bound = _preflight_receipt_binding(
                    self.conn,
                    receipt_sha256=receipt_sha256,
                    dataset_id=dataset_id,
                    identity_scope_id=scope_id,
                    identity_snapshot_id=snapshot_id,
                )

                self.conn.execute(
                    """
                    INSERT INTO recording_identity_evidence(
                        evidence_digest, schema_id, evidence_json
                    )
                    SELECT ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM recording_identity_evidence
                        WHERE evidence_digest = ?
                    );
                    """,
                    (
                        evidence.claim_sha256,
                        SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID,
                        evidence_json,
                        evidence.claim_sha256,
                    ),
                )
                if recording is None:
                    self.conn.execute(
                        """
                        INSERT INTO recordings(
                            recording_id, session_uuid, recording_path, camera_id,
                            created_utc, updated_utc
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        (
                            recording_id,
                            session_uuid,
                            str(recording_path),
                            source_identity.camera_id,
                            decided_at,
                            decided_at,
                        ),
                    )
                else:
                    self.conn.execute(
                        """
                        UPDATE recordings
                        SET session_uuid = COALESCE(session_uuid, ?),
                            recording_path = COALESCE(recording_path, ?),
                            camera_id = COALESCE(camera_id, ?),
                            updated_utc = ?
                        WHERE recording_id = ?;
                        """,
                        (
                            session_uuid,
                            str(recording_path),
                            source_identity.camera_id,
                            decided_at,
                            recording_id,
                        ),
                    )

                if dataset is None:
                    self.conn.execute(
                        """
                        INSERT INTO datasets(
                            dataset_id, session_uuid, zarr_path, recording_id,
                            path_hash
                        ) VALUES (?, ?, ?, ?, ?);
                        """,
                        (
                            dataset_id,
                            session_uuid,
                            str(resolved_path),
                            recording_id,
                            path_hash,
                        ),
                    )
                else:
                    self.conn.execute(
                        """
                        UPDATE datasets
                        SET session_uuid = ?, recording_id = ?
                        WHERE dataset_id = ?;
                        """,
                        (
                            session_uuid,
                            recording_id,
                            dataset_id,
                        ),
                    )

                if current is None:
                    self.conn.execute(
                        """
                        INSERT INTO recording_identity_revisions(
                            identity_snapshot_id, identity_scope_id,
                            recording_id, session_uuid, identity_revision,
                            supersedes_identity_snapshot_id, schema_id,
                            revision_kind, decided_by, decided_at_utc,
                            correction_reason, evidence_digest,
                            initiating_dataset_id, registry_schema_version
                        ) VALUES (?, ?, ?, ?, 1, NULL, ?, 'initial', ?, ?,
                                  NULL, ?, ?, ?);
                        """,
                        (
                            snapshot_id,
                            scope_id,
                            recording_id,
                            session_uuid,
                            IDENTITY_REVISION_SCHEMA_ID,
                            actor,
                            decided_at,
                            evidence.claim_sha256,
                            dataset_id,
                            registry_schema_version,
                        ),
                    )
                    self.conn.execute(
                        """
                        INSERT INTO recording_identity_current(
                            identity_scope_id, recording_id, session_uuid,
                            identity_snapshot_id, identity_revision,
                            updated_at_utc
                        ) VALUES (?, ?, ?, ?, 1, ?);
                        """,
                        (
                            scope_id,
                            recording_id,
                            session_uuid,
                            snapshot_id,
                            decided_at,
                        ),
                    )

                if binding is None:
                    self.conn.execute(
                        """
                        INSERT INTO dataset_recording_identity_current(
                            dataset_id, identity_scope_id, recording_id,
                            session_uuid, identity_snapshot_id,
                            identity_revision, projected_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            dataset_id,
                            scope_id,
                            recording_id,
                            session_uuid,
                            snapshot_id,
                            revision,
                            decided_at,
                        ),
                    )
                if not receipt_bound:
                    self.conn.execute(
                        """
                        INSERT INTO recording_import_receipt_bindings(
                            receipt_sha256, dataset_id, identity_scope_id,
                            identity_snapshot_id, bound_by, bound_at_utc,
                            registry_schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            receipt_sha256,
                            dataset_id,
                            scope_id,
                            snapshot_id,
                            actor,
                            decided_at,
                            registry_schema_version,
                        ),
                    )
                disposition = (
                    "authority_created"
                    if current is None
                    else "dataset_bound"
                    if binding is None
                    else "exact_replay"
                )
                try:
                    final_evidence = _collect_regular_source_recording_claim(
                        resolved_path
                    )
                except RecordingIdentityAuthorityError as exc:
                    raise RecordingIdentityProjectionConflict(
                        "source-recording identity changed during projection"
                    ) from exc
                if final_evidence != evidence:
                    raise RecordingIdentityProjectionConflict(
                        "source-recording identity changed during projection"
                    )
                _verify_live_import_receipt(
                    resolved_path=resolved_path,
                    evidence=final_evidence,
                    receipt=import_receipt,
                    require_current_producer_code=True,
                )
                return RecordingIdentityProjectionResult(
                    dataset_id=dataset_id,
                    identity_scope_id=scope_id,
                    identity_snapshot_id=snapshot_id,
                    identity_revision=revision,
                    evidence_digest=evidence.claim_sha256,
                    receipt_sha256=receipt_sha256,
                    disposition=disposition,
                )

    def read_verified_recording_identity(
        self,
        dataset_id: str,
    ) -> VerifiedRecordingIdentity:
        """Verify the current registry chain against the dataset's live sources."""

        requested_id = _exact_text(dataset_id, field="dataset_id")
        _schema_ready(self.conn)
        row = self.conn.execute(
            """
            SELECT d.dataset_id, d.recording_id AS dataset_recording_id,
                   d.session_uuid AS dataset_session_uuid, d.zarr_path,
                   d.path_hash,
                   b.identity_scope_id, b.recording_id AS binding_recording_id,
                   b.session_uuid AS binding_session_uuid,
                   b.identity_snapshot_id, b.identity_revision,
                   c.recording_id AS current_recording_id,
                   c.session_uuid AS current_session_uuid,
                   r.recording_id AS revision_recording_id,
                   r.session_uuid AS revision_session_uuid,
                   recording.recording_path AS recording_path,
                   recording.camera_id AS recording_camera_id,
                   r.evidence_digest AS revision_evidence_digest
            FROM datasets d
            JOIN dataset_recording_identity_current b
              ON b.dataset_id = d.dataset_id
            JOIN recording_identity_current c
              ON c.identity_scope_id = b.identity_scope_id
             AND c.identity_snapshot_id = b.identity_snapshot_id
             AND c.identity_revision = b.identity_revision
            JOIN recordings recording
              ON recording.recording_id = c.recording_id
             AND recording.session_uuid = c.session_uuid
            JOIN recording_identity_revisions r
              ON r.identity_scope_id = c.identity_scope_id
             AND r.identity_snapshot_id = c.identity_snapshot_id
             AND r.identity_revision = c.identity_revision
            WHERE d.dataset_id = ?;
            """,
            (requested_id,),
        ).fetchone()
        if row is None:
            raise RecordingIdentityProjectionConflict(
                f"dataset has no complete recording identity binding: {requested_id}"
            )
        identities = {
            str(row[field])
            for field in (
                "dataset_recording_id",
                "binding_recording_id",
                "current_recording_id",
                "revision_recording_id",
            )
        }
        sessions = {
            str(row[field])
            for field in (
                "dataset_session_uuid",
                "binding_session_uuid",
                "current_session_uuid",
                "revision_session_uuid",
            )
        }
        if len(identities) != 1 or len(sessions) != 1:
            raise RecordingIdentityProjectionConflict(
                "recording identity registry projections disagree"
            )
        recording_id = identities.pop()
        session_uuid = sessions.pop()
        _validate_current_revision(self.conn, row)
        revision_evidence = _load_stored_evidence(
            self.conn,
            str(row["revision_evidence_digest"]),
            required=True,
        )
        assert revision_evidence is not None
        if (
            revision_evidence.identity.recording_id != recording_id
            or revision_evidence.identity.session_uuid != session_uuid
        ):
            raise RecordingIdentityProjectionConflict(
                "revision evidence resolves to different identity"
            )
        raw_path = row["zarr_path"]
        if type(raw_path) is not str or not raw_path:
            raise RecordingIdentityProjectionConflict(
                "bound dataset has no current Zarr locator"
            )
        resolved_path = _resolve_regular_source_target(Path(raw_path))
        if str(row["path_hash"] or "") != canonical_dataset_path_hash(resolved_path):
            raise RecordingIdentityProjectionConflict(
                "bound dataset path hash does not match its current locator"
            )
        live_evidence = _collect_regular_source_recording_claim(resolved_path)
        if live_evidence != revision_evidence:
            raise RecordingIdentityProjectionConflict(
                "live source-recording identity differs from its dataset binding"
            )
        expected_recording_path = str(
            recording_directory_for_source_target(resolved_path).resolve()
        )
        if (
            row["recording_camera_id"] != live_evidence.identity.camera_id
            or row["recording_path"] != expected_recording_path
        ):
            raise RecordingIdentityProjectionConflict(
                "recording context projection differs from the live source claim"
            )
        return VerifiedRecordingIdentity(
            dataset_id=requested_id,
            recording_id=recording_id,
            session_uuid=session_uuid,
            zarr_path=resolved_path,
            identity_scope_id=str(row["identity_scope_id"]),
            identity_snapshot_id=str(row["identity_snapshot_id"]),
            identity_revision=int(row["identity_revision"]),
            evidence_digest=str(row["revision_evidence_digest"]),
        )

    def read_verified_recording_import(
        self,
        dataset_id: str,
    ) -> VerifiedRecordingImport:
        """Read one current-import receipt and its live acquisition authority.

        This is deliberately separate from :meth:`read_verified_recording_identity`:
        ordinary identity consumers remain opt-in, while this stricter boundary
        admits only a receipt issued by the current importer and bound in the
        registry to the exact current identity snapshot.
        """

        identity = self.read_verified_recording_identity(dataset_id)
        try:
            rows = self.conn.execute(
                """
                SELECT receipt_sha256
                FROM recording_import_receipt_bindings
                WHERE dataset_id = ? AND identity_scope_id = ?
                  AND identity_snapshot_id = ?
                ORDER BY receipt_sha256;
                """,
                (
                    identity.dataset_id,
                    identity.identity_scope_id,
                    identity.identity_snapshot_id,
                ),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RecordingIdentityAuthorityError(
                "recording import receipt bindings are unavailable"
            ) from exc
        if len(rows) != 1:
            raise RecordingIdentityProjectionConflict(
                "dataset current identity must have exactly one import receipt binding"
            )
        receipt_sha256 = str(rows[0]["receipt_sha256"])

        from fisheye.shared.recording_import_receipt import (
            RecordingImportReceipt,
            RecordingImportReceiptError,
            recording_import_receipt_path,
        )

        receipt_path = recording_import_receipt_path(
            identity.zarr_path,
            receipt_sha256,
        )
        try:
            receipt = RecordingImportReceipt.from_path(receipt_path)
        except RecordingImportReceiptError as exc:
            raise RecordingIdentityAuthorityError(
                "bound recording import receipt is missing or malformed"
            ) from exc
        if receipt.receipt_sha256 != receipt_sha256:
            raise RecordingIdentityProjectionConflict(
                "bound receipt digest does not match its registry binding"
            )
        binding_evidence = _load_stored_evidence(
            self.conn,
            identity.evidence_digest,
            required=True,
        )
        assert binding_evidence is not None
        try:
            _receipt_digest, ownership, frame, frame_path = (
                _verify_live_import_receipt(
                resolved_path=identity.zarr_path,
                evidence=binding_evidence,
                receipt=receipt,
            )
            )
        except RecordingIdentityAuthorityError:
            raise
        except Exception as exc:
            raise RecordingIdentityAuthorityError(
                "bound recording import receipt or acquisition authority is stale"
            ) from exc
        return VerifiedRecordingImport(
            identity=identity,
            receipt=receipt,
            receipt_path=receipt_path,
            acquisition_ownership=ownership,
            acquisition_frame=frame,
            acquisition_authority_path=frame_path,
        )


__all__ = [
    "IDENTITY_REVISION_SCHEMA_ID",
    "RecordingIdentityAuthorityError",
    "RecordingImportBindingNotFound",
    "RecordingIdentityProjectionConflict",
    "RecordingIdentityProjectionResult",
    "VerifiedRecordingIdentity",
    "VerifiedRecordingImport",
    "RegistryRecordingIdentityMixin",
    "canonical_dataset_path_hash",
    "collect_regular_source_recording_identity",
    "recording_directory_for_source_target",
]

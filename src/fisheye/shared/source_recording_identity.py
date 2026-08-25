"""Strict producer contract for current source-recording identity."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
import unicodedata

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)


SOURCE_RECORDING_IDENTITY_PROFILE = "palette.source_recording_identity.v2"
SOURCE_RECORDING_IDENTITY_PROFILE_ATTR = "source_recording_identity_profile"
SOURCE_RECORDING_ID_MAPPING_PROFILE = (
    "palette.source_recording_id.session_camera_sha256.v1"
)
SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR = "recording_id_mapping_profile"
SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID = (
    "palette.source_recording_identity_claim.v2"
)
SOURCE_RECORDING_IDENTITY_CLAIM_SOURCE_ROLES = (
    "recording_manifest",
    "zarr_root_direct_metadata",
)
MAX_RECORDING_MANIFEST_BYTES = 4 * 1024 * 1024

SOURCE_ANALYSIS_CLASSIFICATION = {
    "artifact_schema_id": "recording_analysis_v1",
    "artifact_kind": "source_recording",
    "zarr_origin": "source",
    "zarr_use": "analysis",
    "zarr_purpose": "analysis",
}


class SourceRecordingIdentityError(ValueError):
    """A current producer identity declaration is absent or malformed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token}")


def load_strict_json_object(
    path: Path,
    *,
    max_bytes: int = MAX_RECORDING_MANIFEST_BYTES,
) -> dict[str, Any]:
    """Read one stable, bounded UTF-8 JSON object without duplicate keys."""

    source = Path(path)
    try:
        before = source.stat()
        with source.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        after = source.stat()
    except OSError as exc:
        raise SourceRecordingIdentityError(
            f"could not read source-recording identity document {source}: {exc}"
        ) from exc
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise SourceRecordingIdentityError(
            f"source-recording identity document changed while read: {source}"
        )
    if before.st_size > max_bytes:
        raise SourceRecordingIdentityError(
            f"source-recording identity document exceeds {max_bytes} bytes: {source}"
        )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceRecordingIdentityError(
            f"source-recording identity document is not strict JSON: {source}: {exc}"
        ) from exc
    if type(payload) is not dict:
        raise SourceRecordingIdentityError(
            f"source-recording identity document must be an object: {source}"
        )
    return payload


def _require_exact_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value:
        raise SourceRecordingIdentityError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise SourceRecordingIdentityError(
            f"{field} must not have surrounding whitespace"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SourceRecordingIdentityError(
            f"{field} must contain valid Unicode text"
        ) from exc
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise SourceRecordingIdentityError(
            f"{field} must not contain control characters"
        )
    return value


def require_source_identity_text(value: Any, *, field: str) -> str:
    """Validate an identity scalar without trimming or coercion."""

    return _require_exact_text(value, field=field)


def recording_id_from_session_camera(
    *,
    session_uuid: Any,
    camera_id: Any,
) -> str:
    """Map one exact acquisition-session/camera pair to a stable recording ID."""

    document = {
        "schema_id": SOURCE_RECORDING_ID_MAPPING_PROFILE,
        "session_uuid": _require_exact_text(session_uuid, field="session_uuid"),
        "camera_id": _require_exact_text(camera_id, field="camera_id"),
    }
    return f"source_recording_{canonical_json_sha256(document)}"


@dataclass(frozen=True, slots=True)
class SourceRecordingIdentity:
    """Canonical identity of one camera recording and its parent session."""

    recording_id: str
    session_uuid: str
    camera_id: str
    recording_id_mapping_profile: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SourceRecordingIdentity":
        profile = payload.get(SOURCE_RECORDING_IDENTITY_PROFILE_ATTR)
        if profile != SOURCE_RECORDING_IDENTITY_PROFILE:
            raise SourceRecordingIdentityError(
                f"{SOURCE_RECORDING_IDENTITY_PROFILE_ATTR} must equal "
                f"{SOURCE_RECORDING_IDENTITY_PROFILE!r}"
            )
        identity = cls(
            recording_id=_require_exact_text(
                payload.get("recording_id"), field="recording_id"
            ),
            session_uuid=_require_exact_text(
                payload.get("session_uuid"), field="session_uuid"
            ),
            camera_id=_require_exact_text(payload.get("camera_id"), field="camera_id"),
            recording_id_mapping_profile=(
                _require_exact_text(
                    payload.get(SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR),
                    field=SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR,
                )
                if payload.get(SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR) is not None
                else None
            ),
        )
        if identity.recording_id_mapping_profile is not None:
            if (
                identity.recording_id_mapping_profile
                != SOURCE_RECORDING_ID_MAPPING_PROFILE
            ):
                raise SourceRecordingIdentityError(
                    f"{SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR} is unsupported"
                )
            expected = recording_id_from_session_camera(
                session_uuid=identity.session_uuid,
                camera_id=identity.camera_id,
            )
            if identity.recording_id != expected:
                raise SourceRecordingIdentityError(
                    "recording_id does not match its declared session/camera mapping"
                )
        return identity

    def manifest_fields(self) -> dict[str, str]:
        fields = {
            SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: SOURCE_RECORDING_IDENTITY_PROFILE,
            "recording_id": self.recording_id,
            "session_uuid": self.session_uuid,
            "camera_id": self.camera_id,
        }
        if self.recording_id_mapping_profile is not None:
            fields[SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR] = (
                self.recording_id_mapping_profile
            )
        return fields

    def analysis_root_fields(self) -> dict[str, str]:
        return {**self.manifest_fields(), **SOURCE_ANALYSIS_CLASSIFICATION}


def _identity_claim_payload(identity: SourceRecordingIdentity) -> dict[str, Any]:
    return {
        "schema_id": SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID,
        "canonicalization": CANONICAL_JSON_DIGEST_ALGORITHM,
        "identity": identity.manifest_fields(),
        "verified_source_roles": list(
            SOURCE_RECORDING_IDENTITY_CLAIM_SOURCE_ROLES
        ),
        "root_classification": dict(SOURCE_ANALYSIS_CLASSIFICATION),
    }


@dataclass(frozen=True, slots=True)
class SourceRecordingIdentityClaim:
    """One complete current-v2 identity observed equally at both source roles."""

    identity: SourceRecordingIdentity
    claim_sha256: str

    @classmethod
    def create(
        cls,
        identity: SourceRecordingIdentity,
    ) -> "SourceRecordingIdentityClaim":
        if type(identity) is not SourceRecordingIdentity:
            raise SourceRecordingIdentityError(
                "identity claim requires SourceRecordingIdentity"
            )
        normalized = SourceRecordingIdentity.from_mapping(identity.manifest_fields())
        return cls(
            identity=normalized,
            claim_sha256=canonical_json_sha256(_identity_claim_payload(normalized)),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SourceRecordingIdentityClaim":
        if type(value) is not dict or set(value) != {
            "schema_id",
            "canonicalization",
            "identity",
            "verified_source_roles",
            "root_classification",
            "claim_sha256",
        }:
            raise SourceRecordingIdentityError(
                "source-recording identity claim has an unsupported shape"
            )
        identity_payload = value["identity"]
        if type(identity_payload) is not dict:
            raise SourceRecordingIdentityError(
                "source-recording identity claim identity must be an object"
            )
        identity = SourceRecordingIdentity.from_mapping(identity_payload)
        rebuilt = cls.create(identity)
        if rebuilt.as_dict() != value:
            raise SourceRecordingIdentityError(
                "source-recording identity claim is not canonical"
            )
        return rebuilt

    def as_dict(self) -> dict[str, Any]:
        return {
            **_identity_claim_payload(self.identity),
            "claim_sha256": self.claim_sha256,
        }


def load_source_recording_identity(path: Path) -> tuple[dict[str, Any], SourceRecordingIdentity]:
    payload = load_strict_json_object(path)
    return payload, SourceRecordingIdentity.from_mapping(payload)


def _direct_root_attributes(root: Path) -> tuple[int, dict[str, Any]]:
    root = Path(root)
    if not root.is_dir():
        raise SourceRecordingIdentityError(
            f"source-recording Zarr root must be a directory: {root}"
        )
    metadata_path = root / "zarr.json"
    if not metadata_path.exists():
        legacy_metadata_path = root / ".zgroup"
        if not legacy_metadata_path.is_file():
            raise SourceRecordingIdentityError(
                f"source-recording Zarr root metadata is missing: {root}"
            )
        legacy = load_strict_json_object(
            legacy_metadata_path,
            max_bytes=MAX_RECORDING_MANIFEST_BYTES,
        )
        if legacy.get("zarr_format") != 2:
            raise SourceRecordingIdentityError(
                f"legacy Zarr root does not declare zarr_format 2: "
                f"{legacy_metadata_path}"
            )
        return 2, {}
    document = load_strict_json_object(
        metadata_path,
        max_bytes=MAX_RECORDING_MANIFEST_BYTES,
    )
    if type(document.get("zarr_format")) is not int or document["zarr_format"] != 3:
        raise SourceRecordingIdentityError(
            f"source-recording root metadata does not declare v3 zarr_format: "
            f"{metadata_path}"
        )
    if document.get("node_type") != "group":
        raise SourceRecordingIdentityError(
            f"source-recording root metadata does not declare a group: {metadata_path}"
        )
    attributes = document.get("attributes")
    if type(attributes) is not dict:
        raise SourceRecordingIdentityError(
            f"source-recording root attributes must be an object: {metadata_path}"
        )
    return 3, attributes


def _declared_identity_profile(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> str | None:
    if SOURCE_RECORDING_IDENTITY_PROFILE_ATTR not in payload:
        return None
    profile = payload[SOURCE_RECORDING_IDENTITY_PROFILE_ATTR]
    if type(profile) is not str:
        raise SourceRecordingIdentityError(
            f"{source} {SOURCE_RECORDING_IDENTITY_PROFILE_ATTR} must be an exact string"
        )
    if profile != SOURCE_RECORDING_IDENTITY_PROFILE:
        raise SourceRecordingIdentityError(
            f"{source} {SOURCE_RECORDING_IDENTITY_PROFILE_ATTR} is unsupported: "
            f"{profile!r}"
        )
    return profile


def _claim_from_source_mappings(
    manifest: Mapping[str, Any],
    root_attributes: Mapping[str, Any],
) -> SourceRecordingIdentityClaim:
    manifest_identity = SourceRecordingIdentity.from_mapping(manifest)
    for field, expected in SOURCE_ANALYSIS_CLASSIFICATION.items():
        if root_attributes.get(field) != expected:
            raise SourceRecordingIdentityError(
                f"source-recording root requires exact {field}={expected!r}; "
                f"observed {root_attributes.get(field)!r}"
            )
    root_identity = SourceRecordingIdentity.from_mapping(root_attributes)
    for field in (
        "recording_id",
        "session_uuid",
        "camera_id",
        "recording_id_mapping_profile",
    ):
        manifest_value = getattr(manifest_identity, field)
        root_value = getattr(root_identity, field)
        if manifest_value != root_value:
            raise SourceRecordingIdentityError(
                f"manifest/root {field} conflict: "
                f"manifest={manifest_value!r}, root={root_value!r}"
            )
    return SourceRecordingIdentityClaim.create(manifest_identity)


def load_source_recording_identity_claim(
    manifest_path: Path,
    root: Path,
) -> SourceRecordingIdentityClaim:
    """Load one exact current-v2 claim from its two declared source roles."""

    manifest = load_strict_json_object(Path(manifest_path))
    zarr_format, root_attributes = _direct_root_attributes(Path(root))
    if zarr_format != 3:
        raise SourceRecordingIdentityError(
            "current source-recording identity requires a Zarr v3 root"
        )
    return _claim_from_source_mappings(manifest, root_attributes)


def _recording_manifest_for_root(root: Path) -> Path | None:
    candidates: list[Path] = []
    for parent in Path(root).resolve().parents:
        if parent.name.casefold() == "recordings":
            break
        candidate = parent / "recording_manifest.json"
        if candidate.is_file():
            candidates.append(candidate)
    if len(candidates) > 1:
        raise SourceRecordingIdentityError(
            "source target has multiple recording-manifest ancestors"
        )
    return candidates[0] if candidates else None


def _is_explicit_non_source_root(attributes: Mapping[str, Any]) -> bool:
    return any(
        field in attributes and attributes[field] != expected
        for field, expected in (
            ("artifact_kind", "source_recording"),
            ("zarr_use", "analysis"),
            ("zarr_purpose", "analysis"),
        )
    )


def load_source_recording_identity_profile(root: Path) -> str | None:
    """Classify one target without allowing a one-sided current declaration.

    When a recording manifest ancestor exists, its profile and the direct Zarr
    root profile are classified together.  A current declaration on only one
    side is rejected, except that a root explicitly classified as a non-source
    artifact may coexist beneath a current recording manifest.  This prevents a
    stripped current root marker from falling through to a legacy writer.
    """

    root = Path(root)
    zarr_format, attributes = _direct_root_attributes(root)
    root_profile = _declared_identity_profile(attributes, source="Zarr root")
    manifest_path = _recording_manifest_for_root(root)
    if manifest_path is None:
        return root_profile

    manifest = load_strict_json_object(manifest_path)
    manifest_profile = _declared_identity_profile(
        manifest,
        source="recording manifest",
    )
    if root_profile == manifest_profile == SOURCE_RECORDING_IDENTITY_PROFILE:
        if zarr_format != 3:
            raise SourceRecordingIdentityError(
                "current source-recording identity requires a Zarr v3 root"
            )
        _claim_from_source_mappings(manifest, attributes)
        return SOURCE_RECORDING_IDENTITY_PROFILE
    if manifest_profile == SOURCE_RECORDING_IDENTITY_PROFILE:
        if root_profile is None and _is_explicit_non_source_root(attributes):
            return None
        raise SourceRecordingIdentityError(
            "current source-recording manifest and Zarr root profiles disagree"
        )
    if root_profile == SOURCE_RECORDING_IDENTITY_PROFILE:
        raise SourceRecordingIdentityError(
            "current source-recording Zarr root and recording manifest profiles disagree"
        )
    if all(
        attributes.get(field) == expected
        for field, expected in SOURCE_ANALYSIS_CLASSIFICATION.items()
    ):
        raise SourceRecordingIdentityError(
            "source-recording classification is missing its current identity profile"
        )
    return None


__all__ = [
    "MAX_RECORDING_MANIFEST_BYTES",
    "SOURCE_ANALYSIS_CLASSIFICATION",
    "SOURCE_RECORDING_IDENTITY_PROFILE",
    "SOURCE_RECORDING_IDENTITY_PROFILE_ATTR",
    "SOURCE_RECORDING_IDENTITY_CLAIM_SCHEMA_ID",
    "SOURCE_RECORDING_IDENTITY_CLAIM_SOURCE_ROLES",
    "SOURCE_RECORDING_ID_MAPPING_PROFILE",
    "SOURCE_RECORDING_ID_MAPPING_PROFILE_ATTR",
    "SourceRecordingIdentity",
    "SourceRecordingIdentityClaim",
    "SourceRecordingIdentityError",
    "load_source_recording_identity",
    "load_source_recording_identity_claim",
    "load_source_recording_identity_profile",
    "load_strict_json_object",
    "recording_id_from_session_camera",
    "require_source_identity_text",
]

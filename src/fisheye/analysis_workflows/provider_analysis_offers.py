"""Strict in-memory contracts for provider-bound Phase 4 analysis offers.

This module deliberately stops at the contract boundary.  It does not open
Zarr, resolve selectors, compile temporal selections, or compute a metric.
Each record names the immutable inputs that a later metric implementation is
allowed to consume.  The records are canonical JSON envelopes so an offer can
be copied between processes without changing its identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

PROVIDER_IDENTITY_SCHEMA_ID = "palette.provider_identity"
PROVIDER_IDENTITY_SCHEMA_VERSION = 1
PROVIDER_REQUIREMENTS_SCHEMA_ID = "palette.provider_requirements"
PROVIDER_REQUIREMENTS_SCHEMA_VERSION = 1
TEMPORAL_SELECTION_IDENTITY_SCHEMA_ID = "palette.resolved_temporal_selection_identity"
TEMPORAL_SELECTION_IDENTITY_SCHEMA_VERSION = 1
ANALYSIS_OFFER_SCHEMA_ID = "palette.provider_analysis_offer"
ANALYSIS_OFFER_SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSIONED_ID_RE = re.compile(
    r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*(?:\.v|_v)[1-9][0-9]*$"
)
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ARRAY_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")

_RESERVED_SELECTOR_NAMES = frozenset(
    {
        "active",
        "authoritative_run",
        "current",
        "default",
        "latest",
        "latest_complete",
        "latest_pending",
        "selected",
    }
)


class ProviderAnalysisOfferError(ValueError):
    """Raised when a provider-bound contract is not exact and immutable."""


class ProviderRole(str, Enum):
    """The independent consumer role supplied by one provider."""

    POSITION = "position"
    BODY_FRAME = "body_frame"
    MOTION = "motion"


class ProviderKind(str, Enum):
    """The kind of published provider behind a consumer role."""

    DETECTION = "detection"
    KEYPOINT = "keypoint"
    SUBJECT_MASK = "subject_mask"
    TRACK_MOTION = "track_motion"


class ScientificReadiness(str, Enum):
    """The only readiness dimension in this contract slice."""

    READY = "ready"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_RECORDING_AUTHORITY = "blocked_recording_authority"
    BLOCKED_TEMPORAL_AUTHORITY = "blocked_temporal_authority"
    STALE_LINEAGE = "stale_lineage"
    INVALID_CONTRACT = "invalid_contract"


_ROLE_KINDS: dict[ProviderRole, frozenset[ProviderKind]] = {
    ProviderRole.POSITION: frozenset(
        {ProviderKind.DETECTION, ProviderKind.KEYPOINT, ProviderKind.SUBJECT_MASK}
    ),
    ProviderRole.BODY_FRAME: frozenset(
        {ProviderKind.KEYPOINT, ProviderKind.SUBJECT_MASK}
    ),
    ProviderRole.MOTION: frozenset({ProviderKind.TRACK_MOTION}),
}


def _plain_json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ProviderAnalysisOfferError(f"{name} must be one nonempty JSON object.")
    if any(type(key) is not str for key in value):
        raise ProviderAnalysisOfferError(f"{name} keys must be strings.")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProviderAnalysisOfferError(f"{name} must be strict JSON.") from exc
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise ProviderAnalysisOfferError(f"{name} must be one JSON object.")
    return decoded


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProviderAnalysisOfferError(
            f"{name} has an inexact field set (missing={missing!r}, extra={extra!r})."
        )


def _require_schema(
    value: Mapping[str, Any], *, schema_id: str, schema_version: int, name: str
) -> None:
    if (
        value.get("schema_id") != schema_id
        or value.get("schema_version") != schema_version
    ):
        raise ProviderAnalysisOfferError(f"{name} schema identity is invalid.")


def _require_sha256(
    value: object, *, name: str, allow_none: bool = False
) -> str | None:
    if allow_none and value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ProviderAnalysisOfferError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _require_versioned_id(value: object, *, name: str) -> str:
    if type(value) is not str or _VERSIONED_ID_RE.fullmatch(value) is None:
        raise ProviderAnalysisOfferError(
            f"{name} must be one lowercase versioned identifier ending in .vN."
        )
    return value


def _require_id(value: object, *, name: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ProviderAnalysisOfferError(f"{name} must be one lowercase identifier.")
    return value


def _require_recording_id(value: object, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ProviderAnalysisOfferError(
            "recording_id must be one nonempty exact identifier."
        )
    return value


def _require_run_path(value: object, *, name: str = "run_path") -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderAnalysisOfferError(
            f"{name} must be one relative canonical immutable run path."
        )
    if value.startswith("/") or value.endswith("/") or "\\" in value or "//" in value:
        raise ProviderAnalysisOfferError(f"{name} is not canonical.")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ProviderAnalysisOfferError(f"{name} is not canonical.")
    parts = path.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ProviderAnalysisOfferError(f"{name} contains traversal components.")
    if len(parts) < 3 or parts[0] != "analysis":
        raise ProviderAnalysisOfferError(
            f"{name} must identify one concrete path under analysis/."
        )
    if any(part in _RESERVED_SELECTOR_NAMES for part in parts):
        raise ProviderAnalysisOfferError(
            f"{name} must not contain a selector or alias component."
        )
    return value


def _require_array_names(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ProviderAnalysisOfferError(
            f"{name} must be one nonempty ordered list of validity arrays."
        )
    names = tuple(value)
    if any(
        type(item) is not str or _ARRAY_PATH_RE.fullmatch(item) is None
        for item in names
    ):
        raise ProviderAnalysisOfferError(
            f"{name} contains an invalid array name or path."
        )
    if len(set(names)) != len(names):
        raise ProviderAnalysisOfferError(f"{name} contains duplicate array names.")
    return names


def _binding_record(
    value: object,
    *,
    name: str,
    parser: Any,
    expected_digest: str | None = None,
) -> Any:
    if not isinstance(value, Mapping):
        raise ProviderAnalysisOfferError(f"{name} binding must be one object.")
    binding = _plain_json_object(value, name=f"{name} binding")
    _exact_fields(binding, {"record", "sha256"}, name=f"{name} binding")
    digest = _require_sha256(binding["sha256"], name=f"{name} binding sha256")
    assert digest is not None
    if expected_digest is not None and digest != expected_digest:
        raise ProviderAnalysisOfferError(f"{name} digest differs from expectation.")
    record = parser(binding["record"])
    if record.sha256 != digest:
        raise ProviderAnalysisOfferError(f"{name} binding digest is stale.")
    return record


@dataclass(frozen=True)
class ProviderIdentity:
    """One exact immutable provider identity for one independent role."""

    role: ProviderRole
    kind: ProviderKind
    modality: str
    provider_id: str
    run_path: str
    recording_id: str | None
    manifest_sha256: str
    decoded_content_sha256: str | None
    coordinate_authority_sha256: str | None
    timing_authority_sha256: str | None
    validity_array_names: tuple[str, ...]

    def __post_init__(self) -> None:
        role = _coerce_enum(self.role, ProviderRole, name="provider role")
        kind = _coerce_enum(self.kind, ProviderKind, name="provider kind")
        if kind not in _ROLE_KINDS[role]:
            raise ProviderAnalysisOfferError(
                f"Provider kind {kind.value!r} is incompatible with role {role.value!r}."
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "kind", kind)
        _require_versioned_id(self.modality, name="provider modality")
        _require_versioned_id(self.provider_id, name="provider_id")
        object.__setattr__(self, "run_path", _require_run_path(self.run_path))
        _require_recording_id(self.recording_id, allow_none=True)
        _require_sha256(self.manifest_sha256, name="manifest_sha256")
        _require_sha256(
            self.decoded_content_sha256,
            name="decoded_content_sha256",
            allow_none=True,
        )
        coordinate = _require_sha256(
            self.coordinate_authority_sha256,
            name="coordinate_authority_sha256",
            allow_none=True,
        )
        _require_sha256(
            self.timing_authority_sha256,
            name="timing_authority_sha256",
            allow_none=True,
        )
        if coordinate is None:
            raise ProviderAnalysisOfferError(
                "Every provider must bind an exact coordinate authority digest."
            )
        object.__setattr__(
            self,
            "validity_array_names",
            _require_array_names(
                self.validity_array_names,
                name="validity_array_names",
            ),
        )

    @property
    def record(self) -> dict[str, Any]:
        return {
            "schema_id": PROVIDER_IDENTITY_SCHEMA_ID,
            "schema_version": PROVIDER_IDENTITY_SCHEMA_VERSION,
            "role": self.role.value,
            "kind": self.kind.value,
            "modality": self.modality,
            "provider_id": self.provider_id,
            "run_path": self.run_path,
            "recording_id": self.recording_id,
            "manifest_sha256": self.manifest_sha256,
            "decoded_content_sha256": self.decoded_content_sha256,
            "coordinate_authority_sha256": self.coordinate_authority_sha256,
            "timing_authority_sha256": self.timing_authority_sha256,
            "validity_array_names": list(self.validity_array_names),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.record)

    @property
    def digest(self) -> str:
        return self.sha256

    @property
    def source_run_path(self) -> str:
        return self.run_path

    def as_envelope(self) -> dict[str, Any]:
        return {"record": self.record, "sha256": self.sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "ProviderIdentity":
        record = _plain_json_object(value, name="provider identity record")
        _exact_fields(
            record,
            {
                "schema_id",
                "schema_version",
                "role",
                "kind",
                "modality",
                "provider_id",
                "run_path",
                "recording_id",
                "manifest_sha256",
                "decoded_content_sha256",
                "coordinate_authority_sha256",
                "timing_authority_sha256",
                "validity_array_names",
            },
            name="provider identity record",
        )
        _require_schema(
            record,
            schema_id=PROVIDER_IDENTITY_SCHEMA_ID,
            schema_version=PROVIDER_IDENTITY_SCHEMA_VERSION,
            name="provider identity",
        )
        return cls(
            role=_coerce_enum(record["role"], ProviderRole, name="provider role"),
            kind=_coerce_enum(record["kind"], ProviderKind, name="provider kind"),
            modality=record["modality"],
            provider_id=record["provider_id"],
            run_path=record["run_path"],
            recording_id=record["recording_id"],
            manifest_sha256=record["manifest_sha256"],
            decoded_content_sha256=record["decoded_content_sha256"],
            coordinate_authority_sha256=record["coordinate_authority_sha256"],
            timing_authority_sha256=record["timing_authority_sha256"],
            validity_array_names=tuple(record["validity_array_names"]),
        )

    @classmethod
    def from_envelope(
        cls, value: Mapping[str, Any], *, expected_digest: str | None = None
    ) -> "ProviderIdentity":
        return _binding_record(
            value,
            name="provider identity",
            parser=cls.from_record,
            expected_digest=expected_digest,
        )


@dataclass(frozen=True)
class ProviderRequirements:
    """Independent exact requirements for position, body frame, and motion."""

    position: ProviderIdentity | None = None
    body_frame: ProviderIdentity | None = None
    motion: ProviderIdentity | None = None

    def __post_init__(self) -> None:
        values = {
            ProviderRole.POSITION: self.position,
            ProviderRole.BODY_FRAME: self.body_frame,
            ProviderRole.MOTION: self.motion,
        }
        if not any(value is not None for value in values.values()):
            raise ProviderAnalysisOfferError(
                "Provider requirements must declare at least one required role."
            )
        for role, provider in values.items():
            if provider is not None:
                if not isinstance(provider, ProviderIdentity):
                    raise ProviderAnalysisOfferError(
                        f"{role.value} requirement must be a ProviderIdentity."
                    )
                if provider.role is not role:
                    raise ProviderAnalysisOfferError(
                        f"Provider role {provider.role.value!r} cannot satisfy the "
                        f"{role.value!r} requirement."
                    )

    @property
    def required_roles(self) -> tuple[str, ...]:
        return tuple(
            role.value
            for role, provider in (
                (ProviderRole.POSITION, self.position),
                (ProviderRole.BODY_FRAME, self.body_frame),
                (ProviderRole.MOTION, self.motion),
            )
            if provider is not None
        )

    @property
    def record(self) -> dict[str, Any]:
        return {
            "schema_id": PROVIDER_REQUIREMENTS_SCHEMA_ID,
            "schema_version": PROVIDER_REQUIREMENTS_SCHEMA_VERSION,
            "required_roles": list(self.required_roles),
            "position": None if self.position is None else self.position.as_envelope(),
            "body_frame": (
                None if self.body_frame is None else self.body_frame.as_envelope()
            ),
            "motion": None if self.motion is None else self.motion.as_envelope(),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.record)

    @property
    def digest(self) -> str:
        return self.sha256

    def as_envelope(self) -> dict[str, Any]:
        return {"record": self.record, "sha256": self.sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "ProviderRequirements":
        record = _plain_json_object(value, name="provider requirements record")
        _exact_fields(
            record,
            {
                "schema_id",
                "schema_version",
                "required_roles",
                "position",
                "body_frame",
                "motion",
            },
            name="provider requirements record",
        )
        _require_schema(
            record,
            schema_id=PROVIDER_REQUIREMENTS_SCHEMA_ID,
            schema_version=PROVIDER_REQUIREMENTS_SCHEMA_VERSION,
            name="provider requirements",
        )
        required_roles = record["required_roles"]
        if not isinstance(required_roles, list) or any(
            type(role) is not str for role in required_roles
        ):
            raise ProviderAnalysisOfferError(
                "provider requirements required_roles must be a string list."
            )
        identities: dict[str, ProviderIdentity | None] = {}
        for role in ("position", "body_frame", "motion"):
            binding = record[role]
            identities[role] = (
                None if binding is None else ProviderIdentity.from_envelope(binding)
            )
        result = cls(**identities)
        if tuple(required_roles) != result.required_roles:
            raise ProviderAnalysisOfferError(
                "provider requirements required_roles does not match its bindings."
            )
        return result

    @classmethod
    def from_envelope(
        cls, value: Mapping[str, Any], *, expected_digest: str | None = None
    ) -> "ProviderRequirements":
        return _binding_record(
            value,
            name="provider requirements",
            parser=cls.from_record,
            expected_digest=expected_digest,
        )


@dataclass(frozen=True)
class TemporalSelectionIdentity:
    """One already-resolved, exact temporal selection or epoch identity."""

    selection_id: str
    run_path: str
    recording_id: str
    source_timeline_sha256: str
    resolved_sha256: str

    def __post_init__(self) -> None:
        _require_versioned_id(self.selection_id, name="temporal selection_id")
        object.__setattr__(
            self, "run_path", _require_run_path(self.run_path, name="temporal run_path")
        )
        _require_recording_id(self.recording_id)
        _require_sha256(self.source_timeline_sha256, name="source_timeline_sha256")
        _require_sha256(self.resolved_sha256, name="resolved_sha256")

    @property
    def record(self) -> dict[str, Any]:
        return {
            "schema_id": TEMPORAL_SELECTION_IDENTITY_SCHEMA_ID,
            "schema_version": TEMPORAL_SELECTION_IDENTITY_SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "run_path": self.run_path,
            "recording_id": self.recording_id,
            "source_timeline_sha256": self.source_timeline_sha256,
            "resolved_sha256": self.resolved_sha256,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.record)

    @property
    def digest(self) -> str:
        return self.sha256

    def as_envelope(self) -> dict[str, Any]:
        return {"record": self.record, "sha256": self.sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "TemporalSelectionIdentity":
        record = _plain_json_object(value, name="temporal selection record")
        _exact_fields(
            record,
            {
                "schema_id",
                "schema_version",
                "selection_id",
                "run_path",
                "recording_id",
                "source_timeline_sha256",
                "resolved_sha256",
            },
            name="temporal selection record",
        )
        _require_schema(
            record,
            schema_id=TEMPORAL_SELECTION_IDENTITY_SCHEMA_ID,
            schema_version=TEMPORAL_SELECTION_IDENTITY_SCHEMA_VERSION,
            name="temporal selection",
        )
        return cls(
            selection_id=record["selection_id"],
            run_path=record["run_path"],
            recording_id=record["recording_id"],
            source_timeline_sha256=record["source_timeline_sha256"],
            resolved_sha256=record["resolved_sha256"],
        )

    @classmethod
    def from_envelope(
        cls, value: Mapping[str, Any], *, expected_digest: str | None = None
    ) -> "TemporalSelectionIdentity":
        return _binding_record(
            value,
            name="temporal selection",
            parser=cls.from_record,
            expected_digest=expected_digest,
        )


@dataclass(frozen=True)
class AnalysisOffer:
    """An immutable, provider-bound, selector-ineligible analysis offer."""

    analysis_class_id: str
    analysis_class_version: int
    computation_id: str
    computation_version: int
    temporal_selection: TemporalSelectionIdentity
    provider_requirements: ProviderRequirements
    scientific_readiness: ScientificReadiness

    def __post_init__(self) -> None:
        _require_id(self.analysis_class_id, name="analysis_class_id")
        _require_id(self.computation_id, name="computation_id")
        if (
            type(self.analysis_class_version) is not int
            or self.analysis_class_version < 1
        ):
            raise ProviderAnalysisOfferError(
                "analysis_class_version must be one positive integer."
            )
        if type(self.computation_version) is not int or self.computation_version < 1:
            raise ProviderAnalysisOfferError(
                "computation_version must be one positive integer."
            )
        if not isinstance(self.temporal_selection, TemporalSelectionIdentity):
            raise ProviderAnalysisOfferError(
                "temporal_selection must be one TemporalSelectionIdentity."
            )
        if not isinstance(self.provider_requirements, ProviderRequirements):
            raise ProviderAnalysisOfferError(
                "provider_requirements must be one ProviderRequirements."
            )
        object.__setattr__(
            self,
            "scientific_readiness",
            _coerce_enum(
                self.scientific_readiness,
                ScientificReadiness,
                name="scientific readiness",
            ),
        )
        if self.scientific_readiness is ScientificReadiness.READY:
            for role in ("position", "body_frame", "motion"):
                provider = getattr(self.provider_requirements, role)
                if provider is None:
                    continue
                if provider.recording_id != self.temporal_selection.recording_id:
                    raise ProviderAnalysisOfferError(
                        "A ready offer must bind every provider to the temporal "
                        "selection's exact recording identity."
                    )
                if provider.timing_authority_sha256 is None:
                    raise ProviderAnalysisOfferError(
                        "A ready offer must bind every provider to an exact timing "
                        "authority digest."
                    )

    @property
    def selector_eligible(self) -> bool:
        return False

    @property
    def record(self) -> dict[str, Any]:
        return {
            "schema_id": ANALYSIS_OFFER_SCHEMA_ID,
            "schema_version": ANALYSIS_OFFER_SCHEMA_VERSION,
            "analysis_class_id": self.analysis_class_id,
            "analysis_class_version": self.analysis_class_version,
            "computation_id": self.computation_id,
            "computation_version": self.computation_version,
            "temporal_selection": self.temporal_selection.as_envelope(),
            "provider_requirements": self.provider_requirements.as_envelope(),
            "readiness": {"scientific": self.scientific_readiness.value},
            "selector_eligible": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.record)

    @property
    def digest(self) -> str:
        return self.sha256

    def as_envelope(self) -> dict[str, Any]:
        return {"record": self.record, "sha256": self.sha256}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "AnalysisOffer":
        record = _plain_json_object(value, name="analysis offer record")
        _exact_fields(
            record,
            {
                "schema_id",
                "schema_version",
                "analysis_class_id",
                "analysis_class_version",
                "computation_id",
                "computation_version",
                "temporal_selection",
                "provider_requirements",
                "readiness",
                "selector_eligible",
            },
            name="analysis offer record",
        )
        _require_schema(
            record,
            schema_id=ANALYSIS_OFFER_SCHEMA_ID,
            schema_version=ANALYSIS_OFFER_SCHEMA_VERSION,
            name="analysis offer",
        )
        if record["selector_eligible"] is not False:
            raise ProviderAnalysisOfferError(
                "analysis offers are permanently selector-ineligible in v1."
            )
        readiness = _plain_json_object(record["readiness"], name="offer readiness")
        _exact_fields(readiness, {"scientific"}, name="offer readiness")
        return cls(
            analysis_class_id=record["analysis_class_id"],
            analysis_class_version=record["analysis_class_version"],
            computation_id=record["computation_id"],
            computation_version=record["computation_version"],
            temporal_selection=TemporalSelectionIdentity.from_envelope(
                record["temporal_selection"]
            ),
            provider_requirements=ProviderRequirements.from_envelope(
                record["provider_requirements"]
            ),
            scientific_readiness=_coerce_enum(
                readiness["scientific"],
                ScientificReadiness,
                name="scientific readiness",
            ),
        )

    @classmethod
    def from_envelope(
        cls, value: Mapping[str, Any], *, expected_digest: str | None = None
    ) -> "AnalysisOffer":
        return _binding_record(
            value,
            name="analysis offer",
            parser=cls.from_record,
            expected_digest=expected_digest,
        )


def _coerce_enum(value: object, enum_type: type[Enum], *, name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if type(value) is str:
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ProviderAnalysisOfferError(f"Unknown {name}: {value!r}.") from exc
    raise ProviderAnalysisOfferError(f"{name} must be a known string value.")


def require_provider_identity(
    value: Mapping[str, Any], *, expected_digest: str | None = None
) -> ProviderIdentity:
    """Parse one exact provider-identity envelope."""

    return ProviderIdentity.from_envelope(value, expected_digest=expected_digest)


def require_provider_requirements(
    value: Mapping[str, Any], *, expected_digest: str | None = None
) -> ProviderRequirements:
    """Parse one exact provider-requirements envelope."""

    return ProviderRequirements.from_envelope(value, expected_digest=expected_digest)


def require_analysis_offer(
    value: Mapping[str, Any], *, expected_digest: str | None = None
) -> AnalysisOffer:
    """Parse one exact analysis-offer envelope."""

    return AnalysisOffer.from_envelope(value, expected_digest=expected_digest)


__all__ = [
    "ANALYSIS_OFFER_SCHEMA_ID",
    "ANALYSIS_OFFER_SCHEMA_VERSION",
    "AnalysisOffer",
    "ProviderAnalysisOfferError",
    "ProviderIdentity",
    "ProviderKind",
    "ProviderRequirements",
    "ProviderRole",
    "PROVIDER_IDENTITY_SCHEMA_ID",
    "PROVIDER_IDENTITY_SCHEMA_VERSION",
    "PROVIDER_REQUIREMENTS_SCHEMA_ID",
    "PROVIDER_REQUIREMENTS_SCHEMA_VERSION",
    "ScientificReadiness",
    "TEMPORAL_SELECTION_IDENTITY_SCHEMA_ID",
    "TEMPORAL_SELECTION_IDENTITY_SCHEMA_VERSION",
    "TemporalSelectionIdentity",
    "require_analysis_offer",
    "require_provider_identity",
    "require_provider_requirements",
]

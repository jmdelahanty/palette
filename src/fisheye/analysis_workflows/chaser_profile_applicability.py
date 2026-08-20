"""Deterministic capability and applicability planning for chaser profiles.

This module is intentionally independent from Zarr and from the cluster runner.
It consumes explicit capability assessments plus an already dependency-ordered
module selection.  It never discovers a provider, chooses a selector, or turns
an absent authority into scientific inapplicability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


CHASER_PROFILE_APPLICABILITY_SCHEMA_ID = (
    "palette.chaser_profile_applicability_plan"
)
CHASER_PROFILE_APPLICABILITY_SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ChaserProfileApplicabilityError(ValueError):
    """Raised when a profile plan is ambiguous or internally inconsistent."""


class CapabilityState(str, Enum):
    """State of one explicitly assessed scientific input capability."""

    READY = "ready"
    NOT_APPLICABLE = "not_applicable"
    MISSING = "missing"
    INVALID = "invalid"
    REVIEW_REQUIRED = "review_required"
    STALE = "stale"


class ModuleApplicabilityState(str, Enum):
    """Planning or terminal state for one selected analysis module."""

    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    BLOCKED_MISSING_CAPABILITY = "blocked_missing_capability"
    BLOCKED_INVALID_SOURCE = "blocked_invalid_source"
    BLOCKED_REVIEW_REQUIRED = "blocked_review_required"
    STALE = "stale"
    COMPLETE = "complete"


class ProfileReadiness(str, Enum):
    """Whether a plan may claim full-profile completion."""

    PLANNED = "planned"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    NOT_CLAIMED_REDUCED_PROFILE = "not_claimed_reduced_profile"


_REQUIREMENT_CLASSES = frozenset(
    {"required", "conditional_required", "optional"}
)
_PROFILE_SCOPES = frozenset({"full", "reduced"})

CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID = "chaser_temporal_alignment"
PHYSICAL_PRESENTATION_REQUIRED = "physical_presentation_required"
INPUT_PROVENANCE_PROXY_ALLOWED = "input_provenance_proxy_allowed"
CONTROLLER_INPUT_PROVENANCE_PROXY = "controller_input_provenance_proxy"
PRESENTATION_TIME_UNAVAILABLE = "presentation_time_unavailable"
LATEST_LOGGED_CPU_STATE_PROXY_POLICY_ID = (
    "latest_logged_cpu_state_per_input_acquisition_proxy_v1"
)

_PROXY_ALIGNMENT_EVIDENCE_FIELDS = frozenset(
    {
        "temporal_alignment_requirement",
        "temporal_alignment_class",
        "proxy_policy_id",
        "proxy_projection_sha256",
        "proxy_run_path",
        "proxy_manifest_sha256",
        "proxy_selector_eligible",
        "physical_presentation_verified",
        "presentation_timestamp_available",
        "camera_presentation_clock_transform_available",
        "camera_exposure_reference",
        "scientific_use_class",
        "behavioral_denominator",
        "carry_policy",
    }
)
_UNAVAILABLE_PHYSICAL_EVIDENCE_FIELDS = frozenset(
    {
        "temporal_alignment_requirement",
        "temporal_alignment_class",
        "physical_presentation_verified",
        "presentation_timestamp_available",
        "camera_presentation_clock_transform_available",
        "camera_exposure_reference",
        "scientific_use_class",
    }
)


def _require_id(value: object, *, name: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ChaserProfileApplicabilityError(
            f"{name} must be one lowercase controlled identifier."
        )
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ChaserProfileApplicabilityError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _strict_json(value: object, *, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ChaserProfileApplicabilityError(
            f"{name} must contain strict JSON values."
        ) from exc


def _validate_chaser_temporal_alignment_assessment(
    *,
    state: CapabilityState,
    reason_code: str,
    evidence: Mapping[str, Any],
) -> None:
    """Reject implicit fallback or physical claims unsupported by evidence."""

    requirement = evidence.get("temporal_alignment_requirement")
    if requirement == INPUT_PROVENANCE_PROXY_ALLOWED:
        if state is not CapabilityState.READY or reason_code != "proxy_alignment_ready":
            raise ChaserProfileApplicabilityError(
                "input-provenance proxy alignment must be an explicit ready capability."
            )
        if set(evidence) != _PROXY_ALIGNMENT_EVIDENCE_FIELDS:
            raise ChaserProfileApplicabilityError(
                "proxy temporal-alignment evidence has an inexact field set."
            )
        expected = {
            "temporal_alignment_class": CONTROLLER_INPUT_PROVENANCE_PROXY,
            "proxy_policy_id": LATEST_LOGGED_CPU_STATE_PROXY_POLICY_ID,
            "physical_presentation_verified": False,
            "presentation_timestamp_available": False,
            "camera_presentation_clock_transform_available": False,
            "camera_exposure_reference": "unknown",
            "scientific_use_class": "exploratory_proxy",
            "behavioral_denominator": "unique_input_acquisition_frames",
            "carry_policy": "no_carry_across_unmapped_input_acquisitions",
        }
        for field, expected_value in expected.items():
            if evidence.get(field) != expected_value:
                raise ChaserProfileApplicabilityError(
                    f"proxy temporal-alignment evidence has invalid {field}."
                )
        _require_sha256(
            evidence.get("proxy_projection_sha256"),
            name="proxy_projection_sha256",
        )
        _require_sha256(
            evidence.get("proxy_manifest_sha256"),
            name="proxy_manifest_sha256",
        )
        run_path = evidence.get("proxy_run_path")
        if type(run_path) is not str or not run_path or run_path != run_path.strip():
            raise ChaserProfileApplicabilityError(
                "proxy_run_path must be one explicit immutable run path."
            )
        if evidence.get("proxy_selector_eligible") is not False:
            raise ChaserProfileApplicabilityError(
                "The explicit proxy input must remain selector-ineligible."
            )
        return

    if requirement == PHYSICAL_PRESENTATION_REQUIRED:
        if (
            state is not CapabilityState.MISSING
            or reason_code != PRESENTATION_TIME_UNAVAILABLE
        ):
            raise ChaserProfileApplicabilityError(
                "current physical-presentation alignment must remain explicitly "
                "unavailable."
            )
        if set(evidence) != _UNAVAILABLE_PHYSICAL_EVIDENCE_FIELDS:
            raise ChaserProfileApplicabilityError(
                "unavailable physical-alignment evidence has an inexact field set."
            )
        expected = {
            "temporal_alignment_class": "unavailable",
            "physical_presentation_verified": False,
            "presentation_timestamp_available": False,
            "camera_presentation_clock_transform_available": False,
            "camera_exposure_reference": "unknown",
            "scientific_use_class": "unsupported",
        }
        for field, expected_value in expected.items():
            if evidence.get(field) != expected_value:
                raise ChaserProfileApplicabilityError(
                    f"physical temporal-alignment evidence has invalid {field}."
                )
        return

    raise ChaserProfileApplicabilityError(
        "chaser temporal alignment must declare one controlled requirement."
    )


def input_provenance_proxy_alignment_assessment(
    *,
    proxy_projection_sha256: str,
    proxy_run_path: str,
    proxy_manifest_sha256: str,
) -> "CapabilityAssessment":
    """Build the only ready current-recording temporal-alignment assessment."""

    return CapabilityAssessment(
        capability_id=CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
        state=CapabilityState.READY,
        reason_code="proxy_alignment_ready",
        evidence={
            "temporal_alignment_requirement": INPUT_PROVENANCE_PROXY_ALLOWED,
            "temporal_alignment_class": CONTROLLER_INPUT_PROVENANCE_PROXY,
            "proxy_policy_id": LATEST_LOGGED_CPU_STATE_PROXY_POLICY_ID,
            "proxy_projection_sha256": _require_sha256(
                proxy_projection_sha256,
                name="proxy_projection_sha256",
            ),
            "proxy_run_path": proxy_run_path,
            "proxy_manifest_sha256": _require_sha256(
                proxy_manifest_sha256,
                name="proxy_manifest_sha256",
            ),
            "proxy_selector_eligible": False,
            "physical_presentation_verified": False,
            "presentation_timestamp_available": False,
            "camera_presentation_clock_transform_available": False,
            "camera_exposure_reference": "unknown",
            "scientific_use_class": "exploratory_proxy",
            "behavioral_denominator": "unique_input_acquisition_frames",
            "carry_policy": "no_carry_across_unmapped_input_acquisitions",
        },
    )


def unavailable_physical_presentation_alignment_assessment() -> "CapabilityAssessment":
    """Build the fail-closed assessment for current physical-alignment requests."""

    return CapabilityAssessment(
        capability_id=CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
        state=CapabilityState.MISSING,
        reason_code=PRESENTATION_TIME_UNAVAILABLE,
        evidence={
            "temporal_alignment_requirement": PHYSICAL_PRESENTATION_REQUIRED,
            "temporal_alignment_class": "unavailable",
            "physical_presentation_verified": False,
            "presentation_timestamp_available": False,
            "camera_presentation_clock_transform_available": False,
            "camera_exposure_reference": "unknown",
            "scientific_use_class": "unsupported",
        },
    )


@dataclass(frozen=True)
class CapabilityAssessment:
    """An explicit assessment of one protocol or source capability.

    ``reason_code`` is intentionally separate from ``state`` so operators can
    distinguish, for example, an absent trial feature from a missing trial
    ledger.  Evidence records stay compact; row evidence belongs in arrays.
    """

    capability_id: str
    state: CapabilityState
    reason_code: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            _require_id(self.capability_id, name="capability_id"),
        )
        try:
            state = CapabilityState(self.state)
        except ValueError as exc:
            raise ChaserProfileApplicabilityError(
                f"unknown capability state: {self.state!r}"
            ) from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "reason_code",
            _require_id(self.reason_code, name="capability reason_code"),
        )
        evidence = _strict_json(dict(self.evidence), name="capability evidence")
        if not isinstance(evidence, dict):  # pragma: no cover - defensive
            raise ChaserProfileApplicabilityError(
                "capability evidence must be one JSON object."
            )
        if self.capability_id == CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID:
            _validate_chaser_temporal_alignment_assessment(
                state=state,
                reason_code=self.reason_code,
                evidence=evidence,
            )
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "state": self.state.value,
            "reason_code": self.reason_code,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CapabilityAssessment":
        if not isinstance(value, Mapping) or set(value) != {
            "capability_id",
            "state",
            "reason_code",
            "evidence",
        }:
            raise ChaserProfileApplicabilityError(
                "capability assessment has an inexact field set."
            )
        evidence = value["evidence"]
        if not isinstance(evidence, Mapping):
            raise ChaserProfileApplicabilityError(
                "capability assessment evidence must be one object."
            )
        return cls(
            capability_id=value["capability_id"],  # type: ignore[arg-type]
            state=CapabilityState(value["state"]),
            reason_code=value["reason_code"],  # type: ignore[arg-type]
            evidence=evidence,
        )


@dataclass(frozen=True)
class ModuleApplicabilityDecision:
    module_id: str
    requirement_class: str
    required_capabilities: tuple[str, ...]
    depends_on: tuple[str, ...]
    state: ModuleApplicabilityState
    reason_code: str
    implicated_capability_ids: tuple[str, ...]
    implicated_dependency_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "module_id",
            _require_id(self.module_id, name="module decision module_id"),
        )
        if self.requirement_class not in _REQUIREMENT_CLASSES:
            raise ChaserProfileApplicabilityError(
                f"unknown module requirement class: {self.requirement_class!r}"
            )
        try:
            state = ModuleApplicabilityState(self.state)
        except ValueError as exc:
            raise ChaserProfileApplicabilityError(
                f"unknown module applicability state: {self.state!r}"
            ) from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "reason_code",
            _require_id(self.reason_code, name="module decision reason_code"),
        )
        for field_name in (
            "required_capabilities",
            "depends_on",
            "implicated_capability_ids",
            "implicated_dependency_ids",
        ):
            values = tuple(
                _require_id(value, name=f"module decision {field_name}")
                for value in getattr(self, field_name)
            )
            if len(set(values)) != len(values):
                raise ChaserProfileApplicabilityError(
                    f"module decision {field_name} must be unique."
                )
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "requirement_class": self.requirement_class,
            "required_capabilities": list(self.required_capabilities),
            "depends_on": list(self.depends_on),
            "state": self.state.value,
            "reason_code": self.reason_code,
            "implicated_capability_ids": list(self.implicated_capability_ids),
            "implicated_dependency_ids": list(self.implicated_dependency_ids),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ModuleApplicabilityDecision":
        expected = {
            "module_id",
            "requirement_class",
            "required_capabilities",
            "depends_on",
            "state",
            "reason_code",
            "implicated_capability_ids",
            "implicated_dependency_ids",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ChaserProfileApplicabilityError(
                "module applicability decision has an inexact field set."
            )
        sequence_fields = (
            "required_capabilities",
            "depends_on",
            "implicated_capability_ids",
            "implicated_dependency_ids",
        )
        if any(
            isinstance(value[field], str) or not isinstance(value[field], Sequence)
            for field in sequence_fields
        ):
            raise ChaserProfileApplicabilityError(
                "module decision ID collections must be arrays."
            )
        return cls(
            module_id=value["module_id"],  # type: ignore[arg-type]
            requirement_class=value["requirement_class"],  # type: ignore[arg-type]
            required_capabilities=tuple(value["required_capabilities"]),
            depends_on=tuple(value["depends_on"]),
            state=ModuleApplicabilityState(value["state"]),
            reason_code=value["reason_code"],  # type: ignore[arg-type]
            implicated_capability_ids=tuple(value["implicated_capability_ids"]),
            implicated_dependency_ids=tuple(value["implicated_dependency_ids"]),
        )


@dataclass(frozen=True)
class ChaserProfileApplicabilityPlan:
    recording_id: str
    profile_id: str
    profile_version: int
    profile_sha256: str
    profile_scope: str
    capability_assessments: tuple[CapabilityAssessment, ...]
    module_decisions: tuple[ModuleApplicabilityDecision, ...]
    execution_order: tuple[str, ...]
    explicit_enable: tuple[str, ...]
    explicit_disable: tuple[str, ...]
    readiness: ProfileReadiness

    def __post_init__(self) -> None:
        if type(self.recording_id) is not str or not self.recording_id.strip():
            raise ChaserProfileApplicabilityError(
                "recording_id must be one nonempty exact identifier."
            )
        object.__setattr__(
            self,
            "profile_id",
            _require_id(self.profile_id, name="profile_id"),
        )
        if type(self.profile_version) is not int or self.profile_version < 1:
            raise ChaserProfileApplicabilityError(
                "profile_version must be one positive integer."
            )
        object.__setattr__(
            self,
            "profile_sha256",
            _require_sha256(self.profile_sha256, name="profile_sha256"),
        )
        if self.profile_scope not in _PROFILE_SCOPES:
            raise ChaserProfileApplicabilityError(
                f"profile_scope must be one of {sorted(_PROFILE_SCOPES)!r}."
            )
        try:
            readiness = ProfileReadiness(self.readiness)
        except ValueError as exc:
            raise ChaserProfileApplicabilityError(
                f"unknown profile readiness: {self.readiness!r}"
            ) from exc
        object.__setattr__(self, "readiness", readiness)
        if self.execution_order != tuple(
            row.module_id for row in self.module_decisions
        ):
            raise ChaserProfileApplicabilityError(
                "execution_order must exactly match module decision order."
            )
        if len(set(self.execution_order)) != len(self.execution_order):
            raise ChaserProfileApplicabilityError(
                "module decisions must have unique module IDs."
            )
        seen: set[str] = set()
        for row in self.module_decisions:
            missing = set(row.depends_on) - seen
            if missing:
                raise ChaserProfileApplicabilityError(
                    f"module {row.module_id!r} has unresolved dependencies "
                    f"{sorted(missing)!r}."
                )
            seen.add(row.module_id)
        capability_ids = tuple(
            row.capability_id for row in self.capability_assessments
        )
        if capability_ids != tuple(sorted(capability_ids)) or len(
            set(capability_ids)
        ) != len(capability_ids):
            raise ChaserProfileApplicabilityError(
                "capability assessments must be unique and sorted by ID."
            )
        expected_readiness = _profile_readiness(
            profile_scope=self.profile_scope,
            decisions=self.module_decisions,
        )
        if readiness is not expected_readiness:
            raise ChaserProfileApplicabilityError(
                "profile readiness disagrees with its module decisions."
            )

    def record(self) -> dict[str, Any]:
        return {
            "schema_id": CHASER_PROFILE_APPLICABILITY_SCHEMA_ID,
            "schema_version": CHASER_PROFILE_APPLICABILITY_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "profile": {
                "profile_id": self.profile_id,
                "profile_version": self.profile_version,
                "profile_sha256": self.profile_sha256,
                "profile_scope": self.profile_scope,
            },
            "capability_assessments": [
                assessment.to_dict()
                for assessment in self.capability_assessments
            ],
            "module_decisions": [
                decision.to_dict() for decision in self.module_decisions
            ],
            "execution_order": list(self.execution_order),
            "explicit_overrides": {
                "enable": list(self.explicit_enable),
                "disable": list(self.explicit_disable),
            },
            "readiness": self.readiness.value,
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.record())

    def as_envelope(self) -> dict[str, Any]:
        return {"record": self.record(), "sha256": self.sha256}

    @classmethod
    def from_record(cls, value: object) -> "ChaserProfileApplicabilityPlan":
        expected = {
            "schema_id",
            "schema_version",
            "recording_id",
            "profile",
            "capability_assessments",
            "module_decisions",
            "execution_order",
            "explicit_overrides",
            "readiness",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ChaserProfileApplicabilityError(
                "chaser profile applicability plan has an inexact field set."
            )
        if (
            value["schema_id"] != CHASER_PROFILE_APPLICABILITY_SCHEMA_ID
            or value["schema_version"]
            != CHASER_PROFILE_APPLICABILITY_SCHEMA_VERSION
        ):
            raise ChaserProfileApplicabilityError(
                "chaser profile applicability plan schema identity is invalid."
            )
        profile = value["profile"]
        overrides = value["explicit_overrides"]
        if not isinstance(profile, Mapping) or set(profile) != {
            "profile_id",
            "profile_version",
            "profile_sha256",
            "profile_scope",
        }:
            raise ChaserProfileApplicabilityError(
                "applicability plan profile binding is invalid."
            )
        if not isinstance(overrides, Mapping) or set(overrides) != {
            "enable",
            "disable",
        }:
            raise ChaserProfileApplicabilityError(
                "applicability plan explicit overrides are invalid."
            )
        for field_name in (
            "capability_assessments",
            "module_decisions",
            "execution_order",
        ):
            if isinstance(value[field_name], str) or not isinstance(
                value[field_name], Sequence
            ):
                raise ChaserProfileApplicabilityError(
                    f"applicability plan {field_name} must be an array."
                )
        if any(
            isinstance(overrides[field], str)
            or not isinstance(overrides[field], Sequence)
            for field in ("enable", "disable")
        ):
            raise ChaserProfileApplicabilityError(
                "applicability plan overrides must be arrays."
            )
        return cls(
            recording_id=value["recording_id"],  # type: ignore[arg-type]
            profile_id=profile["profile_id"],  # type: ignore[arg-type]
            profile_version=profile["profile_version"],  # type: ignore[arg-type]
            profile_sha256=profile["profile_sha256"],  # type: ignore[arg-type]
            profile_scope=profile["profile_scope"],  # type: ignore[arg-type]
            capability_assessments=tuple(
                CapabilityAssessment.from_dict(row)
                for row in value["capability_assessments"]
            ),
            module_decisions=tuple(
                ModuleApplicabilityDecision.from_dict(row)
                for row in value["module_decisions"]
            ),
            execution_order=tuple(value["execution_order"]),
            explicit_enable=tuple(overrides["enable"]),
            explicit_disable=tuple(overrides["disable"]),
            readiness=ProfileReadiness(value["readiness"]),
        )


def _module_fields(module: object) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    try:
        module_id = _require_id(getattr(module, "module_id"), name="module_id")
        requirement_class = str(getattr(module, "requirement_class"))
        required_capabilities = tuple(getattr(module, "required_capabilities"))
        depends_on = tuple(getattr(module, "depends_on"))
    except (AttributeError, TypeError) as exc:
        raise ChaserProfileApplicabilityError(
            "selected modules must expose module_id, requirement_class, "
            "required_capabilities, and depends_on."
        ) from exc
    if requirement_class not in _REQUIREMENT_CLASSES:
        raise ChaserProfileApplicabilityError(
            f"module {module_id!r} has unknown requirement class "
            f"{requirement_class!r}."
        )
    capabilities = tuple(
        _require_id(value, name=f"{module_id} required capability")
        for value in required_capabilities
    )
    dependencies = tuple(
        _require_id(value, name=f"{module_id} dependency") for value in depends_on
    )
    if len(set(capabilities)) != len(capabilities):
        raise ChaserProfileApplicabilityError(
            f"module {module_id!r} repeats a required capability."
        )
    if len(set(dependencies)) != len(dependencies):
        raise ChaserProfileApplicabilityError(
            f"module {module_id!r} repeats a dependency."
        )
    return module_id, requirement_class, capabilities, dependencies


def _capability_block(
    assessments: Sequence[CapabilityAssessment],
    *,
    requirement_class: str,
) -> tuple[ModuleApplicabilityState, str, tuple[str, ...]] | None:
    by_state = {
        state: tuple(
            assessment.capability_id
            for assessment in assessments
            if assessment.state is state
        )
        for state in CapabilityState
    }
    # A conditional/optional module is scientifically inapplicable as soon as
    # one of its required capabilities is explicitly declared inapplicable.
    # Missing or invalid evidence for another capability must not turn work
    # that the protocol cannot support into a false operational failure.  A
    # profile-owned required module remains fail-closed instead.
    if by_state[CapabilityState.NOT_APPLICABLE]:
        if requirement_class == "required":
            return (
                ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY,
                "required_capability_declared_not_applicable",
                by_state[CapabilityState.NOT_APPLICABLE],
            )
        return (
            ModuleApplicabilityState.INAPPLICABLE,
            "capability_not_applicable",
            by_state[CapabilityState.NOT_APPLICABLE],
        )
    if by_state[CapabilityState.INVALID]:
        return (
            ModuleApplicabilityState.BLOCKED_INVALID_SOURCE,
            "capability_invalid",
            by_state[CapabilityState.INVALID],
        )
    if by_state[CapabilityState.REVIEW_REQUIRED]:
        return (
            ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED,
            "capability_review_required",
            by_state[CapabilityState.REVIEW_REQUIRED],
        )
    if by_state[CapabilityState.STALE]:
        return (
            ModuleApplicabilityState.STALE,
            "capability_stale",
            by_state[CapabilityState.STALE],
        )
    if by_state[CapabilityState.MISSING]:
        return (
            ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY,
            "capability_missing",
            by_state[CapabilityState.MISSING],
        )
    return None


def _dependency_block(
    dependencies: Sequence[ModuleApplicabilityDecision],
    *,
    requirement_class: str,
) -> tuple[ModuleApplicabilityState, str, tuple[str, ...]] | None:
    terminal = {
        ModuleApplicabilityState.BLOCKED_INVALID_SOURCE,
        ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED,
        ModuleApplicabilityState.STALE,
        ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY,
        ModuleApplicabilityState.INAPPLICABLE,
    }
    implicated = tuple(row.module_id for row in dependencies if row.state in terminal)
    if not implicated:
        return None
    states = {row.state for row in dependencies if row.module_id in implicated}
    if (
        requirement_class != "required"
        and ModuleApplicabilityState.INAPPLICABLE in states
    ):
        state = ModuleApplicabilityState.INAPPLICABLE
    elif ModuleApplicabilityState.BLOCKED_INVALID_SOURCE in states:
        state = ModuleApplicabilityState.BLOCKED_INVALID_SOURCE
    elif ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED in states:
        state = ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED
    elif ModuleApplicabilityState.STALE in states:
        state = ModuleApplicabilityState.STALE
    elif ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY in states:
        state = ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    elif requirement_class == "required":
        state = ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    else:
        state = ModuleApplicabilityState.INAPPLICABLE
    return state, "dependency_not_applicable_or_blocked", implicated


def _profile_readiness(
    *,
    profile_scope: str,
    decisions: Sequence[ModuleApplicabilityDecision],
) -> ProfileReadiness:
    if profile_scope == "reduced":
        return ProfileReadiness.NOT_CLAIMED_REDUCED_PROFILE
    readiness_rows = tuple(
        row
        for row in decisions
        if row.requirement_class in {"required", "conditional_required"}
        and row.state is not ModuleApplicabilityState.INAPPLICABLE
    )
    if any(
        row.state
        in {
            ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY,
            ModuleApplicabilityState.BLOCKED_INVALID_SOURCE,
            ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED,
            ModuleApplicabilityState.STALE,
        }
        for row in readiness_rows
    ):
        return ProfileReadiness.BLOCKED
    if readiness_rows and all(
        row.state is ModuleApplicabilityState.COMPLETE for row in readiness_rows
    ):
        return ProfileReadiness.COMPLETE
    return ProfileReadiness.PLANNED


def plan_chaser_profile_applicability(
    *,
    recording_id: str,
    profile_id: str,
    profile_version: int,
    profile_sha256: str,
    profile_scope: str,
    selected_modules: Sequence[object],
    capability_assessments: Iterable[CapabilityAssessment],
    completed_module_ids: Iterable[str] = (),
    explicit_enable: Iterable[str] = (),
    explicit_disable: Iterable[str] = (),
) -> ChaserProfileApplicabilityPlan:
    """Plan selected modules from explicit capabilities without fail-open rules."""

    if type(recording_id) is not str or not recording_id.strip():
        raise ChaserProfileApplicabilityError(
            "recording_id must be one nonempty exact identifier."
        )
    profile_id = _require_id(profile_id, name="profile_id")
    if type(profile_version) is not int or profile_version < 1:
        raise ChaserProfileApplicabilityError(
            "profile_version must be one positive integer."
        )
    profile_sha256 = _require_sha256(profile_sha256, name="profile_sha256")
    if profile_scope not in _PROFILE_SCOPES:
        raise ChaserProfileApplicabilityError(
            f"profile_scope must be one of {sorted(_PROFILE_SCOPES)!r}."
        )

    assessments = tuple(capability_assessments)
    if any(not isinstance(row, CapabilityAssessment) for row in assessments):
        raise ChaserProfileApplicabilityError(
            "capability_assessments must contain CapabilityAssessment values."
        )
    assessment_by_id = {row.capability_id: row for row in assessments}
    if len(assessment_by_id) != len(assessments):
        raise ChaserProfileApplicabilityError(
            "capability assessments must have unique capability_id values."
        )
    assessments = tuple(sorted(assessments, key=lambda row: row.capability_id))

    enabled = tuple(_require_id(value, name="enabled module") for value in explicit_enable)
    disabled = tuple(
        _require_id(value, name="disabled module") for value in explicit_disable
    )
    if len(set(enabled)) != len(enabled) or len(set(disabled)) != len(disabled):
        raise ChaserProfileApplicabilityError("explicit overrides must be unique.")
    if set(enabled) & set(disabled):
        raise ChaserProfileApplicabilityError(
            "a module cannot be explicitly enabled and disabled."
        )
    completed = {
        _require_id(value, name="completed module") for value in completed_module_ids
    }

    decisions: list[ModuleApplicabilityDecision] = []
    decision_by_id: dict[str, ModuleApplicabilityDecision] = {}
    selected_ids: list[str] = []
    for module in selected_modules:
        module_id, requirement_class, capabilities, dependencies = _module_fields(module)
        if module_id in decision_by_id:
            raise ChaserProfileApplicabilityError(
                f"selected module {module_id!r} is duplicated."
            )
        missing_dependencies = tuple(
            dependency for dependency in dependencies if dependency not in decision_by_id
        )
        if missing_dependencies:
            raise ChaserProfileApplicabilityError(
                f"selected modules are not dependency ordered for {module_id!r}; "
                f"unresolved={list(missing_dependencies)!r}."
            )

        required_assessment_rows: list[CapabilityAssessment] = []
        for capability_id in capabilities:
            assessment = assessment_by_id.get(capability_id)
            if assessment is None:
                if capability_id == CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID:
                    assessment = unavailable_physical_presentation_alignment_assessment()
                else:
                    assessment = CapabilityAssessment(
                        capability_id=capability_id,
                        state=CapabilityState.MISSING,
                        reason_code="capability_not_assessed",
                        evidence={},
                    )
            required_assessment_rows.append(assessment)
        required_assessments = tuple(required_assessment_rows)
        capability_block = _capability_block(
            required_assessments,
            requirement_class=requirement_class,
        )
        dependency_block = _dependency_block(
            tuple(decision_by_id[value] for value in dependencies),
            requirement_class=requirement_class,
        )
        if capability_block is not None:
            state, reason_code, implicated_capabilities = capability_block
            implicated_dependencies: tuple[str, ...] = ()
        elif dependency_block is not None:
            state, reason_code, implicated_dependencies = dependency_block
            implicated_capabilities = ()
        elif module_id in completed:
            dependencies_complete = all(
                decision_by_id[value].state is ModuleApplicabilityState.COMPLETE
                for value in dependencies
            )
            if dependencies_complete:
                state = ModuleApplicabilityState.COMPLETE
                reason_code = "validated_immutable_product_complete"
            else:
                state = ModuleApplicabilityState.APPLICABLE
                reason_code = "dependency_execution_pending"
            implicated_capabilities = ()
            implicated_dependencies = ()
        else:
            state = ModuleApplicabilityState.APPLICABLE
            reason_code = "requirements_satisfied_execution_pending"
            implicated_capabilities = ()
            implicated_dependencies = ()

        decision = ModuleApplicabilityDecision(
            module_id=module_id,
            requirement_class=requirement_class,
            required_capabilities=capabilities,
            depends_on=dependencies,
            state=state,
            reason_code=reason_code,
            implicated_capability_ids=implicated_capabilities,
            implicated_dependency_ids=implicated_dependencies,
        )
        decisions.append(decision)
        decision_by_id[module_id] = decision
        selected_ids.append(module_id)

    unknown_completed = sorted(completed - set(selected_ids))
    if unknown_completed:
        raise ChaserProfileApplicabilityError(
            f"completed modules are not selected: {unknown_completed!r}."
        )

    readiness = _profile_readiness(
        profile_scope=profile_scope,
        decisions=decisions,
    )

    return ChaserProfileApplicabilityPlan(
        recording_id=recording_id,
        profile_id=profile_id,
        profile_version=profile_version,
        profile_sha256=profile_sha256,
        profile_scope=profile_scope,
        capability_assessments=assessments,
        module_decisions=tuple(decisions),
        execution_order=tuple(selected_ids),
        explicit_enable=enabled,
        explicit_disable=disabled,
        readiness=readiness,
    )


def require_chaser_profile_applicability_plan(
    value: object,
    *,
    expected_sha256: str | None = None,
) -> ChaserProfileApplicabilityPlan:
    """Parse one exact digest-bound plan envelope and reject rehashed swaps."""

    if not isinstance(value, Mapping) or set(value) != {"record", "sha256"}:
        raise ChaserProfileApplicabilityError(
            "applicability plan envelope must contain exact record and sha256 fields."
        )
    digest = _require_sha256(value["sha256"], name="plan envelope sha256")
    record = value["record"]
    if not isinstance(record, Mapping):
        raise ChaserProfileApplicabilityError(
            "applicability plan envelope record must be one object."
        )
    if canonical_json_sha256(record) != digest:
        raise ChaserProfileApplicabilityError(
            "applicability plan envelope digest is invalid."
        )
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256,
        name="expected plan sha256",
    ):
        raise ChaserProfileApplicabilityError(
            "applicability plan digest differs from expectation."
        )
    plan = ChaserProfileApplicabilityPlan.from_record(record)
    if plan.sha256 != digest:
        raise ChaserProfileApplicabilityError(
            "parsed applicability plan identity changed during normalization."
        )
    return plan


__all__ = [
    "CHASER_PROFILE_APPLICABILITY_SCHEMA_ID",
    "CHASER_PROFILE_APPLICABILITY_SCHEMA_VERSION",
    "CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID",
    "CONTROLLER_INPUT_PROVENANCE_PROXY",
    "INPUT_PROVENANCE_PROXY_ALLOWED",
    "LATEST_LOGGED_CPU_STATE_PROXY_POLICY_ID",
    "PHYSICAL_PRESENTATION_REQUIRED",
    "PRESENTATION_TIME_UNAVAILABLE",
    "CapabilityAssessment",
    "CapabilityState",
    "ChaserProfileApplicabilityError",
    "ChaserProfileApplicabilityPlan",
    "ModuleApplicabilityDecision",
    "ModuleApplicabilityState",
    "ProfileReadiness",
    "input_provenance_proxy_alignment_assessment",
    "plan_chaser_profile_applicability",
    "require_chaser_profile_applicability_plan",
    "unavailable_physical_presentation_alignment_assessment",
]

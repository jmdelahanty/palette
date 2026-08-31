"""Closed validation for exact relative-frame child bindings.

Some successors bind a relative-frame child by its immutable scientific
identity alone.  Receipt-backed consumers may enrich that same identity with
the digest and verification mode of the validation receipt they used.  The
two records are equivalent only when their exact child path and manifest
digest agree; receipt evidence is retained but does not create a different
scientific child.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    CHASER_RELATIVE_FRAME_RUN_PARENT_PATH,
)
from fisheye.analysis_workflows.chaser_relative_frame_validation_receipt import (
    VERIFICATION_MODE as RECEIPT_BOUND_TARGETED_ARRAY_REHASH_MODE,
)


MINIMAL_EXACT_CHILD_PROFILE = "minimal_exact_child_v1"
RECEIPT_BOUND_PROFILE = "receipt_bound_targeted_array_rehash_v1"
NORMALIZED_IDENTITY_FIELDS = frozenset({"run_path", "manifest_sha256"})
RECEIPT_BOUND_FIELDS = frozenset(
    {
        "run_path",
        "manifest_sha256",
        "validation_receipt_sha256",
        "verification_mode",
    }
)
_FORBIDDEN_CHILD_NAMES = frozenset(
    {
        ".",
        "..",
        "authoritative",
        "authoritative_run",
        "active",
        "active_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "latest",
        "latest_any",
        "latest_complete",
        "latest_pending",
        "latest_provider",
        "selected",
        "selected_run",
    }
)


class ExactRelativeFrameBindingError(ValueError):
    """Raised when an exact relative-frame binding is malformed or stale."""


def _sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactRelativeFrameBindingError(
            f"{field} must be one lowercase SHA-256 digest."
        )
    return value


def _exact_child_path(value: Any, *, parent: str) -> str:
    if type(value) is not str or value != value.strip().strip("/"):
        raise ExactRelativeFrameBindingError(
            "relative-frame run_path must be one exact child path."
        )
    prefix = f"{parent}/"
    name = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or not name
        or "/" in name
        or name in _FORBIDDEN_CHILD_NAMES
    ):
        raise ExactRelativeFrameBindingError(
            f"relative-frame run_path must be one exact child below {parent!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class ExactRelativeFrameBinding:
    """Validated binding plus optional sealed receipt evidence."""

    run_path: str
    manifest_sha256: str
    profile_id: str
    validation_receipt_sha256: str | None = None
    verification_mode: str | None = None

    @property
    def normalized_identity(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "run_path": self.run_path,
                "manifest_sha256": self.manifest_sha256,
            }
        )

    def provenance_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "normalized_identity": self.normalized_identity,
                "binding_profile": self.profile_id,
                "validation_receipt_sha256": self.validation_receipt_sha256,
                "verification_mode": self.verification_mode,
            }
        )


@dataclass(frozen=True, slots=True)
class ExactRelativeFrameBindingProof:
    """Proof that two closed binding profiles name one scientific child."""

    expected: ExactRelativeFrameBinding
    observed: ExactRelativeFrameBinding

    @property
    def normalized_identity(self) -> Mapping[str, str]:
        return self.expected.normalized_identity

    def provenance_record(self) -> Mapping[str, Any]:
        receipt_bindings = tuple(
            binding
            for binding in (self.expected, self.observed)
            if binding.profile_id == RECEIPT_BOUND_PROFILE
        )
        receipt_digests = tuple(
            dict.fromkeys(
                binding.validation_receipt_sha256 for binding in receipt_bindings
            )
        )
        verification_modes = tuple(
            dict.fromkeys(binding.verification_mode for binding in receipt_bindings)
        )
        if not receipt_bindings:
            receipt_evidence_relationship = "no_receipt_evidence"
        elif len(receipt_bindings) == 1:
            receipt_evidence_relationship = "one_receipt_bound_one_minimal"
        elif len(receipt_digests) == 1:
            receipt_evidence_relationship = "shared_receipt"
        else:
            receipt_evidence_relationship = "independent_receipts_same_exact_child"
        return MappingProxyType(
            {
                "normalized_identity": self.normalized_identity,
                "expected_binding_profile": self.expected.profile_id,
                "observed_binding_profile": self.observed.profile_id,
                "expected_validation_receipt_sha256": (
                    self.expected.validation_receipt_sha256
                ),
                "observed_validation_receipt_sha256": (
                    self.observed.validation_receipt_sha256
                ),
                "expected_verification_mode": self.expected.verification_mode,
                "observed_verification_mode": self.observed.verification_mode,
                "validation_receipt_sha256": (
                    receipt_digests[0] if len(receipt_digests) == 1 else None
                ),
                "verification_mode": (
                    verification_modes[0] if len(verification_modes) == 1 else None
                ),
                "validation_receipt_sha256s": receipt_digests,
                "verification_modes": verification_modes,
                "receipt_evidence_relationship": receipt_evidence_relationship,
                "validation_behavior": (
                    "binding_schema_and_identity_validated_receipt_digest_not_reopened"
                ),
            }
        )


def validate_exact_relative_frame_binding(
    value: Any,
    *,
    parent: str = CHASER_RELATIVE_FRAME_RUN_PARENT_PATH,
    label: str = "relative-frame binding",
) -> ExactRelativeFrameBinding:
    """Validate one of the two accepted, closed relative-child profiles."""

    if not isinstance(value, Mapping):
        raise ExactRelativeFrameBindingError(f"{label} must be one object.")
    fields = frozenset(value)
    if fields == NORMALIZED_IDENTITY_FIELDS:
        profile_id = MINIMAL_EXACT_CHILD_PROFILE
        receipt_sha256 = None
        verification_mode = None
    elif fields == RECEIPT_BOUND_FIELDS:
        profile_id = RECEIPT_BOUND_PROFILE
        receipt_sha256 = _sha256(
            value.get("validation_receipt_sha256"),
            field=f"{label} validation_receipt_sha256",
        )
        verification_mode = value.get("verification_mode")
        if verification_mode != RECEIPT_BOUND_TARGETED_ARRAY_REHASH_MODE:
            raise ExactRelativeFrameBindingError(
                f"{label} verification_mode is unsupported."
            )
    else:
        raise ExactRelativeFrameBindingError(
            f"{label} has an unrecognized closed field set: "
            f"{sorted(str(field) for field in fields)!r}."
        )
    return ExactRelativeFrameBinding(
        run_path=_exact_child_path(value.get("run_path"), parent=parent),
        manifest_sha256=_sha256(
            value.get("manifest_sha256"), field=f"{label} manifest_sha256"
        ),
        profile_id=profile_id,
        validation_receipt_sha256=receipt_sha256,
        verification_mode=verification_mode,
    )


def require_same_exact_relative_frame_child(
    expected: Any,
    observed: Any,
    *,
    parent: str = CHASER_RELATIVE_FRAME_RUN_PARENT_PATH,
    expected_label: str = "expected relative-frame binding",
    observed_label: str = "observed relative-frame binding",
) -> ExactRelativeFrameBindingProof:
    """Require two valid profiles to name one exact immutable child."""

    expected_binding = validate_exact_relative_frame_binding(
        expected, parent=parent, label=expected_label
    )
    observed_binding = validate_exact_relative_frame_binding(
        observed, parent=parent, label=observed_label
    )
    if (
        expected_binding.run_path != observed_binding.run_path
        or expected_binding.manifest_sha256 != observed_binding.manifest_sha256
    ):
        raise ExactRelativeFrameBindingError(
            "Relative-frame bindings name different exact scientific children."
        )
    return ExactRelativeFrameBindingProof(
        expected=expected_binding,
        observed=observed_binding,
    )


__all__ = [
    "ExactRelativeFrameBinding",
    "ExactRelativeFrameBindingError",
    "ExactRelativeFrameBindingProof",
    "MINIMAL_EXACT_CHILD_PROFILE",
    "RECEIPT_BOUND_PROFILE",
    "require_same_exact_relative_frame_child",
    "validate_exact_relative_frame_binding",
]

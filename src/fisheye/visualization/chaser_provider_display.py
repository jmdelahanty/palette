"""Explicit semantic display labels for sealed chaser position providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

POSITION_PROVIDER_DISPLAY_POLICY_ID = (
    "explicit_semantic_position_provider_labels_identity_in_receipt_v1"
)
POSITION_PROVIDER_ROLE_ORDER = ("keypoint", "detection")
_LABELS = {
    "keypoint": ("Keypoint-derived position", "Keypoint"),
    "detection": ("Detection-derived position", "Detection"),
}


@dataclass(frozen=True, slots=True)
class PositionProviderDisplayBinding:
    """One caller-declared provider role and its retained exact identity."""

    role: str
    display_label: str
    compact_label: str
    provider_id: str

    def provenance_record(self) -> dict[str, str]:
        """Return the complete display binding without hiding source identity."""

        return {
            "policy_id": POSITION_PROVIDER_DISPLAY_POLICY_ID,
            "provider_role": self.role,
            "display_label": self.display_label,
            "compact_label": self.compact_label,
            "provider_id": self.provider_id,
            "role_source": "explicit_validated_caller_contract",
            "provider_id_parsing": "prohibited",
        }


def position_provider_display_binding(
    *, provider_role: str, provider_id: str
) -> PositionProviderDisplayBinding:
    """Bind an explicit canonical role to labels while retaining the exact ID."""

    if provider_role not in _LABELS:
        raise ValueError(
            "Position provider role must be exactly 'keypoint' or 'detection'."
        )
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("Position provider identity must be a non-empty string.")
    display_label, compact_label = _LABELS[provider_role]
    return PositionProviderDisplayBinding(
        role=provider_role,
        display_label=display_label,
        compact_label=compact_label,
        provider_id=provider_id,
    )


def paired_position_provider_display_bindings(
    *,
    provider_ids: Sequence[str],
    provider_roles: Sequence[str],
) -> tuple[PositionProviderDisplayBinding, PositionProviderDisplayBinding]:
    """Require the canonical explicit keypoint/detection comparison order."""

    if tuple(provider_roles) != POSITION_PROVIDER_ROLE_ORDER:
        raise ValueError(
            "Paired position providers must use explicit keypoint/detection order."
        )
    if len(provider_ids) != 2:
        raise ValueError("Paired position providers require exactly two identities.")
    if len(set(provider_ids)) != 2:
        raise ValueError("Paired position provider identities must remain distinct.")
    return (
        position_provider_display_binding(
            provider_role="keypoint", provider_id=provider_ids[0]
        ),
        position_provider_display_binding(
            provider_role="detection", provider_id=provider_ids[1]
        ),
    )


__all__ = [
    "POSITION_PROVIDER_DISPLAY_POLICY_ID",
    "POSITION_PROVIDER_ROLE_ORDER",
    "PositionProviderDisplayBinding",
    "paired_position_provider_display_bindings",
    "position_provider_display_binding",
]
